"""Vector store wrapper (ChromaDB, embedded) + pluggable embedders.

The store owns the embedder explicitly (we pass precomputed embeddings to Chroma
rather than registering a Chroma EmbeddingFunction). That keeps the embedding model
fully under our control and insulated from Chroma's EmbeddingFunction API churn.

Three embedders ship:
  * DefaultEmbedder  - the real, local ONNX all-MiniLM-L6-v2 (384-dim, $0, no API,
                       256-token context -- silently truncates longer chunks).
  * BgeEmbedder      - local ONNX BAAI/bge-large-en-v1.5 (1024-dim, $0, no API,
                       512-token context). Asymmetric: queries get an instruction
                       prefix (see BGE_QUERY_PREFIX), documents do not.
  * HashEmbedder     - a deterministic offline bag-of-words embedder for tests.

Every embedder exposes a `dim` (int) and a `query_prefix` (str, default "") so
VectorStore can (a) guard against dimension-mismatched stores/embedders and
(b) apply retrieval-instruction prefixes only where the model wants them.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

import chromadb

BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _stable_bucket(token: str, dim: int) -> int:
    """Process-stable hash bucket (builtin hash() is randomized per process)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class DimensionMismatchError(ValueError):
    """Raised when an embedder's output dim doesn't match the store it's opening.

    Chroma has no native cross-model guard: mixing a 384-dim (MiniLM) and a
    1024-dim (bge) embedder against the same collection either raises an opaque
    Rust-layer error deep inside `.add()`/`.query()` or, worse, silently degrades
    similarity math. We record `embed_dim` in collection metadata at creation
    time and fail fast, at open time, with a clear message instead.
    """


class Embedder(Protocol):
    dim: int
    query_prefix: str

    def __call__(self, texts: Sequence[str]) -> list[list[float]]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic, offline bag-of-words embedder over a hashed vocabulary.

    Word overlap -> high cosine similarity. No model download, no network, $0.
    Used by the test suite so retrieval wiring is provable without the ONNX model.
    """

    query_prefix = ""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _TOKEN_RE.findall((text or "").lower()):
                vec[_stable_bucket(tok, self.dim)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


class DefaultEmbedder:
    """Local ONNX all-MiniLM-L6-v2 embedder (384-dim). $0, CPU, no API key.

    Lazily constructs Chroma's bundled embedding function on first call so that
    importing this module never triggers the one-time model download.

    Known limitation (see 2026-06-30 research note): the underlying tokenizer
    caps at 256 word-pieces, so chunks longer than ~180 words are silently
    truncated -- only the head of the chunk is ever embedded. BgeEmbedder below
    is the upgrade path (512-token context) and is symmetric-scored; MiniLM
    remains the default embedder until the bge store is cut over.
    """

    dim = 384
    query_prefix = ""

    def __init__(self) -> None:
        self._fn = None

    def _ensure(self):
        if self._fn is None:
            from chromadb.utils import embedding_functions as ef

            self._fn = ef.DefaultEmbeddingFunction()
        return self._fn

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        fn = self._ensure()
        result = fn(list(texts))
        return [list(map(float, v)) for v in result]


class BgeEmbedder:
    """Local ONNX BAAI/bge-large-en-v1.5 embedder (1024-dim). $0, CPU, no API key.

    Backed by `fastembed` (pure onnxruntime + huggingface_hub, no torch). First
    call triggers a one-time ~1.3GB model download into the fastembed cache
    (~/.cache/fastembed by default); subsequent calls are fully local.

    512-token context (vs MiniLM's 256) -- see BGE_MODEL_NAME's model card.
    Asymmetric encoder: bge's training scheme means QUERIES should carry a
    retrieval-instruction prefix but DOCUMENTS should not. This class never
    prefixes on `__call__` (used for documents by VectorStore.add); the prefix
    is applied by VectorStore.query via `query_prefix`, exactly once, on the
    query text only.
    """

    dim = 1024
    query_prefix = BGE_QUERY_PREFIX

    def __init__(self) -> None:
        self._model = None

    def _ensure(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=BGE_MODEL_NAME)
        return self._model

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure()
        return [vec.tolist() for vec in model.embed(list(texts))]


class VectorStore:
    """Thin wrapper over a Chroma collection with an explicit embedder.

    path=None -> in-memory (ephemeral) client. Otherwise a persistent on-disk store.
    Cosine space; ids are caller-supplied and upserted (idempotent re-ingest).
    """

    def __init__(
        self,
        *,
        path: str | None,
        collection_name: str,
        embedder: Embedder,
    ) -> None:
        self.embedder = embedder
        if path is None:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=path)
        # get_or_create_collection only APPLIES metadata on first creation; on a
        # later open of an existing collection the metadata arg here is ignored
        # and the collection's original metadata (incl. embed_dim) wins -- so
        # this doubles as the guard's source of truth, not just a write.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embed_dim": embedder.dim},
        )
        stored_dim = (self._collection.metadata or {}).get("embed_dim")
        if stored_dim is not None and stored_dim != embedder.dim:
            raise DimensionMismatchError(
                f"Collection {collection_name!r} at {path!r} was built with "
                f"embed_dim={stored_dim}, but {type(embedder).__name__} produces "
                f"dim={embedder.dim}. Mixing embedders in one store corrupts "
                "similarity search -- point this embedder at a different store "
                "path/collection (e.g. a bge-specific store dir) instead."
            )

    def add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not ids:
            return
        embeddings = self.embedder(list(documents))
        self._collection.upsert(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas),
            embeddings=embeddings,
        )

    def query(self, text: str, k: int = 5) -> list[dict]:
        if k <= 0:
            return []
        # Query-side instruction prefix (e.g. bge's "Represent this sentence for
        # searching relevant passages: "). Documents embedded via .add() never
        # see this -- bge is an asymmetric encoder and only wants it on queries.
        prefixed = self.embedder.query_prefix + text
        q = self.embedder([prefixed])[0]
        res = self._collection.query(query_embeddings=[q], n_results=k)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits: list[dict] = []
        for i, doc in enumerate(docs):
            hits.append(
                {
                    "document": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def all_metadatas(self) -> list[dict]:
        res = self._collection.get(include=["metadatas"])
        return list(res.get("metadatas") or [])
