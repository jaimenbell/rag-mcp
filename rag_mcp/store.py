"""Vector store wrapper (ChromaDB, embedded) + pluggable embedders.

The store owns the embedder explicitly (we pass precomputed embeddings to Chroma
rather than registering a Chroma EmbeddingFunction). That keeps the embedding model
fully under our control and insulated from Chroma's EmbeddingFunction API churn.

Two embedders ship:
  * DefaultEmbedder  - the real, local ONNX all-MiniLM-L6-v2 (384-dim, $0, no API).
  * HashEmbedder     - a deterministic offline bag-of-words embedder for tests.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

import chromadb


def _stable_bucket(token: str, dim: int) -> int:
    """Process-stable hash bucket (builtin hash() is randomized per process)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class Embedder(Protocol):
    def __call__(self, texts: Sequence[str]) -> list[list[float]]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic, offline bag-of-words embedder over a hashed vocabulary.

    Word overlap -> high cosine similarity. No model download, no network, $0.
    Used by the test suite so retrieval wiring is provable without the ONNX model.
    """

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
    """

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
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
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
        q = self.embedder([text])[0]
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
