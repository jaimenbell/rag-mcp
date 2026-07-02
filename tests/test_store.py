"""Store-level guards for the bge-large upgrade (see 2026-06-30 research note).

Covers, all offline/$0/tmp_path-only:
  * dimension-mismatch guard (a MiniLM-shaped store must reject a bge-shaped
    embedder and vice versa -- mixing dims corrupts cosine similarity)
  * config seam default (RAG_MCP_EMBEDDER defaults to MiniLM until cutover)
  * query-instruction prefix applied on queries only, never on documents
  * bge's wider (512 vs 256 token) context isn't truncated by OUR code before
    it ever reaches the model
"""
from __future__ import annotations

import os

import pytest

from rag_mcp.config import Config
from rag_mcp.store import (
    BGE_QUERY_PREFIX,
    BgeEmbedder,
    DefaultEmbedder,
    DimensionMismatchError,
    HashEmbedder,
    VectorStore,
)


# ---------------------------------------------------------------------------
# Dimension-mismatch guard
# ---------------------------------------------------------------------------


def test_store_records_embed_dim_on_creation(tmp_path):
    store = VectorStore(
        path=str(tmp_path / "store.chroma"),
        collection_name="knowledge",
        embedder=HashEmbedder(dim=384),
    )
    assert store._collection.metadata["embed_dim"] == 384


def test_bge_shaped_embedder_rejected_by_minilm_shaped_store(tmp_path):
    db = str(tmp_path / "store.chroma")
    # First open "creates" the store shaped like MiniLM (384-dim).
    VectorStore(path=db, collection_name="knowledge", embedder=HashEmbedder(dim=384))
    # Reopening the SAME store/collection with a bge-shaped (1024-dim) embedder
    # must fail fast, not corrupt the collection or silently degrade search.
    with pytest.raises(DimensionMismatchError):
        VectorStore(path=db, collection_name="knowledge", embedder=HashEmbedder(dim=1024))


def test_minilm_shaped_embedder_rejected_by_bge_shaped_store(tmp_path):
    db = str(tmp_path / "store.chroma")
    VectorStore(path=db, collection_name="knowledge", embedder=HashEmbedder(dim=1024))
    with pytest.raises(DimensionMismatchError):
        VectorStore(path=db, collection_name="knowledge", embedder=HashEmbedder(dim=384))


def test_matching_dim_reopens_cleanly(tmp_path):
    db = str(tmp_path / "store.chroma")
    VectorStore(path=db, collection_name="knowledge", embedder=HashEmbedder(dim=384))
    # No raise: same dim, different embedder instance -- this is the normal
    # "process restarted" path and must keep working.
    reopened = VectorStore(path=db, collection_name="knowledge", embedder=HashEmbedder(dim=384))
    assert reopened.count() == 0


def test_different_collection_name_sidesteps_mismatch(tmp_path):
    # Real cutover shape: same store DIR, a bge-specific collection name, so a
    # differently-dimensioned embedder is fine as long as it's a fresh collection.
    db = str(tmp_path / "store.chroma")
    VectorStore(path=db, collection_name="knowledge", embedder=HashEmbedder(dim=384))
    bge_side = VectorStore(
        path=db, collection_name="knowledge_bge", embedder=HashEmbedder(dim=1024)
    )
    assert bge_side.count() == 0


def test_real_embedder_dims_declared():
    # Sanity: the two real (non-test) embedders declare the dims the research
    # note verified (384 for MiniLM, 1024 for bge-large-en-v1.5).
    assert DefaultEmbedder.dim == 384
    assert BgeEmbedder.dim == 1024


# ---------------------------------------------------------------------------
# Query-instruction prefix (bge is asymmetric: queries only, never documents)
# ---------------------------------------------------------------------------


class _RecordingEmbedder:
    """Captures exactly what text VectorStore hands to __call__."""

    dim = 8
    query_prefix = "PREFIX: "

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[0.0] * self.dim for _ in texts]


def test_query_prefix_applied_to_queries_only(tmp_path):
    embedder = _RecordingEmbedder()
    store = VectorStore(
        path=str(tmp_path / "store.chroma"), collection_name="knowledge", embedder=embedder
    )
    store.add(
        ids=["a"], documents=["a document about cats"], metadatas=[{"source": "cats.md"}]
    )
    store.query("cats", k=1)

    doc_call, query_call = embedder.calls
    assert doc_call == ["a document about cats"], "documents must NOT get the prefix"
    assert query_call == ["PREFIX: cats"], "queries MUST get the prefix, exactly once"


def test_default_and_hash_embedders_have_no_query_prefix():
    assert DefaultEmbedder.query_prefix == ""
    assert HashEmbedder().query_prefix == ""


def test_bge_embedder_declares_the_researched_instruction_prefix():
    # Exact string from the 2026-06-30 research note's bge guidance.
    assert BgeEmbedder.query_prefix == BGE_QUERY_PREFIX
    assert BGE_QUERY_PREFIX == "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Truncation: our wrapper must not re-impose MiniLM's 256-token ceiling
# ---------------------------------------------------------------------------


def test_bge_embedder_passes_full_text_through_untruncated(monkeypatch):
    """A >256-token chunk must reach the model whole; bge's own 512-token
    tokenizer decides what (if anything) gets cut, not our wrapper.
    """
    long_text = " ".join(f"word{i}" for i in range(400))  # ~400 tokens, > MiniLM's 256

    class _FakeModel:
        def embed(self, texts):
            _FakeModel.last_seen = list(texts)
            return [_Vec([0.0] * 1024) for _ in texts]

    class _Vec(list):
        def tolist(self):
            return list(self)

    embedder = BgeEmbedder()
    embedder._model = _FakeModel()  # skip the real ~1.3GB download in unit tests

    embedder([long_text])

    seen = _FakeModel.last_seen[0]
    assert seen == long_text, "wrapper must not pre-truncate before handing off to bge"
    assert seen.split()[-1] == "word399", "tail content must survive to the embedding input"


def test_bge_query_prefix_does_not_truncate_the_underlying_text(tmp_path):
    long_text = " ".join(f"word{i}" for i in range(400))

    class _Vec(list):
        def tolist(self):
            return list(self)

    class _FakeModel:
        def embed(self, texts):
            _FakeModel.last_seen = list(texts)
            return [_Vec([0.0] * 1024) for _ in texts]

    embedder = BgeEmbedder()
    embedder._model = _FakeModel()
    store = VectorStore(
        path=str(tmp_path / "bge.chroma"), collection_name="knowledge", embedder=embedder
    )
    store.query(long_text, k=1)

    seen = _FakeModel.last_seen[0]
    assert seen.startswith(BGE_QUERY_PREFIX)
    assert seen.endswith("word399")


# ---------------------------------------------------------------------------
# Config seam: MiniLM stays the DEFAULT until cutover
# ---------------------------------------------------------------------------


def test_config_default_embedder_is_still_minilm(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_MCP_EMBEDDER", raising=False)
    monkeypatch.setenv("RAG_MCP_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_MCP_DB_PATH", str(tmp_path / "store.chroma"))
    cfg = Config.from_env()
    assert cfg.embedder_name == "default"
    assert isinstance(cfg.make_embedder(), DefaultEmbedder)


def test_config_opts_into_bge_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_MCP_EMBEDDER", "bge")
    monkeypatch.setenv("RAG_MCP_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_MCP_DB_PATH", str(tmp_path / "store-bge.chroma"))
    cfg = Config.from_env()
    assert cfg.embedder_name == "bge"
    assert isinstance(cfg.make_embedder(), BgeEmbedder)
