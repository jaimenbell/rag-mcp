"""Shared test fixtures.

Tests run fully offline + $0: they use the deterministic HashEmbedder (bag-of-words
over a hashed vocabulary) instead of the real ONNX model, so the suite is fast and
needs no network. Word overlap -> high cosine similarity, which is enough to prove
retrieval wiring without downloading a model.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from rag_mcp.store import HashEmbedder, VectorStore

# Production store guard: the live index is production data. This fleet has a
# recurring "tests pollute prod artifacts" problem, so we assert the real store
# dirs are byte-for-byte untouched by the whole test session.
#
# BOTH stores are watched. Until 2026-07-30 this guarded only `store.chroma`,
# which the 2026-07-02 bge cutover had already demoted to a stale 184 KB
# rollback copy -- the actual live index moved to `store-bge.chroma` and was
# guarded by nothing. The guard was pointed at the wrong artifact and so could
# not have detected pollution of the store it claimed to protect.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_STORES = (
    _REPO_ROOT / "store-bge.chroma",  # live (bge, 1024-dim) since 2026-07-02
    _REPO_ROOT / "store.chroma",  # legacy MiniLM store, kept for rollback
)


def _store_fingerprint(path: Path):
    if not path.exists():
        return None
    # (mtime_ns, size) of the dir and every file underneath -> any write shows up.
    entries = [(path.stat().st_mtime_ns, -1)]
    for p in sorted(path.rglob("*")):
        st = p.stat()
        entries.append((st.st_mtime_ns, st.st_size))
    return tuple(entries)


@pytest.fixture(scope="session", autouse=True)
def _guard_real_store():
    before = {p: _store_fingerprint(p) for p in _REAL_STORES}
    yield
    for path in _REAL_STORES:
        assert _store_fingerprint(path) == before[path], (
            f"TEST POLLUTION: real store {path} was modified during the test "
            f"session (fingerprint changed). Tests must use tmp dirs only."
        )


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=256)


@pytest.fixture
def store(embedder: HashEmbedder) -> VectorStore:
    # In-memory (ephemeral) store. Chroma shares one in-process ephemeral instance,
    # so a unique collection name per test keeps fixtures isolated.
    return VectorStore(
        path=None, collection_name=f"test_{uuid.uuid4().hex}", embedder=embedder
    )


@pytest.fixture
def corpus(tmp_path):
    """A tiny mixed corpus: two real notes, one empty, one garbled, one non-md."""
    (tmp_path / "cats.md").write_text(
        "# Cats\n\nCats are small carnivorous mammals. A cat purrs when content.\n",
        encoding="utf-8",
    )
    (tmp_path / "dogs.md").write_text(
        "# Dogs\n\n## Behavior\n\nDogs are loyal pack animals. A dog barks to alert its owner.\n",
        encoding="utf-8",
    )
    (tmp_path / "empty.md").write_text("   \n\n  ", encoding="utf-8")
    # Garbled: invalid UTF-8 bytes, .md extension -> must be skipped, not fatal.
    (tmp_path / "garbled.md").write_bytes(b"\xff\xfe\x00\x80\x81 not valid utf8 \xc3\x28")
    # Non-markdown -> must be ignored by the loader.
    (tmp_path / "notes.txt").write_text("plain text, not markdown", encoding="utf-8")
    return tmp_path
