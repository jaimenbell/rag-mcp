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

# Production store guard: the live index (~44k chunks) is production data. This
# fleet has a recurring "tests pollute prod artifacts" problem, so we assert the
# real store dir is byte-for-byte untouched by the whole test session.
_REAL_STORE = Path(__file__).resolve().parent.parent / "store.chroma"


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
    before = _store_fingerprint(_REAL_STORE)
    yield
    after = _store_fingerprint(_REAL_STORE)
    assert after == before, (
        f"TEST POLLUTION: real store {_REAL_STORE} was modified during the test "
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
