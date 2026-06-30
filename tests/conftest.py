"""Shared test fixtures.

Tests run fully offline + $0: they use the deterministic HashEmbedder (bag-of-words
over a hashed vocabulary) instead of the real ONNX model, so the suite is fast and
needs no network. Word overlap -> high cosine similarity, which is enough to prove
retrieval wiring without downloading a model.
"""
from __future__ import annotations

import pytest

from rag_mcp.store import HashEmbedder, VectorStore


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=256)


@pytest.fixture
def store(embedder: HashEmbedder) -> VectorStore:
    # In-memory (ephemeral) store: no disk, fast, isolated per test.
    return VectorStore(path=None, collection_name="test", embedder=embedder)


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
