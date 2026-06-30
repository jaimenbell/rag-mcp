"""Phase 1 - chunking boundaries + metadata."""
from __future__ import annotations

from rag_mcp.chunking import Chunk, chunk_markdown


def test_empty_text_yields_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []


def test_single_heading_chunk():
    chunks = chunk_markdown("# Title\n\nbody text here")
    assert len(chunks) == 1
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert c.heading == "Title"
    assert "body text here" in c.text
    assert c.chunk_index == 0


def test_heading_tracked_across_sections():
    text = "# Alpha\n\nfirst body\n\n# Beta\n\nsecond body"
    chunks = chunk_markdown(text)
    headings = {c.heading for c in chunks}
    assert headings == {"Alpha", "Beta"}
    by_heading = {c.heading: c.text for c in chunks}
    assert "first body" in by_heading["Alpha"]
    assert "second body" in by_heading["Beta"]


def test_body_before_any_heading_has_none_heading():
    chunks = chunk_markdown("intro paragraph with no heading")
    assert len(chunks) == 1
    assert chunks[0].heading is None


def test_long_section_splits_with_overlap():
    body = "word " * 600  # ~3000 chars, single section
    text = "# Long\n\n" + body
    chunks = chunk_markdown(text, max_chars=1000, overlap=150)
    assert len(chunks) > 1
    # Consecutive chunks share an overlap window of `overlap` chars.
    for a, b in zip(chunks, chunks[1:]):
        assert a.text[-150:] == b.text[:150]


def test_chunk_indices_sequential_and_unique():
    body = "word " * 600
    text = "# A\n\n" + body + "\n\n# B\n\n" + body
    chunks = chunk_markdown(text, max_chars=1000, overlap=150)
    idxs = [c.chunk_index for c in chunks]
    assert idxs == list(range(len(chunks)))
    assert len(set(idxs)) == len(idxs)
