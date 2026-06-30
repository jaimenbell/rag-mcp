"""Markdown chunking: split a document into heading-scoped, overlapping chunks.

Each chunk records the heading it lives under and a sequential index within the
document, so downstream retrieval can cite (source path + heading + chunk index).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP = 150


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    heading: str | None
    text: str


def _sections(text: str) -> list[tuple[str | None, str]]:
    """Split text into (heading, body) sections on markdown ATX headings."""
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((current_heading, body))

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            buf = []
            current_heading = m.group(2).strip()
        else:
            buf.append(line)
    flush()
    return sections


def chunk_markdown(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk markdown into <= max_chars windows overlapping by `overlap` chars.

    Returns an empty list for empty/whitespace-only input. Body text appearing
    before any heading is kept with heading=None.
    """
    if not text or not text.strip():
        return []
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    step = max_chars - overlap
    chunks: list[Chunk] = []
    idx = 0
    for heading, body in _sections(text):
        if len(body) <= max_chars:
            chunks.append(Chunk(idx, heading, body))
            idx += 1
            continue
        start = 0
        while start < len(body):
            piece = body[start : start + max_chars]
            chunks.append(Chunk(idx, heading, piece))
            idx += 1
            if start + max_chars >= len(body):
                break
            start += step
    return chunks
