"""Ingest pipeline: walk a corpus dir, chunk markdown, embed + store with metadata.

Deterministic + idempotent: a chunk's id is `<relative-path>::<chunk_index>`, so
re-ingesting the same corpus upserts in place rather than duplicating.
Empty or undecodable files are skipped, never fatal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, chunk_markdown
from .store import VectorStore

MARKDOWN_EXTS = (".md", ".markdown")

# Vault-relative POSIX path prefixes excluded from ingest by default.
# Files whose relative path starts with any of these strings are silently skipped.
# CLI --exclude flags AUGMENT (not replace) these defaults.
EXCLUDE_PREFIXES = ("infrastructure/claude-config-backup/",)


@dataclass
class IngestReport:
    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    chunks_added: int = 0


def iter_corpus_files(
    root: Path | str,
    exts: Iterable[str] = MARKDOWN_EXTS,
    exclude_prefixes: tuple[str, ...] = EXCLUDE_PREFIXES,
) -> list[Path]:
    """Return sorted list of corpus markdown files, skipping excluded prefix subtrees.

    Args:
        root: corpus root directory.
        exts: file extensions to include (case-insensitive).
        exclude_prefixes: vault-relative POSIX prefixes to skip. Files whose
            relative path starts with any of these strings are excluded. Pass
            ``()`` to disable all exclusions.
    """
    root = Path(root)
    exts = tuple(e.lower() for e in exts)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in exts
        and not any(
            p.relative_to(root).as_posix().startswith(px) for px in exclude_prefixes
        )
    )


def load_file(path: Path | str) -> str | None:
    """Read a UTF-8 text file. Returns None (skip, not fatal) if it cannot decode."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def ingest(
    root: Path | str,
    store: VectorStore,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    exclude_prefixes: tuple[str, ...] = EXCLUDE_PREFIXES,
) -> IngestReport:
    """Ingest all corpus files into *store*, skipping excluded prefix subtrees.

    Args:
        exclude_prefixes: vault-relative POSIX prefixes to skip (see
            :data:`EXCLUDE_PREFIXES`). Passed straight through to
            :func:`iter_corpus_files`. Pass ``()`` to disable all exclusions.
    """
    root = Path(root)
    report = IngestReport()
    for path in iter_corpus_files(root, exclude_prefixes=exclude_prefixes):
        report.files_seen += 1
        text = load_file(path)
        if text is None or not text.strip():
            report.files_skipped += 1
            continue
        chunks = chunk_markdown(text, max_chars=max_chars, overlap=overlap)
        if not chunks:
            report.files_skipped += 1
            continue
        rel = path.relative_to(root).as_posix()
        ids = [f"{rel}::{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": rel,
                "heading": c.heading if c.heading is not None else "",
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        store.add(ids=ids, documents=documents, metadatas=metadatas)
        report.files_ingested += 1
        report.chunks_added += len(chunks)
    return report
