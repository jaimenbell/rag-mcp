"""Ingest pipeline: walk a corpus dir, chunk markdown, embed + store with metadata.

Deterministic + idempotent: a chunk's id is `<relative-path>::<chunk_index>`, so
re-ingesting the same corpus upserts in place rather than duplicating.
Empty or undecodable files are skipped, never fatal.

INCREMENTAL by default (2026-07-30): a manifest inside the store dir records a
content hash per file, so a run only embeds what actually changed. Embedding is
essentially the whole cost of a run, so this is what makes frequent scheduling
affordable -- see rag_mcp.manifest for the trust rules. Pass ``incremental=False``
to force a full re-embed. Incremental runs also PRUNE, which upsert alone never
did: chunks of deleted files, and trailing chunks of files that got shorter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, chunk_markdown
from .manifest import IngestManifest, RunIdentity, content_hash, load_manifest
from .store import VectorStore

MARKDOWN_EXTS = (".md", ".markdown")

# Vault-relative POSIX path prefixes excluded from ingest by default.
# Files whose relative path starts with any of these strings are silently skipped.
# CLI --exclude flags AUGMENT (not replace) these defaults.
EXCLUDE_PREFIXES = (
    "infrastructure/claude-config-backup/",
    # Index-noise: link-audit sweep records are dense wikilink lists that
    # bge-large matches aggressively on almost any vault-topic query
    # (2/6 A/B regressions traced here at the 2026-07-02 cutover).
    "Routines/Wikilink Scan",
)


@dataclass
class IngestReport:
    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    chunks_added: int = 0
    # Incremental-path counters.
    files_unchanged: int = 0  # content hash matched the manifest -> not re-embedded
    chunks_deleted: int = 0  # pruned: deleted files + trailing chunks of shrunk files
    incremental: bool = False  # False when the run re-embedded everything


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
    incremental: bool = True,
    store_dir: Path | str | None = None,
) -> IngestReport:
    """Ingest all corpus files into *store*, skipping excluded prefix subtrees.

    Args:
        exclude_prefixes: vault-relative POSIX prefixes to skip (see
            :data:`EXCLUDE_PREFIXES`). Passed straight through to
            :func:`iter_corpus_files`. Pass ``()`` to disable all exclusions.
        incremental: when True (default), skip re-embedding files whose content
            hash matches the manifest, and prune stale chunks. Requires a
            persistent store; an in-memory store has nowhere to keep a manifest
            and always does a full pass.
        store_dir: where the manifest lives. Defaults to the store's own
            directory, so the manifest and the store are created and destroyed
            together.
    """
    root = Path(root)
    report = IngestReport()

    if store_dir is None:
        store_dir = store.path
    # Reading the manifest (i.e. skipping unchanged files) is what --full/--clean
    # turn off. WRITING it is unconditional on a persistent store: a full rebuild
    # that left no manifest behind would force the very next run to be full too.
    can_persist = store_dir is not None
    use_manifest = incremental and can_persist
    identity = RunIdentity(
        embedder=type(store.embedder).__name__,
        embed_dim=store.embedder.dim,
        collection=store.collection_name,
        max_chars=max_chars,
        overlap=overlap,
    )

    previous = load_manifest(store_dir, identity) if use_manifest else IngestManifest(identity)
    # Desync guard: a populated manifest against an empty store would skip every
    # file and leave the index permanently empty. Trust the store, not the file.
    if previous.files and store.count() == 0:
        previous = IngestManifest(identity)

    current = IngestManifest(identity)
    report.incremental = bool(use_manifest and previous.files)
    seen_rels: set[str] = set()

    for path in iter_corpus_files(root, exclude_prefixes=exclude_prefixes):
        report.files_seen += 1
        text = load_file(path)
        if text is None or not text.strip():
            report.files_skipped += 1
            continue
        rel = path.relative_to(root).as_posix()
        digest = content_hash(text)
        seen_rels.add(rel)

        if use_manifest and previous.unchanged(rel, digest):
            # Content identical to what is already embedded -- the expensive
            # chunk+embed+upsert is pure waste here. This is the whole point.
            report.files_unchanged += 1
            current.record(rel, digest, previous.chunk_count(rel))
            continue

        chunks = chunk_markdown(text, max_chars=max_chars, overlap=overlap)
        if not chunks:
            report.files_skipped += 1
            seen_rels.discard(rel)
            continue
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

        # A file that got SHORTER leaves orphaned trailing chunks behind:
        # upsert only overwrites 0..n-1, it never removes the old n..m-1.
        stale = _trailing_ids(rel, len(chunks), previous.chunk_count(rel))
        if stale:
            store.delete(ids=stale)
            report.chunks_deleted += len(stale)

        current.record(rel, digest, len(chunks))
        report.files_ingested += 1
        report.chunks_added += len(chunks)

    # Files that vanished from the corpus since the last run: upsert can never
    # prune these, which is why a weekly full rebuild was needed to drop them.
    if use_manifest:
        for rel in set(previous.files) - seen_rels:
            gone = [f"{rel}::{i}" for i in range(previous.chunk_count(rel))]
            if gone:
                store.delete(ids=gone)
                report.chunks_deleted += len(gone)

    if can_persist:
        current.save(store_dir)

    return report


def _trailing_ids(rel: str, new_count: int, old_count: int) -> list[str]:
    """Ids for chunks that existed last run but do not exist now."""
    if old_count <= new_count:
        return []
    return [f"{rel}::{i}" for i in range(new_count, old_count)]
