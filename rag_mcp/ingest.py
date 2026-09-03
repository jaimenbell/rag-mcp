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

SNAPSHOT DE-DUPLICATION (2026-07-30): the manifest's skip is a WHOLE-FILE hash,
which cannot see the duplication that actually poisons retrieval -- daily
snapshot files that repeat yesterday's paragraphs verbatim inside a file whose
hash still changed (a timestamp moved, one number moved). Those repeats crowd
top-k with byte-identical copies of one status line. A second, CHUNK-level pass
collapses them; see rag_mcp.snapshots for the scope and the citability
guarantees. Pass ``dedupe_snapshots=False`` to disable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import yaml

from .chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, Chunk, chunk_markdown
from .manifest import IngestManifest, RunIdentity, content_hash, load_manifest
from .snapshots import (
    EMPTY_PLAN,
    SnapshotPlan,
    build_plan,
    chunk_key,
    format_repeat_dates,
    group_series,
)
from .store import VectorStore

MARKDOWN_EXTS = (".md", ".markdown")

# Default parent-directory name + basenames (case-insensitive) for a corpus's
# own session-bookkeeping mirrors -- see `_doc_class` docstring for why these
# need a filename fallback in addition to frontmatter. This is a CALLER
# convention, not a library-wide assumption: `ingest()`'s `handoff_mirror_dir`
# / `handoff_mirror_basenames` parameters (and the CLI's matching flags, which
# AUGMENT rather than replace these defaults -- same pattern as EXCLUDE_PREFIXES
# / --exclude) let a different corpus override both without editing this file.
DEFAULT_HANDOFF_MIRROR_DIR = "context"
DEFAULT_HANDOFF_MIRROR_BASENAMES = frozenset({"handoff.md", "active.md", "resume.md"})

# Anchored "handoff" filename-TOKEN matcher (fixed 2026-09-03; see
# `_doc_class`'s docstring for the rule this replaced). The naive check it
# replaced -- `"handoff" in basename` -- false-positived on ANY basename
# containing the word anywhere, e.g. "handoff-skill-redesign-spec.md" (a note
# ABOUT the handoff skill, not a session handoff record). This requires
# "handoff" to appear as a bounded TOKEN, not a bare substring:
#
#   LEADING form (`_HANDOFF_LEADING_RE`): "handoff" at the START of the
#   basename -- optionally after a YYYY-MM-DD date prefix -- followed by
#   ".md" or a NARROWLY shaped suffix: nothing, a digit-led run
#   ("-2026-09-03"), a short <=3-char alnum code ("-x", "-pm"), a run ENDING
#   in a version tag ("-alphahive-world-v6"), or a run ENDING in a
#   YYYY-MM-DD date ("-cross-project-2026-05-22"). Free text that fits none
#   of those shapes ("-skill-redesign-spec") does NOT match, so a title that
#   merely starts with the word stays "note".
#
#   TRAILING form (`_HANDOFF_TRAILING_RE`): a `-`/`_`/space separator then
#   "handoff.md" (real mirrors sometimes name the token as a TRAILING word
#   instead of a leading one, e.g. "morning-dispatch-handoff.md").
_HANDOFF_TOKEN_SUFFIX = (
    r"(?:[-_.](?:\d[\w.-]*|[a-z0-9]{1,3}|[\w.-]*-v\d[\w.-]*|[\w.-]*\d{4}-\d{2}-\d{2}))?"
)
_HANDOFF_LEADING_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}[-_ ])?handoff" + _HANDOFF_TOKEN_SUFFIX + r"\.md$",
    re.IGNORECASE,
)
_HANDOFF_TRAILING_RE = re.compile(r"[-_ ]handoff\.md$", re.IGNORECASE)

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

# Bumped when `_doc_class`/`_metadata` change in a way that requires backfilling
# ALREADY-EMBEDDED chunks (added 2026-09-03 for the doc_class classifier). A
# per-file version is stored in the manifest (`IngestManifest.record`'s
# `meta_version` / `.meta_version(rel)`); an incremental run whose file is
# otherwise unchanged (same content hash) but whose stored `metav` is behind
# this constant does a METADATA-ONLY refresh -- re-chunk (cheap) and
# `store.update_metadatas` -- without re-embedding (expensive, and the whole
# point of the incremental path). A pre-feature manifest has no `metav` at
# all, which `IngestManifest.meta_version` reads as 0 -- always stale -- so
# the very next incremental run backfills every file exactly once. Do NOT bump
# `manifest.MANIFEST_VERSION` for this: that forces a full re-embed, which a
# metadata-only change never needs.
CURRENT_METADATA_VERSION = 1


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
    # Snapshot-dedupe counter. STANDING, not a delta: how many chunks across the
    # whole corpus are verbatim repeats of the previous snapshot in their series
    # and therefore deliberately not embedded. Steady on a quiet day; it is the
    # size of the collapse, not the work this run did.
    chunks_deduped: int = 0


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
    """Read a UTF-8 text file. Returns None (skip, not fatal) if it cannot decode.

    ``utf-8-sig`` (not plain ``utf-8``): transparently strips a leading BOM
    when present and is byte-identical to ``utf-8`` when it is not, so this
    is a strict superset -- no existing file's decode changes. Without it a
    BOM-prefixed file's text starts with U+FEFF, which breaks
    ``_parse_frontmatter``'s ``text.startswith("---")`` check and silently
    drops that file's frontmatter (including a ``type: handoff`` classifier).
    """
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, OSError):
        return None


def _plan_snapshots(
    root: Path,
    paths: list[Path],
    *,
    max_chars: int,
    overlap: int,
) -> tuple[SnapshotPlan, dict[str, str], dict[str, list[Chunk]]]:
    """Pre-pass over dated snapshot series only: decide what collapses.

    Runs before the main loop because a chunk's fate depends on the file BEFORE
    it in its series, and the main loop learns that too late to act on it.

    Only snapshot-series files are touched (measured 2026-07-30: 316 of 2814 on
    the live vault) and only chunking happens -- no embedding, which is the cost
    that matters. Their text is handed back so the main loop does not re-read
    them. Measured added cost of this whole pass: 0.099s against a documented
    ~1.8s no-change tick.
    """
    by_rel = {p.relative_to(root).as_posix(): p for p in paths}
    series = group_series(by_rel.keys())
    if not series:
        return EMPTY_PLAN, {}, {}

    texts: dict[str, str] = {}
    chunks_by_rel: dict[str, list[Chunk]] = {}
    keys_by_rel: dict[str, list[str]] = {}
    for rels in series.values():
        for rel in rels:
            text = load_file(by_rel[rel])
            if text is None or not text.strip():
                continue
            chunks = chunk_markdown(text, max_chars=max_chars, overlap=overlap)
            if not chunks:
                continue
            texts[rel] = text
            chunks_by_rel[rel] = chunks
            keys_by_rel[rel] = [chunk_key(c.heading, c.text) for c in chunks]
    return build_plan(series, keys_by_rel), texts, chunks_by_rel


def _parse_frontmatter(text: str) -> dict:
    """Parse a leading YAML frontmatter block. Tolerant by design.

    Frontmatter is an optional Obsidian convention, not a hard contract, so
    absence or invalid YAML must never be fatal -- both return ``{}``, which
    ``_doc_class`` treats identically to "no frontmatter opinion".
    """
    if not text.startswith("---"):
        return {}
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return {}
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            block = "\n".join(lines[1:i])
            try:
                data = yaml.safe_load(block)
            except yaml.YAMLError:
                return {}
            return data if isinstance(data, dict) else {}
    return {}  # no closing delimiter found


def _doc_class(
    rel: str,
    text: str,
    *,
    handoff_mirror_dir: str = DEFAULT_HANDOFF_MIRROR_DIR,
    handoff_mirror_basenames: frozenset[str] | set[str] = DEFAULT_HANDOFF_MIRROR_BASENAMES,
) -> str:
    """Classify a document as "handoff" (session/agent bookkeeping) or "note".

    Primary signal: YAML frontmatter ``type: handoff`` or a ``tags`` list
    containing "handoff" (case-insensitive) -- the durable, content-driven
    design this feature was specced around.

    Fallback: some session-bookkeeping mirrors carry no such frontmatter at
    all, or frontmatter with no ``type``/``tags`` field naming them a
    handoff, so frontmatter alone cannot classify them "handoff". This
    fallback closes that gap from the FILENAME instead: a file directly
    under ``handoff_mirror_dir`` (matched case-insensitively) whose basename
    is either one of ``handoff_mirror_basenames`` exactly, or matches the
    anchored ``_HANDOFF_LEADING_RE``/``_HANDOFF_TRAILING_RE`` "handoff"
    TOKEN patterns (see their definitions for the exact shape), classifies
    "handoff". Deliberately a bounded TOKEN match, never a bare substring
    check, so a title that merely contains the word
    ("handoff-skill-redesign-spec.md") stays "note" while a real
    dated/versioned mirror name ("handoff-alphahive-world-v5.md") still
    matches. Scoped to exactly ``handoff_mirror_dir`` -- never the whole
    corpus -- so an unrelated same-named file elsewhere stays "note".

    ``handoff_mirror_dir``/``handoff_mirror_basenames`` default to
    ``DEFAULT_HANDOFF_MIRROR_DIR``/``DEFAULT_HANDOFF_MIRROR_BASENAMES`` so
    existing behavior is unchanged; a caller can override either via
    :func:`ingest`'s matching parameters (and the CLI's
    ``--handoff-mirror-dir``/``--handoff-mirror-basename`` flags) instead of
    editing this module.
    """
    fm = _parse_frontmatter(text)
    fm_type = str(fm.get("type", "") or "").strip().lower()
    fm_tags = fm.get("tags") or []
    if not isinstance(fm_tags, list):
        fm_tags = [fm_tags]
    fm_tags_lower = {str(t).strip().lower() for t in fm_tags}
    if fm_type == "handoff" or "handoff" in fm_tags_lower:
        return "handoff"

    path = PurePosixPath(rel)
    basename = path.name.lower()
    if path.parent.name.lower() == handoff_mirror_dir.lower() and (
        basename in handoff_mirror_basenames
        or _HANDOFF_LEADING_RE.match(basename)
        or _HANDOFF_TRAILING_RE.search(basename)
    ):
        return "handoff"

    return "note"


def _metadata(rel: str, chunk: Chunk, plan: SnapshotPlan, doc_class: str) -> dict:
    """Citation metadata for one chunk.

    For a snapshot-series chunk this also carries the per-date citability that
    de-duplication would otherwise destroy: the chunk's own date, and the dates
    of the later snapshots whose identical copy was suppressed in its favor.
    """
    meta: dict = {
        "source": rel,
        "heading": chunk.heading if chunk.heading is not None else "",
        "chunk_index": chunk.chunk_index,
        "doc_class": doc_class,
    }
    if plan.is_snapshot(rel):
        dates = plan.repeat_dates(rel, chunk.chunk_index)
        meta["snapshot_date"] = plan.dates.get(rel, "")
        meta["repeat_count"] = 1 + len(dates)
        meta["repeat_dates"] = format_repeat_dates(dates)
    return meta


def ingest(
    root: Path | str,
    store: VectorStore,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    exclude_prefixes: tuple[str, ...] = EXCLUDE_PREFIXES,
    incremental: bool = True,
    store_dir: Path | str | None = None,
    dedupe_snapshots: bool = True,
    handoff_mirror_dir: str = DEFAULT_HANDOFF_MIRROR_DIR,
    handoff_mirror_basenames: frozenset[str] | set[str] = DEFAULT_HANDOFF_MIRROR_BASENAMES,
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
        dedupe_snapshots: when True (default), collapse chunks in a dated
            snapshot series that are byte-identical to the previous snapshot's.
            The first occurrence is always kept and stays attributable to its own
            date; see :mod:`rag_mcp.snapshots`.
        handoff_mirror_dir: parent directory name (not a path) eligible for the
            ``_doc_class`` filename fallback. Defaults to
            :data:`DEFAULT_HANDOFF_MIRROR_DIR` (``"context"``) -- a different
            corpus can override this instead of editing the module.
        handoff_mirror_basenames: basenames (matched case-insensitively) the
            ``_doc_class`` filename fallback treats as handoff mirrors.
            Defaults to :data:`DEFAULT_HANDOFF_MIRROR_BASENAMES`.
    """
    root = Path(root)
    report = IngestReport()
    handoff_mirror_basenames = frozenset(b.lower() for b in handoff_mirror_basenames)

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

    paths = iter_corpus_files(root, exclude_prefixes=exclude_prefixes)
    if dedupe_snapshots:
        plan, snapshot_texts, snapshot_chunks = _plan_snapshots(
            root, paths, max_chars=max_chars, overlap=overlap
        )
    else:
        plan, snapshot_texts, snapshot_chunks = EMPTY_PLAN, {}, {}
    report.chunks_deduped = plan.suppressed

    # Metadata-only refreshes, applied after the loop so an id is never updated
    # before the add that created it.
    refresh_ids: list[str] = []
    refresh_metas: list[dict] = []

    for path in paths:
        report.files_seen += 1
        rel = path.relative_to(root).as_posix()
        text = snapshot_texts.get(rel)
        if text is None:
            text = load_file(path)
        if text is None or not text.strip():
            report.files_skipped += 1
            continue
        digest = content_hash(text)
        seen_rels.add(rel)

        kept_idx = plan.kept.get(rel)  # None => not a snapshot-series file
        hash_same = bool(use_manifest and previous.unchanged(rel, digest))
        prev_ids = previous.stored_ids(rel)
        # True when this file's STORED chunks predate the current metadata
        # rules (e.g. a pre-doc_class-feature manifest, or a future schema
        # bump) -- see CURRENT_METADATA_VERSION. Forces a metadata-only
        # refresh below instead of the plain unchanged-skip.
        meta_stale = use_manifest and previous.meta_version(rel) < CURRENT_METADATA_VERSION

        # Fast path: an ordinary file whose content is unchanged, whose stored
        # id set is the full one, AND whose stored metadata is current. The
        # `kept` check matters when snapshot dedupe gets turned off: those
        # files' suppressed chunks must come BACK, and the content hash alone
        # would happily skip them forever.
        if (
            hash_same
            and kept_idx is None
            and prev_ids == {f"{rel}::{i}" for i in range(previous.chunk_count(rel))}
            and not meta_stale
        ):
            report.files_unchanged += 1
            current.record(rel, digest, previous.chunk_count(rel), meta_version=CURRENT_METADATA_VERSION)
            continue

        chunks = snapshot_chunks.get(rel)
        if chunks is None:
            chunks = chunk_markdown(text, max_chars=max_chars, overlap=overlap)
        if not chunks:
            report.files_skipped += 1
            seen_rels.discard(rel)
            continue

        selected = [chunks[i] for i in kept_idx] if kept_idx is not None else list(chunks)
        new_ids = {f"{rel}::{c.chunk_index}" for c in selected}
        span_h = plan.span_signature(rel)
        # Classified once per file (not per chunk) so every chunk of a doc
        # carries the identical doc_class.
        doc_class = _doc_class(
            rel,
            text,
            handoff_mirror_dir=handoff_mirror_dir,
            handoff_mirror_basenames=handoff_mirror_basenames,
        )

        # An unchanged file only needs the chunks that are MISSING from the store
        # (normally none). Re-embedding the rest to record a metadata change
        # would cost the entire saving the manifest exists to deliver.
        to_add = (
            [c for c in selected if f"{rel}::{c.chunk_index}" not in prev_ids]
            if hash_same
            else selected
        )

        if to_add:
            store.add(
                ids=[f"{rel}::{c.chunk_index}" for c in to_add],
                documents=[c.text for c in to_add],
                metadatas=[_metadata(rel, c, plan, doc_class) for c in to_add],
            )
            report.files_ingested += 1
            report.chunks_added += len(to_add)
        elif hash_same:
            report.files_unchanged += 1
        else:
            # Changed, but every chunk it now produces is a verbatim repeat of
            # the previous snapshot -- processed, nothing new to embed.
            report.files_ingested += 1

        # Everything the manifest says is in the store but should not be:
        # chunks of a file that got shorter, and chunks newly suppressed as
        # duplicates. Upsert alone could never remove either.
        stale = sorted(prev_ids - new_ids)
        if stale:
            store.delete(ids=stale)
            report.chunks_deleted += len(stale)

        # Refresh metadata (never embeddings) for chunks that are already
        # stored and were NOT just (re)written above, in two cases: a
        # survivor's repeat span grew on a day its own file did not change,
        # or this file's stored metadata predates the current schema
        # (meta_stale -- see CURRENT_METADATA_VERSION).
        if meta_stale or (kept_idx is not None and span_h != previous.span_hash(rel)):
            added_ids = {f"{rel}::{c.chunk_index}" for c in to_add}
            for c in selected:
                cid = f"{rel}::{c.chunk_index}"
                if cid in added_ids or cid not in prev_ids:
                    continue  # just written with fresh metadata, or not stored
                refresh_ids.append(cid)
                refresh_metas.append(_metadata(rel, c, plan, doc_class))

        current.record(
            rel,
            digest,
            len(chunks),
            kept=[c.chunk_index for c in selected],
            span_hash=span_h,
            meta_version=CURRENT_METADATA_VERSION,
        )

    if refresh_ids:
        store.update_metadatas(ids=refresh_ids, metadatas=refresh_metas)

    # Files that vanished from the corpus since the last run: upsert can never
    # prune these, which is why a weekly full rebuild was needed to drop them.
    if use_manifest:
        for rel in set(previous.files) - seen_rels:
            gone = sorted(previous.stored_ids(rel))
            if gone:
                store.delete(ids=gone)
                report.chunks_deleted += len(gone)

    if can_persist:
        current.save(store_dir)

    return report
