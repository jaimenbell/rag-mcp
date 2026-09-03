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

import hashlib
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

# Leading-stem filename matcher (REWRITTEN 2026-09-03, RM-fixafter2 slice 2 --
# replaces the leading+trailing regex pair from the same-day doc_class fix).
#
# GRAMMAR (one sentence, this IS the spec -- see tests/test_ingest.py's
# leading-stem-grammar section for the FIRES/SILENT proof): a basename
# (extension stripped, any of MARKDOWN_EXTS) classifies "handoff" iff its
# FIRST '-'/'_'/'.'-delimited token -- optionally after a YYYY-MM-DD date
# prefix -- case-insensitively equals one of `_MIRROR_STEMS`. If more of the
# basename follows the stem, it MUST start with its own '-'/'_'/'.' separator
# (that is what makes the stem a bounded TOKEN, not a prefix) -- there is no
# constraint on the SHAPE of that trailing text once the separator is there.
# A basename where the stem is immediately followed by more text with NO
# separator ("handoff2026.md") does NOT match -- "handoff2026" is one
# token, not the stem "handoff". "." is deliberately IN the separator class
# (finding #9, RM-fixafter3) -- a versioned mirror name like "handoff.v2.md"
# is a realistic filename shape; it was narrowed to "[-_]" in the slice-2
# rewrite above with no test covering "." at all, which silently flipped
# that shape from "handoff" to "note".
#
# Two deliberate changes from the prior design:
#   1. Stems widened from {"handoff"} to {"handoff", "active", "resume"} --
#      the prior design only recognized "handoff" as a LEADING token; real
#      vault mirrors named "RESUME-world-v6.md" / "ACTIVE-projectx.md" were
#      false negatives because the exact-basename set only covered the bare
#      "resume.md"/"active.md" forms, not a versioned/dated variant.
#   2. The TRAILING form ("word-handoff.md") is DROPPED, and the LEADING
#      form's suffix is no longer constrained to a narrow shape (digit-led /
#      <=3-char code / -vN / trailing date). The narrow shape was itself
#      under-inclusive (RESUME-world-v6.md, handoff-projectx.md,
#      resume-2026-09-03.md all missed it) while the trailing form was
#      OVER-inclusive (context/notes-on-handoff.md -> handoff, a live false
#      positive). Trading the narrow-shape precision for a simple
#      first-token rule closes both: leading-only removes the trailing false
#      positive, and dropping the shape constraint removes the leading false
#      negatives. The accepted cost is a title that merely STARTS with a stem
#      word (e.g. "handoff-skill-redesign-spec.md") now also matches --
#      acceptable because doc_class is a convenience filter scoped to
#      `handoff_mirror_dir`, not a security boundary.
_MIRROR_STEMS = ("handoff", "active", "resume")
DEFAULT_HANDOFF_MIRROR_STEMS = _MIRROR_STEMS
_MIRROR_STEM_ALT = "|".join(_MIRROR_STEMS)
_MARKDOWN_EXT_ALT = "|".join(re.escape(ext) for ext in MARKDOWN_EXTS)
_HANDOFF_LEADING_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}[-_ ])?(?:"
    + _MIRROR_STEM_ALT
    + r")(?:[-_.].*)?(?:"
    + _MARKDOWN_EXT_ALT
    + r")$",
    re.IGNORECASE,
)


def _handoff_leading_re(stems: tuple[str, ...]) -> re.Pattern[str]:
    """Leading-stem regex for *stems* -- reuses the precompiled default.

    Finding #6 (RM-fixafter3): `_MIRROR_STEMS` {"handoff", "active",
    "resume"} matched as a free leading token independently of
    `handoff_mirror_basenames`, with no way to opt a corpus out of "active"/
    "resume" specifically (a real business file like
    "context/active-clients-2026.md" always classified "handoff"). Compiling
    per-call for a non-default `stems` keeps the common (default) case at
    the original precompiled-constant cost.
    """
    if stems == _MIRROR_STEMS:
        return _HANDOFF_LEADING_RE
    stem_alt = "|".join(re.escape(s) for s in stems)
    return re.compile(
        r"^(?:\d{4}-\d{2}-\d{2}[-_ ])?(?:"
        + stem_alt
        + r")(?:[-_.].*)?(?:"
        + _MARKDOWN_EXT_ALT
        + r")$",
        re.IGNORECASE,
    )


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
# otherwise unchanged (same content hash) but whose stored `metav` is stale
# relative to `_effective_metadata_version()` (see below) does a
# METADATA-ONLY refresh -- re-chunk (cheap) and `store.update_metadatas` --
# without re-embedding (expensive, and the whole point of the incremental
# path). A pre-feature manifest has no `metav` at all, which
# `IngestManifest.meta_version` reads as 0 -- always stale -- so the very
# next incremental run backfills every file exactly once. Do NOT bump
# `manifest.MANIFEST_VERSION` for this: that forces a full re-embed, which a
# metadata-only change never needs.
CURRENT_METADATA_VERSION = 1


def _is_bare_dirname(value: str) -> bool:
    """True iff *value*, once stripped, is a single bare path component.

    `_doc_class`'s directory comparison is against `path.parent.name`, which
    pathlib always yields as a bare component (no slashes) for a real file's
    relative path. A caller-supplied `handoff_mirror_dir` that isn't ALSO
    bare -- "context/" (trailing slash), "/context" (leading slash), "a/b"
    (embedded separator), or "" (empty) -- can therefore never equal it:
    instead of raising, it silently disables the WHOLE filename fallback for
    every file in the corpus (finding #4, RM-fixafter3). Used both as
    `ingest()`'s outer guard and as `_doc_class`'s own belt-and-braces check
    for callers that bypass `ingest()`.
    """
    value = value.strip()
    return bool(value) and PurePosixPath(value).name == value


def _classifier_config_fingerprint(
    handoff_mirror_dir: str,
    handoff_mirror_basenames: frozenset[str],
    handoff_mirror_stems: tuple[str, ...] = DEFAULT_HANDOFF_MIRROR_STEMS,
) -> int:
    """Stable small int fingerprint of the doc_class classifier config.

    Finding #2: `meta_stale` keyed only off `CURRENT_METADATA_VERSION` means
    changing `handoff_mirror_dir`/`handoff_mirror_basenames` against an
    already-embedded store was a silent no-op -- the store kept whatever
    doc_class the OLD config produced until an operator remembered to run
    --full/--clean. Folding this fingerprint into the effective metav (see
    `_effective_metadata_version`) makes a config change itself a
    metadata-stale trigger, with no extra machinery: the existing
    metadata-only-refresh path (re-chunk + `update_metadatas`, no re-embed)
    just fires for a different reason.

    Finding #3 (RM-fixafter3): the CONFIG VALUES alone don't cover the
    classifier's GRAMMAR -- an edit to the stems tuple, the leading-stem
    regex skeleton (date prefix / separator class), or MARKDOWN_EXTS was a
    silent no-op too. `_handoff_leading_re(handoff_mirror_stems).pattern` is
    the fully-built regex text, which already embeds all three (stems are
    interpolated into the alternation, MARKDOWN_EXTS into the extension
    alternation, and the skeleton is the rest of the literal pattern) --
    hashing that one string covers all three at once, and ties the
    fingerprint directly to `_doc_class`'s actual matching behavior rather
    than to a hand-picked list of "the constants that mattered as of this
    writing" that a future edit could add a fourth one to and miss.
    """
    payload = "\x1f".join(
        [
            handoff_mirror_dir.strip().lower(),
            *sorted(handoff_mirror_basenames),
            _handoff_leading_re(handoff_mirror_stems).pattern,
        ]
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _effective_metadata_version(
    handoff_mirror_dir: str,
    handoff_mirror_basenames: frozenset[str],
    handoff_mirror_stems: tuple[str, ...] = DEFAULT_HANDOFF_MIRROR_STEMS,
) -> int:
    """The metav value staleness is compared against for THIS run's config.

    Combines `CURRENT_METADATA_VERSION` (a schema-shape bump) with the
    classifier config's fingerprint (a config-value AND grammar change) into
    one int, so a single `!=` comparison against the stored `metav` catches
    any kind of drift. Deliberately `!=`, not `<` -- a fingerprint has no
    meaningful ordering, only equality.
    """
    return (
        CURRENT_METADATA_VERSION << 32
    ) + _classifier_config_fingerprint(
        handoff_mirror_dir, handoff_mirror_basenames, handoff_mirror_stems
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
    # Snapshot-dedupe counter. STANDING, not a delta: how many chunks across the
    # whole corpus are verbatim repeats of the previous snapshot in their series
    # and therefore deliberately not embedded. Steady on a quiet day; it is the
    # size of the collapse, not the work this run did.
    chunks_deduped: int = 0
    # Metadata-only refresh counter (finding #9): chunks whose metadata was
    # rewritten via store.update_metadatas() this run WITHOUT a re-embed --
    # the backfill triggered by CURRENT_METADATA_VERSION / a classifier
    # config change (see meta_stale). Zero on a quiet run; lets an operator
    # see a backfill happened from the --quiet JSON summary line alone,
    # without diffing manifests by hand.
    chunks_metadata_refreshed: int = 0
    # Refresh ids the post-update presence check (finding #12) found still
    # missing from the store after store.update_metadatas() -- a manifest/
    # store desync. That file's metav is rolled back (not stamped current)
    # so the next incremental run retries it. Zero on a synced store.
    chunks_metadata_missing: int = 0


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
    handoff_mirror_stems: tuple[str, ...] = DEFAULT_HANDOFF_MIRROR_STEMS,
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
    anchored ``_HANDOFF_LEADING_RE`` "mirror stem" pattern (see its
    definition for the exact grammar -- leading-stem-only, {"handoff",
    "active", "resume"}, unconstrained trailing text once separated from
    the stem by its own "-"/"_"). Scoped to exactly
    ``handoff_mirror_dir`` -- never the whole corpus -- so an unrelated
    same-named file elsewhere stays "note", and to the stem appearing as the
    FIRST token, so a title that merely CONTAINS the word later
    ("research-handoff.md") stays "note".

    ``handoff_mirror_dir``/``handoff_mirror_basenames``/``handoff_mirror_stems``
    default to ``DEFAULT_HANDOFF_MIRROR_DIR``/``DEFAULT_HANDOFF_MIRROR_BASENAMES``/
    ``DEFAULT_HANDOFF_MIRROR_STEMS`` so existing behavior is unchanged; a
    caller can override any of them via :func:`ingest`'s matching parameters
    (and the CLI's ``--handoff-mirror-dir``/``--handoff-mirror-basename``
    flags for the first two) instead of editing this module.
    ``handoff_mirror_stems`` narrows the free leading-token match without
    touching ``handoff_mirror_basenames`` -- e.g. a corpus that uses
    "active-"/"resume-" filenames for something other than session mirrors
    can pass ``handoff_mirror_stems=("handoff",)`` to opt those two out
    while keeping exact basenames (``active.md``, ``resume.md``) working via
    ``handoff_mirror_basenames`` as before.
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
    # Belt-and-braces (finding #4, RM-fixafter3): `ingest()`'s own guard is
    # the primary defense, but `_doc_class` has a public default and is
    # callable directly -- a malformed handoff_mirror_dir ("context/",
    # "/context") can never equal `path.parent.name` (always bare), so
    # degrade safely to "no match" rather than trusting the caller.
    mirror_dir = handoff_mirror_dir.strip()
    leading_re = _handoff_leading_re(handoff_mirror_stems)
    if _is_bare_dirname(mirror_dir) and path.parent.name.lower() == mirror_dir.lower() and (
        basename in handoff_mirror_basenames or leading_re.match(basename)
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
    handoff_mirror_stems: tuple[str, ...] = DEFAULT_HANDOFF_MIRROR_STEMS,
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
        handoff_mirror_stems: leading-token stems (matched case-insensitively,
            free trailing text) the ``_doc_class`` filename fallback treats as
            handoff mirrors, independently of ``handoff_mirror_basenames``.
            Defaults to :data:`DEFAULT_HANDOFF_MIRROR_STEMS` (``("handoff",
            "active", "resume")``) -- narrow this (e.g. to ``("handoff",)``)
            for a corpus that uses "active-"/"resume-" filenames for
            something other than session mirrors.
    """
    if not handoff_mirror_dir or not handoff_mirror_dir.strip():
        raise ValueError(
            "handoff_mirror_dir must not be empty/whitespace-only -- an empty "
            "value widens the doc_class filename fallback to the whole corpus "
            "root (PurePosixPath('root.md').parent.name == '' matches '')."
        )
    if not _is_bare_dirname(handoff_mirror_dir):
        raise ValueError(
            "handoff_mirror_dir must be a single bare directory name -- no "
            "leading/trailing slash, no embedded path separator (e.g. "
            "'context', not 'context/' or '/context'). `path.parent.name` "
            "(what _doc_class compares against) is always bare, so a "
            "malformed value would otherwise silently disable the whole "
            "filename fallback for every file instead of raising (finding #4)."
        )
    # Finding #2 (RM-fixafter3): normalize ONCE here so every downstream use
    # of handoff_mirror_dir -- the classifier-config fingerprint (which folds
    # in .strip().lower()) AND _doc_class's own directory comparison (which
    # only .lower()'d it) -- sees the identical value. Before this, a padded
    # value like " context " produced the SAME metav as "context" (the
    # fingerprint strips) while classifying every real file differently
    # (doc_class compared unstripped) -- a silent drift no refresh would catch.
    handoff_mirror_dir = handoff_mirror_dir.strip()

    root = Path(root)
    report = IngestReport()
    handoff_mirror_basenames = frozenset(b.lower() for b in handoff_mirror_basenames)
    effective_metav = _effective_metadata_version(
        handoff_mirror_dir, handoff_mirror_basenames, handoff_mirror_stems
    )

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
    # id -> owning rel, so a post-update presence check (finding #12) can
    # roll back the right file's metav if its refresh did not fully land.
    refresh_rel_of_id: dict[str, str] = {}

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
        # rules (e.g. a pre-doc_class-feature manifest, a future schema
        # bump, OR a changed handoff_mirror_dir/handoff_mirror_basenames --
        # see _effective_metadata_version). Forces a metadata-only refresh
        # below instead of the plain unchanged-skip.
        meta_stale = use_manifest and previous.meta_version(rel) != effective_metav

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
            current.record(rel, digest, previous.chunk_count(rel), meta_version=effective_metav)
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
            handoff_mirror_stems=handoff_mirror_stems,
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
        # or this file's stored metadata predates the current schema/config
        # (meta_stale -- see _effective_metadata_version).
        if meta_stale or (kept_idx is not None and span_h != previous.span_hash(rel)):
            added_ids = {f"{rel}::{c.chunk_index}" for c in to_add}
            for c in selected:
                cid = f"{rel}::{c.chunk_index}"
                if cid in added_ids or cid not in prev_ids:
                    continue  # just written with fresh metadata, or not stored
                refresh_ids.append(cid)
                refresh_metas.append(_metadata(rel, c, plan, doc_class))
                refresh_rel_of_id[cid] = rel

        current.record(
            rel,
            digest,
            len(chunks),
            kept=[c.chunk_index for c in selected],
            span_hash=span_h,
            meta_version=effective_metav,
        )

    if refresh_ids:
        store.update_metadatas(ids=refresh_ids, metadatas=refresh_metas)
        # Chroma's collection.update() silently ignores ids it doesn't have
        # (warns, does not raise) -- verify presence rather than trusting the
        # lack of an exception (finding #12). A manifest/store desync (an
        # out-of-band delete, a crash between a prior add and manifest.save)
        # would otherwise bake a permanently-missing doc_class in behind an
        # exit-0 run: current.record() above already stamped this file's
        # metav as current, so without this check the next incremental run
        # would believe the refresh landed and never retry it.
        present = store.existing_ids(refresh_ids)
        missing = [cid for cid in refresh_ids if cid not in present]
        if missing:
            # Finding #1 (RM-fixafter3): popping `metav` alone forces
            # meta_stale again next run, but `prev_ids` on that next run is
            # built from `kept`/`chunks`, which still claim the missing id is
            # stored -- so `to_add` never includes it and every subsequent
            # run repeats the same no-op metadata-refresh attempt forever
            # (the id genuinely does not exist, so update_metadatas() keeps
            # silently skipping it). Make the manifest's belief match
            # reality: drop the missing indices from this file's
            # stored-index set too, so the next incremental run's `to_add`
            # sees them as genuinely absent and re-embeds them.
            missing_by_rel: dict[str, set[str]] = {}
            for cid in missing:
                missing_by_rel.setdefault(refresh_rel_of_id[cid], set()).add(cid)
            for rel, missing_ids_for_rel in missing_by_rel.items():
                entry = current.files[rel]
                entry.pop("metav", None)
                chunk_total = int(entry.get("chunks", 0))
                stored_now = entry.get("kept")
                stored_indices = (
                    list(stored_now)
                    if isinstance(stored_now, list)
                    else list(range(chunk_total))
                )
                missing_indices = {int(cid[len(rel) + 2 :]) for cid in missing_ids_for_rel}
                entry["kept"] = [i for i in stored_indices if i not in missing_indices]
            report.chunks_metadata_missing = len(missing)
        report.chunks_metadata_refreshed = len(refresh_ids) - len(missing)

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
