"""Ingest manifest: what is already embedded, so re-ingest can skip unchanged files.

Why this exists
---------------
Ingest used to re-read, re-chunk and **re-embed every file on every run**. Chunk
ids are deterministic (``<relative-path>::<chunk_index>``) so the result was
correct -- but the embedding work was redone from scratch, which is the entire
cost of a run (~2.5h for ~50k chunks on the bge CPU path). That forced a
once-daily 03:00 schedule, so a note written at 03:05 stayed invisible to
``search_knowledge`` for nearly 24 hours.

The manifest records a content hash per ingested file. A run then embeds only
what actually changed, which makes runs cheap enough to schedule every few
minutes and collapses the visibility window.

Safety: the manifest is only trusted when the *identity* of the embedding run
matches (embedder, dimension, collection, chunking parameters). Change any of
them -- e.g. the 2026-07-02 MiniLM -> bge cutover -- and every file is treated
as changed, so a re-embed can never be silently skipped into a mixed store.

The manifest lives INSIDE the store directory, so ``--clean`` (which rmtree's
the store) drops it automatically and the two can never outlive each other.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

MANIFEST_FILENAME = "ingest-manifest.json"

# Bump when the on-disk shape changes INCOMPATIBLY; an unrecognized version is
# ignored (treated as "no manifest") rather than mis-parsed.
#
# Deliberately still 1 after the 2026-07-30 snapshot-dedupe change. That change
# adds two OPTIONAL per-file keys (`kept`, `spanh`); a manifest written before it
# simply lacks them and reads as "every chunk index is stored, no spans
# recorded", which is exactly the old behavior. Bumping would have forced the
# live 50k-chunk store through a ~2h43m full re-embed to gain nothing.
MANIFEST_VERSION = 1


def content_hash(text: str) -> str:
    """Stable content fingerprint for a file's decoded text.

    Hashes the DECODED TEXT, not raw bytes, so a pure line-ending change (which
    produces identical chunks) does not force a needless re-embed. Not mtime:
    mtime moves when content does not (touch, sync, restore) and, worse, can
    stay put when content does (same-second writes, archive extraction).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunIdentity:
    """Everything that changes what a file's embedded chunks would look like.

    If any field differs from the stored manifest, the manifest is not
    applicable and every file must be re-embedded.
    """

    embedder: str
    embed_dim: int
    collection: str
    max_chars: int
    overlap: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder,
            "embed_dim": self.embed_dim,
            "collection": self.collection,
            "max_chars": self.max_chars,
            "overlap": self.overlap,
        }


@dataclass
class IngestManifest:
    """Maps corpus-relative path -> what is currently embedded for that file.

    Per file: ``sha256`` (content hash), ``chunks`` (how many chunks the file
    produces), and two OPTIONAL keys added for snapshot de-duplication:

      * ``kept``  - the chunk indices actually embedded. Absent means "all of
        them", which is the only thing that was ever true before dedupe existed.
        This is the record of which ids are IN THE STORE, so pruning is a set
        difference against it rather than a trailing-tail special case.
      * ``spanh`` - fingerprint of the repeat-dates carried by this file's
        surviving chunks. A survivor's span grows every day the repeat continues
        even though the file itself never changes, and this is how an unchanged
        file's metadata still gets refreshed exactly when it needs to be.
    """

    identity: RunIdentity
    files: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- queries ---------------------------------------------------------

    def unchanged(self, rel: str, digest: str) -> bool:
        record = self.files.get(rel)
        return record is not None and record.get("sha256") == digest

    def chunk_count(self, rel: str) -> int:
        record = self.files.get(rel)
        return int(record.get("chunks", 0)) if record else 0

    def stored_indices(self, rel: str) -> tuple[int, ...]:
        """Chunk indices believed to be in the store for *rel*.

        Falls back to ``range(chunks)`` when ``kept`` is absent -- both for
        pre-dedupe manifests and for ordinary (non-snapshot) files, where every
        chunk is stored and writing the list out would just bloat the file.
        """
        record = self.files.get(rel)
        if not record:
            return ()
        kept = record.get("kept")
        if isinstance(kept, list):
            return tuple(kept)
        return tuple(range(self.chunk_count(rel)))

    def stored_ids(self, rel: str) -> set[str]:
        return {f"{rel}::{i}" for i in self.stored_indices(rel)}

    def span_hash(self, rel: str) -> str:
        record = self.files.get(rel)
        if not record:
            return ""
        value = record.get("spanh")
        return value if isinstance(value, str) else ""

    def record(
        self,
        rel: str,
        digest: str,
        chunks: int,
        kept: Sequence[int] | None = None,
        span_hash: str = "",
    ) -> None:
        entry: dict[str, Any] = {"sha256": digest, "chunks": chunks}
        # Only persist `kept` when it actually differs from "everything" -- the
        # manifest covers ~2.8k files and an index list per file would triple it
        # for no information.
        if kept is not None and tuple(kept) != tuple(range(chunks)):
            entry["kept"] = list(kept)
        if span_hash:
            entry["spanh"] = span_hash
        self.files[rel] = entry

    # -- persistence -----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": MANIFEST_VERSION,
                "identity": self.identity.as_dict(),
                "files": self.files,
            },
            indent=1,
            sort_keys=True,
        )

    def save(self, store_dir: Path | str) -> Path:
        """Write atomically so a crash mid-write cannot leave a torn manifest.

        A torn manifest would be worse than none: it could claim a file is
        embedded when it is not, and that file would stay stale forever.
        """
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        target = store_dir / MANIFEST_FILENAME
        fd, tmp_name = tempfile.mkstemp(dir=str(store_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(self.to_json())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)  # atomic on Windows and POSIX
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return target


def load_manifest(store_dir: Path | str, identity: RunIdentity) -> IngestManifest:
    """Load the manifest for *identity*, or an EMPTY one if it cannot be trusted.

    Returns an empty manifest (meaning "re-embed everything") when the file is
    missing, unreadable, malformed, a different schema version, or was written
    by a different run identity. Every failure mode degrades to the old
    full-rebuild behavior -- never to a wrong skip.
    """
    path = Path(store_dir) / MANIFEST_FILENAME
    empty = IngestManifest(identity=identity)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if not isinstance(raw, dict) or raw.get("version") != MANIFEST_VERSION:
        return empty
    if raw.get("identity") != identity.as_dict():
        return empty
    files = raw.get("files")
    if not isinstance(files, dict):
        return empty
    clean: dict[str, dict[str, Any]] = {}
    for rel, record in files.items():
        if not (
            isinstance(rel, str)
            and isinstance(record, dict)
            and isinstance(record.get("sha256"), str)
            and isinstance(record.get("chunks"), int)
        ):
            continue
        entry: dict[str, Any] = {"sha256": record["sha256"], "chunks": record["chunks"]}
        # Optional, added 2026-07-30. Each is dropped independently if malformed:
        # a bad `kept` degrades to "all chunks stored" (so pruning under-deletes
        # rather than deleting live chunks), a bad `spanh` degrades to "" (so the
        # metadata refresh fires once too often rather than never).
        kept = record.get("kept")
        if isinstance(kept, list) and all(isinstance(i, int) for i in kept):
            entry["kept"] = list(kept)
        spanh = record.get("spanh")
        if isinstance(spanh, str):
            entry["spanh"] = spanh
        clean[rel] = entry
    return IngestManifest(identity=identity, files=clean)
