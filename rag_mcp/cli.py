"""CLI: ingest a corpus, or run a one-off query against the store.

  python -m rag_mcp.cli ingest <corpus_dir> --db <db_path> [--embedder default|hash]
  python -m rag_mcp.cli query "<text>" --db <db_path> --corpus <corpus_dir> [-k 5] [--doc-class note|handoff]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .ingest import (
    DEFAULT_HANDOFF_MIRROR_BASENAMES,
    DEFAULT_HANDOFF_MIRROR_DIR,
    EXCLUDE_PREFIXES,
    ingest,
)
from .lock import LockHeld, ReingestLock
from .store import BgeEmbedder, DefaultEmbedder, HashEmbedder, VectorStore


def _embedder(name: str):
    if name == "hash":
        return HashEmbedder()
    if name == "bge":
        return BgeEmbedder()
    return DefaultEmbedder()


# How long --clean will wait for other processes to let go of the store before
# giving up. Must comfortably exceed rag_mcp.server._WATCH_INTERVAL_S (1 s), the
# period at which a running MCP server notices the lock and drops its handle;
# the generous margin covers a server that is mid-query when the lock is taken.
CLEAN_HANDLE_WAIT_S = 120.0
_CLEAN_POLL_S = 2.0


def _rmtree_waiting_for_readers(db_path: Path) -> None:
    """Delete the store dir, waiting out other processes' open handles.

    ROOT CAUSE (2026-08-16 rebuild failure, root-caused 2026-08-21). A Chroma
    PersistentClient holds the collection's HNSW segment files open for the life
    of the client. On Windows that makes the store dir un-deletable
    (PermissionError WinError 32 on data_level0.bin) AND un-renameable
    (WinError 5), so there is no way to sidestep the handle -- the holder must
    actually let go. The long-lived MCP server is such a holder, and it never
    participates in ReingestLock, which only coordinates the two ingest paths
    with each other.

    The other half of this handshake lives in rag_mcp.server: a watcher thread
    sees the lock WE already hold and releases the server's store, and
    _ensure_store refuses to re-open while the lock is live. So by the time we
    are here the holder is already being told to let go; this loop just waits
    for it. We hold the lock throughout, so no new ingest can start meanwhile.

    Raises the original PermissionError if the wait expires, after naming what
    is likely holding it -- a bare traceback here previously told the operator
    nothing about which process to look at.
    """
    deadline = time.monotonic() + CLEAN_HANDLE_WAIT_S
    attempt = 0
    while True:
        attempt += 1
        try:
            shutil.rmtree(db_path)
            if attempt > 1:
                waited = CLEAN_HANDLE_WAIT_S - (deadline - time.monotonic())
                print(
                    f"[rag-mcp] --clean: removed store {db_path} "
                    f"after waiting {waited:.0f}s for {attempt - 1} blocked attempt(s)",
                    file=sys.stderr,
                )
            else:
                print(f"[rag-mcp] --clean: removed store {db_path}", file=sys.stderr)
            return
        except PermissionError as exc:
            if time.monotonic() >= deadline:
                print(
                    f"[rag-mcp] --clean: FAILED to remove {db_path} after "
                    f"{CLEAN_HANDLE_WAIT_S:.0f}s -- another process still holds "
                    f"{getattr(exc, 'filename', 'a file in the store')} open. "
                    "The usual holder is a long-running rag-mcp MCP server "
                    "(run_server.py); it should release automatically, so a "
                    "server predating this fix, or an unrelated reader, is the "
                    "likely cause.",
                    file=sys.stderr,
                )
                raise
            time.sleep(_CLEAN_POLL_S)


def _cmd_ingest(args: argparse.Namespace) -> int:
    db_path = Path(args.db).resolve()
    # Cross-process mutex: both the daily upsert and the weekly --clean rebuild
    # acquire the SAME lock. If a live run holds it, fail fast (non-zero) and let
    # the scheduler retry next cycle -- never queue/block. The destructive --clean
    # delete happens strictly AFTER acquisition so it can never race a live write.
    try:
        lock = ReingestLock(db_path).acquire()
    except LockHeld as exc:
        print(f"[rag-mcp] SKIP: {exc}", file=sys.stderr)
        return 3
    try:
        if args.clean and db_path.exists():
            _rmtree_waiting_for_readers(db_path)
        store = VectorStore(
            path=str(db_path),
            collection_name=args.collection,
            embedder=_embedder(args.embedder),
        )
        # CLI --exclude flags AUGMENT the built-in defaults; they do not replace them.
        exclude_prefixes = EXCLUDE_PREFIXES + tuple(args.exclude)
        handoff_mirror_basenames = DEFAULT_HANDOFF_MIRROR_BASENAMES | {
            b.lower() for b in args.handoff_mirror_basename
        }
        report = ingest(
            args.corpus_dir,
            store,
            exclude_prefixes=exclude_prefixes,
            # --clean just wiped the store (and its manifest), so there is
            # nothing to skip against; keep the flag honest rather than relying
            # on the manifest happening to be gone.
            incremental=not args.full and not args.clean,
            dedupe_snapshots=not args.no_snapshot_dedupe,
            handoff_mirror_dir=args.handoff_mirror_dir,
            handoff_mirror_basenames=handoff_mirror_basenames,
        )
    finally:
        lock.release()
    summary = {
        "files_seen": report.files_seen,
        "files_ingested": report.files_ingested,
        "files_unchanged": report.files_unchanged,
        "files_skipped": report.files_skipped,
        "chunks_added": report.chunks_added,
        "chunks_deleted": report.chunks_deleted,
        "chunks_deduped": report.chunks_deduped,
        "incremental": report.incremental,
        "store_count": store.count(),
    }
    # --quiet keeps a high-frequency schedule's log readable: one line per run
    # instead of a 10-line block (at a 15-minute cadence that is ~96 runs/day).
    print(json.dumps(summary) if args.quiet else json.dumps(summary, indent=2))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    from .search import search_knowledge

    store = VectorStore(
        path=str(Path(args.db).resolve()),
        collection_name=args.collection,
        embedder=_embedder(args.embedder),
    )
    result = search_knowledge(
        args.text,
        k=args.k,
        store=store,
        corpus_root=Path(args.corpus).resolve(),
        doc_class=args.doc_class,
    )
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag_mcp.cli")
    parser.add_argument("--collection", default="knowledge")
    parser.add_argument(
        "--embedder", default="default", choices=["default", "hash", "bge"]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="ingest a corpus dir")
    p_ing.add_argument("corpus_dir")
    p_ing.add_argument("--db", required=True)
    p_ing.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "vault-relative POSIX prefix to exclude from ingestion (repeatable). "
            "These AUGMENT the built-in defaults (infrastructure/claude-config-backup/) "
            "rather than replacing them."
        ),
    )
    p_ing.add_argument(
        "--clean",
        action="store_true",
        help=(
            "clean rebuild: delete the store dir before ingesting to prune chunks "
            "for deleted/renamed vault notes. The delete runs AFTER the reingest "
            "lock is acquired, so it can never race a concurrent write."
        ),
    )
    p_ing.add_argument(
        "--full",
        action="store_true",
        help=(
            "re-embed every file, ignoring the incremental manifest. Ingest is "
            "incremental by DEFAULT: files whose content hash is unchanged since "
            "the last run are not re-embedded, which is what makes frequent "
            "scheduling affordable. Use --full to force a rebuild in place "
            "(unlike --clean it does not delete the store first)."
        ),
    )
    p_ing.add_argument(
        "--no-snapshot-dedupe",
        action="store_true",
        help=(
            "embed every chunk of every dated snapshot file, including ones "
            "byte-identical to the previous day's. Dedupe is ON by default: "
            "daily report series (fleet-health-YYYY-MM-DD and friends) repeat "
            "unchanged status blocks verbatim, which crowds top-k with copies of "
            "one line. The first occurrence is always kept and the collapsed "
            "dates are recorded in each surviving chunk's metadata, so per-date "
            "questions still answer. Turning this off re-embeds them on the next "
            "run."
        ),
    )
    p_ing.add_argument(
        "--quiet",
        action="store_true",
        help="emit the run summary as a single JSON line (for frequent schedules).",
    )
    p_ing.add_argument(
        "--handoff-mirror-dir",
        default=DEFAULT_HANDOFF_MIRROR_DIR,
        metavar="DIRNAME",
        help=(
            "parent directory name (not a path) eligible for the doc_class "
            f"filename fallback. Default: {DEFAULT_HANDOFF_MIRROR_DIR!r} (the "
            "vault convention)."
        ),
    )
    p_ing.add_argument(
        "--handoff-mirror-basename",
        action="append",
        default=[],
        metavar="BASENAME",
        help=(
            "extra basename (case-insensitive) to treat as a handoff mirror in "
            "doc_class classification (repeatable). AUGMENTS the built-in "
            f"defaults ({sorted(DEFAULT_HANDOFF_MIRROR_BASENAMES)}) rather than "
            "replacing them -- same pattern as --exclude."
        ),
    )
    p_ing.set_defaults(func=_cmd_ingest)

    p_q = sub.add_parser("query", help="query the store")
    p_q.add_argument("text")
    p_q.add_argument("--db", required=True)
    p_q.add_argument("--corpus", required=True, help="corpus root (auth scope)")
    p_q.add_argument("-k", type=int, default=5)
    p_q.add_argument(
        "--doc-class",
        dest="doc_class",
        default=None,
        help=(
            "restrict results to chunks with this exact doc_class "
            "(e.g. \"note\" to exclude session/agent handoff bookkeeping "
            "mirrors). Omit for no filter."
        ),
    )
    p_q.set_defaults(func=_cmd_query)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    # Root-caused 2026-07-12: after the bge cutover (fastembed/onnxruntime +
    # chromadb), the interpreter can hang past a clean `main()` return because
    # those libs leave non-daemon background threads alive -- normal
    # interpreter shutdown (which sys.exit() triggers) blocks joining them.
    # The scheduled reingest/reingest-clean tasks then run past their
    # ExecutionTimeLimit and get hard-killed by Task Scheduler, leaving the
    # cross-process ReingestLock held by a still-"alive" PID for up to an
    # hour -- which starves the weekly --clean run 30 min later (LockHeld)
    # and, via the shared log file staying open that whole time, silently
    # drops its log output too. All ingest work (incl. lock release) and
    # stdout/stderr output are already complete by the time main() returns,
    # so skip the graceful-shutdown thread joins entirely via os._exit();
    # flush explicitly first since os._exit() bypasses the atexit stdio flush.
    _exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_code)
