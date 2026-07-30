"""CLI: ingest a corpus, or run a one-off query against the store.

  python -m rag_mcp.cli ingest <corpus_dir> --db <db_path> [--embedder default|hash]
  python -m rag_mcp.cli query "<text>" --db <db_path> --corpus <corpus_dir> [-k 5]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from .ingest import EXCLUDE_PREFIXES, ingest
from .lock import LockHeld, ReingestLock
from .store import BgeEmbedder, DefaultEmbedder, HashEmbedder, VectorStore


def _embedder(name: str):
    if name == "hash":
        return HashEmbedder()
    if name == "bge":
        return BgeEmbedder()
    return DefaultEmbedder()


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
            shutil.rmtree(db_path)
            print(f"[rag-mcp] --clean: removed store {db_path}", file=sys.stderr)
        store = VectorStore(
            path=str(db_path),
            collection_name=args.collection,
            embedder=_embedder(args.embedder),
        )
        # CLI --exclude flags AUGMENT the built-in defaults; they do not replace them.
        exclude_prefixes = EXCLUDE_PREFIXES + tuple(args.exclude)
        report = ingest(
            args.corpus_dir,
            store,
            exclude_prefixes=exclude_prefixes,
            # --clean just wiped the store (and its manifest), so there is
            # nothing to skip against; keep the flag honest rather than relying
            # on the manifest happening to be gone.
            incremental=not args.full and not args.clean,
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
        args.text, k=args.k, store=store, corpus_root=Path(args.corpus).resolve()
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
        "--quiet",
        action="store_true",
        help="emit the run summary as a single JSON line (for frequent schedules).",
    )
    p_ing.set_defaults(func=_cmd_ingest)

    p_q = sub.add_parser("query", help="query the store")
    p_q.add_argument("text")
    p_q.add_argument("--db", required=True)
    p_q.add_argument("--corpus", required=True, help="corpus root (auth scope)")
    p_q.add_argument("-k", type=int, default=5)
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
