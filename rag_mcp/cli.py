"""CLI: ingest a corpus, or run a one-off query against the store.

  python -m rag_mcp.cli ingest <corpus_dir> --db <db_path> [--embedder default|hash]
  python -m rag_mcp.cli query "<text>" --db <db_path> --corpus <corpus_dir> [-k 5]
"""
from __future__ import annotations

import argparse
import json
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
        report = ingest(args.corpus_dir, store, exclude_prefixes=exclude_prefixes)
    finally:
        lock.release()
    print(
        json.dumps(
            {
                "files_seen": report.files_seen,
                "files_ingested": report.files_ingested,
                "files_skipped": report.files_skipped,
                "chunks_added": report.chunks_added,
                "store_count": store.count(),
            },
            indent=2,
        )
    )
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
    sys.exit(main())
