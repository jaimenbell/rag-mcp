"""CLI: ingest a corpus, or run a one-off query against the store.

  python -m rag_mcp.cli ingest <corpus_dir> --db <db_path> [--embedder default|hash]
  python -m rag_mcp.cli query "<text>" --db <db_path> --corpus <corpus_dir> [-k 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .ingest import ingest
from .store import DefaultEmbedder, HashEmbedder, VectorStore


def _embedder(name: str):
    return HashEmbedder() if name == "hash" else DefaultEmbedder()


def _cmd_ingest(args: argparse.Namespace) -> int:
    store = VectorStore(
        path=str(Path(args.db).resolve()),
        collection_name=args.collection,
        embedder=_embedder(args.embedder),
    )
    report = ingest(args.corpus_dir, store)
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
    parser.add_argument("--embedder", default="default", choices=["default", "hash"])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="ingest a corpus dir")
    p_ing.add_argument("corpus_dir")
    p_ing.add_argument("--db", required=True)
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
