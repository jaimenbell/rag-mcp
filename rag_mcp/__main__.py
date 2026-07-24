"""Console entry point for the rag-mcp MCP server.

Wires `python -m rag_mcp` and the `rag-mcp` console script (see
[project.scripts] in pyproject.toml) to the stdio server in rag_mcp.server.

Unlike rag_mcp.server._main's warmup (best-effort by design: it lets an MCP
client spawn the server even before the corpus store exists, deferring errors
to the per-call fail-soft path), this entry point validates configuration
EAGERLY, before the stdio transport starts, and fails LOUD -- a clear,
actionable stderr message, never a raw traceback -- when it is missing.

No corpus root / db path is hardcoded here: everything comes from the
RAG_MCP_* env vars via rag_mcp.config.Config, so this behaves identically on
any machine (this repo is secret-scrubbed; a hardcoded default would either
silently work only on the author's machine or leak a local path).

run_server.py at the repo root remains the operational wrapper used by
mcp.yaml / the local mcp-factory hub; it is unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .config import Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-mcp",
        description=(
            "rag-mcp MCP server (stdio transport). Configure via the "
            "RAG_MCP_CORPUS_ROOT and RAG_MCP_DB_PATH environment variables "
            "(see README's Configuration section) before starting."
        ),
    )
    parser.parse_args(argv)  # only supports -h/--help; unknown args -> exit 2

    try:
        Config.from_env()
    except ValueError as exc:
        print(f"[rag-mcp] configuration error: {exc}", file=sys.stderr)
        print(
            "[rag-mcp] set RAG_MCP_CORPUS_ROOT and RAG_MCP_DB_PATH before "
            "starting the server (see README's Configuration section).",
            file=sys.stderr,
        )
        return 1

    from .server import _main as _server_main

    asyncio.run(_server_main())
    return 0


if __name__ == "__main__":
    sys.exit(main())
