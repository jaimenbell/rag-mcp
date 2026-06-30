#!/usr/bin/env python3
r"""rag-mcp MCP server - exposes a single retrieval tool over stdio.

Tool: search_knowledge(query, k) -> top-k chunks with citations.

Reliability (see rag_mcp.search): auth-scoped to the configured corpus root,
fail-soft (structured errors, never crashes the agent), version-pinned deps.

Configured via env (see rag_mcp.config):
  RAG_MCP_CORPUS_ROOT, RAG_MCP_DB_PATH, RAG_MCP_COLLECTION, RAG_MCP_EMBEDDER

Expressible in the mcp-factory manifest model via mcp.yaml (runtime.script -> this file).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .search import MAX_K, search_knowledge

server = Server("rag-mcp")

# Lazily-opened store/root so importing the module never touches the model or disk.
_STATE: dict[str, Any] = {"store": None, "root": None}

_TOOL = types.Tool(
    name="search_knowledge",
    description=(
        "Retrieve the most relevant passages from the configured knowledge corpus for a "
        "natural-language query. Returns the passage text plus a CITATION (source file + "
        "heading + chunk index) for each hit so answers are traceable. Auth-scoped to the "
        "corpus root and fail-soft: a down/empty store returns a structured error, never an "
        "exception. Use when asked to look something up in the knowledge base / docs."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
            },
            "k": {
                "type": "number",
                "description": f"Max passages to return, 1-{MAX_K} (default 5).",
            },
        },
        "required": ["query"],
    },
)


def _ensure_store():
    if _STATE["store"] is None:
        # Imported lazily so a missing/invalid config surfaces as a structured
        # error from call_tool rather than an import-time crash.
        from .config import Config

        cfg = Config.from_env()
        _STATE["store"] = cfg.open_store()
        _STATE["root"] = cfg.corpus_root
    return _STATE["store"], _STATE["root"]


def _run_search(arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        store, root = _ensure_store()
    except Exception as exc:  # noqa: BLE001 - config/init failure -> structured error
        return {
            "ok": False,
            "error": {"type": "config_error", "message": str(exc)},
            "results": [],
        }
    return search_knowledge(
        arguments.get("query", ""),
        k=arguments.get("k", 5),
        store=store,
        corpus_root=root,
    )


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [_TOOL]


@server.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if name != "search_knowledge":
        payload: dict[str, Any] = {
            "ok": False,
            "error": {"type": "unknown_tool", "message": f"Unknown tool: {name}"},
            "results": [],
        }
    else:
        payload = _run_search(arguments or {})
    return [types.TextContent(type="text", text=json.dumps(payload))]


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(_main())
