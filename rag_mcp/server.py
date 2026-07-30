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

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .search import MAX_K, search_knowledge

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
    # snake_case as of mcp 2.0.0 (the camelCase alias still constructs, but
    # attribute access is snake_case only); serializes to "inputSchema" on the wire.
    input_schema={
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


async def list_tools(
    ctx: Any = None, params: types.PaginatedRequestParams | None = None
) -> types.ListToolsResult:
    return types.ListToolsResult(tools=[_TOOL])


async def call_tool(
    ctx: Any, params: types.CallToolRequestParams
) -> types.CallToolResult:
    if params.name != "search_knowledge":
        payload: dict[str, Any] = {
            "ok": False,
            "error": {"type": "unknown_tool", "message": f"Unknown tool: {params.name}"},
            "results": [],
        }
    else:
        # Run the (synchronous, potentially slow: embed + vector query) search off
        # the event-loop thread. Blocking the loop here starves the stdio transport
        # streams -- the response can't be written until the handler yields.
        payload = await anyio.to_thread.run_sync(_run_search, params.arguments or {})
    # Fail-soft contract unchanged: an unknown tool / config error is a structured
    # ok:false PAYLOAD, not a protocol-level error, so is_error stays False.
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))]
    )


# mcp 2.0.0 replaced the @server.list_tools()/@server.call_tool() decorators with
# constructor handler kwargs (handlers take (ctx, params) and return Result models).
server = Server("rag-mcp", on_list_tools=list_tools, on_call_tool=call_tool)


def _warm() -> None:
    """Import heavy deps + open the store once, at startup, before transport.

    The first call to _ensure_store lazily triggers chromadb's -- hence numpy's --
    first C-extension import. Doing that on the event-loop thread DURING a tool
    call deadlocks on Windows: by then an anyio worker thread is blocked in a
    native stdin.readline(), and numpy's first `import` never completes, so the
    handler never returns a response (the client sees a hang / BrokenResourceError).

    Forcing the import + store open here -- before stdio_server() starts its
    reader thread -- makes every later call a no-op import and cannot deadlock.
    Best-effort: a missing/invalid store must not stop the server from starting;
    the per-call fail-soft path still returns a structured error in that case.
    """
    try:
        _ensure_store()
    except Exception:  # noqa: BLE001 - warmup is best-effort; call_tool re-checks.
        pass


async def _main() -> None:
    # Warm before the stdio transport (and its worker threads) exist, so the
    # first-time numpy/chromadb import can never race a blocked stdin.readline().
    await anyio.to_thread.run_sync(_warm)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(_main())
