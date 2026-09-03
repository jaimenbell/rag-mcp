#!/usr/bin/env python3
r"""rag-mcp MCP server - exposes a single retrieval tool over stdio.

Tool: search_knowledge(query, k, doc_class=None) -> top-k chunks with citations.

Reliability (see rag_mcp.search): auth-scoped to the configured corpus root,
fail-soft (structured errors, never crashes the agent), version-pinned deps.

Configured via env (see rag_mcp.config):
  RAG_MCP_CORPUS_ROOT, RAG_MCP_DB_PATH, RAG_MCP_COLLECTION, RAG_MCP_EMBEDDER

Expressible in the mcp-factory manifest model via mcp.yaml (runtime.script -> this file).
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .lock import live_holder
from .search import ALLOWED_DOC_CLASSES, MAX_K, search_knowledge

# Lazily-opened store/root so importing the module never touches the model or disk.
_STATE: dict[str, Any] = {"store": None, "root": None, "db_path": None}

# Guards _STATE against the reindex watcher thread racing a tool call. The store
# is opened and closed from two threads now, so "check then use" is no longer
# safe without it.
_LOCK = threading.Lock()

# How often the watcher asks "is a reingest running?". The reingest's own retry
# window (rag_mcp.cli) must comfortably exceed this -- that pairing is what makes
# the handshake work, and both sides say so.
_WATCH_INTERVAL_S = 1.0


class ReindexInProgress(RuntimeError):
    """A --clean rebuild currently owns the store, so there is nothing to read.

    Distinct from a config error on purpose: this is transient and self-healing,
    and reporting it as a config failure would send a reader off debugging their
    environment during what is really a scheduled maintenance window.
    """

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
            "doc_class": {
                "type": "string",
                # Derived from ALLOWED_DOC_CLASSES (finding #11) so this
                # schema can never drift from what search_knowledge()
                # actually accepts -- sorted for a stable, readable wire shape.
                "enum": sorted(ALLOWED_DOC_CLASSES),
                "description": (
                    "Optional metadata filter: restrict results to chunks with this "
                    "exact doc_class (e.g. \"note\" to exclude session/agent handoff "
                    "bookkeeping mirrors). Case-sensitive; a value outside this enum "
                    "returns an invalid_doc_class error rather than an empty result. "
                    "Omit for no filter."
                ),
            },
        },
        "required": ["query"],
    },
)


def _release_store() -> bool:
    """Drop this process's handle on the store. Returns True if one was held.

    Clearing _STATE BEFORE closing is deliberate: a concurrent tool call must
    never be handed a store that is mid-close.
    """
    with _LOCK:
        store = _STATE["store"]
        _STATE["store"] = None
    if store is None:
        return False
    try:
        store.close()
    except Exception:  # noqa: BLE001 - releasing must never take the server down
        pass
    return True


def _reindex_running(db_path) -> bool:
    """True while a live reingest owns the store (rag_mcp.lock.ReingestLock).

    Reuses live_holder rather than re-deriving liveness, so this can never
    disagree with what the lock itself considers a live holder -- a second,
    drifting definition of "running" is how this kind of check starts lying.
    """
    try:
        return db_path is not None and live_holder(db_path) is not None
    except Exception:  # noqa: BLE001 - an unreadable lock must not stop reads
        return False


def _ensure_store():
    with _LOCK:
        if _STATE["store"] is not None:
            return _STATE["store"], _STATE["root"]

    # Imported lazily so a missing/invalid config surfaces as a structured
    # error from call_tool rather than an import-time crash.
    from .config import Config

    cfg = Config.from_env()
    with _LOCK:
        _STATE["db_path"] = cfg.db_path

    # Do NOT re-open mid-rebuild. Without this the watcher would drop the handle
    # and the very next search would immediately grab a new one, re-blocking the
    # rmtree the release just unblocked.
    if _reindex_running(cfg.db_path):
        raise ReindexInProgress(
            f"a --clean rebuild is currently rewriting the store at {cfg.db_path}; "
            "search is unavailable until it finishes"
        )

    store = cfg.open_store()
    with _LOCK:
        if _STATE["store"] is None:
            _STATE["store"] = store
            _STATE["root"] = cfg.corpus_root
            return store, cfg.corpus_root
        existing, root = _STATE["store"], _STATE["root"]
    # Lost the open race with another thread -- close OUR duplicate, or it would
    # keep a handle nothing references and re-block the next rebuild.
    try:
        store.close()
    except Exception:  # noqa: BLE001
        pass
    return existing, root


def _watch_for_reindex(stop=None) -> None:
    """Release the store whenever a reingest takes the lock; poll forever.

    A request-driven check is NOT sufficient, and that is the whole reason this
    thread exists: the weekly rebuild fires at 03:30 Sunday, when the server is
    typically idle, so a server that only re-checked on each tool call would
    hold its handle straight through the rebuild and block it -- the exact
    2026-08-16 failure.
    """
    stop = stop if stop is not None else threading.Event()
    while not stop.wait(_WATCH_INTERVAL_S):
        try:
            db_path = _STATE.get("db_path")
            if db_path is None:
                from .config import Config

                try:
                    db_path = Config.from_env().db_path
                except Exception:  # noqa: BLE001 - config absent; retry next tick
                    continue
                with _LOCK:
                    _STATE["db_path"] = db_path
            if _reindex_running(db_path):
                _release_store()
        except Exception:  # noqa: BLE001 - a watcher that dies is worse than one that retries
            continue


def _run_search(arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        store, root = _ensure_store()
    except ReindexInProgress as exc:
        # Transient and self-healing -- deliberately NOT reported as a config
        # error, which would send the caller debugging their environment during
        # a scheduled rebuild.
        return {
            "ok": False,
            "error": {"type": "reindex_in_progress", "message": str(exc)},
            "results": [],
        }
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
        doc_class=arguments.get("doc_class"),
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
    # Daemon so it never keeps the interpreter alive at shutdown. Started AFTER
    # warm so db_path is already resolved on the first tick in the common case.
    threading.Thread(
        target=_watch_for_reindex, name="rag-mcp-reindex-watch", daemon=True
    ).start()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(_main())
