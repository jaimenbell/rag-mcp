"""End-to-end REAL-stdio transport test for the search_knowledge tool.

The rest of the suite exercises the handler in-process (calling srv.call_tool /
the SDK wrapper directly), which never runs the actual stdio dispatch path. That
gap hid a hard deadlock: the FIRST call_tool lazily triggered chromadb/numpy's
first C-extension import ON the event-loop thread while an anyio worker was
blocked in a native stdin.readline(), and numpy's import deadlocked (Windows),
so the tool never returned a response through real transport.

This test spawns the server as a subprocess and drives it exactly like a real
MCP client -- initialize, then call_tool -- over stdin/stdout, asserting a
well-formed result comes back within a bounded time. It is fully hermetic:
a tiny tmp hash-embedder store, RAG_MCP_EMBEDDER=hash (no model download, no
network), and it never touches the production vault store.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rag_mcp.ingest import ingest
from rag_mcp.store import HashEmbedder, VectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
# Bound every transport step so a regression of the deadlock fails fast (the bug
# manifested as an unbounded hang) instead of stalling the whole suite.
_STEP_TIMEOUT = 30.0


def _build_store(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "dogs.md").write_text(
        "# Dogs\n\n## Behavior\n\nDogs are loyal pack animals. "
        "A dog barks to alert its owner.\n",
        encoding="utf-8",
    )
    (corpus / "cats.md").write_text(
        "# Cats\n\nCats are small carnivorous mammals. A cat purrs when content.\n",
        encoding="utf-8",
    )
    db = tmp_path / "store.chroma"
    ingest(
        corpus,
        VectorStore(path=str(db), collection_name="knowledge", embedder=HashEmbedder()),
    )
    return corpus, db


async def _drive(corpus: Path, db: Path) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "RAG_MCP_CORPUS_ROOT": str(corpus),
            "RAG_MCP_DB_PATH": str(db),
            "RAG_MCP_COLLECTION": "knowledge",
            "RAG_MCP_EMBEDDER": "hash",  # offline, deterministic, no model download
        }
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rag_mcp.server"],
        env=env,
        cwd=str(REPO_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=_STEP_TIMEOUT)
            tools = await asyncio.wait_for(session.list_tools(), timeout=_STEP_TIMEOUT)
            assert "search_knowledge" in {t.name for t in tools.tools}
            result = await asyncio.wait_for(
                session.call_tool("search_knowledge", {"query": "dog barks", "k": 2}),
                timeout=_STEP_TIMEOUT,
            )
    assert result.content, "call_tool returned no content"
    return json.loads(result.content[0].text)


def test_search_knowledge_over_real_stdio(tmp_path):
    """A real MCP client can initialize and call search_knowledge over stdio.

    Regression guard for the first-call numpy-import deadlock: before the fix
    this call_tool never returned and the client hit anyio.BrokenResourceError /
    a hang; the wait_for timeouts here would fire.
    """
    corpus, db = _build_store(tmp_path)
    payload = asyncio.run(_drive(corpus, db))
    assert payload["ok"] is True
    assert payload["results"], "expected at least one hit"
    sources = {r["citation"]["source"] for r in payload["results"]}
    assert "dogs.md" in sources
    # Every result carries a citation (the tool's core contract).
    for r in payload["results"]:
        assert r["citation"]["source"]
