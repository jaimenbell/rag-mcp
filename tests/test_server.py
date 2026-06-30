"""Phase 2 - MCP server wiring: tool schema + call_tool returns structured JSON."""
from __future__ import annotations

import asyncio
import json

import pytest

from rag_mcp import server as srv
from rag_mcp.ingest import ingest
from rag_mcp.store import HashEmbedder, VectorStore


@pytest.fixture
def wired(corpus, tmp_path, monkeypatch):
    """Configure the server to a hash-embedder store over the test corpus."""
    db = tmp_path / "srv.chroma"
    store = VectorStore(path=str(db), collection_name="knowledge", embedder=HashEmbedder())
    ingest(corpus, store)
    # Inject server state directly (bypasses env / real ONNX model).
    srv._STATE["store"] = store
    srv._STATE["root"] = corpus
    yield
    srv._STATE["store"] = None
    srv._STATE["root"] = None


def test_list_tools_exposes_search_knowledge():
    tools = asyncio.run(srv.list_tools())
    names = {t.name for t in tools}
    assert "search_knowledge" in names
    tool = next(t for t in tools if t.name == "search_knowledge")
    props = tool.inputSchema["properties"]
    assert "query" in props and "k" in props
    assert tool.inputSchema["required"] == ["query"]


def test_call_tool_returns_results(wired):
    out = asyncio.run(srv.call_tool("search_knowledge", {"query": "dog barks", "k": 2}))
    payload = json.loads(out[0].text)
    assert payload["ok"] is True
    assert payload["results"]
    assert payload["results"][0]["citation"]["source"] == "dogs.md"


def test_call_tool_unknown_tool_is_structured():
    out = asyncio.run(srv.call_tool("nope", {}))
    payload = json.loads(out[0].text)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "unknown_tool"
