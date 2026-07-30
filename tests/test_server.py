"""Phase 2 - MCP server wiring: tool schema + call_tool returns structured JSON."""
from __future__ import annotations

import asyncio
import json

import pytest
from mcp import types

from rag_mcp import server as srv
from rag_mcp.ingest import ingest
from rag_mcp.store import HashEmbedder, VectorStore


def _call(name: str, arguments: dict | None = None):
    """Drive the mcp 2.0.0 call_tool handler: (ctx, CallToolRequestParams) -> CallToolResult."""
    params = types.CallToolRequestParams(name=name, arguments=arguments or {})
    return asyncio.run(srv.call_tool(None, params))


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
    result = asyncio.run(srv.list_tools())
    tools = result.tools
    names = {t.name for t in tools}
    assert "search_knowledge" in names
    tool = next(t for t in tools if t.name == "search_knowledge")
    props = tool.input_schema["properties"]
    assert "query" in props and "k" in props
    assert tool.input_schema["required"] == ["query"]


def test_tool_serializes_input_schema_under_the_wire_name():
    """snake_case in Python, camelCase on the wire -- a client still sees inputSchema."""
    wire = srv._TOOL.model_dump(by_alias=True, exclude_none=True)
    assert "inputSchema" in wire
    assert wire["inputSchema"]["required"] == ["query"]


def test_call_tool_returns_results(wired):
    out = _call("search_knowledge", {"query": "dog barks", "k": 2})
    payload = json.loads(out.content[0].text)
    assert payload["ok"] is True
    assert payload["results"]
    assert payload["results"][0]["citation"]["source"] == "dogs.md"


def test_call_tool_unknown_tool_is_structured():
    out = _call("nope", {})
    payload = json.loads(out.content[0].text)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "unknown_tool"
    # Fail-soft: a structured payload, never a protocol-level error flag.
    assert out.is_error is False
