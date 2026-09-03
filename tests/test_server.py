"""Phase 2 - MCP server wiring: tool schema + call_tool returns structured JSON."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from mcp import types

from rag_mcp import server as srv
from rag_mcp.ingest import ingest
from rag_mcp.search import search_knowledge
from rag_mcp.store import HashEmbedder, VectorStore

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MCP_YAML_PATH = _REPO_ROOT / "mcp.yaml"


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


# ---------------------------------------------------------------------------
# doc_class filter wiring (RM-ragmcp-docclass slice 3)
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_corpus(tmp_path):
    (tmp_path / "note.md").write_text(
        "# Note\n\nA loyal dog barks at the mail carrier every single day.\n",
        encoding="utf-8",
    )
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff.md").write_text(
        "---\ntype: handoff\n---\n"
        "# Handoff\n\nA loyal dog barks at the mail carrier every single day.\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def mixed_wired(mixed_corpus, tmp_path):
    db = tmp_path / "srv_mixed.chroma"
    store = VectorStore(path=str(db), collection_name="knowledge", embedder=HashEmbedder())
    ingest(mixed_corpus, store)
    srv._STATE["store"] = store
    srv._STATE["root"] = mixed_corpus
    yield store, mixed_corpus
    srv._STATE["store"] = None
    srv._STATE["root"] = None


def test_tool_schema_exposes_doc_class_as_optional():
    tool = srv._TOOL
    props = tool.input_schema["properties"]
    assert "doc_class" in props
    assert "doc_class" not in tool.input_schema["required"]


def test_call_tool_doc_class_filter_excludes_handoff(mixed_wired):
    out = _call("search_knowledge", {"query": "loyal dog barks", "k": 5, "doc_class": "note"})
    payload = json.loads(out.content[0].text)
    assert payload["ok"] is True
    assert payload["results"]
    assert all(r["citation"]["source"] == "note.md" for r in payload["results"])


def test_call_tool_doc_class_parity_with_direct_search_knowledge(mixed_wired):
    # PARITY: the server tool path and the direct search_knowledge() path must
    # return identical results for the same query + filter -- the tool is a
    # thin pass-through, never a second implementation of the filter.
    store, root = mixed_wired
    direct = search_knowledge(
        "loyal dog barks", k=5, store=store, corpus_root=root, doc_class="note"
    )
    out = _call("search_knowledge", {"query": "loyal dog barks", "k": 5, "doc_class": "note"})
    via_tool = json.loads(out.content[0].text)
    assert via_tool == direct


# ---------------------------------------------------------------------------
# mcp.yaml / server.py tool-schema parity (RM-fixafter-ragmcp slice 8) --
# mcp.yaml's `tools[].args` is a second, hand-maintained description of the
# same tool schema server.py's `_TOOL.input_schema` defines. Nothing kept
# them in sync (mcp.yaml had no `doc_class` arg at all until this slice) --
# this test FIRES the next time either drifts from the other.
# ---------------------------------------------------------------------------


def test_mcp_yaml_tool_args_match_server_schema_properties():
    raw = yaml.safe_load(_MCP_YAML_PATH.read_text(encoding="utf-8"))
    tool_entry = next(t for t in raw["tools"] if t["name"] == "search_knowledge")
    yaml_arg_names = {arg["name"] for arg in tool_entry["args"]}

    schema_props = set(srv._TOOL.input_schema["properties"])

    assert yaml_arg_names == schema_props, (
        f"mcp.yaml args {sorted(yaml_arg_names)} != server.py tool schema "
        f"properties {sorted(schema_props)} -- keep both in sync by hand "
        "whenever either changes."
    )
