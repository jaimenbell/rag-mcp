---
title: rag-mcp - Portfolio Entry
type: portfolio
date: 2026-06-29
tags: [rag, retrieval, embeddings, mcp, portfolio]
---

# rag-mcp - RAG retrieval as a hardened MCP tool

## Blurb (verbatim - 3-4 lines)

> **rag-mcp** is a retrieval-augmented-generation MCP tool: one tool, `search_knowledge(query, k)`,
> that embeds a query, vector-searches a local corpus, and returns the passages **with citations**
> (source file + heading + chunk index) so every answer is traceable. It is **auth-scoped** (results
> can't escape the configured corpus root), **fail-soft** (a down or empty store returns a structured
> error instead of crashing the agent), and **version-pinned**. Stack is fully local and $0 - ONNX
> `all-MiniLM-L6-v2` embeddings + an embedded ChromaDB store - and the server is expressible in the
> mcp-factory manifest model. 51 tests, all green.

## Stack
- **Embeddings:** local ONNX `all-MiniLM-L6-v2` (384-dim, CPU, $0, no API key).
- **Vector store:** ChromaDB embedded `PersistentClient` (zero-infra, local file store).
- **Server:** `mcp` Python SDK over stdio; one tool, `search_knowledge`.
- **Manifest:** `mcp.yaml` validated against the mcp-factory `Manifest` loader (referenced, not scaffolded).

## Reliability properties (each test-proven)
| Property | What it means | Proof |
|---|---|---|
| **Cited** | Every hit carries source + heading + chunk index | `test_search.py::test_citation_fields_present` |
| **Auth-scoped** | Results confined to the corpus root; `../` / absolute sources refused | `test_out_of_corpus_source_refused` |
| **Fail-soft** | Down/empty store -> structured error, never an exception | `test_store_unreachable_*`, `test_empty_store_*` |
| **Bounded** | `k` clamped to [1, 20]; empty query rejected | `test_k_bounds_respected`, `test_invalid_query_rejected` |

## Verification (reproducible)
- `python -m pytest` -> **51 passed** (chunking 6, cli 3, ingest 11, lock 7, search 7, server 3, server_stdio 1, store 13).
- Real ingest of a 12-file corpus -> **129 chunks** with the live ONNX embedder.
- Live retrieval (real embedder) returns correct citations, e.g. a "single remaining gate to revenue"
  query ranks the right note first; an out-of-domain query stays in-scope (`dropped_out_of_scope` reported).
- MCP stdio handshake verified end-to-end: `initialize` + `tools/list` over a real subprocess
  advertise `search_knowledge`; `call_tool` execution + JSON citations covered by the server tests.

## Privacy note
The local dogfood corpus is the personal vault and is **never** exposed externally. Any client-facing
demo uses a neutral/public corpus, not your corpus.

## Demo shotlist (~45-60s, for a future Loom - NOT recorded here)
1. **(0-8s) The pitch.** Show `mcp.yaml` + the tool description: "retrieval with citations, auth-scoped, fail-soft."
2. **(8-20s) Ingest.** `python -m rag_mcp.cli ingest <docs> --db <db>` -> show the JSON report (files/chunks).
3. **(20-35s) Retrieve with citations.** `... cli query "a real question" --corpus <root>` -> highlight each
   result's `citation` (source + heading) - "traceable answers, not a black box."
4. **(35-48s) The reliability story.** Point store at a missing path -> show the structured `store_unreachable`
   error (agent keeps running); show an `../`-escaping source getting dropped (`dropped_out_of_scope`).
5. **(48-60s) It's a real MCP tool.** Show `pytest` 51 green + the `mcp.yaml` validating in the mcp-factory model.
