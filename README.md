---
title: rag-mcp
type: project-readme
tags: [rag, retrieval, embeddings, mcp]
---

# rag-mcp

[![CI](https://github.com/jaimenbell/rag-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/jaimenbell/rag-mcp/actions/workflows/ci.yml)

> A minimal, honest **RAG-over-a-corpus MCP retrieval tool**. One tool,
> `search_knowledge(query, k)`, that embeds a query, vector-searches a local corpus, and
> returns passages **with citations** (source + heading + chunk index) so answers are traceable.

Built to slot into the [mcp-factory](https://github.com/jaimenbell/mcp-factory) manifest model.
Fully local + **$0** (no paid embedding API).

## Why it's safe to put in front of a real corpus
- **Cited** - every hit carries `source` + `heading` + `chunk_index`.
- **Auth-scoped** - results are confined to the configured corpus root; sources that escape it
  (absolute paths, `..` traversal) are refused.
- **Fail-soft** - a down or empty store returns a *structured error*, never an exception that
  crashes the calling agent.
- **Bounded** - `k` is clamped to `[1, 20]`; empty queries are rejected.
- **Version-pinned** deps (`requirements.txt`).

## Stack
| Layer | Choice |
|---|---|
| Embeddings | local ONNX `all-MiniLM-L6-v2` (384-dim, CPU, $0) -- **default**. `bge-large-en-v1.5` (1024-dim, 512-token context) available opt-in via `RAG_MCP_EMBEDDER=bge`; see [CUTOVER.md](./CUTOVER.md). |
| Vector store | ChromaDB embedded `PersistentClient` (zero-infra) |
| Server | `mcp` Python SDK, stdio transport |

## Quick start
```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt

# Ingest a corpus (markdown)
python -m rag_mcp.cli ingest path/to/docs --db ./store.chroma

# One-off query (corpus root = the auth scope)
python -m rag_mcp.cli query "your question" --db ./store.chroma --corpus path/to/docs -k 5

# Run as an MCP server (stdio); configure via env first
#   RAG_MCP_CORPUS_ROOT, RAG_MCP_DB_PATH, RAG_MCP_COLLECTION, RAG_MCP_EMBEDDER
python run_server.py
```

## As an MCP server
Register via `mcp.yaml` (validated against mcp-factory's `Manifest` loader). The tool is
`search_knowledge(query, k)`; it reads the store configured by the `RAG_MCP_*` env vars.

## Tests
```bash
python -m pytest        # 60 passed
```

## Layout
```
rag_mcp/
  chunking.py   heading-scoped, overlapping markdown chunks
  store.py      VectorStore (Chroma) + Embedder protocol (MiniLM default + BgeEmbedder opt-in + offline HashEmbedder)
  ingest.py     idempotent ingest pipeline with source/heading/chunk-index metadata
  search.py     search_knowledge: cited, auth-scoped, fail-soft, bounded
  server.py     MCP stdio server exposing search_knowledge
  config.py     env-driven Config
  cli.py        ingest + query CLI
run_server.py   MCP entrypoint (referenced by mcp.yaml)
mcp.yaml        manifest (mcp-factory model)
```

<!-- MCP registry ownership marker -->
mcp-name: io.github.jaimenbell/rag-mcp
