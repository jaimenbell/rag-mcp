---
title: rag-mcp - Stack Decisions
type: project-decisions
date: 2026-06-29
tags: [rag, mcp, retrieval, embeddings, decisions]
status: phase-0-complete
---

# rag-mcp - Stack Decisions (Phase 0)

A minimal, honest **RAG-over-a-corpus MCP retrieval tool**. One MCP tool,
`search_knowledge(query, k)`, that embeds a query, retrieves top-k chunks from a
local vector store, and returns the text **with citations** (source path + heading)
so every answer is traceable. Built to slot into the `mcp-factory` manifest model.

> [!important] Hard rails
> - **$0 / LOCAL only.** No paid embedding API (no OpenAI/Cohere). All inference is local CPU.
> - **Privacy.** The vault is a *local-only* dogfood corpus. Its contents are never sent
>   anywhere external, and any client-facing demo uses a neutral/public corpus.
> - **Honesty.** Only claim what ships. The resume/draft RAG-claim update is done by the
>   command-center session, not here.

## Build location

`C:/Users/jaime/projects/rag-mcp` - a **new, isolated local repo** (`git init`).
Sole-writer-safe; does not touch sales-reps, alphahive, or ai-content-factory.

## Decisions

| Choice | Pick | One-line why |
|---|---|---|
| **Embeddings** | ChromaDB default `all-MiniLM-L6-v2` (ONNX, 384-dim) | Fully local CPU, $0, no API key, no torch; downloads ~80 MB once then runs offline. |
| **Vector store** | **ChromaDB** embedded `PersistentClient` | Zero-infra (Docker is NOT installed on this box), $0, local file store - smallest honest first cut. |
| **Embedder abstraction** | Pluggable `Embedder` protocol | Tests inject a deterministic offline fake (hash-based); real dogfood uses the ONNX model. Keeps the suite fast + offline. |
| **MCP layer** | `mcp` SDK stdio server (`mcp.server.Server`) | Matches the mcp-factory generated-server style (`list_tools`/`call_tool`, stdio). |
| **Corpus (dogfood)** | The Obsidian vault, **local only** | Per the privacy rail - never exposed; client demos would use a neutral corpus. |

### Why not the alternatives
- **Qdrant / pgvector (Docker):** Docker is not installed here; pgvector needs a running
  Postgres. Chroma embedded is the zero-infra MVP. The store layer is wrapped behind one
  module so swapping to Qdrant later is a localized change.
- **vLLM Qwen2.5-14B (already running on :8000):** that is a *chat/completion* model, not an
  embedding model. Pulling a separate local embedding model adds infra; Chroma's bundled
  ONNX MiniLM is the smaller, self-contained $0 path.

## mcp-factory fit

The server is a standalone stdio MCP server (`rag_mcp/server.py`) referenced by an
`mcp.yaml` manifest via `runtime.script` - exactly the "reference an existing server" path
the factory takes. So it doubles as an mcp-factory showcase: **a RAG MCP server expressible
in the factory's manifest model**, carrying the same production properties the factory pitches
(scoped access, fail-soft errors, version-pinned deps, a real test suite).

## Reliability story (the pitch, proven in Phase 2 tests)
- **Auth-scoped:** the tool can only read inside the configured corpus root; out-of-corpus
  path access is refused.
- **Fail-soft:** vector store unreachable / empty -> a structured error object, never an
  exception that kills the agent.
- **Cited:** every hit carries source path + heading + chunk id.
- **Version-pinned:** dependencies pinned in `requirements.txt`.

## Exit (Phase 0)
- [x] Stack chosen + recorded.
- [x] Local embeddings confirmed reachable (MiniLM ONNX -> 384-dim vectors).
- [x] Build dir initialized (`git init`), isolated venv created, deps installed.
