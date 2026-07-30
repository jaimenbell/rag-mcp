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
| Server | `mcp` Python SDK 2.x, stdio transport, protocol revision **2026-07-28** |

## Protocol revision
Pinned to `mcp==2.0.0`, the first SDK release implementing MCP protocol revision
**2026-07-28**. The server serves **both eras on the same stdio connection** -- the
client's first frame picks:

| Client opens with | Negotiated revision | Notes |
|---|---|---|
| a per-request `_meta` envelope (or a `server/discover` probe) | `2026-07-28` | stateless per-request envelope; no `initialize` |
| the classic `initialize` handshake | `2025-11-25` | handshake era caps here -- expected, not a downgrade |

`2026-07-28` is **not reachable via the `initialize` handshake**; it is a "modern"
revision reached through `server/discover` or an inline `_meta` version stamp. Era
selection is automatic and per-connection -- there is no server-side flag.

`tests/test_protocol_version.py` asserts both paths end-to-end, so a dependency
rollback that silently drops the server to an older revision fails CI instead of
passing quietly.

## Quick start
```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt

# Ingest a corpus (markdown). Incremental by default: only files whose content
# changed since the last run are re-embedded.
python -m rag_mcp.cli ingest path/to/docs --db ./store.chroma

# Force a rebuild in place (ignore the manifest, re-embed everything)
python -m rag_mcp.cli ingest path/to/docs --db ./store.chroma --full

# One-off query (corpus root = the auth scope)
python -m rag_mcp.cli query "your question" --db ./store.chroma --corpus path/to/docs -k 5

# Run as an MCP server (stdio); configure via env first
#   RAG_MCP_CORPUS_ROOT, RAG_MCP_DB_PATH, RAG_MCP_COLLECTION, RAG_MCP_EMBEDDER
python run_server.py        # operational entrypoint (referenced by mcp.yaml)
python -m rag_mcp           # same server, via the packaged console entry point
rag-mcp                     # after `pip install jaimenbell-rag-mcp` -- console script
```

## Keeping the index fresh (incremental ingest)
Ingest is **incremental by default**. A manifest inside the store dir records a
SHA-256 of each file's decoded text; a run re-embeds only what actually changed,
and prunes what upsert alone never could (chunks of deleted/renamed notes, and
trailing chunks of notes that got shorter).

Measured on a live 2808-file / 26.6 MiB corpus (bge, CPU):

| Run | Cost |
|---|---|
| tick with no changes | **~0.7s** (walk + read + hash everything) |
| full re-embed | ~2h33m (50,109 chunks at ~5.5 chunks/sec) |

That is what makes a frequent schedule affordable: `reingest.bat` is meant to run
**every 15 minutes** instead of once daily at 03:00, which had left a note written
at 03:05 invisible to `search_knowledge` for nearly 24 hours.

The manifest is only trusted when the **run identity** matches -- embedder, embedding
dimension, collection and chunking parameters. Change any of them and every file is
re-embedded, so an embedder swap can never be silently half-applied. A missing,
corrupt, or mismatched manifest, or a manifest against an empty store, all degrade
to a full rebuild; nothing degrades to a wrong skip.

`--full` rebuilds in place (ignores the manifest, keeps the store); `--clean`
deletes the store first. Both still WRITE a manifest, so the next run is cheap.
`reingest-clean.bat` (weekly) remains a belt-and-braces reset.

## As an MCP server
Register via `mcp.yaml` (validated against mcp-factory's `Manifest` loader). The tool is
`search_knowledge(query, k)`; it reads the store configured by the `RAG_MCP_*` env vars.

## Tests
```bash
python -m pytest        # 126 passed
```

## Layout
```
rag_mcp/
  chunking.py   heading-scoped, overlapping markdown chunks
  store.py      VectorStore (Chroma) + Embedder protocol (MiniLM default + BgeEmbedder opt-in + offline HashEmbedder)
  ingest.py     idempotent ingest pipeline with source/heading/chunk-index metadata; incremental by default
  manifest.py   per-file content hashes -> skip unchanged files, prune stale chunks
  search.py     search_knowledge: cited, auth-scoped, fail-soft, bounded
  server.py     MCP stdio server exposing search_knowledge
  config.py     env-driven Config
  cli.py        ingest + query CLI
  __main__.py   console entrypoint (`python -m rag_mcp` / `rag-mcp` script); fails loud on missing config
run_server.py   operational MCP entrypoint (referenced by mcp.yaml)
mcp.yaml        manifest (mcp-factory model)
```

<!-- MCP registry ownership marker -->
mcp-name: io.github.jaimenbell/rag-mcp
