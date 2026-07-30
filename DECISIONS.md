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

`C:/Users/<you>/projects/rag-mcp` - a **new, isolated local repo** (`git init`).
Sole-writer-safe; does not touch sales-reps, alphahive, or ai-content-factory.

## Decisions

| Choice | Pick | One-line why |
|---|---|---|
| **Embeddings** | ChromaDB default `all-MiniLM-L6-v2` (ONNX, 384-dim) | Fully local CPU, $0, no API key, no torch; downloads ~80 MB once then runs offline. |
| **Vector store** | **ChromaDB** embedded `PersistentClient` | Zero-infra (Docker is NOT installed on this box), $0, local file store - smallest honest first cut. |
| **Embedder abstraction** | Pluggable `Embedder` protocol | Tests inject a deterministic offline fake (hash-based); real dogfood uses the ONNX model. Keeps the suite fast + offline. |
| **MCP layer** | `mcp` SDK stdio server (`mcp.server.Server`) | Matches the mcp-factory generated-server style (`list_tools`/`call_tool`, stdio). Migrated to SDK 2.x / protocol revision 2026-07-28 -- see below. |
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

## MCP SDK 2.x / protocol revision 2026-07-28 (2026-07-30)

Bumped `mcp==1.28.1` -> `mcp==2.0.0`. Rationale: 1.28.1's newest protocol revision is
`2025-11-25`; **2.0.0 is the first release implementing `2026-07-28`** (released the same
day, 2026-07-28).

### What the revision actually changes for this server
`2026-07-28` is a **"modern" revision**: a stateless **per-request `_meta` envelope**
instead of a stateful `initialize` handshake. It is *not* reachable via `initialize` --
`HANDSHAKE_PROTOCOL_VERSIONS` still tops out at `2025-11-25`. Servers reach it through
`server/discover` (which SDK 2.x auto-registers on every `Server`) or an inline `_meta`
version stamp.

The era is chosen **per connection, by the client's first frame**
(`mcp/server/runner.py::serve_dual_era_loop`), and works over **stdio** -- it is not
HTTP-only. There is no server-side flag and no new `run()` parameter; a legacy client
still gets a correct `2025-11-25` handshake connection on the same server object.

### Breaking changes we actually had to absorb
| Change | Before (1.28.1) | After (2.0.0) |
|---|---|---|
| Handler registration | `@server.list_tools()` / `@server.call_tool()` decorators | constructor kwargs `Server(..., on_list_tools=, on_call_tool=)` (or `add_request_handler`) |
| Handler signature | `call_tool(name, arguments)` | `call_tool(ctx, params: CallToolRequestParams)` |
| Handler return type | bare `list[Tool]` / `list[TextContent]` | `ListToolsResult` / `CallToolResult` models |
| Model field names | camelCase attrs (`tool.inputSchema`, `result.isError`) | snake_case attrs (`tool.input_schema`, `result.is_error`); camelCase still accepted on *construction* and still used on the wire |
| Version constants | `mcp.shared.version` | separate `mcp_types.version` package |

Unchanged: `mcp.server.Server`, `mcp.server.stdio.stdio_server`, `server.run(...)`,
`create_initialization_options()`, and the `types.Tool` / `types.TextContent` names. The
`_warm()` numpy-deadlock guard and the fail-soft contract are untouched -- an unknown tool
is still a structured `ok:false` payload with `is_error=False`, not a protocol error.

### Guard
`tests/test_protocol_version.py` asserts the SDK's `LATEST_PROTOCOL_VERSION`, that the
pyproject pin is exact and >=2, that the installed version matches the pin, that
`server/discover` advertises `2026-07-28`, and -- end to end through a real client -- that
modern and `auto` modes negotiate `2026-07-28` while a legacy client still gets
`2025-11-25`. A silent dependency rollback fails CI instead of quietly downgrading the
protocol.

> [!note] `MCPServer` not adopted
> SDK 2.x also ships `mcp.server.MCPServer`, an ergonomic `@app.tool()` facade. It is a
> thin wrapper over the same lowlevel `Server` and negotiates identically, so it buys no
> protocol capability. Staying on the lowlevel `Server` keeps the diff minimal and the
> mcp-factory manifest fit unchanged.

## Incremental ingest / closing the reingest window (2026-07-30)

### Problem
`reingest.bat` ran once daily at 03:00 and **re-embedded the entire corpus every
run** -- there was no mtime check, no hash check and no manifest. Chunk ids are
deterministic (`<rel>::<idx>`) so upsert kept the result correct, but the whole
cost of a run was redone from scratch: measured 2026-07-30, 50,109 chunks in
2h32m53s (~5.5 chunks/sec on the bge CPU path).

That cost is what forced the once-daily cadence, and the cadence is what created
the window: a note written at 03:05 was invisible to `search_knowledge` for
**nearly 24 hours**.

### Fix
A manifest (`ingest-manifest.json`, inside the store dir) records a SHA-256 of
each file's decoded text plus its chunk count. Ingest is now incremental by
default and skips the chunk+embed+upsert for any file whose hash is unchanged.

Measured on the live 2808-file / 26.6 MiB corpus: a tick with **no** changes costs
**0.68s** (walk + read + hash of every file). Only genuinely edited files are
re-embedded, at ~5.5 chunks/sec.

| | before | after |
|---|---|---|
| cost of a run with no changes | ~2h33m | **~0.7s** |
| cost of a day's churn (~19 files, ~533 chunks) | ~2h33m | **~1.6 min** |
| affordable cadence | daily | **every 15 min** |
| worst-case invisibility window | ~24h | **~15 min** |

Hashing the DECODED TEXT, not mtime: mtime moves when content does not (touch,
sync, restore) and can fail to move when content does (same-second writes,
archive extraction). Reading every file costs 0.23s total, so there is no reason
to accept mtime's weaker guarantee.

### Pruning (a latent bug fixed on the way)
Upsert only ever overwrote ids `0..n-1`; it never removed anything. So a note that
got **shorter** left orphaned trailing chunks in the index, and a **deleted or
renamed** note left all of its chunks. Only the weekly `--clean` rebuild cleared
them. The manifest knows the previous chunk count per file, so incremental ingest
now prunes both cases on every run.

### Trust rules
The manifest is honored only when the **run identity** matches: embedder class,
embedding dimension, collection name, and chunking parameters. This is the
2026-07-02 MiniLM -> bge cutover class of bug -- a manifest written by a different
embedder must never authorize a skip. Missing, corrupt, wrong-version or
mismatched-identity manifests, and a populated manifest against an empty store,
ALL degrade to a full re-embed. Every failure mode degrades to doing too much
work, never to a wrong skip, because a wrong skip leaves a note permanently stale
in the index with no symptom.

The manifest lives inside the store directory so `--clean`'s rmtree drops both
together and they cannot outlive each other. It is written atomically
(tmp + `os.replace`) -- a torn manifest could claim a file is embedded when it is
not. `--full` and `--clean` do not READ the manifest but still WRITE one, so a
rebuild does not force the next scheduled run to be full as well.

### Also fixed
`tests/conftest.py`'s production-store guard watched `store.chroma`, which the
2026-07-02 bge cutover had demoted to a stale 184 KB rollback copy. The live index
(`store-bge.chroma`, ~799 MB) was guarded by nothing -- the guard was pointed at
the wrong artifact and could not have detected pollution of the store it claimed
to protect. It now fingerprints both.
