# CUTOVER: MiniLM -> bge-large-en-v1.5

Status: **NOT cut over.** MiniLM (384-dim) is still the default embedder and the
live store (`store.chroma`, `knowledge` collection) is untouched. The bge path
(`RAG_MCP_EMBEDDER=bge`, 1024-dim, 512-token context) ships alongside it on
branch `feat/bge-embedder`; a full bge index of your corpus lives in a SEPARATE
store dir so rollback is instant.

Research grounding: `<your-corpus>/research/<the bge-vs-MiniLM design note>`

## Why two stores

MiniLM vectors are 384-dim; bge vectors are 1024-dim. One Chroma collection
cannot hold both — `VectorStore` now records `embed_dim` in collection metadata
and raises `DimensionMismatchError` if you open a store with the wrong embedder.
So the cutover is a REPOINT, not an in-place migration:

- live MiniLM index: `C:\Users\jaime\projects\rag-mcp\store.chroma`
- new bge index:     built as `store-bge.chroma` (see report for final location)

## Operator cutover steps (in order)

1. **Merge** `feat/bge-embedder` into `master` in `C:\Users\jaime\projects\rag-mcp`
   (branch was developed in the `rag-mcp-bge` worktree; live checkout untouched).
2. **Install the new dep** into the LIVE venv:
   `C:\Users\jaime\projects\rag-mcp\.venv\Scripts\python -m pip install fastembed==0.8.0`
3. **Move/copy the built bge store** into the live repo, e.g.
   `C:\Users\jaime\projects\rag-mcp\store-bge.chroma` (or rebuild in place with
   `python -m rag_mcp.cli --embedder bge ingest "<your-corpus-path>" --db .\store-bge.chroma`
   — ~2.5h CPU). Note: the bge model weights cache in `%TEMP%\fastembed_cache`
   (~1.3GB); a `%TEMP%` purge forces a one-time re-download.
4. **Flip the batch jobs** — in `reingest.bat` AND `reingest-clean.bat`:
   - `set "STORE=C:\Users\jaime\projects\rag-mcp\store-bge.chroma"`
   - `set "RAG_MCP_EMBEDDER=bge"`
   - pass `--embedder %RAG_MCP_EMBEDDER%` on the `rag_mcp.cli` line (the .bat
     currently sets the env var but the CLI takes the flag; the flag is what counts).
   The schtasks (`rag-mcp-reingest` daily 03:00, `rag-mcp-reingest-clean` Sun
   03:30) call these .bats by path, so NO schtask re-registration is needed.
   CAUTION: daily bge upsert re-embeds every chunk at bge speed (~2.5h vs ~11min
   MiniLM). 03:00 start still finishes well before 06:00 weekday start, but
   verify the first run's log (`logs\reingest.log`).
5. **Flip the MCP server env** — wherever `run_server.py` gets its env
   (`~/.claude.json` entry / mcp.yaml `env:` block):
   - `RAG_MCP_EMBEDDER=bge`
   - `RAG_MCP_DB_PATH=C:\Users\jaime\projects\rag-mcp\store-bge.chroma`
   Restart the MCP server (restart Claude Code session) to pick it up.
6. **Smoke test**:
   `python -m rag_mcp.cli --embedder bge query "dead-but-GREEN liveness gate" --db .\store-bge.chroma --corpus "<your-corpus-path>" -k 3`
   Expect top hit `<a known doc in your corpus>` with score > MiniLM's 0.57.
7. **After ~1 week green**: delete the old `store.chroma` (or keep as archive).

## Rollback (instant)

Revert steps 4-5: point `STORE`/`RAG_MCP_DB_PATH` back to `store.chroma` and
`RAG_MCP_EMBEDDER` back to `default`. The MiniLM store keeps receiving daily
upserts until you flip the .bats, so it stays fresh during any bake period —
if you want a bake period with BOTH stores fresh, duplicate the .bat lines to
run both ingests (MiniLM first, bge second; the ReingestLock note below).

## Locking note

The reingest mutex (`3e0d682`, branch `fix/reingest-mutex`) locks per-store
(`<store>.lock` next to the store dir). If both a MiniLM and a bge ingest run
in the same window they use DIFFERENT stores, so they won't corrupt each other,
but they will compete for CPU. Stagger them (e.g. 03:00 / 03:30 offsets).
NOTE: that mutex branch is NOT yet merged to master — merge order with this
branch is free, but if both land, re-run the combined test suite.
