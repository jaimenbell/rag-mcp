@echo off
REM ============================================================================
REM rag-mcp WEEKLY CLEAN rebuild routine
REM Deletes the Chroma store then re-ingests from scratch.
REM Registered as scheduled task "rag-mcp-reingest-clean" (weekly Sun 03:30).
REM
REM 2026-07-30: this is now a BELT-AND-BRACES rebuild, not the only way to prune.
REM reingest.bat became incremental and prunes deleted/renamed notes and trailing
REM chunks of shortened notes on every tick, so orphans no longer accumulate for
REM a week. This full rebuild still earns its place as a periodic reset: it
REM recovers from a corrupt store, an embedder change, or any manifest/store
REM desync, none of which an incremental run is guaranteed to notice.
REM
REM Lock interaction with the 15-minute incremental task: both take the SAME
REM cross-process lock, so they can never corrupt each other. Whichever starts
REM first wins; the other exits 3 as a no-op. A ~1s tick therefore has a small
REM (~0.1%) chance of making this weekly rebuild skip a week -- acceptable now
REM that pruning no longer depends on it. It also means the ~2.5h rebuild
REM silently no-ops the incremental ticks for its duration, which is correct.
REM ============================================================================

set "REPO=%USERPROFILE%\projects\rag-mcp"
REM Corpus path from the UNTRACKED local file (see reingest.bat header).
if not exist "%REPO%\local-corpus.txt" (
  echo [%DATE% %TIME%] FATAL: local-corpus.txt missing -- see reingest.bat header>> "%REPO%\logs\reingest.log"
  exit /b 1
)
set /p VAULT=<"%REPO%\local-corpus.txt"
REM CUTOVER 2026-07-02: bge store (see reingest.bat header). First clean rebuild
REM (Sun 03:30) also purges the Wikilink-Scan noise chunks baked into the
REM initial build before the ingest exclude landed.
set "STORE=%REPO%\store-bge.chroma"
set "PY=%REPO%\.venv\Scripts\python.exe"
set "LOG=%REPO%\logs\reingest.log"

set "RAG_MCP_COLLECTION=knowledge"
set "RAG_MCP_EMBEDDER=bge"

if not exist "%REPO%\logs" mkdir "%REPO%\logs"

echo ==================================================================>> "%LOG%"
echo [%DATE% %TIME%] rag-mcp CLEAN rebuild START (pruning orphans)>> "%LOG%"
REM The store delete is done INSIDE the Python entrypoint (--clean), strictly
REM AFTER the cross-process reingest lock is acquired, so it can never race a
REM concurrent daily run mid-write. Do NOT rmdir the store here (outside the lock).
cd /d "%REPO%"
"%PY%" -m rag_mcp.cli --embedder %RAG_MCP_EMBEDDER% ingest "%VAULT%" --db "%STORE%" --clean >> "%LOG%" 2>&1
echo [%DATE% %TIME%] rag-mcp CLEAN rebuild DONE (exit %ERRORLEVEL%)>> "%LOG%"
