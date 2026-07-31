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

REM REPO self-locates from THIS SCRIPT's own directory (%~dp0), which satisfies
REM two constraints at once and is why neither earlier form was right:
REM   - It is GENERIC. This is a public repo; a hardcoded personal path would
REM     undo the 2026-07-22 public-scrub (see the local-corpus.txt note below).
REM   - It is SCHEDULER-SAFE. %USERPROFILE% (introduced e268652, 2026-07-28) is
REM     not guaranteed to resolve under a no-profile/S4U task context to what an
REM     interactive shell sees; the script's own location always does.
set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"
set "LOG=%REPO%\logs\reingest.log"

REM The logs dir must exist BEFORE the guard below, or the guard's own FATAL
REM echo is written into a nonexistent directory and is silently lost -- which
REM is exactly why the 2026-07-26 exit-1 could not be diagnosed from the log.
if not exist "%REPO%\logs" mkdir "%REPO%\logs"

REM Unconditional path diagnostics, so a future guard trip is explainable from
REM the log alone rather than being invisible.
echo [%DATE% %TIME%] rag-mcp CLEAN rebuild INVOKED (REPO=%REPO%, USERPROFILE=%USERPROFILE%, CD=%CD%)>> "%LOG%"

REM Corpus path from the UNTRACKED local file (see reingest.bat header).
if not exist "%REPO%\local-corpus.txt" (
  echo [%DATE% %TIME%] FATAL: local-corpus.txt missing -- see reingest.bat header>> "%LOG%"
  exit /b 1
)
set /p VAULT=<"%REPO%\local-corpus.txt"
REM CUTOVER 2026-07-02: bge store (see reingest.bat header). First clean rebuild
REM (Sun 03:30) also purges the Wikilink-Scan noise chunks baked into the
REM initial build before the ingest exclude landed.
set "STORE=%REPO%\store-bge.chroma"
set "PY=%REPO%\.venv\Scripts\python.exe"

set "RAG_MCP_COLLECTION=knowledge"
set "RAG_MCP_EMBEDDER=bge"

echo ==================================================================>> "%LOG%"
echo [%DATE% %TIME%] rag-mcp CLEAN rebuild START (pruning orphans)>> "%LOG%"
REM The store delete is done INSIDE the Python entrypoint (--clean), strictly
REM AFTER the cross-process reingest lock is acquired, so it can never race a
REM concurrent daily run mid-write. Do NOT rmdir the store here (outside the lock).
cd /d "%REPO%"
"%PY%" -m rag_mcp.cli --embedder %RAG_MCP_EMBEDDER% ingest "%VAULT%" --db "%STORE%" --clean >> "%LOG%" 2>&1
REM Capture immediately, on its own line. Do NOT read %ERRORLEVEL% inside a
REM parenthesized block -- it expands at parse time there and freezes.
set "RC=%ERRORLEVEL%"
echo [%DATE% %TIME%] rag-mcp CLEAN rebuild DONE (exit %RC%)>> "%LOG%"

REM Exit code 3 is the cross-process lock being held by the 15-minute
REM incremental task (rag_mcp/cli.py:38). Per this file's header that is a
REM correct, by-design no-op, NOT a failure -- so it must not surface to the
REM scheduler as a red run. Every other nonzero code propagates honestly.
REM Before 2026-07-31 the tail was a bare echo with no exit /b, which made this
REM script report exit 0 even when the ingest genuinely failed (proven on this
REM box by isolated repro).
if "%RC%"=="3" (
  echo [%DATE% %TIME%] rag-mcp CLEAN rebuild SKIPPED -- reingest lock held by the incremental task; reporting success, will retry next Sunday>> "%LOG%"
  exit /b 0
)
exit /b %RC%
