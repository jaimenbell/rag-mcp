@echo off
REM ============================================================================
REM rag-mcp WEEKLY CLEAN rebuild routine
REM Deletes the Chroma store then re-ingests from scratch. Unlike the daily
REM upsert (reingest.bat), a clean rebuild PRUNES chunks for vault notes that
REM were deleted/renamed since the last full build (upsert never removes those).
REM Registered as scheduled task "rag-mcp-reingest-clean" (weekly Sun 03:30,
REM offset 30 min after the daily 03:00 task so they never collide).
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
