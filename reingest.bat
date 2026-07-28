@echo off
REM ============================================================================
REM rag-mcp daily re-ingest routine
REM Re-embeds your corpus into the Chroma store so search_knowledge stays current.
REM Idempotent: rag_mcp ingest UPSERTS by id, so re-running does not duplicate.
REM Registered as scheduled task "rag-mcp-reingest" (daily 03:00, catch-up on).
REM Caveat: upsert does NOT prune chunks for files DELETED from your corpus.
REM         Run a clean rebuild (delete store.chroma then ingest) occasionally
REM         to drop orphaned chunks -- see reingest-clean.bat note below.
REM ============================================================================

set "REPO=%USERPROFILE%\projects\rag-mcp"
REM Corpus path comes from an UNTRACKED local file so the public repo stays
REM generic (2026-07-22: the public-scrub sanitized a hardcoded path here and
REM the nightly ingest silently ran against a placeholder). Create it once:
REM   echo C:\path\to\your\corpus> local-corpus.txt
if not exist "%REPO%\local-corpus.txt" (
  echo [%DATE% %TIME%] FATAL: local-corpus.txt missing -- see reingest.bat header>> "%REPO%\logs\reingest.log"
  exit /b 1
)
set /p VAULT=<"%REPO%\local-corpus.txt"
REM CUTOVER 2026-07-02: bge-large-en-v1.5 store (1024-dim). Old MiniLM store
REM kept at store.chroma for instant rollback (repoint STORE + EMBEDDER back).
REM bge upsert takes ~2.5h CPU (vs ~11min MiniLM) -- 03:00 start finishes ~05:30.
set "STORE=%REPO%\store-bge.chroma"
set "PY=%REPO%\.venv\Scripts\python.exe"
set "LOG=%REPO%\logs\reingest.log"

set "RAG_MCP_COLLECTION=knowledge"
set "RAG_MCP_EMBEDDER=bge"

if not exist "%REPO%\logs" mkdir "%REPO%\logs"

echo ==================================================================>> "%LOG%"
echo [%DATE% %TIME%] rag-mcp re-ingest START>> "%LOG%"
cd /d "%REPO%"
"%PY%" -m rag_mcp.cli --embedder %RAG_MCP_EMBEDDER% ingest "%VAULT%" --db "%STORE%" >> "%LOG%" 2>&1
echo [%DATE% %TIME%] rag-mcp re-ingest DONE (exit %ERRORLEVEL%)>> "%LOG%"
