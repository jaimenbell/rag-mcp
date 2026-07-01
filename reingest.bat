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

set "REPO=C:\Users\jaime\projects\rag-mcp"
set "VAULT=<your-corpus-path>"
set "STORE=C:\Users\jaime\projects\rag-mcp\store.chroma"
set "PY=%REPO%\.venv\Scripts\python.exe"
set "LOG=%REPO%\logs\reingest.log"

set "RAG_MCP_COLLECTION=knowledge"
set "RAG_MCP_EMBEDDER=default"

if not exist "%REPO%\logs" mkdir "%REPO%\logs"

echo ==================================================================>> "%LOG%"
echo [%DATE% %TIME%] rag-mcp re-ingest START>> "%LOG%"
cd /d "%REPO%"
"%PY%" -m rag_mcp.cli ingest "%VAULT%" --db "%STORE%" >> "%LOG%" 2>&1
echo [%DATE% %TIME%] rag-mcp re-ingest DONE (exit %ERRORLEVEL%)>> "%LOG%"
