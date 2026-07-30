@echo off
REM ============================================================================
REM rag-mcp INCREMENTAL re-ingest routine
REM Re-embeds only the corpus files whose CONTENT CHANGED since the last run, so
REM search_knowledge stays current without paying for a full re-embed each time.
REM
REM Registered as scheduled task "rag-mcp-reingest". Intended cadence: EVERY 15
REM MINUTES. It used to be daily 03:00, which meant a note written at 03:05 was
REM invisible to search_knowledge for nearly 24 hours.
REM
REM Cost, measured 2026-07-30 on the live 2808-file / 26.6 MiB corpus:
REM   tick with NO changes : ~0.7s   (walk + read + sha256 of every file)
REM   full re-embed        : ~2h33m  (50,109 chunks at ~5.5 chunks/sec, bge CPU)
REM A normal tick costs about a second; only genuinely edited files are
REM re-embedded, at roughly 5.5 chunks/sec.
REM
REM FIRST RUN AFTER DEPLOY has no manifest and therefore does a FULL ~2.5h
REM embed. That is expected and self-limiting: ticks firing while it runs hit the
REM cross-process lock and exit 3 (a no-op) -- they never queue or collide.
REM
REM Incremental ingest also PRUNES, which the old upsert-only run never did:
REM chunks of deleted/renamed notes, and trailing chunks of notes that got
REM shorter. reingest-clean.bat (weekly Sun 03:30) stays as the belt-and-braces
REM full rebuild.
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
set "STORE=%REPO%\store-bge.chroma"
set "PY=%REPO%\.venv\Scripts\python.exe"
set "LOG=%REPO%\logs\reingest.log"

set "RAG_MCP_COLLECTION=knowledge"
set "RAG_MCP_EMBEDDER=bge"

if not exist "%REPO%\logs" mkdir "%REPO%\logs"

REM Rotate at ~5 MB so a 15-minute cadence cannot grow the log without bound.
for %%A in ("%LOG%") do if %%~zA GTR 5000000 move /y "%LOG%" "%LOG%.1" >nul

REM One line per tick (--quiet) instead of a banner + 10-line pretty JSON: at
REM every 15 minutes this runs ~96 times a day and the old format was unreadable.
cd /d "%REPO%"
"%PY%" -m rag_mcp.cli --embedder %RAG_MCP_EMBEDDER% ingest "%VAULT%" --db "%STORE%" --quiet >> "%LOG%" 2>&1
REM exit 3 == another run holds the lock (e.g. the weekly clean rebuild). At this
REM cadence that is a normal no-op, not an error worth alerting on.
if errorlevel 1 echo [%DATE% %TIME%] rag-mcp incremental re-ingest exited nonzero>> "%LOG%"
