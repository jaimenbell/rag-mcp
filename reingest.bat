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
REM Cost, RE-MEASURED IN PRODUCTION 2026-07-30 on the live 2810-file corpus
REM (the earlier figures here were estimates and both were wrong):
REM   tick with NO changes : ~1.8s   (walk + read + sha256 of every file, plus
REM                                   interpreter start and bge model load).
REM                                   Measured 1.836s / 1.796s back-to-back with
REM                                   files_unchanged=2810, chunks_added=0.
REM                                   The old "~0.7s" here was 2.6x optimistic
REM                                   and is what made a "sub-second" acceptance
REM                                   criterion get written into three handoffs.
REM   full re-embed        : ~2h43m  (lock acquired 10:15:02, manifest written
REM                                   ~12:58; 50,311 chunks, bge CPU). The old
REM                                   "~2h33m" understated it.
REM A normal tick costs about two seconds; only genuinely edited files are
REM re-embedded. A tick that re-embeds one changed note measured ~9.5s.
REM
REM FIRST RUN AFTER DEPLOY has no manifest and therefore does a FULL ~2.7h
REM embed. That is expected and self-limiting -- but NOT by the mechanism this
REM comment used to claim. The scheduled task "rag-mcp-reingest" is registered
REM MultipleInstances=IgnoreNew, so Task Scheduler REFUSES an overlapping tick
REM before cmd.exe or python ever start: it records LastTaskResult 2147946720
REM (0x800710E0) and writes NOTHING to this log. Verified 2026-07-30, when the
REM 10:15 full embed ran ~2h43m and the 10:30/10:45/11:00/11:15/... ticks left
REM no trace at all. The cross-process lock + exit 3 path below IS real and is
REM the belt to that braces -- it catches a MANUAL run, or the weekly
REM reingest-clean.bat, overlapping a scheduled one -- but it is not what
REM protects the scheduled cadence from itself. A silent gap in this log during
REM a long embed is therefore EXPECTED, not a sign the schedule died.
REM
REM Incremental ingest also PRUNES, which the old upsert-only run never did:
REM chunks of deleted/renamed notes, and trailing chunks of notes that got
REM shorter. reingest-clean.bat (weekly Sun 03:30) stays as the belt-and-braces
REM full rebuild.
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
REM echo goes into a nonexistent directory and is silently lost.
if not exist "%REPO%\logs" mkdir "%REPO%\logs"

REM Corpus path comes from an UNTRACKED local file so the public repo stays
REM generic (2026-07-22: the public-scrub sanitized a hardcoded path here and
REM the nightly ingest silently ran against a placeholder). Create it once:
REM   echo C:\path\to\your\corpus> local-corpus.txt
if not exist "%REPO%\local-corpus.txt" (
  echo [%DATE% %TIME%] FATAL: local-corpus.txt missing -- see reingest.bat header ^(REPO=%REPO%, USERPROFILE=%USERPROFILE%^)>> "%LOG%"
  exit /b 1
)
set /p VAULT=<"%REPO%\local-corpus.txt"
REM CUTOVER 2026-07-02: bge-large-en-v1.5 store (1024-dim). Old MiniLM store
REM kept at store.chroma for instant rollback (repoint STORE + EMBEDDER back).
set "STORE=%REPO%\store-bge.chroma"
set "PY=%REPO%\.venv\Scripts\python.exe"

set "RAG_MCP_COLLECTION=knowledge"
set "RAG_MCP_EMBEDDER=bge"

REM Rotate at ~5 MB so a 15-minute cadence cannot grow the log without bound.
for %%A in ("%LOG%") do if %%~zA GTR 5000000 move /y "%LOG%" "%LOG%.1" >nul

REM One line per tick (--quiet) instead of a banner + 10-line pretty JSON: at
REM every 15 minutes this runs ~96 times a day and the old format was unreadable.
cd /d "%REPO%"
"%PY%" -m rag_mcp.cli --embedder %RAG_MCP_EMBEDDER% ingest "%VAULT%" --db "%STORE%" --quiet >> "%LOG%" 2>&1
REM Capture immediately, on its own line. Do NOT read %ERRORLEVEL% inside a
REM parenthesized block -- it expands at parse time there and freezes.
set "RC=%ERRORLEVEL%"

REM exit 3 == another run holds the lock (e.g. the weekly clean rebuild). At this
REM cadence that is a normal no-op, not an error worth alerting on, so it is
REM mapped to success rather than surfaced to the scheduler as a red run.
REM Every other nonzero code now PROPAGATES: before 2026-07-31 the tail was a
REM bare conditional echo with no exit /b, so this script reported exit 0 even
REM when the ingest genuinely failed.
if "%RC%"=="3" exit /b 0
if not "%RC%"=="0" echo [%DATE% %TIME%] rag-mcp incremental re-ingest exited nonzero (exit %RC%)>> "%LOG%"
exit /b %RC%
