"""Cross-process reingest lock tests. All state lives in tmp_path (never the real store).

Covers: acquire/release lifecycle, fail-fast on a live holder, stale-lock reclaim
(dead PID and age ceiling), and that a --clean rebuild is blocked while the lock
is held by a live process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

from rag_mcp.cli import main
from rag_mcp.lock import (
    DEFAULT_STALE_AFTER_S,
    FULL_REEMBED_OBSERVED_S,
    LockHeld,
    ReingestLock,
    live_holder,
)


def _dead_pid() -> int:
    """A PID guaranteed not to be running: spawn a trivial child and fully reap it."""
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


def test_lock_acquired_and_released_on_normal_exit(tmp_path):
    store = tmp_path / "store.chroma"
    lock_file = tmp_path / "store.chroma.lock"
    with ReingestLock(store) as lock:
        assert lock_file.exists()
        holder = json.loads(lock_file.read_text(encoding="utf-8"))
        assert holder["pid"] == os.getpid()
        assert holder["acquired"]  # ISO timestamp present
    assert not lock_file.exists()  # released on normal exit


def test_second_process_fails_fast_while_held(tmp_path):
    store = tmp_path / "store.chroma"
    with ReingestLock(store):
        with pytest.raises(LockHeld):
            # Second acquirer sees a live (this-process) holder -> fail fast.
            ReingestLock(store).acquire()


def test_stale_lock_dead_pid_reclaimed(tmp_path):
    store = tmp_path / "store.chroma"
    lock_file = tmp_path / "store.chroma.lock"
    lock_file.write_text(
        json.dumps(
            {
                "pid": _dead_pid(),
                "acquired": datetime.now(timezone.utc).isoformat(),
                "host": "somehost",
            }
        ),
        encoding="utf-8",
    )
    # Dead holder -> reclaimable even though the timestamp is fresh.
    with ReingestLock(store) as lock:
        holder = json.loads(lock_file.read_text(encoding="utf-8"))
        assert holder["pid"] == os.getpid()
    assert not lock_file.exists()


def test_stale_lock_age_ceiling_reclaimed(tmp_path):
    store = tmp_path / "store.chroma"
    lock_file = tmp_path / "store.chroma.lock"
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    # Use THIS live pid so only the age ceiling (not liveness) can trigger reclaim.
    lock_file.write_text(
        json.dumps({"pid": os.getpid(), "acquired": old, "host": "h"}),
        encoding="utf-8",
    )
    with ReingestLock(store, stale_after_s=2 * 60 * 60):  # 2h ceiling, lock is 5h old
        holder = json.loads(lock_file.read_text(encoding="utf-8"))
        assert datetime.fromisoformat(holder["acquired"]) > datetime.fromisoformat(old)


def test_live_holder_mid_full_reembed_not_reclaimed(tmp_path):
    """A live run that has held the lock longer than 2h must NOT be stolen.

    Regression for 2026-07-30: DEFAULT_STALE_AFTER_S was 2h while a full
    re-embed of the live corpus measured 2h32m53s. At the new 15-minute
    cadence the tick firing at the 2h mark saw a LIVE holder, applied the age
    ceiling anyway, and reclaimed the lock -- starting a SECOND concurrent
    writer against the same Chroma store. That is precisely the corruption
    this module exists to prevent, and it was reachable on every first run
    after deploy (no manifest -> full embed -> >2h).
    """
    store = tmp_path / "store.chroma"
    lock_file = tmp_path / "store.chroma.lock"
    # Live pid, held 3h: past the old ceiling but still a legitimate full embed.
    held_since = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    lock_file.write_text(
        json.dumps({"pid": os.getpid(), "acquired": held_since, "host": "h"}),
        encoding="utf-8",
    )
    with pytest.raises(LockHeld):
        ReingestLock(store).acquire()  # DEFAULT ceiling, not an override


def test_default_ceiling_exceeds_observed_full_reembed():
    """Pin the RELATIONSHIP, not the bare number.

    The ceiling is only safe while it clears the longest legitimate run. If the
    corpus grows enough to slow a full re-embed toward the ceiling, this fails
    and forces the ceiling up -- instead of a concurrent writer appearing with
    no symptom until the index is already corrupt.
    """
    assert DEFAULT_STALE_AFTER_S >= 2 * FULL_REEMBED_OBSERVED_S


def test_live_holder_reports_a_running_writer(tmp_path):
    """live_holder is how a caller asks "is production writing this store NOW?"."""
    store = tmp_path / "store.chroma"
    (tmp_path / "store.chroma.lock").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "acquired": datetime.now(timezone.utc).isoformat(),
                "host": "h",
            }
        ),
        encoding="utf-8",
    )
    holder = live_holder(store)
    assert holder is not None
    assert holder["pid"] == os.getpid()


def test_live_holder_is_none_when_no_lock_exists(tmp_path):
    assert live_holder(tmp_path / "store.chroma") is None


def test_live_holder_ignores_a_dead_writer(tmp_path):
    """A leftover lock from a crashed run is not a live writer."""
    store = tmp_path / "store.chroma"
    (tmp_path / "store.chroma.lock").write_text(
        json.dumps(
            {
                "pid": _dead_pid(),
                "acquired": datetime.now(timezone.utc).isoformat(),
                "host": "h",
            }
        ),
        encoding="utf-8",
    )
    assert live_holder(store) is None


def test_fresh_live_lock_not_reclaimed(tmp_path):
    store = tmp_path / "store.chroma"
    lock_file = tmp_path / "store.chroma.lock"
    lock_file.write_text(
        json.dumps(
            {
                "pid": os.getpid(),  # live
                "acquired": datetime.now(timezone.utc).isoformat(),  # fresh
                "host": "h",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LockHeld):
        ReingestLock(store).acquire()


def test_clean_blocked_while_lock_held(tmp_path, capsys):
    """--clean must fail fast (rc 3) and NOT delete the store while the lock is live."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("# Note\n\nhello world", encoding="utf-8")
    store = tmp_path / "store.chroma"
    store.mkdir()
    sentinel = store / "sentinel.txt"
    sentinel.write_text("do not delete me", encoding="utf-8")

    # Hold the lock, then try a --clean ingest through the CLI: it must refuse.
    with ReingestLock(store):
        rc = main(
            [
                "--embedder", "hash",
                "ingest", str(corpus),
                "--db", str(store),
                "--clean",
            ]
        )
    assert rc == 3  # fail-fast exit code
    assert "SKIP" in capsys.readouterr().err
    assert sentinel.exists()  # destructive delete never happened


def test_clean_deletes_store_after_acquiring_lock(tmp_path):
    """When the lock is free, --clean removes the old store then rebuilds it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("# Note\n\nhello world", encoding="utf-8")
    store = tmp_path / "store.chroma"
    store.mkdir()
    stale = store / "orphan.txt"
    stale.write_text("chunk for a deleted note", encoding="utf-8")

    rc = main(
        ["--embedder", "hash", "ingest", str(corpus), "--db", str(store), "--clean"]
    )
    assert rc == 0
    assert not stale.exists()  # old store dir was wiped by --clean
    assert store.exists()      # rebuilt fresh
    assert not (tmp_path / "store.chroma.lock").exists()  # lock released
