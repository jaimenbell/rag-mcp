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
from rag_mcp.lock import LockHeld, ReingestLock


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
