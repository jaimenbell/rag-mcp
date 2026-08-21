"""A long-lived reader must not block the weekly --clean rebuild.

ROOT CAUSE (rebuild of 2026-08-16 died; root-caused 2026-08-21). A Chroma
PersistentClient holds the collection's HNSW segment files open for the life of
the client. On Windows an open handle makes the containing directory
un-deletable (PermissionError WinError 32 on data_level0.bin) and also
un-renameable (WinError 5), so the holder must genuinely let go -- there is no
rename-aside trick. The long-lived MCP server is exactly such a holder and never
participates in ReingestLock, which only coordinates the two ingest paths with
each other. That is why the 03:30 Sunday rebuild failed while the two earlier
ones (server down) succeeded.

THE POSITIVE CONTROL IS THE POINT of this module. test_open_reader_blocks_rmtree
asserts the hazard still EXISTS -- if a future Chroma stopped holding the handle,
or a future Windows stopped minding, that test fails and tells us this whole
mechanism is obsolete, rather than the fix quietly guarding nothing. Its negative
control sits beside it: with no reader, the same delete just works.

All state lives in tmp_path; the real store is never touched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from rag_mcp import server as server_mod
from rag_mcp.lock import ReingestLock
from rag_mcp.store import HashEmbedder, VectorStore

WINDOWS_ONLY = pytest.mark.skipif(
    os.name != "nt",
    reason="POSIX unlinks files that are still open, so a held handle cannot "
    "block rmtree there; this failure mode is Windows-specific.",
)

_READY_WAIT_S = 120.0


def _seed_store(path: Path) -> None:
    """Create a populated store, leaving NO handle open in this process.

    Seeding runs in a child that then exits, deliberately rather than opening
    the store here and calling close(). If seeding depended on close(), every
    test in this module would fail with AttributeError the moment close() was
    missing -- including the positive control, which must be able to demonstrate
    the hazard whether or not the fix is present. A control that only passes
    once the fix exists is not a control.
    """
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[2])
        from rag_mcp.store import HashEmbedder, VectorStore
        s = VectorStore(path=sys.argv[1], collection_name="knowledge",
                        embedder=HashEmbedder())
        s.add(ids=["a"], documents=["alpha beta gamma"],
              metadatas=[{"source": "a.md"}])
        # Query so the HNSW segment (data_level0.bin) is actually materialised
        # and opened -- a store that was only written may not have loaded the
        # segment, which would make this scenario silently un-reproducible.
        s.query("alpha", k=1)
        """
    )
    repo_root = str(Path(__file__).resolve().parent.parent)
    proc = subprocess.run(
        [sys.executable, "-c", code, str(path), repo_root],
        capture_output=True,
    )
    assert proc.returncode == 0, (
        "seeding child failed: " + proc.stderr.decode(errors="replace")[-1500:]
    )


def _spawn_reader(store_path: Path, ready: Path, release: Path) -> subprocess.Popen:
    """A separate process holding an open VectorStore, as the MCP server does."""
    code = textwrap.dedent(
        """
        import sys, time, pathlib
        sys.path.insert(0, sys.argv[4])
        from rag_mcp.store import HashEmbedder, VectorStore
        store_path, ready, release = sys.argv[1], sys.argv[2], sys.argv[3]
        s = VectorStore(path=store_path, collection_name="knowledge",
                        embedder=HashEmbedder())
        s.query("alpha", k=1)
        pathlib.Path(ready).write_text("up")
        while not pathlib.Path(release).exists():
            time.sleep(0.05)
        """
    )
    repo_root = str(Path(__file__).resolve().parent.parent)
    return subprocess.Popen(
        [sys.executable, "-c", code, str(store_path), str(ready), str(release),
         repo_root],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _await(flag: Path, proc: subprocess.Popen | None = None) -> None:
    deadline = time.monotonic() + _READY_WAIT_S
    while not flag.exists():
        if proc is not None and proc.poll() is not None:
            _, err = proc.communicate()
            raise AssertionError(
                "reader died before signalling ready: "
                + err.decode(errors="replace")[-1500:]
            )
        if time.monotonic() > deadline:
            raise AssertionError(f"flag {flag} never appeared")
        time.sleep(0.05)


@WINDOWS_ONLY
def test_open_reader_blocks_rmtree_positive_control(tmp_path: Path) -> None:
    """KNOWN-BAD: a second process holding the store makes --clean's delete fail.

    This is the control that proves the hazard is real. If it ever stops firing,
    the release machinery below is guarding nothing and should be reconsidered --
    that is a finding, not a pass.
    """
    store_path = tmp_path / "store.chroma"
    _seed_store(store_path)
    ready, release = tmp_path / "ready", tmp_path / "release"
    proc = _spawn_reader(store_path, ready, release)
    try:
        _await(ready, proc)
        with pytest.raises(PermissionError) as excinfo:
            shutil.rmtree(store_path)
        assert excinfo.value.winerror == 32, (
            "expected WinError 32 (file in use); a different error means the "
            "reproduction no longer matches the production failure"
        )
    finally:
        release.write_text("go")
        proc.wait(timeout=60)


def test_rmtree_succeeds_with_no_reader_negative_control(tmp_path: Path) -> None:
    """KNOWN-GOOD: with nothing holding the store, the same delete just works."""
    store_path = tmp_path / "store.chroma"
    _seed_store(store_path)
    shutil.rmtree(store_path)
    assert not store_path.exists()


def test_close_releases_the_handle(tmp_path: Path) -> None:
    """VectorStore.close() must actually free the OS handle, not just look tidy.

    Guards the private-API calls inside close(): if a future Chroma renames or
    drops _system.stop()/clear_system_cache(), this fails loudly instead of the
    server silently holding the store open forever.
    """
    store_path = tmp_path / "store.chroma"
    store = VectorStore(
        path=str(store_path), collection_name="knowledge", embedder=HashEmbedder()
    )
    store.add(ids=["a"], documents=["alpha"], metadatas=[{"source": "a.md"}])
    store.query("alpha", k=1)
    store.close()
    shutil.rmtree(store_path)  # would raise WinError 32 if the handle survived
    assert not store_path.exists()


def test_ensure_store_refuses_to_reopen_during_reindex(tmp_path: Path, monkeypatch) -> None:
    """The server must not re-grab the store while a reingest owns it.

    Without this the watcher's release is pointless: the next search would open
    a fresh handle and re-block the rmtree that the release just unblocked.
    """
    store_path = tmp_path / "store.chroma"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed_store(store_path)

    monkeypatch.setenv("RAG_MCP_CORPUS_ROOT", str(corpus))
    monkeypatch.setenv("RAG_MCP_DB_PATH", str(store_path))
    monkeypatch.setenv("RAG_MCP_EMBEDDER", "hash")
    monkeypatch.setitem(server_mod._STATE, "store", None)
    monkeypatch.setitem(server_mod._STATE, "db_path", None)

    lock = ReingestLock(store_path).acquire()
    try:
        with pytest.raises(server_mod.ReindexInProgress):
            server_mod._ensure_store()
    finally:
        lock.release()

    # And once the reingest is done, reads resume without a restart.
    store, root = server_mod._ensure_store()
    try:
        assert root == corpus.resolve()
    finally:
        server_mod._release_store()


def test_search_reports_reindex_distinctly_from_config_error(tmp_path: Path, monkeypatch) -> None:
    """A rebuild window must not masquerade as a broken config.

    Reported as its own error type so a caller is not sent off debugging their
    environment during scheduled maintenance.
    """
    store_path = tmp_path / "store.chroma"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed_store(store_path)

    monkeypatch.setenv("RAG_MCP_CORPUS_ROOT", str(corpus))
    monkeypatch.setenv("RAG_MCP_DB_PATH", str(store_path))
    monkeypatch.setenv("RAG_MCP_EMBEDDER", "hash")
    monkeypatch.setitem(server_mod._STATE, "store", None)
    monkeypatch.setitem(server_mod._STATE, "db_path", None)

    lock = ReingestLock(store_path).acquire()
    try:
        payload = server_mod._run_search({"query": "alpha", "k": 1})
    finally:
        lock.release()

    assert payload["ok"] is False
    assert payload["error"]["type"] == "reindex_in_progress"
    assert payload["results"] == []


def test_watcher_releases_the_store_when_a_reingest_takes_the_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """The idle-server case: nothing queries, so only the watcher can notice.

    The weekly rebuild runs at 03:30 Sunday against an idle server. A check that
    only ran per tool call would sleep straight through it -- which is the actual
    2026-08-16 failure -- so the watcher is what makes the handshake work.
    """
    store_path = tmp_path / "store.chroma"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed_store(store_path)

    monkeypatch.setenv("RAG_MCP_CORPUS_ROOT", str(corpus))
    monkeypatch.setenv("RAG_MCP_DB_PATH", str(store_path))
    monkeypatch.setenv("RAG_MCP_EMBEDDER", "hash")
    monkeypatch.setitem(server_mod._STATE, "store", None)
    monkeypatch.setitem(server_mod._STATE, "db_path", None)

    server_mod._ensure_store()
    assert server_mod._STATE["store"] is not None, "precondition: a handle is held"

    stop = threading.Event()
    watcher = threading.Thread(
        target=server_mod._watch_for_reindex, args=(stop,), daemon=True
    )
    watcher.start()
    lock = ReingestLock(store_path).acquire()
    try:
        deadline = time.monotonic() + 30
        while server_mod._STATE["store"] is not None:
            assert time.monotonic() < deadline, "watcher never released the store"
            time.sleep(0.1)
        # The real acceptance criterion: the rebuild's delete can now proceed.
        shutil.rmtree(store_path)
        assert not store_path.exists()
    finally:
        stop.set()
        watcher.join(timeout=10)
        lock.release()
        server_mod._release_store()
