"""Cross-process mutual exclusion for the re-ingest routines.

Both the daily upsert re-ingest and the weekly ``--clean`` rebuild write to the
SAME on-disk Chroma store. Without a lock, a ``--clean`` run that deletes
``store.chroma`` mid-write of a concurrent daily run corrupts the index. This
module provides a single, stdlib-only, cross-process lock guarding every
invocation path (both modes acquire the SAME lock).

Design:
  * Atomic create via ``os.open(O_CREAT | O_EXCL)`` -- no TOCTOU window.
  * Lock file sits next to the store (same volume) and holds PID + ISO timestamp.
  * A held lock is honoured ONLY if the holder looks alive: reclaim (steal) the
    lock when the holder PID is dead OR the lock is older than ``stale_after``
    (reingest takes ~13 min; ceiling defaults to 2 h).
  * On a live conflict we FAIL FAST (raise :class:`LockHeld`) -- callers exit
    non-zero and let the scheduler retry next cycle. We never queue or block.
"""
from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

# Default staleness ceiling: a real reingest takes ~13 min. 2 h is a generous
# ceiling that reclaims a genuinely wedged lock without racing a slow-but-live run.
DEFAULT_STALE_AFTER_S = 2 * 60 * 60


class LockHeld(RuntimeError):
    """Raised when the lock is currently held by a live process."""

    def __init__(self, holder: dict, lock_path: Path) -> None:
        self.holder = holder
        self.lock_path = lock_path
        super().__init__(
            f"reingest lock held by pid={holder.get('pid')} "
            f"since {holder.get('acquired')} ({lock_path})"
        )


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check that never signals/kills the target process.

    Conservative on ambiguity: an access-denied probe counts as ALIVE so we never
    steal a lock from a running-but-unqueryable process. The ``stale_after`` age
    ceiling is the backstop for a truly wedged holder.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        ERROR_INVALID_PARAMETER = 87  # OpenProcess -> no such pid
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Only a "no such process" error proves the pid is dead. Any other
            # failure (e.g. access denied) is treated as alive (do not reclaim).
            return kernel32.GetLastError() != ERROR_INVALID_PARAMETER
        try:
            STILL_ACTIVE = 259
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    # POSIX: signal 0 performs error-checking without delivering a signal.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_holder(lock_path: Path) -> dict | None:
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _is_stale(holder: dict | None, stale_after_s: float) -> bool:
    """A holder is stale (reclaimable) if its PID is dead OR it is too old."""
    if holder is None:
        return True  # unreadable/corrupt lock -> reclaim
    pid = holder.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return True
    acquired = holder.get("acquired")
    try:
        ts = datetime.fromisoformat(acquired).timestamp()
    except (TypeError, ValueError):
        return True  # missing/garbled timestamp -> treat as stale
    return (time.time() - ts) > stale_after_s


def _write_lock(fd: int) -> dict:
    holder = {
        "pid": os.getpid(),
        "acquired": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
    }
    os.write(fd, json.dumps(holder).encode("utf-8"))
    return holder


class ReingestLock:
    """Context manager acquiring an exclusive, cross-process reingest lock.

    Usage::

        with ReingestLock(store_path):
            ...  # clean/delete + ingest, guaranteed single-writer

    Raises :class:`LockHeld` immediately if a live process holds the lock.
    The lock file is ``<store_path>.lock`` (sibling of the store, same volume).
    """

    def __init__(
        self,
        store_path: str | os.PathLike,
        *,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
    ) -> None:
        store = Path(store_path)
        self.lock_path = store.with_name(store.name + ".lock")
        self.stale_after_s = stale_after_s
        self._fd: int | None = None
        self.holder: dict | None = None

    def acquire(self) -> "ReingestLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # One reclaim retry: create -> on collision, check staleness, steal, retry.
        for attempt in range(2):
            try:
                fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                holder = _read_holder(self.lock_path)
                if attempt == 0 and _is_stale(holder, self.stale_after_s):
                    # Reclaim the stale lock, then loop once to re-create atomically.
                    try:
                        self.lock_path.unlink()
                    except FileNotFoundError:
                        pass  # another process just reclaimed it; retry create
                    continue
                raise LockHeld(holder or {}, self.lock_path)
            else:
                self.holder = _write_lock(fd)
                os.close(fd)
                self._fd = fd
                return self
        # Both attempts collided with a freshly re-created live lock.
        raise LockHeld(_read_holder(self.lock_path) or {}, self.lock_path)

    def release(self) -> None:
        # Only remove a lock we still own (guards against deleting a reclaimer's lock).
        holder = _read_holder(self.lock_path)
        if holder is not None and holder.get("pid") == os.getpid():
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
        self._fd = None

    def __enter__(self) -> "ReingestLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
