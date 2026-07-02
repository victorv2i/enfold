"""Cross-process serialization for multi-step fact writes."""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - posix-only deployment target
    fcntl = None


_held_lock = threading.RLock()
_held_by_path = {}


@contextlib.contextmanager
def cross_process_write_lock(db_path: str) -> Iterator[None]:
    """Serialize multi-transaction writes across processes sharing *db_path*.

    The lock is keyed by the canonical database path and uses a sidecar file
    next to the database. It is reentrant within the current process, so a
    write path that already holds the sidecar can safely call another helper
    that also asks for it.
    """
    if fcntl is None:  # pragma: no cover
        yield
        return

    canonical = str(Path(db_path).expanduser().resolve())
    lock_path = f"{canonical}.mcp-write.lock"

    with _held_lock:
        state = _held_by_path.get(canonical)
        if state is not None:
            state["count"] += 1
            nested = True
            fh = state["fh"]
        else:
            nested = False
            fh = open(lock_path, "a+")
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            _held_by_path[canonical] = {"count": 1, "fh": fh}

    try:
        yield
    finally:
        if nested:
            with _held_lock:
                state = _held_by_path[canonical]
                state["count"] -= 1
            return

        with _held_lock:
            state = _held_by_path.pop(canonical, None)
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
