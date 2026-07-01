from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

TERMINAL_EXTRACT_QUEUE_STATUSES = ("done", "completed", "succeeded", "failed", "dead")


@dataclass(frozen=True)
class BackupResult:
    source: Path
    destination: Path
    quick_check: str
    bytes: int


def _sqlite_uri(path: Path, mode: str) -> str:
    return f"file:{path}?mode={mode}"


def connect_readonly(path: str | Path) -> sqlite3.Connection:
    """Open an existing SQLite database read-only without creating it."""
    db_path = Path(path)
    conn = sqlite3.connect(_sqlite_uri(db_path, "ro"), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def quick_check(path: str | Path) -> str:
    """Return SQLite PRAGMA quick_check for an existing database."""
    with closing(connect_readonly(path)) as conn:
        row = conn.execute("PRAGMA quick_check").fetchone()
    return str(row[0]) if row else ""


def _remove_sqlite_sidecars(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            candidate.unlink()


def backup_sqlite_db(src: str | Path, dst: str | Path, *, overwrite: bool = False) -> BackupResult:
    """Copy a SQLite DB with sqlite3's backup API and verify the copy.

    This intentionally avoids shutil.copy for live/WAL databases. It is safe for
    read-only evaluation snapshots and does not mutate the source database.
    """
    source = Path(src)
    destination = Path(dst)

    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different paths")
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _remove_sqlite_sidecars(destination)

    with closing(connect_readonly(source)) as src_conn:
        with closing(sqlite3.connect(destination)) as dst_conn:
            src_conn.backup(dst_conn)
            dst_conn.commit()

    check = quick_check(destination)
    if check.lower() != "ok":
        raise sqlite3.DatabaseError(f"backup quick_check failed: {check}")

    return BackupResult(
        source=source,
        destination=destination,
        quick_check=check,
        bytes=destination.stat().st_size,
    )
