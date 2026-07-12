"""Cron-friendly rehearsal of the newest verified SQLite backup.

Run with ``python -m enfold.backup_rehearsal --help``. The live database is
opened read-only and the restored artifact exists only in a temporary folder.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Sequence

from .backup import BackupError, restore_database, sqlite_file_uri


class RehearsalError(RuntimeError):
    """Raised after a failed rehearsal report has been persisted."""


@dataclass(frozen=True, slots=True)
class RestoreRehearsalReport:
    status: str
    rehearsed_at: str
    live_database: str
    backup: str
    quick_check: tuple[str, ...]
    live_fact_count: int | None
    restored_fact_count: int | None
    fact_count_tolerance: int
    error: str | None


def _newest_backup(directory: Path) -> Path:
    candidates = [path for path in directory.glob("*.sqlite") if path.is_file()]
    if not candidates:
        raise BackupError(f"no SQLite backups found in {directory}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _fact_count(path: Path) -> int:
    with sqlite3.connect(sqlite_file_uri(path.resolve(), mode="ro"), uri=True) as conn:
        return int(conn.execute("SELECT count(*) FROM facts").fetchone()[0])


def _write_report(state_dir: Path, report: RestoreRehearsalReport) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.rehearsed_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    destination = state_dir / f"restore-rehearsal-{stamp}.json"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=state_dir
    )
    try:
        payload = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        return destination
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def rehearse_latest_backup(
    live_database: str | os.PathLike[str],
    backup_directory: str | os.PathLike[str],
    state_directory: str | os.PathLike[str],
    *,
    fact_count_tolerance: int = 0,
) -> RestoreRehearsalReport:
    """Restore the newest backup, validate it, and persist pass/fail evidence."""

    if fact_count_tolerance < 0:
        raise ValueError("fact count tolerance must be non-negative")
    live = Path(live_database).expanduser().resolve()
    backups = Path(backup_directory).expanduser().resolve()
    state = Path(state_directory).expanduser().resolve()
    rehearsed_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    backup: Path | None = None
    quick_check: tuple[str, ...] = ()
    live_count: int | None = None
    restored_count: int | None = None
    error: str | None = None
    try:
        backup = _newest_backup(backups)
        live_count = _fact_count(live)
        with tempfile.TemporaryDirectory(prefix="enfold-restore-rehearsal-") as work:
            restored = Path(work) / "restored.sqlite"
            restore_database(backup, restored)
            with sqlite3.connect(
                sqlite_file_uri(restored.resolve(), mode="ro"), uri=True
            ) as conn:
                quick_check = tuple(
                    str(row[0]) for row in conn.execute("PRAGMA quick_check")
                )
                restored_count = int(
                    conn.execute("SELECT count(*) FROM facts").fetchone()[0]
                )
        if quick_check != ("ok",):
            raise BackupError(f"restored database quick_check failed: {quick_check}")
        difference = abs(live_count - restored_count)
        if difference > fact_count_tolerance:
            raise BackupError(
                "restored fact count differs from live by "
                f"{difference}, exceeding tolerance {fact_count_tolerance}"
            )
        status = "passed"
    except (BackupError, OSError, sqlite3.DatabaseError) as exc:
        status = "failed"
        error = str(exc)

    report = RestoreRehearsalReport(
        status=status,
        rehearsed_at=rehearsed_at,
        live_database=str(live),
        backup="" if backup is None else str(backup.resolve()),
        quick_check=quick_check,
        live_fact_count=live_count,
        restored_fact_count=restored_count,
        fact_count_tolerance=fact_count_tolerance,
        error=error,
    )
    _write_report(state, report)
    if status != "passed":
        raise RehearsalError(f"restore rehearsal failed: {error}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("live_database", help="live SQLite database (opened read-only)")
    parser.add_argument("backup_directory", help="directory containing *.sqlite backups")
    parser.add_argument("state_directory", help="directory for dated JSON reports")
    parser.add_argument("--fact-count-tolerance", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = rehearse_latest_backup(
            args.live_database,
            args.backup_directory,
            args.state_directory,
            fact_count_tolerance=args.fact_count_tolerance,
        )
    except (RehearsalError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
