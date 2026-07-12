"""Verified SQLite backup and restore helpers.

All copies use SQLite's online backup API, which includes committed WAL pages.
Plain filesystem copying is intentionally absent.  Callers choose the source
and destination explicitly; this module has no knowledge of the live Hermes
database path.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TypeAlias
from urllib.parse import quote


Database: TypeAlias = sqlite3.Connection | str | os.PathLike[str]


_LOGGER = logging.getLogger(__name__)


class BackupError(RuntimeError):
    """Raised when a copy is unsafe or verification fails."""


@contextlib.contextmanager
def maintenance_database_lock(
    database: str | os.PathLike[str], *, timeout: float = 0.0
) -> Iterator[None]:
    """Exclude daemon and legacy writers for an entire maintenance operation.

    Locks are always acquired in daemon-then-legacy order.  The stable
    sidecars are deliberately retained after release so no waiter can be
    stranded on an unlinked inode.  Acquisition is bounded so an operator
    receives a refusal instead of an indefinite hang.
    """

    if timeout < 0:
        raise BackupError("maintenance lock timeout must be non-negative")
    canonical = Path(database).expanduser().resolve()
    canonical.parent.mkdir(parents=True, exist_ok=True)
    sidecars = (
        (canonical.with_name(canonical.name + ".enfold.lock"), "Enfold daemon"),
        (canonical.with_name(canonical.name + ".mcp-write.lock"), "legacy writer"),
    )
    held: list[int] = []
    deadline = time.monotonic() + timeout
    try:
        for sidecar, owner in sidecars:
            flags = os.O_RDWR | os.O_CREAT
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(sidecar, flags, 0o600)
            except OSError as exc:
                raise BackupError(
                    f"cannot open {owner} lock sidecar {sidecar}: {exc}"
                ) from exc
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                    raise BackupError(
                        f"{owner} lock sidecar must be a regular file owned by this user"
                    )
                if info.st_mode & 0o022:
                    raise BackupError(
                        f"{owner} lock sidecar must not be group/world writable"
                    )
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError as exc:
                        if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                            raise
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise BackupError(
                                f"{owner} owns the database; refusing maintenance "
                                "after the bounded lock wait"
                            ) from exc
                        time.sleep(min(0.05, remaining))
            except BaseException:
                os.close(fd)
                raise
            held.append(fd)
        yield
    finally:
        for fd in reversed(held):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


@dataclass(frozen=True)
class VerificationReport:
    integrity_check: tuple[str, ...]
    foreign_key_violations: tuple[tuple[object, ...], ...]
    fts_tables_checked: tuple[str, ...]
    fts_errors: tuple[str, ...]
    row_counts: dict[str, int]

    @property
    def ok(self) -> bool:
        return (
            self.integrity_check == ("ok",)
            and not self.foreign_key_violations
            and not self.fts_errors
        )


@dataclass(frozen=True)
class SecondaryCopyReport:
    status: str
    destination: str | None
    encrypted: bool
    error: str | None


@dataclass(frozen=True)
class CopyReport:
    operation: str
    source: VerificationReport
    destination: VerificationReport
    secondary: SecondaryCopyReport | None = None

    @property
    def row_counts_match(self) -> bool:
        return self.source.row_counts == self.destination.row_counts

    @property
    def ok(self) -> bool:
        return self.source.ok and self.destination.ok and self.row_counts_match


def sqlite_file_uri(path: str | os.PathLike[str], *, mode: str) -> str:
    """Build a SQLite file URI with an encoded filesystem path component."""

    if mode not in {"ro", "rw", "rwc"}:
        raise ValueError(f"unsupported SQLite URI mode: {mode}")
    encoded_path = quote(Path(path).as_posix(), safe="/")
    return f"file:{encoded_path}?mode={mode}"


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _visible_tables(conn: sqlite3.Connection) -> tuple[tuple[str, str], ...]:
    """Return countable user tables, excluding SQLite/FTS shadow tables."""

    try:
        rows = conn.execute("PRAGMA table_list").fetchall()
    except sqlite3.DatabaseError:
        rows = []
    if rows:
        return tuple(
            sorted(
                (str(row[1]), str(row[2]))
                for row in rows
                if row[0] == "main"
                and row[2] in {"table", "virtual"}
                and not str(row[1]).startswith("sqlite_")
            )
        )

    # SQLite before table_list: identify virtual tables and their conventional
    # shadow-table names from sqlite_master.
    master = conn.execute(
        "SELECT name, COALESCE(sql, '') FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    virtual = {
        str(name)
        for name, sql in master
        if str(sql).lstrip().upper().startswith("CREATE VIRTUAL TABLE")
    }
    shadows = {
        f"{name}_{suffix}"
        for name in virtual
        for suffix in ("data", "idx", "content", "docsize", "config", "segments", "segdir", "stat")
    }
    return tuple(
        sorted(
            (str(name), "virtual" if name in virtual else "table")
            for name, _ in master
            if name not in shadows
        )
    )


def _fts5_tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT name, COALESCE(sql, '') FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return tuple(
        sorted(
            str(name)
            for name, sql in rows
            if "USING FTS5" in " ".join(str(sql).upper().split())
        )
    )


def verify_database(
    conn: sqlite3.Connection, *, check_fts: bool = True
) -> VerificationReport:
    """Collect integrity, FK, FTS5, and row-count evidence from ``conn``."""

    integrity = tuple(str(row[0]) for row in conn.execute("PRAGMA integrity_check"))
    foreign_keys = tuple(tuple(row) for row in conn.execute("PRAGMA foreign_key_check"))
    tables = _visible_tables(conn)
    row_counts = {
        name: int(conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(name)}").fetchone()[0])
        for name, _kind in tables
    }

    checked: list[str] = []
    errors: list[str] = []
    for table in _fts5_tables(conn) if check_fts else ():
        checked.append(table)
        savepoint = "enfold_fts_integrity_check"
        try:
            conn.execute(f"SAVEPOINT {savepoint}")
            conn.execute(
                f"INSERT INTO {_quote_identifier(table)}({_quote_identifier(table)}) "
                "VALUES ('integrity-check')"
            )
        except sqlite3.DatabaseError as exc:
            errors.append(f"{table}: {exc}")
        finally:
            # The FTS command is logically read-only, but INSERT syntax can
            # open a transaction.  A savepoint ensures verification never
            # leaves transactional state behind or alters indexed content.
            try:
                conn.execute(f"ROLLBACK TO {savepoint}")
                conn.execute(f"RELEASE {savepoint}")
            except sqlite3.DatabaseError:
                pass

    return VerificationReport(
        integrity_check=integrity,
        foreign_key_violations=foreign_keys,
        fts_tables_checked=tuple(checked),
        fts_errors=tuple(errors),
        row_counts=row_counts,
    )


def _assert_valid(report: VerificationReport, label: str) -> None:
    if not report.ok:
        raise BackupError(f"{label} database failed verification: {report}")


def _open_database(
    value: Database, *, destination: bool, overwrite: bool
) -> tuple[sqlite3.Connection, bool]:
    if isinstance(value, sqlite3.Connection):
        if value.in_transaction:
            raise BackupError("database connection has an active transaction")
        if destination and not overwrite:
            populated = value.execute(
                "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1"
            ).fetchone()
            if populated:
                raise BackupError("destination connection is not empty")
        return value, False

    path = Path(value).expanduser()
    if destination:
        if path.exists() and not overwrite:
            raise BackupError(f"destination already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(path), True
    if not path.is_file():
        raise BackupError(f"source database does not exist: {path}")
    return sqlite3.connect(sqlite_file_uri(path, mode="ro"), uri=True), True


def _path_identity(value: Database) -> Path | None:
    if isinstance(value, sqlite3.Connection):
        return None
    return Path(value).expanduser().resolve()


def _destination_path(value: Database) -> Path | None:
    if isinstance(value, sqlite3.Connection):
        return None
    return Path(value).expanduser()


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    """Persist a completed rename on filesystems that support directory fsync."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _assert_no_destination_sidecars(path: Path) -> None:
    """Refuse to replace a database that may still have active WAL state.

    Replacing only the main file while a previous generation's WAL/SHM files
    remain is not a safe database-level atomic operation.  The maintenance
    workflow must close/checkpoint all destination writers first.
    """

    sidecars = [Path(f"{path}{suffix}") for suffix in ("-wal", "-shm")]
    present = [candidate for candidate in sidecars if candidate.exists()]
    if present:
        names = ", ".join(str(candidate) for candidate in present)
        raise BackupError(
            "destination has SQLite WAL sidecars; close/checkpoint all writers "
            f"before overwrite: {names}"
        )


def _atomic_path_destination(
    source: sqlite3.Connection,
    destination_path: Path,
    *,
    overwrite: bool,
) -> VerificationReport:
    """Copy into a same-directory artifact, verify it, then atomically publish."""

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    existed = destination_path.exists()
    if existed and not overwrite:
        raise BackupError(f"destination already exists: {destination_path}")
    # Sidecars without a main file can be leftovers from a crashed/removed
    # generation and are just as unsafe to pair with the new snapshot.
    _assert_no_destination_sidecars(destination_path)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    destination: sqlite3.Connection | None = None
    published = False
    try:
        destination = sqlite3.connect(temporary_path)
        source.backup(destination)
        destination_report = verify_database(destination)
        _assert_valid(destination_report, "destination")
        destination.close()
        destination = None

        # Preserve the existing artifact's permission bits; new artifacts stay
        # private (mkstemp creates them with mode 0600).
        if existed:
            mode = stat.S_IMODE(destination_path.stat().st_mode)
            os.chmod(temporary_path, mode)
        _fsync_file(temporary_path)

        # Recheck immediately before publication so an ordinary concurrent
        # creator is not silently overwritten when overwrite=False.
        if not overwrite and destination_path.exists():
            raise BackupError(f"destination already exists: {destination_path}")
        os.replace(temporary_path, destination_path)
        published = True
        _fsync_directory(destination_path.parent)
        return destination_report
    finally:
        if destination is not None:
            destination.close()
        if not published:
            temporary_path.unlink(missing_ok=True)


def _copy(
    source_value: Database,
    destination_value: Database,
    *,
    operation: str,
    overwrite: bool,
) -> CopyReport:
    source_path = _path_identity(source_value)
    destination_path = _path_identity(destination_value)
    if source_value is destination_value or (
        source_path is not None and source_path == destination_path
    ):
        raise BackupError("source and destination must be different databases")
    source, close_source = _open_database(
        source_value, destination=False, overwrite=False
    )
    destination: sqlite3.Connection | None = None
    close_destination = False
    try:
        # The source may intentionally be read-only.  The copied artifact is
        # writable and receives the FTS-specific check; integrity_check still
        # covers the source's FTS shadow tables before copying.
        source_report = verify_database(source, check_fts=False)
        _assert_valid(source_report, "source")
        destination_path_value = _destination_path(destination_value)
        if destination_path_value is not None:
            destination_report = _atomic_path_destination(
                source,
                destination_path_value,
                overwrite=overwrite,
            )
        else:
            destination, close_destination = _open_database(
                destination_value, destination=True, overwrite=overwrite
            )
            if overwrite:
                populated = destination.execute(
                    "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1"
                ).fetchone()
                if populated:
                    raise BackupError(
                        "cannot atomically overwrite a populated database connection; "
                        "use a filesystem path"
                    )
            source.backup(destination)
            destination_report = verify_database(destination)
        _assert_valid(destination_report, "destination")
        report = CopyReport(operation, source_report, destination_report)
        if not report.row_counts_match:
            raise BackupError(
                "row-count verification failed: source and destination differ"
            )
        return report
    finally:
        if close_destination and destination is not None:
            destination.close()
        if close_source:
            source.close()


def _publish_secondary_backup(
    primary: Path,
    secondary_directory: Path,
    *,
    age_recipient_path: Path | None,
) -> Path:
    """Atomically publish a secondary copy, encrypted when explicitly configured.

    Key creation and custody are deliberately outside Enfold. ``age`` accepts
    an age recipients file or an age identity file through ``-R``; operators
    provision that file separately and give this function only its path.
    """

    age = shutil.which("age")
    encrypted = age_recipient_path is not None
    if encrypted and age is None:
        raise BackupError(
            "age recipient/key path was configured but the age executable "
            "was not found; refusing plaintext secondary backup"
        )
    if not encrypted and age is None:
        _LOGGER.warning(
            "age executable not found; writing unencrypted secondary backup"
        )
    secondary_directory.mkdir(parents=True, exist_ok=True)
    name = f"{primary.name}.age" if encrypted else primary.name
    destination = secondary_directory / name
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".tmp", dir=secondary_directory
    )
    os.close(fd)
    temporary = Path(temporary_name)
    published = False
    try:
        if encrypted:
            temporary.unlink()
            subprocess.run(
                [
                    str(age),
                    "-R",
                    os.fspath(age_recipient_path),
                    "-o",
                    os.fspath(temporary),
                    os.fspath(primary),
                ],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            shutil.copyfile(primary, temporary)
        os.chmod(temporary, 0o600)
        _fsync_file(temporary)
        os.replace(temporary, destination)
        published = True
        _fsync_directory(secondary_directory)
        return destination
    finally:
        if not published:
            temporary.unlink(missing_ok=True)


def backup_database(
    source: Database,
    destination: Database,
    *,
    overwrite: bool = False,
    secondary_directory: str | os.PathLike[str] | None = None,
    age_recipient_path: str | os.PathLike[str] | None = None,
) -> CopyReport:
    """Create and verify an online SQLite backup and optional secondary copy.

    Secondary replication is best-effort by design: its outcome is reported,
    while failure never changes the successful primary backup result.
    """

    if age_recipient_path is not None and secondary_directory is None:
        raise BackupError(
            "age recipient/key path requires a secondary backup directory"
        )
    primary_report = _copy(source, destination, operation="backup", overwrite=overwrite)
    secondary_report: SecondaryCopyReport | None = None
    if secondary_directory is not None:
        primary = _destination_path(destination)
        recipient = (
            None
            if age_recipient_path is None
            else Path(age_recipient_path).expanduser()
        )
        try:
            if primary is None:
                raise BackupError(
                    "secondary backup requires a filesystem primary destination"
                )
            published = _publish_secondary_backup(
                primary.resolve(),
                Path(secondary_directory).expanduser(),
                age_recipient_path=recipient,
            )
            secondary_report = SecondaryCopyReport(
                status="succeeded",
                destination=str(published.resolve()),
                encrypted=recipient is not None,
                error=None,
            )
        except (BackupError, OSError, subprocess.SubprocessError) as exc:
            secondary_report = SecondaryCopyReport(
                status="failed",
                destination=None,
                encrypted=recipient is not None,
                error=str(exc),
            )
            _LOGGER.exception("secondary backup failed; primary backup remains valid")
    return CopyReport(
        primary_report.operation,
        primary_report.source,
        primary_report.destination,
        secondary_report,
    )


def restore_database(
    backup: Database, destination: Database, *, overwrite: bool = False
) -> CopyReport:
    """Restore and verify a SQLite backup using the backup API."""

    return _copy(backup, destination, operation="restore", overwrite=overwrite)


create_backup = backup_database
restore_backup = restore_database
