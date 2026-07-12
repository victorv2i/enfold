import fcntl
import os
import sqlite3
from pathlib import Path

import pytest

from enfold.backup import (
    BackupError,
    backup_database,
    maintenance_database_lock,
    restore_database,
    verify_database,
)


def _make_database(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE parent(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE child(
            id INTEGER PRIMARY KEY,
            parent_id INTEGER NOT NULL REFERENCES parent(id)
        );
        CREATE VIRTUAL TABLE notes_fts USING fts5(content);
        INSERT INTO parent(value) VALUES ('one'), ('two');
        INSERT INTO child(parent_id) VALUES (1);
        INSERT INTO notes_fts(content) VALUES ('memory with receipts');
        """
    )
    conn.commit()
    conn.close()


def test_backup_uses_verified_snapshot_with_row_count_evidence(tmp_path):
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    _make_database(source)

    report = backup_database(source, backup)

    assert report.ok
    assert report.operation == "backup"
    assert report.row_counts_match
    assert report.destination.row_counts["parent"] == 2
    assert report.destination.row_counts["notes_fts"] == 1
    assert report.destination.fts_tables_checked == ("notes_fts",)


def test_restore_round_trip_is_verified(tmp_path):
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    restored = tmp_path / "restored.db"
    _make_database(source)
    backup_database(source, backup)

    report = restore_database(backup, restored)

    assert report.ok
    with sqlite3.connect(restored) as conn:
        assert conn.execute("SELECT value FROM parent ORDER BY id").fetchall() == [
            ("one",),
            ("two",),
        ]


def test_backup_refuses_to_overwrite_without_opt_in(tmp_path):
    source = tmp_path / "source.db"
    destination = tmp_path / "existing.db"
    _make_database(source)
    _make_database(destination)

    with pytest.raises(BackupError, match="already exists"):
        backup_database(source, destination)


def test_foreign_key_check_is_part_of_verification(tmp_path):
    path = tmp_path / "invalid.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE parent(id INTEGER PRIMARY KEY);
        CREATE TABLE child(parent_id INTEGER REFERENCES parent(id));
        INSERT INTO child VALUES (99);
        """
    )
    conn.commit()

    report = verify_database(conn)

    assert not report.ok
    assert report.foreign_key_violations
    conn.close()


def test_connection_with_active_transaction_is_rejected(tmp_path):
    destination = tmp_path / "backup.db"
    source = sqlite3.connect(":memory:")
    source.execute("CREATE TABLE facts(id INTEGER)")
    source.execute("INSERT INTO facts VALUES (1)")

    with pytest.raises(BackupError, match="active transaction"):
        backup_database(source, destination)


def test_same_path_is_rejected_even_with_overwrite(tmp_path):
    path = tmp_path / "memory.db"
    _make_database(path)

    with pytest.raises(BackupError, match="different databases"):
        backup_database(path, path, overwrite=True)


def test_path_source_can_be_read_only(tmp_path):
    source = tmp_path / "readonly.db"
    destination = tmp_path / "backup.db"
    _make_database(source)
    source.chmod(0o444)

    report = backup_database(source, destination)

    assert report.ok
    assert report.destination.fts_tables_checked == ("notes_fts",)


def test_backup_percent_encodes_sqlite_uri_path(tmp_path):
    source = tmp_path / "live?tenant=1.sqlite"
    destination = tmp_path / "backup.sqlite"
    _make_database(source)

    report = backup_database(source, destination)

    assert report.ok
    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT COUNT(*) FROM parent").fetchone() == (2,)


@pytest.mark.parametrize("copy", [backup_database, restore_database])
def test_failed_overwrite_preserves_existing_destination(tmp_path, monkeypatch, copy):
    source = tmp_path / "source.db"
    destination = tmp_path / "existing.db"
    _make_database(source)
    _make_database(destination)
    with sqlite3.connect(destination) as conn:
        conn.execute("UPDATE parent SET value = 'original' WHERE id = 1")
        conn.commit()

    def fail_before_publish(_path):
        raise OSError("simulated sync failure")

    monkeypatch.setattr("enfold.backup._fsync_file", fail_before_publish)

    with pytest.raises(OSError, match="simulated sync failure"):
        copy(source, destination, overwrite=True)

    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT value FROM parent WHERE id = 1").fetchone() == (
            "original",
        )
    assert not list(tmp_path.glob(".existing.db.*.tmp"))


def test_overwrite_stages_snapshot_on_same_filesystem(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    destination = tmp_path / "nested" / "existing.db"
    destination.parent.mkdir()
    _make_database(source)
    _make_database(destination)
    replacements = []
    real_replace = __import__("os").replace

    def record_replace(staged, published):
        replacements.append((Path(staged), Path(published)))
        real_replace(staged, published)

    monkeypatch.setattr("enfold.backup.os.replace", record_replace)

    report = backup_database(source, destination, overwrite=True)

    assert report.ok
    assert len(replacements) == 1
    staged, published = replacements[0]
    assert staged.parent == destination.parent
    assert published == destination


def test_overwrite_refuses_destination_with_wal_sidecars(tmp_path):
    source = tmp_path / "source.db"
    destination = tmp_path / "existing.db"
    _make_database(source)
    _make_database(destination)
    wal = Path(f"{destination}-wal")
    wal.touch()

    with pytest.raises(BackupError, match="WAL sidecars"):
        backup_database(source, destination, overwrite=True)

    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT COUNT(*) FROM parent").fetchone() == (2,)


def test_backup_includes_committed_rows_still_in_source_wal(tmp_path):
    source = tmp_path / "source.db"
    destination = tmp_path / "backup.db"
    writer = sqlite3.connect(source)
    writer.execute("PRAGMA journal_mode = WAL")
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    writer.execute("CREATE TABLE facts(value TEXT NOT NULL)")
    writer.execute("INSERT INTO facts VALUES ('committed in wal')")
    writer.commit()
    assert Path(f"{source}-wal").exists()

    try:
        report = backup_database(source, destination)
    finally:
        writer.close()

    assert report.ok
    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT value FROM facts").fetchall() == [
            ("committed in wal",)
        ]


def test_populated_connection_cannot_be_atomically_overwritten(tmp_path):
    source = tmp_path / "source.db"
    _make_database(source)
    destination = sqlite3.connect(":memory:")
    destination.execute("CREATE TABLE sentinel(value TEXT)")
    destination.execute("INSERT INTO sentinel VALUES ('keep me')")
    destination.commit()

    try:
        with pytest.raises(BackupError, match="cannot atomically overwrite"):
            backup_database(source, destination, overwrite=True)
        assert destination.execute("SELECT value FROM sentinel").fetchone() == (
            "keep me",
        )
    finally:
        destination.close()


def test_maintenance_lock_refuses_legacy_writer_and_releases_daemon_lock(tmp_path):
    database = tmp_path / "memory.db"
    _make_database(database)
    legacy_sidecar = Path(f"{database.resolve()}.mcp-write.lock")
    legacy_fd = os.open(legacy_sidecar, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(legacy_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(BackupError, match="legacy writer owns"):
            with maintenance_database_lock(database, timeout=0.01):
                pytest.fail("maintenance entered while legacy writer held its lock")
    finally:
        fcntl.flock(legacy_fd, fcntl.LOCK_UN)
        os.close(legacy_fd)

    # Failure on the second lock must roll back the first acquisition.
    daemon_sidecar = Path(f"{database.resolve()}.enfold.lock")
    daemon_fd = os.open(daemon_sidecar, os.O_RDWR)
    try:
        fcntl.flock(daemon_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(daemon_fd, fcntl.LOCK_UN)
        os.close(daemon_fd)


def test_maintenance_lock_is_held_for_entire_critical_section(tmp_path):
    database = tmp_path / "memory.db"
    _make_database(database)

    with maintenance_database_lock(database):
        for suffix in (".enfold.lock", ".mcp-write.lock"):
            fd = os.open(Path(f"{database.resolve()}{suffix}"), os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
