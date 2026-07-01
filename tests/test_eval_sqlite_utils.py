from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory_eval.sqlite_utils import backup_sqlite_db, connect_readonly, quick_check


def _make_wal_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO facts (content) VALUES (?)", ("fact living in sqlite",))
    conn.commit()
    conn.close()


def test_backup_sqlite_db_uses_consistent_backup_api_copy(tmp_path):
    src = tmp_path / "source.db"
    dst = tmp_path / "copy.db"
    _make_wal_db(src)

    result = backup_sqlite_db(src, dst)

    assert result.source == src
    assert result.destination == dst
    assert result.quick_check == "ok"
    assert result.bytes > 0
    with sqlite3.connect(dst) as conn:
        rows = conn.execute("SELECT content FROM facts").fetchall()
    assert rows == [("fact living in sqlite",)]


def test_backup_sqlite_db_refuses_to_overwrite_unless_requested(tmp_path):
    src = tmp_path / "source.db"
    dst = tmp_path / "copy.db"
    _make_wal_db(src)
    dst.write_text("not sqlite")

    with pytest.raises(FileExistsError):
        backup_sqlite_db(src, dst)

    result = backup_sqlite_db(src, dst, overwrite=True)
    assert result.quick_check == "ok"


def test_backup_sqlite_db_refuses_same_source_and_destination(tmp_path):
    src = tmp_path / "source.db"
    _make_wal_db(src)

    with pytest.raises(ValueError, match="source and destination"):
        backup_sqlite_db(src, src, overwrite=True)


def test_connect_readonly_does_not_create_missing_db(tmp_path):
    missing = tmp_path / "missing.db"

    with pytest.raises(sqlite3.OperationalError):
        connect_readonly(missing)

    assert not missing.exists()


def test_quick_check_reports_ok_for_valid_copy(tmp_path):
    src = tmp_path / "source.db"
    dst = tmp_path / "copy.db"
    _make_wal_db(src)
    backup_sqlite_db(src, dst)

    assert quick_check(dst) == "ok"
