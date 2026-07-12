from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sqlite3

import pytest

from enfold.backup import backup_database
from enfold.backup_rehearsal import RehearsalError, rehearse_latest_backup
from enfold.schema import migrate
from enfold.server import _open_existing_v1, load_config


def _database(path: Path, *, facts: int = 2) -> Path:
    with sqlite3.connect(path) as conn:
        migrate(conn)
        for index in range(facts):
            conn.execute(
                "INSERT INTO facts(content, scope) VALUES (?, 'private')",
                (f"durable fact {index}",),
            )
    return path


def test_secondary_backup_uses_age_when_available(tmp_path, monkeypatch):
    source = _database(tmp_path / "live.sqlite")
    primary = tmp_path / "primary" / "memory-20260712.sqlite"
    secondary = tmp_path / "offsite"
    recipient = tmp_path / "recipient.txt"
    recipient.write_text("age1testrecipient\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    age = bin_dir / "age"
    age.write_text(
        "#!/bin/sh\n"
        "test \"$1\" = -R\n"
        "test \"$2\" = \"$AGE_EXPECTED_RECIPIENT\"\n"
        "test \"$3\" = -o\n"
        "/bin/cp \"$5\" \"$4\"\n",
        encoding="utf-8",
    )
    age.chmod(0o700)
    monkeypatch.setenv("PATH", os.fspath(bin_dir))
    monkeypatch.setenv("AGE_EXPECTED_RECIPIENT", os.fspath(recipient))

    report = backup_database(
        source,
        primary,
        secondary_directory=secondary,
        age_recipient_path=recipient,
    )

    assert report.ok
    assert report.secondary is not None
    assert report.secondary.status == "succeeded"
    assert report.secondary.encrypted is True
    assert report.secondary.error is None
    encrypted = secondary / f"{primary.name}.age"
    assert encrypted.read_bytes() == primary.read_bytes()
    assert not (secondary / primary.name).exists()


def test_secondary_backup_plain_copy_when_age_is_absent(
    tmp_path, monkeypatch, caplog
):
    source = _database(tmp_path / "live.sqlite")
    primary = tmp_path / "primary.sqlite"
    secondary = tmp_path / "offsite"
    monkeypatch.setenv("PATH", os.fspath(tmp_path / "empty-bin"))

    with caplog.at_level(logging.WARNING, logger="enfold.backup"):
        report = backup_database(
            source, primary, secondary_directory=secondary
        )

    assert report.ok
    assert report.secondary is not None
    assert report.secondary.status == "succeeded"
    assert report.secondary.encrypted is False
    assert (secondary / primary.name).read_bytes() == primary.read_bytes()
    assert "age executable not found" in caplog.text


def test_secondary_backup_without_recipient_stays_plain_when_age_is_available(
    tmp_path, monkeypatch
):
    source = _database(tmp_path / "live.sqlite")
    primary = tmp_path / "primary.sqlite"
    secondary = tmp_path / "offsite"
    monkeypatch.setattr("enfold.backup.shutil.which", lambda _name: "/fake/age")

    report = backup_database(source, primary, secondary_directory=secondary)

    assert report.ok
    assert report.secondary is not None
    assert report.secondary.status == "succeeded"
    assert report.secondary.encrypted is False
    assert (secondary / primary.name).read_bytes() == primary.read_bytes()
    assert not (secondary / f"{primary.name}.age").exists()


def test_secondary_failure_does_not_fail_primary(tmp_path, caplog):
    source = _database(tmp_path / "live.sqlite")
    primary = tmp_path / "primary.sqlite"
    secondary_file = tmp_path / "not-a-directory"
    secondary_file.write_text("occupied", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="enfold.backup"):
        report = backup_database(
            source, primary, secondary_directory=secondary_file
        )

    assert report.ok
    assert report.secondary is not None
    assert report.secondary.status == "failed"
    assert report.secondary.error
    assert primary.is_file()
    assert "secondary backup failed" in caplog.text


def test_configured_recipient_without_age_fails_closed_in_report(
    tmp_path, monkeypatch
):
    source = _database(tmp_path / "live.sqlite")
    primary = tmp_path / "primary.sqlite"
    secondary = tmp_path / "offsite"
    recipient = tmp_path / "recipient.txt"
    recipient.write_text("age1testrecipient\n", encoding="utf-8")
    monkeypatch.setenv("PATH", os.fspath(tmp_path / "empty-bin"))

    report = backup_database(
        source,
        primary,
        secondary_directory=secondary,
        age_recipient_path=recipient,
    )

    assert report.ok
    assert primary.is_file()
    assert not secondary.exists() or list(secondary.iterdir()) == []
    assert report.secondary is not None
    assert report.secondary.status == "failed"
    assert report.secondary.encrypted is True
    assert "age executable" in report.secondary.error


def test_restore_rehearsal_percent_encodes_live_database_uri(tmp_path):
    live = _database(tmp_path / "live?tenant=1.sqlite")
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    backup = _database(backup_dir / "memory-20260712.sqlite")
    state_dir = tmp_path / "state"

    report = rehearse_latest_backup(live, backup_dir, state_dir)

    assert report.status == "passed"
    assert report.backup == str(backup.resolve())
    assert report.live_fact_count == report.restored_fact_count == 2


def test_restore_rehearsal_detects_bit_flipped_backup_and_writes_report(tmp_path):
    live = _database(tmp_path / "live.sqlite")
    backup_dir = tmp_path / "backups"
    damaged = backup_dir / "memory-20260712.sqlite"
    backup_database(live, damaged)
    content = bytearray(damaged.read_bytes())
    content[100] ^= 0xFF
    damaged.write_bytes(content)
    state_dir = tmp_path / "state"

    with pytest.raises(RehearsalError, match="failed"):
        rehearse_latest_backup(live, backup_dir, state_dir)

    reports = list(state_dir.glob("restore-rehearsal-*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["backup"] == str(damaged.resolve())
    assert payload["error"]


def test_restore_rehearsal_uses_newest_backup_and_fact_count_tolerance(tmp_path):
    live = _database(tmp_path / "live.sqlite", facts=3)
    backup_dir = tmp_path / "backups"
    older = backup_dir / "memory-older.sqlite"
    newer = backup_dir / "memory-newer.sqlite"
    backup_database(live, older)
    with sqlite3.connect(live) as conn:
        conn.execute(
            "INSERT INTO facts(content, scope) VALUES ('latest fact', 'private')"
        )
    backup_database(live, newer)
    os.utime(older, (1, 1))

    with sqlite3.connect(live) as conn:
        conn.execute(
            "INSERT INTO facts(content, scope) VALUES ('post-backup fact', 'private')"
        )
    report = rehearse_latest_backup(
        live, backup_dir, tmp_path / "state", fact_count_tolerance=1
    )

    assert report.status == "passed"
    assert report.backup == str(newer.resolve())
    assert report.quick_check == ("ok",)
    assert report.live_fact_count == 5
    assert report.restored_fact_count == 4


def test_synchronous_full_config_is_applied_to_write_connection(tmp_path):
    database = _database(tmp_path / "memory.sqlite")
    config_path = tmp_path / "server.json"
    config_path.write_text(
        json.dumps(
            {
                "database_path": str(database),
                "socket_path": str(tmp_path / "enfold.sock"),
                "grants": {"test": ["private"]},
                "retrieval": {
                    "mode": "ci",
                    "allow_nonproduction": True,
                    "dimensions": 64,
                },
                "synchronous_full": True,
            }
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    config = load_config(config_path)
    connection = _open_existing_v1(config)
    try:
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    finally:
        connection.close()


def test_synchronous_default_remains_normal(tmp_path):
    database = _database(tmp_path / "memory.sqlite")
    config_path = tmp_path / "server.json"
    config_path.write_text(
        json.dumps(
            {
                "database_path": str(database),
                "socket_path": str(tmp_path / "enfold.sock"),
                "grants": {"test": ["private"]},
                "retrieval": {
                    "mode": "ci",
                    "allow_nonproduction": True,
                    "dimensions": 64,
                },
            }
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    connection = _open_existing_v1(load_config(config_path))
    try:
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1
    finally:
        connection.close()
