from __future__ import annotations

import sqlite3

import pytest

from enfold.rehearsal import (
    RehearsalError,
    create_legacy_fixture,
    database_fingerprint,
    prove_unknown_object_fails_closed,
    rehearse_snapshot,
    run_rehearsal,
)


def test_full_synthetic_backup_migrate_smoke_restore_rehearsal(tmp_path):
    report = run_rehearsal(tmp_path / "rehearsal")

    assert report.migrated_schema_version == 1
    assert report.restored_schema_version == 0
    assert report.migrated_fact_count == 5
    assert report.restored_fact_count == 3
    assert report.current_fact_id in report.current_search_fact_ids
    assert report.legacy_fingerprint == report.backup_fingerprint
    assert report.legacy_fingerprint == report.restored_fingerprint
    assert report.rollback_artifact_verified is True
    assert report.restored_legacy_verified is True

    # The destructive restore really removed v1 and reopened the old shape.
    with sqlite3.connect(tmp_path / "rehearsal" / "synthetic-legacy.sqlite") as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
        assert "invalid_at" not in columns
        assert conn.execute("SELECT count(*) FROM embeddings").fetchone() == (2,)
        assert conn.execute("SELECT count(*) FROM extract_queue").fetchone() == (2,)
        assert conn.execute("SELECT count(*) FROM reflection_sources").fetchone() == (2,)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_unknown_legacy_facts_object_fails_closed_without_data_loss(tmp_path):
    assert prove_unknown_object_fails_closed(tmp_path / "unknown.sqlite") is True


def test_rehearsal_refuses_any_path_below_dot_hermes(tmp_path):
    with pytest.raises(RehearsalError, match="must not be below .hermes"):
        run_rehearsal(tmp_path / ".hermes" / "never-touch")


def test_explicit_offline_snapshot_rehearsal_never_modifies_source(tmp_path):
    snapshot = create_legacy_fixture(tmp_path / "copied-realistic-snapshot.sqlite")
    before = database_fingerprint(snapshot)

    report = rehearse_snapshot(snapshot, tmp_path / "snapshot-rehearsal")

    assert report.source_fingerprint == before
    assert report.rollback_fingerprint == before
    assert report.restored_fingerprint == before
    assert report.migrated_schema_version == 1
    assert report.restored_schema_version == 0
    assert report.smoke_search_verified is True
    assert report.smoke_evidence_verified is True
    assert report.source_unchanged is True
    assert database_fingerprint(snapshot) == before


def test_snapshot_rehearsal_refuses_live_or_reused_workdir(tmp_path):
    live = tmp_path / ".hermes" / "snapshot.sqlite"
    live.parent.mkdir()
    live.write_bytes(b"not opened")
    with pytest.raises(RehearsalError, match="must not be below .hermes"):
        rehearse_snapshot(live, tmp_path / "work")

    snapshot = create_legacy_fixture(tmp_path / "snapshot.sqlite")
    workdir = tmp_path / "used"
    rehearse_snapshot(snapshot, workdir)
    with pytest.raises(RehearsalError, match="already contains"):
        rehearse_snapshot(snapshot, workdir)


def test_snapshot_rehearsal_refuses_source_as_workdir(tmp_path):
    snapshot = create_legacy_fixture(tmp_path / "snapshot.sqlite")

    with pytest.raises(RehearsalError, match="separate from the source"):
        rehearse_snapshot(snapshot, snapshot)


def test_failed_snapshot_migration_leaves_source_unchanged_and_unlocks(
    tmp_path, monkeypatch
):
    snapshot = create_legacy_fixture(tmp_path / "snapshot.sqlite")
    before = database_fingerprint(snapshot)

    def fail_migration(_conn):
        raise sqlite3.OperationalError("synthetic migration failure")

    monkeypatch.setattr("enfold.rehearsal.migrate", fail_migration)
    with pytest.raises(sqlite3.OperationalError, match="synthetic migration failure"):
        rehearse_snapshot(snapshot, tmp_path / "failed-rehearsal")

    assert database_fingerprint(snapshot) == before
    # A prior failure must not strand either maintenance lock.
    from enfold.backup import maintenance_database_lock

    with maintenance_database_lock(snapshot):
        pass
