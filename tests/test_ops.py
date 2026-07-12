import json
import sqlite3
import hashlib

import numpy as np
import pytest

from enfold.embeddings import embedding_to_bytes
from enfold.ops import _connect, main
from enfold.core_store import insert_fact
from enfold.schema import migrate
from enfold.rehearsal import create_legacy_fixture
from enfold.server import DatabaseOwnership


def _database(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO facts(content) VALUES ('shared memory')")
    conn.commit()
    conn.close()


def test_schema_status_is_read_only(tmp_path, capsys):
    database = tmp_path / "legacy.db"
    _database(database)

    assert main(["schema-status", str(database)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == 0
    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'schema_migrations'"
        ).fetchone() is None


def test_read_only_connect_percent_encodes_sqlite_uri_path(tmp_path):
    database = tmp_path / "live?tenant=1.sqlite"
    _database(database)

    with _connect(database, read_only=True) as conn:
        assert conn.execute("SELECT content FROM facts").fetchone() == (
            "shared memory",
        )
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO facts(content) VALUES ('must fail')")


def test_migrate_is_explicit_and_reports_versions(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)

    assert main(["migrate", str(database)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["schema_version_before"] == 0
    assert output["schema_version_after"] == 1


def test_migrate_under_hermes_requires_maintenance_override(tmp_path, capsys):
    database = tmp_path / ".hermes" / "memory.db"
    database.parent.mkdir()
    _database(database)

    assert main(["migrate", str(database)]) == 2

    error = capsys.readouterr().err
    assert "maintenance window" in error
    assert "--allow-live" in error
    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'schema_migrations'"
        ).fetchone() is None

    assert main(["migrate", str(database), "--allow-live"]) == 0


def test_backup_may_read_hermes_source_with_explicit_destination(tmp_path, capsys):
    source = tmp_path / ".hermes" / "memory.db"
    source.parent.mkdir()
    destination = tmp_path / "safe" / "memory-backup.db"
    _database(source)

    assert main(["backup", str(source), str(destination)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert destination.is_file()


def test_verify_reports_integrity_and_row_counts(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)

    assert main(["verify", str(database)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["report"]["integrity_check"] == ["ok"]
    assert output["report"]["row_counts"]["facts"] == 1


def test_live_verify_is_read_only_unless_explicit_fts_maintenance(tmp_path, capsys):
    database = tmp_path / ".hermes" / "memory.db"
    database.parent.mkdir()
    _database(database)

    assert main(["verify", str(database)]) == 0
    capsys.readouterr()

    assert main(["verify", str(database), "--check-fts"]) == 2
    assert "maintenance window" in capsys.readouterr().err

    assert main(
        ["verify", str(database), "--check-fts", "--allow-live"]
    ) == 0


def test_restore_under_hermes_requires_maintenance_override(tmp_path, capsys):
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    destination = tmp_path / ".hermes" / "restored.db"
    destination.parent.mkdir()
    _database(source)
    assert main(["backup", str(source), str(backup)]) == 0
    capsys.readouterr()

    assert main(["restore", str(backup), str(destination)]) == 2

    error = capsys.readouterr().err
    assert "maintenance window" in error
    assert "--allow-live" in error
    assert not destination.exists()

    assert main(
        ["restore", str(backup), str(destination), "--allow-live"]
    ) == 0
    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT content FROM facts").fetchone() == (
            "shared memory",
        )


def test_erase_fact_is_explicit_audited_maintenance(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)
    assert main(["migrate", str(database)]) == 0
    capsys.readouterr()

    assert main([
        "erase-fact", str(database), "1",
        "--requested-by", "victor",
        "--reason", "privacy request",
    ]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["report"]["fact_id"] == 1
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT content FROM facts").fetchone()[0] == (
            "[PRIVACY ERASED fact:1]"
        )
        assert conn.execute(
            "SELECT requested_by, reason FROM privacy_erasure_log"
        ).fetchone() == ("victor", "privacy request")


def test_rehearse_command_uses_only_explicit_offline_snapshot(tmp_path, capsys):
    snapshot = create_legacy_fixture(tmp_path / "snapshot.sqlite")
    workdir = tmp_path / "rehearsal"

    assert main(["rehearse", str(snapshot), str(workdir)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["report"]["migrated_schema_version"] == 1
    assert output["report"]["restored_schema_version"] == 0
    assert output["report"]["source_unchanged"] is True


def test_rebuild_vector_index_command_is_explicit_and_idempotent(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)
    assert main(["migrate", str(database)]) == 0
    capsys.readouterr()
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
            "VALUES (1, ?, 2, 'fixture')",
            (embedding_to_bytes(np.asarray((1.0, 0.0), dtype=np.float32)),),
        )
        conn.commit()

    command = [
        "rebuild-vector-index", str(database),
        "--embedding-identity", "fixture", "--dimensions", "2",
    ]
    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(command) == 0
    second = json.loads(capsys.readouterr().out)

    assert first["report"]["indexed_count"] == 1
    assert second["report"] == first["report"]


def test_mutating_maintenance_refuses_daemon_owned_database(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)
    ownership = DatabaseOwnership(database)
    ownership.acquire()
    try:
        assert main(["migrate", str(database)]) == 2
        assert "daemon owns" in capsys.readouterr().err
    finally:
        ownership.release()

    assert main(["migrate", str(database)]) == 0


def test_browse_snapshot_is_policy_filtered_read_only_and_idempotent(tmp_path, capsys):
    database = tmp_path / "memory.db"
    conn = sqlite3.connect(database)
    migrate(conn)
    visible = insert_fact(conn, "Visible browse fact", scope="private")
    insert_fact(conn, "Out of scope fact", scope="work")
    insert_fact(conn, "Sensitive browse fact", scope="private", sensitivity="sensitive")
    superseded = insert_fact(conn, "Superseded browse fact", scope="private")
    conflicted = insert_fact(conn, "Conflicted browse fact", scope="private")
    conn.execute("UPDATE facts SET superseded_by = ? WHERE fact_id = ?", (visible, superseded))
    conn.execute("UPDATE facts SET conflict_group = 'disputed' WHERE fact_id = ?", (conflicted,))
    conn.commit()
    conn.close()
    config = tmp_path / "server.json"
    config.write_text(json.dumps({
        "database_path": str(database),
        "socket_path": str(tmp_path / "enfold.sock"),
        "grants": {"browser": ["private"]},
        "browse_scopes": ["private"],
        "retrieval": {"mode": "ci", "allow_nonproduction": True, "dimensions": 64},
    }), encoding="utf-8")
    config.chmod(0o600)
    destination = tmp_path / "browse" / "browse-snapshot.db"

    assert main(["browse-snapshot", str(config), "--destination", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    first_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    with sqlite3.connect(f"file:{destination}?mode=ro", uri=True) as browse:
        assert browse.execute("SELECT content FROM facts").fetchall() == [("Visible browse fact",)]
        assert browse.execute("SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'visible'").fetchall() == [(visible,)]
    assert (destination.stat().st_mode & 0o222) == 0
    metadata = json.loads((destination.parent / "metadata.json").read_text())
    assert metadata["title"] == "Enfold Second Brain"
    assert metadata["scope_allowlist"] == ["private"]

    assert main(["browse-snapshot", str(config), "--destination", str(destination)]) == 0
    capsys.readouterr()
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == first_digest
