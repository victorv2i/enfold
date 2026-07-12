from __future__ import annotations

from dataclasses import FrozenInstanceError
import sqlite3

import pytest

from enfold.provenance import (
    ConnectionContext,
    WriteOutcome,
    WriteRequest,
    ensure_provenance_schema,
)


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        "CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def test_schema_is_additive_idempotent_and_has_composite_session_key():
    conn = _connection()
    ensure_provenance_schema(conn)
    ensure_provenance_schema(conn)

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {
        "memory_clients",
        "memory_sessions",
        "observations",
        "fact_provenance",
        "memory_write_log",
    }.issubset(tables)
    key_columns = {
        row[1]: row[5] for row in conn.execute("PRAGMA table_info(memory_sessions)")
    }
    assert key_columns["client_id"] > 0
    assert key_columns["session_id"] > 0


def test_schema_requires_existing_fact_store():
    with pytest.raises(RuntimeError, match="facts table"):
        ensure_provenance_schema(sqlite3.connect(":memory:"))


def test_schema_refuses_to_commit_a_callers_transaction():
    conn = _connection()
    conn.execute("INSERT INTO facts (content) VALUES ('uncommitted')")
    with pytest.raises(RuntimeError, match="idle connection"):
        ensure_provenance_schema(conn)
    assert conn.in_transaction is True
    conn.rollback()
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0


def test_schema_setup_rolls_back_all_ddl_on_failure():
    conn = _connection()
    conn.execute("CREATE TABLE fact_provenance (broken TEXT)")
    conn.commit()

    with pytest.raises(sqlite3.DatabaseError):
        ensure_provenance_schema(conn)

    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_clients'"
    ).fetchone() is None


def test_boundary_dataclasses_are_frozen_and_validate_inputs():
    context = ConnectionContext("client-a-1", "client-a", "client-a", "thread-1")
    request = WriteRequest("write-1", "A durable fact", "agent_report")
    outcome = WriteOutcome("uuid", "inserted", 1)

    with pytest.raises(FrozenInstanceError):
        context.agent_id = "client-b"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        request.content = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        outcome.fact_id = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="trust_score"):
        WriteRequest("write-2", "fact", "manual", trust_score=1.5)
    with pytest.raises(ValueError, match="access_scopes"):
        ConnectionContext("x", "client-a", "client-a", "thread", access_scopes=())
