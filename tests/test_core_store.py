import sqlite3

import pytest

from enfold.core_store import (
    active_facts,
    connect_database,
    ensure_core_schema,
    get_active_fact,
    insert_fact,
    link_fact_entities,
    resolve_entity,
    search_fts,
)


def _store(tmp_path):
    conn = connect_database(tmp_path / "memory.db")
    ensure_core_schema(conn)
    conn.commit()
    return conn


def test_connection_factory_applies_sqlite_operational_pragmas(tmp_path):
    conn = connect_database(tmp_path / "memory.db", busy_timeout_ms=4321)

    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 4321
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
    assert conn.row_factory is sqlite3.Row
    conn.close()


def test_owned_schema_has_current_fact_entity_and_fts_shape(tmp_path):
    conn = _store(tmp_path)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
    assert {
        "content",
        "hrr_vector",
        "invalid_at",
        "superseded_by",
        "memory_kind",
        "subject_key",
        "predicate_key",
        "scope",
        "conflict_group",
    } <= columns
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    assert {"facts", "entities", "fact_entities", "facts_fts"} <= tables
    conn.close()


def test_insert_fact_never_commits_and_fts_follows_transaction(tmp_path):
    conn = _store(tmp_path)

    fact_id = insert_fact(conn, "Client A changed the project architecture")
    assert conn.in_transaction
    assert search_fts(conn, "architecture")[0]["fact_id"] == fact_id
    conn.rollback()

    assert active_facts(conn) == []
    assert search_fts(conn, "architecture") == []
    conn.close()


def test_fresh_store_preserves_repeated_historical_content(tmp_path):
    """State can legitimately change X -> Y -> X without losing history."""

    conn = _store(tmp_path)
    first = insert_fact(conn, "Victor prefers the compact layout")
    middle = insert_fact(conn, "Victor prefers the spacious layout")
    latest = insert_fact(conn, "Victor prefers the compact layout")
    conn.execute(
        "UPDATE facts SET superseded_by = ? WHERE fact_id = ?", (middle, first)
    )
    conn.execute(
        "UPDATE facts SET superseded_by = ? WHERE fact_id = ?", (latest, middle)
    )
    conn.commit()

    rows = conn.execute(
        "SELECT fact_id, content FROM facts ORDER BY fact_id"
    ).fetchall()
    assert [row["fact_id"] for row in rows] == [first, middle, latest]
    assert [row["fact_id"] for row in active_facts(conn)] == [latest]
    conn.close()


def test_active_reads_exclude_invalid_and_superseded_facts(tmp_path):
    conn = _store(tmp_path)
    current = insert_fact(conn, "Current project state")
    invalid = insert_fact(conn, "Old invalid project state")
    superseded = insert_fact(conn, "Old superseded project state")
    conn.execute(
        "UPDATE facts SET invalid_at = CURRENT_TIMESTAMP WHERE fact_id = ?", (invalid,)
    )
    conn.execute(
        "UPDATE facts SET superseded_by = ? WHERE fact_id = ?", (current, superseded)
    )
    conn.commit()

    assert [row["fact_id"] for row in active_facts(conn)] == [current]
    assert get_active_fact(conn, invalid) is None
    assert [row["fact_id"] for row in search_fts(conn, "project")] == [current]
    conn.close()


def test_scope_predicate_is_enforced_before_active_and_fts_results(tmp_path):
    conn = _store(tmp_path)
    private = insert_fact(conn, "Victor private planning", scope="private")
    work = insert_fact(conn, "Victor work planning", scope="work")
    conn.commit()

    assert [row["fact_id"] for row in active_facts(conn, allowed_scopes=("work",))] == [work]
    assert [row["fact_id"] for row in search_fts(conn, "planning", allowed_scopes=("private",))] == [private]
    assert active_facts(conn, allowed_scopes=()) == []
    assert search_fts(conn, "planning", allowed_scopes=()) == []
    assert [row["fact_id"] for row in active_facts(conn)] == [private]
    conn.close()


def test_unresolved_conflicts_are_not_returned_as_settled_truth(tmp_path):
    conn = _store(tmp_path)
    settled = insert_fact(conn, "Victor uses the compact layout")
    disputed = insert_fact(conn, "Victor uses the spacious layout")
    conn.execute(
        "UPDATE facts SET conflict_group = 'layout-conflict' WHERE fact_id = ?",
        (disputed,),
    )
    conn.commit()

    assert [row["fact_id"] for row in active_facts(conn)] == [settled]
    assert get_active_fact(conn, disputed) is None
    assert search_fts(conn, "spacious") == []
    conn.close()


def test_entity_resolution_and_links_share_callers_transaction(tmp_path):
    conn = _store(tmp_path)
    fact_id = insert_fact(conn, "Victor uses Enfold")
    victor = resolve_entity(conn, "Victor", entity_type="person")
    assert resolve_entity(conn, "victor") == victor
    link_fact_entities(conn, fact_id, (victor,))
    conn.rollback()

    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM fact_entities").fetchone()[0] == 0
    conn.close()


def test_legacy_schema_reads_as_private_and_rejects_scope_loss():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE facts(
            fact_id INTEGER PRIMARY KEY,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5
        );
        INSERT INTO facts(content) VALUES ('legacy shared memory');
        """
    )

    assert len(active_facts(conn, allowed_scopes=("private",))) == 1
    assert active_facts(conn, allowed_scopes=("work",)) == []
    with pytest.raises(ValueError, match="cannot persist"):
        insert_fact(conn, "scoped legacy fact", scope="work")


def test_schema_initializer_preserves_existing_legacy_facts_table(tmp_path):
    conn = sqlite3.connect(tmp_path / "legacy.db")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE facts(
            fact_id INTEGER PRIMARY KEY,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5
        )
        """
    )
    conn.execute(
        "INSERT INTO facts(content, tags) VALUES ('fact before FTS migration', '')"
    )
    conn.commit()

    ensure_core_schema(conn)
    fact_id = insert_fact(conn, "legacy-compatible fact")
    conn.commit()

    assert search_fts(conn, "compatible")[0]["fact_id"] == fact_id
    assert search_fts(conn, "migration")[0]["content"] == "fact before FTS migration"
    columns = {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
    assert "scope" not in columns
    conn.close()
