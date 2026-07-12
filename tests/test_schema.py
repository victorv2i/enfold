import sqlite3

import pytest

from enfold.schema import (
    Migration,
    MigrationError,
    SchemaLedgerError,
    SchemaTooNewError,
    migrate,
    require_compatible_schema,
    schema_version,
)


def test_legacy_database_is_version_zero_and_read_is_non_mutating():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY)")

    assert schema_version(conn) == 0
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE name IN ('schema_migrations', 'enfold_meta')"
    ).fetchall() == []


def test_explicit_migration_creates_consistent_version_ledger():
    conn = sqlite3.connect(":memory:")

    assert migrate(conn) == 1
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "facts",
        "entities",
        "facts_fts",
        "memory_clients",
        "observations",
        "fact_provenance",
        "fact_conflicts",
    } <= tables
    assert schema_version(conn) == 1
    assert conn.execute("SELECT version FROM schema_migrations").fetchall() == [(1,)]
    assert conn.execute(
        "SELECT value FROM enfold_meta WHERE key = 'schema_version'"
    ).fetchone() == ("1",)
    assert migrate(conn) == 1


def test_newer_schema_fails_closed():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    conn.execute("INSERT INTO schema_migrations VALUES (2, 'later')")
    conn.execute("UPDATE enfold_meta SET value = '2' WHERE key = 'schema_version'")
    conn.commit()

    with pytest.raises(SchemaTooNewError):
        require_compatible_schema(conn)


def test_incomplete_or_disagreeing_ledger_fails_closed():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT)")
    with pytest.raises(SchemaLedgerError):
        schema_version(conn)


def test_v1_ledger_cannot_mask_a_partially_shaped_database():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        CREATE TABLE enfold_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_migrations VALUES (1, 'now');
        INSERT INTO enfold_meta VALUES ('schema_version', '1');
        """
    )

    with pytest.raises(SchemaLedgerError, match="required shape is incomplete"):
        schema_version(conn)


def test_v1_shape_rejects_reintroduced_content_uniqueness():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    conn.execute("CREATE UNIQUE INDEX legacy_content_unique ON facts(content)")
    conn.commit()

    with pytest.raises(SchemaLedgerError, match="content_unique"):
        schema_version(conn)


def test_failed_legacy_shape_upgrade_rolls_back_and_remains_v0():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hrr_vector BLOB
        );
        CREATE TABLE fact_provenance(broken TEXT);
        """
    )

    with pytest.raises(MigrationError):
        migrate(conn)

    assert schema_version(conn) == 0
    columns = {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
    assert "invalid_at" not in columns
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='memory_clients'"
    ).fetchone() is None


def test_current_legacy_fact_store_migrates_through_one_complete_v1_path():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hrr_vector BLOB
        );
        INSERT INTO facts(content, category, tags)
        VALUES ('legacy fact survives migration', 'project', 'legacy');
        """
    )

    assert migrate(conn) == 1

    assert schema_version(conn) == 1
    assert conn.execute("SELECT content FROM facts").fetchone() == (
        "legacy fact survives migration",
    )
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' "
        "AND name='idx_facts_active_scope'"
    ).fetchone() == (1,)


def test_legacy_unique_content_is_rebuilt_for_value_flip_flops():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hrr_vector BLOB
        );
        CREATE TABLE legacy_links (
            link_id INTEGER PRIMARY KEY,
            fact_id INTEGER NOT NULL REFERENCES facts(fact_id)
        );
        INSERT INTO facts(content, category, tags, trust_score)
        VALUES ('X', 'project', 'first', 0.8), ('Y', 'general', 'second', 0.4);
        INSERT INTO legacy_links(fact_id) VALUES (1), (2);
        """
    )

    assert migrate(conn) == 1
    third_id = conn.execute("INSERT INTO facts(content) VALUES ('X')").lastrowid
    conn.commit()

    assert third_id == 3
    assert conn.execute(
        "SELECT fact_id, content FROM facts ORDER BY fact_id"
    ).fetchall() == [(1, "X"), (2, "Y"), (3, "X")]
    assert conn.execute(
        "SELECT category, tags, trust_score FROM facts WHERE fact_id=1"
    ).fetchone() == ("project", "first", 0.8)
    assert conn.execute(
        "SELECT fact_id FROM legacy_links ORDER BY link_id"
    ).fetchall() == [(1,), (2,)]
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("PRAGMA foreign_keys").fetchone() == (1,)
    assert conn.execute(
        "SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'X' ORDER BY rowid"
    ).fetchall() == [(1,), (3,)]


def test_real_legacy_trust_index_is_verified_and_recreated():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            content         TEXT NOT NULL UNIQUE,
            category        TEXT DEFAULT 'general',
            tags            TEXT DEFAULT '',
            trust_score     REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count   INTEGER DEFAULT 0,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hrr_vector      BLOB,
            valid_from      TIMESTAMP,
            invalid_at      TIMESTAMP,
            superseded_by   INTEGER
        );
        CREATE INDEX idx_facts_category ON facts(category);
        CREATE INDEX idx_facts_trust    ON facts(trust_score DESC);
        INSERT INTO facts(content, category, trust_score)
        VALUES ('high trust survives', 'project', 0.9),
               ('low trust survives', 'general', 0.2);
        """
    )

    assert migrate(conn) == 1

    assert conn.execute(
        "SELECT fact_id, content, trust_score FROM facts ORDER BY fact_id"
    ).fetchall() == [
        (1, "high trust survives", 0.9),
        (2, "low trust survives", 0.2),
    ]
    index_row = next(
        row for row in conn.execute("PRAGMA index_list(facts)")
        if row[1] == "idx_facts_trust"
    )
    assert (index_row[2], index_row[3], index_row[4]) == (0, "c", 0)
    assert [
        (row[2], row[3], row[4])
        for row in conn.execute('PRAGMA index_xinfo("idx_facts_trust")')
        if row[5]
    ] == [("trust_score", 1, "BINARY")]


def test_malformed_legacy_trust_index_fails_closed_and_rolls_back():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY,
            content TEXT NOT NULL UNIQUE,
            trust_score REAL DEFAULT 0.5
        );
        CREATE INDEX idx_facts_trust ON facts(trust_score ASC);
        INSERT INTO facts VALUES (7, 'preserve me', 0.8);
        """
    )

    with pytest.raises(MigrationError, match="index:idx_facts_trust"):
        migrate(conn)

    assert schema_version(conn) == 0
    assert conn.execute("SELECT * FROM facts").fetchall() == [
        (7, "preserve me", 0.8)
    ]
    assert conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' "
        "AND name='idx_facts_trust'"
    ).fetchone() == (
        "CREATE INDEX idx_facts_trust ON facts(trust_score ASC)",
    )
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='facts_v1_rebuild'"
    ).fetchone() is None


def test_unique_rebuild_with_unknown_fact_columns_fails_closed_and_rolls_back():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY,
            content TEXT NOT NULL UNIQUE,
            extension_payload TEXT
        );
        INSERT INTO facts VALUES (7, 'preserve me', 'extension data');
        """
    )

    with pytest.raises(MigrationError, match="unknown columns"):
        migrate(conn)

    assert schema_version(conn) == 0
    assert conn.execute("SELECT * FROM facts").fetchall() == [
        (7, "preserve me", "extension data")
    ]
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='facts_v1_rebuild'"
    ).fetchone() is None


def test_migration_rolls_back_schema_and_ledger_together():
    conn = sqlite3.connect(":memory:")

    def broken(c):
        c.execute("CREATE TABLE should_rollback (id INTEGER)")
        raise ValueError("boom")

    with pytest.raises(MigrationError):
        migrate(conn, migrations={1: Migration(1, "broken", broken)})

    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE name IN "
        "('should_rollback', 'schema_migrations', 'enfold_meta')"
    ).fetchall() == []


def test_migration_refuses_to_own_callers_transaction():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE caller_data(id INTEGER)")
    conn.execute("INSERT INTO caller_data VALUES (1)")

    with pytest.raises(MigrationError, match="existing transaction"):
        migrate(conn)
    assert conn.execute("SELECT COUNT(*) FROM caller_data").fetchone() == (1,)


def test_migration_rejects_incompatible_nullable_legacy_queue_and_rolls_back():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT NOT NULL);
        CREATE TABLE extract_queue(
            id INTEGER PRIMARY KEY,
            payload TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
        );
        INSERT INTO extract_queue(id, payload) VALUES (1, 'legacy transcript');
        """
    )

    with pytest.raises(MigrationError, match="payload must be NOT NULL"):
        migrate(conn)

    assert schema_version(conn) == 0
    assert conn.execute("SELECT payload FROM extract_queue").fetchone() == (
        "legacy transcript",
    )
