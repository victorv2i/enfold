"""Standalone SQLite fact storage for Enfold.

The schema and FTS trigger design descend from the MIT-licensed holographic
memory foundation by dusterbloom for NousResearch/hermes-agent.  See the
repository ``NOTICE`` for full attribution.  This module is an Enfold-owned
implementation: it imports no Hermes modules and can be used by the daemon,
MCP bridge, or maintenance tools directly.

Schema creation and fact mutations never commit.  Transaction boundaries
belong to the caller so provenance, temporal changes, and fact insertion can
be made atomic by a higher-level write service.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


DEFAULT_BUSY_TIMEOUT_MS = 10_000
DEFAULT_SCOPE = "private"


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS facts (
        fact_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        content           TEXT NOT NULL,
        category          TEXT DEFAULT 'general',
        tags              TEXT DEFAULT '',
        trust_score       REAL DEFAULT 0.5,
        retrieval_count   INTEGER DEFAULT 0,
        helpful_count     INTEGER DEFAULT 0,
        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        hrr_vector        BLOB,
        valid_from        TIMESTAMP,
        invalid_at        TIMESTAMP,
        superseded_by     INTEGER,
        memory_kind       TEXT,
        subject_key       TEXT,
        predicate_key     TEXT,
        object_value      TEXT,
        object_entity_id  INTEGER,
        confidence        REAL,
        source_authority  REAL,
        scope             TEXT NOT NULL DEFAULT 'private',
        sensitivity       TEXT NOT NULL DEFAULT 'normal',
        correction_status TEXT,
        schema_version    INTEGER,
        conflict_group    TEXT,
        FOREIGN KEY (superseded_by) REFERENCES facts(fact_id),
        FOREIGN KEY (object_entity_id) REFERENCES entities(entity_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entities (
        entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        entity_type TEXT DEFAULT 'unknown',
        aliases     TEXT DEFAULT '',
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fact_entities (
        fact_id   INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
        entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
        PRIMARY KEY (fact_id, entity_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
    "CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)",
    "CREATE INDEX IF NOT EXISTS idx_facts_active_scope ON facts(invalid_at, scope)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id)
    """,
    """
    CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
        INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
        INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF content, tags ON facts BEGIN
        INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
        INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
    END
    """,
)


def connect_database(
    path: str | Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    """Open a configured standalone Enfold SQLite connection.

    This does not create tables.  Call :func:`ensure_core_schema` explicitly
    for a new store or run the migration layer for an existing store.
    """

    if busy_timeout_ms < 0:
        raise ValueError("busy_timeout_ms must be non-negative")
    conn = sqlite3.connect(
        str(Path(path).expanduser()),
        check_same_thread=check_same_thread,
        timeout=busy_timeout_ms / 1000,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_core_schema(conn: sqlite3.Connection) -> None:
    """Create the owned fact/entity/FTS schema without committing.

    ``IF NOT EXISTS`` preserves the current legacy holographic tables.  This
    function intentionally does not add columns to an existing ``facts``
    table; additive upgrades belong to explicit migrations.
    """

    had_fts = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'facts_fts'"
    ).fetchone() is not None

    # entities must exist before facts because fresh facts declares an entity
    # foreign key.  SQLite permits the reverse, but deterministic ordering is
    # friendlier to schema inspection tools.
    order = (1, 0, *range(2, len(_SCHEMA_STATEMENTS)))
    for index in order:
        if index == 5:
            columns = _fact_columns(conn)
            if not {"invalid_at", "scope"}.issubset(columns):
                continue
        conn.execute(_SCHEMA_STATEMENTS[index])
    if not had_fts:
        # External-content FTS tables do not automatically index rows that
        # predate their creation.  This remains inside the caller transaction.
        conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")


def _fact_columns(conn: sqlite3.Connection) -> frozenset[str]:
    return frozenset(str(row[1]) for row in conn.execute("PRAGMA table_info(facts)"))


def _required_text(value: str, name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    return cleaned


def insert_fact(
    conn: sqlite3.Connection,
    content: str,
    *,
    category: str = "general",
    tags: str = "",
    trust_score: float = 0.5,
    memory_kind: str | None = None,
    subject_key: str | None = None,
    predicate_key: str | None = None,
    object_value: str | None = None,
    confidence: float | None = None,
    source_authority: float | None = None,
    scope: str = DEFAULT_SCOPE,
    sensitivity: str = "normal",
    valid_from: str | None = None,
) -> int:
    """Insert one fact and return its id without committing the transaction.

    Optional modern fields are omitted when opening a legacy facts table that
    predates them.  ``scope`` cannot be silently weakened: a non-private scope
    requires a schema that can persist it.
    """

    content = _required_text(content, "content")
    category = _required_text(category, "category")
    scope = _required_text(scope, "scope")
    sensitivity = _required_text(sensitivity, "sensitivity")
    if not 0 <= trust_score <= 1:
        raise ValueError("trust_score must be between 0 and 1")
    for value, name in ((confidence, "confidence"), (source_authority, "source_authority")):
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    available = _fact_columns(conn)
    values: dict[str, Any] = {
        "content": content,
        "category": category,
        "tags": tags,
        "trust_score": trust_score,
        "memory_kind": memory_kind,
        "subject_key": subject_key,
        "predicate_key": predicate_key,
        "object_value": object_value,
        "confidence": confidence,
        "source_authority": source_authority,
        "scope": scope,
        "sensitivity": sensitivity,
        "valid_from": valid_from,
    }
    if "scope" not in available and scope != DEFAULT_SCOPE:
        raise ValueError("legacy facts schema cannot persist a non-private scope")
    modern_defaults: dict[str, Any] = {
        "memory_kind": memory_kind,
        "subject_key": subject_key,
        "predicate_key": predicate_key,
        "object_value": object_value,
        "confidence": confidence,
        "source_authority": source_authority,
        "valid_from": valid_from,
    }
    unsupported = [
        name
        for name, value in modern_defaults.items()
        if value is not None and name not in available
    ]
    if unsupported:
        raise ValueError(
            "legacy facts schema cannot persist fields: " + ", ".join(unsupported)
        )
    columns = [name for name in values if name in available]
    placeholders = ", ".join("?" for _ in columns)
    cursor = conn.execute(
        f"INSERT INTO facts ({', '.join(columns)}) VALUES ({placeholders})",
        tuple(values[name] for name in columns),
    )
    return int(cursor.lastrowid)


def build_scope_predicate(
    allowed_scopes: Sequence[str] | None,
    *,
    column: str = "scope",
    scope_column_available: bool = True,
) -> tuple[str, tuple[str, ...]]:
    """Build a parameterized SQL scope predicate.

    ``None`` retains legacy unrestricted behavior.  An empty sequence denies
    every row.  A legacy table without ``scope`` is treated as private.
    ``column`` is selected by trusted call sites, not user input.
    """

    if column not in {"scope", "f.scope"}:
        raise ValueError("unsupported scope column")
    if allowed_scopes is None:
        return "1", ()
    scopes = tuple(dict.fromkeys(_required_text(value, "scope") for value in allowed_scopes))
    if not scopes:
        return "0", ()
    if not scope_column_available:
        return ("1", ()) if DEFAULT_SCOPE in scopes else ("0", ())
    placeholders = ", ".join("?" for _ in scopes)
    return f"{column} IN ({placeholders})", scopes


def _active_predicates(columns: frozenset[str], *, prefix: str = "") -> list[str]:
    predicates: list[str] = []
    if "invalid_at" in columns:
        predicates.append(f"{prefix}invalid_at IS NULL")
    if "superseded_by" in columns:
        predicates.append(f"{prefix}superseded_by IS NULL")
    if "conflict_group" in columns:
        predicates.append(f"{prefix}conflict_group IS NULL")
    return predicates


def active_facts(
    conn: sqlite3.Connection,
    *,
    allowed_scopes: Sequence[str] | None = (DEFAULT_SCOPE,),
    category: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return current facts authorized by the supplied scope set."""

    if limit < 1:
        return []
    columns = _fact_columns(conn)
    predicates = _active_predicates(columns)
    scope_sql, scope_params = build_scope_predicate(
        allowed_scopes, scope_column_available="scope" in columns
    )
    predicates.append(scope_sql)
    params: list[Any] = list(scope_params)
    if category is not None:
        predicates.append("category = ?")
        params.append(category)
    params.append(limit)
    rows = conn.execute(
        f"SELECT * FROM facts WHERE {' AND '.join(predicates)} "
        "ORDER BY fact_id DESC LIMIT ?",
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def get_active_fact(
    conn: sqlite3.Connection,
    fact_id: int,
    *,
    allowed_scopes: Sequence[str] | None = (DEFAULT_SCOPE,),
) -> dict[str, Any] | None:
    """Return one current authorized fact, or ``None``."""

    columns = _fact_columns(conn)
    predicates = ["fact_id = ?", *_active_predicates(columns)]
    scope_sql, scope_params = build_scope_predicate(
        allowed_scopes, scope_column_available="scope" in columns
    )
    predicates.append(scope_sql)
    row = conn.execute(
        f"SELECT * FROM facts WHERE {' AND '.join(predicates)}",
        (fact_id, *scope_params),
    ).fetchone()
    return dict(row) if row is not None else None


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    *,
    allowed_scopes: Sequence[str] | None = (DEFAULT_SCOPE,),
    category: str | None = None,
    min_trust: float = 0.0,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search current authorized facts using the owned FTS5 index."""

    query = _required_text(query, "query")
    if limit < 1:
        return []
    columns = _fact_columns(conn)
    predicates = ["facts_fts MATCH ?", *_active_predicates(columns, prefix="f.")]
    params: list[Any] = [query]
    scope_sql, scope_params = build_scope_predicate(
        allowed_scopes,
        column="f.scope",
        scope_column_available="scope" in columns,
    )
    predicates.append(scope_sql)
    params.extend(scope_params)
    if category is not None:
        predicates.append("f.category = ?")
        params.append(category)
    predicates.append("f.trust_score >= ?")
    params.extend((min_trust, limit))
    rows = conn.execute(
        f"""
        SELECT f.*, bm25(facts_fts) AS fts_rank
        FROM facts_fts
        JOIN facts AS f ON f.fact_id = facts_fts.rowid
        WHERE {' AND '.join(predicates)}
        ORDER BY fts_rank, f.fact_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def settled_fact_events(
    conn: sqlite3.Connection,
    *,
    allowed_scopes: Sequence[str] | None = (DEFAULT_SCOPE,),
    since: str | None = None,
    until: str | None = None,
) -> list[dict[str, Any]]:
    """Return scope-authorized events that represent settled fact history.

    Creation events for members of an unresolved conflict are withheld. A
    supersession is settled history even when the losing fact retains its
    conflict-group marker, and a resolution is emitted for the selected fact.
    Window bounds are half-open when supplied and must be supplied together.
    """

    if (since is None) != (until is None):
        raise ValueError("since and until must be supplied together")
    columns = _fact_columns(conn)
    required = {
        "scope", "conflict_group", "invalid_at", "superseded_by", "created_at"
    }
    if not required.issubset(columns):
        raise RuntimeError("fact store does not support settled event projections")
    scope_sql, scope_params = build_scope_predicate(
        allowed_scopes, column="f.scope", scope_column_available=True
    )
    window_sql = ""
    window_params: tuple[str, ...] = ()
    if since is not None and until is not None:
        window_sql = (
            "AND julianday(changed_at) >= julianday(?) "
            "AND julianday(changed_at) < julianday(?)"
        )
        window_params = (since, until)
    selected = ", ".join(f"f.{name}" for name in sorted(columns))
    rows = conn.execute(
        f"""
        WITH fact_events AS (
            SELECT 'created' AS kind, f.created_at AS changed_at, f.fact_id
            FROM facts f
            WHERE {scope_sql} AND f.conflict_group IS NULL
            UNION ALL
            SELECT 'superseded', f.invalid_at, f.fact_id
            FROM facts f
            WHERE {scope_sql} AND f.invalid_at IS NOT NULL
              AND f.superseded_by IS NOT NULL
            UNION ALL
            SELECT 'resolved', c.resolved_at, c.resolution_fact_id
            FROM fact_conflicts c
            JOIN facts f ON f.fact_id = c.resolution_fact_id
            WHERE {scope_sql} AND c.resolved_at IS NOT NULL
        )
        SELECT e.kind, e.changed_at, {selected}
        FROM fact_events e
        JOIN facts f ON f.fact_id = e.fact_id
        WHERE e.changed_at IS NOT NULL {window_sql}
        ORDER BY julianday(e.changed_at), e.changed_at,
                 CASE e.kind WHEN 'created' THEN 0 WHEN 'superseded' THEN 1 ELSE 2 END,
                 f.fact_id
        """,
        (*scope_params, *scope_params, *scope_params, *window_params),
    ).fetchall()
    return [dict(row) for row in rows]


def historical_facts_by_id(
    conn: sqlite3.Connection,
    fact_ids: Iterable[int],
    *,
    allowed_scopes: Sequence[str] | None = (DEFAULT_SCOPE,),
) -> list[dict[str, Any]]:
    """Return historical facts by id without weakening scope authorization."""

    ids = tuple(dict.fromkeys(int(fact_id) for fact_id in fact_ids))
    if not ids:
        return []
    columns = _fact_columns(conn)
    scope_sql, scope_params = build_scope_predicate(
        allowed_scopes,
        column="f.scope",
        scope_column_available="scope" in columns,
    )
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT f.* FROM facts f WHERE f.fact_id IN ({placeholders}) "
        f"AND {scope_sql} ORDER BY f.fact_id",
        (*ids, *scope_params),
    ).fetchall()
    return [dict(row) for row in rows]


def resolve_entity(
    conn: sqlite3.Connection,
    name: str,
    *,
    entity_type: str = "unknown",
    aliases: str = "",
) -> int:
    """Resolve or insert an entity without committing."""

    name = _required_text(name, "name")
    row = conn.execute(
        "SELECT entity_id FROM entities WHERE name = ? COLLATE NOCASE "
        "ORDER BY entity_id LIMIT 1",
        (name,),
    ).fetchone()
    if row is not None:
        return int(row[0])
    cursor = conn.execute(
        "INSERT INTO entities(name, entity_type, aliases) VALUES (?, ?, ?)",
        (name, entity_type, aliases),
    )
    return int(cursor.lastrowid)


def link_fact_entities(
    conn: sqlite3.Connection, fact_id: int, entity_ids: Iterable[int]
) -> None:
    """Link a fact to entities without committing."""

    conn.executemany(
        "INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES (?, ?)",
        ((fact_id, int(entity_id)) for entity_id in entity_ids),
    )


# Concise aliases for callers that treat this module as the store API.
connect = connect_database
initialize_schema = ensure_core_schema
