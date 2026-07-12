"""Explicit schema versioning for Enfold-owned SQLite databases.

This module deliberately does *not* open Enfold's live database and does not
run migrations at import time.  Callers must pass an already-open connection
to :func:`migrate` from an explicit maintenance command.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib


SUPPORTED_SCHEMA_VERSION = 1
SCHEMA_VERSION_KEY = "schema_version"


class SchemaError(RuntimeError):
    """Base class for schema ledger and migration failures."""


class SchemaTooNewError(SchemaError):
    """Raised when a database was created by a newer Enfold release."""


class SchemaLedgerError(SchemaError):
    """Raised when the two version records disagree or are malformed."""


class MigrationError(SchemaError):
    """Raised when an explicit migration cannot be applied safely."""


@dataclass(frozen=True)
class Migration:
    """One ordered, explicit, transactional schema change."""

    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone() is not None


_V1_COLUMNS: Mapping[str, frozenset[str]] = {
    "facts": frozenset(
        {
            "fact_id", "content", "category", "tags", "trust_score",
            "retrieval_count", "helpful_count", "created_at", "updated_at",
            "hrr_vector", "valid_from", "invalid_at", "superseded_by",
            "memory_kind", "subject_key", "predicate_key", "object_value",
            "object_entity_id", "confidence", "source_authority", "scope",
            "sensitivity", "correction_status", "schema_version", "conflict_group",
        }
    ),
    "entities": frozenset({"entity_id", "name", "entity_type", "aliases", "created_at"}),
    "fact_entities": frozenset({"fact_id", "entity_id"}),
    "memory_clients": frozenset({"client_id", "surface", "created_at"}),
    "memory_sessions": frozenset(
        {"session_id", "client_id", "agent_id", "capabilities_json", "access_scopes_json"}
    ),
    "observations": frozenset(
        {
            "observation_id", "client_id", "session_id", "source_type",
            "project_root", "repository", "branch", "commit_sha",
            "content_sha256", "recorded_at", "scope", "sensitivity",
        }
    ),
    "fact_provenance": frozenset(
        {"fact_id", "observation_id", "relation", "created_at"}
    ),
    "memory_write_log": frozenset(
        {
            "write_id", "idempotency_key", "client_id", "operation", "outcome",
            "recorded_at", "request_sha256",
        }
    ),
    "privacy_erasure_log": frozenset(
        {
            "erasure_id", "fact_id", "requested_by", "reason", "erased_at",
            "affected_observations", "affected_embeddings", "affected_queue_rows",
        }
    ),
    "embedding_jobs": frozenset(
        {
            "job_id", "fact_id", "document_identity", "embedding_version",
            "dimensions", "content_sha256", "status", "attempts",
            "available_at", "lease_token", "lease_expires_at",
        }
    ),
    "fact_embeddings": frozenset(
        {"fact_id", "embedding", "dim", "embedding_identity"}
    ),
    "fact_conflicts": frozenset(
        {
            "conflict_id", "scope", "subject_key", "predicate_key",
            "detected_at", "resolved_at",
        }
    ),
    "fact_conflict_members": frozenset({"conflict_id", "fact_id"}),
    "schema_migrations": frozenset({"version", "applied_at"}),
    "enfold_meta": frozenset({"key", "value"}),
    "facts_fts": frozenset({"content", "tags"}),
    "extract_queue": frozenset(
        {
            "id", "payload", "created_at", "attempts", "last_error",
            "status", "not_before", "lease_owner", "lease_until",
            "lease_token", "payload_hash",
        }
    ),
}

_V1_INDEXES = frozenset(
    {
        "idx_entities_name",
        "idx_facts_category",
        "idx_facts_active_scope",
        "uq_facts_current_state_slot",
        "idx_fact_conflict_members_fact",
        "idx_observations_session",
        "idx_fact_provenance_observation",
        "idx_memory_write_log_session",
        "idx_extract_queue_status",
        "uq_extract_queue_active_payload_hash",
        "idx_embedding_jobs_claim",
        "idx_fact_embeddings_fact_id",
        "idx_fact_embeddings_identity_dim",
    }
)
_V1_TRIGGERS = frozenset({"facts_ai", "facts_ad", "facts_au"})

# These indexes were created by released legacy Enfold schemas but are not
# required by v1.  Rebuilding ``facts`` drops them, so each admitted shape has
# an explicit, canonical recreation statement.  Names alone are never enough:
# callers must pass the semantic shape check below.
_SAFE_LEGACY_FACT_INDEX_SQL: Mapping[str, str] = {
    "idx_facts_trust": (
        "CREATE INDEX idx_facts_trust ON facts(trust_score DESC)"
    ),
}


def _safe_legacy_fact_index_sql(
    conn: sqlite3.Connection, name: str
) -> str | None:
    """Return recreation SQL only for an exact known legacy index shape."""

    if name != "idx_facts_trust":
        return None
    index_row = next(
        (row for row in conn.execute("PRAGMA index_list(facts)") if row[1] == name),
        None,
    )
    if index_row is None:
        return None
    # idx_facts_trust was an ordinary, non-unique, non-partial index.  Checking
    # xinfo (rather than sqlite_master formatting) accepts harmless whitespace
    # differences while rejecting expressions, extra keys, alternate order,
    # and alternate collations.
    if bool(index_row[2]) or str(index_row[3]) != "c" or bool(index_row[4]):
        return None
    key_columns = [
        (str(row[2]), bool(row[3]), str(row[4]).upper())
        for row in conn.execute(f'PRAGMA index_xinfo("{name}")')
        if bool(row[5])
    ]
    if key_columns != [("trust_score", True, "BINARY")]:
        return None
    return _SAFE_LEGACY_FACT_INDEX_SQL[name]


def _content_has_unique_constraint(conn: sqlite3.Connection) -> bool:
    for row in conn.execute("PRAGMA index_list(facts)"):
        if not bool(row[2]):
            continue
        columns = [
            str(info[2])
            for info in conn.execute(f'PRAGMA index_info("{str(row[1])}")')
        ]
        if columns == ["content"]:
            return True
    return False


def _rebuild_legacy_unique_facts(conn: sqlite3.Connection) -> bool:
    """Remove legacy ``UNIQUE(content)`` while retaining row identities."""

    if not _content_has_unique_constraint(conn):
        return False

    from .core_store import _SCHEMA_STATEMENTS

    existing = [str(row[1]) for row in conn.execute("PRAGMA table_info(facts)")]
    unknown = sorted(set(existing) - set(_V1_COLUMNS["facts"]))
    if unknown:
        raise MigrationError(
            "cannot safely rebuild legacy facts table with unknown columns: "
            + ", ".join(unknown)
        )
    known_indexes = {
        "idx_facts_category",
        "idx_facts_active_scope",
        "uq_facts_current_state_slot",
    }
    unexpected_indexes: list[str] = []
    legacy_indexes_to_recreate: list[str] = []
    for row in conn.execute("PRAGMA index_list(facts)"):
        name = str(row[1])
        columns = [
            str(info[2])
            for info in conn.execute(f'PRAGMA index_info("{name}")')
        ]
        if name in known_indexes or (bool(row[2]) and columns == ["content"]):
            continue
        recreation_sql = _safe_legacy_fact_index_sql(conn, name)
        if recreation_sql is not None:
            legacy_indexes_to_recreate.append(recreation_sql)
            continue
        unexpected_indexes.append(name)
    unexpected_triggers = [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='facts'"
        )
        if str(row[0]) not in _V1_TRIGGERS
    ]
    if unexpected_indexes or unexpected_triggers:
        objects = [
            *(f"index:{name}" for name in unexpected_indexes),
            *(f"trigger:{name}" for name in unexpected_triggers),
        ]
        raise MigrationError(
            "cannot safely rebuild legacy facts table with unknown dependent "
            "objects: " + ", ".join(objects)
        )
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='facts_v1_rebuild'"
    ).fetchone() is not None:
        raise MigrationError("temporary facts_v1_rebuild object already exists")

    create_sql = _SCHEMA_STATEMENTS[0].replace(
        "CREATE TABLE IF NOT EXISTS facts (",
        "CREATE TABLE facts_v1_rebuild (",
        1,
    )
    conn.execute(create_sql)
    target = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(facts_v1_rebuild)")
    }
    copied = [column for column in existing if column in target]
    quoted = ", ".join(f'"{column}"' for column in copied)
    conn.execute(
        f"INSERT INTO facts_v1_rebuild ({quoted}) SELECT {quoted} FROM facts"
    )
    before = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    after = conn.execute("SELECT COUNT(*) FROM facts_v1_rebuild").fetchone()[0]
    if before != after:
        raise MigrationError(
            f"facts rebuild row count mismatch: source={before}, rebuilt={after}"
        )
    conn.execute("DROP TABLE facts")
    conn.execute("ALTER TABLE facts_v1_rebuild RENAME TO facts")
    for statement in legacy_indexes_to_recreate:
        conn.execute(statement)
    return True


def _ensure_extraction_queue_schema(conn: sqlite3.Connection) -> None:
    """Provision the v1 daemon queue, preserving compatible legacy rows."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS extract_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            not_before REAL,
            lease_owner TEXT,
            lease_until REAL,
            lease_token TEXT,
            payload_hash TEXT,
            proposal_json TEXT,
            proposal_hash TEXT
        )
        """
    )
    info = conn.execute("PRAGMA table_info(extract_queue)").fetchall()
    columns = {str(row[1]): row for row in info}
    if "id" not in columns and "queue_id" in columns:
        if int(columns["queue_id"][5]) != 1:
            raise MigrationError("extract_queue.queue_id must be the primary key")
        conn.execute("ALTER TABLE extract_queue RENAME COLUMN queue_id TO id")
        info = conn.execute("PRAGMA table_info(extract_queue)").fetchall()
        columns = {str(row[1]): row for row in info}
    if "id" not in columns or int(columns["id"][5]) != 1:
        raise MigrationError("extract_queue.id must be the primary key")
    if "payload" not in columns or int(columns["payload"][3]) != 1:
        raise MigrationError("extract_queue.payload must be NOT NULL")
    for name, declaration in (
        # SQLite cannot add a non-constant CURRENT_TIMESTAMP default to a
        # populated legacy table. New stores receive the default above;
        # compatible legacy stores are backfilled below.
        ("created_at", "TEXT"),
        ("attempts", "INTEGER NOT NULL DEFAULT 0"),
        ("last_error", "TEXT"),
        ("status", "TEXT NOT NULL DEFAULT 'pending'"),
        ("not_before", "REAL"),
        ("lease_owner", "TEXT"),
        ("lease_until", "REAL"),
        ("lease_token", "TEXT"),
        ("payload_hash", "TEXT"),
        # A proposal snapshot is deliberately nullable.  Existing pending
        # work has not reached a model yet; paired non-NULL values mean a
        # validated batch was durably recorded before any fact write.
        ("proposal_json", "TEXT"),
        ("proposal_hash", "TEXT"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE extract_queue ADD COLUMN {name} {declaration}")
    conn.execute(
        "UPDATE extract_queue SET created_at = CURRENT_TIMESTAMP "
        "WHERE created_at IS NULL"
    )
    snapshot_mismatch = conn.execute(
        "SELECT id FROM extract_queue WHERE "
        "(proposal_json IS NULL) != (proposal_hash IS NULL) LIMIT 1"
    ).fetchone()
    if snapshot_mismatch is not None:
        raise MigrationError("extract_queue proposal snapshot columns are inconsistent")
    invalid = conn.execute(
        "SELECT id FROM extract_queue WHERE payload IS NULL OR status IS NULL "
        "OR attempts IS NULL OR attempts < 0 LIMIT 1"
    ).fetchone()
    if invalid is not None:
        raise MigrationError("extract_queue contains incompatible null/negative rows")
    active_hashes: set[str] = set()
    for row_id, payload, status, digest in conn.execute(
        "SELECT id, payload, status, payload_hash FROM extract_queue ORDER BY id"
    ):
        calculated = hashlib.sha256(str(payload).encode("utf-8")).hexdigest()
        candidate = str(digest) if digest else calculated
        if str(status) in {"pending", "processing"}:
            if candidate in active_hashes:
                # Preserve the row while preventing a false uniqueness claim.
                candidate = None
            else:
                active_hashes.add(candidate)
        conn.execute(
            "UPDATE extract_queue SET payload_hash = ? WHERE id = ?",
            (candidate, int(row_id)),
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_extract_queue_status "
        "ON extract_queue(status, not_before, lease_until, id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_extract_queue_active_payload_hash "
        "ON extract_queue(payload_hash) WHERE payload_hash IS NOT NULL "
        "AND status IN ('pending', 'processing')"
    )


def verify_schema_shape(conn: sqlite3.Connection, version: int) -> None:
    """Fail closed unless all objects required by ``version`` are present."""

    if version < 1:
        return
    missing: list[str] = []
    for table, required_columns in _V1_COLUMNS.items():
        if not _table_exists(conn, table):
            missing.append(f"table:{table}")
            continue
        actual = {
            str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')
        }
        for column in sorted(required_columns - actual):
            missing.append(f"column:{table}.{column}")
    indexes = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    missing.extend(f"index:{name}" for name in sorted(_V1_INDEXES - indexes))
    if "uq_facts_current_state_slot" in indexes:
        slot_columns = tuple(
            str(row[2])
            for row in conn.execute(
                'PRAGMA index_info("uq_facts_current_state_slot")'
            )
        )
        if slot_columns != ("scope", "subject_key", "predicate_key"):
            missing.append("index-shape:uq_facts_current_state_slot")
    triggers = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
    }
    missing.extend(f"trigger:{name}" for name in sorted(_V1_TRIGGERS - triggers))
    if _table_exists(conn, "facts") and _content_has_unique_constraint(conn):
        missing.append("constraint:facts.content_unique")
    if missing:
        preview = ", ".join(missing[:8])
        suffix = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise SchemaLedgerError(
            f"schema ledger claims v{version}, but required shape is incomplete: "
            f"{preview}{suffix}"
        )


def _migration_001_complete_schema(conn: sqlite3.Connection) -> None:
    """Build and verify the complete standalone v1 shape atomically."""

    # Imports are intentionally local: standalone modules do not import the
    # migration coordinator, avoiding cycles and import-time schema effects.
    from .core_store import ensure_core_schema
    from .provenance import ensure_provenance_schema
    from .state_slots import ensure_state_slot_schema

    # Some early/minimal stores have a facts table but only a subset of the
    # holographic columns.  Add every safely additive base column before the
    # core initializer creates indexes and the external-content FTS table.
    if _table_exists(conn, "facts"):
        base_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(facts)")
        }
        for name, definition in (
            ("category", "TEXT DEFAULT 'general'"),
            ("tags", "TEXT DEFAULT ''"),
            ("trust_score", "REAL DEFAULT 0.5"),
            ("retrieval_count", "INTEGER DEFAULT 0"),
            ("helpful_count", "INTEGER DEFAULT 0"),
            ("created_at", "TIMESTAMP"),
            ("updated_at", "TIMESTAMP"),
            ("hrr_vector", "BLOB"),
        ):
            if name not in base_columns:
                conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {definition}")
                base_columns.add(name)

    ensure_core_schema(conn)
    rebuilt_facts = _rebuild_legacy_unique_facts(conn)

    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(facts)")}
    for name, column_type in (
        ("valid_from", "TIMESTAMP"),
        ("invalid_at", "TIMESTAMP"),
        ("superseded_by", "INTEGER"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {column_type}")
            columns.add(name)

    if not ensure_state_slot_schema(conn):
        raise MigrationError(
            "cannot install v1 current-state uniqueness invariant; resolve "
            "duplicate active state slots first"
        )
    # A legacy facts table lacked the columns needed by the active-scope
    # index during the first core pass.  Re-running the idempotent initializer
    # after typed columns exist installs that remaining owned index.
    ensure_core_schema(conn)
    if rebuilt_facts:
        conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
    ensure_provenance_schema(conn, manage_transaction=False)
    _ensure_extraction_queue_schema(conn)

    conn.execute(
        """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE enfold_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    verify_schema_shape(conn, 1)
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise MigrationError(
            f"v1 migration produced {len(violations)} foreign-key violation(s)"
        )


MIGRATIONS: Mapping[int, Migration] = {
    1: Migration(1, "complete_standalone_schema", _migration_001_complete_schema),
}


def _needs_extraction_queue_patch(conn: sqlite3.Connection) -> bool:
    """Whether a v1 store predates durable extraction proposal snapshots.

    The snapshot columns are an additive v1 queue hardening.  They intentionally
    are not part of the immutable v1 ledger shape: that keeps an already-v1
    database readable long enough for an explicit ``migrate`` to add them.
    """

    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(extract_queue)")
    }
    if not {"proposal_json", "proposal_hash"}.issubset(columns):
        return True
    return conn.execute(
        "SELECT 1 FROM extract_queue WHERE "
        "(proposal_json IS NULL) != (proposal_hash IS NULL) LIMIT 1"
    ).fetchone() is not None


def _apply_extraction_queue_patch(conn: sqlite3.Connection) -> None:
    """Atomically add v1 extraction snapshot columns when explicitly requested."""

    if not _needs_extraction_queue_patch(conn):
        return
    foreign_keys_enabled = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    try:
        if foreign_keys_enabled:
            conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN IMMEDIATE")
        _ensure_extraction_queue_schema(conn)
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise MigrationError(
                f"extraction queue patch left {len(violations)} foreign-key violation(s)"
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        if isinstance(exc, SchemaError):
            raise
        raise MigrationError("extraction queue snapshot patch failed") from exc
    finally:
        if foreign_keys_enabled:
            conn.execute("PRAGMA foreign_keys=ON")


def schema_version(conn: sqlite3.Connection) -> int:
    """Return the recorded schema version without mutating ``conn``.

    Legacy databases with neither Enfold version table are version zero.  Once
    either table exists, both must exist and agree; ambiguity fails closed.
    """

    has_ledger = _table_exists(conn, "schema_migrations")
    has_meta = _table_exists(conn, "enfold_meta")
    if not has_ledger and not has_meta:
        return 0
    if has_ledger != has_meta:
        raise SchemaLedgerError(
            "incomplete Enfold schema ledger: schema_migrations and enfold_meta "
            "must either both exist or both be absent"
        )

    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    ledger_version = row[0] if row and row[0] is not None else None
    meta_row = conn.execute(
        "SELECT value FROM enfold_meta WHERE key = ?", (SCHEMA_VERSION_KEY,)
    ).fetchone()
    if ledger_version is None or meta_row is None:
        raise SchemaLedgerError("Enfold schema version records are missing")
    try:
        meta_version = int(meta_row[0])
    except (TypeError, ValueError) as exc:
        raise SchemaLedgerError("enfold_meta schema_version is not an integer") from exc
    if ledger_version != meta_version:
        raise SchemaLedgerError(
            f"schema version records disagree: ledger={ledger_version}, "
            f"meta={meta_version}"
        )
    verify_schema_shape(conn, ledger_version)
    return ledger_version


def require_compatible_schema(
    conn: sqlite3.Connection, *, supported_version: int = SUPPORTED_SCHEMA_VERSION
) -> int:
    """Fail closed if ``conn`` uses a schema newer than this code supports."""

    version = schema_version(conn)
    if version > supported_version:
        raise SchemaTooNewError(
            f"database schema {version} is newer than supported schema "
            f"{supported_version}; upgrade Enfold before opening this database"
        )
    return version


def migrate(
    conn: sqlite3.Connection,
    *,
    target_version: int = SUPPORTED_SCHEMA_VERSION,
    migrations: Mapping[int, Migration] = MIGRATIONS,
) -> int:
    """Explicitly migrate ``conn`` to ``target_version``.

    Each migration is committed in its own ``BEGIN IMMEDIATE`` transaction.
    The schema change, ledger row, and metadata version therefore succeed or
    roll back together.  An existing caller transaction is rejected because
    committing or nesting it would violate ownership of that transaction.
    """

    if conn.in_transaction:
        raise MigrationError("cannot migrate inside an existing transaction")
    if target_version < 0 or target_version > SUPPORTED_SCHEMA_VERSION:
        raise MigrationError(
            f"target schema {target_version} is outside supported range "
            f"0..{SUPPORTED_SCHEMA_VERSION}"
        )

    current = require_compatible_schema(conn)
    if target_version < current:
        raise MigrationError(
            f"downgrades are not supported (database={current}, target={target_version})"
        )

    for version in range(current + 1, target_version + 1):
        migration = migrations.get(version)
        if migration is None or migration.version != version:
            raise MigrationError(f"no explicit migration registered for version {version}")
        foreign_keys_enabled = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
        try:
            if foreign_keys_enabled:
                conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("BEGIN IMMEDIATE")
            migration.apply(conn)
            conn.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (version, _utc_now()),
            )
            conn.execute(
                "INSERT INTO enfold_meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (SCHEMA_VERSION_KEY, str(version)),
            )
            verify_schema_shape(conn, version)
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise MigrationError(
                    f"migration {version} left {len(violations)} "
                    "foreign-key violation(s)"
                )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            if isinstance(exc, SchemaError):
                raise
            raise MigrationError(
                f"migration {version} ({migration.name}) failed"
            ) from exc
        finally:
            if foreign_keys_enabled:
                conn.execute("PRAGMA foreign_keys=ON")
    # Proposal snapshots harden the v1 extraction queue without changing the
    # public schema ledger.  Existing v1 stores remain readable, but explicit
    # maintenance runs upgrade the queue before a processor is allowed to use
    # it for automatic writes.
    if target_version >= 1:
        _apply_extraction_queue_patch(conn)
    return schema_version(conn)


# Verbose aliases make call sites self-documenting and retain a small API.
get_schema_version = schema_version
assert_schema_compatible = require_compatible_schema
apply_migrations = migrate
