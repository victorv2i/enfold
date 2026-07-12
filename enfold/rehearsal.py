"""Synthetic end-to-end migration and rollback rehearsal.

The harness is deliberately path-explicit and refuses ``.hermes`` paths.  It
never discovers a user database or imports the legacy Hermes provider.  Its
fixture resembles the legacy store closely enough to exercise FTS, opaque
embedding payloads, extraction/reflection data, and foreign-key children.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

from .backup import (
    backup_database,
    maintenance_database_lock,
    restore_database,
    verify_database,
)
from .core_store import search_fts
from .policy import MemoryPolicy
from .provenance import ConnectionContext, WriteRequest
from .schema import MigrationError, migrate, schema_version
from .state_slots import StateCandidate, read_current_state
from .write_service import FactWriteResult, MemoryWriteService


class RehearsalError(RuntimeError):
    """Raised when a rehearsal invariant is not proven."""


@dataclass(frozen=True, slots=True)
class RehearsalReport:
    legacy_fingerprint: str
    backup_fingerprint: str
    restored_fingerprint: str
    migrated_schema_version: int
    restored_schema_version: int
    migrated_fact_count: int
    restored_fact_count: int
    current_fact_id: int
    current_search_fact_ids: tuple[int, ...]
    rollback_artifact_verified: bool
    restored_legacy_verified: bool


@dataclass(frozen=True, slots=True)
class SnapshotRehearsalReport:
    source_snapshot: str
    source_fingerprint: str
    rollback_fingerprint: str
    restored_fingerprint: str
    migrated_schema_version: int
    smoke_fact_id: int
    smoke_search_verified: bool
    smoke_evidence_verified: bool
    restored_schema_version: int
    source_unchanged: bool


_LEGACY_TABLES = (
    "facts",
    "entities",
    "fact_entities",
    "embeddings",
    "extract_queue",
    "reflection_runs",
    "reflection_sources",
    "legacy_notes",
)


def _safe_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if ".hermes" in resolved.parts:
        raise RehearsalError("rehearsal paths must not be below .hermes")
    return resolved


def create_legacy_fixture(path: str | Path, *, unknown_fact_object: bool = False) -> Path:
    """Create a deterministic, synthetic legacy database at an unused path."""

    destination = _safe_path(path)
    if destination.exists():
        raise RehearsalError(f"fixture destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(destination)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
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
            CREATE TABLE entities (
                entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                entity_type TEXT DEFAULT 'unknown',
                aliases TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE fact_entities (
                fact_id INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
                entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
                PRIMARY KEY (fact_id, entity_id)
            );
            CREATE VIRTUAL TABLE facts_fts
            USING fts5(content, tags, content=facts, content_rowid=fact_id);
            CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, content, tags)
                VALUES (new.fact_id, new.content, new.tags);
            END;
            CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                VALUES ('delete', old.fact_id, old.content, old.tags);
            END;
            CREATE TRIGGER facts_au AFTER UPDATE OF content, tags ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                VALUES ('delete', old.fact_id, old.content, old.tags);
                INSERT INTO facts_fts(rowid, content, tags)
                VALUES (new.fact_id, new.content, new.tags);
            END;
            CREATE TABLE embeddings (
                fact_id INTEGER PRIMARY KEY REFERENCES facts(fact_id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE extract_queue (
                queue_id INTEGER PRIMARY KEY,
                fact_id INTEGER REFERENCES facts(fact_id) ON DELETE SET NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE reflection_runs (
                run_id INTEGER PRIMARY KEY,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE reflection_sources (
                run_id INTEGER NOT NULL REFERENCES reflection_runs(run_id),
                fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
                PRIMARY KEY (run_id, fact_id)
            );
            CREATE TABLE legacy_notes (
                note_id INTEGER PRIMARY KEY,
                body TEXT NOT NULL
            );
            CREATE VIEW legacy_notes_view AS
                SELECT note_id, body FROM legacy_notes;

            INSERT INTO facts(
                content, category, tags, trust_score, retrieval_count,
                helpful_count, hrr_vector
            ) VALUES
                ('Victor builds Enfold on the synthetic lab', 'project',
                 'enfold,legacy', 0.9, 7, 2, X'01020304'),
                ('Synthetic morning briefing uses model alpha', 'preference',
                 'briefing,model', 0.8, 3, 1, X'05060708'),
                ('A fictional teammate prefers concise reports', 'people',
                 'fictional,preference', 0.7, 1, 1, NULL);
            INSERT INTO entities(name, entity_type, aliases)
                VALUES ('Enfold', 'project', 'shared memory');
            INSERT INTO fact_entities VALUES (1, 1);
            INSERT INTO embeddings VALUES
                (1, 'synthetic-embed-v1', 4, X'0000803F000000000000000000000000',
                 '2026-07-11T12:00:00Z'),
                (2, 'synthetic-embed-v1', 4, X'000000000000803F0000000000000000',
                 '2026-07-11T12:01:00Z');
            INSERT INTO extract_queue VALUES
                (1, 3, '{"source":"synthetic transcript"}', 'done',
                 '2026-07-11T12:02:00Z'),
                (2, NULL, '{"source":"synthetic pending"}', 'pending',
                 '2026-07-11T12:03:00Z');
            INSERT INTO reflection_runs VALUES
                (1, 'Synthetic project reflection', '2026-07-11T12:04:00Z');
            INSERT INTO reflection_sources VALUES (1, 1), (1, 2);
            INSERT INTO legacy_notes VALUES
                (1, 'An extension-owned table migration must preserve.');
            """
        )
        if unknown_fact_object:
            conn.execute(
                "CREATE INDEX extension_fact_content ON facts(content, category)"
            )
        conn.commit()
        if conn.execute("PRAGMA foreign_key_check").fetchall():
            raise RehearsalError("synthetic legacy fixture has foreign-key violations")
    finally:
        conn.close()
    return destination


def _legacy_fingerprint(path: Path) -> str:
    """Fingerprint legacy data and extension objects, excluding timestamps in DDL."""

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        payload: dict[str, object] = {}
        for table in _LEGACY_TABLES:
            columns = [
                str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')
            ]
            rows = conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            payload[table] = {"columns": columns, "rows": rows}
        payload["fts"] = conn.execute(
            "SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'synthetic' ORDER BY rowid"
        ).fetchall()
        payload["notes_view"] = conn.execute(
            "SELECT * FROM legacy_notes_view ORDER BY note_id"
        ).fetchall()
        payload["foreign_keys"] = conn.execute("PRAGMA foreign_key_check").fetchall()
        encoded = json.dumps(payload, sort_keys=True, default=lambda value: value.hex())
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    finally:
        conn.close()


def database_fingerprint(path: str | Path) -> str:
    """Stream a deterministic logical fingerprint of an arbitrary snapshot."""

    source = _safe_path(path)
    digest = hashlib.sha256()
    conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        objects = conn.execute(
            """
            SELECT type, name, tbl_name, COALESCE(sql, '')
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        for item in objects:
            digest.update(repr(tuple(item)).encode("utf-8"))
            digest.update(b"\0")
        tables = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table in tables:
            digest.update(f"table:{table}\0".encode())
            try:
                cursor = conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
            except sqlite3.OperationalError:
                cursor = conn.execute(f'SELECT * FROM "{table}"')
            for row in cursor:
                normalized = tuple(
                    value.hex() if isinstance(value, bytes) else value for value in row
                )
                digest.update(repr(normalized).encode("utf-8"))
                digest.update(b"\0")
        return digest.hexdigest()
    finally:
        conn.close()


def _fact_writer(
    conn: sqlite3.Connection, request: WriteRequest, _observation_id: int
) -> FactWriteResult:
    cursor = conn.execute(
        """
        INSERT INTO facts(
            content, category, tags, trust_score, source_authority,
            scope, sensitivity, correction_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request.content,
            request.category,
            request.tags,
            request.trust_score,
            request.source_authority,
            request.scope,
            request.sensitivity,
            request.correction_status,
        ),
    )
    return FactWriteResult(int(cursor.lastrowid))


def _write_state(
    service: MemoryWriteService,
    context: ConnectionContext,
    *,
    key: str,
    model: str,
    valid_from: str,
) -> int:
    content = f"Synthetic morning briefing uses model {model}"
    request = WriteRequest(
        idempotency_key=key,
        content=content,
        source_type="synthetic_rehearsal",
        category="preference",
        tags="briefing,model,rehearsal",
        trust_score=0.9,
        source_authority=0.9,
    )
    candidate = StateCandidate(
        content=content,
        subject_key="synthetic:morning-briefing",
        predicate_key="model",
        object_value=model,
        source_authority=0.9,
        valid_from=valid_from,
    )
    outcome = service.write(context, request, state_candidate=candidate)
    if outcome.fact_id is None:
        raise RehearsalError(f"smoke write failed with outcome {outcome.outcome}")
    return outcome.fact_id


def run_rehearsal(directory: str | Path) -> RehearsalReport:
    """Run backup, migration, state-write/search, and destructive rollback."""

    root = _safe_path(directory)
    root.mkdir(parents=True, exist_ok=True)
    database = create_legacy_fixture(root / "synthetic-legacy.sqlite")
    rollback = root / "synthetic-rollback.sqlite"
    legacy_fingerprint = _legacy_fingerprint(database)

    backup_report = backup_database(database, rollback)
    if not backup_report.ok:
        raise RehearsalError("verified backup failed")
    backup_fingerprint = _legacy_fingerprint(rollback)
    if backup_fingerprint != legacy_fingerprint:
        raise RehearsalError("rollback artifact does not match legacy source")

    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        migrated_version = migrate(conn)
        if migrated_version != 1:
            raise RehearsalError("migration did not reach schema version 1")
        context = ConnectionContext(
            client_id="rehearsal-codex",
            surface="codex",
            agent_id="synthetic-codex",
            session_id="synthetic-rehearsal",
            project_root=str(root),
            repository="synthetic/enfold",
            branch="rehearsal",
            commit_sha="0" * 40,
            capabilities=("memory.read", "memory.write"),
        )
        service = MemoryWriteService(
            conn,
            _fact_writer,
            MemoryPolicy({context.client_id: ("private",)}),
        )
        first_id = _write_state(
            service,
            context,
            key="rehearsal-state-alpha",
            model="alpha-v1",
            valid_from="2026-07-11T12:00:00Z",
        )
        current_id = _write_state(
            service,
            context,
            key="rehearsal-state-beta",
            model="beta-v2",
            valid_from="2026-07-12T12:00:00Z",
        )
        current = read_current_state(
            conn, "synthetic:morning-briefing", "model"
        )
        if current is None or current.fact_id != current_id:
            raise RehearsalError("current-state smoke did not select replacement")
        hits = search_fts(conn, "Synthetic morning briefing", limit=20)
        search_ids = tuple(int(hit["fact_id"]) for hit in hits)
        if current_id not in search_ids or first_id in search_ids:
            raise RehearsalError("current search leaked the superseded state")
        migrated_count = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
        migrated_verify = verify_database(conn)
        if not migrated_verify.ok:
            raise RehearsalError("migrated database failed integrated verification")
    finally:
        conn.close()

    restore_report = restore_database(rollback, database, overwrite=True)
    if not restore_report.ok:
        raise RehearsalError("verified restore failed")
    restored_fingerprint = _legacy_fingerprint(database)
    if restored_fingerprint != legacy_fingerprint:
        raise RehearsalError("restored legacy data differs from pre-migration data")

    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as restored:
        restored_version = schema_version(restored)
        restored_count = int(restored.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
        restored_ok = verify_database(restored, check_fts=False).ok
        if restored_version != 0 or not restored_ok:
            raise RehearsalError("restored legacy database is not independently usable")
    # Reopen the rollback artifact after restore; it remains an independent,
    # old-schema recovery point rather than being consumed by restoration.
    with sqlite3.connect(f"file:{rollback}?mode=ro", uri=True) as artifact:
        artifact_ok = (
            schema_version(artifact) == 0
            and verify_database(artifact, check_fts=False).ok
            and artifact.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 3
        )
    if not artifact_ok:
        raise RehearsalError("rollback artifact is not independently usable")

    return RehearsalReport(
        legacy_fingerprint=legacy_fingerprint,
        backup_fingerprint=backup_fingerprint,
        restored_fingerprint=restored_fingerprint,
        migrated_schema_version=migrated_version,
        restored_schema_version=restored_version,
        migrated_fact_count=migrated_count,
        restored_fact_count=restored_count,
        current_fact_id=current_id,
        current_search_fact_ids=search_ids,
        rollback_artifact_verified=artifact_ok,
        restored_legacy_verified=restored_ok,
    )


def prove_unknown_object_fails_closed(path: str | Path) -> bool:
    """Prove migration refuses an extension-owned facts index atomically."""

    database = create_legacy_fixture(path, unknown_fact_object=True)
    before = _legacy_fingerprint(database)
    conn = sqlite3.connect(database)
    try:
        try:
            migrate(conn)
        except MigrationError:
            pass
        else:
            raise RehearsalError("migration accepted an unknown facts-dependent object")
        if schema_version(conn) != 0:
            raise RehearsalError("failed migration changed the schema ledger")
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' "
            "AND name='extension_fact_content'"
        ).fetchone() is None:
            raise RehearsalError("failed migration lost the extension-owned index")
    finally:
        conn.close()
    after = _legacy_fingerprint(database)
    if after != before:
        raise RehearsalError("failed migration changed legacy data")
    return True


def rehearse_snapshot(
    snapshot_path: str | Path, workdir: str | Path
) -> SnapshotRehearsalReport:
    """Rehearse v1 on an explicit offline snapshot without modifying it."""

    source = _safe_path(snapshot_path)
    root = _safe_path(workdir)
    if not source.is_file():
        raise RehearsalError(f"snapshot does not exist: {source}")
    if root == source or source in root.parents:
        raise RehearsalError("rehearsal workdir must be separate from the source snapshot")
    if root.exists():
        if not root.is_dir():
            raise RehearsalError("rehearsal workdir must be a directory")
        if any(root.iterdir()):
            raise RehearsalError(
                "rehearsal workdir already contains files; it must be empty"
            )
    else:
        root.mkdir(parents=True)
    candidate = root / "migration-candidate.sqlite"
    rollback = root / "rollback-artifact.sqlite"
    with maintenance_database_lock(source):
        source_fingerprint = database_fingerprint(source)
        copied = backup_database(source, candidate)
        if not copied.ok:
            raise RehearsalError("snapshot copy verification failed")
        rollback_report = backup_database(candidate, rollback)
        if not rollback_report.ok:
            raise RehearsalError("rollback artifact verification failed")
        rollback_fingerprint = database_fingerprint(rollback)
        if rollback_fingerprint != source_fingerprint:
            raise RehearsalError("rollback artifact differs from source snapshot")

        conn = sqlite3.connect(candidate)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            migrated_version = migrate(conn)
            context = ConnectionContext(
                client_id="snapshot-rehearsal",
                surface="codex",
                agent_id="codex-rehearsal",
                session_id="snapshot-rehearsal",
                access_scopes=("private",),
            )
            service = MemoryWriteService(
                conn,
                _fact_writer,
                MemoryPolicy({"snapshot-rehearsal": ("private",)}),
            )
            smoke_id = _write_state(
                service,
                context,
                key="snapshot-rehearsal-state",
                model="rehearsal-only-model",
                valid_from="2099-01-01T00:00:00Z",
            )
            search_verified = smoke_id in {
                int(row["fact_id"])
                for row in search_fts(
                    conn,
                    '"rehearsal" OR "only" OR "model"',
                    allowed_scopes=("private",),
                    limit=10,
                )
            }
            evidence_verified = conn.execute(
                "SELECT 1 FROM fact_provenance WHERE fact_id = ?", (smoke_id,)
            ).fetchone() is not None
            if not search_verified or not evidence_verified:
                raise RehearsalError("integrated migrated-store smoke failed")
            if not verify_database(conn).ok:
                raise RehearsalError("migrated candidate verification failed")
        finally:
            conn.close()

        restored = restore_database(rollback, candidate, overwrite=True)
        if not restored.ok:
            raise RehearsalError("candidate rollback failed")
        restored_fingerprint = database_fingerprint(candidate)
        if restored_fingerprint != source_fingerprint:
            raise RehearsalError("restored candidate differs from source snapshot")
        with sqlite3.connect(f"file:{candidate}?mode=ro", uri=True) as conn:
            restored_version = schema_version(conn)
        unchanged = database_fingerprint(source) == source_fingerprint
        if not unchanged:
            raise RehearsalError("source snapshot changed during rehearsal")
    return SnapshotRehearsalReport(
        str(source),
        source_fingerprint,
        rollback_fingerprint,
        restored_fingerprint,
        migrated_version,
        smoke_id,
        search_verified,
        evidence_verified,
        restored_version,
        unchanged,
    )
