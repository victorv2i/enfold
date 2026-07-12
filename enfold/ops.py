"""Explicit Enfold schema, backup, verification, and restore operations.

This module is intentionally separate from provider initialization.  It never
chooses a database path implicitly and requires an explicit maintenance-window
override before migrating or restoring anything below a ``.hermes`` directory.

Run with ``python -m enfold.ops --help``.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .backup import (
    BackupError,
    backup_database,
    maintenance_database_lock,
    restore_database,
    sqlite_file_uri,
    verify_database,
)
from .erasure import ErasureError, erase_fact
from .rehearsal import RehearsalError, rehearse_snapshot
from .schema import (
    SUPPORTED_SCHEMA_VERSION,
    SchemaError,
    migrate,
    require_compatible_schema,
)
from .server import load_config
from .sqlite_vec_index import SQLiteVecError, rebuild_sqlite_vec_index


LIVE_PATH_MESSAGE = (
    "refusing to modify a database under .hermes; stop all Hermes, Codex, "
    "Claude, MCP, and Enfold writers during a maintenance window, then pass "
    "--allow-live explicitly"
)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_under_hermes(path: str | Path) -> bool:
    return ".hermes" in _resolved(path).parts


def _require_live_override(path: str | Path, *, allow_live: bool) -> None:
    if _is_under_hermes(path) and not allow_live:
        raise BackupError(LIVE_PATH_MESSAGE)


def _connect(path: str | Path, *, read_only: bool) -> sqlite3.Connection:
    resolved = _resolved(path)
    if not resolved.is_file():
        raise BackupError(f"database does not exist: {resolved}")
    mode = "ro" if read_only else "rw"
    return sqlite3.connect(sqlite_file_uri(resolved, mode=mode), uri=True)


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _schema_status(args: argparse.Namespace) -> None:
    with _connect(args.database, read_only=True) as conn:
        version = require_compatible_schema(conn)
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "schema_version": version,
            "supported_schema_version": SUPPORTED_SCHEMA_VERSION,
            "compatible": True,
        }
    )


def _migrate(args: argparse.Namespace) -> None:
    _require_live_override(args.database, allow_live=args.allow_live)
    with maintenance_database_lock(args.database):
        with _connect(args.database, read_only=False) as conn:
            before = require_compatible_schema(conn)
            after = migrate(conn, target_version=args.target)
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "schema_version_before": before,
            "schema_version_after": after,
        }
    )


def _backup(args: argparse.Namespace) -> None:
    report = backup_database(
        args.source,
        args.destination,
        overwrite=args.overwrite,
        secondary_directory=args.secondary_directory,
        age_recipient_path=args.age_recipient_path,
    )
    _print_json(
        {
            "source": str(_resolved(args.source)),
            "destination": str(_resolved(args.destination)),
            "report": asdict(report),
            "ok": report.ok,
        }
    )


def _verify(args: argparse.Namespace) -> None:
    # FTS5's integrity command uses INSERT syntax even though it rolls back to
    # a savepoint. Keep ordinary verification genuinely read-only; the fuller
    # FTS check is an explicit maintenance operation.
    if args.check_fts:
        _require_live_override(args.database, allow_live=args.allow_live)
    if args.check_fts:
        with maintenance_database_lock(args.database):
            with _connect(args.database, read_only=False) as conn:
                report = verify_database(conn, check_fts=True)
    else:
        with _connect(args.database, read_only=True) as conn:
            report = verify_database(conn, check_fts=False)
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "report": asdict(report),
            "ok": report.ok,
        }
    )
    if not report.ok:
        raise BackupError("database verification failed")


def _restore(args: argparse.Namespace) -> None:
    _require_live_override(args.destination, allow_live=args.allow_live)
    with maintenance_database_lock(args.destination):
        report = restore_database(
            args.backup, args.destination, overwrite=args.overwrite
        )
    _print_json(
        {
            "backup": str(_resolved(args.backup)),
            "destination": str(_resolved(args.destination)),
            "report": asdict(report),
            "ok": report.ok,
        }
    )


def _erase_fact(args: argparse.Namespace) -> None:
    _require_live_override(args.database, allow_live=args.allow_live)
    with maintenance_database_lock(args.database):
        with _connect(args.database, read_only=False) as conn:
            report = erase_fact(
                conn,
                args.fact_id,
                requested_by=args.requested_by,
                reason=args.reason,
            )
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "report": asdict(report),
            "ok": True,
        }
    )


def _rehearse(args: argparse.Namespace) -> None:
    report = rehearse_snapshot(args.snapshot, args.workdir)
    _print_json({"report": asdict(report), "ok": True})


def _browse_metadata(scopes: tuple[str, ...]) -> dict[str, object]:
    return {
        "title": "Enfold Second Brain",
        "databases": {
            "browse-snapshot": {
                "tables": {"facts": {"fts_table": "facts_fts"}}
            }
        },
        "scope_allowlist": list(scopes),
        "filters": {"lifecycle": "settled_active", "sensitivity": "normal_only"},
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _browse_snapshot(args: argparse.Namespace) -> None:
    """Copy approved current facts from a read-only live snapshot into SQLite."""

    config = load_config(args.config, allow_live=True)
    destination = _resolved(
        args.destination or "~/.local/state/enfold/browse/browse-snapshot.db"
    )
    metadata_path = destination.with_name("metadata.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = _connect(config.database_path, read_only=True)
    temporary_source = tempfile.NamedTemporaryFile(
        prefix="enfold-browse-source-", suffix=".db", delete=False
    )
    temporary_source.close()
    temporary_destination = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".db", dir=destination.parent, delete=False
    )
    temporary_destination.close()
    try:
        with sqlite3.connect(temporary_source.name) as snapshot:
            source.backup(snapshot)
        source.close()
        source = None
        with sqlite3.connect(temporary_source.name) as snapshot, sqlite3.connect(
            temporary_destination.name
        ) as browse:
            require_compatible_schema(snapshot)
            browse.executescript(
                """
                CREATE TABLE facts (
                    fact_id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    trust_score REAL,
                    memory_kind TEXT,
                    subject_key TEXT,
                    predicate_key TEXT,
                    object_value TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    scope TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE facts_fts USING fts5(content);
                """
            )
            placeholders = ",".join("?" for _ in config.browse_scopes)
            rows = snapshot.execute(
                f"""
                SELECT fact_id, content, COALESCE(category, 'general'), COALESCE(tags, ''),
                       trust_score, memory_kind, subject_key, predicate_key,
                       object_value, created_at, updated_at, scope
                FROM facts
                WHERE invalid_at IS NULL AND superseded_by IS NULL
                  AND conflict_group IS NULL
                  AND COALESCE(sensitivity, 'normal') = 'normal'
                  AND scope IN ({placeholders})
                ORDER BY fact_id
                """,
                config.browse_scopes,
            )
            facts = list(rows)
            browse.executemany(
                "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", facts
            )
            browse.executemany(
                "INSERT INTO facts_fts(rowid, content) VALUES (?, ?)",
                ((row[0], row[1]) for row in facts),
            )
        os.chmod(temporary_destination.name, 0o400)
        os.replace(temporary_destination.name, destination)
        _atomic_json(metadata_path, _browse_metadata(config.browse_scopes))
    finally:
        if source is not None:
            source.close()
        Path(temporary_source.name).unlink(missing_ok=True)
        Path(temporary_destination.name).unlink(missing_ok=True)
    _print_json(
        {
            "database": str(destination),
            "metadata": str(metadata_path),
            "scope_allowlist": list(config.browse_scopes),
            "ok": True,
        }
    )


def _rebuild_vector_index(args: argparse.Namespace) -> None:
    _require_live_override(args.database, allow_live=args.allow_live)
    with maintenance_database_lock(args.database):
        with _connect(args.database, read_only=False) as conn:
            require_compatible_schema(conn)
            report = rebuild_sqlite_vec_index(
                conn, args.embedding_identity, args.dimensions
            )
    _print_json({
        "database": str(_resolved(args.database)),
        "report": asdict(report),
        "ok": True,
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser(
        "schema-status", help="inspect schema compatibility without modifying it"
    )
    status.add_argument("database", help="explicit SQLite database path")
    status.set_defaults(handler=_schema_status)

    migration = commands.add_parser(
        "migrate", help="explicitly apply registered schema migrations"
    )
    migration.add_argument("database", help="explicit SQLite database path")
    migration.add_argument(
        "--target", type=int, default=SUPPORTED_SCHEMA_VERSION, help="target version"
    )
    migration.add_argument(
        "--allow-live",
        action="store_true",
        help="allow a .hermes path after all writers are stopped for maintenance",
    )
    migration.set_defaults(handler=_migrate)

    backup = commands.add_parser(
        "backup", help="create a verified backup using SQLite's backup API"
    )
    backup.add_argument("source", help="explicit source SQLite database path")
    backup.add_argument("destination", help="explicit destination backup path")
    backup.add_argument("--overwrite", action="store_true")
    backup.add_argument(
        "--secondary-directory",
        help="best-effort secondary destination directory",
    )
    backup.add_argument(
        "--age-recipient-path",
        help="age recipients or identity file used with age -R",
    )
    backup.set_defaults(handler=_backup)

    verify = commands.add_parser(
        "verify", help="run integrity, foreign-key, FTS, and row-count checks"
    )
    verify.add_argument("database", help="explicit SQLite database path")
    verify.add_argument(
        "--check-fts",
        action="store_true",
        help="run the FTS5 write-syntax integrity command inside a rollback",
    )
    verify.add_argument(
        "--allow-live",
        action="store_true",
        help="allow --check-fts under .hermes during a maintenance window",
    )
    verify.set_defaults(handler=_verify)

    restore = commands.add_parser(
        "restore", help="restore a verified backup using SQLite's backup API"
    )
    restore.add_argument("backup", help="explicit source backup path")
    restore.add_argument("destination", help="explicit restore destination path")
    restore.add_argument("--overwrite", action="store_true")
    restore.add_argument(
        "--allow-live",
        action="store_true",
        help="allow a .hermes path after all writers are stopped for maintenance",
    )
    restore.set_defaults(handler=_restore)

    erasure = commands.add_parser(
        "erase-fact",
        help="privacy/legal erasure of a fact and all known content copies",
    )
    erasure.add_argument("database", help="explicit schema-v1 SQLite database path")
    erasure.add_argument("fact_id", type=int)
    erasure.add_argument("--requested-by", required=True)
    erasure.add_argument("--reason", required=True)
    erasure.add_argument(
        "--allow-live",
        action="store_true",
        help="allow a .hermes path after every writer is stopped",
    )
    erasure.set_defaults(handler=_erase_fact)

    rehearsal = commands.add_parser(
        "rehearse",
        help="backup/migrate/smoke/restore an explicit offline snapshot copy",
    )
    rehearsal.add_argument("snapshot", help="offline snapshot outside .hermes")
    rehearsal.add_argument("workdir", help="empty artifact directory outside .hermes")
    rehearsal.set_defaults(handler=_rehearse)

    browse = commands.add_parser(
        "browse-snapshot",
        help="build a policy-filtered SQLite snapshot for a local Datasette browser",
    )
    browse.add_argument("config", help="explicit Enfold server JSON configuration")
    browse.add_argument(
        "--destination",
        help="snapshot path (default: ~/.local/state/enfold/browse/browse-snapshot.db)",
    )
    browse.set_defaults(handler=_browse_snapshot)

    vector_rebuild = commands.add_parser(
        "rebuild-vector-index",
        help="atomically rebuild sqlite-vec from canonical fact embeddings",
    )
    vector_rebuild.add_argument("database", help="explicit schema-v1 SQLite database path")
    vector_rebuild.add_argument("--embedding-identity", required=True)
    vector_rebuild.add_argument("--dimensions", required=True, type=int)
    vector_rebuild.add_argument(
        "--allow-live", action="store_true",
        help="allow a .hermes path after every writer is stopped",
    )
    vector_rebuild.set_defaults(handler=_rebuild_vector_index)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (
        BackupError,
        ErasureError,
        RehearsalError,
        SchemaError,
        SQLiteVecError,
        sqlite3.DatabaseError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
