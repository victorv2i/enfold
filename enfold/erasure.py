"""Explicit privacy erasure for a stopped, schema-v1 Enfold store.

Ordinary correction preserves history.  This maintenance operation is for
privacy or legal erasure and scrubs every known materialized content copy in
one transaction.  It never opens a database or chooses a path implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
import uuid

from .schema import SUPPORTED_SCHEMA_VERSION, require_compatible_schema
from .sqlite_vec_index import SQLiteVecIndex


class ErasureError(RuntimeError):
    """A requested erasure could not be completed safely."""


@dataclass(frozen=True, slots=True)
class ErasureReport:
    erasure_id: str
    fact_id: int
    affected_observations: int
    affected_embeddings: int
    affected_queue_rows: int
    invalidated_insights: int
    resolved_conflicts: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def erase_fact(
    conn: sqlite3.Connection,
    fact_id: int,
    *,
    requested_by: str,
    reason: str,
) -> ErasureReport:
    """Scrub one fact and its evidence copies without deleting audit identity."""

    if conn.in_transaction:
        raise ErasureError("privacy erasure requires an idle connection")
    if isinstance(fact_id, bool) or not isinstance(fact_id, int) or fact_id <= 0:
        raise ValueError("fact_id must be a positive integer")
    requested_by = requested_by.strip()
    reason = reason.strip()
    if not requested_by or not reason:
        raise ValueError("requested_by and reason must not be empty")
    version = require_compatible_schema(conn)
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ErasureError(
            f"privacy erasure requires schema v{SUPPORTED_SCHEMA_VERSION}; found v{version}"
        )

    row = conn.execute(
        "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    if row is None:
        raise ErasureError("fact was not found")
    original = str(row[0])
    vector_index = SQLiteVecIndex.open_configured(conn, warn=False)
    erasure_id = str(uuid.uuid4())
    erased_at = _now()
    placeholder = f"[PRIVACY ERASED fact:{fact_id}]"
    observation_ids = tuple(
        int(row[0])
        for row in conn.execute(
            "SELECT DISTINCT observation_id FROM fact_provenance WHERE fact_id = ?",
            (fact_id,),
        )
    )
    linked_queue_ids: set[int] = set()
    linked_payload_hashes: set[str] = set()
    for metadata_row in conn.execute(
        "SELECT metadata_json FROM observations WHERE observation_id IN "
        f"({','.join('?' for _ in observation_ids)})" if observation_ids else
        "SELECT metadata_json FROM observations WHERE 0",
        observation_ids,
    ):
        try:
            metadata = json.loads(str(metadata_row[0] or "{}"))
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            continue
        queue_id = metadata.get("extraction_queue_id")
        payload_hash = metadata.get("extraction_payload_sha256")
        if isinstance(queue_id, int) and queue_id > 0:
            linked_queue_ids.add(queue_id)
        if isinstance(payload_hash, str) and payload_hash:
            linked_payload_hashes.add(payload_hash)

    embeddings = 0
    queue_rows = 0
    invalidated_insights = 0
    resolved_conflicts = 0
    try:
        conn.execute("BEGIN IMMEDIATE")
        for observation_id in observation_ids:
            conn.execute(
                """
                UPDATE observations
                SET source_uri = NULL, content = ?, content_sha256 = ?,
                    asserted_by = NULL, metadata_json = '{}', redacted_at = ?
                WHERE observation_id = ?
                """,
                (
                    "[PRIVACY ERASED]",
                    f"erased:{erasure_id}:{observation_id}",
                    erased_at,
                    observation_id,
                ),
            )
        conn.execute(
            "UPDATE fact_provenance SET evidence_excerpt = NULL WHERE fact_id = ?",
            (fact_id,),
        )

        fact_columns = _columns(conn, "facts")
        assignments = [
            "content = ?", "tags = ''", "invalid_at = COALESCE(invalid_at, ?)",
            "conflict_group = NULL",
        ]
        params: list[object] = [placeholder, erased_at]
        for column in (
            "hrr_vector", "object_value", "object_entity_id", "subject_key",
            "predicate_key",
        ):
            if column in fact_columns:
                assignments.append(f"{column} = NULL")
        if "memory_kind" in fact_columns:
            assignments.append("memory_kind = 'fact'")
        conn.execute(
            f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
            (*params, fact_id),
        )
        if _table_exists(conn, "fact_entities"):
            conn.execute("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))

        for table in ("fact_embeddings", "embeddings"):
            if _table_exists(conn, table) and "fact_id" in _columns(conn, table):
                embeddings += conn.execute(
                    f'DELETE FROM "{table}" WHERE fact_id = ?', (fact_id,)
                ).rowcount
        if vector_index is not None:
            vector_index.delete_in_transaction(fact_id)

        if _table_exists(conn, "embedding_jobs"):
            conn.execute(
                """
                UPDATE embedding_jobs
                SET content_sha256 = ?, status = 'completed',
                    lease_token = NULL, lease_owner = NULL,
                    lease_expires_at = NULL, last_error = 'privacy_erased',
                    completed_at = ?, updated_at = ?
                WHERE fact_id = ?
                """,
                (f"erased:{erasure_id}:{fact_id}", erased_at, erased_at, fact_id),
            )

        if _table_exists(conn, "extract_queue"):
            queue_columns = _columns(conn, "extract_queue")
            id_column = "id" if "id" in queue_columns else "queue_id"
            if {id_column, "payload"}.issubset(queue_columns):
                rows = conn.execute(
                    f'SELECT "{id_column}", payload FROM extract_queue'
                ).fetchall()
                for queue_id, payload in rows:
                    payload_text = str(payload)
                    payload_digest = hashlib.sha256(payload_text.encode()).hexdigest()
                    stored_hash = None
                    if "payload_hash" in queue_columns:
                        stored = conn.execute(
                            f'SELECT payload_hash FROM extract_queue WHERE "{id_column}" = ?',
                            (queue_id,),
                        ).fetchone()
                        stored_hash = str(stored[0]) if stored and stored[0] else None
                    linked = (
                        int(queue_id) in linked_queue_ids
                        or payload_digest in linked_payload_hashes
                        or stored_hash in linked_payload_hashes
                    )
                    if not linked and (not original or original not in payload_text):
                        continue
                    replacement = json.dumps(
                        {"privacy_erased": True, "queue_id": int(queue_id)},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    updates = ["payload = ?"]
                    values: list[object] = [replacement]
                    if "payload_hash" in queue_columns:
                        updates.append("payload_hash = ?")
                        values.append(hashlib.sha256(replacement.encode()).hexdigest())
                    if "last_error" in queue_columns:
                        updates.append("last_error = NULL")
                    conn.execute(
                        f'UPDATE extract_queue SET {", ".join(updates)} '
                        f'WHERE "{id_column}" = ?',
                        (*values, queue_id),
                    )
                    queue_rows += 1

        # Derived insights cite source ids in tags. Invalidate them without
        # calling the legacy helper, whose internal commit would break atomicity.
        if {"category", "tags", "invalid_at"}.issubset(fact_columns):
            for insight_id, tags in conn.execute(
                "SELECT fact_id, tags FROM facts "
                "WHERE category='insight' AND invalid_at IS NULL"
            ):
                marker = "source_facts:"
                text = str(tags or "")
                cited: set[int] = set()
                if marker in text:
                    values = text.split(marker, 1)[1].split()[0].strip(",")
                    cited = {int(value) for value in values.split(",") if value.isdigit()}
                if fact_id in cited:
                    conn.execute(
                        "UPDATE facts SET invalid_at = ? WHERE fact_id = ?",
                        (erased_at, int(insight_id)),
                    )
                    invalidated_insights += 1

        if _table_exists(conn, "fact_conflict_members"):
            conflict_ids = tuple(
                str(row[0])
                for row in conn.execute(
                    "SELECT conflict_id FROM fact_conflict_members WHERE fact_id = ?",
                    (fact_id,),
                )
            )
            conn.execute(
                "DELETE FROM fact_conflict_members WHERE fact_id = ?", (fact_id,)
            )
            for conflict_id in conflict_ids:
                remaining = [
                    int(row[0])
                    for row in conn.execute(
                        "SELECT fact_id FROM fact_conflict_members WHERE conflict_id = ?",
                        (conflict_id,),
                    )
                ]
                if len(remaining) <= 1:
                    winner = remaining[0] if remaining else None
                    if winner is not None:
                        conn.execute(
                            "UPDATE facts SET conflict_group = NULL WHERE fact_id = ?",
                            (winner,),
                        )
                    conn.execute(
                        """
                        UPDATE fact_conflicts
                        SET resolved_at = ?, resolution_fact_id = ?,
                            resolved_by = ?, resolution_reason = ?
                        WHERE conflict_id = ? AND resolved_at IS NULL
                        """,
                        (
                            erased_at,
                            winner,
                            requested_by,
                            "privacy erasure removed a conflict member",
                            conflict_id,
                        ),
                    )
                    resolved_conflicts += 1

        if _table_exists(conn, "memory_write_log"):
            write_columns = _columns(conn, "memory_write_log")
            clauses: list[str] = []
            values: list[object] = []
            if "fact_id" in write_columns:
                clauses.append("fact_id = ?")
                values.append(fact_id)
            if observation_ids and "observation_id" in write_columns:
                clauses.append(
                    f"observation_id IN ({','.join('?' for _ in observation_ids)})"
                )
                values.extend(observation_ids)
            if clauses:
                conn.execute(
                    "UPDATE memory_write_log SET detail_json = ? WHERE "
                    + " OR ".join(f"({clause})" for clause in clauses),
                    ('{"privacy_erased":true}', *values),
                )
            if original:
                for write_id, detail in conn.execute(
                    "SELECT write_id, detail_json FROM memory_write_log"
                ).fetchall():
                    if original in str(detail):
                        conn.execute(
                            "UPDATE memory_write_log SET detail_json = ? WHERE write_id = ?",
                            ('{"privacy_erased":true}', write_id),
                        )

        conn.execute(
            """
            INSERT INTO privacy_erasure_log(
                erasure_id, fact_id, requested_by, reason, erased_at,
                affected_observations, affected_embeddings, affected_queue_rows
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                erasure_id,
                fact_id,
                requested_by,
                reason,
                erased_at,
                len(observation_ids),
                embeddings,
                queue_rows,
            ),
        )
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ErasureError("privacy erasure would violate foreign keys")
        indexed = conn.execute(
            "SELECT content FROM facts_fts WHERE rowid = ?", (fact_id,)
        ).fetchone()
        if indexed is not None and str(indexed[0]) != placeholder:
            raise ErasureError("FTS content did not follow privacy erasure")
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise

    return ErasureReport(
        erasure_id,
        fact_id,
        len(observation_ids),
        embeddings,
        queue_rows,
        invalidated_insights,
        resolved_conflicts,
    )
