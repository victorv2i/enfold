"""Daemon-owned, model-free extraction queue boundary.

This module deliberately does not create or migrate the queue and never calls
an extraction model.  An explicit migration/adapter must provide the durable
``extract_queue`` table.  Enqueue happens only after the caller's fact write
transaction has committed, preserving the write-path latency and rollback
contract while retaining payload-hash idempotency.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sqlite3
from typing import Any, Mapping

from .provenance import ConnectionContext


MAX_EXTRACTION_PAYLOAD_BYTES = 12 * 1024
_REQUIRED_COLUMNS = frozenset({"id", "payload", "status", "payload_hash"})


class ExtractionQueueUnavailable(RuntimeError):
    """The explicitly provisioned durable extraction queue is unavailable."""


@dataclass(frozen=True, slots=True)
class ExtractionEnqueueResult:
    queue_id: int
    payload_sha256: str
    replayed: bool


class ExtractionEnqueuer:
    """Append canonical attributed transcripts to an existing durable queue."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(extract_queue)")
        }
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ExtractionQueueUnavailable(
                "extract_queue is not provisioned with required columns: "
                + ", ".join(missing)
            )

    def enqueue_after_commit(
        self,
        context: ConnectionContext,
        transcript: str,
        *,
        source: str,
        scope: str = "private",
        metadata: Mapping[str, Any] | None = None,
    ) -> ExtractionEnqueueResult:
        """Enqueue one transcript without running a model.

        The connection must be idle.  This makes ordering explicit: the
        authoritative write commits first; queue insertion is a separate,
        idempotent transaction and can be retried safely after a crash.
        """

        if self._conn.in_transaction:
            raise RuntimeError("extraction enqueue must run after commit")
        transcript = transcript.strip()
        source = source.strip()
        scope = scope.strip()
        if not transcript or not source or not scope:
            raise ValueError("transcript, source, and scope must be non-empty")
        if scope not in context.access_scopes:
            raise ValueError("extraction scope must be present in context access scopes")
        envelope = {
            "version": 1,
            "transcript": transcript,
            "source": source,
            "scope": scope,
            "provenance": {
                "client_id": context.client_id,
                "surface": context.surface,
                "agent_id": context.agent_id,
                "session_id": context.session_id,
                "parent_agent_id": context.parent_agent_id,
                "project_root": context.project_root,
                "repository": context.repository,
                "branch": context.branch,
                "commit_sha": context.commit_sha,
                "access_scopes": list(context.access_scopes),
            },
            "metadata": dict(metadata or {}),
        }
        try:
            payload = json.dumps(
                envelope,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("extraction metadata must contain JSON values") from exc
        if len(payload.encode("utf-8")) > MAX_EXTRACTION_PAYLOAD_BYTES:
            raise ValueError("canonical extraction payload exceeds size limit")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            existing = self._conn.execute(
                """
                SELECT id FROM extract_queue
                WHERE payload_hash = ? AND status IN ('pending', 'processing')
                ORDER BY id LIMIT 1
                """,
                (digest,),
            ).fetchone()
            if existing is not None:
                self._conn.commit()
                return ExtractionEnqueueResult(int(existing[0]), digest, True)
            cursor = self._conn.execute(
                "INSERT INTO extract_queue(payload, payload_hash, status) "
                "VALUES (?, ?, 'pending')",
                (payload, digest),
            )
            queue_id = int(cursor.lastrowid)
            self._conn.commit()
            return ExtractionEnqueueResult(queue_id, digest, False)
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
