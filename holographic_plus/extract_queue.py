"""Persistent queue for LLM fact extraction.

Rows live in the same SQLite database as the fact store, so queued work
survives gateway restarts and crashes. The provider enqueues an already
formatted transcript at session end and before context compression, and a
single daemon worker drains the queue with retry and backoff. Rows that keep
failing are kept with status 'dead' for inspection instead of being silently
dropped.

The table is created lazily on first use, mirroring EmbedStore's migration
style: the parent plugin stays unaware of it and no parent schema changes
are needed.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Transcripts are capped before storage; the formatter already truncates to
# roughly this size, the cap here is a hard safety bound on row size.
MAX_PAYLOAD_BYTES = 12 * 1024

STATUS_PENDING = "pending"
STATUS_DEAD = "dead"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS extract_queue (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    payload    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    status     TEXT NOT NULL DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_extract_queue_status
    ON extract_queue(status, id);
"""


class ExtractQueue:
    """CRUD for the extract_queue table on a shared SQLite connection."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        lock: Optional["threading.RLock"] = None,
    ) -> None:
        self._conn = conn
        # Share the parent store's lock when provided so queue writes and the
        # parent's fact writes serialize on the same connection.
        self._lock = lock if lock is not None else threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def enqueue(self, payload: str) -> int:
        """Insert a pending row and return its id.

        Payloads above MAX_PAYLOAD_BYTES keep their tail, since transcripts
        put the most recent (most relevant) turns last.
        """
        encoded = payload.encode("utf-8")
        if len(encoded) > MAX_PAYLOAD_BYTES:
            payload = encoded[-MAX_PAYLOAD_BYTES:].decode("utf-8", errors="ignore")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO extract_queue (payload) VALUES (?)", (payload,)
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def mark_done(self, row_id: int) -> None:
        """Delete a successfully processed row."""
        with self._lock:
            self._conn.execute("DELETE FROM extract_queue WHERE id = ?", (row_id,))
            self._conn.commit()

    def mark_failed(self, row_id: int, error: str, max_attempts: int) -> int:
        """Record a failed attempt; mark the row dead once attempts reach the cap.

        Returns the new attempt count (0 if the row no longer exists).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT attempts FROM extract_queue WHERE id = ?", (row_id,)
            ).fetchone()
            if row is None:
                return 0
            attempts = int(row[0]) + 1
            status = STATUS_DEAD if attempts >= max_attempts else STATUS_PENDING
            self._conn.execute(
                """
                UPDATE extract_queue
                SET attempts = ?, last_error = ?, status = ?
                WHERE id = ?
                """,
                (attempts, (error or "")[:500], status, row_id),
            )
            self._conn.commit()
            return attempts

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def next_pending(
        self, max_attempts: int, exclude_ids=()
    ) -> Optional[Dict[str, Any]]:
        """Return the oldest pending row as a dict, or None when drained.

        Rows whose id is in *exclude_ids* are skipped; the worker uses this
        as an in-memory bound when recording failures in the DB is broken.

        Index-based row access so this works with any row_factory.
        """
        exclude = [int(i) for i in exclude_ids]
        sql = """
            SELECT id, payload, attempts FROM extract_queue
            WHERE status = ? AND attempts < ?
        """
        params: list = [STATUS_PENDING, max_attempts]
        if exclude:
            placeholders = ",".join("?" * len(exclude))
            sql += f" AND id NOT IN ({placeholders})"
            params.extend(exclude)
        sql += " ORDER BY id LIMIT 1"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return {"id": int(row[0]), "payload": row[1], "attempts": int(row[2])}

    def pending_count(self) -> int:
        """Number of rows still waiting to be processed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM extract_queue WHERE status = ?",
                (STATUS_PENDING,),
            ).fetchone()
        return int(row[0])

    def dead_count(self) -> int:
        """Number of rows that exhausted their retries (kept for inspection)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM extract_queue WHERE status = ?",
                (STATUS_DEAD,),
            ).fetchone()
        return int(row[0])
