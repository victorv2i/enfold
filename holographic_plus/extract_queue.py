"""Persistent queue for LLM fact extraction.

Rows live in the same SQLite database as the fact store, so queued work
survives gateway restarts and crashes. The provider enqueues an already
formatted transcript at session end and before context compression, and a
single daemon worker drains the queue with retry and backoff. Rows that keep
failing are kept with status 'dead' for inspection instead of being silently
dropped.

Provider quota errors (plan-limit windows, e.g. a 429 with
``resets_in_seconds``) are special: they reset on the provider's schedule,
not ours, so they reschedule the row via ``not_before`` without consuming a
retry attempt. The 48h age cap (``created_at``) is the only thing that kills
a quota-limited row.

The table is created lazily on first use, mirroring EmbedStore's migration
style: the parent plugin stays unaware of it and no parent schema changes
are needed.
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

# Transcripts are capped before storage; the formatter already truncates to
# roughly this size, the cap here is a hard safety bound on row size.
MAX_PAYLOAD_BYTES = 12 * 1024

STATUS_PENDING = "pending"
STATUS_DEAD = "dead"

# Failure messages matching any of these substrings (case-insensitive) are
# classified as provider quota / rate-limit errors and rescheduled instead of
# consuming retry attempts.
QUOTA_ERROR_PATTERNS = (
    "429",
    "usage_limit_reached",
    "usage_limit",
    "rate limit",
    "quota",
)

# Jitter added on top of a parsed reset window so retries do not land exactly
# on the reset boundary, and the fallback delay when a quota error does not
# say when its window reopens.
QUOTA_JITTER_MIN = 60.0        # seconds
QUOTA_JITTER_MAX = 300.0       # seconds
QUOTA_FALLBACK_DELAY = 1800.0  # seconds

# Rows older than this (by created_at) are marked dead on their next failure
# regardless of error classification; quota-limited rows therefore retry
# patiently for up to 48h and no longer.
MAX_ROW_AGE_SECONDS = 48 * 3600

_AGE_CAP_NOTE = " (dead: 48h age cap)"

_RESETS_IN_RE = re.compile(r"resets_in_seconds\D{0,12}(\d+)", re.IGNORECASE)
_RESETS_AT_RE = re.compile(r"resets_at\D{0,12}(\d{9,12})", re.IGNORECASE)


def is_quota_error(error: Optional[str]) -> bool:
    """True when a failure message looks like a provider quota / rate limit."""
    text = (error or "").lower()
    return any(pattern in text for pattern in QUOTA_ERROR_PATTERNS)


def quota_retry_delay(error: Optional[str], now: Optional[float] = None) -> float:
    """Seconds to wait before retrying a quota-limited row.

    Parses ``resets_in_seconds`` (or a ``resets_at`` epoch) from the error
    text and adds a small jitter. Falls back to QUOTA_FALLBACK_DELAY when the
    error does not say when the window reopens.
    """
    text = error or ""
    match = _RESETS_IN_RE.search(text)
    if match:
        return float(match.group(1)) + random.uniform(QUOTA_JITTER_MIN, QUOTA_JITTER_MAX)
    match = _RESETS_AT_RE.search(text)
    if match:
        current = time.time() if now is None else now
        remaining = max(0.0, float(match.group(1)) - current)
        return remaining + random.uniform(QUOTA_JITTER_MIN, QUOTA_JITTER_MAX)
    return QUOTA_FALLBACK_DELAY


_SCHEMA = """
CREATE TABLE IF NOT EXISTS extract_queue (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    payload    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    status     TEXT NOT NULL DEFAULT 'pending',
    not_before REAL
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
            self._ensure_not_before_column()
            self._conn.commit()

    def _ensure_not_before_column(self) -> None:
        """Lazy migration: add not_before to tables created before quota retries.

        Mirrors EmbedStore's migration style: a PRAGMA table_info guard, then
        a single ALTER TABLE ... ADD COLUMN. Idempotent and safe on every
        startup; new tables already have the column from _SCHEMA.
        """
        info = self._conn.execute("PRAGMA table_info(extract_queue)").fetchall()
        cols = {row[1] for row in info}
        if "not_before" not in cols:
            try:
                self._conn.execute("ALTER TABLE extract_queue ADD COLUMN not_before REAL")
            except sqlite3.OperationalError as exc:
                # Two processes racing this same check-then-add on a fresh db
                # (e.g. two MCP server instances starting at once): the
                # loser's ALTER TABLE is a no-op, not a real failure.
                if "duplicate column name" not in str(exc).lower():
                    raise

    @staticmethod
    def _age_exceeded(created_epoch) -> bool:
        """True when a row's created_at (epoch string/float) is past the age cap."""
        if created_epoch is None:
            return False
        return time.time() - float(created_epoch) >= MAX_ROW_AGE_SECONDS

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

        Rows past MAX_ROW_AGE_SECONDS are marked dead regardless of the
        attempt count, with last_error noting the age cap.

        Returns the new attempt count (0 if the row no longer exists).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT attempts, strftime('%s', created_at) FROM extract_queue WHERE id = ?",
                (row_id,),
            ).fetchone()
            if row is None:
                return 0
            attempts = int(row[0]) + 1
            message = (error or "")[:500]
            if self._age_exceeded(row[1]):
                status = STATUS_DEAD
                message += _AGE_CAP_NOTE
            else:
                status = STATUS_DEAD if attempts >= max_attempts else STATUS_PENDING
            self._conn.execute(
                """
                UPDATE extract_queue
                SET attempts = ?, last_error = ?, status = ?
                WHERE id = ?
                """,
                (attempts, message, status, row_id),
            )
            self._conn.commit()
            return attempts

    def mark_quota_failed(self, row_id: int, error: str, not_before: float) -> bool:
        """Reschedule a quota-limited row without consuming a retry attempt.

        The row stays pending and next_pending() skips it until *not_before*
        (epoch seconds). Rows past MAX_ROW_AGE_SECONDS are marked dead with
        last_error noting the age cap.

        Returns True when the row was rescheduled, False when it went dead by
        the age cap or no longer exists.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT strftime('%s', created_at) FROM extract_queue WHERE id = ?",
                (row_id,),
            ).fetchone()
            if row is None:
                return False
            message = (error or "")[:500]
            if self._age_exceeded(row[0]):
                self._conn.execute(
                    """
                    UPDATE extract_queue
                    SET last_error = ?, status = ?, not_before = NULL
                    WHERE id = ?
                    """,
                    (message + _AGE_CAP_NOTE, STATUS_DEAD, row_id),
                )
                self._conn.commit()
                return False
            self._conn.execute(
                "UPDATE extract_queue SET last_error = ?, not_before = ? WHERE id = ?",
                (message, float(not_before), row_id),
            )
            self._conn.commit()
            return True

    def revive_dead(self, ids: Optional[Iterable[int]] = None) -> int:
        """Reset dead rows to pending so the worker retries them.

        Attempts go back to 0 and not_before is cleared; the last error is
        kept with ' (revived)' appended for inspection. With *ids* of None
        every dead row is revived, otherwise only the listed ids.

        Returns the number of rows revived.
        """
        sql = """
            UPDATE extract_queue
            SET status = ?, attempts = 0, not_before = NULL,
                last_error = COALESCE(last_error, '') || ' (revived)'
            WHERE status = ?
        """
        params: list = [STATUS_PENDING, STATUS_DEAD]
        if ids is not None:
            id_list = [int(i) for i in ids]
            if not id_list:
                return 0
            placeholders = ",".join("?" * len(id_list))
            sql += f" AND id IN ({placeholders})"
            params.extend(id_list)
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return int(cur.rowcount)

    def revive_recent_quota_dead(self) -> int:
        """Revive dead rows that were killed by quota errors and are still young.

        One-shot recovery for rows dead-lettered before quota errors stopped
        consuming attempts: any dead row younger than MAX_ROW_AGE_SECONDS
        whose last_error matches QUOTA_ERROR_PATTERNS goes back to pending.

        Returns the number of rows revived.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, last_error FROM extract_queue
                WHERE status = ?
                  AND (strftime('%s', 'now') - strftime('%s', created_at)) < ?
                """,
                (STATUS_DEAD, int(MAX_ROW_AGE_SECONDS)),
            ).fetchall()
        quota_ids = [int(row[0]) for row in rows if is_quota_error(row[1])]
        if not quota_ids:
            return 0
        return self.revive_dead(quota_ids)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def next_pending(
        self, max_attempts: int, exclude_ids=()
    ) -> Optional[Dict[str, Any]]:
        """Return the oldest due pending row as a dict, or None when drained.

        Rows whose not_before is in the future (quota reschedules) are
        skipped until due. Rows whose id is in *exclude_ids* are skipped; the
        worker uses this as an in-memory bound when recording failures in the
        DB is broken.

        Index-based row access so this works with any row_factory.
        """
        exclude = [int(i) for i in exclude_ids]
        sql = """
            SELECT id, payload, attempts FROM extract_queue
            WHERE status = ? AND attempts < ?
              AND (not_before IS NULL OR not_before <= ?)
        """
        params: list = [STATUS_PENDING, max_attempts, time.time()]
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
