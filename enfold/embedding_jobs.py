"""Durable model-free embedding outbox and explicit leased processor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3
import threading
import time
import uuid

from .hybrid_retrieval import SQLiteStoredEmbeddingWriter


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    document_identity: str
    embedding_version: str
    dimensions: int
    model_fingerprint: str = "v1"
    prefix_policy: str = "none"
    query_prefix: str = ""
    document_prefix: str = ""

    def __post_init__(self) -> None:
        if self.document_identity.count(":document:") != 1:
            raise ValueError("document identity must contain one document role")
        if not self.embedding_version.strip() or not self.document_identity.endswith(
            f":{self.embedding_version}"
        ):
            raise ValueError("embedding version must be the final identity component")
        if self.dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        if self.model_fingerprint != self.embedding_version:
            raise ValueError("model fingerprint must equal the identity version component")
        if self.prefix_policy == "none":
            if self.query_prefix or self.document_prefix:
                raise ValueError("none prefix policy requires empty query/document prefixes")
        elif self.prefix_policy.startswith("sha256-"):
            digest = hashlib.sha256(
                f"{self.query_prefix}\0{self.document_prefix}".encode("utf-8")
            ).hexdigest()
            if self.prefix_policy != f"sha256-{digest}":
                raise ValueError("document prefix does not match its fingerprint policy")
        else:
            raise ValueError("prefix policy must be none or sha256-<full digest>")
        if f":document:{self.prefix_policy}:{self.embedding_version}" not in (
            self.document_identity
        ):
            raise ValueError("document identity is not bound to prefix policy/version")


@dataclass(frozen=True, slots=True)
class ClaimedEmbeddingJob:
    job_id: int
    fact_id: int
    content_sha256: str
    attempts: int
    lease_token: str


@dataclass(frozen=True, slots=True)
class ProcessResult:
    job_id: int
    fact_id: int
    outcome: str


class EmbeddingOutbox:
    """Enqueue work inside fact transactions and inspect activation safety."""

    def __init__(self, conn: sqlite3.Connection, spec: EmbeddingSpec):
        self.conn = conn
        self.spec = spec
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(embedding_jobs)")}
        required = {
            "job_id", "fact_id", "document_identity", "embedding_version",
            "dimensions", "content_sha256", "status", "attempts",
            "available_at", "lease_token", "lease_owner", "lease_expires_at",
        }
        if not required <= columns:
            raise RuntimeError("embedding_jobs outbox is not provisioned")

    def enqueue_in_transaction(self, fact_id: int) -> int:
        if not self.conn.in_transaction:
            raise RuntimeError("embedding job enqueue must share the fact transaction")
        row = self.conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("cannot enqueue an embedding for a missing fact")
        digest = hashlib.sha256(str(row[0]).encode("utf-8")).hexdigest()
        now = _stamp(_now())
        self.conn.execute(
            """
            INSERT INTO embedding_jobs(
                fact_id, document_identity, embedding_version, dimensions,
                content_sha256, status, attempts, available_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
            ON CONFLICT(fact_id, document_identity) DO UPDATE SET
                content_sha256 = excluded.content_sha256,
                embedding_version = excluded.embedding_version,
                dimensions = excluded.dimensions,
                status = 'pending', attempts = 0,
                available_at = excluded.available_at,
                lease_token = NULL, lease_owner = NULL,
                lease_expires_at = NULL, last_error = NULL,
                completed_at = NULL, updated_at = excluded.updated_at
            WHERE embedding_jobs.status = 'completed'
              AND NOT EXISTS (
                SELECT 1 FROM fact_embeddings AS e
                WHERE e.fact_id = excluded.fact_id
                  AND e.embedding_identity = excluded.document_identity
                  AND e.dim = excluded.dimensions
              )
            """,
            (
                fact_id, self.spec.document_identity, self.spec.embedding_version,
                self.spec.dimensions, digest, now, now, now,
            ),
        )
        row = self.conn.execute(
            "SELECT job_id FROM embedding_jobs WHERE fact_id = ? AND document_identity = ?",
            (fact_id, self.spec.document_identity),
        ).fetchone()
        return int(row[0])

    def enqueue_backfill(self) -> int:
        """Queue every active fact missing this vector; never revives dead letters."""

        if self.conn.in_transaction:
            raise RuntimeError("embedding backfill requires an idle connection")
        now = _stamp(_now())
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            rows = self.conn.execute(
                """
                SELECT f.fact_id, f.content
                FROM facts AS f
                LEFT JOIN fact_embeddings AS e
                  ON e.fact_id = f.fact_id
                 AND e.embedding_identity = ? AND e.dim = ?
                WHERE f.invalid_at IS NULL AND f.superseded_by IS NULL
                  AND f.conflict_group IS NULL AND e.fact_id IS NULL
                ORDER BY f.fact_id
                """,
                (self.spec.document_identity, self.spec.dimensions),
            ).fetchall()
            changed = 0
            for fact_id, content in rows:
                digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
                cursor = self.conn.execute(
                    """
                    INSERT INTO embedding_jobs(
                        fact_id, document_identity, embedding_version, dimensions,
                        content_sha256, status, attempts, available_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                    ON CONFLICT(fact_id, document_identity) DO UPDATE SET
                        content_sha256 = excluded.content_sha256,
                        embedding_version = excluded.embedding_version,
                        dimensions = excluded.dimensions,
                        status = 'pending', attempts = 0,
                        available_at = excluded.available_at,
                        lease_token = NULL, lease_owner = NULL,
                        lease_expires_at = NULL, last_error = NULL,
                        completed_at = NULL, updated_at = excluded.updated_at
                    WHERE embedding_jobs.status = 'completed'
                    """,
                    (
                        fact_id, self.spec.document_identity,
                        self.spec.embedding_version, self.spec.dimensions,
                        digest, now, now, now,
                    ),
                )
                changed += int(cursor.rowcount > 0)
            self.conn.commit()
            return changed
        except BaseException:
            if self.conn.in_transaction:
                self.conn.rollback()
            raise

    def health(self) -> dict[str, object]:
        rows = self.conn.execute(
            """
            SELECT status, COUNT(*) FROM embedding_jobs
            WHERE document_identity = ? GROUP BY status
            """,
            (self.spec.document_identity,),
        ).fetchall()
        counts = {str(status): int(count) for status, count in rows}
        oldest = self.conn.execute(
            "SELECT MIN(created_at) FROM embedding_jobs WHERE document_identity = ? "
            "AND status IN ('pending', 'processing')",
            (self.spec.document_identity,),
        ).fetchone()[0]
        pending_age = (
            None
            if oldest is None
            else max(0.0, (_now() - datetime.fromisoformat(str(oldest))).total_seconds())
        )
        uncovered = int(self.conn.execute(
            """
            SELECT COUNT(*)
            FROM facts AS f
            LEFT JOIN fact_embeddings AS e
              ON e.fact_id = f.fact_id
             AND e.embedding_identity = ? AND e.dim = ?
            LEFT JOIN embedding_jobs AS j
              ON j.fact_id = f.fact_id AND j.document_identity = ?
             AND j.embedding_version = ? AND j.dimensions = ?
             AND j.status IN ('pending', 'processing')
            WHERE f.invalid_at IS NULL AND f.superseded_by IS NULL
              AND f.conflict_group IS NULL
              AND e.fact_id IS NULL AND j.job_id IS NULL
            """,
            (
                self.spec.document_identity, self.spec.dimensions,
                self.spec.document_identity, self.spec.embedding_version,
                self.spec.dimensions,
            ),
        ).fetchone()[0])
        return {
            "pending": counts.get("pending", 0),
            "processing": counts.get("processing", 0),
            "completed": counts.get("completed", 0),
            "dead_letter": counts.get("dead_letter", 0),
            "uncovered_active": uncovered,
            "activation_safe": uncovered == 0 and counts.get("dead_letter", 0) == 0,
            "oldest_pending_age_seconds": pending_age,
        }


class EmbeddingJobProcessor:
    """Explicit one-job leased worker. No thread is started by this class."""

    def __init__(
        self,
        outbox: EmbeddingOutbox,
        writer: SQLiteStoredEmbeddingWriter,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        max_attempts: int = 5,
    ):
        if not worker_id.strip() or lease_seconds < 1 or max_attempts < 1:
            raise ValueError("worker id, lease, and max attempts must be valid")
        if writer.document_identity != outbox.spec.document_identity:
            raise ValueError("processor writer identity does not match its outbox")
        if writer.dimensions != outbox.spec.dimensions:
            raise ValueError("processor writer dimensions do not match its outbox")
        if writer.embedding_version != outbox.spec.embedding_version:
            raise ValueError("processor writer version does not match its outbox")
        if writer.model_fingerprint != outbox.spec.model_fingerprint:
            raise ValueError("processor writer model fingerprint does not match its outbox")
        if writer.prefix_policy != outbox.spec.prefix_policy:
            raise ValueError("processor writer prefix policy does not match its outbox")
        self.outbox = outbox
        self.conn = outbox.conn
        self.writer = writer
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts

    def claim(self, *, now: datetime | None = None) -> ClaimedEmbeddingJob | None:
        if self.conn.in_transaction:
            raise RuntimeError("embedding claim requires an idle connection")
        instant = now or _now()
        stamp = _stamp(instant)
        token = uuid.uuid4().hex
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                """
                UPDATE embedding_jobs
                SET status = 'dead_letter', lease_token = NULL, lease_owner = NULL,
                    lease_expires_at = NULL, last_error = 'attempts_exhausted',
                    updated_at = ?
                WHERE document_identity = ? AND embedding_version = ?
                  AND dimensions = ? AND attempts >= ?
                  AND (
                    status = 'pending'
                    OR (status = 'processing' AND (
                        lease_expires_at IS NULL OR lease_expires_at <= ?
                    ))
                  )
                """,
                (
                    stamp, self.outbox.spec.document_identity,
                    self.outbox.spec.embedding_version,
                    self.outbox.spec.dimensions, self.max_attempts, stamp,
                ),
            )
            row = self.conn.execute(
                """
                SELECT job_id, fact_id, content_sha256, attempts
                FROM embedding_jobs
                WHERE document_identity = ? AND embedding_version = ?
                  AND dimensions = ?
                  AND attempts < ?
                  AND (
                    (status = 'pending' AND available_at <= ?)
                    OR (status = 'processing' AND lease_expires_at <= ?)
                  )
                ORDER BY job_id LIMIT 1
                """,
                (
                    self.outbox.spec.document_identity,
                    self.outbox.spec.embedding_version,
                    self.outbox.spec.dimensions, self.max_attempts, stamp, stamp,
                ),
            ).fetchone()
            if row is None:
                self.conn.commit()
                return None
            attempts = int(row[3]) + 1
            self.conn.execute(
                """
                UPDATE embedding_jobs
                SET status = 'processing', attempts = ?, lease_token = ?,
                    lease_owner = ?, lease_expires_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    attempts, token, self.worker_id,
                    _stamp(instant + timedelta(seconds=self.lease_seconds)),
                    stamp, int(row[0]),
                ),
            )
            self.conn.commit()
            return ClaimedEmbeddingJob(
                int(row[0]), int(row[1]), str(row[2]), attempts, token
            )
        except BaseException:
            if self.conn.in_transaction:
                self.conn.rollback()
            raise

    def process_one(self, *, now: datetime | None = None) -> ProcessResult | None:
        job = self.claim(now=now)
        if job is None:
            return None
        row = self.conn.execute(
            """
            SELECT content, invalid_at, superseded_by, conflict_group
            FROM facts WHERE fact_id = ?
            """,
            (job.fact_id,),
        ).fetchone()
        if row is None or any(value is not None for value in row[1:]):
            outcome = self._settle_inactive(job)
            return ProcessResult(job.job_id, job.fact_id, outcome)
        current_hash = hashlib.sha256(str(row[0]).encode("utf-8")).hexdigest()
        try:
            embedding = self.writer.embed_document(str(row[0]))
        except Exception as exc:
            outcome = self._fail(job, type(exc).__name__, now=now)
            return ProcessResult(job.job_id, job.fact_id, outcome)
        outcome = self._commit_prepared(job, current_hash, embedding)
        return ProcessResult(job.job_id, job.fact_id, outcome)

    def _settle_inactive(self, job: ClaimedEmbeddingJob) -> str:
        """Atomically skip an inactive fact, or requeue if it became active."""

        now = _stamp(_now())
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            fact = self.conn.execute(
                "SELECT content, invalid_at, superseded_by, conflict_group FROM facts "
                "WHERE fact_id = ?",
                (job.fact_id,),
            ).fetchone()
            if fact is not None and all(value is None for value in fact[1:]):
                digest = hashlib.sha256(str(fact[0]).encode("utf-8")).hexdigest()
                cursor = self.conn.execute(
                    """
                    UPDATE embedding_jobs SET status = 'pending', content_sha256 = ?,
                        available_at = ?, lease_token = NULL, lease_owner = NULL,
                        lease_expires_at = NULL, updated_at = ?
                    WHERE job_id = ? AND status = 'processing' AND lease_token = ?
                    """,
                    (digest, now, now, job.job_id, job.lease_token),
                )
                outcome = "requeued_active"
            else:
                cursor = self.conn.execute(
                    """
                    UPDATE embedding_jobs SET status = 'completed',
                        lease_token = NULL, lease_owner = NULL,
                        lease_expires_at = NULL, last_error = 'skipped_inactive',
                        completed_at = ?, updated_at = ?
                    WHERE job_id = ? AND status = 'processing' AND lease_token = ?
                    """,
                    (now, now, job.job_id, job.lease_token),
                )
                outcome = "skipped_inactive"
            if cursor.rowcount != 1:
                raise RuntimeError("embedding job lease was lost while settling")
            self.conn.commit()
            return outcome
        except BaseException:
            if self.conn.in_transaction:
                self.conn.rollback()
            raise

    def _commit_prepared(
        self, job: ClaimedEmbeddingJob, expected_hash: str, embedding: bytes
    ) -> str:
        """Revalidate lease and fact after the model call, then atomically store."""

        now = _stamp(_now())
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            lease = self.conn.execute(
                """
                SELECT 1 FROM embedding_jobs
                WHERE job_id = ? AND status = 'processing' AND lease_token = ?
                  AND document_identity = ? AND embedding_version = ?
                  AND dimensions = ?
                """,
                (
                    job.job_id, job.lease_token,
                    self.outbox.spec.document_identity,
                    self.outbox.spec.embedding_version,
                    self.outbox.spec.dimensions,
                ),
            ).fetchone()
            if lease is None:
                raise RuntimeError("embedding job lease was lost after model call")
            fact = self.conn.execute(
                "SELECT content, invalid_at, superseded_by, conflict_group FROM facts "
                "WHERE fact_id = ?",
                (job.fact_id,),
            ).fetchone()
            if fact is None or any(value is not None for value in fact[1:]):
                self.conn.execute(
                    """
                    UPDATE embedding_jobs SET status = 'completed',
                        lease_token = NULL, lease_owner = NULL,
                        lease_expires_at = NULL, last_error = 'skipped_stale',
                        completed_at = ?, updated_at = ? WHERE job_id = ?
                    """,
                    (now, now, job.job_id),
                )
                self.conn.commit()
                return "skipped_stale"
            actual_hash = hashlib.sha256(str(fact[0]).encode("utf-8")).hexdigest()
            if actual_hash != expected_hash:
                self.conn.execute(
                    """
                    UPDATE embedding_jobs SET status = 'pending', content_sha256 = ?,
                        available_at = ?, lease_token = NULL, lease_owner = NULL,
                        lease_expires_at = NULL, updated_at = ? WHERE job_id = ?
                    """,
                    (actual_hash, now, now, job.job_id),
                )
                self.conn.commit()
                return "requeued_changed"
            self.writer.upsert_in_transaction(job.fact_id, embedding)
            self.conn.execute(
                """
                UPDATE embedding_jobs SET status = 'completed', content_sha256 = ?,
                    lease_token = NULL, lease_owner = NULL,
                    lease_expires_at = NULL, last_error = NULL,
                    completed_at = ?, updated_at = ? WHERE job_id = ?
                """,
                (actual_hash, now, now, job.job_id),
            )
            self.conn.commit()
            return "embedded"
        except BaseException:
            if self.conn.in_transaction:
                self.conn.rollback()
            raise

    def _complete(
        self, job: ClaimedEmbeddingJob, outcome: str, *, content_sha256: str | None = None
    ) -> None:
        now = _stamp(_now())
        cursor = self.conn.execute(
            """
            UPDATE embedding_jobs
            SET status = 'completed', content_sha256 = COALESCE(?, content_sha256),
                lease_token = NULL, lease_owner = NULL, lease_expires_at = NULL,
                last_error = ?, completed_at = ?, updated_at = ?
            WHERE job_id = ? AND status = 'processing' AND lease_token = ?
            """,
            (content_sha256, outcome, now, now, job.job_id, job.lease_token),
        )
        if cursor.rowcount != 1:
            self.conn.rollback()
            raise RuntimeError("embedding job lease was lost before completion")
        self.conn.commit()

    def _fail(
        self, job: ClaimedEmbeddingJob, error_type: str, *, now: datetime | None
    ) -> str:
        instant = now or _now()
        dead = job.attempts >= self.max_attempts
        status = "dead_letter" if dead else "pending"
        available = instant + timedelta(seconds=min(300, 2 ** job.attempts))
        cursor = self.conn.execute(
            """
            UPDATE embedding_jobs
            SET status = ?, available_at = ?, lease_token = NULL,
                lease_owner = NULL, lease_expires_at = NULL,
                last_error = ?, updated_at = ?
            WHERE job_id = ? AND status = 'processing' AND lease_token = ?
            """,
            (
                status, _stamp(available), error_type, _stamp(instant),
                job.job_id, job.lease_token,
            ),
        )
        if cursor.rowcount != 1:
            self.conn.rollback()
            raise RuntimeError("embedding job lease was lost before failure handling")
        self.conn.commit()
        return status


class SupervisedEmbeddingWorker:
    """Bounded daemon-owned processor loop with observable lifecycle state."""

    def __init__(self, processor: EmbeddingJobProcessor, *, poll_seconds: float = 1.0,
                 drain_limit: int = 8):
        if poll_seconds <= 0 or drain_limit < 1:
            raise ValueError("worker polling configuration is invalid")
        self.processor = processor
        self.poll_seconds = poll_seconds
        self.drain_limit = drain_limit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._heartbeat: float | None = None
        self._last_success: float | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("embedding worker already started")
        self._thread = threading.Thread(
            target=self._run, name="enfold-embedding-worker", daemon=False
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self._heartbeat = time.monotonic()
            try:
                for _ in range(self.drain_limit):
                    if self._stop.is_set():
                        break
                    result = self.processor.process_one()
                    if result is None:
                        state = self.processor.outbox.health()
                        if not (
                            state["pending"]
                            or state["processing"]
                            or state["dead_letter"]
                        ):
                            with self._lock:
                                self._last_error = None
                        break
                    if result.outcome == "embedded":
                        with self._lock:
                            self._last_success = time.monotonic()
                            self._last_error = None
                    elif result.outcome in {"pending", "dead_letter"}:
                        with self._lock:
                            self._last_error = f"job_{result.outcome}"
                    else:
                        with self._lock:
                            self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = type(exc).__name__
            self._stop.wait(self.poll_seconds)

    def health(self, *, stale_after: float = 10.0) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            heartbeat = self._heartbeat
            return {
                "running": bool(self._thread and self._thread.is_alive()),
                "heartbeat_age_seconds": None if heartbeat is None else now - heartbeat,
                "heartbeat_stale": heartbeat is None or now - heartbeat > stale_after,
                "last_success_age_seconds": (
                    None if self._last_success is None else now - self._last_success
                ),
                "last_error": self._last_error,
            }

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("embedding worker did not stop cleanly")
