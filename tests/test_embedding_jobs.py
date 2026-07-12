from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3

import pytest

from enfold.core_store import connect_database
from enfold.embedding_jobs import (
    EmbeddingJobProcessor,
    EmbeddingOutbox,
    EmbeddingSpec,
    SupervisedEmbeddingWorker,
)
from enfold.hybrid_retrieval import (
    SQLiteStoredEmbeddingWriter,
    SQLiteVersionedEmbeddingBackend,
    StoredEmbeddingError,
)
from enfold.policy import MemoryPolicy
from enfold.provenance import ConnectionContext, WriteRequest
from enfold.schema import migrate
from enfold.write_service import FactWriteResult, MemoryWriteService


SPEC = EmbeddingSpec("fake:model:document:none:v1", "v1", 2)


class FakeEmbedder:
    def __init__(self, vector=(1.0, 0.0), callback=None):
        self.vector = vector
        self.callback = callback
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        if self.callback is not None:
            self.callback()
        return self.vector


def _db(tmp_path):
    path = tmp_path / "jobs.db"
    conn = connect_database(path)
    migrate(conn)
    return path, conn


def _fact(conn, content="durable memory", **fields):
    values = {
        "scope": "private",
        "trust_score": 0.8,
        "schema_version": 1,
        **fields,
    }
    columns = ("content", *values)
    cursor = conn.execute(
        f"INSERT INTO facts({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        (content, *values.values()),
    )
    return int(cursor.lastrowid)


def _writer(conn, embedder):
    return SQLiteStoredEmbeddingWriter(
        conn,
        embedder,
        document_identity=SPEC.document_identity,
        embedding_version="v1",
        model_fingerprint="v1",
        prefix_policy="none",
        dimensions=SPEC.dimensions,
    )


def test_fact_and_embedding_job_commit_atomically_and_replay_is_idempotent(tmp_path):
    _path, conn = _db(tmp_path)
    outbox = EmbeddingOutbox(conn, SPEC)

    def fact_writer(connection, request, observation_id):
        del observation_id
        return FactWriteResult(_fact(connection, request.content))

    service = MemoryWriteService(
        conn,
        fact_writer,
        MemoryPolicy({"client-a": ("private",)}),
        embedding_enqueue=outbox.enqueue_in_transaction,
    )
    context = ConnectionContext("client-a", "client-a", "client-a", "session")
    request = WriteRequest("key-1", "atomic vector job", "test")

    first = service.write(context, request)
    replay = service.write(context, request)

    assert replay.replayed is True
    assert replay.fact_id == first.fact_id
    assert conn.execute("SELECT COUNT(*) FROM embedding_jobs").fetchone()[0] == 1
    row = conn.execute(
        "SELECT status, document_identity, embedding_version, dimensions "
        "FROM embedding_jobs"
    ).fetchone()
    assert tuple(row) == ("pending", SPEC.document_identity, "v1", 2)
    conn.close()


def test_outbox_failure_rolls_back_the_fact_transaction(tmp_path):
    _path, conn = _db(tmp_path)

    def fact_writer(connection, request, observation_id):
        del observation_id
        return FactWriteResult(_fact(connection, request.content))

    def fail_enqueue(_fact_id):
        raise RuntimeError("fixture enqueue failure")

    service = MemoryWriteService(
        conn,
        fact_writer,
        MemoryPolicy({"client-a": ("private",)}),
        embedding_enqueue=fail_enqueue,
    )
    with pytest.raises(RuntimeError, match="fixture enqueue failure"):
        service.write(
            ConnectionContext("client-a", "client-a", "client-a", "session"),
            WriteRequest("key-1", "must roll back", "test"),
        )
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    conn.close()


def test_backfill_processor_success_and_crash_safe_lease_reclaim(tmp_path):
    _path, conn = _db(tmp_path)
    fact_id = _fact(conn)
    conn.commit()
    outbox = EmbeddingOutbox(conn, SPEC)
    assert outbox.enqueue_backfill() == 1
    embedder = FakeEmbedder()
    processor = EmbeddingJobProcessor(
        outbox, _writer(conn, embedder), worker_id="worker-a", lease_seconds=5
    )
    start = datetime(2030, 1, 1, tzinfo=timezone.utc)
    abandoned = processor.claim(now=start)
    assert abandoned is not None

    other = EmbeddingJobProcessor(
        outbox, _writer(conn, embedder), worker_id="worker-b", lease_seconds=5
    )
    assert other.claim(now=start + timedelta(seconds=4)) is None
    result = other.process_one(now=start + timedelta(seconds=6))

    assert result is not None and result.outcome == "embedded"
    assert conn.execute(
        "SELECT COUNT(*) FROM fact_embeddings WHERE fact_id = ?", (fact_id,)
    ).fetchone()[0] == 1
    assert outbox.health()["activation_safe"] is True
    conn.close()


def test_retry_then_dead_letter_is_bounded_and_health_degrades(tmp_path):
    _path, conn = _db(tmp_path)
    _fact(conn)
    conn.commit()
    outbox = EmbeddingOutbox(conn, SPEC)
    outbox.enqueue_backfill()
    processor = EmbeddingJobProcessor(
        outbox,
        _writer(conn, FakeEmbedder(None)),
        worker_id="worker",
        max_attempts=2,
    )
    start = datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert processor.process_one(now=start).outcome == "pending"
    assert processor.process_one(now=start + timedelta(seconds=10)).outcome == "dead_letter"
    health = outbox.health()
    assert health["dead_letter"] == 1
    assert health["activation_safe"] is False
    conn.close()


def test_expired_exhausted_lease_dead_letters_without_another_model_call(tmp_path):
    _path, conn = _db(tmp_path)
    _fact(conn)
    conn.commit()
    outbox = EmbeddingOutbox(conn, SPEC)
    outbox.enqueue_backfill()
    embedder = FakeEmbedder()
    processor = EmbeddingJobProcessor(
        outbox, _writer(conn, embedder), worker_id="worker", max_attempts=1,
        lease_seconds=1,
    )
    start = datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert processor.claim(now=start) is not None
    assert processor.claim(now=start + timedelta(seconds=2)) is None
    assert embedder.calls == []
    assert outbox.health()["dead_letter"] == 1
    conn.close()


def test_final_attempt_live_lease_is_not_dead_lettered_by_second_claimer(tmp_path):
    _path, conn = _db(tmp_path)
    _fact(conn)
    conn.commit()
    outbox = EmbeddingOutbox(conn, SPEC)
    outbox.enqueue_backfill()
    start = datetime(2030, 1, 1, tzinfo=timezone.utc)
    first = EmbeddingJobProcessor(
        outbox, _writer(conn, FakeEmbedder()), worker_id="first",
        max_attempts=1, lease_seconds=30,
    )
    second = EmbeddingJobProcessor(
        outbox, _writer(conn, FakeEmbedder()), worker_id="second",
        max_attempts=1, lease_seconds=30,
    )
    claimed = first.claim(now=start)
    assert claimed is not None and claimed.attempts == 1

    assert second.claim(now=start + timedelta(seconds=10)) is None
    live = conn.execute(
        "SELECT status, lease_token, lease_owner FROM embedding_jobs"
    ).fetchone()
    assert tuple(live) == ("processing", claimed.lease_token, "first")

    assert second.claim(now=start + timedelta(seconds=31)) is None
    assert conn.execute("SELECT status FROM embedding_jobs").fetchone()[0] == "dead_letter"
    conn.close()


def test_writer_identity_rejects_unbound_model_or_prefix_configuration(tmp_path):
    _path, conn = _db(tmp_path)
    with pytest.raises(ValueError, match="fingerprint"):
        SQLiteStoredEmbeddingWriter(
            conn,
            FakeEmbedder(),
            document_identity=SPEC.document_identity,
            embedding_version="v1",
            model_fingerprint="different",
            prefix_policy="none",
            dimensions=2,
        )
    with pytest.raises(ValueError, match="none prefix"):
        SQLiteStoredEmbeddingWriter(
            conn,
            FakeEmbedder(),
            document_identity=SPEC.document_identity,
            embedding_version="v1",
            model_fingerprint="v1",
            prefix_policy="none",
            dimensions=2,
            document_prefix="different semantics",
        )
    with pytest.raises(ValueError, match="none prefix"):
        SQLiteStoredEmbeddingWriter(
            conn, FakeEmbedder(), document_identity=SPEC.document_identity,
            embedding_version="v1", model_fingerprint="v1",
            prefix_policy="none", dimensions=2, query_prefix="query: ",
        )
    conn.close()


def test_supervised_worker_heartbeat_processes_and_joins_cleanly(tmp_path):
    import time

    _path, conn = _db(tmp_path)
    _fact(conn)
    conn.commit()
    outbox = EmbeddingOutbox(conn, SPEC)
    outbox.enqueue_backfill()
    worker = SupervisedEmbeddingWorker(
        EmbeddingJobProcessor(
            outbox, _writer(conn, FakeEmbedder()), worker_id="daemon"
        ),
        poll_seconds=0.01,
    )
    worker.start()
    deadline = time.monotonic() + 1.0
    while outbox.health()["pending"] and time.monotonic() < deadline:
        time.sleep(0.01)
    health = worker.health(stale_after=1.0)
    worker.stop()
    assert health["running"] is True
    assert health["heartbeat_stale"] is False
    assert outbox.health()["pending"] == 0
    assert worker.health()["running"] is False
    conn.close()


def test_nonempty_prefix_binding_uses_one_query_document_pair_hash(tmp_path):
    _path, conn = _db(tmp_path)
    query_prefix = "query: "
    document_prefix = "document: "
    digest = hashlib.sha256(
        f"{query_prefix}\0{document_prefix}".encode("utf-8")
    ).hexdigest()
    policy = f"sha256-{digest}"
    identity = f"fake:model:document:{policy}:v1"
    spec = EmbeddingSpec(
        identity, "v1", 2, model_fingerprint="v1",
        prefix_policy=policy, query_prefix=query_prefix,
        document_prefix=document_prefix,
    )
    writer = SQLiteStoredEmbeddingWriter(
        conn, FakeEmbedder(), document_identity=identity,
        embedding_version="v1", model_fingerprint="v1",
        prefix_policy=policy, query_prefix=query_prefix,
        document_prefix=document_prefix, dimensions=2,
    )
    EmbeddingJobProcessor(
        EmbeddingOutbox(conn, spec), writer, worker_id="binding-test"
    )
    conn.close()


def test_supervisor_surfaces_processor_retry_as_last_error(tmp_path):
    import time

    _path, conn = _db(tmp_path)
    _fact(conn)
    conn.commit()
    outbox = EmbeddingOutbox(conn, SPEC)
    outbox.enqueue_backfill()
    worker = SupervisedEmbeddingWorker(
        EmbeddingJobProcessor(
            outbox, _writer(conn, FakeEmbedder(None)), worker_id="daemon"
        ),
        poll_seconds=0.01,
    )
    worker.start()
    deadline = time.monotonic() + 1
    while worker.health()["last_error"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    health = worker.health()
    worker.stop()
    assert health["last_error"] == "job_pending"
    conn.close()


@pytest.mark.parametrize("mutation", ["invalidate", "supersede", "erase"])
def test_post_model_revalidation_never_stores_stale_or_erased_content(tmp_path, mutation):
    path, conn = _db(tmp_path)
    fact_id = _fact(conn, "private content")
    replacement = _fact(conn, "replacement") if mutation == "supersede" else None
    conn.commit()
    outbox = EmbeddingOutbox(conn, SPEC)
    outbox.enqueue_backfill()

    def mutate():
        racer = sqlite3.connect(path)
        if mutation == "supersede":
            racer.execute(
                "UPDATE facts SET invalid_at = ?, superseded_by = ? WHERE fact_id = ?",
                ("2026-01-01T00:00:00Z", replacement, fact_id),
            )
        else:
            racer.execute(
                "UPDATE facts SET content = ?, invalid_at = ? WHERE fact_id = ?",
                (
                    "[PRIVACY ERASED]" if mutation == "erase" else "private content",
                    "2026-01-01T00:00:00Z",
                    fact_id,
                ),
            )
        racer.commit()
        racer.close()

    processor = EmbeddingJobProcessor(
        outbox, _writer(conn, FakeEmbedder(callback=mutate)), worker_id="worker"
    )
    result = processor.process_one()

    assert result.outcome == "skipped_stale"
    assert conn.execute(
        "SELECT COUNT(*) FROM fact_embeddings WHERE fact_id = ?", (fact_id,)
    ).fetchone()[0] == 0
    conn.close()


def test_conflict_winner_revives_completed_skip_and_gets_embedded(tmp_path):
    _path, conn = _db(tmp_path)
    fact_id = _fact(conn, conflict_group="conflict-1")
    conn.commit()
    outbox = EmbeddingOutbox(conn, SPEC)
    conn.execute("BEGIN IMMEDIATE")
    outbox.enqueue_in_transaction(fact_id)
    conn.commit()
    processor = EmbeddingJobProcessor(
        outbox, _writer(conn, FakeEmbedder()), worker_id="worker"
    )
    assert processor.process_one().outcome == "skipped_inactive"

    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE facts SET conflict_group = NULL WHERE fact_id = ?", (fact_id,))
    outbox.enqueue_in_transaction(fact_id)
    conn.commit()

    assert processor.process_one().outcome == "embedded"
    assert conn.execute(
        "SELECT COUNT(*) FROM fact_embeddings WHERE fact_id = ?", (fact_id,)
    ).fetchone()[0] == 1
    conn.close()


def test_pending_exact_job_allows_zero_dense_but_dead_or_missing_fails(tmp_path):
    _path, conn = _db(tmp_path)
    fact_id = _fact(conn)
    conn.commit()
    outbox = EmbeddingOutbox(conn, SPEC)
    outbox.enqueue_backfill()
    backend = SQLiteVersionedEmbeddingBackend(
        conn,
        FakeEmbedder(),
        query_identity="fake:model:query:none:v1",
        document_identity=SPEC.document_identity,
        embedding_version="v1",
        dimensions=2,
    )
    vectors = backend.load_documents(((fact_id, "durable memory"),))
    assert tuple(vectors[0]) == (0.0, 0.0)

    conn.execute("UPDATE embedding_jobs SET status = 'dead_letter'")
    conn.commit()
    with pytest.raises(StoredEmbeddingError, match="lack the configured"):
        SQLiteVersionedEmbeddingBackend(
            conn,
            FakeEmbedder(),
            query_identity="fake:model:query:none:v1",
            document_identity=SPEC.document_identity,
            embedding_version="v1",
            dimensions=2,
        )
    conn.close()
