from __future__ import annotations

import sqlite3

import pytest

from enfold.extraction_enqueue import ExtractionEnqueuer
from enfold.extraction_processor import (
    ExtractedMemory,
    ExtractionProcessor,
    ExtractionProcessorUnavailable,
)
from enfold.policy import MemoryPolicy
from enfold.provenance import ConnectionContext
from enfold.schema import migrate
from enfold.service import EnfoldService


def _setup(tmp_path):
    conn = sqlite3.connect(tmp_path / "processor.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    context = ConnectionContext(
        client_id="hermes-install",
        surface="hermes",
        agent_id="wonny",
        session_id="hermes-session-1",
        parent_agent_id="orchestrator",
        repository="enfold",
        branch="processor",
        access_scopes=("private", "work"),
    )
    service = EnfoldService(
        conn, MemoryPolicy({"hermes-install": ("private", "work")})
    )
    return conn, context, service


class FakeExtractor:
    identity = "fake-extractor:v1"

    def __init__(self, proposals=(), failures=0):
        self.proposals = tuple(proposals)
        self.failures = failures
        self.calls = 0

    def extract(self, envelope):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary fake model failure")
        assert envelope.context.agent_id == "wonny"
        return self.proposals


def _enqueue(conn, context, *, scope="private", transcript="Victor uses Enfold."):
    return ExtractionEnqueuer(conn).enqueue_after_commit(
        context, transcript, source="session_end", scope=scope
    )


def test_fake_extraction_applies_authoritative_attributed_writes(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context, scope="work")
    extractor = FakeExtractor(
        [
            ExtractedMemory(
                "Victor uses Enfold as a shared second brain.",
                category="preference",
                tags="enfold,second-brain",
                scope="work",
                evidence_excerpt="Victor uses Enfold.",
            )
        ]
    )

    result = ExtractionProcessor(conn, service, extractor).process_one()

    assert result.outcome == "completed"
    assert result.writes == 1
    assert conn.execute("SELECT count(*) FROM extract_queue").fetchone()[0] == 0
    row = conn.execute(
        "SELECT client_id, session_id, performed_by, asserted_by, scope, metadata_json "
        "FROM observations"
    ).fetchone()
    assert tuple(row[:5]) == (
        "hermes-install",
        "hermes-session-1",
        "wonny",
        "fake-extractor:v1",
        "work",
    )
    assert '"extractor_identity":"fake-extractor:v1"' in row[5]
    session = conn.execute(
        "SELECT agent_id, parent_agent_id FROM memory_sessions"
    ).fetchone()
    assert tuple(session) == ("wonny", "orchestrator")
    conn.close()


def test_retry_then_success_and_exhaustion_dead_letters(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    clock = [100.0]
    flaky = FakeExtractor([ExtractedMemory("A durable preference.")], failures=1)
    worker = ExtractionProcessor(
        conn,
        service,
        flaky,
        max_attempts=3,
        retry_delay_seconds=5,
        clock=lambda: clock[0],
    )

    first = worker.process_one()
    assert first.outcome == "retry"
    assert worker.process_one().outcome == "idle"
    clock[0] += 5
    assert worker.process_one().outcome == "completed"

    _enqueue(conn, context, transcript="A second transcript.")
    broken = ExtractionProcessor(
        conn,
        service,
        FakeExtractor(failures=99),
        max_attempts=2,
        retry_delay_seconds=0,
        clock=lambda: clock[0],
    )
    assert broken.process_one().outcome == "retry"
    result = broken.process_one()
    assert result.outcome == "dead"
    assert result.error == "extractor_failed"
    row = conn.execute(
        "SELECT status, attempts, last_error FROM extract_queue"
    ).fetchone()
    assert tuple(row[:2]) == ("dead", 2)
    assert row[2] == "extractor_failed"
    conn.close()


def test_secret_output_is_dead_lettered_before_any_write(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    extractor = FakeExtractor(
        [ExtractedMemory("api_key = abcdefghijklmnopqrstuv")]
    )

    result = ExtractionProcessor(conn, service, extractor).process_one()

    assert result.outcome == "dead"
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert tuple(conn.execute(
        "SELECT status, attempts FROM extract_queue"
    ).fetchone()) == ("dead", 1)
    conn.close()


def test_crash_after_write_reclaims_lease_and_replays_idempotently(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    clock = [100.0]
    extractor = FakeExtractor([ExtractedMemory("Crash-safe extracted fact.")])
    crashed = ExtractionProcessor(
        conn,
        service,
        extractor,
        worker_id="worker-a",
        lease_seconds=10,
        clock=lambda: clock[0],
    )
    crashed._complete = lambda _row_id, _token: (_ for _ in ()).throw(KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        crashed.process_one()
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
    assert conn.execute("SELECT status FROM extract_queue").fetchone()[0] == "processing"

    clock[0] += 11
    recovered = ExtractionProcessor(
        conn,
        service,
        extractor,
        worker_id="worker-b",
        clock=lambda: clock[0],
    )
    assert recovered.process_one().outcome == "completed"
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 1
    conn.close()


def test_processor_refuses_minimal_enqueue_only_schema(tmp_path):
    conn = sqlite3.connect(tmp_path / "minimal.db")
    conn.execute(
        "CREATE TABLE extract_queue (id INTEGER PRIMARY KEY, payload TEXT, "
        "status TEXT, payload_hash TEXT)"
    )
    with pytest.raises(ExtractionProcessorUnavailable, match="attempts"):
        ExtractionProcessor(conn, object(), FakeExtractor())
    conn.close()


def test_fencing_token_blocks_stale_completion_with_stable_worker_id(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    clock = [100.0]
    first = ExtractionProcessor(
        conn, service, FakeExtractor(), worker_id="stable-worker",
        lease_seconds=5, clock=lambda: clock[0], max_attempts=2,
    )
    claimed = first._claim()
    assert claimed is not None
    row_id, _payload, _digest, attempts, stale_token = claimed
    assert attempts == 1

    clock[0] = 106.0
    second = ExtractionProcessor(
        conn, service, FakeExtractor(), worker_id="stable-worker",
        lease_seconds=5, clock=lambda: clock[0], max_attempts=2,
    )
    reclaimed = second._claim()
    assert reclaimed is not None
    assert reclaimed[3] == 2
    assert reclaimed[4] != stale_token
    with pytest.raises(RuntimeError, match="lease was lost"):
        first._complete(row_id, stale_token)
    second._fail(row_id, reclaimed[4], "failed", permanent=False)
    assert tuple(conn.execute(
        "SELECT status, attempts FROM extract_queue WHERE id = ?", (row_id,)
    ).fetchone()) == ("dead", 2)


def test_late_service_failure_reports_applied_writes_and_resumes_idempotently(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    extractor = FakeExtractor([
        ExtractedMemory("First durable proposal."),
        ExtractedMemory("Second durable proposal."),
    ])

    class FailSecond:
        calls = 0

        def handle(self, client_context, request):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic service interruption")
            return service.handle(client_context, request)

    partial = ExtractionProcessor(
        conn, FailSecond(), extractor, retry_delay_seconds=0
    ).process_one()
    assert partial.outcome == "retry"
    assert partial.writes == 1
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1

    resumed = ExtractionProcessor(
        conn, service, extractor, retry_delay_seconds=0
    ).process_one()
    assert resumed.outcome == "completed"
    # The first proposal replays its deterministic write; the second is added.
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM fact_provenance").fetchone()[0] == 2
