"""Persistent extraction queue: unit behavior, worker drain, retry, restart."""

import json
import sqlite3
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import fake_hermes
import pytest

MESSAGES = [
    {"role": "user", "content": "I always use pnpm for node projects, remember that."},
    {"role": "assistant", "content": "Noted, pnpm is your package manager of choice."},
    {"role": "user", "content": "Also the deploy target for my web apps is vercel."},
    {"role": "assistant", "content": "Got it, vercel is the deploy target for web apps."},
]

FACTS_JSON = json.dumps([
    {"content": "The user uses pnpm as their package manager for node projects.",
     "category": "tool", "tags": "pnpm,node"},
    {"content": "The user deploys web apps to vercel.",
     "category": "tool", "tags": "deploy,vercel"},
    {"content": "The dashboard rewrite is an active project.",
     "category": "project", "tags": "dashboard"},
])


def _llm_response(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


def _extraction_cfg():
    return {"extraction_provider": "testprov", "extraction_model": "testmodel"}


# ---------------------------------------------------------------------------
# ExtractQueue unit behavior
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_queue(hp, tmp_path):
    conn = sqlite3.connect(str(tmp_path / "queue.db"), check_same_thread=False)
    queue = hp.extract_queue.ExtractQueue(conn)
    yield queue
    conn.close()


def test_enqueue_and_fifo_pending(raw_queue):
    first = raw_queue.enqueue("payload one")
    second = raw_queue.enqueue("payload two")
    assert raw_queue.pending_count() == 2
    row = raw_queue.next_pending(max_attempts=5)
    assert row["id"] == first
    assert row["payload"] == "payload one"
    raw_queue.mark_done(first, row["lease_owner"])
    assert raw_queue.next_pending(max_attempts=5)["id"] == second


def test_identical_active_payload_is_idempotent(raw_queue):
    first = raw_queue.enqueue("same transcript")
    duplicate = raw_queue.enqueue("same transcript")

    assert duplicate == first
    assert raw_queue.pending_count() == 1
    assert raw_queue._conn.execute(
        "SELECT COUNT(*) FROM extract_queue"
    ).fetchone()[0] == 1


def test_identical_concurrent_enqueue_across_connections(hp, tmp_path):
    db_path = tmp_path / "overlap.db"
    connections = [
        sqlite3.connect(str(db_path), check_same_thread=False, timeout=5.0)
        for _ in range(2)
    ]
    queues = [hp.extract_queue.ExtractQueue(conn) for conn in connections]
    barrier = threading.Barrier(2)

    def enqueue(queue):
        barrier.wait()
        return queue.enqueue("overlapping transcript")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            ids = list(pool.map(enqueue, queues))
        assert ids[0] == ids[1]
        assert connections[0].execute(
            "SELECT COUNT(*) FROM extract_queue"
        ).fetchone()[0] == 1
    finally:
        for conn in connections:
            conn.close()


def test_queue_additive_hash_setup_preserves_legacy_duplicates(hp, tmp_path):
    conn = sqlite3.connect(str(tmp_path / "legacy-queue.db"), check_same_thread=False)
    conn.executescript(
        """
        CREATE TABLE extract_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            not_before REAL,
            lease_owner TEXT,
            lease_until REAL
        );
        INSERT INTO extract_queue(payload) VALUES ('legacy duplicate');
        INSERT INTO extract_queue(payload) VALUES ('legacy duplicate');
        """
    )
    queue = hp.extract_queue.ExtractQueue(conn)
    try:
        assert queue.enqueue("legacy duplicate") == 1
        assert conn.execute("SELECT COUNT(*) FROM extract_queue").fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(payload_hash) FROM extract_queue"
        ).fetchone()[0] == 1
        first = queue.next_pending(max_attempts=5, lease_owner="worker")
        assert first["id"] == 1
        assert queue.mark_done(first["id"], "worker") is True
        # The remaining legacy duplicate has a NULL hash, but payload lookup
        # still prevents a third active copy from being created.
        assert queue.enqueue("legacy duplicate") == 2
        assert conn.execute("SELECT COUNT(*) FROM extract_queue").fetchone()[0] == 1
    finally:
        conn.close()


def test_next_pending_claims_row_with_lease(raw_queue):
    row_id = raw_queue.enqueue("payload one")

    first = raw_queue.next_pending(max_attempts=5, lease_owner="worker-a")
    second = raw_queue.next_pending(max_attempts=5, lease_owner="worker-b")

    assert first["id"] == row_id
    assert first["lease_owner"] == "worker-a"
    assert first["lease_until"] > time.time()
    assert second is None
    stored = raw_queue._conn.execute(
        "SELECT status, lease_owner, lease_until FROM extract_queue WHERE id = ?",
        (row_id,),
    ).fetchone()
    assert stored[0] == "processing"
    assert stored[1] == "worker-a"
    assert stored[2] == first["lease_until"]


def test_expired_processing_lease_is_reclaimable(raw_queue):
    row_id = raw_queue.enqueue("payload one")
    first = raw_queue.next_pending(
        max_attempts=5, lease_owner="worker-a", lease_seconds=0.01
    )
    assert first["id"] == row_id
    time.sleep(0.02)

    reclaimed = raw_queue.next_pending(max_attempts=5, lease_owner="worker-b")

    assert reclaimed["id"] == row_id
    assert reclaimed["lease_owner"] == "worker-b"
    assert reclaimed["attempts"] == 0


def test_mark_done_requires_matching_lease_owner(raw_queue):
    row_id = raw_queue.enqueue("payload one")
    raw_queue.next_pending(max_attempts=5, lease_owner="worker-a")

    assert raw_queue.mark_done(row_id, "worker-b") is False
    assert raw_queue.pending_count() == 1
    assert raw_queue.mark_done(row_id, "worker-a") is True
    assert raw_queue.pending_count() == 0


def test_payload_capped_to_12kb_keeping_tail(hp, raw_queue):
    big = ("x" * 1000 + "\n") * 20  # well over the cap
    row_id = raw_queue.enqueue(big)
    row = raw_queue.next_pending(max_attempts=5)
    assert row["id"] == row_id
    stored = row["payload"]
    assert len(stored.encode("utf-8")) <= hp.extract_queue.MAX_PAYLOAD_BYTES
    assert big.endswith(stored)


def test_mark_failed_increments_and_marks_dead(raw_queue):
    row_id = raw_queue.enqueue("doomed payload")
    row = raw_queue.next_pending(max_attempts=3)
    assert raw_queue.mark_failed(
        row_id, "boom 1", max_attempts=3, lease_owner=row["lease_owner"]
    ) == 1
    row = raw_queue.next_pending(max_attempts=3)
    assert raw_queue.mark_failed(
        row_id, "boom 2", max_attempts=3, lease_owner=row["lease_owner"]
    ) == 2
    assert raw_queue.pending_count() == 1
    row = raw_queue.next_pending(max_attempts=3)
    assert raw_queue.mark_failed(
        row_id, "boom 3", max_attempts=3, lease_owner=row["lease_owner"]
    ) == 3
    assert raw_queue.pending_count() == 0
    assert raw_queue.dead_count() == 1
    assert raw_queue.next_pending(max_attempts=3) is None
    # Dead rows keep their last error for inspection
    row = raw_queue._conn.execute(
        "SELECT status, last_error, attempts FROM extract_queue WHERE id = ?", (row_id,)
    ).fetchone()
    assert row[0] == "dead"
    assert row[1] == "boom 3"
    assert row[2] == 3


# ---------------------------------------------------------------------------
# Provider hooks enqueue and the worker drains
# ---------------------------------------------------------------------------

def test_session_end_enqueues_and_worker_inserts_facts(make_provider, aux_module, waiter):
    aux_module.call_llm = lambda **kwargs: _llm_response(FACTS_JSON)
    provider = make_provider(**_extraction_cfg())

    provider.on_session_end(list(MESSAGES))

    assert waiter(lambda: provider._extract_queue.pending_count() == 0)
    assert waiter(
        lambda: len(provider._store.list_facts(min_trust=0.0, limit=50)) == 3
    )
    contents = {f["content"] for f in provider._store.list_facts(min_trust=0.0, limit=50)}
    assert "The user deploys web apps to vercel." in contents


def test_pre_compress_enqueues_and_returns_immediately(make_provider, aux_module, waiter):
    aux_module.call_llm = lambda **kwargs: _llm_response(FACTS_JSON)
    provider = make_provider(**_extraction_cfg())

    result = provider.on_pre_compress(list(MESSAGES))
    assert result == ""
    assert waiter(
        lambda: len(provider._store.list_facts(min_trust=0.0, limit=50)) == 3
    )


def test_pre_compress_skips_trivial_conversations(make_provider):
    provider = make_provider(**_extraction_cfg())
    assert provider.on_pre_compress([{"role": "user", "content": "hi"}]) == ""
    assert provider._extract_queue.pending_count() == 0


def test_nothing_enqueued_without_extraction_model(make_provider):
    provider = make_provider()  # no extraction_provider/model configured
    provider.on_session_end(list(MESSAGES))
    assert provider._extract_queue.pending_count() == 0


def test_failed_extraction_retries_then_marks_dead(make_provider, aux_module, waiter):
    def failing(**kwargs):
        raise RuntimeError("backend down")

    aux_module.call_llm = failing
    provider = make_provider(init=False, **_extraction_cfg())
    provider._queue_max_attempts = 2
    provider.initialize("test-session")

    provider.on_session_end(list(MESSAGES))

    assert waiter(lambda: provider._extract_queue.dead_count() == 1)
    assert provider._extract_queue.pending_count() == 0
    row = provider._store._conn.execute(
        "SELECT attempts, last_error FROM extract_queue"
    ).fetchone()
    assert row["attempts"] == 2
    assert "backend down" in row["last_error"]
    assert provider._store.list_facts(min_trust=0.0, limit=10) == []


def test_dead_rows_skip_the_backoff_wait(make_provider, aux_module, waiter):
    """Marking a row dead must not be followed by a backoff sleep.

    With a 30s backoff base, the old behavior would stall 30s+ between the
    two doomed rows; skipping the wait for never-to-be-retried rows lets both
    die well inside the waiter timeout.
    """
    def failing(**kwargs):
        raise RuntimeError("always down")

    aux_module.call_llm = failing
    provider = make_provider(init=False, **_extraction_cfg())
    provider._queue_max_attempts = 1
    provider._queue_backoff_base = 30.0
    provider._queue_backoff_cap = 60.0
    provider.initialize("test-session")

    provider._extract_queue.enqueue("USER: first doomed transcript")
    provider._extract_queue.enqueue("USER: second doomed transcript")
    provider._queue_wake.set()

    assert waiter(lambda: provider._extract_queue.dead_count() == 2)


def test_in_memory_attempt_bound_when_mark_failed_is_broken(make_provider, aux_module, waiter):
    """A DB-broken failure loop must not re-run the LLM indefinitely.

    When mark_failed itself keeps raising, the in-memory attempts fallback
    caps the LLM runs at max_attempts and then drops the row from this
    process's consideration; the row stays pending for the next restart.
    """
    calls = []

    def failing(**kwargs):
        calls.append(1)
        raise RuntimeError("backend down")

    aux_module.call_llm = failing
    provider = make_provider(init=False, **_extraction_cfg())
    provider._queue_max_attempts = 2
    provider.initialize("test-session")

    def broken_mark_failed(row_id, error, max_attempts, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    provider._extract_queue.mark_failed = broken_mark_failed
    provider._extract_queue.enqueue("USER: transcript whose failures cannot be recorded")
    provider._queue_wake.set()

    assert waiter(lambda: len(calls) == 2)
    time.sleep(0.5)  # several worker poll intervals (0.1s in tests)
    assert len(calls) == 2, "row must be dropped after the in-memory cap"
    assert provider._extract_queue.pending_count() == 1, (
        "the row stays pending in the DB for the next restart"
    )


def test_restart_drains_rows_left_by_previous_run(hp, make_provider, aux_module, tmp_path, waiter):
    # Simulate a crash: a row was queued but never processed
    db_path = tmp_path / "facts.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    queue = hp.extract_queue.ExtractQueue(conn)
    queue.enqueue("USER: I always use pnpm for node projects.")
    conn.close()

    aux_module.call_llm = lambda **kwargs: _llm_response(FACTS_JSON)
    provider = make_provider(**_extraction_cfg())  # same tmp_path db

    assert waiter(lambda: provider._extract_queue.pending_count() == 0)
    assert waiter(
        lambda: len(provider._store.list_facts(min_trust=0.0, limit=50)) == 3
    )


def test_wal_checkpoint_runs_after_successful_batch(hp, make_provider, aux_module, waiter, monkeypatch):
    calls = []
    original = hp.EnfoldProvider._wal_checkpoint
    monkeypatch.setattr(
        hp.EnfoldProvider,
        "_wal_checkpoint",
        lambda self: calls.append(True) or original(self),
    )
    aux_module.call_llm = lambda **kwargs: _llm_response(FACTS_JSON)
    provider = make_provider(**_extraction_cfg())
    init_calls = len(calls)
    assert init_calls >= 1, "initialize() must checkpoint the WAL"

    provider.on_session_end(list(MESSAGES))
    assert waiter(lambda: len(calls) > init_calls), "drain must checkpoint after a batch"


def test_worker_batch_rebuilds_banks_once_per_category(make_provider, aux_module, waiter, monkeypatch):
    aux_module.call_llm = lambda **kwargs: _llm_response(FACTS_JSON)
    provider = make_provider(**_extraction_cfg())

    rebuilds = []
    monkeypatch.setattr(
        fake_hermes.MemoryStore,
        "_rebuild_bank",
        lambda self, category: rebuilds.append(category),
    )

    provider.on_session_end(list(MESSAGES))
    assert waiter(
        lambda: len(provider._store.list_facts(min_trust=0.0, limit=50)) == 3
    )
    # 3 facts in 2 categories: exactly one rebuild per category, not per add
    assert waiter(lambda: sorted(rebuilds) == ["project", "tool"])


def test_inserted_facts_get_embeddings(make_provider, aux_module, waiter):
    aux_module.call_llm = lambda **kwargs: _llm_response(FACTS_JSON)
    provider = make_provider(**_extraction_cfg())
    provider.on_session_end(list(MESSAGES))
    assert waiter(
        lambda: provider._store._conn.execute(
            "SELECT COUNT(*) FROM fact_embeddings"
        ).fetchone()[0] == 3
    )


def test_shutdown_interrupted_extraction_stays_pending(make_provider, aux_module):
    """A teardown that interrupts an in-flight extraction must not burn a retry.

    When the gateway restarts mid-extraction the DB connection is torn down
    under the worker, surfacing as a bad file descriptor. That is not a real
    extraction failure: the row must be left pending with its attempt count
    untouched so the next worker drains it cleanly, instead of consuming a
    retry attempt and logging an error on every restart.
    """
    import threading

    provider = make_provider(init=False, **_extraction_cfg())
    provider.initialize("test-session")
    # Quiesce the auto-started worker so only our controlled drain runs.
    provider._queue_stop.set()
    if provider._queue_worker:
        provider._queue_worker.join(timeout=2.0)

    stop = threading.Event()

    def fail_as_if_shutting_down(**kwargs):
        # Simulate teardown landing mid-extraction: stop is set, then the
        # connection-backed call blows up with a bad file descriptor.
        stop.set()
        raise OSError(9, "Bad file descriptor")

    aux_module.call_llm = fail_as_if_shutting_down

    row_id = provider._extract_queue.enqueue("USER: transcript cut off by a restart")
    provider._drain_extract_queue(stop, provider._extract_queue)

    row = provider._extract_queue._conn.execute(
        "SELECT status, attempts, lease_owner, lease_until FROM extract_queue WHERE id = ?",
        (row_id,),
    ).fetchone()
    assert row[0] == "pending", "shutdown-interrupted row must stay pending"
    assert row[1] == 0, "a shutdown interruption must not consume a retry attempt"
    assert row[2] is None
    assert row[3] is None
    assert provider._extract_queue.dead_count() == 0
