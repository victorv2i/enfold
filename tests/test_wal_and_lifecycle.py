"""WAL checkpoint behavior and re-initialize teardown."""

import threading
import time

import fake_hermes

MESSAGES = [
    {"role": "user", "content": "I always use pnpm for node projects, remember that."},
    {"role": "assistant", "content": "Noted, pnpm is your package manager of choice."},
    {"role": "user", "content": "Also the deploy target for my web apps is vercel."},
    {"role": "assistant", "content": "Got it, vercel is the deploy target for web apps."},
]


def test_wal_checkpoint_truncates_the_wal_file(make_provider, tmp_path):
    provider = make_provider()
    for i in range(10):
        provider._store.add_fact(f"wal growth fact number {i}", category="general")
    wal_path = tmp_path / "facts.db-wal"
    assert wal_path.exists() and wal_path.stat().st_size > 0

    result = provider._wal_checkpoint()
    assert result is not None
    busy, wal_pages, checkpointed = result
    assert busy == 0
    assert wal_path.stat().st_size == 0


def test_wal_checkpoint_safe_without_store(hp):
    provider = hp.EnfoldProvider(config={})
    assert provider._wal_checkpoint() is None


def test_reinitialize_stops_previous_worker_and_pool(make_provider):
    provider = make_provider()
    first_worker = provider._queue_worker
    first_pool = provider._embed_pool
    first_store = provider._store
    assert first_worker.is_alive()

    provider.initialize("second-session")

    assert provider._queue_worker is not first_worker
    assert provider._queue_worker.is_alive()
    assert not first_worker.is_alive(), "previous worker must be stopped"
    assert first_pool._shutdown, "previous embed pool must be shut down"
    # Previous store connection is closed (it raises on use)
    import sqlite3
    import pytest
    with pytest.raises(sqlite3.ProgrammingError):
        first_store._conn.execute("SELECT 1")
    # New store works
    assert provider._store is not first_store
    provider._store.add_fact("fact after re-init", category="general")


def test_reinit_with_stuck_worker_leaks_old_store_open(make_provider, aux_module, waiter):
    """A worker mid LLM call cannot be joined: the old connection must stay open.

    Closing it would yank the SQLite handle out from under the still-running
    writer; leaking it for the rest of the process is the safe choice (and is
    what the parent provider always did).
    """
    release = threading.Event()
    started = threading.Event()

    def blocking_llm(**kwargs):
        started.set()
        release.wait(10.0)
        raise RuntimeError("aborted by test")

    aux_module.call_llm = blocking_llm
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel"
    )
    first_store = provider._store
    provider.on_session_end(list(MESSAGES))
    assert waiter(started.is_set), "worker never reached the LLM call"

    t0 = time.monotonic()
    provider.initialize("second-session")
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, "re-init must not stall on a busy worker (0.5s join)"

    # Old connection stays OPEN and usable; the new store is independent.
    assert provider._store is not first_store
    assert first_store._conn.execute("SELECT 1").fetchone()[0] == 1
    release.set()


def test_backfill_checks_stop_event_per_chunk(make_provider, waiter):
    provider = make_provider()
    # Wait out the initial backfill pass, then create work it never saw:
    # direct store adds do not go through the embedding sidecar.
    assert waiter(lambda: not provider._backfill_thread.is_alive())
    provider._store.add_fact("a fact with no embedding yet", category="general")

    def embedding_count():
        return provider._store._conn.execute(
            "SELECT COUNT(*) FROM fact_embeddings"
        ).fetchone()[0]

    stop = threading.Event()
    stop.set()
    provider._backfill_embeddings(stop)
    assert embedding_count() == 0, "a set stop event must halt before any chunk"

    provider._backfill_embeddings(threading.Event())
    assert embedding_count() == 1


def test_stale_embed_task_after_reinit_does_not_write_new_store(hp, tmp_path, waiter):
    started = threading.Event()
    release = threading.Event()

    class BlockingEmbedder(fake_hermes.FakeEmbedder):
        def embed(self, text):
            started.set()
            release.wait(10.0)
            return super().embed(text)

    class UnavailableEmbedder(fake_hermes.FakeEmbedder):
        def is_available(self):
            return False

    blocking_embedder = BlockingEmbedder()
    embedders = [blocking_embedder, UnavailableEmbedder()]

    class TestProvider(hp.EnfoldProvider):
        def _create_embedder(self):
            return embedders.pop(0)

    provider = TestProvider(config={
        "db_path": str(tmp_path / "facts.db"),
        "hrr_dim": 64,
    })
    provider.initialize("first-session")
    assert waiter(lambda: not provider._backfill_thread.is_alive())

    fact_id = provider._store.add_fact(
        "The stale embed task should not write after reinit.",
        category="general",
    )
    provider._submit_embed(
        provider._embed_and_store,
        fact_id,
        "The stale embed task should not write after reinit.",
    )
    assert waiter(started.is_set)

    provider.initialize("second-session")
    release.set()
    assert waiter(lambda: len(blocking_embedder.embed_calls) == 1)
    embedding_count = provider._store._conn.execute(
        "SELECT COUNT(*) FROM fact_embeddings"
    ).fetchone()[0]
    assert embedding_count == 0

    provider.shutdown()


def test_shutdown_stops_worker(make_provider):
    provider = make_provider()
    worker = provider._queue_worker
    pool = provider._embed_pool
    provider.shutdown()
    assert not worker.is_alive()
    assert pool._shutdown
    assert provider._queue_worker is None
    assert provider._extract_queue is None
    assert provider._embed_store is None
