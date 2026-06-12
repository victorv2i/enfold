"""Quota-aware retries: classification, not_before scheduling, age cap, revival."""

import json
import sqlite3
import time
import types

import pytest

MESSAGES = [
    {"role": "user", "content": "I always use pnpm for node projects, remember that."},
    {"role": "assistant", "content": "Noted, pnpm is your package manager of choice."},
    {"role": "user", "content": "Also the deploy target for my web apps is vercel."},
    {"role": "assistant", "content": "Got it, vercel is the deploy target for web apps."},
]

FACTS_JSON = json.dumps([
    {"content": "Victor uses pnpm as his package manager for node projects.",
     "category": "tool", "tags": "pnpm,node"},
    {"content": "Victor deploys web apps to vercel.",
     "category": "tool", "tags": "deploy,vercel"},
    {"content": "CourseKit is an active project.",
     "category": "project", "tags": "coursekit"},
])

# Shape of the live failure that dead-lettered two rows: plan-limit 429 with
# a reset window far beyond the exponential backoff span.
QUOTA_ERROR = (
    "Error code: 429 - {'type': 'error', 'error': {'type': 'usage_limit_reached', "
    "'message': 'usage limit reached'}, 'resets_in_seconds': 4716}"
)

# The extract_queue schema as it existed before the not_before column.
OLD_SCHEMA = """
CREATE TABLE extract_queue (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    payload    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    status     TEXT NOT NULL DEFAULT 'pending'
);
"""


def _llm_response(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


def _extraction_cfg():
    return {"extraction_provider": "testprov", "extraction_model": "testmodel"}


@pytest.fixture()
def raw_queue(hp, tmp_path):
    conn = sqlite3.connect(str(tmp_path / "queue.db"), check_same_thread=False)
    queue = hp.extract_queue.ExtractQueue(conn)
    yield queue
    conn.close()


def _row(queue, row_id):
    return queue._conn.execute(
        "SELECT attempts, last_error, status, not_before FROM extract_queue WHERE id = ?",
        (row_id,),
    ).fetchone()


def _backdate(queue, row_id, hours):
    queue._conn.execute(
        "UPDATE extract_queue SET created_at = datetime('now', ?) WHERE id = ?",
        (f"-{hours} hours", row_id),
    )
    queue._conn.commit()


# ---------------------------------------------------------------------------
# Classification and delay parsing
# ---------------------------------------------------------------------------

def test_is_quota_error_matches_patterns(hp):
    is_quota = hp.extract_queue.is_quota_error
    assert is_quota(QUOTA_ERROR)
    assert is_quota("Rate Limit exceeded, slow down")
    assert is_quota("monthly QUOTA exhausted")
    assert not is_quota("backend down")
    assert not is_quota(None)
    assert not is_quota("")


def test_quota_retry_delay_parses_resets_in_seconds(hp):
    delay = hp.extract_queue.quota_retry_delay(QUOTA_ERROR)
    assert 4716 + 60 <= delay <= 4716 + 300


def test_quota_retry_delay_parses_resets_at_epoch(hp):
    now = 1750000000.0
    delay = hp.extract_queue.quota_retry_delay(
        "429 quota window, 'resets_at': 1750000600", now=now
    )
    assert 600 + 60 <= delay <= 600 + 300


def test_unparseable_quota_error_falls_back_to_1800s(hp):
    assert hp.extract_queue.quota_retry_delay("Error code: 429 rate limit") == 1800.0
    assert hp.extract_queue.quota_retry_delay("") == 1800.0


# ---------------------------------------------------------------------------
# Queue-level quota scheduling
# ---------------------------------------------------------------------------

def test_mark_quota_failed_does_not_consume_attempts(raw_queue):
    row_id = raw_queue.enqueue("quota limited payload")
    due = time.time() + 4716
    assert raw_queue.mark_quota_failed(row_id, QUOTA_ERROR, due) is True
    attempts, last_error, status, not_before = _row(raw_queue, row_id)
    assert attempts == 0
    assert status == "pending"
    assert not_before == pytest.approx(due)
    assert "usage_limit_reached" in last_error
    assert raw_queue.mark_quota_failed(99999, QUOTA_ERROR, due) is False


def test_next_pending_skips_not_before_until_due(raw_queue):
    first = raw_queue.enqueue("rescheduled payload")
    second = raw_queue.enqueue("due payload")
    raw_queue.mark_quota_failed(first, QUOTA_ERROR, time.time() + 3600)

    row = raw_queue.next_pending(max_attempts=5)
    assert row["id"] == second
    raw_queue.mark_done(second)
    assert raw_queue.next_pending(max_attempts=5) is None
    # Both still count as pending even while one is waiting out its window
    assert raw_queue.pending_count() == 1

    raw_queue._conn.execute(
        "UPDATE extract_queue SET not_before = ? WHERE id = ?",
        (time.time() - 1, first),
    )
    raw_queue._conn.commit()
    assert raw_queue.next_pending(max_attempts=5)["id"] == first


# ---------------------------------------------------------------------------
# 48h age cap
# ---------------------------------------------------------------------------

def test_age_cap_kills_on_plain_failure(raw_queue):
    row_id = raw_queue.enqueue("ancient payload")
    _backdate(raw_queue, row_id, 49)
    attempts = raw_queue.mark_failed(row_id, "backend down", max_attempts=5)
    assert attempts == 1
    _, last_error, status, _ = _row(raw_queue, row_id)
    assert status == "dead"
    assert "48h age cap" in last_error


def test_age_cap_kills_quota_rows_too(raw_queue):
    row_id = raw_queue.enqueue("ancient quota payload")
    _backdate(raw_queue, row_id, 49)
    assert raw_queue.mark_quota_failed(row_id, QUOTA_ERROR, time.time() + 60) is False
    attempts, last_error, status, not_before = _row(raw_queue, row_id)
    assert status == "dead"
    assert attempts == 0
    assert not_before is None
    assert "48h age cap" in last_error


def test_young_rows_unaffected_by_age_cap(raw_queue):
    row_id = raw_queue.enqueue("fresh payload")
    raw_queue.mark_failed(row_id, "backend down", max_attempts=5)
    _, last_error, status, _ = _row(raw_queue, row_id)
    assert status == "pending"
    assert "48h age cap" not in last_error


# ---------------------------------------------------------------------------
# Revival
# ---------------------------------------------------------------------------

def test_revive_dead_resets_rows_to_pending(raw_queue):
    row_id = raw_queue.enqueue("doomed payload")
    raw_queue.mark_quota_failed(row_id, QUOTA_ERROR, time.time() + 3600)
    raw_queue.mark_failed(row_id, "boom", max_attempts=1)
    # mark_failed leaves the stale not_before; revive must clear it
    raw_queue._conn.execute(
        "UPDATE extract_queue SET not_before = ? WHERE id = ?",
        (time.time() + 3600, row_id),
    )
    raw_queue._conn.commit()
    assert raw_queue.dead_count() == 1

    assert raw_queue.revive_dead() == 1
    attempts, last_error, status, not_before = _row(raw_queue, row_id)
    assert status == "pending"
    assert attempts == 0
    assert not_before is None
    assert last_error.endswith("(revived)")
    assert "boom" in last_error
    assert raw_queue.next_pending(max_attempts=5)["id"] == row_id


def test_revive_dead_with_ids_only_touches_those(raw_queue):
    first = raw_queue.enqueue("dead one")
    second = raw_queue.enqueue("dead two")
    raw_queue.mark_failed(first, "boom", max_attempts=1)
    raw_queue.mark_failed(second, "boom", max_attempts=1)

    assert raw_queue.revive_dead(ids=[second]) == 1
    assert _row(raw_queue, first)[2] == "dead"
    assert _row(raw_queue, second)[2] == "pending"
    assert raw_queue.revive_dead(ids=[]) == 0


def test_revive_recent_quota_dead_filters(raw_queue):
    quota_id = raw_queue.enqueue("young quota dead")
    other_id = raw_queue.enqueue("young plain dead")
    old_id = raw_queue.enqueue("old quota dead")
    raw_queue.mark_failed(quota_id, QUOTA_ERROR, max_attempts=1)
    raw_queue.mark_failed(other_id, "backend down", max_attempts=1)
    raw_queue.mark_failed(old_id, QUOTA_ERROR, max_attempts=1)
    _backdate(raw_queue, old_id, 49)

    assert raw_queue.revive_recent_quota_dead() == 1
    assert _row(raw_queue, quota_id)[2] == "pending"
    assert _row(raw_queue, other_id)[2] == "dead"
    assert _row(raw_queue, old_id)[2] == "dead"
    # Idempotent: nothing left to revive
    assert raw_queue.revive_recent_quota_dead() == 0


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migration_adds_column_once_and_is_idempotent(hp, tmp_path):
    conn = sqlite3.connect(str(tmp_path / "old.db"), check_same_thread=False)
    try:
        conn.executescript(OLD_SCHEMA)
        conn.execute(
            "INSERT INTO extract_queue (payload) VALUES (?)", ("pre-migration row",)
        )
        conn.commit()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(extract_queue)")]
        assert "not_before" not in cols

        queue = hp.extract_queue.ExtractQueue(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(extract_queue)")]
        assert cols.count("not_before") == 1
        # Pre-migration row stays retrievable with a NULL not_before
        row = queue.next_pending(max_attempts=5)
        assert row["payload"] == "pre-migration row"

        # Re-running the constructor (restart) must not add it again
        hp.extract_queue.ExtractQueue(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(extract_queue)")]
        assert cols.count("not_before") == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Worker behavior
# ---------------------------------------------------------------------------

def test_worker_quota_failure_keeps_attempts_and_schedules(make_provider, aux_module, waiter):
    calls = []

    def quota_limited(**kwargs):
        calls.append(1)
        raise RuntimeError(QUOTA_ERROR)

    aux_module.call_llm = quota_limited
    provider = make_provider(**_extraction_cfg())
    before = time.time()

    provider.on_session_end(list(MESSAGES))

    def scheduled():
        row = provider._store._conn.execute(
            "SELECT not_before FROM extract_queue"
        ).fetchone()
        return row is not None and row["not_before"] is not None

    assert waiter(scheduled)
    row = provider._store._conn.execute(
        "SELECT attempts, status, not_before FROM extract_queue"
    ).fetchone()
    assert row["attempts"] == 0, "quota errors must not consume attempts"
    assert row["status"] == "pending"
    assert row["not_before"] >= before + 4716 + 60
    assert row["not_before"] <= time.time() + 4716 + 300
    time.sleep(0.5)  # several worker poll intervals (0.1s in tests)
    assert len(calls) == 1, "row must not be retried before not_before"


def test_initialize_auto_revives_young_quota_dead_only(hp, make_provider, aux_module, tmp_path, waiter):
    # Simulate last night: rows dead-lettered by the old fixed-attempt path
    db_path = tmp_path / "facts.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    queue = hp.extract_queue.ExtractQueue(conn)
    quota_id = queue.enqueue("USER: I always use pnpm for node projects.")
    other_id = queue.enqueue("USER: transcript killed by a real failure")
    old_id = queue.enqueue("USER: quota transcript past the age cap")
    queue.mark_failed(quota_id, QUOTA_ERROR, max_attempts=1)
    queue.mark_failed(other_id, "backend down", max_attempts=1)
    queue.mark_failed(old_id, QUOTA_ERROR, max_attempts=1)
    conn.execute(
        "UPDATE extract_queue SET created_at = datetime('now', '-49 hours') WHERE id = ?",
        (old_id,),
    )
    conn.commit()
    conn.close()

    aux_module.call_llm = lambda **kwargs: _llm_response(FACTS_JSON)
    provider = make_provider(**_extraction_cfg())  # same tmp_path db

    # The revived quota row is drained into facts; the others stay dead
    assert waiter(lambda: provider._extract_queue.pending_count() == 0)
    assert waiter(
        lambda: len(provider._store.list_facts(min_trust=0.0, limit=50)) == 3
    )
    statuses = {
        row["id"]: row["status"]
        for row in provider._store._conn.execute("SELECT id, status FROM extract_queue")
    }
    assert quota_id not in statuses, "revived row must be processed and removed"
    assert statuses[other_id] == "dead"
    assert statuses[old_id] == "dead"
    assert provider._extract_queue.dead_count() == 2
