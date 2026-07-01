"""Temporal validity: invalidate-not-delete supersession (temporal.py).

Covers the additive schema migration (idempotent, safe on a live-size
store), value-update detection (the case the near-duplicate gate
deliberately lets through), supersede/history bookkeeping, and the
provider-level write/read wiring: an incoming value-update supersedes the
old fact instead of leaving two live rows, search excludes superseded facts
by default, temporal_filter=false preserves old behaviour, and the legacy
string-prefix filter still works alongside structural supersession.
"""

import sqlite3

import pytest

from holographic_plus.temporal import (
    ensure_temporal_schema,
    _is_value_update,
    find_value_update_target,
    supersede,
    fact_history,
)

_SCHEMA = """
CREATE TABLE facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);
"""


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _add(conn, content, **kw):
    cols = ["content"] + list(kw.keys())
    placeholders = ", ".join("?" * len(cols))
    cur = conn.execute(
        f"INSERT INTO facts ({', '.join(cols)}) VALUES ({placeholders})",
        [content, *kw.values()],
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Migration: idempotent, additive, fast on a live-size store
# ---------------------------------------------------------------------------

def test_migration_adds_nullable_columns():
    conn = _conn()
    ensure_temporal_schema(conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
    assert {"valid_from", "invalid_at", "superseded_by"} <= cols


def test_migration_is_idempotent():
    conn = _conn()
    ensure_temporal_schema(conn)
    ensure_temporal_schema(conn)  # second call must not raise or duplicate columns
    cols = [row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()]
    assert cols.count("invalid_at") == 1


def test_migration_leaves_existing_rows_currently_valid():
    conn = _conn()
    fid = _add(conn, "The Skylark dashboard port is 3100.")
    ensure_temporal_schema(conn)
    row = conn.execute("SELECT invalid_at FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["invalid_at"] is None


def test_migration_is_fast_on_a_few_thousand_rows():
    import time

    conn = _conn()
    conn.executemany(
        "INSERT INTO facts (content) VALUES (?)",
        [(f"fact number {i} has a unique value",) for i in range(3500)],
    )
    conn.commit()
    t0 = time.perf_counter()
    ensure_temporal_schema(conn)
    assert time.perf_counter() - t0 < 1.0


# ---------------------------------------------------------------------------
# Value-update detection: the complement of the near-duplicate gate
# ---------------------------------------------------------------------------

def test_changed_value_same_words_is_a_value_update():
    a = "The Skylark dashboard port is 3100."
    b = "The Skylark dashboard port is 3200."
    assert _is_value_update(b, a) is True


def test_identical_values_is_not_a_value_update():
    # Equal value tokens: a plain restatement, owned by the dedup gate, not here.
    a = "The Skylark dashboard port is 3100."
    assert _is_value_update(a, a) is False


def test_no_value_tokens_is_not_a_value_update():
    a = "The Skylark sandbox service is currently active."
    b = "The Skylark sandbox service is currently archived."
    assert _is_value_update(b, a) is False


def test_unrelated_fact_is_not_a_value_update():
    a = "The Skylark dashboard port is 3100."
    b = "Alex Rivera moved to Springfield last month."
    assert _is_value_update(b, a) is False


def test_find_value_update_target_skips_already_superseded_candidate():
    candidates = [
        {"fact_id": 1, "content": "The Skylark dashboard port is 3100.", "superseded_by": 2},
        {"fact_id": 3, "content": "The Skylark dashboard port is 3100 on staging."},
    ]
    target = find_value_update_target("The Skylark dashboard port is 3200.", candidates)
    assert target is None  # neither candidate is a value-word-for-word match


def test_find_value_update_target_returns_matching_candidate():
    candidates = [
        {"fact_id": 5, "content": "Alex Rivera moved to Springfield last month."},
        {"fact_id": 7, "content": "The Skylark dashboard port is 3100."},
    ]
    target = find_value_update_target("The Skylark dashboard port is 3200.", candidates)
    assert target is not None
    assert target["fact_id"] == 7


# ---------------------------------------------------------------------------
# Supersede + history chain
# ---------------------------------------------------------------------------

def test_supersede_marks_old_fact_invalid():
    conn = _conn()
    ensure_temporal_schema(conn)
    old_id = _add(conn, "The Skylark dashboard port is 3100.")
    new_id = _add(conn, "The Skylark dashboard port is 3200.")

    supersede(conn, old_id, new_id)

    row = conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone()
    assert row["invalid_at"] is not None
    assert row["superseded_by"] == new_id

    new_row = conn.execute(
        "SELECT invalid_at FROM facts WHERE fact_id = ?", (new_id,)
    ).fetchone()
    assert new_row["invalid_at"] is None


def test_supersede_is_idempotent_and_does_not_overwrite_an_existing_link():
    conn = _conn()
    ensure_temporal_schema(conn)
    a = _add(conn, "v1")
    b = _add(conn, "v2")
    c = _add(conn, "v3")

    supersede(conn, a, b)
    supersede(conn, a, c)  # a is already invalid: must be a no-op

    row = conn.execute(
        "SELECT superseded_by FROM facts WHERE fact_id = ?", (a,)
    ).fetchone()
    assert row["superseded_by"] == b


def test_fact_history_returns_full_chain_from_any_link():
    conn = _conn()
    ensure_temporal_schema(conn)
    v1 = _add(conn, "The Skylark dashboard port is 3100.")
    v2 = _add(conn, "The Skylark dashboard port is 3200.")
    v3 = _add(conn, "The Skylark dashboard port is 3300.")
    supersede(conn, v1, v2)
    supersede(conn, v2, v3)

    for start in (v1, v2, v3):
        chain = fact_history(conn, start)
        ids = [f["fact_id"] for f in chain]
        assert ids == [v1, v2, v3]


def test_fact_history_of_a_fact_with_no_chain_is_itself_alone():
    conn = _conn()
    ensure_temporal_schema(conn)
    fid = _add(conn, "A standalone fact with no history.")
    chain = fact_history(conn, fid)
    assert [f["fact_id"] for f in chain] == [fid]


def test_fact_history_missing_fact_returns_empty():
    conn = _conn()
    ensure_temporal_schema(conn)
    assert fact_history(conn, 9999) == []


# ---------------------------------------------------------------------------
# Provider wiring: migration on init, write-path supersession, read-path filter
# ---------------------------------------------------------------------------

def test_migration_runs_on_provider_initialize(make_provider):
    provider = make_provider()
    cols = {
        row[1] for row in provider._store._conn.execute(
            "PRAGMA table_info(facts)"
        ).fetchall()
    }
    assert {"valid_from", "invalid_at", "superseded_by"} <= cols


def test_interactive_add_of_a_value_update_supersedes_the_old_fact(make_provider):
    import json

    provider = make_provider()
    add_result = json.loads(provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark dashboard port is 3100.",
        "category": "project",
    }))
    old_id = add_result["fact_id"]

    update_result = json.loads(provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark dashboard port is 3200.",
        "category": "project",
    }))
    assert update_result["status"] == "added"
    new_id = update_result["fact_id"]
    assert new_id != old_id

    old_row = provider._store._conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone()
    assert old_row["invalid_at"] is not None
    assert old_row["superseded_by"] == new_id


def test_search_excludes_superseded_facts_by_default(make_provider):
    import json

    provider = make_provider()
    provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark dashboard port is 3100.",
        "category": "project",
    })
    provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark dashboard port is 3200.",
        "category": "project",
    })

    results = provider.search("Skylark dashboard port", min_trust=0.0, limit=10)
    contents = [r["content"] for r in results]
    assert "The Skylark dashboard port is 3100." not in contents
    assert "The Skylark dashboard port is 3200." in contents


def test_temporal_filter_off_preserves_old_behaviour(make_provider):
    import json

    provider = make_provider(temporal_filter=False)
    provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark dashboard port is 3100.",
        "category": "project",
    })
    provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark dashboard port is 3200.",
        "category": "project",
    })

    results = provider.search("Skylark dashboard port", min_trust=0.0, limit=10)
    contents = [r["content"] for r in results]
    assert "The Skylark dashboard port is 3100." in contents
    assert "The Skylark dashboard port is 3200." in contents


def test_legacy_string_prefix_filter_still_works_alongside_structural_filter(make_provider):
    provider = make_provider()
    provider._store.add_fact(
        "SUPERSEDED 2026-01-01: old routing fact", category="project"
    )
    provider._store.add_fact("The current routing fact is stable.", category="project")

    results = provider.search("routing fact", min_trust=0.0, limit=10)
    contents = [r["content"] for r in results]
    assert not any(c.startswith("SUPERSEDED") for c in contents)
    assert "The current routing fact is stable." in contents


def test_provider_fact_history_walks_the_chain(make_provider):
    import json

    provider = make_provider()
    r1 = json.loads(provider._handle_fact_store({
        "action": "add", "content": "The Skylark dashboard port is 3100.",
        "category": "project",
    }))
    r2 = json.loads(provider._handle_fact_store({
        "action": "add", "content": "The Skylark dashboard port is 3200.",
        "category": "project",
    }))

    chain = provider.fact_history(r1["fact_id"])
    ids = [f["fact_id"] for f in chain]
    assert ids == [r1["fact_id"], r2["fact_id"]]

    # Same chain reachable from the newer fact too.
    chain_from_new = provider.fact_history(r2["fact_id"])
    assert [f["fact_id"] for f in chain_from_new] == ids


def test_extraction_insert_supersedes_a_value_update(make_provider, aux_module, waiter):
    import json
    import types

    provider = make_provider(extraction_provider="testprov", extraction_model="testmodel")
    old_id = provider._store.add_fact(
        "The Skylark dashboard port is 3100.", category="project"
    )

    aux_module.call_llm = lambda **kwargs: types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(
            content=json.dumps([{
                "content": "The Skylark dashboard port is 3200.",
                "category": "project", "tags": "skylark,port",
            }]),
        ))]
    )

    provider.on_session_end([
        {"role": "user", "content": "The Skylark dashboard port changed to 3200."},
        {"role": "assistant", "content": "Updated, the port is now 3200."},
    ])

    assert waiter(lambda: provider._extract_queue.pending_count() == 0)

    old_row = provider._store._conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone()
    assert old_row["invalid_at"] is not None
    assert old_row["superseded_by"] is not None

    facts = provider._store.list_facts(min_trust=0.0, limit=50)
    contents = [f["content"] for f in facts]
    assert "The Skylark dashboard port is 3200." in contents
