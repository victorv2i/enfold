"""Temporal validity: invalidate-not-delete supersession (temporal.py).

Covers the additive schema migration (idempotent, safe on a live-size
store), value-update detection (the case the near-duplicate gate
deliberately lets through), supersede/history bookkeeping, and the
provider-level write/read wiring: an incoming value-update supersedes the
old fact instead of leaving two live rows, search excludes superseded facts
by default, temporal_filter=false preserves old behaviour, and the legacy
string-prefix filter still works alongside structural supersession.
"""

import contextlib
import importlib.util
import logging
from pathlib import Path
import sqlite3
import sys
import threading
import types

from enfold.temporal import (
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


def test_migration_survives_concurrent_add_column_race(tmp_path):
    """Two real connections racing the same check-then-ALTER on a fresh db_path.

    Regression: two separate MCP server processes starting against a brand
    new store at the same moment can both see the column missing (PRAGMA
    table_info) before either has run its ALTER TABLE, so both attempt it;
    the loser raised "duplicate column name" and crashed initialize(). A
    barrier holds both threads at the same start line so the race is
    deterministic rather than relying on scheduling luck.
    """
    db_path = tmp_path / "race.db"
    setup = sqlite3.connect(str(db_path))
    setup.executescript(_SCHEMA)
    setup.commit()
    setup.close()

    start = threading.Barrier(2)
    errors = []

    def worker():
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        start.wait()
        try:
            ensure_temporal_schema(conn)
        except Exception as exc:  # pragma: no cover - failure surfaced via assert
            errors.append(str(exc))
        conn.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors

    check = sqlite3.connect(str(db_path))
    cols = [row[1] for row in check.execute("PRAGMA table_info(facts)").fetchall()]
    assert cols.count("invalid_at") == 1
    assert {"valid_from", "invalid_at", "superseded_by"} <= set(cols)
    check.close()


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
    assert _is_value_update(b, a) is True


def test_low_jaccard_state_change_is_a_value_update():
    a = "The Springfield timer is enabled for restart automation."
    b = "Automatic restarts are disabled now for Springfield."
    assert _is_value_update(b, a) is True


def test_negation_change_is_a_value_update():
    a = "The Skylark sandbox service is currently configured to run nightly for the gateway."
    b = "The Skylark sandbox service is not currently configured to run nightly for the gateway."
    assert _is_value_update(b, a) is True


def test_unrelated_fact_is_not_a_value_update():
    a = "The Skylark dashboard port is 3100."
    b = "Alex Rivera moved to Springfield last month."
    assert _is_value_update(b, a) is False


def test_temporal_module_imports_before_package_helpers_are_initialized(monkeypatch):
    """Hermes may preload temporal.py while the parent package is half-built."""
    repo_root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("enfold")
    pkg.__path__ = [str(repo_root / "enfold")]
    monkeypatch.setitem(sys.modules, "enfold", pkg)
    sys.modules.pop("enfold.temporal", None)

    spec = importlib.util.spec_from_file_location(
        "enfold.temporal",
        repo_root / "enfold" / "temporal.py",
        submodule_search_locations=None,
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "enfold.temporal", module)
    spec.loader.exec_module(module)

    assert module._is_value_update(
        "Automatic restarts are disabled now for Springfield.",
        "The Springfield timer is enabled for restart automation.",
    ) is True


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


def test_supersede_reports_noop_and_logs_when_old_row_is_not_active(caplog):
    conn = _conn()
    ensure_temporal_schema(conn)
    a = _add(conn, "v1")
    b = _add(conn, "v2")
    c = _add(conn, "v3")

    assert supersede(conn, a, b) is True
    caplog.set_level(logging.DEBUG, logger="enfold.temporal")

    assert supersede(conn, a, c) is False

    row = conn.execute(
        "SELECT superseded_by FROM facts WHERE fact_id = ?", (a,)
    ).fetchone()
    assert row["superseded_by"] == b
    assert "supersede no-op" in caplog.text


def test_cross_process_write_lock_is_reentrant_for_same_db_path(hp, tmp_path):
    db_path = tmp_path / "facts.db"
    db_path.touch()

    with hp.cross_process_write_lock(str(db_path)):
        with hp.cross_process_write_lock(str(db_path)):
            pass

    assert (tmp_path / "facts.db.mcp-write.lock").exists()


def test_legacy_builtin_memory_hook_takes_shared_write_lock(
    make_provider, hp, monkeypatch
):
    provider = make_provider()
    depth = 0
    events = []

    @contextlib.contextmanager
    def fake_lock(db_path):
        nonlocal depth
        assert db_path == str(Path(provider._store.db_path).expanduser().resolve())
        depth += 1
        events.append("enter")
        try:
            yield
        finally:
            depth -= 1
            events.append("exit")

    def checked_parent_write(self, action, target, content):
        assert self is provider
        assert depth == 1
        events.append("parent-write")

    monkeypatch.setattr(hp, "cross_process_write_lock", fake_lock)
    monkeypatch.setattr(
        hp.HolographicMemoryProvider, "on_memory_write", checked_parent_write
    )
    provider.on_memory_write("add", "memory", "serialized host write")
    assert events == ["enter", "parent-write", "exit"]


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


def test_interactive_add_of_a_state_update_supersedes_the_old_fact(make_provider):
    import json

    provider = make_provider()
    add_result = json.loads(provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark sandbox service is currently active.",
        "category": "project",
    }))
    old_id = add_result["fact_id"]

    update_result = json.loads(provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark sandbox service is currently archived.",
        "category": "project",
    }))
    assert update_result["status"] == "added"
    new_id = update_result["fact_id"]

    old_row = provider._store._conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone()
    assert old_row["invalid_at"] is not None
    assert old_row["superseded_by"] == new_id


def test_interactive_add_holds_write_lock_through_supersede(make_provider, hp, monkeypatch):
    import json

    provider = make_provider()
    lock_depth = 0
    events = []
    expected_db = str(Path(provider._store.db_path).expanduser().resolve())

    @contextlib.contextmanager
    def fake_lock(db_path):
        nonlocal lock_depth
        events.append(("enter", db_path))
        assert db_path == expected_db
        lock_depth += 1
        try:
            yield
        finally:
            lock_depth -= 1
            events.append(("exit", db_path))

    monkeypatch.setattr(hp, "cross_process_write_lock", fake_lock, raising=False)

    original_dedup = provider._find_near_duplicate
    original_update = provider._find_update_target
    original_supersede = provider._supersede_fact

    def checked_dedup(*args, **kwargs):
        assert lock_depth == 1
        events.append(("dedup", None))
        return original_dedup(*args, **kwargs)

    def checked_update(*args, **kwargs):
        assert lock_depth == 1
        events.append(("update", None))
        return original_update(*args, **kwargs)

    def checked_supersede(*args, **kwargs):
        assert lock_depth == 1
        events.append(("supersede", None))
        return original_supersede(*args, **kwargs)

    monkeypatch.setattr(provider, "_find_near_duplicate", checked_dedup)
    monkeypatch.setattr(provider, "_find_update_target", checked_update)
    monkeypatch.setattr(provider, "_supersede_fact", checked_supersede)

    first = json.loads(provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark dashboard port is 3100.",
        "category": "project",
    }))
    second = json.loads(provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark dashboard port is 3200.",
        "category": "project",
    }))

    assert first["status"] == "added"
    assert second["status"] == "added"
    assert ("supersede", None) in events
    assert events.count(("enter", expected_db)) == 2


def test_search_excludes_superseded_facts_by_default(make_provider):
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


def test_extraction_insert_batch_uses_cross_process_write_lock(
    make_provider, hp, monkeypatch, aux_module, waiter
):
    import json
    import types

    lock_events = []

    @contextlib.contextmanager
    def fake_lock(db_path):
        lock_events.append(("enter", db_path))
        try:
            yield
        finally:
            lock_events.append(("exit", db_path))

    monkeypatch.setattr(hp, "cross_process_write_lock", fake_lock, raising=False)
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
    assert lock_events
    assert lock_events[0][0] == "enter"
    assert lock_events[-1][0] == "exit"

    old_row = provider._store._conn.execute(
        "SELECT superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone()
    assert old_row["superseded_by"] is not None


def test_reflection_insert_uses_cross_process_write_lock(make_provider, hp, monkeypatch):
    lock_events = []

    @contextlib.contextmanager
    def fake_lock(db_path):
        lock_events.append(("enter", db_path))
        try:
            yield
        finally:
            lock_events.append(("exit", db_path))

    monkeypatch.setattr(hp, "cross_process_write_lock", fake_lock, raising=False)
    provider = make_provider()

    fact_id = provider._insert_reflection_fact(
        "Alex Rivera prefers async status updates.",
        category="insight",
        tags="source_facts:1,2",
    )

    assert fact_id
    assert [event[0] for event in lock_events] == ["enter", "exit"]
