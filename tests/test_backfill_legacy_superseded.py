"""Offline legacy-marker backfill (backfill_legacy_superseded.py).

Converts facts carrying the old 'SUPERSEDED <date>:' / 'STALE/DISABLED
<date>:' / 'Historical/superseded <date>:' content prefix to structural
invalid_at, without rewriting content and without fabricating a
superseded_by link (there is no reliable way to infer the successor fact
from the prefix alone). Dry-run by default; refuses a live .hermes path.
"""

import sqlite3

import pytest

from holographic_plus.backfill_legacy_superseded import (
    plan_backfill,
    execute_backfill,
    GuardRailError,
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
    hrr_vector      BLOB,
    valid_from      TIMESTAMP,
    invalid_at      TIMESTAMP,
    superseded_by   INTEGER
);
"""


def _conn(path=":memory:"):
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    c.commit()
    return c


def _add(conn, content):
    cur = conn.execute("INSERT INTO facts (content) VALUES (?)", (content,))
    conn.commit()
    return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# plan_backfill
# ---------------------------------------------------------------------------

def test_plan_finds_legacy_prefixed_facts():
    conn = _conn()
    a = _add(conn, "SUPERSEDED 2026-01-01: old routing fact")
    _add(conn, "The current routing fact is stable.")
    plan = plan_backfill(conn)
    assert plan.fact_ids == [a]


def test_plan_matches_all_three_legacy_prefixes():
    conn = _conn()
    a = _add(conn, "SUPERSEDED 2026-01-01: old fact one")
    b = _add(conn, "STALE/DISABLED 2026-02-01: old fact two")
    c = _add(conn, "Historical/superseded 2026-03-01: old fact three")
    plan = plan_backfill(conn)
    assert sorted(plan.fact_ids) == sorted([a, b, c])


def test_plan_skips_already_invalid_rows():
    conn = _conn()
    a = _add(conn, "SUPERSEDED 2026-01-01: old routing fact")
    conn.execute(
        "UPDATE facts SET invalid_at = CURRENT_TIMESTAMP WHERE fact_id = ?", (a,)
    )
    conn.commit()
    plan = plan_backfill(conn)
    assert plan.fact_ids == []


def test_plan_ignores_mid_sentence_word():
    conn = _conn()
    _add(conn, "The project supersedes the old plan but is not itself marked")
    plan = plan_backfill(conn)
    assert plan.fact_ids == []


# ---------------------------------------------------------------------------
# execute_backfill: dry-run default, live-path refusal, content preserved
# ---------------------------------------------------------------------------

def test_execute_backfill_dry_run_does_not_modify_rows(tmp_path):
    db_path = str(tmp_path / "facts.db")
    conn = _conn(db_path)
    a = _add(conn, "SUPERSEDED 2026-01-01: old routing fact")
    conn.close()

    result = execute_backfill(db_path)  # dry_run=True by default
    assert result.dry_run is True
    assert result.count == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT invalid_at, content FROM facts WHERE fact_id = ?", (a,)
    ).fetchone()
    assert row["invalid_at"] is None
    assert row["content"] == "SUPERSEDED 2026-01-01: old routing fact"
    conn.close()


def test_execute_backfill_sets_invalid_at_without_rewriting_content(tmp_path):
    db_path = str(tmp_path / "facts.db")
    conn = _conn(db_path)
    a = _add(conn, "SUPERSEDED 2026-01-01: old routing fact")
    conn.close()

    result = execute_backfill(db_path, dry_run=False)
    assert result.dry_run is False
    assert result.updated == 1
    assert result.integrity_ok is True

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT invalid_at, superseded_by, content FROM facts WHERE fact_id = ?", (a,)
    ).fetchone()
    assert row["invalid_at"] is not None
    assert row["superseded_by"] is None  # no successor can be inferred from a prefix alone
    assert row["content"] == "SUPERSEDED 2026-01-01: old routing fact"
    conn.close()


def test_execute_backfill_refuses_live_hermes_path():
    with pytest.raises(GuardRailError):
        execute_backfill("/home/user/.hermes/holographic_plus.db", dry_run=False)


def test_execute_backfill_requires_temporal_schema(tmp_path):
    db_path = str(tmp_path / "facts.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE
        );
        """
    )
    conn.close()
    with pytest.raises(GuardRailError):
        execute_backfill(db_path, dry_run=False)
