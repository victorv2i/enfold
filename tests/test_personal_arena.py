from __future__ import annotations

import json
import sqlite3

import pytest

from enfold.schema import migrate
from memory_eval.personal_arena import (
    load_personal_cases,
    run_personal_arena,
    snapshot_database,
    validate_personal_cases,
)


def _database(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    migrate(conn)
    conn.executemany(
        """
        INSERT INTO facts(
            fact_id, content, category, tags, trust_score, scope, sensitivity,
            schema_version, superseded_by
        ) VALUES (?, ?, ?, '', 0.9, 'private', 'normal', 1, ?)
        """,
        [
            (1, "The Atlas launch date is 2026-07-18.", "project", None),
            (2, "The retired Atlas launch date was 2026-06-30.", "project", 1),
            (3, "Mina owns the Atlas release checklist.", "project", None),
        ],
    )
    conn.commit()
    return conn


def _write_cases(path):
    rows = [
        {
            "id": "atlas-date",
            "query": "When is Atlas scheduled to launch?",
            "expected_fact_ids": [1],
            "forbidden_content_regexes": ["retired Atlas"],
            "category": "project",
        },
        {
            "id": "atlas-owner",
            "query": "Who runs the Atlas release checklist?",
            "expected_content_regexes": ["Mina owns the Atlas release checklist"],
            "category": "project",
        },
        {
            "id": "unknown",
            "query": "What is the approved budget for Project Zaffre?",
            "category": "project",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_personal_cases_rejects_invalid_content_regex(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({
        "id": "bad", "query": "q", "category": "project",
        "expected_content_regexes": ["["],
    }) + "\n")

    with pytest.raises(ValueError, match="invalid content regex"):
        load_personal_cases(path)


def test_personal_arena_uses_snapshot_and_real_hybrid_filters(tmp_path):
    source = tmp_path / "live.db"
    conn = _database(source)
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(cases_path)

    snapshot = tmp_path / "snapshot.db"
    snapshot_database(source, snapshot)
    conn.execute(
        "INSERT INTO facts(fact_id, content, category, tags, trust_score, scope, sensitivity, schema_version) "
        "VALUES (4, 'A source-only fact.', 'project', '', 0.9, 'private', 'normal', 1)"
    )
    conn.commit()
    assert sqlite3.connect(snapshot).execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 3
    conn.close()

    run = run_personal_arena(cases_path, source, dimensions=64, abstention_min_score=0.35)

    assert run.summary["cases"] == 3
    assert run.summary["recall_at_1"] == pytest.approx(1.0)
    assert run.summary["recall_at_3"] == pytest.approx(1.0)
    assert run.summary["stale_leak_rate"] == 0.0
    assert run.summary["abstention_correctness"] == 1.0
    assert run.summary["by_category"]["project"]["cases"] == 3
    assert run.metadata["snapshot_copy"] is True
    assert run.metadata["retrieval"]["embedder_production_ready"] is False


def test_validate_personal_cases_rejects_inactive_expected_fact(tmp_path):
    source = tmp_path / "live.db"
    conn = _database(source)
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps({
        "id": "stale-target", "query": "old Atlas date", "category": "project",
        "expected_fact_ids": [2],
    }) + "\n")

    with pytest.raises(ValueError, match="not an active private fact"):
        validate_personal_cases(conn, load_personal_cases(cases_path))
    conn.close()
