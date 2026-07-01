from __future__ import annotations

import json
import sqlite3

from memory_eval.cases import generate_exact_fact_cases, load_cases, write_json_report
from memory_eval.runner import EvalCase, EvalResult


def test_load_cases_accepts_list_or_wrapped_shape(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({
        "cases": [
            {"id": "one", "query": "q", "gold_fact_id": 42, "stale_fact_ids": [41], "tags": ["smoke"]}
        ]
    }))

    cases = load_cases(path)

    assert cases == [EvalCase(id="one", query="q", gold_fact_id=42, stale_fact_ids=[41], tags=["smoke"])]

    path.write_text(json.dumps([
        {"id": "two", "query": "qq", "gold_fact_id": 7, "category": "project", "min_trust": 0.8}
    ]))
    assert load_cases(path)[0].category == "project"
    assert load_cases(path)[0].min_trust == 0.8


def test_load_cases_accepts_memoryarena_v2_fields(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({
        "cases": [{
            "id": "stale-1",
            "query": "Which setting is current?",
            "gold_fact_id": 10,
            "case_type": "stale_suppression",
            "expected_current_fact_ids": [10, 12],
            "entity_refs": ["Alex", "Hermes"],
            "difficulty": "hard",
            "generation": "hand",
            "privacy_tier": "private",
            "should_abstain": False,
            "provenance": {"source_type": "session_end", "min_sources": 1},
            "answer_rubric": {"must_mention": ["current"], "must_not_mention": ["old"]},
        }]
    }))

    case = load_cases(path)[0]

    assert case.case_type == "stale_suppression"
    assert case.expected_current_fact_ids == [10, 12]
    assert case.entity_refs == ["Alex", "Hermes"]
    assert case.difficulty == "hard"
    assert case.generation == "hand"
    assert case.privacy_tier == "private"
    assert case.provenance == {"source_type": "session_end", "min_sources": 1}
    assert case.answer_rubric == {"must_mention": ["current"], "must_not_mention": ["old"]}


def test_generate_exact_fact_cases_samples_active_facts_without_private_fixture_content(tmp_path):
    db = tmp_path / "facts.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT, trust_score REAL)")
    conn.executemany(
        "INSERT INTO facts VALUES (?, ?, ?, ?)",
        [
            (1, "old low trust", "general", 0.1),
            (2, "current preference", "user_pref", 0.9),
            (3, "current project", "project", 0.7),
            (4, "Superseded 2026-06-28. historical value", "user_pref", 0.9),
            (5, "   stale/disabled historical value", "user_pref", 0.9),
            (6, "historical/superseded project config", "project", 0.9),
        ],
    )
    conn.commit()
    conn.close()

    cases = generate_exact_fact_cases(db, limit=3, min_trust=0.3)

    assert [c.gold_fact_id for c in cases] == [2, 3]
    assert [c.query for c in cases] == ["current preference", "current project"]
    assert all(c.tags == ["exact-fact-smoke"] for c in cases)
    assert all(c.privacy_tier == "private" for c in cases)


def test_generate_exact_fact_cases_respects_category_filter_and_limit(tmp_path):
    db = tmp_path / "facts.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT, trust_score REAL)")
    conn.executemany(
        "INSERT INTO facts VALUES (?, ?, ?, ?)",
        [
            (1, "pref one", "user_pref", 0.9),
            (2, "project one", "project", 0.9),
            (3, "pref two", "user_pref", 0.9),
            (4, "pref three", "user_pref", 0.9),
        ],
    )
    conn.commit()
    conn.close()

    cases = generate_exact_fact_cases(db, limit=2, min_trust=0.3, category="user_pref")

    assert [c.gold_fact_id for c in cases] == [1, 3]
    assert all(c.category == "user_pref" for c in cases)


def test_write_json_report_omits_result_content_by_default(tmp_path):
    out = tmp_path / "report.json"
    case = EvalCase(id="one", query="q", gold_fact_id=1)
    result = EvalResult(
        case=case,
        ranked_fact_ids=[1],
        gold_rank=1,
        stale_leak_ranks=[],
        latency_ms=1.2,
        results=[{"fact_id": 1, "content": "private text", "score": 1.0}],
    )

    write_json_report(out, summary={"cases": 1}, results=[result], metadata={"db": "copy.db"})

    data = json.loads(out.read_text())
    assert data["metadata"] == {"db": "copy.db"}
    assert data["summary"] == {"cases": 1}
    assert data["results"][0]["top_fact_ids"] == [1]
    assert "private text" not in out.read_text()


def test_write_json_report_keeps_private_cases_redacted_even_with_include_text(tmp_path):
    out = tmp_path / "report.json"
    private_case = EvalCase(
        id="private",
        query="private query text",
        gold_fact_id=1,
        privacy_tier="private",
        case_type="current_preference",
        tags=["private-sensitive-tag"],
    )
    public_case = EvalCase(
        id="public",
        query="public query text",
        gold_fact_id=2,
        privacy_tier="public",
        case_type="guardrail",
    )
    results = [
        EvalResult(private_case, [1], 1, [], 1.0, [{"fact_id": 1, "content": "private content"}]),
        EvalResult(public_case, [2], 1, [], 1.0, [{"fact_id": 2, "content": "public content", "score": 1.0}]),
    ]

    write_json_report(out, summary={"cases": 2}, results=results, metadata={}, include_text=True)

    text = out.read_text()
    data = json.loads(text)
    assert data["results"][0]["case_type"] == "current_preference"
    assert data["results"][0]["privacy_tier"] == "private"
    assert data["results"][1]["top_scores"] == [1.0]
    assert data["results"][1]["top_score"] == 1.0
    assert data["results"][1]["score_margin"] is None
    assert "private query text" not in text
    assert "private content" not in text
    assert "private-sensitive-tag" not in text
    assert "public query text" in text
    assert "public content" in text
