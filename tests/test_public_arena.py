from __future__ import annotations

import json

import pytest

from memory_eval.public_arena import (
    EnfoldCoreFtsCurrentProvider,
    EnfoldOfflineHybridProvider,
    PUBLIC_ARENA_PATH,
    REQUIRED_CASE_TYPES,
    load_public_arena,
    main,
    run_public_arena,
    write_public_arena_report,
)


class PerfectFixtureProvider:
    def __init__(self, arena):
        self._cases = {case.query: case for case in arena.cases}
        self._facts = {fact.fact_id: fact for fact in arena.facts}

    def search(self, query, *, category=None, min_trust=0.3, limit=10, bump=False):
        assert bump is False
        case = self._cases[query]
        if case.should_abstain:
            return []
        expected = case.expected_current_fact_ids or [case.gold_fact_id]
        rows = []
        for rank, fact_id in enumerate(expected[:limit]):
            fact = self._facts[fact_id]
            assert category is None or fact.category == category
            assert fact.trust_score >= min_trust
            rows.append(fact.as_search_row(score=1.0 - rank * 0.05))
        return rows


def test_bundled_public_arena_is_synthetic_complete_and_referentially_valid():
    arena = load_public_arena()

    assert arena.version == "1.0"
    assert arena.source_path == PUBLIC_ARENA_PATH.resolve()
    assert len(arena.facts) == 22
    assert len(arena.cases) == 14
    assert {case.case_type for case in arena.cases} == REQUIRED_CASE_TYPES
    assert all(case.privacy_tier == "public" for case in arena.cases)
    assert all(case.generation == "hand" for case in arena.cases)
    assert all(case.provenance == {"source_type": "synthetic_fixture"} for case in arena.cases)


def test_loader_rejects_non_public_case_and_unknown_fact_reference(tmp_path):
    payload = json.loads(PUBLIC_ARENA_PATH.read_text())
    payload["cases"][0]["privacy_tier"] = "private"
    path = tmp_path / "private.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="privacy_tier='public'"):
        load_public_arena(path)

    payload = json.loads(PUBLIC_ARENA_PATH.read_text())
    payload["cases"][0]["gold_fact_id"] = 999999
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unknown gold_fact_id"):
        load_public_arena(path)


def test_loader_rejects_duplicate_fact_ids_and_incomplete_case_shapes(tmp_path):
    payload = json.loads(PUBLIC_ARENA_PATH.read_text())
    payload["facts"][1]["fact_id"] = payload["facts"][0]["fact_id"]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fact_id values must be unique"):
        load_public_arena(path)

    payload = json.loads(PUBLIC_ARENA_PATH.read_text())
    payload["cases"] = [case for case in payload["cases"] if case["case_type"] != "contradiction"]
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="missing required case types.*contradiction"):
        load_public_arena(path)


def test_public_arena_runs_offline_through_shared_runner_and_metrics():
    arena = load_public_arena()
    run = run_public_arena(PerfectFixtureProvider(arena), arena=arena, limit=5)

    assert run.provider_name == "PerfectFixtureProvider"
    assert len(run.results) == 14
    assert run.summary["cases"] == 14
    assert run.summary["answerable"]["recall@1"] == 1.0
    assert run.summary["set_recall"] == 1.0
    assert run.summary["stale_leak@3"]["leaks"] == 0
    assert run.summary["abstention"]["true_abstain"] == 2
    assert set(run.summary["by_case_type"]) == REQUIRED_CASE_TYPES


def test_default_lexical_baseline_needs_no_database_or_embedding_service():
    run = run_public_arena(limit=5)

    assert run.provider_name == "LexicalFixtureProvider"
    assert len(run.results) == len(run.arena.cases)
    assert run.summary["cases"] == 14
    assert run.summary["answerable"]["recall@1"] > 0.0


def test_core_fts_provider_uses_migrated_state_and_never_returns_stale_facts():
    arena = load_public_arena()
    stale_ids = {item.stale_fact_id for item in arena.memory_state.supersessions}

    with EnfoldCoreFtsCurrentProvider(arena) as provider:
        assert provider.schema_version == 1
        stale_rows = provider.connection.execute(
            "SELECT fact_id, invalid_at, superseded_by FROM facts WHERE invalid_at IS NOT NULL"
        ).fetchall()
        assert {row["fact_id"] for row in stale_rows} == stale_ids
        assert all(row["invalid_at"] and row["superseded_by"] for row in stale_rows)
        assert provider.connection.execute("SELECT COUNT(*) FROM fact_conflicts").fetchone()[0] == 2
        assert provider.connection.execute("SELECT COUNT(*) FROM fact_conflict_members").fetchone()[0] == 4
        assert provider.connection.execute(
            "SELECT COUNT(*) FROM facts WHERE conflict_group IS NOT NULL"
        ).fetchone()[0] == 4

        assert provider.search("Maple support coverage", category="work") == []
        conflicts = provider.search_conflicts(
            "Maple support coverage", category="work"
        )
        assert {row["fact_id"] for row in conflicts} == {1301, 1302}
        assert all(row["is_conflict"] for row in conflicts)

        rows = provider.search("current former runtime Comet Amber Quartz", category="system_state")
        assert stale_ids.isdisjoint(row["fact_id"] for row in rows)


def test_core_fts_arena_gate_has_zero_stale_leaks_and_honest_metadata():
    run = run_public_arena(limit=5, provider_kind="core-fts-current")

    assert run.provider_name == "EnfoldCoreFtsCurrentProvider"
    assert run.summary["stale_leak@3"]["leaks"] == 0
    assert run.provider_metadata == {
        "provider_id": "enfold-core-fts-current-v1",
        "retrieval_stack": "standalone_core_store.search_fts+current_state_predicates",
        "schema_version": 1,
        "store": "temporary_synthetic_migrated_sqlite",
        "full_hybrid_production_retriever": False,
        "uses_live_data": False,
        "uses_embedding_service": False,
        "explicit_named_anchor_abstention": True,
    }


def test_core_provider_abstains_when_named_query_anchor_is_absent():
    arena = load_public_arena()
    with EnfoldCoreFtsCurrentProvider(arena) as provider:
        assert provider.search("What budget was approved for Project Ember?") == []


def test_offline_hybrid_arena_is_deterministic_and_honestly_identified():
    first = run_public_arena(limit=5, provider_kind="offline-hybrid-ci")
    second = run_public_arena(limit=5, provider_kind="offline-hybrid-ci")

    assert first.provider_name == "EnfoldOfflineHybridProvider"
    assert first.provider_metadata["full_hybrid_production_retriever"] is False
    assert first.provider_metadata["embedder_production_ready"] is False
    assert first.provider_metadata["ci_embedder_is_semantic_model"] is False
    assert first.provider_metadata["uses_live_data"] is False
    assert first.summary["stale_leak@3"]["leaks"] == 0
    assert [result.ranked_fact_ids for result in first.results] == [
        result.ranked_fact_ids for result in second.results
    ]
    assert [result.scores for result in first.results] == [
        result.scores for result in second.results
    ]


def test_offline_hybrid_keeps_conflicts_on_explicit_api_only():
    arena = load_public_arena()
    with EnfoldOfflineHybridProvider(arena) as provider:
        assert provider.search("Maple support coverage", category="work") == []
        assert {row["fact_id"] for row in provider.search_conflicts(
            "Maple support coverage", category="work"
        )} == {1301, 1302}


def test_public_report_includes_only_deliberately_synthetic_text(tmp_path):
    arena = load_public_arena()
    run = run_public_arena(PerfectFixtureProvider(arena), arena=arena, limit=5)
    output = tmp_path / "public-report.json"

    write_public_arena_report(output, run)

    report = json.loads(output.read_text())
    assert report["metadata"]["synthetic"] is True
    assert report["metadata"]["arena_version"] == "1.0"
    assert report["summary"]["cases"] == 14
    assert report["results"][0]["query"] == arena.cases[0].query
    assert report["results"][0]["top_results"][0]["content"] == arena.facts[0].content


def test_cli_prints_concise_offline_summary(capsys):
    assert main(["--limit", "3"]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["arena"] == "enfold-public-arena"
    assert summary["arena_version"] == "1.0"
    assert summary["provider"] == "EnfoldCoreFtsCurrentProvider"
    assert summary["cases"] == 14
    assert set(summary) == {
        "arena",
        "arena_version",
        "provider",
        "cases",
        "answerable_recall@1",
        "set_recall",
        "set_f1",
        "stale_leaks@3",
        "true_abstain",
        "false_confident",
    }


def test_cli_can_write_full_report_and_respects_limit(tmp_path, capsys):
    output = tmp_path / "arena-report.json"

    assert main(["--limit", "1", "--out", str(output)]) == 0

    capsys.readouterr()
    report = json.loads(output.read_text())
    assert report["metadata"]["provider"] == "EnfoldCoreFtsCurrentProvider"
    assert report["metadata"]["synthetic"] is True
    assert report["metadata"]["provider_metadata"]["full_hybrid_production_retriever"] is False
    assert len(report["metadata"]["fixture_sha256"]) == 64
    assert all(len(result["top_fact_ids"]) <= 1 for result in report["results"])


def test_cli_keeps_lexical_baseline_selectable(capsys):
    assert main(["--provider", "lexical", "--limit", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["provider"] == "LexicalFixtureProvider"


def test_cli_exposes_offline_hybrid_provider(capsys):
    assert main(["--provider", "offline-hybrid-ci", "--limit", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["provider"] == "EnfoldOfflineHybridProvider"


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_cli_rejects_invalid_limit(value):
    with pytest.raises(SystemExit) as exc:
        main(["--limit", value])

    assert exc.value.code == 2
