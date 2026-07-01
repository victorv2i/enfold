from __future__ import annotations

import pytest

from memory_eval.runner import EvalCase, run_retrieval_cases, summarize_results


class FakeProvider:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def search(self, query, *, category=None, min_trust=0.3, limit=10, bump=False):
        self.calls.append({
            "query": query,
            "category": category,
            "min_trust": min_trust,
            "limit": limit,
            "bump": bump,
        })
        return self.responses[query][:limit]


def test_run_retrieval_cases_calls_provider_without_bumping_counts():
    provider = FakeProvider({
        "dark theme": [
            {"fact_id": 2, "content": "Alex prefers dark UI", "score": 0.9},
            {"fact_id": 1, "content": "Alex used light UI", "score": 0.5},
        ]
    })
    cases = [EvalCase(id="pref-ui", query="dark theme", gold_fact_id=2, stale_fact_ids=[1])]

    results = run_retrieval_cases(provider, cases, limit=5)

    assert provider.calls == [{
        "query": "dark theme",
        "category": None,
        "min_trust": 0.3,
        "limit": 5,
        "bump": False,
    }]
    assert results[0].ranked_fact_ids == [2, 1]
    assert results[0].gold_rank == 1
    assert results[0].stale_leak_ranks == [2]
    assert results[0].latency_ms >= 0.0


def test_run_retrieval_cases_respects_case_filters_and_missing_gold():
    provider = FakeProvider({"unknown": [{"fact_id": 9, "content": "wrong"}]})
    cases = [EvalCase(
        id="missing",
        query="unknown",
        gold_fact_id=3,
        category="project",
        min_trust=0.7,
    )]

    results = run_retrieval_cases(provider, cases, limit=1)

    assert provider.calls[0]["category"] == "project"
    assert provider.calls[0]["min_trust"] == 0.7
    assert results[0].ranked_fact_ids == [9]
    assert results[0].gold_rank is None


def test_summarize_results_reports_retrieval_latency_and_stale_leaks():
    provider = FakeProvider({
        "q1": [{"fact_id": 10}, {"fact_id": 11}],
        "q2": [{"fact_id": 22}],
        "q3": [{"fact_id": 31}, {"fact_id": 30}],
    })
    cases = [
        EvalCase(id="hit1", query="q1", gold_fact_id=10),
        EvalCase(id="miss", query="q2", gold_fact_id=20),
        EvalCase(id="hit2", query="q3", gold_fact_id=30, stale_fact_ids=[31]),
    ]

    summary = summarize_results(run_retrieval_cases(provider, cases, limit=3), stale_k=1)

    assert summary["cases"] == 3
    assert summary["recall@1"] == 1 / 3
    assert summary["recall@3"] == 2 / 3
    assert summary["mrr"] == (1 + 0 + 0.5) / 3
    assert summary["stale_leak@1"]["leaks"] == 1
    assert "latency_ms" in summary


def test_summarize_results_uses_case_specific_stale_ids_for_leak_rate():
    provider = FakeProvider({
        "q1": [{"fact_id": 1}],
        "q2": [{"fact_id": 9}],
    })
    cases = [
        EvalCase(id="stale-for-q1-only", query="q1", gold_fact_id=1, stale_fact_ids=[9]),
        EvalCase(id="q2-current", query="q2", gold_fact_id=9),
    ]

    summary = summarize_results(run_retrieval_cases(provider, cases, limit=1), stale_k=1)

    assert summary["stale_leak@1"]["leaks"] == 0
    assert summary["stale_leak@1"]["cases_with_stale_ids"] == 1
    assert summary["stale_leak@1"]["exposed_case_leak_rate"] == 0.0


def test_summarize_results_reports_default_stale_k_three():
    provider = FakeProvider({"q": [{"fact_id": 10}, {"fact_id": 9}]})
    cases = [EvalCase(id="stale-at-two", query="q", gold_fact_id=10, stale_fact_ids=[9])]

    summary = summarize_results(run_retrieval_cases(provider, cases, limit=2))

    assert summary["stale_leak@1"]["leaks"] == 0
    assert summary["stale_leak@3"]["leaks"] == 1
    assert summary["stale_leak@3"]["best_leak_ranks"] == [2]


def test_summarize_results_handles_empty_results_list():
    summary = summarize_results([], score_thresholds=[], decision_rules=[])

    assert summary["cases"] == 0
    assert summary["answerable"]["cases"] == 0
    assert summary["abstention_decision"] == {
        "predicted_abstain": 0,
        "actual_abstain": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_memoryarena_v2_summary_groups_metrics_by_case_type():
    provider = FakeProvider({
        "current": [{"fact_id": 10}, {"fact_id": 11}],
        "stale": [{"fact_id": 21}, {"fact_id": 20}],
        "abstain": [{"fact_id": 30}],
    })
    cases = [
        EvalCase(
            id="cur-1",
            query="current",
            gold_fact_id=10,
            case_type="current_preference",
            expected_current_fact_ids=[10, 11],
        ),
        EvalCase(
            id="stale-1",
            query="stale",
            gold_fact_id=20,
            case_type="stale_suppression",
            stale_fact_ids=[21],
            expected_current_fact_ids=[20],
        ),
        EvalCase(
            id="abs-1",
            query="abstain",
            gold_fact_id=99,
            case_type="abstention",
            should_abstain=True,
        ),
    ]

    summary = summarize_results(run_retrieval_cases(provider, cases, limit=3), stale_k=1)

    assert summary["by_case_type"]["current_preference"]["cases"] == 1
    assert summary["by_case_type"]["current_preference"]["set_recall"] == 1.0
    assert summary["by_case_type"]["stale_suppression"]["stale_leak@1"]["leaks"] == 1
    assert summary["by_case_type"]["abstention"]["abstention"]["false_confident"] == 1


def test_should_abstain_cases_do_not_contribute_to_set_metrics_even_with_expected_ids():
    provider = FakeProvider({"unknown": [{"fact_id": 10, "score": 0.9}]})
    cases = [EvalCase(
        id="conflicting-abstain",
        query="unknown",
        gold_fact_id=-1,
        expected_current_fact_ids=[10],
        should_abstain=True,
    )]

    summary = summarize_results(run_retrieval_cases(provider, cases, limit=1), score_thresholds=[])

    assert summary["set_cases"] == 0
    assert summary["abstention"]["false_confident"] == 1


def test_abstention_metrics_treat_any_expected_current_id_as_answered():
    provider = FakeProvider({"multi": [{"fact_id": 12, "score": 0.9}]})
    cases = [EvalCase(
        id="multi-expected",
        query="multi",
        gold_fact_id=10,
        expected_current_fact_ids=[10, 12],
    )]

    summary = summarize_results(run_retrieval_cases(provider, cases, limit=1), score_thresholds=[])

    assert summary["abstention"]["correct_answer"] == 1
    assert summary["abstention"]["false_abstain"] == 0
    assert summary["set_recall"] == 0.5


def test_abstention_metrics_do_not_count_stale_expected_ids_as_answered():
    provider = FakeProvider({"multi": [{"fact_id": 12, "score": 0.9}]})
    cases = [EvalCase(
        id="multi-expected-stale",
        query="multi",
        gold_fact_id=10,
        expected_current_fact_ids=[10, 12],
        stale_fact_ids=[12],
    )]

    summary = summarize_results(run_retrieval_cases(provider, cases, limit=1), score_thresholds=[])

    assert summary["abstention"]["correct_answer"] == 0
    assert summary["abstention"]["false_abstain"] == 1
    assert summary["set_recall"] == 0.0


def test_set_metrics_filter_stale_predictions_before_precision():
    provider = FakeProvider({"q": [{"fact_id": 5}, {"fact_id": 10}]})
    cases = [EvalCase(
        id="stale-then-current",
        query="q",
        gold_fact_id=10,
        expected_current_fact_ids=[10],
        stale_fact_ids=[5],
    )]

    summary = summarize_results(run_retrieval_cases(provider, cases, limit=2), score_thresholds=[])

    assert summary["set_precision"] == 1.0
    assert summary["set_recall"] == 1.0


def test_score_threshold_summary_turns_low_score_abstention_hits_into_true_abstains():
    provider = FakeProvider({
        "answerable": [{"fact_id": 10, "score": 0.82}, {"fact_id": 11, "score": 0.3}],
        "unknown": [{"fact_id": 99, "score": 0.21}],
    })
    cases = [
        EvalCase(id="ans", query="answerable", gold_fact_id=10, case_type="current_fact"),
        EvalCase(id="abs", query="unknown", gold_fact_id=-1, case_type="abstention", should_abstain=True),
    ]

    results = run_retrieval_cases(provider, cases, limit=2)
    summary = summarize_results(results, score_thresholds=[0.5])

    assert results[0].top_score == 0.82
    assert results[1].top_score == 0.21
    threshold = summary["score_thresholds"]["0.5"]
    assert threshold["accepted_queries"] == 1
    assert threshold["recall@1"] == 0.5
    assert threshold["abstention"]["true_abstain"] == 1
    assert threshold["abstention"]["correct_answer"] == 1


def test_decision_rule_summary_combines_score_and_margin_for_abstention():
    provider = FakeProvider({
        "answerable": [
            {"fact_id": 10, "score": 0.72, "trust_score": 0.9, "category": "project"},
            {"fact_id": 11, "score": 0.55, "trust_score": 0.9, "category": "project"},
        ],
        "ambiguous unknown": [
            {"fact_id": 98, "score": 0.71, "trust_score": 0.8, "category": "general"},
            {"fact_id": 99, "score": 0.70, "trust_score": 0.8, "category": "general"},
        ],
        "weak answerable": [
            {"fact_id": 20, "score": 0.49, "trust_score": 0.9, "category": "project"},
        ],
    })
    cases = [
        EvalCase(id="ans", query="answerable", gold_fact_id=10, case_type="current_fact"),
        EvalCase(id="abs", query="ambiguous unknown", gold_fact_id=-1, case_type="abstention", should_abstain=True),
        EvalCase(id="weak", query="weak answerable", gold_fact_id=20, case_type="current_fact"),
    ]

    results = run_retrieval_cases(provider, cases, limit=2)
    summary = summarize_results(
        results,
        score_thresholds=[],
        decision_rules=[{"name": "score_margin", "min_score": 0.5, "min_margin": 0.05}],
    )

    assert results[0].score_margin == pytest.approx(0.17)
    strict = summary["decision_rules"]["score_margin"]
    assert strict["accepted_queries"] == 1
    assert strict["recall@1"] == pytest.approx(1 / 3)
    assert strict["abstention"]["correct_answer"] == 1
    assert strict["abstention"]["true_abstain"] == 1
    assert strict["abstention"]["false_abstain"] == 1
    assert strict["answerable"]["cases"] == 2
    assert strict["answerable"]["recall@1"] == 0.5
    assert strict["abstention_decision"]["precision"] == 0.5
    assert strict["abstention_decision"]["recall"] == 1.0
    assert strict["abstention_decision"]["f1"] == pytest.approx(2 / 3)
    assert strict["abstention_decision"]["predicted_abstain"] == 2
    assert strict["abstention_decision"]["actual_abstain"] == 1


def test_margin_gate_documents_single_result_behavior():
    provider = FakeProvider({
        "single confident": [{"fact_id": 10, "score": 0.9}],
        "close scores": [{"fact_id": 20, "score": 0.9}, {"fact_id": 21, "score": 0.87}],
    })
    cases = [
        EvalCase(id="single", query="single confident", gold_fact_id=10),
        EvalCase(id="close", query="close scores", gold_fact_id=20),
    ]

    summary = summarize_results(
        run_retrieval_cases(provider, cases, limit=2),
        score_thresholds=[],
        decision_rules=[{"name": "wide_margin", "min_score": 0.5, "min_margin": 0.3}],
    )

    wide = summary["decision_rules"]["wide_margin"]
    assert wide["accepted_queries"] == 1
    assert wide["answerable"]["recall@1"] == 0.5


def test_decision_rule_summary_can_filter_category_trust_and_stale_prefixes():
    provider = FakeProvider({
        "category": [
            {"fact_id": 9, "score": 0.9, "trust_score": 0.95, "category": "user_pref"},
            {"fact_id": 40, "score": 0.7, "trust_score": 0.95, "category": "project"},
        ],
        "stale": [
            {"fact_id": 50, "content": "Superseded 2026-06-28. old value", "score": 0.93, "trust_score": 0.9},
            {"fact_id": 51, "content": "current value", "score": 0.82, "trust_score": 0.9},
        ],
        "low trust": [
            {"fact_id": 60, "score": 0.88, "trust_score": 0.2},
            {"fact_id": 61, "score": 0.81, "trust_score": 0.8},
        ],
    })
    cases = [
        EvalCase(id="cat", query="category", gold_fact_id=40, category="project"),
        EvalCase(id="stale", query="stale", gold_fact_id=51, stale_fact_ids=[50]),
        EvalCase(id="trust", query="low trust", gold_fact_id=61),
    ]

    summary = summarize_results(
        run_retrieval_cases(provider, cases, limit=2),
        score_thresholds=[],
        decision_rules=[{
            "name": "guarded",
            "require_category_match": True,
            "reject_stale_prefix": True,
            "min_trust": 0.5,
        }],
    )

    guarded = summary["decision_rules"]["guarded"]
    assert guarded["accepted_queries"] == 3
    assert guarded["recall@1"] == 1.0
    assert guarded["stale_leak@1"]["leaks"] == 0
