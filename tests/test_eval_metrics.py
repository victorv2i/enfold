from __future__ import annotations

import pytest

from memory_eval.metrics import (
    mean_reciprocal_rank,
    precision_recall_f1,
    percentile,
    recall_at_k,
    stale_leak_rate,
)


def test_recall_at_k_counts_queries_with_missing_gold_as_misses():
    ranked = [["a", "b"], ["x"], []]
    gold = ["b", "missing", "z"]

    assert recall_at_k(ranked, gold, k=1) == pytest.approx(0.0)
    assert recall_at_k(ranked, gold, k=2) == pytest.approx(1 / 3)


def test_mean_reciprocal_rank_uses_zero_for_unfound_gold():
    ranked = [["a", "b", "c"], ["x", "y"], ["m"]]
    gold = ["a", "y", "absent"]

    assert mean_reciprocal_rank(ranked, gold) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_percentile_uses_nearest_rank_with_sorted_copy():
    values = [100.0, 1.0, 50.0, 25.0]

    assert percentile(values, 50) == pytest.approx(25.0)
    assert percentile(values, 95) == pytest.approx(100.0)


def test_precision_recall_f1_handles_empty_predictions_and_labels():
    assert precision_recall_f1([], []) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert precision_recall_f1(["a"], []) == {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    assert precision_recall_f1([], ["a"]) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_precision_recall_f1_scores_set_overlap():
    metrics = precision_recall_f1(["a", "b", "c"], ["b", "c", "d"])

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_stale_leak_rate_counts_any_stale_fact_in_top_k():
    ranked = [[1, 2, 3], [4, 5], [6]]
    stale_ids = {2, 9, 6}

    result = stale_leak_rate(ranked, stale_ids, k=2)

    assert result["queries"] == 3
    assert result["leaks"] == 2
    assert result["leak_rate"] == pytest.approx(2 / 3)
    assert result["best_leak_ranks"] == [2, 1]
