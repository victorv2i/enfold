from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any


def _as_list(values: Iterable[Any]) -> list[Any]:
    return list(values)


def recall_at_k(ranked_per_query: Sequence[Sequence[Any]], gold_ids: Sequence[Any], k: int) -> float:
    """Return the fraction of queries whose gold id appears in the top k results."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(ranked_per_query) != len(gold_ids):
        raise ValueError("ranked_per_query and gold_ids must have the same length")
    if not gold_ids:
        return 0.0
    hits = 0
    for ranked, gold in zip(ranked_per_query, gold_ids, strict=True):
        if gold in list(ranked)[:k]:
            hits += 1
    return hits / len(gold_ids)


def mean_reciprocal_rank(ranked_per_query: Sequence[Sequence[Any]], gold_ids: Sequence[Any]) -> float:
    """Return MRR with unfound gold ids counted as zero."""
    if len(ranked_per_query) != len(gold_ids):
        raise ValueError("ranked_per_query and gold_ids must have the same length")
    if not gold_ids:
        return 0.0
    total = 0.0
    for ranked, gold in zip(ranked_per_query, gold_ids, strict=True):
        ranked_list = list(ranked)
        if gold in ranked_list:
            total += 1.0 / (ranked_list.index(gold) + 1)
    return total / len(gold_ids)


def percentile(values: Iterable[float], p: float) -> float:
    """Nearest-rank percentile for small benchmark samples."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if p < 0 or p > 100:
        raise ValueError("p must be between 0 and 100")
    rank = math.ceil((p / 100.0) * len(vals))
    idx = max(0, min(len(vals) - 1, rank - 1))
    return vals[idx]


def precision_recall_f1(predicted: Iterable[Any], expected: Iterable[Any]) -> dict[str, float]:
    """Compute set precision/recall/F1 for unordered labels."""
    pred = set(_as_list(predicted))
    exp = set(_as_list(expected))
    if not pred and not exp:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not exp:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    overlap = len(pred & exp)
    precision = overlap / len(pred)
    recall = overlap / len(exp)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def stale_leak_rate(ranked_per_query: Sequence[Sequence[Any]], stale_ids: set[Any], k: int) -> dict[str, Any]:
    """Measure queries where any globally stale/superseded fact leaks into top k.

    ``stale_ids`` is applied to every query. For case-specific stale sets, use
    the eval runner's summary path, which reads each case's stale IDs.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    best_leak_ranks: list[int] = []
    for ranked in ranked_per_query:
        top = list(ranked)[:k]
        leak_ranks = [idx + 1 for idx, fact_id in enumerate(top) if fact_id in stale_ids]
        if leak_ranks:
            best_leak_ranks.append(min(leak_ranks))
    queries = len(ranked_per_query)
    leaks = len(best_leak_ranks)
    return {
        "queries": queries,
        "leaks": leaks,
        "leak_rate": 0.0 if queries == 0 else leaks / queries,
        "best_leak_ranks": best_leak_ranks,
    }
