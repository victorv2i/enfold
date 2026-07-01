from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .metrics import mean_reciprocal_rank, percentile, precision_recall_f1, recall_at_k

_DEFAULT_SCORE_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
_DEFAULT_DECISION_RULES: tuple[dict[str, Any], ...] = (
    {"name": "score>=0.5_margin>=0.02", "min_score": 0.5, "min_margin": 0.02},
    {"name": "score>=0.5_margin>=0.05", "min_score": 0.5, "min_margin": 0.05},
    {"name": "score>=0.5_trust>=0.5", "min_score": 0.5, "min_trust": 0.5},
    {"name": "score>=0.5_no_stale_prefix", "min_score": 0.5, "reject_stale_prefix": True},
    {
        "name": "score>=0.5_margin>=0.02_trust>=0.5_no_stale_prefix",
        "min_score": 0.5,
        "min_margin": 0.02,
        "min_trust": 0.5,
        "reject_stale_prefix": True,
    },
)
STALE_CONTENT_PREFIXES = (
    "superseded",
    "stale/disabled",
    "historical/superseded",
)


class SearchProvider(Protocol):
    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        bump: bool = False,
    ) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    gold_fact_id: int
    category: str | None = None
    min_trust: float = 0.3
    stale_fact_ids: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    case_type: str = "exact_fact"
    expected_current_fact_ids: list[int] = field(default_factory=list)
    entity_refs: list[str] = field(default_factory=list)
    provenance: dict[str, Any] | None = None
    answer_rubric: dict[str, Any] | None = None
    difficulty: str = "easy"
    generation: str = "auto"
    privacy_tier: str = "private"
    should_abstain: bool = False


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    ranked_fact_ids: list[int]
    gold_rank: int | None
    stale_leak_ranks: list[int]
    latency_ms: float
    results: list[dict[str, Any]]
    scores: list[float | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.scores:
            object.__setattr__(self, "scores", [_row_score(row) for row in self.results])

    @property
    def top_score(self) -> float | None:
        return self.scores[0] if self.scores else None

    @property
    def second_score(self) -> float | None:
        return self.scores[1] if len(self.scores) > 1 else None

    @property
    def score_margin(self) -> float | None:
        if self.top_score is None or self.second_score is None:
            return None
        return self.top_score - self.second_score


def _fact_id(row: dict[str, Any]) -> int:
    return int(row["fact_id"])


def _row_score(row: dict[str, Any]) -> float | None:
    value = row.get("score")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_trust(row: dict[str, Any]) -> float | None:
    value = row.get("trust_score")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_has_stale_prefix(row: dict[str, Any]) -> bool:
    content = str(row.get("content") or "").strip().lower()
    return any(content.startswith(prefix) for prefix in STALE_CONTENT_PREFIXES)


def _rule_float(rule: dict[str, Any], key: str) -> float | None:
    if key not in rule or rule[key] is None:
        return None
    try:
        return float(rule[key])
    except (TypeError, ValueError):
        return None


def _decision_rule_key(rule: dict[str, Any]) -> str:
    if rule.get("name"):
        return str(rule["name"])
    parts = []
    for key in sorted(rule):
        parts.append(f"{key}={rule[key]}")
    return ",".join(parts) or "unnamed"


def _passes_result_gates(result: EvalResult, decision_rule: dict[str, Any] | None) -> bool:
    if not decision_rule:
        return True
    min_margin = _rule_float(decision_rule, "min_margin")
    # A margin exists only when the provider returned at least two scored rows;
    # single-result queries are left to score/trust/category gates.
    if min_margin is not None and result.score_margin is not None and result.score_margin < min_margin:
        return False
    return True


def _passes_row_gates(
    result: EvalResult,
    row: dict[str, Any],
    score: float | None,
    *,
    score_threshold: float | None,
    decision_rule: dict[str, Any] | None,
) -> bool:
    threshold = score_threshold
    if decision_rule:
        rule_threshold = _rule_float(decision_rule, "min_score")
        if rule_threshold is not None:
            threshold = rule_threshold if threshold is None else max(threshold, rule_threshold)
    if threshold is not None and (score is None or score < threshold):
        return False

    if decision_rule:
        min_trust = _rule_float(decision_rule, "min_trust")
        if min_trust is not None:
            trust = _row_trust(row)
            if trust is None or trust < min_trust:
                return False
        if decision_rule.get("require_category_match") and result.case.category is not None:
            if row.get("category") != result.case.category:
                return False
        if decision_rule.get("reject_stale_prefix") and _row_has_stale_prefix(row):
            return False
    return True


def _ranked_ids_at(
    result: EvalResult,
    score_threshold: float | None = None,
    *,
    decision_rule: dict[str, Any] | None = None,
) -> list[int]:
    if score_threshold is None and decision_rule is None:
        return result.ranked_fact_ids
    if not _passes_result_gates(result, decision_rule):
        return []
    ranked: list[int] = []
    for row, score in zip(result.results, result.scores, strict=False):
        if _passes_row_gates(result, row, score, score_threshold=score_threshold, decision_rule=decision_rule):
            ranked.append(_fact_id(row))
    return ranked


def run_retrieval_cases(provider: SearchProvider, cases: list[EvalCase], *, limit: int = 10) -> list[EvalResult]:
    """Run retrieval cases through a provider without mutating retrieval counts."""
    results: list[EvalResult] = []
    for case in cases:
        started = time.perf_counter()
        rows = provider.search(
            case.query,
            category=case.category,
            min_trust=case.min_trust,
            limit=limit,
            bump=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranked_ids = [_fact_id(row) for row in rows]
        scores = [_row_score(row) for row in rows]
        gold_rank = ranked_ids.index(case.gold_fact_id) + 1 if case.gold_fact_id in ranked_ids else None
        stale = set(case.stale_fact_ids)
        stale_ranks = [idx + 1 for idx, fact_id in enumerate(ranked_ids) if fact_id in stale]
        results.append(EvalResult(
            case=case,
            ranked_fact_ids=ranked_ids,
            gold_rank=gold_rank,
            stale_leak_ranks=stale_ranks,
            latency_ms=latency_ms,
            results=rows,
            scores=scores,
        ))
    return results


def _expected_current_ids(case: EvalCase) -> list[int]:
    if case.should_abstain:
        return []
    if case.expected_current_fact_ids:
        return case.expected_current_fact_ids
    return [case.gold_fact_id]


def _set_metrics(
    results: list[EvalResult],
    *,
    score_threshold: float | None = None,
    decision_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scored = []
    for result in results:
        expected = _expected_current_ids(result.case)
        if not expected:
            continue
        stale = set(result.case.stale_fact_ids)
        predicted = [
            fact_id
            for fact_id in _ranked_ids_at(result, score_threshold, decision_rule=decision_rule)
            if fact_id not in stale
        ]
        scored.append(precision_recall_f1(predicted, expected))
    if not scored:
        return {"set_cases": 0, "set_precision": 0.0, "set_recall": 0.0, "set_f1": 0.0}
    return {
        "set_cases": len(scored),
        "set_precision": sum(row["precision"] for row in scored) / len(scored),
        "set_recall": sum(row["recall"] for row in scored) / len(scored),
        "set_f1": sum(row["f1"] for row in scored) / len(scored),
    }


def _abstention_metrics(
    results: list[EvalResult],
    *,
    score_threshold: float | None = None,
    decision_rule: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Score abstention as a binary decision using any expected current fact as a hit."""
    metrics = {
        "true_abstain": 0,
        "false_abstain": 0,
        "false_confident": 0,
        "correct_answer": 0,
    }
    for result in results:
        ranked = _ranked_ids_at(result, score_threshold, decision_rule=decision_rule)
        if result.case.should_abstain:
            if ranked:
                metrics["false_confident"] += 1
            else:
                metrics["true_abstain"] += 1
        elif not _abstention_hit(result, ranked):
            metrics["false_abstain"] += 1
        else:
            metrics["correct_answer"] += 1
    return metrics


def _abstention_hit(result: EvalResult, ranked: list[int]) -> bool:
    expected = result.case.expected_current_fact_ids or [result.case.gold_fact_id]
    stale = set(result.case.stale_fact_ids)
    return any(fact_id in ranked and fact_id not in stale for fact_id in expected)


def _answerable_metrics(
    results: list[EvalResult],
    *,
    score_threshold: float | None = None,
    decision_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize retrieval quality against the primary gold fact.

    Multi-answer cases are covered separately by ``set_recall`` / ``set_f1``.
    """
    answerable = [result for result in results if not result.case.should_abstain]
    ranked = [_ranked_ids_at(r, score_threshold, decision_rule=decision_rule) for r in answerable]
    gold = [r.case.gold_fact_id for r in answerable]
    return {
        "cases": len(answerable),
        "accepted_queries": sum(1 for row in ranked if row),
        "recall@1": recall_at_k(ranked, gold, 1),
        "recall@3": recall_at_k(ranked, gold, 3),
        "recall@5": recall_at_k(ranked, gold, 5),
        "recall@10": recall_at_k(ranked, gold, 10),
        "mrr": mean_reciprocal_rank(ranked, gold),
    }


def _abstention_decision_metrics(abstention: dict[str, int]) -> dict[str, Any]:
    true_abstain = abstention["true_abstain"]
    false_abstain = abstention["false_abstain"]
    false_confident = abstention["false_confident"]
    precision_denominator = true_abstain + false_abstain
    recall_denominator = true_abstain + false_confident
    precision = 0.0 if precision_denominator == 0 else true_abstain / precision_denominator
    recall = 0.0 if recall_denominator == 0 else true_abstain / recall_denominator
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "predicted_abstain": precision_denominator,
        "actual_abstain": recall_denominator,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _stale_leak_summary(
    results: list[EvalResult],
    *,
    k: int,
    score_threshold: float | None = None,
    decision_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")
    best_leak_ranks: list[int] = []
    cases_with_stale_ids = 0
    for result in results:
        stale = set(result.case.stale_fact_ids)
        if stale:
            cases_with_stale_ids += 1
        top = _ranked_ids_at(result, score_threshold, decision_rule=decision_rule)[:k]
        leak_ranks = [idx + 1 for idx, fact_id in enumerate(top) if fact_id in stale]
        if leak_ranks:
            best_leak_ranks.append(min(leak_ranks))
    queries = len(results)
    leaks = len(best_leak_ranks)
    return {
        "queries": queries,
        "cases_with_stale_ids": cases_with_stale_ids,
        "leaks": leaks,
        "leak_rate": 0.0 if queries == 0 else leaks / queries,
        "exposed_case_leak_rate": 0.0 if cases_with_stale_ids == 0 else leaks / cases_with_stale_ids,
        "best_leak_ranks": best_leak_ranks,
    }


def _top_score_summary(results: list[EvalResult]) -> dict[str, float]:
    top_scores = [score for score in (r.top_score for r in results) if score is not None]
    return {
        "p50": percentile(top_scores, 50),
        "p95": percentile(top_scores, 95),
        "min": min(top_scores) if top_scores else 0.0,
        "max": max(top_scores) if top_scores else 0.0,
    }


def _score_margin_summary(results: list[EvalResult]) -> dict[str, float]:
    margins = [margin for margin in (r.score_margin for r in results) if margin is not None]
    return {
        "p50": percentile(margins, 50),
        "p95": percentile(margins, 95),
        "min": min(margins) if margins else 0.0,
        "max": max(margins) if margins else 0.0,
    }


def _summarize_flat(
    results: list[EvalResult],
    *,
    stale_k: int,
    score_threshold: float | None = None,
    decision_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranked = [_ranked_ids_at(r, score_threshold, decision_rule=decision_rule) for r in results]
    gold = [r.case.gold_fact_id for r in results]
    latencies = [r.latency_ms for r in results]

    abstention = _abstention_metrics(results, score_threshold=score_threshold, decision_rule=decision_rule)
    stale_leak_at_1 = _stale_leak_summary(
        results,
        k=1,
        score_threshold=score_threshold,
        decision_rule=decision_rule,
    )
    summary = {
        "cases": len(results),
        "accepted_queries": sum(1 for row in ranked if row),
        "recall@1": recall_at_k(ranked, gold, 1),
        "recall@3": recall_at_k(ranked, gold, 3),
        "recall@5": recall_at_k(ranked, gold, 5),
        "recall@10": recall_at_k(ranked, gold, 10),
        "mrr": mean_reciprocal_rank(ranked, gold),
        "stale_leak@1": stale_leak_at_1,
        "latency_ms": {
            "mean": 0.0 if not latencies else sum(latencies) / len(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies) if latencies else 0.0,
        },
        "top_score": _top_score_summary(results),
        "score_margin": _score_margin_summary(results),
        "answerable": _answerable_metrics(results, score_threshold=score_threshold, decision_rule=decision_rule),
        "abstention": abstention,
        "abstention_decision": _abstention_decision_metrics(abstention),
    }
    if stale_k != 1:
        summary[f"stale_leak@{stale_k}"] = _stale_leak_summary(
            results,
            k=stale_k,
            score_threshold=score_threshold,
            decision_rule=decision_rule,
        )
    summary.update(_set_metrics(results, score_threshold=score_threshold, decision_rule=decision_rule))
    return summary


def _threshold_key(threshold: float) -> str:
    return f"{threshold:g}"


def summarize_results(
    results: list[EvalResult],
    *,
    stale_k: int = 3,
    score_thresholds: list[float] | tuple[float, ...] | None = _DEFAULT_SCORE_THRESHOLDS,
    decision_rules: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = _DEFAULT_DECISION_RULES,
) -> dict[str, Any]:
    """Summarize retrieval results with ranking, latency, stale-leak, and v2 metrics.

    Top-level recall/MRR include all cases, including should-abstain cases with
    sentinel gold IDs. Use the nested ``answerable`` block for retrieval-only
    recall that excludes abstention/negative cases.
    """
    summary = _summarize_flat(results, stale_k=stale_k)
    if score_thresholds:
        summary["score_thresholds"] = {
            _threshold_key(threshold): _summarize_flat(results, stale_k=stale_k, score_threshold=threshold)
            for threshold in score_thresholds
        }
    if decision_rules:
        summary["decision_rules"] = {
            _decision_rule_key(rule): _summarize_flat(results, stale_k=stale_k, decision_rule=rule)
            for rule in decision_rules
        }
    by_case_type: dict[str, list[EvalResult]] = {}
    for result in results:
        by_case_type.setdefault(result.case.case_type, []).append(result)
    summary["by_case_type"] = {
        case_type: _summarize_flat(case_results, stale_k=stale_k)
        for case_type, case_results in sorted(by_case_type.items())
    }
    return summary
