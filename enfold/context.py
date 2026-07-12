"""Deterministic prompt-context packing for Enfold's v1 fast lane.

This module has no database, model, or transport dependency.  It receives
already-authorized, already-current search rows from the service and turns a
bounded prefix of them into a compact Markdown block.  Keeping packing pure
makes its token-budget and conflict behavior directly testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any


TOKEN_ESTIMATE_METHOD = "unicode_chars_divided_by_four"
_HEADER = "## Enfold Memory\n"
_ELLIPSIS = "…"
TRUNCATION_MARKER = "… [truncated]"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def estimate_tokens(text: str) -> int:
    """Return a deterministic four-characters-per-token estimate for ``text``.

    It intentionally is not a model tokenizer.  The common four Unicode
    characters per token heuristic is stable across agent surfaces and avoids
    the previous fourfold under-filling without adding a tokenizer dependency.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return (len(text) + 3) // 4


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A prompt-ready context payload plus transparent packing metadata."""

    markdown: str
    facts: tuple[dict[str, Any], ...]
    abstained: bool
    token_budget: int
    token_estimate: int
    omitted_fact_count: int
    unsafe_fact_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "markdown": self.markdown,
            "facts": [dict(fact) for fact in self.facts],
            "abstained": self.abstained,
            "token_estimate": {
                "method": TOKEN_ESTIMATE_METHOD,
                "budget": self.token_budget,
                "used": self.token_estimate,
            },
            "omitted_fact_count": self.omitted_fact_count,
            "unsafe_fact_count": self.unsafe_fact_count,
        }


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_RE.findall(left.lower()))
    right_tokens = set(_TOKEN_RE.findall(right.lower()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _embedding_similarity(left: Any, right: Any) -> float | None:
    """Return cosine similarity only for complete, finite matching vectors."""

    if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)):
        return None
    try:
        left_values = tuple(float(value) for value in left)
        right_values = tuple(float(value) for value in right)
    except (TypeError, ValueError):
        return None
    if not left_values or len(left_values) != len(right_values):
        return None
    if not all(math.isfinite(value) for value in (*left_values, *right_values)):
        return None
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if not left_norm or not right_norm:
        return None
    return max(-1.0, min(1.0, sum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ) / (left_norm * right_norm)))


def _similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    dense = _embedding_similarity(left.get("_mmr_embedding"), right.get("_mmr_embedding"))
    if dense is not None:
        return dense
    return _jaccard_similarity(_text(left.get("content")), _text(right.get("content")))


def _mmr_select(
    facts: Sequence[Mapping[str, Any]], *, max_facts: int | None, mmr_lambda: float
) -> Sequence[Mapping[str, Any]]:
    if max_facts is None or len(facts) <= max_facts:
        return facts
    remaining = list(enumerate(facts))
    selected: list[Mapping[str, Any]] = []
    while remaining and len(selected) < max_facts:
        def objective(item: tuple[int, Mapping[str, Any]]) -> tuple[float, int]:
            index, fact = item
            try:
                relevance = float(fact.get("score", 0.0))
            except (TypeError, ValueError):
                relevance = 0.0
            redundancy = max((_similarity(fact, chosen) for chosen in selected), default=0.0)
            return mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy, -index

        best = max(remaining, key=objective)
        remaining.remove(best)
        selected.append(best[1])
    return selected


def _safe_candidate(fact: Mapping[str, Any]) -> bool:
    """Defend the context boundary even if a retriever regresses upstream."""

    return (
        fact.get("invalid_at") is None
        and fact.get("superseded_by") is None
        and fact.get("conflict_group") is None
    )


def _state_slot(fact: Mapping[str, Any]) -> tuple[str, str, str] | None:
    if fact.get("memory_kind") != "state":
        return None
    subject = _text(fact.get("subject_key"))
    predicate = _text(fact.get("predicate_key"))
    scope = _text(fact.get("scope"))
    if not subject or not predicate:
        return None
    return scope, subject, predicate


def _compact_source(attribution: Any) -> str:
    if not isinstance(attribution, Mapping):
        return "unknown"
    for key in ("performed_by", "agent_id", "source_type"):
        value = _text(attribution.get(key))
        if value:
            return value[:64] + (_ELLIPSIS if len(value) > 64 else "")
    return "unknown"


def _line_prefix(fact: Mapping[str, Any]) -> str:
    fact_id = fact.get("fact_id")
    score = fact.get("score")
    try:
        score_text = f"{float(score):.3f}"
    except (TypeError, ValueError):
        score_text = "n/a"
    attribution = fact.get("attribution")
    evidence_count = (
        attribution.get("evidence_count")
        if isinstance(attribution, Mapping)
        else None
    )
    evidence = f" evidence:{evidence_count}" if isinstance(evidence_count, int) else ""
    return f"- [fact:{fact_id} score:{score_text} by:{_compact_source(attribution)}{evidence}] "


def _truncate_to_budget(text: str, budget: int) -> tuple[str, bool]:
    """Return the largest deterministic Unicode-safe prefix fitting ``budget``."""

    if budget <= 0:
        return "", bool(text)
    if estimate_tokens(text) <= budget:
        return text, False
    if estimate_tokens(_ELLIPSIS) > budget:
        return "", True

    low, high = 0, len(text)
    best = ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = text[:midpoint].rstrip() + _ELLIPSIS
        if estimate_tokens(candidate) <= budget:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best, True


def _truncate_to_chars(text: str, maximum: int) -> tuple[str, bool]:
    if len(text) <= maximum:
        return text, False
    prefix = text[:maximum - len(TRUNCATION_MARKER)].rstrip()
    return prefix + TRUNCATION_MARKER, True


def pack_context(
    facts: Sequence[Mapping[str, Any]],
    *,
    token_budget: int,
    max_fact_chars: int | None = None,
    max_facts: int | None = None,
    mmr_lambda: float = 0.7,
) -> ContextPack:
    """Pack ranked safe facts in input order without exceeding ``token_budget``.

    The input order is authoritative ranking order.  At most one current fact
    per exact state slot is selected; non-state facts coexist normally.  The
    returned structured facts carry an explicit ``context_truncated`` marker
    when their Markdown content was shortened to honor the conservative budget.
    """

    if isinstance(token_budget, bool) or not isinstance(token_budget, int):
        raise ValueError("token_budget must be an integer")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if max_fact_chars is not None and max_fact_chars < len(TRUNCATION_MARKER):
        raise ValueError("max_fact_chars is too small for the truncation marker")
    if max_facts is not None and (
        isinstance(max_facts, bool) or not isinstance(max_facts, int) or max_facts < 1
    ):
        raise ValueError("max_facts must be a positive integer")
    if isinstance(mmr_lambda, bool) or not isinstance(mmr_lambda, (int, float)) or not math.isfinite(mmr_lambda) or not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("mmr_lambda must be between 0 and 1")

    header_tokens = estimate_tokens(_HEADER)
    if header_tokens > token_budget:
        return ContextPack("", (), True, token_budget, 0, len(facts), 0)

    markdown = _HEADER
    selected: list[dict[str, Any]] = []
    selected_slots: set[tuple[str, str, str]] = set()
    omitted = 0
    unsafe = 0

    ranked_facts = _mmr_select(facts, max_facts=max_facts, mmr_lambda=float(mmr_lambda))
    omitted += len(facts) - len(ranked_facts)
    for raw_fact in ranked_facts:
        fact = dict(raw_fact)
        fact.pop("_mmr_embedding", None)
        if not _safe_candidate(fact):
            unsafe += 1
            continue
        slot = _state_slot(fact)
        if slot is not None and slot in selected_slots:
            omitted += 1
            continue
        prefix = _line_prefix(fact)
        available = token_budget - estimate_tokens(markdown) - estimate_tokens(prefix) - 1
        content = _text(fact.get("content"))
        content_truncated = False
        if max_fact_chars is not None:
            content, content_truncated = _truncate_to_chars(content, max_fact_chars)
        content, budget_truncated = _truncate_to_budget(content, available)
        truncated = content_truncated or budget_truncated
        if budget_truncated and len(content.rstrip(_ELLIPSIS).rstrip()) < 8:
            omitted += 1
            continue
        if not content:
            omitted += 1
            continue
        line = prefix + content + "\n"
        if estimate_tokens(markdown + line) > token_budget:
            # Defensive against a future estimate implementation change.
            omitted += 1
            continue
        fact["content"] = content
        fact["context_truncated"] = truncated
        if content_truncated:
            fact["content_truncated"] = True
        selected.append(fact)
        if slot is not None:
            selected_slots.add(slot)
        markdown += line

    if not selected:
        return ContextPack("", (), True, token_budget, 0, omitted, unsafe)
    used = estimate_tokens(markdown)
    return ContextPack(
        markdown,
        tuple(selected),
        False,
        token_budget,
        used,
        omitted,
        unsafe,
    )
