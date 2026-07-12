"""Small offline acceptance Arena for Enfold's v1 prompt-context fast lane.

The fixture contains only fictional facts.  It builds an in-memory migrated
store, writes through the public v1 service, and evaluates the read-only
``memory.context`` response without opening a live database or a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any

from enfold.policy import MemoryPolicy
from enfold.protocol import ClientContext, Request
from enfold.schema import migrate
from enfold.service import EnfoldService


CONTEXT_ARENA_PATH = Path(__file__).with_name("fixtures") / "context_arena.json"


@dataclass(frozen=True, slots=True)
class ContextArenaFact:
    key: str
    content: str
    scope: str
    source_authority: float
    state: dict[str, str] | None


@dataclass(frozen=True, slots=True)
class ContextArenaCase:
    id: str
    query: str
    scope: str
    token_budget: int
    expected_fact_keys: tuple[str, ...]
    forbidden_fact_keys: tuple[str, ...]
    abstained: bool
    require_attribution: bool
    require_truncation: bool = False


@dataclass(frozen=True, slots=True)
class ContextArena:
    version: str
    facts: tuple[ContextArenaFact, ...]
    cases: tuple[ContextArenaCase, ...]
    source_path: Path


@dataclass(frozen=True, slots=True)
class ContextArenaResult:
    case_id: str
    fact_ids: tuple[int, ...]
    abstained: bool
    passed: bool
    failures: tuple[str, ...]
    response: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ContextArenaRun:
    arena: ContextArena
    results: tuple[ContextArenaResult, ...]
    fact_ids_by_key: dict[str, int]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _keys(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    keys = tuple(_text(item, f"{label}[]") for item in value)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} must not contain duplicates")
    return keys


def _fact(value: Any, index: int) -> ContextArenaFact:
    label = f"facts[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    state = value.get("state")
    if state is not None:
        if not isinstance(state, dict):
            raise ValueError(f"{label}.state must be an object")
        required = {"subject_key", "predicate_key", "object_value", "valid_from"}
        if set(state) != required:
            raise ValueError(f"{label}.state must contain exactly {sorted(required)}")
        state = {name: _text(state[name], f"{label}.state.{name}") for name in required}
    authority = value.get("source_authority", 0.5)
    if isinstance(authority, bool) or not isinstance(authority, (int, float)):
        raise ValueError(f"{label}.source_authority must be numeric")
    if not 0.0 <= float(authority) <= 1.0:
        raise ValueError(f"{label}.source_authority must be between 0 and 1")
    return ContextArenaFact(
        _text(value.get("key"), f"{label}.key"),
        _text(value.get("content"), f"{label}.content"),
        _text(value.get("scope", "private"), f"{label}.scope"),
        float(authority),
        state,
    )


def _case(value: Any, index: int) -> ContextArenaCase:
    label = f"cases[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    budget = value.get("token_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError(f"{label}.token_budget must be a positive integer")
    for field in ("abstained", "require_attribution"):
        if not isinstance(value.get(field), bool):
            raise ValueError(f"{label}.{field} must be a boolean")
    truncated = value.get("require_truncation", False)
    if not isinstance(truncated, bool):
        raise ValueError(f"{label}.require_truncation must be a boolean")
    return ContextArenaCase(
        _text(value.get("id"), f"{label}.id"),
        _text(value.get("query"), f"{label}.query"),
        _text(value.get("scope"), f"{label}.scope"),
        budget,
        _keys(value.get("expected_fact_keys"), f"{label}.expected_fact_keys"),
        _keys(value.get("forbidden_fact_keys"), f"{label}.forbidden_fact_keys"),
        value["abstained"],
        value["require_attribution"],
        truncated,
    )


def load_context_arena(path: str | Path = CONTEXT_ARENA_PATH) -> ContextArena:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load context Arena fixture {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("context Arena fixture must be an object")
    version = _text(raw.get("arena_version"), "arena_version")
    facts_raw = raw.get("facts")
    cases_raw = raw.get("cases")
    if not isinstance(facts_raw, list) or not isinstance(cases_raw, list):
        raise ValueError("context Arena facts and cases must be lists")
    facts = tuple(_fact(item, index) for index, item in enumerate(facts_raw))
    cases = tuple(_case(item, index) for index, item in enumerate(cases_raw))
    if not facts or not cases:
        raise ValueError("context Arena must contain facts and cases")
    fact_keys = [fact.key for fact in facts]
    if len(fact_keys) != len(set(fact_keys)):
        raise ValueError("context Arena fact keys must be unique")
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("context Arena case ids must be unique")
    known = set(fact_keys)
    for case in cases:
        referenced = set(case.expected_fact_keys) | set(case.forbidden_fact_keys)
        if not referenced <= known:
            raise ValueError(f"context Arena case {case.id} references unknown facts")
        if set(case.expected_fact_keys) & set(case.forbidden_fact_keys):
            raise ValueError(f"context Arena case {case.id} has overlapping expectations")
    return ContextArena(version, facts, cases, source.resolve())


def _write_fact(
    service: EnfoldService,
    context: ClientContext,
    fact: ContextArenaFact,
) -> int:
    params: dict[str, Any] = {
        "idempotency_key": f"context-arena:{fact.key}",
        "content": fact.content,
        "source_type": "synthetic_fixture",
        "scope": fact.scope,
        "source_authority": fact.source_authority,
    }
    if fact.state is not None:
        params["state"] = dict(fact.state)
    response = service.handle(
        context,
        Request(f"context-arena-write:{fact.key}", "memory.write", params),
    )
    if response["fact_id"] is None:
        raise RuntimeError(f"context Arena write was not durable: {fact.key}")
    return int(response["fact_id"])


def run_context_arena(
    arena: ContextArena | None = None,
    *,
    path: str | Path = CONTEXT_ARENA_PATH,
) -> ContextArenaRun:
    """Run all synthetic fast-lane checks on an in-memory v1 service."""

    loaded = arena if arena is not None else load_context_arena(path)
    scopes = tuple(dict.fromkeys(fact.scope for fact in loaded.facts))
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    context = ClientContext(
        client_id="context-arena-client",
        surface="arena",
        agent_id="arena-agent",
        session_id="context-arena",
        access_scopes=scopes,
    )
    service = EnfoldService(
        conn,
        MemoryPolicy({"context-arena-client": scopes}),
    )
    try:
        fact_ids = {
            fact.key: _write_fact(service, context, fact)
            for fact in loaded.facts
        }
        results: list[ContextArenaResult] = []
        for case in loaded.cases:
            response = service.handle(
                context,
                Request(
                    f"context-arena-read:{case.id}",
                    "memory.context",
                    {
                        "query": case.query,
                        "scope": case.scope,
                        "token_budget": case.token_budget,
                    },
                ),
            )
            actual = tuple(int(fact["fact_id"]) for fact in response["facts"])
            expected = {fact_ids[key] for key in case.expected_fact_keys}
            forbidden = {fact_ids[key] for key in case.forbidden_fact_keys}
            failures: list[str] = []
            if bool(response["abstained"]) != case.abstained:
                failures.append("unexpected abstention")
            if not expected <= set(actual):
                failures.append("expected facts missing")
            if forbidden & set(actual):
                failures.append("forbidden fact returned")
            if int(response["token_estimate"]["used"]) > case.token_budget:
                failures.append("token budget exceeded")
            if case.require_attribution and any(
                fact.get("attribution", {}).get("agent_id") != "arena-agent"
                for fact in response["facts"]
            ):
                failures.append("authorized attribution missing")
            if case.require_truncation and not any(
                bool(fact.get("context_truncated")) for fact in response["facts"]
            ):
                failures.append("expected deterministic truncation")
            results.append(
                ContextArenaResult(
                    case.id,
                    actual,
                    bool(response["abstained"]),
                    not failures,
                    tuple(failures),
                    response,
                )
            )
        return ContextArenaRun(loaded, tuple(results), fact_ids)
    finally:
        conn.close()
