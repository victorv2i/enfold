"""Offline, privacy-safe foundation for Enfold's public Memory Arena.

The bundled fixture is synthetic and intentionally independent of Ollama and a
live Enfold database.  Any provider implementing ``SearchProvider`` can be
evaluated against it; the small lexical provider exists only to make fixture and
report plumbing reproducible in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cases import load_cases, write_json_report
from .runner import EvalCase, EvalResult, SearchProvider, run_retrieval_cases, summarize_results

PUBLIC_ARENA_PATH = Path(__file__).with_name("fixtures") / "public_arena.json"
REQUIRED_CASE_TYPES = frozenset({
    "paraphrase",
    "current_state_update",
    "stale_fact_exclusion",
    "contradiction",
    "changed_preference",
    "abstention",
    "multi_fact_retrieval",
})

_TOKEN_RE = re.compile(r"[#a-z0-9]+")
_STOP_WORDS = frozenset({
    "a", "an", "and", "at", "be", "before", "does", "for", "from", "is",
    "its", "of", "on", "should", "the", "to", "was", "what", "when",
    "where", "which", "who", "will",
})


@dataclass(frozen=True)
class ArenaFact:
    fact_id: int
    content: str
    category: str
    trust_score: float
    tags: tuple[str, ...] = ()

    def as_search_row(self, *, score: float) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "content": self.content,
            "category": self.category,
            "trust_score": self.trust_score,
            "tags": list(self.tags),
            "score": score,
        }


@dataclass(frozen=True)
class ArenaSupersession:
    stale_fact_id: int
    current_fact_id: int
    subject_key: str
    predicate_key: str


@dataclass(frozen=True)
class ArenaConflict:
    conflict_id: str
    subject_key: str
    predicate_key: str
    fact_ids: tuple[int, ...]


@dataclass(frozen=True)
class ArenaMemoryState:
    current_fact_ids: tuple[int, ...]
    supersessions: tuple[ArenaSupersession, ...]
    conflicts: tuple[ArenaConflict, ...]


@dataclass(frozen=True)
class PublicArena:
    version: str
    description: str
    facts: tuple[ArenaFact, ...]
    cases: tuple[EvalCase, ...]
    memory_state: ArenaMemoryState
    source_path: Path


@dataclass(frozen=True)
class PublicArenaRun:
    arena: PublicArena
    results: tuple[EvalResult, ...]
    summary: dict[str, Any]
    provider_name: str
    provider_metadata: dict[str, Any]


def _require_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _fact_from_mapping(data: Any, *, index: int) -> ArenaFact:
    label = f"facts[{index}]"
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be an object")
    fact_id = _require_int(data.get("fact_id"), label=f"{label}.fact_id")
    if fact_id <= 0:
        raise ValueError(f"{label}.fact_id must be positive")
    content = data.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{label}.content must be a non-empty string")
    category = data.get("category")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"{label}.category must be a non-empty string")
    trust = data.get("trust_score")
    if isinstance(trust, bool) or not isinstance(trust, (int, float)):
        raise ValueError(f"{label}.trust_score must be numeric")
    if not 0.0 <= float(trust) <= 1.0:
        raise ValueError(f"{label}.trust_score must be between 0 and 1")
    tags = data.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag for tag in tags):
        raise ValueError(f"{label}.tags must be a list of non-empty strings")
    return ArenaFact(fact_id, content.strip(), category.strip(), float(trust), tuple(tags))


def _memory_state_from_mapping(data: Any) -> ArenaMemoryState:
    if not isinstance(data, dict):
        raise ValueError("memory_state must be an object")
    current = data.get("current_fact_ids")
    supersessions = data.get("supersessions")
    conflicts = data.get("conflict_groups")
    if not isinstance(current, list):
        raise ValueError("memory_state.current_fact_ids must be a list")
    if not isinstance(supersessions, list):
        raise ValueError("memory_state.supersessions must be a list")
    if not isinstance(conflicts, list):
        raise ValueError("memory_state.conflict_groups must be a list")

    parsed_supersessions: list[ArenaSupersession] = []
    for index, row in enumerate(supersessions):
        if not isinstance(row, dict):
            raise ValueError(f"memory_state.supersessions[{index}] must be an object")
        parsed_supersessions.append(ArenaSupersession(
            _require_int(row.get("stale_fact_id"), label=f"supersessions[{index}].stale_fact_id"),
            _require_int(row.get("current_fact_id"), label=f"supersessions[{index}].current_fact_id"),
            str(row.get("subject_key", "")).strip(),
            str(row.get("predicate_key", "")).strip(),
        ))
    parsed_conflicts: list[ArenaConflict] = []
    for index, row in enumerate(conflicts):
        if not isinstance(row, dict) or not isinstance(row.get("fact_ids"), list):
            raise ValueError(f"memory_state.conflict_groups[{index}] must contain fact_ids")
        parsed_conflicts.append(ArenaConflict(
            str(row.get("conflict_id", "")).strip(),
            str(row.get("subject_key", "")).strip(),
            str(row.get("predicate_key", "")).strip(),
            tuple(_require_int(value, label=f"conflict_groups[{index}].fact_ids") for value in row["fact_ids"]),
        ))
    return ArenaMemoryState(
        tuple(_require_int(value, label="memory_state.current_fact_ids") for value in current),
        tuple(parsed_supersessions),
        tuple(parsed_conflicts),
    )


def validate_public_arena(arena: PublicArena) -> None:
    """Validate referential integrity and public-fixture safety declarations."""
    if not arena.version.strip():
        raise ValueError("arena_version must be a non-empty string")
    if not arena.facts:
        raise ValueError("public arena must contain facts")
    if not arena.cases:
        raise ValueError("public arena must contain cases")

    fact_ids = [fact.fact_id for fact in arena.facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("public arena fact_id values must be unique")
    known_ids = set(fact_ids)

    state = arena.memory_state
    current_ids = set(state.current_fact_ids)
    stale_ids = {item.stale_fact_id for item in state.supersessions}
    if current_ids & stale_ids:
        raise ValueError("memory_state cannot mark a fact as both current and stale")
    if current_ids | stale_ids != known_ids:
        raise ValueError("memory_state must classify every fact as current or stale")
    for item in state.supersessions:
        if not item.subject_key or not item.predicate_key:
            raise ValueError("supersessions require subject_key and predicate_key")
        if item.current_fact_id not in current_ids or item.stale_fact_id not in stale_ids:
            raise ValueError("supersession endpoints do not match declared memory state")
    conflict_members: set[int] = set()
    for conflict in state.conflicts:
        if not conflict.conflict_id or not conflict.subject_key or not conflict.predicate_key:
            raise ValueError("conflict groups require identifiers and an exact state slot")
        if len(conflict.fact_ids) < 2 or not set(conflict.fact_ids) <= current_ids:
            raise ValueError("conflict groups require at least two current facts")
        conflict_members.update(conflict.fact_ids)

    case_ids = [case.id for case in arena.cases]
    if any(not case_id.strip() for case_id in case_ids):
        raise ValueError("public arena case ids must be non-empty")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("public arena case ids must be unique")

    for case in arena.cases:
        label = f"case {case.id!r}"
        if case.privacy_tier != "public":
            raise ValueError(f"{label} must declare privacy_tier='public'")
        if case.generation != "hand":
            raise ValueError(f"{label} must declare generation='hand'")
        if not isinstance(case.provenance, dict) or case.provenance.get("source_type") != "synthetic_fixture":
            raise ValueError(f"{label} must declare synthetic_fixture provenance")
        if case.should_abstain:
            if case.gold_fact_id >= 0:
                raise ValueError(f"{label} must use a negative sentinel gold_fact_id")
            if case.expected_current_fact_ids:
                raise ValueError(f"{label} cannot declare expected current facts")
        elif case.gold_fact_id not in known_ids:
            raise ValueError(f"{label} references unknown gold_fact_id {case.gold_fact_id}")

        referenced = set(case.stale_fact_ids) | set(case.expected_current_fact_ids)
        unknown = sorted(referenced - known_ids)
        if unknown:
            raise ValueError(f"{label} references unknown fact ids {unknown}")
        overlap = set(case.stale_fact_ids) & set(case.expected_current_fact_ids)
        if overlap:
            raise ValueError(f"{label} marks fact ids as both current and stale: {sorted(overlap)}")
        if not case.should_abstain and case.expected_current_fact_ids:
            if case.gold_fact_id not in case.expected_current_fact_ids:
                raise ValueError(f"{label} gold fact must be included in expected_current_fact_ids")
        if not set(case.stale_fact_ids) <= stale_ids:
            raise ValueError(f"{label} stale ids must be structurally superseded")
        if case.case_type == "contradiction" and not set(case.expected_current_fact_ids) <= conflict_members:
            raise ValueError(f"{label} contradiction facts must belong to a durable conflict group")

    present_types = {case.case_type for case in arena.cases}
    missing_types = sorted(REQUIRED_CASE_TYPES - present_types)
    if missing_types:
        raise ValueError(f"public arena is missing required case types: {missing_types}")


def load_public_arena(path: str | Path = PUBLIC_ARENA_PATH) -> PublicArena:
    """Load and validate a synthetic public Arena bundle."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load public arena fixture {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("public arena fixture must be a JSON object")
    version = payload.get("arena_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("arena_version must be a non-empty string")
    description = payload.get("description", "")
    if not isinstance(description, str):
        raise ValueError("description must be a string")
    fact_rows = payload.get("facts")
    if not isinstance(fact_rows, list):
        raise ValueError("facts must be a list")
    case_rows = payload.get("cases")
    if not isinstance(case_rows, list):
        raise ValueError("cases must be a list")

    facts = tuple(_fact_from_mapping(row, index=index) for index, row in enumerate(fact_rows))
    memory_state = _memory_state_from_mapping(payload.get("memory_state"))
    try:
        cases = tuple(load_cases(source))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid public arena case: {exc}") from exc
    arena = PublicArena(version.strip(), description.strip(), facts, cases, memory_state, source.resolve())
    validate_public_arena(arena)
    return arena


def _tokens(text: str) -> set[str]:
    normalized = text.lower().replace("'s", "")
    return {token for token in _TOKEN_RE.findall(normalized) if token not in _STOP_WORDS}


class LexicalFixtureProvider:
    """Tiny deterministic baseline for CI plumbing, not Enfold's production search."""

    def __init__(self, facts: tuple[ArenaFact, ...], *, min_score: float = 0.01):
        if min_score < 0:
            raise ValueError("min_score cannot be negative")
        self._facts = facts
        self._min_score = min_score

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        bump: bool = False,
    ) -> list[dict[str, Any]]:
        del bump
        if limit <= 0:
            return []
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        ranked: list[tuple[float, ArenaFact]] = []
        for fact in self._facts:
            if category is not None and fact.category != category:
                continue
            if fact.trust_score < min_trust:
                continue
            overlap = len(query_tokens & _tokens(fact.content))
            score = overlap / len(query_tokens)
            if score >= self._min_score:
                ranked.append((score, fact))
        ranked.sort(key=lambda item: (-item[0], -item[1].trust_score, item[1].fact_id))
        return [fact.as_search_row(score=score) for score, fact in ranked[:limit]]


def _fts5_query(text: str) -> str:
    """Build a conservative OR query for Enfold's FTS5 core retriever."""
    tokens = sorted(token.lstrip("#") for token in _tokens(text) if token.lstrip("#"))
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _proper_anchor_tokens(text: str) -> set[str]:
    """Extract explicit named anchors used for conservative abstention."""

    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
    if not words:
        return set()
    sentence_openers = {
        "A", "An", "Does", "How", "Is", "The", "What", "When", "Where",
        "Which", "Who", "Why",
    }
    return {
        word.lower()
        for index, word in enumerate(words)
        if word[0].isupper()
        and not (index == 0 and word in sentence_openers)
    }


class EnfoldCoreFtsCurrentProvider:
    """Temporary v1-store provider using core FTS and current-state predicates.

    This deliberately exercises :func:`enfold.core_store.search_fts`; it is not
    Enfold's full hybrid production retriever and does not call an embedding
    service.  The database is synthetic, temporary, and created by ``migrate``.
    """

    provider_id = "enfold-core-fts-current-v1"

    def __init__(self, arena: PublicArena):
        from enfold.core_store import connect_database
        from enfold.schema import migrate, schema_version

        self._tempdir = tempfile.TemporaryDirectory(prefix="enfold-public-arena-")
        self.db_path = Path(self._tempdir.name) / "arena.db"
        self._conn = connect_database(self.db_path)
        migrate(self._conn)
        self._load_arena(arena)
        self.schema_version = schema_version(self._conn)
        self.metadata = {
            "provider_id": self.provider_id,
            "retrieval_stack": "standalone_core_store.search_fts+current_state_predicates",
            "schema_version": self.schema_version,
            "store": "temporary_synthetic_migrated_sqlite",
            "full_hybrid_production_retriever": False,
            "uses_live_data": False,
            "uses_embedding_service": False,
            "explicit_named_anchor_abstention": True,
        }

    def _load_arena(self, arena: PublicArena) -> None:
        facts = {fact.fact_id: fact for fact in arena.facts}
        slots: dict[int, ArenaSupersession] = {}
        for item in arena.memory_state.supersessions:
            slots[item.stale_fact_id] = item
            slots[item.current_fact_id] = item
        stale = {item.stale_fact_id: item for item in arena.memory_state.supersessions}
        conflict_slots = {
            fact_id: conflict
            for conflict in arena.memory_state.conflicts
            for fact_id in conflict.fact_ids
        }

        try:
            self._conn.execute("BEGIN IMMEDIATE")
            # Current rows are inserted first so superseded_by always points at
            # an existing row with foreign-key enforcement enabled.
            ordered_ids = [
                *arena.memory_state.current_fact_ids,
                *(item.stale_fact_id for item in arena.memory_state.supersessions),
            ]
            for fact_id in ordered_ids:
                fact = facts[fact_id]
                slot = slots.get(fact_id)
                stale_link = stale.get(fact_id)
                self._conn.execute(
                    """
                    INSERT INTO facts(
                        fact_id, content, category, tags, trust_score,
                        valid_from, invalid_at, superseded_by, memory_kind,
                        subject_key, predicate_key, object_value, scope,
                        sensitivity, schema_version, conflict_group
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'private', 'normal', 1, ?)
                    """,
                    (
                        fact.fact_id,
                        fact.content,
                        fact.category,
                        ",".join(fact.tags),
                        fact.trust_score,
                        "2026-01-02T00:00:00+00:00" if stale_link is None else "2026-01-01T00:00:00+00:00",
                        "2026-01-02T00:00:00+00:00" if stale_link else None,
                        stale_link.current_fact_id if stale_link else None,
                        "state" if slot or fact_id in conflict_slots else "fact",
                        (slot.subject_key if slot else conflict_slots[fact_id].subject_key)
                        if slot or fact_id in conflict_slots else None,
                        (slot.predicate_key if slot else conflict_slots[fact_id].predicate_key)
                        if slot or fact_id in conflict_slots else None,
                        fact.content if slot or fact_id in conflict_slots else None,
                        conflict_slots[fact_id].conflict_id
                        if fact_id in conflict_slots else None,
                    ),
                )
            for conflict in arena.memory_state.conflicts:
                self._conn.execute(
                    """
                    INSERT INTO fact_conflicts(
                        conflict_id, scope, subject_key, predicate_key,
                        detected_at, detail_json
                    ) VALUES (?, 'private', ?, ?, ?, ?)
                    """,
                    (
                        conflict.conflict_id,
                        conflict.subject_key,
                        conflict.predicate_key,
                        "2026-01-02T00:00:00+00:00",
                        json.dumps({"source": "synthetic_public_arena"}, sort_keys=True),
                    ),
                )
                self._conn.executemany(
                    "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES (?, ?)",
                    ((conflict.conflict_id, fact_id) for fact_id in conflict.fact_ids),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            self.close()
            raise

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        bump: bool = False,
    ) -> list[dict[str, Any]]:
        from enfold.core_store import search_fts

        del bump
        fts_query = _fts5_query(query)
        if not fts_query or limit <= 0:
            return []
        rows = search_fts(
            self._conn,
            fts_query,
            allowed_scopes=("private",),
            category=category,
            min_trust=min_trust,
            limit=limit,
        )
        anchors = _proper_anchor_tokens(query)
        if anchors:
            rows = [
                row
                for row in rows
                if anchors <= _tokens(f"{row['content']} {row.get('tags', '')}")
            ]
        for row in rows:
            rank = float(row["fts_rank"])
            row["score"] = max(0.0, -rank)
        return rows

    def search_conflicts(
        self,
        query: str,
        *,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        bump: bool = False,
    ) -> list[dict[str, Any]]:
        """Search the explicit conflict view, never the settled-truth index."""

        from enfold.state_slots import list_state_conflicts

        del bump
        query_tokens = _tokens(query)
        if not query_tokens or limit <= 0:
            return []
        ranked: list[tuple[float, dict[str, Any]]] = []
        for conflict in list_state_conflicts(self._conn, "private"):
            placeholders = ",".join("?" for _ in conflict.member_fact_ids)
            rows = self._conn.execute(
                f"""
                SELECT fact_id, content, category, trust_score, tags
                FROM facts WHERE fact_id IN ({placeholders})
                """,
                conflict.member_fact_ids,
            ).fetchall()
            for row in rows:
                if category is not None and row["category"] != category:
                    continue
                if float(row["trust_score"]) < min_trust:
                    continue
                overlap = len(query_tokens & _tokens(str(row["content"])))
                if overlap == 0:
                    continue
                score = overlap / len(query_tokens)
                ranked.append((score, {
                    "fact_id": int(row["fact_id"]),
                    "content": str(row["content"]),
                    "category": str(row["category"]),
                    "trust_score": float(row["trust_score"]),
                    "tags": str(row["tags"] or "").split(","),
                    "score": score,
                    "conflict_id": conflict.conflict_id,
                    "is_conflict": True,
                }))
        ranked.sort(key=lambda item: (-item[0], item[1]["fact_id"]))
        return [row for _, row in ranked[:limit]]

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the temporary connection for deterministic structural tests."""
        return self._conn

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()
            self._conn = None  # type: ignore[assignment]
        tempdir = getattr(self, "_tempdir", None)
        if tempdir is not None:
            tempdir.cleanup()
            self._tempdir = None  # type: ignore[assignment]

    def __enter__(self) -> EnfoldCoreFtsCurrentProvider:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class EnfoldOfflineHybridProvider(EnfoldCoreFtsCurrentProvider):
    """Hybrid v1 provider with the deterministic, non-production CI embedder."""

    provider_id = "enfold-offline-hybrid-ci-v1"

    def __init__(self, arena: PublicArena):
        from enfold.hybrid_retrieval import (
            DeterministicFeatureHashEmbedder,
            HybridRetriever,
        )

        super().__init__(arena)
        embedder = DeterministicFeatureHashEmbedder()
        self._retriever = HybridRetriever(
            self.connection,
            embedder,
            allowed_scopes=("private",),
        )
        self.metadata = {
            "provider_id": self.provider_id,
            **self._retriever.metadata,
            "schema_version": self.schema_version,
            "store": "temporary_synthetic_migrated_sqlite",
            "full_hybrid_production_retriever": False,
            "uses_live_data": False,
            "uses_embedding_service": False,
            "ci_embedder_is_semantic_model": False,
        }

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        bump: bool = False,
    ) -> list[dict[str, Any]]:
        return self._retriever.search(
            query,
            category=category,
            min_trust=min_trust,
            limit=limit,
            bump=bump,
        )


def score_public_arena(
    results: list[EvalResult] | tuple[EvalResult, ...],
    *,
    stale_k: int = 3,
    score_thresholds: list[float] | tuple[float, ...] | None = None,
    decision_rules: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Score Arena results through the shared evaluation metrics pipeline."""
    return summarize_results(
        list(results),
        stale_k=stale_k,
        score_thresholds=score_thresholds,
        decision_rules=decision_rules,
    )


def run_public_arena(
    provider: SearchProvider | None = None,
    *,
    arena: PublicArena | None = None,
    path: str | Path = PUBLIC_ARENA_PATH,
    limit: int = 10,
    provider_kind: str = "lexical",
) -> PublicArenaRun:
    """Run the public case bank with a caller's provider or the offline baseline."""
    loaded = arena if arena is not None else load_public_arena(path)
    validate_public_arena(loaded)
    owned_provider: EnfoldCoreFtsCurrentProvider | None = None
    if provider is not None:
        selected_provider = provider
    elif provider_kind == "lexical":
        selected_provider = LexicalFixtureProvider(loaded.facts)
    elif provider_kind == "core-fts-current":
        owned_provider = EnfoldCoreFtsCurrentProvider(loaded)
        selected_provider = owned_provider
    elif provider_kind == "offline-hybrid-ci":
        owned_provider = EnfoldOfflineHybridProvider(loaded)
        selected_provider = owned_provider
    else:
        raise ValueError(f"unsupported public Arena provider: {provider_kind}")
    try:
        results = tuple(run_retrieval_cases(selected_provider, list(loaded.cases), limit=limit))
        provider_metadata = dict(getattr(selected_provider, "metadata", {}))
    finally:
        if owned_provider is not None:
            owned_provider.close()
    summary = score_public_arena(results)
    return PublicArenaRun(
        arena=loaded,
        results=results,
        summary=summary,
        provider_name=type(selected_provider).__name__,
        provider_metadata=provider_metadata,
    )


def write_public_arena_report(path: str | Path, run: PublicArenaRun) -> None:
    """Write a publishable report containing synthetic queries and result text."""
    write_json_report(
        path,
        summary=run.summary,
        results=list(run.results),
        metadata={
            "arena": "enfold-public-arena",
            "arena_version": run.arena.version,
            "fixture": run.arena.source_path.name,
            "provider": run.provider_name,
            "provider_metadata": run.provider_metadata,
            "fixture_sha256": hashlib.sha256(run.arena.source_path.read_bytes()).hexdigest(),
            "synthetic": True,
        },
        include_text=True,
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _concise_summary(run: PublicArenaRun) -> dict[str, Any]:
    stale = run.summary.get("stale_leak@3", run.summary["stale_leak@1"])
    return {
        "arena": "enfold-public-arena",
        "arena_version": run.arena.version,
        "provider": run.provider_name,
        "cases": run.summary["cases"],
        "answerable_recall@1": run.summary["answerable"]["recall@1"],
        "set_recall": run.summary["set_recall"],
        "set_f1": run.summary["set_f1"],
        "stale_leaks@3": stale["leaks"],
        "true_abstain": run.summary["abstention"]["true_abstain"],
        "false_confident": run.summary["abstention"]["false_confident"],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the bundled offline baseline and optionally write a public report."""
    parser = argparse.ArgumentParser(
        description="Run Enfold's synthetic public Arena without live data or Ollama.",
    )
    parser.add_argument(
        "--provider",
        choices=("core-fts-current", "offline-hybrid-ci", "lexical"),
        default="core-fts-current",
        help="retrieval provider (default: core-fts-current)",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=10,
        help="maximum retrieved facts per case (default: 10)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="optional path for the full publishable JSON report",
    )
    args = parser.parse_args(argv)

    run = run_public_arena(limit=args.limit, provider_kind=args.provider)
    if args.out is not None:
        write_public_arena_report(args.out, run)
    print(json.dumps(_concise_summary(run), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
