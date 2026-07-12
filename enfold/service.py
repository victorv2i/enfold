"""Scoped, provenance-aware Enfold request service.

This module is the storage router behind a transport adapter.  It owns no
socket and opens no database: callers pass a connection that has already been
explicitly migrated to Enfold schema v1.  The protocol context is trusted only
as an identity assertion from the daemon; server-side :class:`MemoryPolicy`
grants still narrow every read and write.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import sqlite3
from typing import Any, Callable, Mapping, Sequence

from .context import TRUNCATION_MARKER, pack_context
from .core_store import insert_fact
from .extraction_enqueue import ExtractionEnqueuer
from .embedding_jobs import EmbeddingOutbox
from .hybrid_retrieval import RetrieverFactory, deterministic_retriever_factory
from .policy import (
    MemoryPolicy,
    UnknownMemoryClient,
    default_credential_screen,
    validate_scope,
)
from .protocol import (
    IMMUTABLE_CONTEXT_FIELDS,
    ClientContext,
    Request,
    RequestHandlingError,
    SUPPORTED_SCHEMA_VERSION,
)
from .provenance import ConnectionContext, WriteRequest
from .projections import changes, entities, entity_dossier, timeline
from .schema import require_compatible_schema
from .state_slots import StateCandidate, list_state_conflicts, resolve_state_conflict
from .temporal import fact_history
from .write_service import (
    FactWriteResult,
    IdempotencyConflict,
    MemoryWriteService,
    NearDedupConfig,
)


class ServiceRequestError(RequestHandlingError):
    """Safe request failure for a transport adapter to serialize."""

    def __init__(self, code: str, message: str):
        super().__init__(code, message)


_FACT_FIELDS = (
    "fact_id", "content", "category", "tags", "trust_score",
    "retrieval_count", "helpful_count", "created_at", "updated_at",
    "valid_from", "invalid_at", "superseded_by", "memory_kind",
    "subject_key", "predicate_key", "object_value", "object_entity_id",
    "confidence", "source_authority", "scope", "sensitivity",
    "correction_status", "schema_version", "conflict_group",
)
_MIN_CONTEXT_TOKEN_BUDGET = 16
_MAX_CONTEXT_TOKEN_BUDGET = 4096


@dataclass(frozen=True, slots=True)
class OutputBounds:
    """Service-layer trust defaults and serialized response limits."""

    default_min_trust: float = 0.3
    search_max_results: int = 20
    context_max_results: int = 12
    max_fact_chars: int = 2_000
    search_max_total_chars: int = 12_000
    context_max_total_chars: int = 16_000
    context_mmr_lambda: float = 0.7

    def __post_init__(self) -> None:
        integer_bounds = (
            self.search_max_results,
            self.context_max_results,
            self.max_fact_chars,
            self.search_max_total_chars,
            self.context_max_total_chars,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_bounds):
            raise ValueError("output bounds must be integers")
        if (
            isinstance(self.default_min_trust, bool)
            or not isinstance(self.default_min_trust, (int, float))
            or not math.isfinite(self.default_min_trust)
        ):
            raise ValueError("default_min_trust must be a finite number")
        if not 0.0 <= self.default_min_trust <= 1.0:
            raise ValueError("default_min_trust must be between 0 and 1")
        if self.search_max_results < 1 or self.context_max_results < 1:
            raise ValueError("result caps must be positive")
        if self.max_fact_chars < len(TRUNCATION_MARKER):
            raise ValueError("max_fact_chars is too small for the truncation marker")
        if self.search_max_total_chars < 512 or self.context_max_total_chars < 512:
            raise ValueError("total character caps must be at least 512")
        if (
            isinstance(self.context_mmr_lambda, bool)
            or not isinstance(self.context_mmr_lambda, (int, float))
            or not math.isfinite(self.context_mmr_lambda)
            or not 0.0 <= self.context_mmr_lambda <= 1.0
        ):
            raise ValueError("context_mmr_lambda must be between 0 and 1")


DEFAULT_OUTPUT_BOUNDS = OutputBounds()


def _check_keys(
    params: Mapping[str, Any],
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = sorted(required - params.keys())
    unknown = sorted(params.keys() - required - optional)
    if missing:
        raise ServiceRequestError("invalid_params", f"missing parameters: {missing}")
    if unknown:
        raise ServiceRequestError("invalid_params", f"unknown parameters: {unknown}")


def _reject_nested_identity(value: Any, *, path: str = "params") -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(IMMUTABLE_CONTEXT_FIELDS & value.keys())
        if forbidden:
            raise ServiceRequestError(
                "invalid_params",
                f"{path} cannot contain connection identity fields: {forbidden}",
            )
        for key, item in value.items():
            _reject_nested_identity(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nested_identity(item, path=f"{path}[{index}]")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceRequestError("invalid_params", f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _number(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServiceRequestError("invalid_params", f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ServiceRequestError("invalid_params", f"{name} must be between 0 and 1")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ServiceRequestError("invalid_params", f"{name} must be a positive integer")
    return value


def _limit(value: Any, *, default: int, maximum: int = 200) -> int:
    if value is None:
        return default
    result = _positive_int(value, "limit")
    if result > maximum:
        raise ServiceRequestError("invalid_params", f"limit must not exceed {maximum}")
    return result


def _token_budget(value: Any) -> int:
    if value is None:
        return 256
    if isinstance(value, bool) or not isinstance(value, int):
        raise ServiceRequestError("invalid_params", "token_budget must be an integer")
    if not _MIN_CONTEXT_TOKEN_BUDGET <= value <= _MAX_CONTEXT_TOKEN_BUDGET:
        raise ServiceRequestError(
            "invalid_params",
            "token_budget must be between "
            f"{_MIN_CONTEXT_TOKEN_BUDGET} and {_MAX_CONTEXT_TOKEN_BUDGET}",
        )
    return value


def _serialized_chars(value: Mapping[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _truncate_fact_content(fact: dict[str, Any], maximum: int) -> bool:
    content = fact.get("content")
    if not isinstance(content, str) or len(content) <= maximum:
        return False
    fact["content"] = content[:maximum - len(TRUNCATION_MARKER)].rstrip() + TRUNCATION_MARKER
    fact["content_truncated"] = True
    return True


def _boolean(value: Any, name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ServiceRequestError("invalid_params", f"{name} must be a boolean")
    return value


def _json_object(value: Any, name: str) -> str:
    if value is None:
        return "{}"
    if not isinstance(value, dict):
        raise ServiceRequestError("invalid_params", f"{name} must be an object")
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ServiceRequestError("invalid_params", f"{name} must contain JSON values") from exc


def _protocol_context(context: ClientContext) -> ConnectionContext:
    """Copy only immutable, handshake-established context into provenance."""

    return ConnectionContext(
        client_id=context.client_id,
        surface=context.surface,
        agent_id=context.agent_id,
        session_id=context.session_id,
        parent_agent_id=context.parent_agent_id,
        project_root=context.project_root,
        repository=context.repository,
        branch=context.branch,
        commit_sha=context.commit_sha,
        access_scopes=context.access_scopes,
    )


def _fact_writer(
    conn: sqlite3.Connection, request: WriteRequest, observation_id: int
) -> FactWriteResult:
    del observation_id
    fact_id = insert_fact(
        conn,
        request.content,
        category=request.category,
        tags=request.tags,
        trust_score=request.trust_score,
        source_authority=request.source_authority,
        scope=request.scope,
        sensitivity=request.sensitivity,
    )
    if request.correction_status is not None:
        conn.execute(
            "UPDATE facts SET correction_status = ? WHERE fact_id = ?",
            (request.correction_status, fact_id),
        )
    return FactWriteResult(fact_id)


class EnfoldService:
    """Route typed protocol requests against one migrated SQLite connection."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        policy: MemoryPolicy,
        *,
        retriever_factory: RetrieverFactory | None = None,
        embedding_outbox: EmbeddingOutbox | None = None,
        extraction_enqueuer: ExtractionEnqueuer | None = None,
        extraction_processing_mode: str = "deferred",
        output_bounds: OutputBounds = DEFAULT_OUTPUT_BOUNDS,
        embedding_identity: str | None = None,
        query_embedder: Callable[[str], object] | None = None,
        near_dedup_enabled: bool = True,
    ):
        if conn.in_transaction:
            raise RuntimeError("EnfoldService requires an idle connection")
        version = require_compatible_schema(conn)
        if version != SUPPORTED_SCHEMA_VERSION:
            raise RuntimeError(
                f"EnfoldService requires schema v{SUPPORTED_SCHEMA_VERSION}; found v{version}"
            )
        # Core retrieval returns mapping-shaped rows.  sqlite3.Row remains
        # index-compatible with the write/schema layers using this connection.
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
            raise RuntimeError("EnfoldService requires foreign key enforcement")
        self._conn = conn
        self._policy = policy
        self._output_bounds = output_bounds
        self._embedding_outbox = embedding_outbox
        self._writes = MemoryWriteService(
            conn,
            _fact_writer,
            policy,
            embedding_enqueue=(
                embedding_outbox.enqueue_in_transaction
                if embedding_outbox is not None else None
            ),
            near_dedup=NearDedupConfig(
                enabled=near_dedup_enabled,
                embedding_identity=embedding_identity,
            ),
            query_embedder=query_embedder,
        )
        self._retriever_factory = (
            retriever_factory or deterministic_retriever_factory()
        )
        self._extraction_enqueuer = extraction_enqueuer
        if extraction_processing_mode not in {
            "deferred", "disabled", "daemon-supervised",
        }:
            raise ValueError("extraction_processing_mode is invalid")
        self._extraction_processing_mode = extraction_processing_mode

    @property
    def retrieval_metadata(self) -> dict[str, Any]:
        """Non-sensitive retrieval capabilities for health/inspection output."""

        retriever = self._retriever_factory(self._conn, ("private",))
        return dict(retriever.metadata)

    def __call__(self, context: ClientContext, request: Request) -> dict[str, Any]:
        return self.handle(context, request)

    def handle(self, context: ClientContext, request: Request) -> dict[str, Any]:
        if request.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ServiceRequestError(
                "incompatible_schema",
                f"request schema {request.schema_version}; service schema {SUPPORTED_SCHEMA_VERSION}",
            )
        _reject_nested_identity(request.params)
        effective = self._authorize(context)
        routes = {
            "memory.write": self._write,
            "memory.search": self._search,
            "memory.context": self._context,
            "memory.evidence": self._evidence,
            "memory.history": self._history,
            "memory.changes": self._changes,
            "memory.timeline": self._timeline,
            "memory.entities": self._entities,
            "memory.entity": self._entity,
            "memory.conflicts": self._conflicts,
            "memory.resolve_conflict": self._resolve_conflict,
            "memory.extraction.enqueue": self._enqueue_extraction,
        }
        route = routes.get(request.method)
        if route is None:
            raise ServiceRequestError("unsupported_method", f"unsupported service method: {request.method}")
        return route(effective, request.params)

    def _authorize(self, context: ClientContext) -> ConnectionContext:
        try:
            return self._policy.authorize_context(_protocol_context(context))
        except UnknownMemoryClient as exc:
            raise ServiceRequestError("access_denied", "memory client is not authorized") from exc
        except PermissionError as exc:
            raise ServiceRequestError("access_denied", "no requested memory scope is authorized") from exc
        except ValueError as exc:
            raise ServiceRequestError("invalid_context", str(exc)) from exc

    def _write(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        required = {"idempotency_key", "content", "source_type"}
        optional = {
            "category", "tags", "trust_score", "source_authority", "source_uri",
            "observation_content", "asserted_by", "observed_at", "scope",
            "sensitivity", "correction_status", "evidence_excerpt", "relation",
            "metadata", "supersede_fact_id", "state",
        }
        _check_keys(params, required, optional)
        metadata = _json_object(params.get("metadata"), "metadata")
        try:
            write = WriteRequest(
                idempotency_key=_text(params["idempotency_key"], "idempotency_key"),
                content=_text(params["content"], "content"),
                source_type=_text(params["source_type"], "source_type"),
                category=_text(params.get("category", "general"), "category"),
                tags=(
                    params.get("tags", "")
                    if isinstance(params.get("tags", ""), str)
                    else self._invalid("tags must be a string")
                ),
                trust_score=_number(params.get("trust_score"), "trust_score", 0.5),
                source_authority=_number(
                    params.get("source_authority"), "source_authority", 0.5
                ),
                source_uri=_optional_text(params.get("source_uri"), "source_uri"),
                observation_content=_optional_text(
                    params.get("observation_content"), "observation_content"
                ),
                asserted_by=_optional_text(params.get("asserted_by"), "asserted_by"),
                # The performing agent is connection provenance, never caller input.
                performed_by=context.agent_id,
                observed_at=_optional_text(params.get("observed_at"), "observed_at"),
                scope=_text(params.get("scope", "private"), "scope"),
                sensitivity=_text(params.get("sensitivity", "normal"), "sensitivity"),
                correction_status=_optional_text(
                    params.get("correction_status"), "correction_status"
                ),
                evidence_excerpt=_optional_text(
                    params.get("evidence_excerpt"), "evidence_excerpt"
                ),
                relation=_text(params.get("relation", "supports"), "relation"),
                metadata_json=metadata,
                supersede_fact_id=(
                    None
                    if params.get("supersede_fact_id") is None
                    else _positive_int(params["supersede_fact_id"], "supersede_fact_id")
                ),
            )
            candidate = self._state_candidate(write, params.get("state"))
            outcome = self._writes.write(context, write, state_candidate=candidate)
        except ServiceRequestError:
            raise
        except IdempotencyConflict as exc:
            raise ServiceRequestError("idempotency_conflict", str(exc)) from exc
        except (PermissionError, ValueError) as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc
        result = asdict(outcome)
        result["detail"] = json.loads(result.pop("detail_json"))
        return result

    @staticmethod
    def _invalid(message: str) -> Any:
        raise ServiceRequestError("invalid_params", message)

    def _state_candidate(
        self, write: WriteRequest, value: Any
    ) -> StateCandidate | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ServiceRequestError("invalid_params", "state must be an object")
        _check_keys(
            value,
            {"subject_key", "predicate_key"},
            {"object_value", "valid_from"},
        )
        return StateCandidate(
            content=write.content,
            subject_key=_text(value["subject_key"], "state.subject_key"),
            predicate_key=_text(value["predicate_key"], "state.predicate_key"),
            object_value=_optional_text(value.get("object_value"), "state.object_value"),
            source_authority=write.source_authority,
            valid_from=_optional_text(value.get("valid_from"), "state.valid_from"),
            scope=write.scope,
        )

    def _search(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"query"}, {"category", "min_trust", "limit"})
        query = _text(params["query"], "query")
        category = _optional_text(params.get("category"), "category")
        bounds = self._output_bounds
        min_trust = _number(
            params.get("min_trust"), "min_trust", bounds.default_min_trust
        )
        requested_limit = _limit(params.get("limit"), default=20)
        limit = min(requested_limit, bounds.search_max_results)
        retriever = self._retriever_factory(self._conn, context.access_scopes)
        rows = retriever.search(
            query,
            category=category,
            min_trust=min_trust,
            limit=limit + 1,
        )
        result_cap_truncated = len(rows) > limit
        rows = rows[:limit]
        facts = []
        content_truncated = False
        for row in rows:
            fact = self._safe_fact(row)
            fact["attribution"] = self._authorized_attribution(
                int(fact["fact_id"]), context.access_scopes
            )
            content_truncated |= _truncate_fact_content(fact, bounds.max_fact_chars)
            facts.append(fact)
        response = {
            "facts": facts,
            "retrieval": dict(retriever.metadata),
            "output_truncated": (
                content_truncated or requested_limit > limit or result_cap_truncated
            ),
        }
        while facts and _serialized_chars(response) > bounds.search_max_total_chars:
            facts.pop()
            response["output_truncated"] = True
        if _serialized_chars(response) > bounds.search_max_total_chars:
            response["retrieval"] = {"output_truncated": True}
        if _serialized_chars(response) > bounds.search_max_total_chars:
            response["retrieval"] = {}
        return response

    def _context(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return a bounded, cited context pack over authorized current facts.

        This is a read-only projection over the ordinary retriever.  The
        retriever applies scope/current/conflict predicates before ranking; the
        pure packer repeats lifecycle checks defensively before formatting.
        """

        _check_keys(params, {"query", "token_budget"}, {"scope", "min_trust"})
        query = _text(params["query"], "query")
        token_budget = _token_budget(params["token_budget"])
        bounds = self._output_bounds
        min_trust = _number(
            params.get("min_trust"), "min_trust", bounds.default_min_trust
        )
        scopes = self._requested_scopes(context, params.get("scope"))
        retriever = self._retriever_factory(self._conn, scopes)
        rows = retriever.search(
            query,
            min_trust=min_trust,
            limit=bounds.context_max_results * 4 + 1,
        )
        candidate_cap = bounds.context_max_results * 4
        result_cap_truncated = len(rows) > candidate_cap
        rows = rows[:candidate_cap]
        candidates: list[dict[str, Any]] = []
        for row in rows:
            fact = self._safe_fact(row)
            if "_mmr_embedding" in row:
                fact["_mmr_embedding"] = row["_mmr_embedding"]
            fact["attribution"] = self._authorized_attribution(
                int(fact["fact_id"]), scopes
            )
            candidates.append(fact)
        output_truncated = result_cap_truncated
        while True:
            packed = pack_context(
                candidates,
                token_budget=token_budget,
                max_fact_chars=bounds.max_fact_chars,
                max_facts=bounds.context_max_results,
                mmr_lambda=bounds.context_mmr_lambda,
            ).as_dict()
            output_truncated |= any(
                bool(fact.get("context_truncated")) for fact in packed["facts"]
            )
            packed["retrieval"] = dict(retriever.metadata)
            packed["output_truncated"] = output_truncated
            if _serialized_chars(packed) <= bounds.context_max_total_chars:
                return packed
            output_truncated = True
            if candidates:
                candidates.pop()
                continue
            packed["retrieval"] = {"output_truncated": True}
            if _serialized_chars(packed) <= bounds.context_max_total_chars:
                return packed
            packed["retrieval"] = {}
            return packed

    def _enqueue_extraction(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"transcript", "source"}, {"scope", "metadata"})
        if self._extraction_enqueuer is None:
            raise ServiceRequestError(
                "extraction_unavailable",
                "durable extraction enqueue is not configured; automatic LLM extraction remains deferred",
            )
        scope = _text(params.get("scope", "private"), "scope")
        try:
            scope = validate_scope(scope)
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc
        if scope not in context.access_scopes:
            raise ServiceRequestError("access_denied", "extraction scope is not authorized")
        if scope == "secret":
            return {
                "outcome": "rejected",
                "reason": "secret durable extraction payloads are disabled",
                "queue_id": None,
            }
        transcript = _text(params["transcript"], "transcript")
        source = _text(params["source"], "source")
        metadata_json = _json_object(params.get("metadata"), "metadata")
        screen_request = WriteRequest(
            idempotency_key="extraction-screen",
            content=transcript,
            source_type="conversation_transcript",
            scope=scope,
            metadata_json=metadata_json,
        )
        decision = default_credential_screen(screen_request)
        if decision is not None:
            return {"outcome": "rejected", "reason": decision.reason, "queue_id": None}
        result = self._extraction_enqueuer.enqueue_after_commit(
            context,
            transcript,
            source=source,
            scope=scope,
            metadata=json.loads(metadata_json),
        )
        return {
            "outcome": "queued",
            "queue_id": result.queue_id,
            "payload_sha256": result.payload_sha256,
            "replayed": result.replayed,
            "automatic_llm_extraction": self._extraction_processing_mode,
        }

    def _evidence(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"fact_id"}, {"limit"})
        fact_id = _positive_int(params["fact_id"], "fact_id")
        limit = _limit(params.get("limit"), default=100)
        fact = self._historical_fact(fact_id, context.access_scopes)
        if fact is None:
            raise ServiceRequestError("not_found", "fact was not found")
        placeholders = ",".join("?" for _ in context.access_scopes)
        rows = self._conn.execute(
            f"""
            SELECT o.observation_id, o.client_id, o.session_id, o.source_type,
                   o.source_uri, o.project_root, o.repository, o.branch,
                   o.commit_sha, o.content, o.asserted_by, o.performed_by,
                   o.observed_at, o.recorded_at, o.scope, o.sensitivity,
                   o.redacted_at, o.metadata_json, p.relation,
                   p.evidence_excerpt, p.created_at
            FROM fact_provenance p
            JOIN observations o ON o.observation_id = p.observation_id
            WHERE p.fact_id = ? AND o.scope IN ({placeholders})
            ORDER BY p.created_at, o.observation_id
            LIMIT ?
            """,
            (fact_id, *context.access_scopes, limit),
        ).fetchall()
        keys = (
            "observation_id", "client_id", "session_id", "source_type",
            "source_uri", "project_root", "repository", "branch", "commit_sha",
            "content", "asserted_by", "performed_by", "observed_at", "recorded_at",
            "scope", "sensitivity", "redacted_at", "metadata", "relation",
            "evidence_excerpt", "provenance_created_at",
        )
        evidence = []
        for row in rows:
            item = dict(zip(keys, row))
            item["metadata"] = json.loads(item["metadata"])
            evidence.append(item)
        return {"fact": fact, "evidence": evidence}

    def _history(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        optional = {"fact_id", "subject_key", "predicate_key", "scope", "limit"}
        _check_keys(params, set(), optional)
        by_id = "fact_id" in params
        by_slot = "subject_key" in params or "predicate_key" in params or "scope" in params
        if by_id == by_slot:
            raise ServiceRequestError(
                "invalid_params",
                "history requires either fact_id or subject_key and predicate_key",
            )
        limit = _limit(params.get("limit"), default=100)
        if by_id:
            fact_id = _positive_int(params["fact_id"], "fact_id")
            anchor = self._historical_fact(fact_id, context.access_scopes)
            if anchor is None:
                raise ServiceRequestError("not_found", "fact was not found")
            if anchor.get("subject_key") and anchor.get("predicate_key"):
                scopes = (str(anchor["scope"]),)
                subject = str(anchor["subject_key"])
                predicate = str(anchor["predicate_key"])
                rows = self._slot_history(scopes, subject, predicate, limit)
            else:
                rows = [
                    row for row in fact_history(self._conn, fact_id)
                    if row.get("scope") in context.access_scopes
                ][:limit]
        else:
            if "subject_key" not in params or "predicate_key" not in params:
                raise ServiceRequestError(
                    "invalid_params", "subject_key and predicate_key are both required"
                )
            scopes = self._requested_scopes(context, params.get("scope"))
            rows = self._slot_history(
                scopes,
                _text(params["subject_key"], "subject_key"),
                _text(params["predicate_key"], "predicate_key"),
                limit,
            )
        return {"facts": rows}

    def _conflicts(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, set(), {"scope", "unresolved_only"})
        scopes = self._requested_scopes(context, params.get("scope"))
        unresolved = _boolean(params.get("unresolved_only"), "unresolved_only", True)
        conflicts: list[dict[str, Any]] = []
        for scope in scopes:
            for record in list_state_conflicts(
                self._conn, scope, unresolved_only=unresolved
            ):
                item = asdict(record)
                item["member_fact_ids"] = list(item["member_fact_ids"])
                item["members"] = [
                    fact
                    for fact_id in item["member_fact_ids"]
                    if (fact := self._historical_fact(fact_id, (scope,))) is not None
                ]
                conflicts.append(item)
        return {"conflicts": conflicts}

    def _changes(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"since", "until"}, {"scope", "limit"})
        scopes = self._requested_scopes(context, params.get("scope"))
        try:
            return changes(
                self._conn,
                _text(params["since"], "since"),
                _text(params["until"], "until"),
                scopes,
                limit=_limit(params.get("limit"), default=100),
            )
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc

    def _timeline(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"subject_or_query"}, {"scope", "limit"})
        scopes = self._requested_scopes(context, params.get("scope"))
        try:
            return timeline(
                self._conn,
                _text(params["subject_or_query"], "subject_or_query"),
                scopes,
                limit=_limit(params.get("limit"), default=100),
            )
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc

    def _entities(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, set(), {"scope", "min_facts", "limit"})
        scopes = self._requested_scopes(context, params.get("scope"))
        min_facts = _positive_int(params.get("min_facts", 1), "min_facts")
        try:
            return entities(
                self._conn,
                scopes,
                min_facts,
                limit=_limit(params.get("limit"), default=100),
            )
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc

    def _entity(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"name"}, {"scope", "limit"})
        scopes = self._requested_scopes(context, params.get("scope"))
        try:
            return entity_dossier(
                self._conn,
                _text(params["name"], "name"),
                scopes,
                limit=_limit(params.get("limit"), default=100),
            )
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc

    def _resolve_conflict(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"conflict_id", "resolution_fact_id", "reason"})
        if not self._policy.can_resolve_conflicts(context.client_id):
            raise ServiceRequestError(
                "access_denied", "memory client is not authorized to resolve conflicts"
            )
        conflict_id = _text(params["conflict_id"], "conflict_id")
        resolution_fact_id = _positive_int(
            params["resolution_fact_id"], "resolution_fact_id"
        )
        reason = _text(params["reason"], "reason")
        placeholders = ",".join("?" for _ in context.access_scopes)
        resolved_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            visible = self._conn.execute(
                f"""
                SELECT scope FROM fact_conflicts
                WHERE conflict_id = ? AND scope IN ({placeholders})
                  AND resolved_at IS NULL
                """,
                (conflict_id, *context.access_scopes),
            ).fetchone()
            if visible is None:
                raise ServiceRequestError("not_found", "conflict was not found")
            self._writes._register_client(context, resolved_at)
            self._writes._register_session(context, resolved_at)
            resolution = resolve_state_conflict(
                self._conn,
                conflict_id,
                resolution_fact_id,
                resolved_by=context.agent_id,
                reason=reason,
                resolved_at=resolved_at,
                resolver_client_id=context.client_id,
                resolver_session_id=context.session_id,
                resolver_agent_id=context.agent_id,
            )
            if self._embedding_outbox is not None:
                self._embedding_outbox.enqueue_in_transaction(resolution_fact_id)
            self._conn.commit()
        except ServiceRequestError:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
        except (ValueError, RuntimeError) as exc:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise ServiceRequestError("invalid_resolution", str(exc)) from exc
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
        result = asdict(resolution)
        result["superseded_fact_ids"] = list(result["superseded_fact_ids"])
        result["scope"] = str(visible[0])
        return {"resolution": result}

    @staticmethod
    def _safe_fact(row: Mapping[str, Any]) -> dict[str, Any]:
        score_fields = (
            "score", "fts_score", "jaccard_score", "dense_score",
            "trust_score_component", "memory_kind_score", "recency_score",
        )
        return {
            key: row[key]
            for key in (*_FACT_FIELDS, *score_fields)
            if key in row
        }

    def _authorized_attribution(
        self, fact_id: int, scopes: Sequence[str]
    ) -> dict[str, Any] | None:
        """Return latest visible provenance plus a visible-only evidence count."""

        placeholders = ",".join("?" for _ in scopes)
        row = self._conn.execute(
            f"""
            SELECT o.performed_by, s.agent_id, o.session_id, o.source_type,
                   o.repository, o.branch, o.commit_sha,
                   COUNT(*) OVER () AS evidence_count
            FROM fact_provenance p
            JOIN observations o ON o.observation_id = p.observation_id
            JOIN memory_sessions s
              ON s.client_id = o.client_id AND s.session_id = o.session_id
            WHERE p.fact_id = ? AND o.scope IN ({placeholders})
            ORDER BY o.recorded_at DESC, o.observation_id DESC
            LIMIT 1
            """,
            (fact_id, *scopes),
        ).fetchone()
        if row is None:
            return None
        keys = (
            "performed_by",
            "agent_id",
            "session_id",
            "source_type",
            "repository",
            "branch",
            "commit_sha",
            "evidence_count",
        )
        return dict(zip(keys, row))

    def _historical_fact(
        self, fact_id: int, scopes: Sequence[str]
    ) -> dict[str, Any] | None:
        placeholders = ",".join("?" for _ in scopes)
        columns = ", ".join(_FACT_FIELDS)
        row = self._conn.execute(
            f"SELECT {columns} FROM facts WHERE fact_id = ? AND scope IN ({placeholders})",
            (fact_id, *scopes),
        ).fetchone()
        return dict(zip(_FACT_FIELDS, row)) if row is not None else None

    def _slot_history(
        self, scopes: Sequence[str], subject: str, predicate: str, limit: int
    ) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in scopes)
        columns = ", ".join(_FACT_FIELDS)
        rows = self._conn.execute(
            f"""
            SELECT {columns} FROM facts
            WHERE scope IN ({placeholders})
              AND subject_key = ? AND predicate_key = ?
            ORDER BY COALESCE(valid_from, created_at), fact_id
            LIMIT ?
            """,
            (*scopes, subject, predicate, limit),
        ).fetchall()
        return [dict(zip(_FACT_FIELDS, row)) for row in rows]

    @staticmethod
    def _requested_scopes(
        context: ConnectionContext, requested: Any
    ) -> tuple[str, ...]:
        if requested is None:
            return context.access_scopes
        try:
            scope = validate_scope(_text(requested, "scope"))
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc
        if scope not in context.access_scopes:
            raise ServiceRequestError("access_denied", "requested memory scope is not authorized")
        return (scope,)
