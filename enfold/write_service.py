"""Transactional, provenance-aware fact write envelope."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from typing import Callable, Optional
import uuid

import numpy as np

from .cluster_merge import NearDuplicateCandidate, find_write_near_duplicates
from .policy import MemoryPolicy, PolicyDecision
from .provenance import ConnectionContext, WriteOutcome, WriteRequest
from .state_slots import (
    SlotDecision,
    StateCandidate,
    add_conflict_member,
    decide_state_write,
    open_state_conflict,
)


class IdempotencyConflict(ValueError):
    """An idempotency key was retried with a different request payload."""


class ClientIdentityConflict(ValueError):
    """A stable client id was reused for a different surface or agent."""


class SessionContextConflict(ValueError):
    """A client/session identity was reused with different immutable context."""


@dataclass(frozen=True, slots=True)
class FactWriteResult:
    """Narrow result contract implemented by the existing fact store."""

    fact_id: int
    outcome: str = "inserted"
    existing_fact_id: Optional[int] = None
    detail_json: str = "{}"

    def __post_init__(self) -> None:
        if self.fact_id <= 0:
            raise ValueError("fact_id must be positive")
        if self.existing_fact_id is not None and self.existing_fact_id <= 0:
            raise ValueError("existing_fact_id must be positive")
        decoded = json.loads(self.detail_json)
        if not isinstance(decoded, dict):
            raise ValueError("detail_json must be a JSON object")
        object.__setattr__(
            self,
            "detail_json",
            json.dumps(decoded, sort_keys=True, separators=(",", ":")),
        )


FactWriter = Callable[
    [sqlite3.Connection, WriteRequest, int],
    FactWriteResult,
]


@dataclass(frozen=True, slots=True)
class NearDedupConfig:
    """Conservative controls for embedding-backed write-time consolidation.

    ``query_embedder`` remains intentionally separate from the durable
    embedding job queue: when it is absent, no embedding identity is supplied,
    or any candidate lacks a stored vector, writes retain the exact-dedup path
    instead of waiting on a job.
    """

    enabled: bool = True
    cosine_threshold: float = 0.97
    candidate_limit: int = 64
    embedding_identity: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("near dedup enabled must be a boolean")
        if not 0.0 <= self.cosine_threshold <= 1.0:
            raise ValueError("near dedup cosine threshold must be between 0 and 1")
        if self.candidate_limit <= 0:
            raise ValueError("near dedup candidate limit must be positive")
        if self.embedding_identity is not None and not self.embedding_identity.strip():
            raise ValueError("near dedup embedding identity must be non-empty")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _request_sha256(
    context: ConnectionContext,
    request: WriteRequest,
    state_candidate: Optional[StateCandidate] = None,
) -> str:
    # Session identity affects the observation and therefore belongs to the
    # replay contract.  Volatile repository context does not: adapters may
    # learn it after a retry without changing the requested write itself.
    payload = {
        "client_id": context.client_id,
        "session_id": context.session_id,
        "request": asdict(request),
        "state_candidate": (
            asdict(state_candidate) if state_candidate is not None else None
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observation_sha256(
    context: ConnectionContext, request: WriteRequest
) -> str:
    payload = {
        "content": request.observation_content or request.content,
        "source_type": request.source_type,
        "source_uri": request.source_uri,
        "asserted_by": request.asserted_by,
        "performed_by": request.performed_by,
        "observed_at": request.observed_at,
        "scope": request.scope,
        "sensitivity": request.sensitivity,
        "metadata_json": request.metadata_json,
        "project_root": context.project_root,
        "repository": context.repository,
        "branch": context.branch,
        "commit_sha": context.commit_sha,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MemoryWriteService:
    """Perform one complete durable write using a caller-provided connection.

    ``fact_writer`` is the only integration point with the existing fact
    store.  It runs inside the same ``BEGIN IMMEDIATE`` transaction and must
    neither commit nor roll back.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        fact_writer: FactWriter,
        policy: MemoryPolicy,
        embedding_enqueue: Callable[[int], int] | None = None,
        *,
        near_dedup: NearDedupConfig | None = None,
        query_embedder: Callable[[str], object] | None = None,
    ):
        self._conn = conn
        self._fact_writer = fact_writer
        self._policy = policy
        self._embedding_enqueue = embedding_enqueue
        self._near_dedup = near_dedup or NearDedupConfig()
        self._query_embedder = query_embedder
        if conn.in_transaction:
            raise RuntimeError("MemoryWriteService requires an idle connection")
        conn.execute("PRAGMA foreign_keys = ON")
        if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
            raise RuntimeError("MemoryWriteService requires foreign key enforcement")

    def write(
        self,
        context: ConnectionContext,
        request: WriteRequest,
        *,
        state_candidate: Optional[StateCandidate] = None,
    ) -> WriteOutcome:
        if self._conn.in_transaction:
            raise RuntimeError("MemoryWriteService requires an idle connection")
        context = self._policy.authorize_context(context)
        self._validate_state_candidate(request, state_candidate)

        request_hash = _request_sha256(context, request, state_candidate)
        recorded_at = _now()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._register_client(context, recorded_at)
            prior = self._load_prior(context.client_id, request.idempotency_key)
            if prior is not None:
                if prior["request_sha256"] != request_hash:
                    raise IdempotencyConflict(
                        "idempotency key was already used for a different request"
                    )
                outcome = self._outcome_from_row(prior, replayed=True)
                self._conn.commit()
                return outcome

            self._register_session(context, recorded_at)
            sensitive_fields = ()
            if state_candidate is not None:
                sensitive_fields = tuple(
                    value for value in (
                        state_candidate.subject_key,
                        state_candidate.predicate_key,
                        state_candidate.object_value,
                    ) if value
                )
            decision = self._policy.evaluate_write(
                request,
                client_id=context.client_id,
                sensitive_fields=sensitive_fields,
            )
            if decision is None and request.scope not in context.access_scopes:
                decision = PolicyDecision(
                    "rejected", "requested write scope is not server-authorized"
                )
            if decision is None:
                decision = self._supersession_policy(request)
            # A factless policy decision must not inspect a secret or
            # unauthorized state slot.  The candidate remains part of the
            # idempotency hash above, but slot lookup is deferred until the
            # write itself is authorized.
            effective_candidate = state_candidate
            if effective_candidate is not None and effective_candidate.valid_from is None:
                effective_candidate = replace(
                    effective_candidate,
                    valid_from=request.observed_at or recorded_at,
                )
            state_decision = (
                decide_state_write(self._conn, effective_candidate)
                if decision is None and state_candidate is not None
                else None
            )
            if (
                decision is None
                and state_decision is not None
                and state_decision.action == "supersede"
            ):
                decision = self._supersession_policy(
                    request,
                    target_id=state_decision.target_fact_id,
                    candidate_authority=effective_candidate.source_authority,
                )
            if decision is not None:
                outcome = self._record_factless_decision(
                    context, request, request_hash, recorded_at, decision
                )
                self._conn.commit()
                return outcome
            observation_id = self._record_observation(
                context, request, recorded_at
            )
            conflict_id: Optional[str] = None
            untyped_duplicate = (
                self._find_untyped_exact_duplicate(request)
                if state_candidate is None
                else None
            )
            near_duplicate = (
                self._find_untyped_near_duplicate(request)
                if state_candidate is None
                and request.supersede_fact_id is None
                and untyped_duplicate is None
                else None
            )
            enqueue_fact_id: Optional[int] = None
            if (
                state_decision is not None
                and state_decision.action in {"dedup", "conflict"}
                and state_decision.target_fact_id is not None
            ):
                if state_decision.action == "conflict":
                    row = self._conn.execute(
                        "SELECT conflict_group FROM facts WHERE fact_id = ?",
                        (state_decision.target_fact_id,),
                    ).fetchone()
                    conflict_id = str(row[0]) if row and row[0] else None
                fact_result = FactWriteResult(
                    state_decision.target_fact_id,
                    outcome=state_decision.action,
                    existing_fact_id=state_decision.target_fact_id,
                )
            elif untyped_duplicate is not None:
                fact_result = FactWriteResult(
                    untyped_duplicate,
                    outcome="dedup",
                    existing_fact_id=untyped_duplicate,
                )
            else:
                if state_decision is not None:
                    conflict_id = self._prepare_state_mutation(
                        state_decision, recorded_at
                    )
                fact_result = self._fact_writer(
                    self._conn, request, observation_id
                )
                if near_duplicate is not None:
                    fact_result, enqueue_fact_id = self._merge_near_duplicate(
                        fact_result, near_duplicate, request, recorded_at
                    )
                else:
                    enqueue_fact_id = fact_result.fact_id
                if effective_candidate is not None and state_decision is not None:
                    self._persist_state_candidate(
                        fact_result, effective_candidate, conflict_id
                    )
                    if state_decision.action == "supersede":
                        self._finish_state_supersession(
                            state_decision.target_fact_id, fact_result.fact_id
                        )
                    elif state_decision.action == "conflict":
                        if conflict_id is None:
                            raise RuntimeError("state conflict group was not established")
                        add_conflict_member(
                            self._conn, conflict_id, fact_result.fact_id
                        )
                    fact_result = FactWriteResult(
                        fact_result.fact_id,
                        outcome=state_decision.action,
                        existing_fact_id=fact_result.existing_fact_id,
                        detail_json=fact_result.detail_json,
                    )
            if enqueue_fact_id is not None and self._embedding_enqueue is not None:
                self._embedding_enqueue(enqueue_fact_id)
            self._authorize_fact_scope(fact_result.fact_id, request.scope)
            self._attach_provenance(
                fact_result.fact_id, observation_id, request, recorded_at
            )
            superseded = False
            if state_candidate is None:
                superseded = self._supersede_if_requested(
                    request.supersede_fact_id, fact_result.fact_id, recorded_at
                )

            detail = json.loads(fact_result.detail_json)
            if near_duplicate is not None:
                detail["near_duplicate"] = {
                    "candidate_fact_id": near_duplicate.fact_id,
                    "cosine": round(near_duplicate.cosine, 6),
                    "survivor_fact_id": fact_result.fact_id,
                }
            if superseded:
                detail["superseded_fact_id"] = request.supersede_fact_id
            if state_decision is not None:
                detail["state_action"] = state_decision.action
                detail["state_slot"] = {
                    "scope": state_decision.scope,
                    "subject_key": state_decision.subject_key,
                    "predicate_key": state_decision.predicate_key,
                }
                if state_decision.action == "supersede":
                    detail["superseded_fact_id"] = state_decision.target_fact_id
                if conflict_id is not None:
                    detail["conflict_id"] = conflict_id
            detail_json = json.dumps(detail, sort_keys=True, separators=(",", ":"))
            write_id = str(uuid.uuid4())
            self._conn.execute(
                """
                INSERT INTO memory_write_log (
                    write_id, idempotency_key, client_id, session_id,
                    operation, outcome, fact_id, existing_fact_id,
                    observation_id, recorded_at, request_sha256, detail_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    write_id,
                    request.idempotency_key,
                    context.client_id,
                    context.session_id,
                    request.operation,
                    fact_result.outcome,
                    fact_result.fact_id,
                    fact_result.existing_fact_id,
                    observation_id,
                    recorded_at,
                    request_hash,
                    detail_json,
                ),
            )
            outcome = WriteOutcome(
                write_id=write_id,
                outcome=fact_result.outcome,
                fact_id=fact_result.fact_id,
                existing_fact_id=fact_result.existing_fact_id,
                observation_id=observation_id,
                detail_json=detail_json,
            )
            self._conn.commit()
            return outcome
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise

    @staticmethod
    def _validate_state_candidate(
        request: WriteRequest, candidate: Optional[StateCandidate]
    ) -> None:
        if candidate is None:
            return
        if candidate.memory_kind != "state":
            raise ValueError("state_candidate must have memory_kind='state'")
        if candidate.content != request.content:
            raise ValueError("state candidate content must match the write request")
        if candidate.scope != request.scope:
            raise ValueError("state candidate scope must match the write request")
        if candidate.source_authority != request.source_authority:
            raise ValueError(
                "state candidate authority must match the write request"
            )
        if request.supersede_fact_id is not None:
            raise ValueError(
                "typed state writes derive supersession from the exact slot"
            )

    def _prepare_state_mutation(
        self, decision: SlotDecision, now: str
    ) -> Optional[str]:
        if decision.action == "supersede":
            if decision.target_fact_id is None:
                raise RuntimeError("supersession decision has no target")
            cursor = self._conn.execute(
                """
                UPDATE facts SET invalid_at = ?
                WHERE fact_id = ? AND scope = ?
                  AND invalid_at IS NULL AND superseded_by IS NULL
                  AND conflict_group IS NULL
                """,
                (now, decision.target_fact_id, decision.scope),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("state supersession target is no longer current")
            return None
        if decision.action != "conflict":
            return None

        placeholders = ",".join("?" for _ in decision.current_fact_ids)
        groups = {
            row[0]
            for row in self._conn.execute(
                f"SELECT conflict_group FROM facts WHERE fact_id IN ({placeholders})",
                decision.current_fact_ids,
            ).fetchall()
            if row[0] is not None
        }
        if groups:
            if len(groups) != 1:
                raise RuntimeError("state slot spans multiple unresolved conflicts")
            return str(next(iter(groups)))
        conflict = open_state_conflict(
            self._conn,
            decision.subject_key,
            decision.predicate_key,
            decision.current_fact_ids,
            scope=decision.scope,
            detected_at=now,
            detail_json=json.dumps({"reason": decision.reason}),
        )
        return conflict.conflict_id

    def _persist_state_candidate(
        self,
        result: FactWriteResult,
        candidate: StateCandidate,
        conflict_id: Optional[str],
    ) -> None:
        if result.existing_fact_id is not None or result.outcome not in {
            "inserted", "add"
        }:
            raise RuntimeError(
                "typed state creation requires the fact writer to insert a new fact"
            )
        cursor = self._conn.execute(
            """
            UPDATE facts
            SET memory_kind = 'state', subject_key = ?, predicate_key = ?,
                object_value = ?, source_authority = ?, valid_from = ?,
                scope = ?, conflict_group = ?
            WHERE fact_id = ? AND invalid_at IS NULL AND superseded_by IS NULL
            """,
            (
                candidate.subject_key,
                candidate.predicate_key,
                candidate.object_value,
                candidate.source_authority,
                candidate.valid_from,
                candidate.scope,
                conflict_id,
                result.fact_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("fact writer returned an unusable state fact")

    def _finish_state_supersession(
        self, old_fact_id: Optional[int], new_fact_id: int
    ) -> None:
        if old_fact_id is None:
            raise RuntimeError("supersession decision has no target")
        cursor = self._conn.execute(
            """
            UPDATE facts SET superseded_by = ?
            WHERE fact_id = ? AND invalid_at IS NOT NULL
              AND superseded_by IS NULL
            """,
            (new_fact_id, old_fact_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("could not finalize state supersession")

    def _record_factless_decision(
        self,
        context: ConnectionContext,
        request: WriteRequest,
        request_hash: str,
        recorded_at: str,
        decision: PolicyDecision,
    ) -> WriteOutcome:
        write_id = str(uuid.uuid4())
        detail_json = json.dumps(
            {"policy_reason": decision.reason},
            sort_keys=True,
            separators=(",", ":"),
        )
        self._conn.execute(
            """
            INSERT INTO memory_write_log (
                write_id, idempotency_key, client_id, session_id,
                operation, outcome, fact_id, existing_fact_id,
                observation_id, recorded_at, request_sha256, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?)
            """,
            (
                write_id,
                request.idempotency_key,
                context.client_id,
                context.session_id,
                request.operation,
                decision.outcome,
                recorded_at,
                request_hash,
                detail_json,
            ),
        )
        return WriteOutcome(
            write_id=write_id,
            outcome=decision.outcome,
            fact_id=None,
            detail_json=detail_json,
        )

    def _supersession_policy(
        self,
        request: WriteRequest,
        *,
        target_id: Optional[int] = None,
        candidate_authority: Optional[float] = None,
    ) -> Optional[PolicyDecision]:
        """Prevent automation from silently replacing protected truth."""

        target_id = target_id or request.supersede_fact_id
        if target_id is None:
            return None
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)")}
        selected = ["fact_id"]
        selected.append(
            "COALESCE(source_authority, 0.5)" if "source_authority" in columns else "0.5"
        )
        selected.append("correction_status" if "correction_status" in columns else "NULL")
        selected.append("memory_kind" if "memory_kind" in columns else "NULL")
        selected.append("conflict_group" if "conflict_group" in columns else "NULL")
        row = self._conn.execute(
            f"SELECT {', '.join(selected)} FROM facts WHERE fact_id = ? AND scope = ?",
            (target_id, request.scope),
        ).fetchone()
        if row is None:
            return PolicyDecision("needs_review", "supersession target is unavailable")
        target_authority = float(row[1])
        correction_status = row[2]
        if request.supersede_fact_id is not None:
            if row[3] == "state":
                return PolicyDecision(
                    "needs_review", "typed state requires state-slot supersession"
                )
            if row[4] is not None:
                return PolicyDecision(
                    "needs_review", "open-conflict members cannot be explicitly superseded"
                )
        candidate_is_human = (
            request.source_type == "human_correction"
            or request.relation == "corrects"
            or request.correction_status in {"human_corrected", "human_confirmed"}
        )
        if correction_status in {"human_corrected", "human_confirmed"} and not candidate_is_human:
            return PolicyDecision("needs_review", "target is protected by human correction")
        authority = (
            request.source_authority
            if candidate_authority is None
            else candidate_authority
        )
        if target_authority > authority:
            return PolicyDecision("needs_review", "target has higher source authority")
        return None

    def _find_untyped_exact_duplicate(self, request: WriteRequest) -> Optional[int]:
        """Find only an exact active duplicate inside the authorized write scope.

        This deliberately avoids semantic guesses at the write boundary and
        includes the scope predicate in the lookup, so neither result nor
        timing depends on facts the caller cannot access.
        """

        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)")}
        required = {"scope", "invalid_at", "superseded_by", "conflict_group"}
        if not required.issubset(columns):
            return None
        row = self._conn.execute(
            """
            SELECT fact_id FROM facts
            WHERE scope = ? AND content = ?
              AND invalid_at IS NULL AND superseded_by IS NULL
              AND conflict_group IS NULL
            ORDER BY fact_id LIMIT 1
            """,
            (request.scope, request.content),
        ).fetchone()
        return int(row[0]) if row is not None else None

    def _find_untyped_near_duplicate(
        self, request: WriteRequest
    ) -> Optional[NearDuplicateCandidate]:
        """Return the strongest safe FTS-bounded embedding match, if available."""
        if (
            not self._near_dedup.enabled
            or self._query_embedder is None
            or self._near_dedup.embedding_identity is None
        ):
            return None
        try:
            query_embedding = np.asarray(
                self._query_embedder(request.content), dtype=np.float32
            )
            candidates = find_write_near_duplicates(
                self._conn,
                content=request.content,
                scope=request.scope,
                query_embedding=query_embedding,
                threshold=self._near_dedup.cosine_threshold,
                candidate_limit=self._near_dedup.candidate_limit,
                embedding_identity=self._near_dedup.embedding_identity,
            )
        except (TypeError, ValueError, sqlite3.DatabaseError):
            # The async embedding job may not have completed yet, or a legacy
            # store may not expose FTS/vector tables. Exact dedup remains safe.
            return None
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (item.trust_score, item.created_at, item.fact_id),
        )

    def _merge_near_duplicate(
        self,
        inserted: FactWriteResult,
        candidate: NearDuplicateCandidate,
        request: WriteRequest,
        recorded_at: str,
    ) -> tuple[FactWriteResult, Optional[int]]:
        """Keep one fact active and retain the other in its history chain."""
        incoming_wins = (request.trust_score, recorded_at) > (
            candidate.trust_score,
            candidate.created_at,
        )
        if incoming_wins:
            self._supersede_near_duplicate(candidate.fact_id, inserted.fact_id, recorded_at)
            return (
                FactWriteResult(
                    inserted.fact_id,
                    outcome="near_dedup",
                    existing_fact_id=candidate.fact_id,
                    detail_json=inserted.detail_json,
                ),
                inserted.fact_id,
            )
        self._supersede_near_duplicate(inserted.fact_id, candidate.fact_id, recorded_at)
        return (
            FactWriteResult(
                candidate.fact_id,
                outcome="near_dedup",
                existing_fact_id=candidate.fact_id,
                detail_json=inserted.detail_json,
            ),
            None,
        )

    def _supersede_near_duplicate(
        self, loser_id: int, survivor_id: int, recorded_at: str
    ) -> None:
        cursor = self._conn.execute(
            """
            UPDATE facts SET invalid_at = ?, superseded_by = ?
            WHERE fact_id = ? AND invalid_at IS NULL AND superseded_by IS NULL
            """,
            (recorded_at, survivor_id, loser_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("near-duplicate candidate is no longer active")

    def _register_client(self, context: ConnectionContext, now: str) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_clients (
                client_id, surface, display_name, created_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(client_id) DO NOTHING
            """,
            (
                context.client_id,
                context.surface,
                context.display_name,
                now,
            ),
        )
        row = self._conn.execute(
            "SELECT surface, disabled_at FROM memory_clients WHERE client_id = ?",
            (context.client_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to register memory client")
        if row[0] != context.surface:
            raise ClientIdentityConflict(
                "client_id is already registered to a different surface"
            )
        if row[1] is not None:
            raise PermissionError("memory client is disabled")

    def _register_session(self, context: ConnectionContext, now: str) -> None:
        capabilities_json = json.dumps(context.capabilities, separators=(",", ":"))
        access_scopes_json = json.dumps(context.access_scopes, separators=(",", ":"))
        self._conn.execute(
            """
            INSERT INTO memory_sessions (
                session_id, client_id, agent_id, parent_agent_id, project_root,
                repository, branch, commit_sha, capabilities_json,
                access_scopes_json, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_id, session_id) DO NOTHING
            """,
            (
                context.session_id,
                context.client_id,
                context.agent_id,
                context.parent_agent_id,
                context.project_root,
                context.repository,
                context.branch,
                context.commit_sha,
                capabilities_json,
                access_scopes_json,
                context.started_at or now,
            ),
        )
        # Branch and commit can advance during one long-running coding
        # session. Keep their latest values here; every observation below
        # records the exact values present for its write.
        self._conn.execute(
            """
            UPDATE memory_sessions SET branch = ?, commit_sha = ?
            WHERE client_id = ? AND session_id = ?
            """,
            (context.branch, context.commit_sha, context.client_id, context.session_id),
        )
        row = self._conn.execute(
            """
            SELECT agent_id, parent_agent_id, project_root, repository,
                   capabilities_json, access_scopes_json
            FROM memory_sessions
            WHERE client_id = ? AND session_id = ?
            """,
            (context.client_id, context.session_id),
        ).fetchone()
        expected = (
            context.agent_id,
            context.parent_agent_id,
            context.project_root,
            context.repository,
            capabilities_json,
            access_scopes_json,
        )
        if row is None or tuple(row) != expected:
            raise SessionContextConflict(
                "client_id/session_id was reused with different connection context"
            )

    def _record_observation(
        self,
        context: ConnectionContext,
        request: WriteRequest,
        now: str,
    ) -> int:
        content_hash = _observation_sha256(context, request)
        self._conn.execute(
            """
            INSERT INTO observations (
                client_id, session_id, source_type, source_uri,
                project_root, repository, branch, commit_sha, content,
                content_sha256, asserted_by, performed_by, observed_at,
                recorded_at, scope, sensitivity, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_id, content_sha256, session_id, source_type)
            DO NOTHING
            """,
            (
                context.client_id,
                context.session_id,
                request.source_type,
                request.source_uri,
                context.project_root,
                context.repository,
                context.branch,
                context.commit_sha,
                request.observation_content or request.content,
                content_hash,
                request.asserted_by,
                request.performed_by,
                request.observed_at,
                now,
                request.scope,
                request.sensitivity,
                request.metadata_json,
            ),
        )
        row = self._conn.execute(
            """
            SELECT observation_id FROM observations
            WHERE client_id = ? AND content_sha256 = ?
              AND session_id = ? AND source_type = ?
            """,
            (
                context.client_id,
                content_hash,
                context.session_id,
                request.source_type,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to record observation")
        return int(row[0])

    def _attach_provenance(
        self,
        fact_id: int,
        observation_id: int,
        request: WriteRequest,
        now: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO fact_provenance (
                fact_id, observation_id, relation, evidence_excerpt, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(fact_id, observation_id, relation) DO NOTHING
            """,
            (
                fact_id,
                observation_id,
                request.relation,
                request.evidence_excerpt,
                now,
            ),
        )

    def _supersede_if_requested(
        self,
        old_fact_id: Optional[int],
        new_fact_id: int,
        now: str,
    ) -> bool:
        if old_fact_id is None:
            return False
        if old_fact_id == new_fact_id:
            raise ValueError("a fact cannot supersede itself")
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(facts)")
        }
        required = {"invalid_at", "superseded_by"}
        if not required.issubset(columns):
            raise RuntimeError(
                "structural supersession requires temporal facts columns"
            )
        request_scope = "private"
        if "scope" in columns:
            replacement = self._conn.execute(
                "SELECT scope FROM facts WHERE fact_id = ?", (new_fact_id,)
            ).fetchone()
            if replacement is None:
                raise RuntimeError("replacement fact is unavailable")
            request_scope = str(replacement[0])
            target = self._conn.execute(
                "SELECT scope FROM facts WHERE fact_id = ? AND scope = ?",
                (old_fact_id, request_scope),
            ).fetchone()
            if target is None:
                raise ValueError("superseded fact is unavailable or no longer current")
        memory_expr = "memory_kind" if "memory_kind" in columns else "NULL"
        conflict_expr = "conflict_group" if "conflict_group" in columns else "NULL"
        protected = self._conn.execute(
            f"SELECT {memory_expr}, {conflict_expr} FROM facts WHERE fact_id = ?",
            (old_fact_id,),
        ).fetchone()
        if protected is None:
            raise ValueError("superseded fact is unavailable or no longer current")
        if protected[0] == "state" or protected[1] is not None:
            raise ValueError("typed or conflicted facts require their dedicated resolution path")
        cursor = self._conn.execute(
            """
            UPDATE facts
            SET invalid_at = ?, superseded_by = ?
            WHERE fact_id = ? AND scope = ?
              AND invalid_at IS NULL AND superseded_by IS NULL
            """,
            (now, new_fact_id, old_fact_id, request_scope),
        )
        if cursor.rowcount != 1:
            raise ValueError("superseded fact is unavailable or no longer current")
        return True

    def _authorize_fact_scope(self, fact_id: int, requested_scope: str) -> None:
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)")}
        if "scope" not in columns:
            if requested_scope != "private":
                raise RuntimeError("legacy facts schema cannot persist requested scope")
            return
        row = self._conn.execute(
            "SELECT scope FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("fact writer returned a missing fact")
        if row[0] != requested_scope:
            raise PermissionError("fact writer persisted a different memory scope")

    def _load_prior(self, client_id: str, idempotency_key: str):
        old_factory = self._conn.row_factory
        try:
            self._conn.row_factory = sqlite3.Row
            return self._conn.execute(
                """
                SELECT * FROM memory_write_log
                WHERE client_id = ? AND idempotency_key = ?
                """,
                (client_id, idempotency_key),
            ).fetchone()
        finally:
            self._conn.row_factory = old_factory

    @staticmethod
    def _outcome_from_row(row: sqlite3.Row, replayed: bool) -> WriteOutcome:
        return WriteOutcome(
            write_id=row["write_id"],
            outcome=row["outcome"],
            fact_id=row["fact_id"],
            existing_fact_id=row["existing_fact_id"],
            observation_id=row["observation_id"],
            replayed=replayed,
            detail_json=row["detail_json"],
        )
