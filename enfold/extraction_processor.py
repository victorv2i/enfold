"""Fail-closed, model-agnostic processing for attributed extraction jobs.

The processor owns no model or persistent worker.  A host supplies an
``Extractor`` and explicitly calls :meth:`ExtractionProcessor.process_one` or
:meth:`~ExtractionProcessor.drain`.  Queue leases make claims crash-safe; a
validated proposal snapshot is persisted before any fact write, so replay
never asks a nondeterministic model to regenerate an already-applied batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import sqlite3
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import uuid

from .policy import default_credential_screen, validate_scope
from .protocol import ClientContext, Request
from .provenance import WriteRequest
from .service import EnfoldService
from .state_slots import StateCandidate, normalize_predicate_key, normalize_subject_key


MAX_EXTRACTED_MEMORIES = 32
AUTOMATIC_TRUST_SCORE = 0.5
AUTOMATIC_SOURCE_AUTHORITY = 0.5
MIN_TYPED_CONFIDENCE = 0.8
_TYPED_KINDS = frozenset({"state", "preference", "commitment", "event"})
_TYPED_FIELDS = frozenset(
    {
        "kind", "subject", "predicate", "object", "value",
        "occurred_at", "valid_from", "negation", "confidence",
    }
)
_REQUIRED_QUEUE_COLUMNS = frozenset(
    {
        "id",
        "payload",
        "status",
        "payload_hash",
        "attempts",
        "last_error",
        "not_before",
        "lease_owner",
        "lease_until",
        "lease_token",
        "proposal_json",
        "proposal_hash",
    }
)
_SAFE_ERROR_CODES = frozenset(
    {
        "adapter_exit",
        "adapter_cleanup_failed",
        "adapter_input_too_large",
        "adapter_invalid_output",
        "adapter_output_too_large",
        "adapter_timeout",
        "adapter_unavailable",
        "extractor_failed",
        "invalid_envelope",
        "invalid_proposal",
        "invalid_snapshot",
        "proposal_credential_rejected",
        "proposal_limit",
        "proposal_scope_rejected",
        "proposal_sensitivity_rejected",
        "snapshot_hash_mismatch",
        "write_policy_rejected",
    }
)


class ExtractionProcessorUnavailable(RuntimeError):
    """The durable queue does not support safe claimed processing."""


class PermanentExtractionError(ValueError):
    """A job is unsafe or malformed and must go directly to dead letter."""

    def __init__(self, message: str, *, error_code: str = "invalid_proposal") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class ExtractionEnvelope:
    transcript: str
    source: str
    scope: str
    context: ClientContext
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExtractedMemory:
    """One model-produced proposal; the authoritative service still decides."""

    content: str
    category: str = "general"
    tags: str = ""
    trust_score: float = 0.5
    source_authority: float = 0.5
    evidence_excerpt: str | None = None
    scope: str | None = None
    sensitivity: str = "normal"
    state: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Extractor(Protocol):
    """Host-provided model adapter. Implementations may call a model."""

    @property
    def identity(self) -> str:
        """Stable, non-secret extractor/model identity for provenance."""

    def extract(self, envelope: ExtractionEnvelope) -> Sequence[ExtractedMemory]:
        """Return structured proposals without writing storage directly."""


@dataclass(frozen=True, slots=True)
class ExtractionProcessResult:
    outcome: str
    queue_id: int | None
    writes: int = 0
    attempts: int = 0
    error: str | None = None


class ExtractionProcessor:
    """Claim durable jobs and apply proposals through ``EnfoldService``."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        service: EnfoldService,
        extractor: Extractor,
        *,
        worker_id: str | None = None,
        max_attempts: int = 3,
        lease_seconds: float = 300.0,
        heartbeat_seconds: float | None = None,
        retry_delay_seconds: float = 1.0,
        clock: Callable[[], float] = time.time,
    ):
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(extract_queue)")
        }
        missing = sorted(_REQUIRED_QUEUE_COLUMNS - columns)
        if missing:
            raise ExtractionProcessorUnavailable(
                "extract_queue lacks claimed-processing columns: " + ", ".join(missing)
            )
        identity = getattr(extractor, "identity", None)
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("extractor identity must be a non-empty string")
        if max_attempts <= 0 or lease_seconds <= 0 or retry_delay_seconds < 0:
            raise ValueError("invalid extraction retry/lease configuration")
        effective_heartbeat = (
            min(30.0, lease_seconds / 3.0)
            if heartbeat_seconds is None
            else heartbeat_seconds
        )
        if effective_heartbeat <= 0 or effective_heartbeat >= lease_seconds:
            raise ValueError("heartbeat_seconds must be positive and shorter than lease_seconds")
        if conn.in_transaction:
            raise RuntimeError("ExtractionProcessor requires an idle connection")
        self._conn = conn
        self._service = service
        self._extractor = extractor
        self._worker_id = worker_id or f"extractor-{uuid.uuid4().hex}"
        self._max_attempts = max_attempts
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = float(effective_heartbeat)
        self._retry_delay = retry_delay_seconds
        self._clock = clock

    def process_one(self) -> ExtractionProcessResult:
        """Process one due job, or return ``idle`` without model activity."""

        row = self._claim()
        if row is None:
            return ExtractionProcessResult("idle", None)
        row_id, payload, digest, attempts, lease_token = row
        writes = 0
        try:
            envelope = self._decode_envelope(payload)
            snapshot_json, snapshot_hash = self._load_snapshot(row_id, lease_token)
            if snapshot_json is None:
                proposals = self._extract_with_heartbeat(
                    envelope, row_id, lease_token
                )
                snapshot_json, snapshot_hash = self._make_snapshot(proposals, envelope)
                self._persist_snapshot(
                    row_id, lease_token, snapshot_json, snapshot_hash
                )
            prepared = self._prepare_snapshot(
                snapshot_json, snapshot_hash, envelope, row_id, digest
            )
            for index, params in enumerate(prepared):
                response = self._service.handle(
                    envelope.context,
                    Request(
                        f"extract-{row_id}-{index}",
                        "memory.write",
                        params,
                    ),
                )
                if response["outcome"] in {"rejected", "needs_review"}:
                    raise PermanentExtractionError(
                        "authoritative write policy rejected extraction",
                        error_code="write_policy_rejected",
                    )
                writes += 1
            self._complete(row_id, lease_token)
            return ExtractionProcessResult("completed", row_id, writes, attempts)
        except PermanentExtractionError as exc:
            error_code = self._safe_error_code(exc)
            attempts = self._fail(row_id, lease_token, error_code, permanent=True)
            return ExtractionProcessResult("dead", row_id, writes, attempts, error_code)
        except Exception as exc:
            error_code = self._safe_error_code(exc)
            attempts, outcome = self._fail(row_id, lease_token, error_code, permanent=False), "retry"
            if attempts >= self._max_attempts:
                outcome = "dead"
            return ExtractionProcessResult(outcome, row_id, writes, attempts, error_code)

    def drain(self, *, limit: int = 10) -> tuple[ExtractionProcessResult, ...]:
        """Process at most ``limit`` jobs; never loops indefinitely."""

        if isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be positive")
        results: list[ExtractionProcessResult] = []
        for _ in range(limit):
            result = self.process_one()
            if result.outcome == "idle":
                break
            results.append(result)
        return tuple(results)

    @property
    def health(self) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT status, count(*) FROM extract_queue GROUP BY status"
        ).fetchall()
        counts = {str(row[0]): int(row[1]) for row in rows}
        return {
            "configured": True,
            "mode": "explicit_host_driven",
            "extractor": self._extractor.identity,
            "pending": counts.get("pending", 0) + counts.get("processing", 0),
            "dead": counts.get("dead", 0),
        }

    def _claim(self) -> tuple[int, str, str, int, str] | None:
        now = self._clock()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                """
                UPDATE extract_queue
                SET status = 'dead', last_error = 'attempts_exhausted',
                    lease_owner = NULL, lease_until = NULL, lease_token = NULL
                WHERE status = 'processing' AND lease_until IS NOT NULL
                  AND lease_until <= ? AND attempts >= ?
                """,
                (now, self._max_attempts),
            )
            row = self._conn.execute(
                """
                SELECT id, payload, payload_hash, attempts
                FROM extract_queue
                WHERE attempts < ?
                  AND (not_before IS NULL OR not_before <= ?)
                  AND (status = 'pending' OR
                       (status = 'processing' AND lease_until IS NOT NULL
                        AND lease_until <= ?))
                ORDER BY id LIMIT 1
                """,
                (self._max_attempts, now, now),
            ).fetchone()
            if row is None:
                self._conn.commit()
                return None
            row_id = int(row[0])
            lease_token = uuid.uuid4().hex
            attempts = int(row[3]) + 1
            cursor = self._conn.execute(
                """
                UPDATE extract_queue
                SET status = 'processing', attempts = ?, lease_owner = ?,
                    lease_until = ?, lease_token = ?
                WHERE id = ? AND attempts < ?
                """,
                (
                    attempts, self._worker_id, now + self._lease_seconds,
                    lease_token, row_id, self._max_attempts,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("extraction claim was lost")
            self._conn.commit()
            payload = str(row[1])
            digest = str(row[2] or hashlib.sha256(payload.encode()).hexdigest())
            return row_id, payload, digest, attempts, lease_token
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise

    def _complete(self, row_id: int, lease_token: str) -> None:
        cursor = self._conn.execute(
            "DELETE FROM extract_queue WHERE id = ? AND status = 'processing' "
            "AND lease_owner = ? AND lease_token = ?",
            (row_id, self._worker_id, lease_token),
        )
        self._conn.commit()
        if cursor.rowcount != 1:
            raise RuntimeError("extraction lease was lost before completion")

    def _renew(self, row_id: int, lease_token: str) -> None:
        """Extend one live lease without permitting a stale worker to revive it."""

        now = self._clock()
        cursor = self._conn.execute(
            """
            UPDATE extract_queue
            SET lease_until = ?
            WHERE id = ? AND status = 'processing' AND lease_owner = ?
              AND lease_token = ? AND lease_until IS NOT NULL AND lease_until > ?
            """,
            (
                now + self._lease_seconds,
                row_id,
                self._worker_id,
                lease_token,
                now,
            ),
        )
        self._conn.commit()
        if cursor.rowcount != 1:
            raise RuntimeError("extraction lease was lost before renewal")

    def _fail(
        self, row_id: int, lease_token: str, error_code: str, *, permanent: bool
    ) -> int:
        row = self._conn.execute(
            "SELECT attempts FROM extract_queue WHERE id = ? AND status = 'processing' "
            "AND lease_owner = ? AND lease_token = ?",
            (row_id, self._worker_id, lease_token),
        ).fetchone()
        if row is None:
            raise RuntimeError("extraction lease was lost while recording failure")
        attempts = int(row[0])
        dead = permanent or attempts >= self._max_attempts
        self._conn.execute(
            """
            UPDATE extract_queue
            SET attempts = ?, last_error = ?, status = ?, not_before = ?,
                lease_owner = NULL, lease_until = NULL, lease_token = NULL
            WHERE id = ? AND lease_owner = ? AND lease_token = ?
            """,
            (
                attempts,
                self._safe_error_code(error_code),
                "dead" if dead else "pending",
                None if dead else self._clock() + self._retry_delay,
                row_id,
                self._worker_id,
                lease_token,
            ),
        )
        self._conn.commit()
        return attempts

    def _extract_with_heartbeat(
        self,
        envelope: ExtractionEnvelope,
        row_id: int,
        lease_token: str,
    ) -> tuple[ExtractedMemory, ...]:
        """Run a model outside SQLite while renewing only the current fence."""

        done = threading.Event()
        result: dict[str, Any] = {}

        def invoke() -> None:
            try:
                result["proposals"] = tuple(self._extractor.extract(envelope))
            except BaseException as exc:  # relay the original model failure
                result["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(
            target=invoke,
            name="enfold-extraction-call",
            daemon=True,
        )
        thread.start()
        while not done.wait(self._heartbeat_seconds):
            self._renew(row_id, lease_token)
        error = result.get("error")
        if error is not None:
            raise error
        proposals = result.get("proposals")
        if not isinstance(proposals, tuple):
            raise RuntimeError("extractor did not return proposals")
        return proposals

    def _load_snapshot(
        self, row_id: int, lease_token: str
    ) -> tuple[str | None, str | None]:
        row = self._conn.execute(
            """
            SELECT proposal_json, proposal_hash FROM extract_queue
            WHERE id = ? AND status = 'processing' AND lease_owner = ?
              AND lease_token = ?
            """,
            (row_id, self._worker_id, lease_token),
        ).fetchone()
        if row is None:
            raise RuntimeError("extraction lease was lost before snapshot load")
        proposal_json = row[0]
        proposal_hash = row[1]
        if (proposal_json is None) != (proposal_hash is None):
            raise PermanentExtractionError(
                "proposal snapshot is inconsistent", error_code="invalid_snapshot"
            )
        if proposal_json is None:
            return None, None
        if not isinstance(proposal_json, str) or not isinstance(proposal_hash, str):
            raise PermanentExtractionError(
                "proposal snapshot is malformed", error_code="invalid_snapshot"
            )
        return proposal_json, proposal_hash

    def _persist_snapshot(
        self,
        row_id: int,
        lease_token: str,
        proposal_json: str,
        proposal_hash: str,
    ) -> None:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            cursor = self._conn.execute(
                """
                UPDATE extract_queue
                SET proposal_json = ?, proposal_hash = ?
                WHERE id = ? AND status = 'processing' AND lease_owner = ?
                  AND lease_token = ? AND proposal_json IS NULL AND proposal_hash IS NULL
                """,
                (proposal_json, proposal_hash, row_id, self._worker_id, lease_token),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("extraction lease was lost before snapshot persistence")
            self._conn.commit()
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise

    @staticmethod
    def _decode_envelope(payload: str) -> ExtractionEnvelope:
        try:
            data = json.loads(payload)
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("unsupported extraction envelope version")
            provenance = data["provenance"]
            if not isinstance(provenance, dict):
                raise ValueError("provenance must be an object")
            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            scope = validate_scope(str(data.get("scope", "private")))
            context = ClientContext.from_dict(provenance)
            if scope not in context.access_scopes or scope == "secret":
                raise ValueError("extraction scope is unauthorized or secret")
            transcript = str(data["transcript"]).strip()
            source = str(data["source"]).strip()
            if not transcript or not source:
                raise ValueError("transcript and source must be non-empty")
            return ExtractionEnvelope(transcript, source, scope, context, metadata)
        except (KeyError, TypeError, ValueError) as exc:
            raise PermanentExtractionError(
                f"invalid extraction envelope: {exc}", error_code="invalid_envelope"
            ) from exc

    def _make_snapshot(
        self,
        proposals: Sequence[ExtractedMemory],
        envelope: ExtractionEnvelope,
    ) -> tuple[str, str]:
        if len(proposals) > MAX_EXTRACTED_MEMORIES:
            raise PermanentExtractionError(
                "extractor returned too many memories", error_code="proposal_limit"
            )
        normalized: list[dict[str, Any]] = []
        for proposal in proposals:
            if not isinstance(proposal, ExtractedMemory):
                raise PermanentExtractionError(
                    "extractor returned an invalid proposal", error_code="invalid_proposal"
                )
            if not isinstance(proposal.content, str) or not proposal.content.strip():
                raise PermanentExtractionError(
                    "proposal content must be non-empty text", error_code="invalid_proposal"
                )
            if proposal.scope is not None:
                try:
                    requested_scope = validate_scope(proposal.scope)
                except (TypeError, ValueError) as exc:
                    raise PermanentExtractionError(
                        "proposal scope is invalid", error_code="proposal_scope_rejected"
                    ) from exc
                if requested_scope != envelope.scope:
                    raise PermanentExtractionError(
                        "automatic extraction cannot change envelope scope",
                        error_code="proposal_scope_rejected",
                    )
            if proposal.sensitivity not in {"normal", "sensitive"}:
                raise PermanentExtractionError(
                    "proposal sensitivity is not permitted",
                    error_code="proposal_sensitivity_rejected",
                )
            if not isinstance(proposal.category, str) or not proposal.category.strip():
                raise PermanentExtractionError(
                    "proposal category must be non-empty text", error_code="invalid_proposal"
                )
            if not isinstance(proposal.tags, str):
                raise PermanentExtractionError(
                    "proposal tags must be text", error_code="invalid_proposal"
                )
            if proposal.evidence_excerpt is not None and not isinstance(
                proposal.evidence_excerpt, str
            ):
                raise PermanentExtractionError(
                    "proposal evidence excerpt must be text", error_code="invalid_proposal"
                )
            item = {
                    "category": proposal.category.strip(),
                    "content": proposal.content.strip(),
                    "evidence_excerpt": proposal.evidence_excerpt,
                    "sensitivity": proposal.sensitivity,
                    "tags": proposal.tags,
                }
            typed = self._normalize_typed_fields(proposal.state, item["content"])
            if typed is not None:
                item["typed"] = typed
            normalized.append(item)
        snapshot = {
            "extractor_identity": self._extractor.identity,
            "proposals": normalized,
            "version": 1,
        }
        try:
            proposal_json = json.dumps(
                snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise PermanentExtractionError(
                "proposal snapshot is not JSON", error_code="invalid_proposal"
            ) from exc
        return proposal_json, hashlib.sha256(proposal_json.encode("utf-8")).hexdigest()

    def _prepare_snapshot(
        self,
        proposal_json: str,
        proposal_hash: str,
        envelope: ExtractionEnvelope,
        row_id: int,
        digest: str,
    ) -> tuple[dict[str, Any], ...]:
        if hashlib.sha256(proposal_json.encode("utf-8")).hexdigest() != proposal_hash:
            raise PermanentExtractionError(
                "proposal snapshot hash does not match", error_code="snapshot_hash_mismatch"
            )
        try:
            snapshot = json.loads(proposal_json)
            canonical = json.dumps(
                snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise PermanentExtractionError(
                "proposal snapshot is malformed", error_code="invalid_snapshot"
            ) from exc
        if canonical != proposal_json or not isinstance(snapshot, dict):
            raise PermanentExtractionError(
                "proposal snapshot is not canonical", error_code="invalid_snapshot"
            )
        if snapshot.get("version") != 1:
            raise PermanentExtractionError(
                "proposal snapshot version is unsupported", error_code="invalid_snapshot"
            )
        identity = snapshot.get("extractor_identity")
        proposals = snapshot.get("proposals")
        if not isinstance(identity, str) or not identity.strip() or not isinstance(proposals, list):
            raise PermanentExtractionError(
                "proposal snapshot is malformed", error_code="invalid_snapshot"
            )
        if len(proposals) > MAX_EXTRACTED_MEMORIES:
            raise PermanentExtractionError(
                "proposal snapshot exceeds proposal limit", error_code="proposal_limit"
            )
        prepared: list[dict[str, Any]] = []
        for index, proposal in enumerate(proposals):
            base_fields = {
                "category", "content", "evidence_excerpt", "sensitivity", "tags"
            }
            if (
                not isinstance(proposal, dict)
                or not base_fields.issubset(proposal)
                or set(proposal) - base_fields - {"typed"}
            ):
                raise PermanentExtractionError(
                    "proposal snapshot has unsupported fields", error_code="invalid_snapshot"
                )
            content = proposal["content"]
            category = proposal["category"]
            tags = proposal["tags"]
            excerpt = proposal["evidence_excerpt"]
            sensitivity = proposal["sensitivity"]
            typed = proposal.get("typed")
            if (
                not isinstance(content, str)
                or not content
                or not isinstance(category, str)
                or not category
                or not isinstance(tags, str)
                or (excerpt is not None and (not isinstance(excerpt, str) or not excerpt))
                or sensitivity not in {"normal", "sensitive"}
            ):
                raise PermanentExtractionError(
                    "proposal snapshot contains invalid fields", error_code="invalid_snapshot"
                )
            if typed is not None and not self._is_normalized_typed_fields(typed):
                raise PermanentExtractionError(
                    "proposal snapshot contains invalid typed fields",
                    error_code="invalid_snapshot",
                )
            metadata = {
                "extraction_queue_id": row_id,
                "extraction_payload_sha256": digest,
                "extractor_identity": identity,
                "extraction_source": envelope.source,
                "proposal_snapshot_sha256": proposal_hash,
            }
            if typed is not None:
                metadata.update(
                    {
                        "extracted_kind": typed["kind"],
                        "extracted_confidence": typed["confidence"],
                        "extracted_negation": typed["negation"],
                    }
                )
            try:
                request = WriteRequest(
                    idempotency_key=(
                        f"extract:{digest}:{proposal_hash[:24]}:{index}"
                    ),
                    content=content,
                    source_type="automatic_extraction",
                    category=category,
                    tags=tags,
                    trust_score=AUTOMATIC_TRUST_SCORE,
                    source_authority=AUTOMATIC_SOURCE_AUTHORITY,
                    observation_content=envelope.transcript,
                    asserted_by=identity,
                    scope=envelope.scope,
                    sensitivity=sensitivity,
                    evidence_excerpt=excerpt,
                    relation="derived_from",
                    metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False),
                )
            except (TypeError, ValueError, RecursionError) as exc:
                raise PermanentExtractionError(
                    "invalid extraction proposal", error_code="invalid_proposal"
                ) from exc
            decision = default_credential_screen(request)
            if decision is not None:
                raise PermanentExtractionError(
                    "credential-shaped proposal rejected",
                    error_code="proposal_credential_rejected",
                )
            params: dict[str, Any] = {
                "idempotency_key": request.idempotency_key,
                "content": request.content,
                "source_type": request.source_type,
                "category": request.category,
                "tags": request.tags,
                "trust_score": request.trust_score,
                "source_authority": request.source_authority,
                "observation_content": request.observation_content,
                "asserted_by": request.asserted_by,
                "scope": request.scope,
                "sensitivity": request.sensitivity,
                "evidence_excerpt": request.evidence_excerpt,
                "relation": request.relation,
                "metadata": metadata,
            }
            if typed is not None and typed["kind"] == "state":
                params["state"] = {
                    "subject_key": typed["subject_key"],
                    "predicate_key": typed["predicate_key"],
                    "object_value": typed["object_value"],
                    "valid_from": typed["valid_from"],
                }
            prepared.append(params)
        return tuple(prepared)

    @staticmethod
    def _normalize_typed_fields(
        value: Any, content: str
    ) -> dict[str, Any] | None:
        """Return a safe typed payload, or abstain without dropping content."""

        if value is None or not isinstance(value, Mapping):
            return None
        if not value or set(value) - _TYPED_FIELDS:
            return None
        kind = value.get("kind")
        confidence = value.get("confidence")
        negation = value.get("negation", False)
        if (
            kind not in _TYPED_KINDS
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not MIN_TYPED_CONFIDENCE <= float(confidence) <= 1.0
            or not isinstance(negation, bool)
        ):
            return None
        if "object" in value and "value" in value:
            return None
        if "occurred_at" in value and "valid_from" in value:
            return None
        object_value = value.get("object", value.get("value"))
        if negation:
            if object_value is not None:
                return None
            object_value = None
        elif not isinstance(object_value, str) or not object_value.strip():
            return None
        else:
            object_value = object_value.strip()
        valid_from = value.get("valid_from", value.get("occurred_at"))
        if valid_from is not None and (
            not isinstance(valid_from, str) or not valid_from.strip()
        ):
            return None
        try:
            subject_key = normalize_subject_key(value.get("subject"))
            predicate_key = normalize_predicate_key(value.get("predicate"))
            # Reuse the slot type's timestamp validation rather than maintaining
            # a subtly different parser at the model boundary.
            StateCandidate(
                content=content,
                subject_key=subject_key,
                predicate_key=predicate_key,
                object_value=object_value,
                valid_from=valid_from,
            )
        except (TypeError, ValueError):
            return None
        return {
            "confidence": float(confidence),
            "kind": kind,
            "negation": negation,
            "object_value": object_value,
            "predicate_key": predicate_key,
            "subject_key": subject_key,
            "valid_from": valid_from.strip() if valid_from is not None else None,
        }

    @staticmethod
    def _is_normalized_typed_fields(value: Any) -> bool:
        if not isinstance(value, dict) or set(value) != {
            "confidence", "kind", "negation", "object_value",
            "predicate_key", "subject_key", "valid_from",
        }:
            return False
        normalized = ExtractionProcessor._normalize_typed_fields(
            {
                "confidence": value["confidence"],
                "kind": value["kind"],
                "negation": value["negation"],
                "subject": value["subject_key"],
                "predicate": value["predicate_key"],
                "value": value["object_value"],
                "valid_from": value["valid_from"],
            },
            "snapshot validation",
        )
        return normalized == value

    @staticmethod
    def _safe_error_code(exc: BaseException | str) -> str:
        """Return an allowlisted operational code, never adapter/model text."""

        candidate = (
            exc if isinstance(exc, str) else getattr(exc, "error_code", None)
        )
        if isinstance(candidate, str) and candidate in _SAFE_ERROR_CODES:
            return candidate
        return "extractor_failed"
