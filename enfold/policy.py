"""Deterministic server-side privacy and write policy.

Clients may *request* scopes, but they never grant scopes to themselves.  A
trusted daemon constructs this policy from local configuration and resolves a
connection to the intersection of requested and configured scopes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Optional

from .provenance import ConnectionContext, WriteRequest


BASE_SCOPES = frozenset({"private", "work", "public", "sensitive", "secret"})
SENSITIVITIES = frozenset({"normal", "sensitive", "secret"})
PROVENANCE_RELATIONS = frozenset(
    {"supports", "contradicts", "verifies", "corrects", "derived_from"}
)
CORRECTION_STATUSES = frozenset(
    {"unreviewed", "human_corrected", "human_confirmed"}
)
_PROJECT_SCOPE = re.compile(r"^project:[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class UnknownMemoryClient(PermissionError):
    """A client without a server-side grant attempted to connect."""


def validate_scope(value: str) -> str:
    if value in BASE_SCOPES or _PROJECT_SCOPE.fullmatch(value):
        return value
    raise ValueError(f"unsupported memory scope: {value!r}")


def validate_sensitivity(value: str) -> str:
    if value not in SENSITIVITIES:
        raise ValueError(f"unsupported memory sensitivity: {value!r}")
    return value


def validate_relation(value: str) -> str:
    if value not in PROVENANCE_RELATIONS:
        raise ValueError(f"unsupported provenance relation: {value!r}")
    return value


def validate_correction_status(value: Optional[str]) -> Optional[str]:
    if value is not None and value not in CORRECTION_STATUSES:
        raise ValueError(f"unsupported correction status: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    outcome: str
    reason: str

    def __post_init__(self) -> None:
        if self.outcome not in {"rejected", "needs_review"}:
            raise ValueError("policy outcome must be rejected or needs_review")
        if not self.reason.strip():
            raise ValueError("policy reason must not be empty")


CredentialScreen = Callable[[WriteRequest], Optional[PolicyDecision]]


_CREDENTIAL_SHAPES = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:ghp|github_pat|sk-[A-Za-z0-9_-]{8})[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"(?i)\b(?:api[_ -]?key|client[_ -]?secret|password|token)\b['\"]?\s*[:=]\s*['\"]?[^\s'\"]{8,}"
    ),
    re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^\s:/]+:[^\s/@]+@"),
    re.compile(r"(?i)\b(?:accountkey|sharedaccesssignature|aws_secret_access_key)\s*=\s*[^\s;]{8,}"),
)


def default_credential_screen(
    request: WriteRequest, extra_text: Iterable[str] = ()
) -> Optional[PolicyDecision]:
    """Reject common credential shapes without retaining the matched value."""

    candidate = "\n".join(
        part
        for part in (
            request.content,
            request.observation_content or "",
            request.evidence_excerpt or "",
            request.source_uri or "",
            request.asserted_by or "",
            request.category,
            request.tags,
            request.metadata_json,
            *extra_text,
        )
        if part
    )
    if any(pattern.search(candidate) for pattern in _CREDENTIAL_SHAPES):
        return PolicyDecision("rejected", "credential-shaped content")
    return None


class MemoryPolicy:
    """Immutable grants and deterministic pre-write policy.

    ``client_scope_grants`` comes from server configuration or a trusted
    registry. Unknown clients fail closed. Custom screens supplement (and do
    not disable) the built-in credential screen.
    """

    def __init__(
        self,
        client_scope_grants: Mapping[str, tuple[str, ...]],
        *,
        credential_screens: tuple[CredentialScreen, ...] = (),
        correction_authorities: tuple[str, ...] = (),
        conflict_resolution_authorities: tuple[str, ...] = (),
    ) -> None:
        normalized: dict[str, tuple[str, ...]] = {}
        for client_id, scopes in client_scope_grants.items():
            key = client_id.strip()
            if not key:
                raise ValueError("client grant id must not be empty")
            validated = tuple(dict.fromkeys(validate_scope(scope) for scope in scopes))
            if not validated:
                raise ValueError("client scope grant must not be empty")
            normalized[key] = validated
        self._grants = MappingProxyType(normalized)
        self._screens = credential_screens
        self._correction_authorities = frozenset(correction_authorities)
        self._conflict_resolution_authorities = frozenset(
            conflict_resolution_authorities
        )

    def authorize_context(self, context: ConnectionContext) -> ConnectionContext:
        grants = self._grants.get(context.client_id)
        if grants is None:
            raise UnknownMemoryClient(
                f"memory client {context.client_id!r} has no server-side scope grant"
            )
        requested = tuple(validate_scope(scope) for scope in context.access_scopes)
        effective = tuple(scope for scope in requested if scope in grants)
        if not effective:
            raise PermissionError("requested scopes do not intersect server-side grants")
        return replace(context, access_scopes=effective)

    def can_assert_correction(self, client_id: str) -> bool:
        return client_id in self._correction_authorities

    def can_resolve_conflicts(self, client_id: str) -> bool:
        return client_id in self._conflict_resolution_authorities

    def evaluate_write(
        self,
        request: WriteRequest,
        *,
        client_id: str | None = None,
        sensitive_fields: Iterable[str] = (),
    ) -> Optional[PolicyDecision]:
        validate_scope(request.scope)
        validate_sensitivity(request.sensitivity)
        validate_relation(request.relation)
        validate_correction_status(request.correction_status)
        if request.scope == "secret" or request.sensitivity == "secret":
            return PolicyDecision("rejected", "secret durable writes are disabled")
        claims_correction = (
            request.source_type == "human_correction"
            or request.relation == "corrects"
            or request.correction_status in {"human_corrected", "human_confirmed"}
        )
        if claims_correction and (
            client_id is None or not self.can_assert_correction(client_id)
        ):
            return PolicyDecision(
                "needs_review", "client is not authorized to assert human correction"
            )
        decision = default_credential_screen(request, sensitive_fields)
        if decision is not None:
            return decision
        for screen in self._screens:
            decision = screen(request)
            if decision is not None:
                return decision
        return None
