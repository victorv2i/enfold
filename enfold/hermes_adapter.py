"""Hermes-facing adapter for Enfold's standalone daemon protocol.

This module deliberately imports no Hermes package and opens no database.  A
Hermes integration supplies lifecycle context, then all canonical memory work
goes through :class:`EnfoldClient`.  Keeping the host boundary this small makes
the adapter testable without installing or importing Hermes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from .client import ClientConfig, EnfoldClient, EnfoldTransportError
from .mcp_proxy import MemoryMCPProxy, MemoryTransport
from .protocol import ClientContext


HERMES_CLIENT_ID = "hermes-native"
_READ_METHODS = frozenset(
    {
        "memory.search", "memory.context", "memory.evidence", "memory.history",
        "memory.conflicts",
    }
)


class ReadOnlyDegradedProvider(Protocol):
    """Optional, explicitly configured source for unavailable-daemon reads.

    Implementations must be read-only.  Enfold never supplies a SQLite-backed
    implementation here, so this adapter cannot accidentally create a second
    writer or bypass the daemon's transaction and policy boundaries.
    """

    def read(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        context: ClientContext,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class DegradedReadResult:
    """A visibly non-authoritative result from an opt-in read fallback."""

    result: Any
    reason: str = "Enfold daemon unavailable; result is read-only and may be stale"
    degraded: bool = True


def _host_value(host: object, name: str, default: Any = None) -> Any:
    if isinstance(host, Mapping):
        return host.get(name, default)
    return getattr(host, name, default)


@dataclass(frozen=True, slots=True)
class HermesSessionContext:
    """Immutable identity captured from one Hermes lifecycle session."""

    agent_id: str
    session_id: str
    access_scopes: tuple[str, ...]
    parent_agent_id: str | None = None
    project_root: str | None = None
    repository: str | None = None
    branch: str | None = None
    commit_sha: str | None = None

    def __post_init__(self) -> None:
        # Canonical protocol validation prevents subtly different validation at
        # the Hermes boundary and daemon handshake.
        self.to_client_context()

    @classmethod
    def from_host(
        cls,
        host: object,
        *,
        access_scopes: tuple[str, ...],
    ) -> "HermesSessionContext":
        """Capture context from a mapping or attribute-shaped Hermes host.

        Scopes are never inferred from the host.  They must be selected by
        trusted adapter configuration and are narrowed again by daemon policy.
        """

        return cls(
            agent_id=_host_value(host, "agent_id"),
            session_id=_host_value(host, "session_id"),
            access_scopes=tuple(access_scopes),
            parent_agent_id=_host_value(host, "parent_agent_id"),
            project_root=_host_value(host, "project_root"),
            repository=_host_value(host, "repository"),
            branch=_host_value(host, "branch"),
            commit_sha=_host_value(host, "commit_sha"),
        )

    def to_client_context(
        self, *, client_id: str = HERMES_CLIENT_ID
    ) -> ClientContext:
        return ClientContext(
            client_id=client_id,
            surface="hermes",
            agent_id=self.agent_id,
            session_id=self.session_id,
            parent_agent_id=self.parent_agent_id,
            project_root=self.project_root,
            repository=self.repository,
            branch=self.branch,
            commit_sha=self.commit_sha,
            access_scopes=self.access_scopes,
        )


@dataclass(frozen=True, slots=True)
class HermesAdapterConfig:
    socket_path: Path
    client_id: str = HERMES_CLIENT_ID
    connect_timeout: float = 2.0
    request_timeout: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "socket_path", Path(self.socket_path))
        if not self.socket_path.is_absolute():
            raise ValueError("socket_path must be absolute")
        if not self.client_id.strip():
            raise ValueError("client_id must not be empty")


class HermesProtocolAdapter:
    """Factory mapping Hermes session lifecycle to isolated memory sessions."""

    def __init__(
        self,
        config: HermesAdapterConfig,
        *,
        transport_factory: Any = None,
        degraded_provider: ReadOnlyDegradedProvider | None = None,
    ) -> None:
        self.config = config
        self._transport_factory = transport_factory or EnfoldClient
        self._degraded_provider = degraded_provider

    def open_session(self, lifecycle: HermesSessionContext) -> "HermesMemorySession":
        """Create a session whose provenance cannot change between requests."""

        context = lifecycle.to_client_context(client_id=self.config.client_id)
        client_config = ClientConfig(
            socket_path=self.config.socket_path,
            context=context,
            connect_timeout=self.config.connect_timeout,
            request_timeout=self.config.request_timeout,
        )
        transport = self._transport_factory(client_config)
        return HermesMemorySession(
            context,
            transport,
            degraded_provider=self._degraded_provider,
        )

    def open_host_session(
        self,
        host: object,
        *,
        access_scopes: tuple[str, ...],
    ) -> "HermesMemorySession":
        return self.open_session(
            HermesSessionContext.from_host(host, access_scopes=access_scopes)
        )


class HermesMemorySession:
    """Memory operations bound to one immutable Hermes session context."""

    def __init__(
        self,
        context: ClientContext,
        transport: MemoryTransport,
        *,
        degraded_provider: ReadOnlyDegradedProvider | None = None,
    ) -> None:
        self.context = context
        self._proxy = MemoryMCPProxy(transport)
        self._degraded_provider = degraded_provider

    def write(
        self,
        content: str,
        *,
        event_id: str,
        source_type: str,
        scope: str,
        **fields: Any,
    ) -> Any:
        """Write durably with a retry-stable key derived from the host event.

        Transport failures intentionally propagate.  Writes are never queued,
        cached, or redirected to a fallback store because doing so could lose
        provenance, bypass policy, or later duplicate the canonical write.
        """

        params = dict(fields)
        forbidden = {"idempotency_key", "content", "source_type", "scope"} & params.keys()
        if forbidden:
            raise ValueError(f"reserved write fields cannot be overridden: {sorted(forbidden)}")
        params.update(
            idempotency_key=self.idempotency_key(event_id),
            content=content,
            source_type=source_type,
            scope=scope,
        )
        return self._proxy.write(params)

    def search(self, query: str, **filters: Any) -> Any:
        return self._read("memory.search", {"query": query, **filters})

    def memory_context(
        self,
        query: str,
        *,
        token_budget: int,
        scope: str | None = None,
    ) -> Any:
        """Return compact, cited current memory suitable for turn prefetch."""

        params: dict[str, Any] = {"query": query, "token_budget": token_budget}
        if scope is not None:
            params["scope"] = scope
        return self._read("memory.context", params)

    def evidence(self, fact_id: int, *, limit: int | None = None) -> Any:
        params: dict[str, Any] = {"fact_id": fact_id}
        if limit is not None:
            params["limit"] = limit
        return self._read("memory.evidence", params)

    def history(self, **selector: Any) -> Any:
        return self._read("memory.history", selector)

    def conflicts(
        self, *, scope: str | None = None, unresolved_only: bool = True
    ) -> Any:
        params: dict[str, Any] = {"unresolved_only": unresolved_only}
        if scope is not None:
            params["scope"] = scope
        return self._read("memory.conflicts", params)

    def resolve_conflict(
        self, conflict_id: str, resolution_fact_id: int, *, reason: str
    ) -> Any:
        """Settle a visible conflict through the authoritative daemon."""

        return self._proxy.resolve_conflict(
            {
                "conflict_id": conflict_id,
                "resolution_fact_id": resolution_fact_id,
                "reason": reason,
            }
        )

    def enqueue_extraction(
        self,
        transcript: str,
        *,
        source: str,
        scope: str = "private",
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Queue an attributed transcript for the daemon-owned processor."""

        return self._proxy.enqueue_extraction(
            {
                "transcript": transcript,
                "source": source,
                "scope": scope,
                "metadata": dict(metadata or {}),
            }
        )

    def idempotency_key(self, event_id: str) -> str:
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("event_id must be a non-empty stable Hermes event id")
        material = "\0".join(
            (self.context.client_id, self.context.session_id, event_id.strip())
        ).encode("utf-8")
        return f"hermes-v1:{sha256(material).hexdigest()}"

    def _read(self, method: str, params: Mapping[str, Any]) -> Any:
        if method not in _READ_METHODS:
            raise ValueError(f"degraded reads do not support method: {method}")
        proxy_method = getattr(self._proxy, method.removeprefix("memory."))
        try:
            return proxy_method(params)
        except EnfoldTransportError:
            if self._degraded_provider is None:
                raise
            result = self._degraded_provider.read(
                method, dict(params), context=self.context
            )
            return DegradedReadResult(result)
