"""Thin, transport-agnostic memory tool proxy.

This is an adapter foundation, not an MCP server registration.  It imports no
Hermes modules and can be given any transport implementing ``request``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class MemoryTransport(Protocol):
    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> Any: ...


class MemoryMCPProxy:
    """One-to-one memory methods suitable for a future MCP tool wrapper."""

    def __init__(self, transport: MemoryTransport):
        self._transport = transport

    def write(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.write", params)

    def search(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.search", params)

    def context(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.context", params)

    def evidence(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.evidence", params)

    def history(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.history", params)

    def changes(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.changes", params)

    def timeline(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.timeline", params)

    def entities(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.entities", params)

    def entity(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.entity", params)

    def conflicts(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.conflicts", params)

    def resolve_conflict(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.resolve_conflict", params)

    def enqueue_extraction(self, params: Mapping[str, Any]) -> Any:
        return self._call("memory.extraction.enqueue", params)

    def _call(self, method: str, params: Mapping[str, Any]) -> Any:
        if not isinstance(params, Mapping):
            raise TypeError("tool params must be a mapping")
        return self._transport.request(method, dict(params))
