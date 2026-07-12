"""Memory Arena provider that exercises the versioned daemon client path."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Mapping, Protocol


class MemoryProtocolClient(Protocol):
    def request(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class DaemonProtocolArenaProvider:
    """Translate Arena search calls into ``memory.search`` protocol requests."""

    client: MemoryProtocolClient
    provider_id: str = "enfold-daemon-protocol-v1"
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def metadata(self) -> dict[str, Any]:
        ordered = sorted(self.latencies_ms)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0.0
        return {
            "provider_id": self.provider_id,
            "transport_path": "versioned-daemon-protocol",
            "requests": len(ordered),
            "latency_ms_max": max(ordered, default=0.0),
            "latency_ms_p95": p95,
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
        del bump
        params: dict[str, Any] = {
            "query": query,
            "min_trust": min_trust,
            "limit": limit,
        }
        if category is not None:
            params["category"] = category
        started = time.perf_counter()
        response = self.client.request("memory.search", params)
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        facts = response.get("facts")
        if not isinstance(facts, list) or not all(isinstance(row, dict) for row in facts):
            raise ValueError("daemon memory.search returned an invalid facts payload")
        return facts

    def search_conflicts(
        self,
        query: str,
        *,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        bump: bool = False,
    ) -> list[dict[str, Any]]:
        """Read the explicit conflict view; never mix it into settled truth."""

        del bump
        started = time.perf_counter()
        response = self.client.request("memory.conflicts", {"unresolved_only": True})
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        conflicts = response.get("conflicts")
        if not isinstance(conflicts, list):
            raise ValueError("daemon memory.conflicts returned an invalid payload")
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for conflict in conflicts:
            if not isinstance(conflict, dict) or not isinstance(conflict.get("members"), list):
                raise ValueError("daemon memory.conflicts returned invalid members")
            for member in conflict["members"]:
                if not isinstance(member, dict):
                    raise ValueError("daemon memory.conflicts returned an invalid member")
                if category is not None and member.get("category") != category:
                    continue
                if float(member.get("trust_score") or 0.0) < min_trust:
                    continue
                content_tokens = set(re.findall(r"[a-z0-9]+", str(member.get("content", "")).lower()))
                overlap = len(query_tokens & content_tokens)
                if not overlap:
                    continue
                row = dict(member)
                row.update({
                    "score": overlap / max(1, len(query_tokens)),
                    "conflict_id": conflict.get("conflict_id"),
                    "is_conflict": True,
                })
                ranked.append((float(row["score"]), int(row["fact_id"]), row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [row for _, _, row in ranked[:limit]]
