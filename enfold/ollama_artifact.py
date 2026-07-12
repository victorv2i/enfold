"""Fail-closed immutable-artifact attestation for local Ollama models.

The embedding endpoint accepts a mutable model tag. Before Enfold uses that tag
for stored retrieval, this module resolves the local Ollama model registry and
requires its recorded manifest digest to match the configured SHA-256 identity
exactly. Provider payloads and observed digests deliberately never leave this
boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import re
from typing import Any, Protocol
import urllib.request


_CONFIG_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROVIDER_DIGEST = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class ArtifactAttestationError(RuntimeError):
    """The configured immutable model artifact could not be verified."""


def require_sha256_digest(value: object, *, name: str = "artifact digest") -> str:
    """Return a canonical immutable SHA-256 digest or fail closed."""

    if not isinstance(value, str) or not _CONFIG_DIGEST.fullmatch(value):
        raise ArtifactAttestationError(
            f"{name} must be sha256:<64 lowercase hexadecimal characters>"
        )
    return value


@dataclass(frozen=True, slots=True)
class ArtifactAttestation:
    """A successful attestation, deliberately free of provider payload data."""

    provider: str = "ollama"

    def safe_state(self) -> dict[str, str]:
        return {"provider": self.provider, "status": "verified"}


class ArtifactAttestor(Protocol):
    """Injectable boundary for resolving a configured model to its artifact."""

    def attest(self, *, model: str, expected_digest: str) -> ArtifactAttestation:
        """Return only a successful safe attestation or raise on failure."""


def _model_aliases(model: str) -> frozenset[str]:
    """Match Ollama's implicit ``:latest`` tag without broadening identity."""

    if not isinstance(model, str) or not model.strip():
        raise ArtifactAttestationError("configured Ollama model must be non-empty")
    cleaned = model.strip()
    aliases = {cleaned}
    final_segment = cleaned.rsplit("/", 1)[-1]
    if ":" not in final_segment:
        aliases.add(f"{cleaned}:latest")
    return frozenset(aliases)


class LocalOllamaArtifactAttestor:
    """Resolve an Ollama tag through the local registry's ``/api/tags`` API."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = 5.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Ollama base_url must be non-empty")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("Ollama artifact attestation timeout must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._opener = opener

    def attest(self, *, model: str, expected_digest: str) -> ArtifactAttestation:
        expected = require_sha256_digest(expected_digest)
        aliases = _model_aliases(model)
        payload = self._load_registry()
        models = payload.get("models") if isinstance(payload, Mapping) else None
        if not isinstance(models, list):
            raise ArtifactAttestationError("local Ollama registry response is invalid")

        matches: list[str] = []
        for record in models:
            if not isinstance(record, Mapping):
                continue
            names = {
                item.strip()
                for item in (record.get("name"), record.get("model"))
                if isinstance(item, str) and item.strip()
            }
            if not names.intersection(aliases):
                continue
            digest = record.get("digest")
            if not isinstance(digest, str):
                raise ArtifactAttestationError(
                    "matched local Ollama model has no immutable digest"
                )
            normalized = _PROVIDER_DIGEST.fullmatch(digest)
            if normalized is None:
                raise ArtifactAttestationError(
                    "matched local Ollama model has an invalid immutable digest"
                )
            matches.append(f"sha256:{normalized.group(1)}")

        if not matches:
            raise ArtifactAttestationError("configured Ollama model is not present locally")
        if len(set(matches)) != 1:
            raise ArtifactAttestationError(
                "configured Ollama model resolves to multiple artifacts"
            )
        if matches[0] != expected:
            raise ArtifactAttestationError("configured Ollama artifact digest does not match")
        return ArtifactAttestation()

    def _load_registry(self) -> Mapping[str, Any]:
        request = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
        try:
            with self._opener(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ArtifactAttestationError(
                "local Ollama artifact attestation is unavailable"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ArtifactAttestationError("local Ollama registry response is invalid")
        return payload
