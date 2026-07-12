from __future__ import annotations

import json

import pytest

from enfold.ollama_artifact import (
    ArtifactAttestationError,
    LocalOllamaArtifactAttestor,
    require_sha256_digest,
)


_DIGEST = "sha256:" + "b" * 64


class _Response:
    def __init__(self, payload: object):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _opener(payload: object, calls: list[tuple[str, float]]):
    def open_request(request, *, timeout):
        calls.append((request.full_url, timeout))
        return _Response(payload)

    return open_request


def test_attests_exact_local_artifact_without_exposing_provider_payload():
    calls: list[tuple[str, float]] = []
    attestor = LocalOllamaArtifactAttestor(
        base_url="http://fixture-ollama:11434/",
        timeout=3,
        opener=_opener(
            {
                "models": [
                    {"name": "embeddinggemma:latest", "digest": _DIGEST[7:]}
                ]
            },
            calls,
        ),
    )

    result = attestor.attest(model="embeddinggemma", expected_digest=_DIGEST)

    assert result.safe_state() == {"provider": "ollama", "status": "verified"}
    assert calls == [("http://fixture-ollama:11434/api/tags", 3.0)]


@pytest.mark.parametrize(
    ("configured", "message"),
    [
        ("b" * 64, "sha256"),
        ("sha256:" + "B" * 64, "lowercase"),
        ("sha256:" + "b" * 63, "64 lowercase"),
        (None, "sha256"),
    ],
)
def test_configured_artifact_digest_must_be_canonical_sha256(configured, message):
    with pytest.raises(ArtifactAttestationError, match=message):
        require_sha256_digest(configured)


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ({"models": []}, "not present"),
        ({"models": [{"name": "embeddinggemma", "digest": "not-a-digest"}]}, "invalid immutable"),
        ({"models": [{"name": "embeddinggemma"}]}, "no immutable"),
        ({"models": "not-a-list"}, "response is invalid"),
    ],
)
def test_attestation_rejects_missing_or_malformed_provider_artifact(
    payload, expected_message
):
    attestor = LocalOllamaArtifactAttestor(opener=_opener(payload, []))
    with pytest.raises(ArtifactAttestationError, match=expected_message):
        attestor.attest(model="embeddinggemma", expected_digest=_DIGEST)


def test_attestation_rejects_digest_mismatch_and_transport_failure():
    mismatch = LocalOllamaArtifactAttestor(
        opener=_opener(
            {"models": [{"name": "embeddinggemma", "digest": "c" * 64}]}, []
        )
    )
    with pytest.raises(ArtifactAttestationError, match="does not match"):
        mismatch.attest(model="embeddinggemma", expected_digest=_DIGEST)

    def unavailable(_request, *, timeout):
        del timeout
        raise OSError("fixture unavailable")

    unavailable_attestor = LocalOllamaArtifactAttestor(opener=unavailable)
    with pytest.raises(ArtifactAttestationError, match="attestation is unavailable"):
        unavailable_attestor.attest(model="embeddinggemma", expected_digest=_DIGEST)
