from __future__ import annotations

import pytest

from enfold.policy import (
    MemoryPolicy,
    PolicyDecision,
    UnknownMemoryClient,
    validate_relation,
    validate_scope,
    validate_sensitivity,
)
from enfold.provenance import ConnectionContext, WriteRequest


def _context(**changes):
    values = {
        "client_id": "client-a-install",
        "surface": "client-a",
        "agent_id": "client-a",
        "session_id": "thread-1",
        "access_scopes": ("private",),
    }
    values.update(changes)
    return ConnectionContext(**values)


def _request(**changes):
    values = {
        "idempotency_key": "write-1",
        "content": "A safe durable fact",
        "source_type": "agent_report",
    }
    values.update(changes)
    return WriteRequest(**values)


def test_requested_scopes_are_intersected_with_server_grants():
    policy = MemoryPolicy(
        {"client-a-install": ("private", "work", "project:enfold")}
    )

    effective = policy.authorize_context(
        _context(access_scopes=("public", "project:enfold", "private"))
    )

    assert effective.access_scopes == ("private", "project:enfold")


def test_unknown_client_and_empty_grant_intersection_fail_closed():
    policy = MemoryPolicy({"client-a-install": ("private",)})

    with pytest.raises(UnknownMemoryClient, match="no server-side scope grant"):
        policy.authorize_context(
            _context(client_id="unknown", access_scopes=("private",))
        )
    with pytest.raises(PermissionError, match="do not intersect"):
        policy.authorize_context(_context(access_scopes=("work",)))


@pytest.mark.parametrize("scope", ["private ", "team", "project:", "project:a/b"])
def test_scope_vocabulary_is_strict(scope):
    with pytest.raises(ValueError, match="unsupported memory scope"):
        validate_scope(scope)


def test_sensitivity_and_relation_vocabularies_are_strict():
    with pytest.raises(ValueError, match="sensitivity"):
        validate_sensitivity("confidential")
    with pytest.raises(ValueError, match="relation"):
        validate_relation("mentions")


def test_secret_classification_and_common_credential_shapes_are_rejected():
    policy = MemoryPolicy({"client-a-install": ("private", "secret")})

    assert policy.evaluate_write(_request(sensitivity="secret")) == PolicyDecision(
        "rejected", "secret durable writes are disabled"
    )
    decision = policy.evaluate_write(
        _request(content="Authorization: Bearer abcdefghijklmnopqrstuvwxyz")
    )
    assert decision == PolicyDecision("rejected", "credential-shaped content")


def test_credential_screen_covers_metadata_tags_and_source_uri():
    policy = MemoryPolicy({"client-a-install": ("private",)})

    for request in (
        _request(metadata_json='{"api_key":"abcdefgh12345678"}'),
        _request(tags="token: abcdefgh12345678"),
        _request(source_uri="https://example.test/?token=abcdefgh12345678"),
    ):
        assert policy.evaluate_write(request) == PolicyDecision(
            "rejected", "credential-shaped content"
        )


def test_custom_credential_screen_can_request_human_review():
    def internal_identifier_screen(request):
        if "PAYROLL-ID" in request.content:
            return PolicyDecision("needs_review", "possible internal identifier")
        return None

    policy = MemoryPolicy(
        {"client-a-install": ("private",)},
        credential_screens=(internal_identifier_screen,),
    )

    assert policy.evaluate_write(_request(content="PAYROLL-ID 123")) == PolicyDecision(
        "needs_review", "possible internal identifier"
    )


def test_correction_claims_require_explicit_server_authority():
    request = _request(correction_status="human_corrected")
    denied = MemoryPolicy({"client-a-install": ("private",)})
    allowed = MemoryPolicy(
        {"client-a-install": ("private",)},
        correction_authorities=("client-a-install",),
    )

    assert denied.evaluate_write(request, client_id="client-a-install") == PolicyDecision(
        "needs_review", "client is not authorized to assert human correction"
    )
    assert allowed.evaluate_write(request, client_id="client-a-install") is None


@pytest.mark.parametrize(
    "value",
    [
        "postgresql://user:password123@example.test/db",
        # Joined at runtime so secret scanners never see a contiguous token.
        "-".join(("xoxb", "1234567890", "abcdefghijklmnop")),
        "eyJabcdefghij.abcdefghijkl.abcdefghijkl",
        "AccountKey=abcdefghijklmnop1234",
    ],
)
def test_expanded_credential_shapes_are_rejected(value):
    policy = MemoryPolicy({"client-a-install": ("private",)})
    assert policy.evaluate_write(_request(asserted_by=value)) == PolicyDecision(
        "rejected", "credential-shaped content"
    )
