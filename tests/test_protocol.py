from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from enfold.protocol import (
    CAPABILITY_CHANGES,
    CAPABILITY_CONFLICTS,
    CAPABILITY_CONTEXT,
    CAPABILITY_RESOLVE_CONFLICT,
    CAPABILITY_ENQUEUE_EXTRACTION,
    CAPABILITY_EVIDENCE,
    CAPABILITY_HEALTH,
    CAPABILITY_HISTORY,
    CAPABILITY_ENTITIES,
    CAPABILITY_ENTITY,
    CAPABILITY_SEARCH,
    CAPABILITY_TIMELINE,
    CAPABILITY_WRITE,
    MAX_FRAME_SIZE,
    ClientContext,
    FrameTooLargeError,
    Handshake,
    HandshakeResponse,
    ProtocolError,
    ProtocolValidationError,
    ProtocolVersion,
    ProtocolVersionMismatch,
    Request,
    Response,
    SUPPORTED_CAPABILITIES,
    decode_frame,
    encode_frame,
    negotiate_handshake,
    required_capability,
)


def context() -> ClientContext:
    return ClientContext(
        client_id="client-a-install-1",
        surface="client-a",
        agent_id="client-a",
        session_id="thread-123",
        project_root="/workspace/enfold",
        repository="enfold",
        branch="main",
        commit_sha="abc123",
        access_scopes=("private", "work"),
    )


def test_protocol_exposes_required_methods_and_capabilities():
    assert required_capability("health") == CAPABILITY_HEALTH
    assert required_capability("memory.write") == CAPABILITY_WRITE
    assert required_capability("memory.search") == CAPABILITY_SEARCH
    assert required_capability("memory.context") == CAPABILITY_CONTEXT
    assert required_capability("memory.evidence") == CAPABILITY_EVIDENCE
    assert required_capability("memory.history") == CAPABILITY_HISTORY
    assert required_capability("memory.changes") == CAPABILITY_CHANGES
    assert required_capability("memory.timeline") == CAPABILITY_TIMELINE
    assert required_capability("memory.entities") == CAPABILITY_ENTITIES
    assert required_capability("memory.entity") == CAPABILITY_ENTITY
    assert required_capability("memory.conflicts") == CAPABILITY_CONFLICTS
    assert required_capability("memory.resolve_conflict") == CAPABILITY_RESOLVE_CONFLICT
    assert set(SUPPORTED_CAPABILITIES) == {
        CAPABILITY_HEALTH,
        CAPABILITY_WRITE,
        CAPABILITY_SEARCH,
        CAPABILITY_CONTEXT,
        CAPABILITY_EVIDENCE,
        CAPABILITY_HISTORY,
        CAPABILITY_CHANGES,
        CAPABILITY_TIMELINE,
        CAPABILITY_ENTITIES,
        CAPABILITY_ENTITY,
        CAPABILITY_CONFLICTS,
        CAPABILITY_RESOLVE_CONFLICT,
        CAPABILITY_ENQUEUE_EXTRACTION,
    }
    with pytest.raises(ProtocolValidationError, match="unsupported method"):
        required_capability("memory.delete_everything")


def test_handshake_round_trip_preserves_immutable_context():
    original = Handshake(context(), capabilities=(CAPABILITY_HEALTH, CAPABILITY_SEARCH))

    decoded = decode_frame(encode_frame(original))

    assert decoded == original
    assert isinstance(decoded, Handshake)
    with pytest.raises(FrozenInstanceError):
        decoded.context.agent_id = "spoofed"


def test_request_and_response_round_trip_as_canonical_json_lines():
    request = Request("req-1", "memory.search", {"limit": 5, "query": "current project"})
    encoded = encode_frame(request)

    assert encoded.endswith(b"\n")
    assert encoded.count(b"\n") == 1
    assert encoded == encode_frame(decode_frame(encoded))
    assert b'"params":{"limit":5,"query":"current project"}' in encoded

    response = Response("req-1", True, result={"facts": [3, 7]})
    assert decode_frame(encode_frame(response)) == response


def test_requests_cover_every_supported_method():
    for index, method in enumerate(
        ("health", "memory.write", "memory.search", "memory.context", "memory.evidence", "memory.history", "memory.changes", "memory.timeline", "memory.entities", "memory.entity", "memory.conflicts", "memory.resolve_conflict", "memory.extraction.enqueue")
    ):
        request = Request(f"req-{index}", method, {})
        assert decode_frame(encode_frame(request)) == request


def test_request_cannot_override_handshake_identity_or_scope():
    for field in ("client_id", "agent_id", "session_id", "access_scopes"):
        with pytest.raises(ProtocolValidationError, match="immutable connection context"):
            Request("req-spoof", "memory.write", {field: "forged"})


def test_typed_error_is_required_for_failed_response():
    error = ProtocolError(
        "conflict",
        "state slot has unresolved values",
        retryable=False,
        details={"conflict_id": "conflict-1", "fact_ids": [3, 4]},
    )
    response = Response("req-2", False, error=error)

    assert decode_frame(encode_frame(response)) == response
    with pytest.raises(ProtocolValidationError, match="typed error"):
        Response("req-2", False)
    with pytest.raises(ProtocolValidationError, match="cannot include an error"):
        Response("req-2", True, result={}, error=error)
    with pytest.raises(ProtocolValidationError, match="cannot include a result"):
        Response("req-2", False, result={}, error=error)


def test_handshake_negotiates_intersection_in_server_order():
    hello = Handshake(
        context(),
        capabilities=(CAPABILITY_SEARCH, CAPABILITY_HEALTH, "future.optional"),
    )

    response = negotiate_handshake(
        hello,
        service_version="1.0.0",
        server_capabilities=(CAPABILITY_HEALTH, CAPABILITY_WRITE, CAPABILITY_SEARCH),
    )

    assert response == HandshakeResponse(
        accepted=True,
        capabilities=(CAPABILITY_HEALTH, CAPABILITY_SEARCH),
        service_version="1.0.0",
    )


def test_major_version_mismatch_is_explicitly_refused():
    hello = Handshake(context(), protocol=ProtocolVersion(2, 0))

    refused = negotiate_handshake(hello, service_version="1.0.0")

    assert refused.accepted is False
    assert refused.error is not None
    assert refused.error.code == "incompatible_protocol_major"
    assert refused.error.retryable is False
    assert refused.capabilities == ()

    encoded = encode_frame(hello)
    with pytest.raises(ProtocolVersionMismatch, match="major 2"):
        decode_frame(encoded)
    assert decode_frame(encoded, refuse_major_mismatch=False) == hello


def test_decode_rejects_unknown_missing_and_extra_fields():
    valid = json.loads(encode_frame(Request("req-1", "health")).decode())

    unknown_type = dict(valid, type="notification")
    with pytest.raises(ProtocolValidationError, match="unknown frame type"):
        decode_frame(json.dumps(unknown_type))

    missing = dict(valid)
    del missing["method"]
    with pytest.raises(ProtocolValidationError, match="missing frame fields"):
        decode_frame(json.dumps(missing))

    extra = dict(valid, recorded_by="client-a")
    with pytest.raises(ProtocolValidationError, match="unknown frame fields"):
        decode_frame(json.dumps(extra))


def test_decode_rejects_multiple_frames_invalid_utf8_and_non_finite_numbers():
    frame = encode_frame(Request("req-1", "health"))
    with pytest.raises(ProtocolValidationError, match="exactly one"):
        decode_frame(frame + frame)
    with pytest.raises(ProtocolValidationError, match="UTF-8"):
        decode_frame(b"\xff\n")
    with pytest.raises(ProtocolValidationError, match="non-finite"):
        decode_frame(
            b'{"method":"health","params":{"x":NaN},"protocol":{"major":1,"minor":0},'
            b'"request_id":"r","schema_version":1,"type":"request"}\n'
        )
    with pytest.raises(ProtocolValidationError, match="duplicate JSON object key"):
        decode_frame(
            b'{"method":"health","method":"memory.search","params":{},'
            b'"protocol":{"major":1,"minor":0},"request_id":"r",'
            b'"schema_version":1,"type":"request"}\n'
        )
    with pytest.raises(ProtocolValidationError, match="valid Unicode"):
        decode_frame("\ud800")


def test_encode_and_decode_enforce_byte_limit():
    request = Request("req-large", "memory.search", {"query": "x" * 200})
    encoded = encode_frame(request)

    with pytest.raises(FrameTooLargeError):
        encode_frame(request, max_frame_size=len(encoded) - 1)
    with pytest.raises(FrameTooLargeError):
        decode_frame(encoded, max_frame_size=len(encoded) - 1)
    with pytest.raises(FrameTooLargeError):
        decode_frame(b"x" * (MAX_FRAME_SIZE + 1))


def test_validation_rejects_non_json_params_bad_tokens_and_invalid_context():
    with pytest.raises(ProtocolValidationError, match="non-JSON"):
        Request("req-1", "memory.search", {"bad": object()})
    with pytest.raises(ProtocolValidationError, match="valid protocol token"):
        Request("has spaces", "health")
    with pytest.raises(ProtocolValidationError, match="access_scopes must not be empty"):
        ClientContext("client", "client-a", "agent", "session", access_scopes=())
    with pytest.raises(ProtocolValidationError, match="schema_version"):
        Request("req-1", "health", schema_version=-1)


def test_frame_size_is_measured_as_utf8_bytes_not_characters():
    request = Request("req-unicode", "memory.search", {"query": "🧠" * 8})
    encoded = encode_frame(request)

    assert len(encoded) > len(encoded.decode("utf-8"))
    with pytest.raises(FrameTooLargeError):
        encode_frame(request, max_frame_size=len(encoded.decode("utf-8")))
