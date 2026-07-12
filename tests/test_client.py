from __future__ import annotations

import socket

import pytest

from enfold.client import (
    ClientConfig,
    EnfoldClient,
    EnfoldHandshakeError,
    EnfoldProtocolError,
    EnfoldRemoteError,
    EnfoldTransportError,
)
from enfold.daemon import DaemonConfig, RequestError, UnixJsonDaemon
from enfold.protocol import (
    CAPABILITY_HEALTH,
    CAPABILITY_SEARCH,
    CAPABILITY_WRITE,
    ClientContext,
)


def _context() -> ClientContext:
    return ClientContext(
        client_id="client-a-install-1",
        surface="client-a",
        agent_id="client-a",
        session_id="thread-123",
        project_root="/workspace/enfold",
        access_scopes=("private", "work"),
    )


def _config(path, **changes) -> ClientConfig:
    values = {
        "socket_path": path,
        "context": _context(),
        "connect_timeout": 0.5,
        "request_timeout": 0.5,
    }
    values.update(changes)
    return ClientConfig(**values)


def _daemon(path, handler, **changes) -> UnixJsonDaemon:
    values = {
        "socket_path": path,
        "server_version": "1.0-test",
        "client_timeout": 1.0,
        "shutdown_timeout": 1.0,
    }
    values.update(changes)
    return UnixJsonDaemon(DaemonConfig(**values), handler)


def test_client_negotiates_context_and_returns_typed_result(tmp_path):
    path = tmp_path / "enfold.sock"
    seen = []

    def handler(context, request):
        seen.append((context, request))
        return {"query": request.params["query"], "writer": context.agent_id}

    daemon = _daemon(path, handler)
    daemon.start()
    try:
        result = EnfoldClient(_config(path)).request(
            "memory.search", {"query": "deployment"}, request_id="req-fixed"
        )
    finally:
        daemon.shutdown()

    assert result == {"query": "deployment", "writer": "client-a"}
    assert seen[0][0] == _context()
    assert seen[0][1].request_id == "req-fixed"


def test_client_reconnects_and_handshakes_for_every_request(tmp_path):
    path = tmp_path / "enfold.sock"
    contexts = []
    daemon = _daemon(
        path,
        lambda context, request: contexts.append(context) or request.params,
        client_timeout=0.05,
    )
    daemon.start()
    try:
        client = EnfoldClient(_config(path))
        assert client.request("memory.write", {"n": 1}) == {"n": 1}
        # A second operation succeeds without relying on an idle pooled socket.
        assert client.request("memory.write", {"n": 2}) == {"n": 2}
    finally:
        daemon.shutdown()
    assert contexts == [_context(), _context()]


def test_client_exposes_remote_error_without_losing_type_information(tmp_path):
    path = tmp_path / "enfold.sock"

    def handler(context, request):
        raise RequestError("needs_review", "human confirmation required")

    daemon = _daemon(path, handler)
    daemon.start()
    try:
        with pytest.raises(EnfoldRemoteError) as raised:
            EnfoldClient(_config(path)).request(
                "memory.write", {"content": "uncertain"}, request_id="req-review"
            )
    finally:
        daemon.shutdown()

    assert raised.value.code == "needs_review"
    assert raised.value.message == "human confirmation required"
    assert raised.value.retryable is False
    assert raised.value.details == {}
    assert raised.value.request_id == "req-review"


def test_client_rejects_missing_local_or_negotiated_capability(tmp_path):
    path = tmp_path / "enfold.sock"
    local = EnfoldClient(
        _config(path, capabilities=(CAPABILITY_HEALTH, CAPABILITY_SEARCH))
    )
    with pytest.raises(EnfoldProtocolError, match="did not request capability"):
        local.request("memory.write", {})

    daemon = _daemon(
        path,
        lambda context, request: {},
        server_capabilities=(CAPABILITY_HEALTH, CAPABILITY_SEARCH),
    )
    daemon.start()
    try:
        client = EnfoldClient(
            _config(path, capabilities=(CAPABILITY_HEALTH, CAPABILITY_WRITE))
        )
        with pytest.raises(EnfoldProtocolError, match="did not negotiate"):
            client.request("memory.write", {})
    finally:
        daemon.shutdown()


def test_client_surfaces_handshake_refusal_and_transport_failure(tmp_path):
    path = tmp_path / "enfold.sock"
    daemon = _daemon(path, lambda context, request: {}, schema_version=2)
    daemon.start()
    try:
        with pytest.raises(EnfoldHandshakeError) as raised:
            EnfoldClient(_config(path)).request("health")
        assert raised.value.code == "incompatible_schema"
    finally:
        daemon.shutdown()

    with pytest.raises(EnfoldTransportError):
        EnfoldClient(_config(path)).request("health")


def test_client_rejects_oversized_response(tmp_path):
    path = tmp_path / "enfold.sock"
    # The daemon turns its oversized result into a small typed remote error.
    daemon = _daemon(
        path,
        lambda context, request: {"content": "x" * 1000},
        max_frame_bytes=512,
    )
    daemon.start()
    try:
        client = EnfoldClient(_config(path, max_frame_bytes=512))
        with pytest.raises(EnfoldRemoteError) as raised:
            client.request("memory.search", {})
        assert raised.value.code == "response_too_large"
    finally:
        daemon.shutdown()


def test_client_config_requires_absolute_socket_path():
    with pytest.raises(ValueError, match="absolute"):
        ClientConfig(socket_path="relative.sock", context=_context())


def test_client_timeout_is_typed_transport_error(tmp_path):
    path = tmp_path / "hung.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    try:
        client = EnfoldClient(
            _config(path, connect_timeout=0.05, request_timeout=0.05)
        )
        with pytest.raises(EnfoldTransportError):
            client.request("health")
    finally:
        listener.close()
