from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from enfold.client import (
    ClientConfig,
    EnfoldRemoteError,
    EnfoldTransportError,
)
from enfold.mcp_stdio import build_server, parse_config
from enfold.protocol import ClientContext, ProtocolError


class FakeToolError(Exception):
    pass


class FakeMCP:
    def __init__(self, name):
        self.name = name
        self.tools = {}
        self.runs = []

    def tool(self):
        def register(function):
            self.tools[function.__name__] = function
            return function

        return register

    def run(self, *, transport):
        self.runs.append(transport)


class RecordingTransport:
    instances = []

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.result = {"ok": True}
        self.error = None
        self.__class__.instances.append(self)

    def request(self, method, params=None, *, request_id=None):
        self.calls.append((method, params, request_id))
        if self.error:
            raise self.error
        return self.result


def _config(tmp_path: Path) -> ClientConfig:
    return ClientConfig(
        socket_path=tmp_path / "enfold.sock",
        context=ClientContext(
            client_id="client-a-install",
            surface="client-a",
            agent_id="client-a",
            session_id="thread-7",
            project_root="/workspace/project",
            access_scopes=("private", "work"),
        ),
    )


@pytest.fixture
def harness(tmp_path):
    RecordingTransport.instances.clear()
    server = build_server(
        _config(tmp_path),
        server_factory=FakeMCP,
        transport_factory=RecordingTransport,
        tool_error_type=FakeToolError,
    )
    return server, RecordingTransport.instances[0]


def test_registers_only_v1_memory_tools_without_opening_transport(harness):
    server, transport = harness

    assert server.name == "enfold-memory-v1"
    assert set(server.tools) == {
        "memory_write",
        "memory_search",
        "memory_context",
        "memory_evidence",
        "memory_history",
        "memory_changes",
        "memory_timeline",
        "memory_entities",
        "memory_entity",
        "memory_conflicts",
        "memory_resolve_conflict",
        "memory_extraction_enqueue",
    }
    assert transport.calls == []

    forbidden = {
        "client_id",
        "surface",
        "agent_id",
        "session_id",
        "parent_agent_id",
        "project_root",
        "repository",
        "branch",
        "commit_sha",
        "access_scopes",
        "performed_by",
    }
    for tool in server.tools.values():
        assert forbidden.isdisjoint(inspect.signature(tool).parameters)


def test_tools_are_one_to_one_json_safe_protocol_calls(harness):
    server, transport = harness
    transport.result = {"facts": [{"fact_id": 1}], "tuple": (1, 2)}

    result = server.tools["memory_search"]("needle", limit=4)

    assert result == {"facts": [{"fact_id": 1}], "tuple": [1, 2]}
    assert transport.calls == [
        (
            "memory.search",
            {"query": "needle", "category": None, "min_trust": 0.0, "limit": 4},
            None,
        )
    ]


def test_write_forwards_memory_fields_but_not_connection_identity(harness):
    server, transport = harness

    server.tools["memory_write"](
        "client-a:thread-7:1",
        "Enfold uses one daemon",
        "agent_report",
        asserted_by="Victor",
        state={"subject_key": "enfold", "predicate_key": "architecture"},
    )

    method, params, _ = transport.calls[0]
    assert method == "memory.write"
    assert params["idempotency_key"] == "client-a:thread-7:1"
    assert params["asserted_by"] == "Victor"
    assert params["state"]["subject_key"] == "enfold"
    assert "agent_id" not in params
    assert "session_id" not in params
    assert "performed_by" not in params


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        (
            "memory_context",
            ("current project",),
            ("memory.context", {"query": "current project", "token_budget": 256}),
        ),
        ("memory_evidence", (9,), ("memory.evidence", {"fact_id": 9, "limit": 100})),
        ("memory_history", (), ("memory.history", {"limit": 100})),
        (
            "memory_conflicts",
            (),
            ("memory.conflicts", {"unresolved_only": True}),
        ),
        (
            "memory_changes",
            ("2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"),
            (
                "memory.changes",
                {
                    "since": "2026-07-01T00:00:00Z",
                    "until": "2026-07-02T00:00:00Z",
                    "limit": 100,
                },
            ),
        ),
        (
            "memory_timeline",
            ("Victor",),
            ("memory.timeline", {"subject_or_query": "Victor", "limit": 100}),
        ),
        (
            "memory_entities",
            (),
            ("memory.entities", {"min_facts": 1, "limit": 100}),
        ),
        (
            "memory_entity",
            ("Victor",),
            ("memory.entity", {"name": "Victor", "limit": 100}),
        ),
    ],
)
def test_read_tools_forward_to_proxy(harness, name, args, expected):
    server, transport = harness
    server.tools[name](*args)
    assert transport.calls[-1][:2] == expected


def test_resolve_conflict_forwards_audited_decision(harness):
    server, transport = harness
    server.tools["memory_resolve_conflict"]("conflict-1", 9, "Victor confirmed it")
    assert transport.calls[-1][:2] == (
        "memory.resolve_conflict",
        {
            "conflict_id": "conflict-1",
            "resolution_fact_id": 9,
            "reason": "Victor confirmed it",
        },
    )


def test_extraction_enqueue_is_explicit_scoped_model_free_protocol_call(harness):
    server, transport = harness
    server.tools["memory_extraction_enqueue"](
        "USER: Victor prefers concise responses.",
        "session_end",
        metadata={"hook": "session_end"},
    )
    assert transport.calls[-1][:2] == (
        "memory.extraction.enqueue",
        {
            "transcript": "USER: Victor prefers concise responses.",
            "source": "session_end",
            "scope": "private",
            "metadata": {"hook": "session_end"},
        },
    )


def test_remote_error_is_typed_json_mcp_error(harness):
    server, transport = harness
    transport.error = EnfoldRemoteError(
        ProtocolError(
            "needs_review",
            "human confirmation required",
            details={"fact_id": 4},
        ),
        request_id="req-4",
    )

    with pytest.raises(FakeToolError) as raised:
        server.tools["memory_write"]("key", "claim", "agent_report")

    payload = json.loads(str(raised.value))
    assert payload == {
        "code": "needs_review",
        "details": {"fact_id": 4},
        "message": "human confirmation required",
        "request_id": "req-4",
        "retryable": False,
    }


def test_transport_error_is_retryable_and_non_json_result_is_typed(harness):
    server, transport = harness
    transport.error = EnfoldTransportError("socket unavailable")
    with pytest.raises(FakeToolError) as raised:
        server.tools["memory_search"]("query")
    assert json.loads(str(raised.value))["code"] == "daemon_unavailable"
    assert json.loads(str(raised.value))["retryable"] is True

    transport.error = None
    transport.result = {"bad": object()}
    with pytest.raises(FakeToolError) as raised:
        server.tools["memory_search"]("query")
    assert json.loads(str(raised.value))["code"] == "invalid_daemon_result"


def test_parse_config_uses_explicit_environment_identity(tmp_path):
    config = parse_config(
        [],
        environ={
            "ENFOLD_SOCKET_PATH": str(tmp_path / "enfold.sock"),
            "ENFOLD_CLIENT_ID": "client-b-install",
            "ENFOLD_SURFACE": "client-b",
            "ENFOLD_AGENT_ID": "client-b",
            "ENFOLD_SESSION_ID": "session-9",
            "ENFOLD_ACCESS_SCOPES": "private,work",
        },
    )

    assert config.socket_path == tmp_path / "enfold.sock"
    assert config.context.surface == "client-b"
    assert config.context.access_scopes == ("private", "work")
    assert "health" not in config.capabilities
    assert "memory.extraction.enqueue" in config.capabilities


def test_parse_config_cli_overrides_environment_and_requires_absolute_socket(tmp_path):
    config = parse_config(
        [
            "--socket-path", str(tmp_path / "cli.sock"),
            "--client-id", "cli-install",
            "--surface", "client-a",
            "--agent-id", "client-a",
            "--session-id", "cli-session",
            "--access-scope", "project",
        ],
        environ={"ENFOLD_ACCESS_SCOPES": "private"},
    )
    assert config.context.access_scopes == ("project",)

    with pytest.raises(SystemExit):
        parse_config(
            [
                "--socket-path", "relative.sock",
                "--client-id", "client",
                "--surface", "client-a",
                "--agent-id", "client-a",
                "--session-id", "session",
            ],
            environ={},
        )
