from __future__ import annotations

import json

import pytest

from enfold.client import EnfoldRemoteError, EnfoldTransportError
from enfold.protocol import ProtocolError
EnfoldV1MemoryProvider = pytest.importorskip(
    "integrations.hermes_enfold_v1",
    reason="Hermes bridge dependencies are not installed in the standalone test environment",
).EnfoldV1MemoryProvider


class FakeMemorySession:
    def __init__(self, context, *, offline=False, error=None):
        self.context = context
        self.offline = offline
        self.error = error
        self.calls = []

    def _call(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        if self.offline:
            raise EnfoldTransportError("test daemon offline")
        if self.error is not None:
            raise self.error
        if method == "search":
            return {"facts": [{"fact_id": 7, "content": "Victor uses Enfold"}]}
        return {"method": method, "args": args, "kwargs": kwargs}

    def search(self, *args, **kwargs):
        return self._call("search", *args, **kwargs)

    def write(self, *args, **kwargs):
        return self._call("write", *args, **kwargs)

    def evidence(self, *args, **kwargs):
        return self._call("evidence", *args, **kwargs)

    def history(self, *args, **kwargs):
        return self._call("history", *args, **kwargs)

    def conflicts(self, *args, **kwargs):
        return self._call("conflicts", *args, **kwargs)

    def enqueue_extraction(self, *args, **kwargs):
        return self._call("enqueue_extraction", *args, **kwargs)


class FakeAdapter:
    def __init__(self, config, *, offline=False, error=None):
        self.config = config
        self.offline = offline
        self.error = error
        self.sessions = []

    def open_session(self, context):
        session = FakeMemorySession(context, offline=self.offline, error=self.error)
        self.sessions.append(session)
        return session


def provider(tmp_path, *, offline=False, error=None):
    adapters = []

    def factory(config):
        adapter = FakeAdapter(config, offline=offline, error=error)
        adapters.append(adapter)
        return adapter

    instance = EnfoldV1MemoryProvider(
        adapter_factory=factory,
        environ={
            "ENFOLD_SOCKET_PATH": str(tmp_path / "enfold.sock"),
            "ENFOLD_HERMES_CLIENT_ID": "hermes-test-install",
            "ENFOLD_HERMES_SCOPES": "private,project:enfold",
        },
    )
    return instance, adapters


def test_lifecycle_prefetch_and_session_switch_preserve_host_provenance(tmp_path):
    memory, adapters = provider(tmp_path)
    memory.initialize(
        "session-main",
        agent_identity="wonny",
        agent_context="primary",
        agent_workspace="/work/enfold",
        repository="victor/enfold",
        branch="v1",
        commit_sha="abc123",
    )
    first = adapters[0].sessions[0]
    assert first.context.agent_id == "wonny"
    assert first.context.session_id == "session-main"
    assert first.context.access_scopes == ("private", "project:enfold")
    assert first.context.repository == "victor/enfold"
    assert memory.prefetch("what memory system?") == (
        "## Enfold Shared Memory\n- Victor uses Enfold"
    )

    memory.on_session_switch("session-branched", parent_session_id="session-main")
    assert adapters[0].sessions[-1].context.session_id == "session-branched"
    memory.shutdown()
    with pytest.raises(RuntimeError, match="not initialized"):
        memory.prefetch("after shutdown")


def test_explicit_tool_write_has_stable_event_and_session_attribution(tmp_path):
    memory, adapters = provider(tmp_path)
    memory.initialize("session-1", agent_identity="wonny")
    payload = json.loads(
        memory.handle_tool_call(
            "enfold_memory",
            {
                "action": "add",
                "event_id": "tool-call-99",
                "content": "Client A reviewed the Enfold daemon",
                "scope": "private",
                "category": "project",
            },
        )
    )
    assert payload["ok"] is True
    method, args, kwargs = adapters[0].sessions[0].calls[-1]
    assert method == "write"
    assert args == ("Client A reviewed the Enfold daemon",)
    assert kwargs["event_id"] == "tool-call-99"
    assert kwargs["source_type"] == "hermes_explicit_tool"
    assert adapters[0].sessions[0].context.session_id == "session-1"


def test_builtin_write_and_delegation_capture_parent_child_provenance(tmp_path):
    memory, adapters = provider(tmp_path)
    memory.initialize("parent-session", agent_identity="wonny")
    memory.on_memory_write(
        "add",
        "user",
        "Victor prefers concise responses",
        {"event_id": "builtin-1", "session_id": "parent-session"},
    )
    parent_write = adapters[0].sessions[0].calls[-1]
    assert parent_write[2]["source_type"] == "hermes_builtin_memory_write"
    assert parent_write[2]["category"] == "user_pref"

    memory.on_delegation(
        "review locking",
        "Lock ownership is safe",
        child_session_id="child-session",
        child_agent_id="reviewer-1",
    )
    child = adapters[0].sessions[-1]
    assert child.context.session_id == "child-session"
    assert child.context.agent_id == "reviewer-1"
    assert child.context.parent_agent_id == "wonny"
    assert child.calls[-1][2]["source_type"] == "hermes_delegation_result"
    assert "performed_by" not in child.calls[-1][2]


def test_builtin_hook_strips_nested_host_identity_metadata(tmp_path):
    memory, adapters = provider(tmp_path)
    memory.initialize("trusted-session", agent_identity="wonny")
    memory.on_memory_write(
        "add",
        "memory",
        "A deliberately attributed fact",
        {
            "event_id": "host-event-1",
            "session_id": "spoofed-session",
            "client_id": "spoofed-client",
            "nested": {"agent_id": "spoofed-agent", "safe": "kept"},
        },
    )
    call = adapters[0].sessions[0].calls[-1]
    assert adapters[0].sessions[0].context.session_id == "trusted-session"
    assert call[2]["metadata"] == {
        "action": "add",
        "target": "memory",
        "nested": {"safe": "kept"},
    }


def test_session_hooks_enqueue_attributed_transcripts_without_local_model(tmp_path):
    memory, adapters = provider(tmp_path)
    memory.initialize("parent-session", agent_identity="wonny")
    messages = [
        {"role": "user", "content": "This is a meaningful user message for extraction."},
        {"role": "assistant", "content": "This is a meaningful assistant response for extraction."},
        {"role": "user", "content": "Another meaningful user message about durable preferences."},
        {"role": "assistant", "content": "Another meaningful response about the shared memory."},
    ]

    assert memory.on_pre_compress(messages) == ""
    memory.on_session_end(messages)

    calls = [call for call in adapters[0].sessions[0].calls if call[0] == "enqueue_extraction"]
    assert [call[2]["source"] for call in calls] == ["pre_compress", "session_end"]
    assert all(call[2]["scope"] == "private" for call in calls)
    assert calls[0][1][0].startswith("USER: This is a meaningful")


def test_session_hook_utf8_safely_caps_long_transcript(tmp_path):
    memory, adapters = provider(tmp_path)
    memory.initialize("long-session", agent_identity="wonny")
    messages = [
        {"role": "user", "content": "old " * 4000},
        {"role": "assistant", "content": "recent 🧠 " * 2000},
    ]

    memory.on_session_end(messages)

    call = adapters[0].sessions[0].calls[-1]
    assert call[0] == "enqueue_extraction"
    transcript = call[1][0]
    assert len(transcript.encode("utf-8")) <= 10 * 1024
    assert "recent 🧠" in transcript


def test_daemon_unavailable_is_empty_prefetch_and_explicit_retryable_tool_error(tmp_path):
    memory, _ = provider(tmp_path, offline=True)
    memory.initialize("session-1", agent_identity="wonny")
    assert memory.prefetch("Victor") == ""
    result = json.loads(
        memory.handle_tool_call(
            "enfold_memory",
            {
                "action": "add",
                "event_id": "event-1",
                "content": "never spool this write",
            },
        )
    )
    assert result == {
        "error": "daemon_unavailable",
        "message": "test daemon offline",
        "ok": False,
        "retryable": True,
    }


def test_remote_errors_are_stable_and_lifecycle_hooks_remain_nonbreaking(tmp_path, caplog):
    remote = EnfoldRemoteError(
        ProtocolError("access_denied", "scope is not authorized")
    )
    memory, _ = provider(tmp_path, error=remote)
    memory.initialize("session-1", agent_identity="wonny")

    result = json.loads(
        memory.handle_tool_call(
            "enfold_memory", {"action": "search", "query": "private project"}
        )
    )
    assert result == {
        "error": "access_denied",
        "message": "scope is not authorized",
        "ok": False,
        "retryable": False,
    }
    assert memory.prefetch("private project") == ""
    memory.on_memory_write("add", "memory", "must not crash the host")
    memory.on_session_end(
        [{"role": "user", "content": "A durable message long enough to enqueue."}]
    )
    memory.on_delegation(
        "task", "result", child_session_id="child", child_agent_id="reviewer"
    )
    assert "[access_denied]" in caplog.text


def test_explicit_write_requires_host_event_id(tmp_path):
    memory, _ = provider(tmp_path)
    memory.initialize("session-1", agent_identity="wonny")
    result = json.loads(
        memory.handle_tool_call(
            "enfold_memory", {"action": "add", "content": "fact"}
        )
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_request"
