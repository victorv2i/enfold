"""Optional Hermes MemoryProvider bridge for the standalone Enfold v1 service.

This directory is a staging artifact.  It is not imported, installed, or
registered by Enfold itself.  During a controlled maintenance window it can be
copied to ``$HERMES_HOME/plugins/enfold_v1`` after the daemon is ready.
"""

from __future__ import annotations

from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from agent.memory_provider import MemoryProvider
from enfold.client import EnfoldClientError, EnfoldRemoteError, EnfoldTransportError
from enfold.hermes_adapter import (
    HermesAdapterConfig,
    HermesMemorySession,
    HermesProtocolAdapter,
    HermesSessionContext,
)


logger = logging.getLogger(__name__)
_MAX_TRANSCRIPT_BYTES = 10 * 1024
_IDENTITY_METADATA_KEYS = frozenset({
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
})


def _strip_identity_metadata(value: Any) -> Any:
    """Remove host identity claims before they enter daemon request params."""

    if isinstance(value, Mapping):
        return {
            key: _strip_identity_metadata(item)
            for key, item in value.items()
            if key not in _IDENTITY_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_strip_identity_metadata(item) for item in value]
    return value


def _client_failure(exc: EnfoldClientError) -> tuple[str, bool]:
    if isinstance(exc, EnfoldRemoteError):
        return exc.code, exc.retryable
    if isinstance(exc, EnfoldTransportError):
        return "daemon_unavailable", True
    return "client_error", False


def _format_messages(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            lines.append(f"{role.upper()}: {content.strip()}")
    transcript = "\n\n".join(lines)
    encoded = transcript.encode("utf-8")
    if len(encoded) <= _MAX_TRANSCRIPT_BYTES:
        return transcript
    # Recent turns normally carry the active decisions and preferences.  Keep
    # a UTF-8-safe tail while leaving envelope room under the daemon's cap.
    return encoded[-_MAX_TRANSCRIPT_BYTES:].decode("utf-8", errors="ignore")


_TOOL_SCHEMA = {
    "name": "enfold_memory",
    "description": "Search or explicitly write scoped Enfold memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "add", "evidence", "history", "conflicts"],
            },
            "query": {"type": "string"},
            "content": {"type": "string"},
            "event_id": {
                "type": "string",
                "description": "Stable host event/tool-call id required for add.",
            },
            "scope": {"type": "string", "default": "private"},
            "category": {"type": "string"},
            "tags": {"type": "string"},
            "fact_id": {"type": "integer"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            "subject_key": {"type": "string"},
            "predicate_key": {"type": "string"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not values:
        raise ValueError("ENFOLD_HERMES_SCOPES must contain at least one scope")
    return values


def _stable_event(prefix: str, values: Mapping[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}:{sha256(encoded.encode('utf-8')).hexdigest()}"


class EnfoldV1MemoryProvider(MemoryProvider):
    """Thin Hermes lifecycle adapter; all canonical state lives in the daemon."""

    def __init__(self, *, adapter_factory=HermesProtocolAdapter, environ=None) -> None:
        self._adapter_factory = adapter_factory
        self._environ = os.environ if environ is None else environ
        self._adapter: HermesProtocolAdapter | None = None
        self._sessions: dict[str, HermesMemorySession] = {}
        self._session_id = ""
        self._identity = "hermes"
        self._agent_context = "primary"
        self._parent_session_id = ""
        self._scopes: tuple[str, ...] = ("private",)
        self._host_fields: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "enfold_v1"

    def is_available(self) -> bool:
        try:
            return Path(self._socket_value()).expanduser().is_absolute()
        except (TypeError, ValueError):
            return False

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Hermes session_id must not be empty")
        socket_path = Path(self._socket_value()).expanduser()
        self._scopes = _csv(self._environ.get("ENFOLD_HERMES_SCOPES", "private"))
        self._identity = str(kwargs.get("agent_identity") or "hermes")
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._parent_session_id = str(kwargs.get("parent_session_id") or "")
        self._host_fields = {
            "project_root": kwargs.get("project_root") or kwargs.get("agent_workspace"),
            "repository": kwargs.get("repository"),
            "branch": kwargs.get("branch"),
            "commit_sha": kwargs.get("commit_sha"),
        }
        config = HermesAdapterConfig(
            socket_path=socket_path,
            client_id=self._environ.get("ENFOLD_HERMES_CLIENT_ID", "hermes-install"),
            connect_timeout=float(self._environ.get("ENFOLD_HERMES_CONNECT_TIMEOUT", "2")),
            request_timeout=float(self._environ.get("ENFOLD_HERMES_REQUEST_TIMEOUT", "5")),
        )
        self._adapter = self._adapter_factory(config)
        self._sessions.clear()
        self._session_id = session_id.strip()
        self._session_for(self._session_id)

    def _socket_value(self) -> str:
        return self._environ.get(
            "ENFOLD_SOCKET_PATH", str(Path.home() / ".hermes" / "enfold-v1.sock")
        )

    def _session_for(
        self,
        session_id: str,
        *,
        agent_id: str | None = None,
        parent_agent_id: str | None = None,
    ) -> HermesMemorySession:
        if self._adapter is None:
            raise RuntimeError("Enfold Hermes provider is not initialized")
        key = f"{session_id}\0{agent_id or self._identity}\0{parent_agent_id or ''}"
        existing = self._sessions.get(key)
        if existing is not None:
            return existing
        context = HermesSessionContext(
            agent_id=agent_id or self._identity,
            session_id=session_id,
            parent_agent_id=parent_agent_id,
            access_scopes=self._scopes,
            **self._host_fields,
        )
        opened = self._adapter.open_session(context)
        self._sessions[key] = opened
        return opened

    def _current(self, session_id: str = "") -> HermesMemorySession:
        return self._session_for(session_id or self._session_id)

    def system_prompt_block(self) -> str:
        return (
            "# Enfold Shared Memory\n"
            "Use enfold_memory for deliberate durable facts and scoped recall. "
            "Writes are attributed to this Hermes agent and session."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not query.strip():
            return ""
        try:
            result = self._current(session_id).search(query, limit=5)
        except EnfoldClientError as exc:
            code, _retryable = _client_failure(exc)
            logger.warning("Enfold Hermes prefetch failed [%s]: %s", code, exc)
            return ""
        facts = result.get("facts", []) if isinstance(result, Mapping) else []
        lines = [f"- {fact.get('content', '')}" for fact in facts if fact.get("content")]
        return "## Enfold Shared Memory\n" + "\n".join(lines) if lines else ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [dict(_TOOL_SCHEMA)]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if tool_name != "enfold_memory":
            return json.dumps({"ok": False, "error": "unknown_tool"})
        try:
            session = self._current(str(kwargs.get("session_id") or ""))
            action = args.get("action")
            if action == "search":
                result = session.search(
                    str(args.get("query") or ""), limit=int(args.get("limit", 10))
                )
            elif action == "add":
                event_id = str(args.get("event_id") or "").strip()
                if not event_id:
                    raise ValueError("event_id is required for an explicit write")
                result = session.write(
                    str(args.get("content") or ""),
                    event_id=event_id,
                    source_type="hermes_explicit_tool",
                    scope=str(args.get("scope") or "private"),
                    category=str(args.get("category") or "general"),
                    tags=str(args.get("tags") or ""),
                )
            elif action == "evidence":
                result = session.evidence(int(args["fact_id"]), limit=args.get("limit"))
            elif action == "history":
                selector = {
                    key: args[key]
                    for key in ("subject_key", "predicate_key", "scope")
                    if args.get(key) is not None
                }
                result = session.history(**selector)
            elif action == "conflicts":
                result = session.conflicts(scope=args.get("scope"))
            else:
                raise ValueError("unsupported Enfold memory action")
            return json.dumps({"ok": True, "result": result}, sort_keys=True)
        except EnfoldClientError as exc:
            code, retryable = _client_failure(exc)
            return json.dumps(
                {
                    "ok": False,
                    "error": code,
                    "retryable": retryable,
                    "message": str(exc),
                },
                sort_keys=True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return json.dumps({"ok": False, "error": "invalid_request", "message": str(exc)}, sort_keys=True)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if action not in {"add", "replace"} or not content:
            return
        details = _strip_identity_metadata(dict(metadata or {}))
        event_id = str(details.pop("event_id", "") or _stable_event(
            "builtin-memory", {"session": self._session_id, "action": action, "target": target, "content": content, "metadata": details}
        ))
        try:
            self._current().write(
                content,
                event_id=event_id,
                source_type="hermes_builtin_memory_write",
                scope="private",
                category="user_pref" if target == "user" else "general",
                metadata={"action": action, "target": target, **details},
            )
        except EnfoldClientError as exc:
            code, _retryable = _client_failure(exc)
            logger.warning("Enfold Hermes memory-write hook failed [%s]: %s", code, exc)

    def _enqueue_messages(
        self, messages: List[Dict[str, Any]], *, source: str
    ) -> bool:
        transcript = _format_messages(messages)
        if not transcript:
            return False
        try:
            self._current().enqueue_extraction(
                transcript,
                source=source,
                scope="private",
                metadata={"host": "hermes", "hook": source},
            )
            return True
        except EnfoldClientError as exc:
            code, _retryable = _client_failure(exc)
            logger.warning("Enfold extraction enqueue failed [%s]: %s", code, exc)
            return False
        except ValueError as exc:
            logger.warning("Enfold extraction enqueue rejected [invalid_request]: %s", exc)
            return False

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        meaningful = [
            message
            for message in messages
            if message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
            and len(message["content"].strip()) > 20
        ]
        if len(meaningful) >= 4:
            self._enqueue_messages(messages, source="pre_compress")
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._enqueue_messages(messages, source="session_end")

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        if not child_session_id or not result:
            return
        child_agent = str(kwargs.get("child_agent_id") or f"{self._identity}:subagent")
        child = self._session_for(
            child_session_id,
            agent_id=child_agent,
            parent_agent_id=self._identity,
        )
        try:
            child.write(
                result,
                event_id=_stable_event(
                    "delegation", {"child_session_id": child_session_id, "task": task, "result": result}
                ),
                source_type="hermes_delegation_result",
                scope="private",
                metadata={"task": task, "parent_session": self._session_id},
            )
        except EnfoldClientError as exc:
            code, _retryable = _client_failure(exc)
            logger.warning("Enfold delegation hook failed [%s]: %s", code, exc)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        self._parent_session_id = parent_session_id
        self._session_id = new_session_id
        self._session_for(new_session_id)

    def shutdown(self) -> None:
        self._sessions.clear()
        self._adapter = None
        self._session_id = ""


def register(ctx: Any) -> None:
    ctx.register_memory_provider(EnfoldV1MemoryProvider())
