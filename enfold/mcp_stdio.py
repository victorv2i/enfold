"""MCP stdio adapter for the standalone Enfold v1 daemon.

This process is deliberately only a protocol bridge.  It never opens SQLite
and imports no Hermes modules.  Connection identity is fixed at process
startup from explicit command-line arguments (or their documented environment
variables), then negotiated with the daemon on every tool call.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import sys
from typing import Any, Protocol

from .client import (
    ClientConfig,
    EnfoldClient,
    EnfoldClientError,
    EnfoldProtocolError,
    EnfoldRemoteError,
    EnfoldTransportError,
)
from .mcp_proxy import MemoryMCPProxy
from .protocol import (
    CAPABILITY_CONFLICTS,
    CAPABILITY_CONTEXT,
    CAPABILITY_ENQUEUE_EXTRACTION,
    CAPABILITY_RESOLVE_CONFLICT,
    CAPABILITY_EVIDENCE,
    CAPABILITY_HISTORY,
    CAPABILITY_SEARCH,
    CAPABILITY_WRITE,
    ClientContext,
)


MEMORY_CAPABILITIES = (
    CAPABILITY_WRITE,
    CAPABILITY_SEARCH,
    CAPABILITY_CONTEXT,
    CAPABILITY_EVIDENCE,
    CAPABILITY_HISTORY,
    CAPABILITY_CONFLICTS,
    CAPABILITY_RESOLVE_CONFLICT,
    CAPABILITY_ENQUEUE_EXTRACTION,
)


class MCPApp(Protocol):
    def tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def run(self, *, transport: str) -> Any: ...


class MCPBridgeError(RuntimeError):
    """JSON-serializable typed failure at the MCP boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = dict(details or {})
        self.request_id = request_id

    def payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
            "request_id": self.request_id,
        }


def _json_safe(value: Any, *, label: str) -> Any:
    """Validate and normalize a value to plain JSON containers."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise MCPBridgeError(
            "invalid_daemon_result",
            f"Enfold returned a non-JSON {label}",
        ) from exc


def _proxy_call(operation: Callable[[Mapping[str, Any]], Any], params: dict[str, Any]) -> Any:
    try:
        return _json_safe(operation(params), label="result")
    except EnfoldRemoteError as exc:
        raise MCPBridgeError(
            exc.code,
            exc.message,
            retryable=exc.retryable,
            details=_json_safe(exc.details, label="error details"),
            request_id=exc.request_id,
        ) from exc
    except EnfoldTransportError as exc:
        raise MCPBridgeError(
            "daemon_unavailable", str(exc), retryable=True
        ) from exc
    except EnfoldProtocolError as exc:
        raise MCPBridgeError("protocol_error", str(exc)) from exc
    except EnfoldClientError as exc:
        raise MCPBridgeError("client_error", str(exc)) from exc


def _typed_tool_error(error_type: type[Exception], exc: MCPBridgeError) -> Exception:
    payload = _json_safe(exc.payload(), label="error")
    return error_type(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _load_mcp() -> tuple[type[Any], type[Exception]]:
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.server.fastmcp.exceptions import ToolError
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "The optional 'mcp' package is required for the Enfold stdio proxy. "
            "Install it with: pip install 'enfold[mcp]'"
        ) from exc
    return FastMCP, ToolError


def build_server(
    config: ClientConfig,
    *,
    server_factory: Callable[[str], MCPApp] | None = None,
    transport_factory: Callable[[ClientConfig], Any] = EnfoldClient,
    tool_error_type: type[Exception] | None = None,
) -> MCPApp:
    """Register Enfold v1 tools without opening a database or socket.

    The socket is opened later by :class:`EnfoldClient`, once per tool call.
    Injectable factories keep tests independent of the optional MCP package
    and of a running daemon.
    """

    if server_factory is None or tool_error_type is None:
        fast_mcp, mcp_tool_error = _load_mcp()
        server_factory = server_factory or fast_mcp
        tool_error_type = tool_error_type or mcp_tool_error
    server = server_factory("enfold-memory-v1")
    proxy = MemoryMCPProxy(transport_factory(config))

    def invoke(operation: Callable[[Mapping[str, Any]], Any], params: dict[str, Any]) -> Any:
        try:
            return _proxy_call(operation, params)
        except MCPBridgeError as exc:
            raise _typed_tool_error(tool_error_type, exc) from exc

    @server.tool()
    def memory_write(
        idempotency_key: str,
        content: str,
        source_type: str,
        category: str = "general",
        tags: str = "",
        trust_score: float = 0.5,
        source_authority: float = 0.5,
        source_uri: str | None = None,
        observation_content: str | None = None,
        asserted_by: str | None = None,
        observed_at: str | None = None,
        scope: str = "private",
        sensitivity: str = "normal",
        correction_status: str | None = None,
        evidence_excerpt: str | None = None,
        relation: str = "supports",
        metadata: dict[str, Any] | None = None,
        supersede_fact_id: int | None = None,
        state: dict[str, Any] | None = None,
    ) -> Any:
        """Write one durable memory with evidence and idempotency protection.

        Writer identity, agent, session, project, repository, branch, commit,
        and granted scopes come from this proxy's startup context, not tool
        arguments. ``asserted_by`` identifies the subject making a claim and
        is not connection identity.
        """

        return invoke(
            proxy.write,
            {
                "idempotency_key": idempotency_key,
                "content": content,
                "source_type": source_type,
                "category": category,
                "tags": tags,
                "trust_score": trust_score,
                "source_authority": source_authority,
                "source_uri": source_uri,
                "observation_content": observation_content,
                "asserted_by": asserted_by,
                "observed_at": observed_at,
                "scope": scope,
                "sensitivity": sensitivity,
                "correction_status": correction_status,
                "evidence_excerpt": evidence_excerpt,
                "relation": relation,
                "metadata": metadata,
                "supersede_fact_id": supersede_fact_id,
                "state": state,
            },
        )

    @server.tool()
    def memory_search(
        query: str,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 20,
    ) -> Any:
        """Search active memories visible to this proxy's granted scopes."""

        return invoke(
            proxy.search,
            {
                "query": query,
                "category": category,
                "min_trust": min_trust,
                "limit": limit,
            },
        )

    @server.tool()
    def memory_context(
        query: str,
        token_budget: int = 256,
        scope: str | None = None,
    ) -> Any:
        """Return compact, cited current memory for this proxy's granted scopes."""

        params: dict[str, Any] = {
            "query": query,
            "token_budget": token_budget,
        }
        if scope is not None:
            params["scope"] = scope
        return invoke(proxy.context, params)

    @server.tool()
    def memory_evidence(fact_id: int, limit: int = 100) -> Any:
        """Return a fact and its visible source observations/provenance."""

        return invoke(proxy.evidence, {"fact_id": fact_id, "limit": limit})

    @server.tool()
    def memory_history(
        fact_id: int | None = None,
        subject_key: str | None = None,
        predicate_key: str | None = None,
        scope: str | None = None,
        limit: int = 100,
    ) -> Any:
        """Return history by fact ID or by subject/predicate state slot."""

        params: dict[str, Any] = {"limit": limit}
        if fact_id is not None:
            params["fact_id"] = fact_id
        if subject_key is not None:
            params["subject_key"] = subject_key
        if predicate_key is not None:
            params["predicate_key"] = predicate_key
        if scope is not None:
            params["scope"] = scope
        return invoke(proxy.history, params)

    @server.tool()
    def memory_changes(
        since: str,
        until: str,
        scope: str | None = None,
        limit: int = 100,
    ) -> Any:
        """Return created, superseded, and resolved facts in a half-open time window."""

        params: dict[str, Any] = {"since": since, "until": until, "limit": limit}
        if scope is not None:
            params["scope"] = scope
        return invoke(proxy.changes, params)

    @server.tool()
    def memory_timeline(
        subject_or_query: str,
        scope: str | None = None,
        limit: int = 100,
    ) -> Any:
        """Return chronological settled fact events for a subject or query."""

        params: dict[str, Any] = {"subject_or_query": subject_or_query, "limit": limit}
        if scope is not None:
            params["scope"] = scope
        return invoke(proxy.timeline, params)

    @server.tool()
    def memory_entities(
        scope: str | None = None,
        min_facts: int = 1,
        limit: int = 100,
    ) -> Any:
        """Rank visible entities derived from current fact subjects and tags."""

        params: dict[str, Any] = {"min_facts": min_facts, "limit": limit}
        if scope is not None:
            params["scope"] = scope
        return invoke(proxy.entities, params)

    @server.tool()
    def memory_entity(
        name: str,
        scope: str | None = None,
        limit: int = 100,
    ) -> Any:
        """Return current facts, recent changes, and open conflicts for an entity."""

        params: dict[str, Any] = {"name": name, "limit": limit}
        if scope is not None:
            params["scope"] = scope
        return invoke(proxy.entity, params)

    @server.tool()
    def memory_conflicts(
        scope: str | None = None,
        unresolved_only: bool = True,
    ) -> Any:
        """List visible state conflicts, optionally including resolved ones."""

        params: dict[str, Any] = {"unresolved_only": unresolved_only}
        if scope is not None:
            params["scope"] = scope
        return invoke(proxy.conflicts, params)

    @server.tool()
    def memory_resolve_conflict(
        conflict_id: str,
        resolution_fact_id: int,
        reason: str,
    ) -> Any:
        """Resolve a state conflict to one member with resolver audit provenance."""

        return invoke(
            proxy.resolve_conflict,
            {
                "conflict_id": conflict_id,
                "resolution_fact_id": resolution_fact_id,
                "reason": reason,
            },
        )

    @server.tool()
    def memory_extraction_enqueue(
        transcript: str,
        source: str,
        scope: str = "private",
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Queue a scoped transcript for deferred, daemon-owned extraction.

        The proxy never invokes a model or writes SQLite. Connection identity
        is fixed at startup and identity-shaped nested metadata is rejected by
        the service boundary.
        """

        return invoke(
            proxy.enqueue_extraction,
            {
                "transcript": transcript,
                "source": source,
                "scope": scope,
                "metadata": metadata,
            },
        )

    return server


def _env(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    return value if value and value.strip() else None


def _parse_scopes(values: Sequence[str] | None, fallback: str | None) -> tuple[str, ...]:
    raw = list(values or ())
    if not raw and fallback:
        raw = [fallback]
    scopes = tuple(part.strip() for value in raw for part in value.split(",") if part.strip())
    if not scopes:
        return ("private",)
    if len(scopes) != len(set(scopes)):
        raise ValueError("access scopes must not contain duplicates")
    return scopes


def _parser(environ: Mapping[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enfold-mcp-proxy",
        description=(
            "Run a stdio MCP bridge to an already-running Enfold v1 Unix daemon. "
            "This command never opens SQLite. Identity is fixed at startup."
        ),
        epilog=(
            "Environment equivalents: ENFOLD_SOCKET_PATH, ENFOLD_CLIENT_ID, "
            "ENFOLD_SURFACE, ENFOLD_AGENT_ID, ENFOLD_SESSION_ID, "
            "ENFOLD_PARENT_AGENT_ID, ENFOLD_PROJECT_ROOT, ENFOLD_REPOSITORY, "
            "ENFOLD_BRANCH, ENFOLD_COMMIT_SHA, ENFOLD_ACCESS_SCOPES."
        ),
    )
    parser.add_argument("--socket-path", default=_env(environ, "ENFOLD_SOCKET_PATH"), help="absolute Enfold daemon Unix socket path [ENFOLD_SOCKET_PATH]")
    parser.add_argument("--client-id", default=_env(environ, "ENFOLD_CLIENT_ID"), help="stable client installation ID [ENFOLD_CLIENT_ID]")
    parser.add_argument("--surface", default=_env(environ, "ENFOLD_SURFACE"), help="agent surface, e.g. mcp-client-a or mcp-client-b [ENFOLD_SURFACE]")
    parser.add_argument("--agent-id", default=_env(environ, "ENFOLD_AGENT_ID"), help="writer agent ID [ENFOLD_AGENT_ID]")
    parser.add_argument("--session-id", default=_env(environ, "ENFOLD_SESSION_ID"), help="session/thread ID [ENFOLD_SESSION_ID]")
    parser.add_argument("--parent-agent-id", default=_env(environ, "ENFOLD_PARENT_AGENT_ID"))
    parser.add_argument("--project-root", default=_env(environ, "ENFOLD_PROJECT_ROOT"))
    parser.add_argument("--repository", default=_env(environ, "ENFOLD_REPOSITORY"))
    parser.add_argument("--branch", default=_env(environ, "ENFOLD_BRANCH"))
    parser.add_argument("--commit-sha", default=_env(environ, "ENFOLD_COMMIT_SHA"))
    parser.add_argument("--access-scope", action="append", dest="access_scopes", metavar="SCOPE", help="granted scope; repeat or comma-separate [ENFOLD_ACCESS_SCOPES; default private]")
    parser.add_argument("--connect-timeout", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    return parser


def parse_config(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ClientConfig:
    env = os.environ if environ is None else environ
    parser = _parser(env)
    args = parser.parse_args(argv)
    missing = [
        flag
        for flag, value in (
            ("--socket-path", args.socket_path),
            ("--client-id", args.client_id),
            ("--surface", args.surface),
            ("--agent-id", args.agent_id),
            ("--session-id", args.session_id),
        )
        if not value
    ]
    if missing:
        parser.error("required startup identity missing: " + ", ".join(missing))
    try:
        context = ClientContext(
            client_id=args.client_id,
            surface=args.surface,
            agent_id=args.agent_id,
            session_id=args.session_id,
            parent_agent_id=args.parent_agent_id,
            project_root=args.project_root,
            repository=args.repository,
            branch=args.branch,
            commit_sha=args.commit_sha,
            access_scopes=_parse_scopes(
                args.access_scopes, _env(env, "ENFOLD_ACCESS_SCOPES")
            ),
        )
        return ClientConfig(
            socket_path=Path(args.socket_path).expanduser(),
            context=context,
            capabilities=MEMORY_CAPABILITIES,
            connect_timeout=args.connect_timeout,
            request_timeout=args.request_timeout,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError("argparse.error did not exit")


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    try:
        server = build_server(config)
    except RuntimeError as exc:
        print(f"enfold MCP proxy startup failed: {exc}", file=sys.stderr)
        return 2
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
