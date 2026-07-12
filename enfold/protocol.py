"""Versioned JSON-lines contract for a future local Enfold daemon.

This module defines frames and compatibility rules only.  It opens no sockets,
databases, or services and has no import-time side effects.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Union

from .schema import SUPPORTED_SCHEMA_VERSION


PROTOCOL_MAJOR = 1
PROTOCOL_MINOR = 0
MAX_FRAME_SIZE = 1024 * 1024
MAX_JSON_DEPTH = 32

CAPABILITY_HEALTH = "health"
CAPABILITY_WRITE = "memory.write"
CAPABILITY_SEARCH = "memory.search"
CAPABILITY_CONTEXT = "memory.context"
CAPABILITY_EVIDENCE = "memory.evidence"
CAPABILITY_HISTORY = "memory.history"
CAPABILITY_PROJECTIONS = CAPABILITY_SEARCH
CAPABILITY_CHANGES = CAPABILITY_PROJECTIONS
CAPABILITY_TIMELINE = CAPABILITY_PROJECTIONS
CAPABILITY_ENTITIES = CAPABILITY_PROJECTIONS
CAPABILITY_ENTITY = CAPABILITY_PROJECTIONS
CAPABILITY_CONFLICTS = "memory.conflicts"
CAPABILITY_RESOLVE_CONFLICT = "memory.resolve_conflict"
CAPABILITY_ENQUEUE_EXTRACTION = "memory.extraction.enqueue"

SUPPORTED_CAPABILITIES = (
    CAPABILITY_HEALTH,
    CAPABILITY_WRITE,
    CAPABILITY_SEARCH,
    CAPABILITY_CONTEXT,
    CAPABILITY_EVIDENCE,
    CAPABILITY_HISTORY,
    CAPABILITY_CONFLICTS,
    CAPABILITY_RESOLVE_CONFLICT,
    CAPABILITY_ENQUEUE_EXTRACTION,
)
METHOD_CAPABILITIES = {
    "health": CAPABILITY_HEALTH,
    "memory.write": CAPABILITY_WRITE,
    "memory.search": CAPABILITY_SEARCH,
    "memory.context": CAPABILITY_CONTEXT,
    "memory.evidence": CAPABILITY_EVIDENCE,
    "memory.history": CAPABILITY_HISTORY,
    "memory.changes": CAPABILITY_CHANGES,
    "memory.timeline": CAPABILITY_TIMELINE,
    "memory.entities": CAPABILITY_ENTITIES,
    "memory.entity": CAPABILITY_ENTITY,
    "memory.conflicts": CAPABILITY_CONFLICTS,
    "memory.resolve_conflict": CAPABILITY_RESOLVE_CONFLICT,
    "memory.extraction.enqueue": CAPABILITY_ENQUEUE_EXTRACTION,
}
IMMUTABLE_CONTEXT_FIELDS = frozenset({
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
})

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_MISSING = object()


class ProtocolValidationError(ValueError):
    """A frame is malformed or violates the protocol contract."""


class FrameTooLargeError(ProtocolValidationError):
    """A frame exceeds the configured byte limit."""


class ProtocolVersionMismatch(ProtocolValidationError):
    """A peer uses an incompatible protocol major version."""


class RequestHandlingError(ValueError):
    """Safe application failure that transports may serialize verbatim."""

    def __init__(self, code: str, message: str):
        if not isinstance(code, str) or not code.strip():
            raise ValueError("request error code must not be empty")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("request error message must not be empty")
        super().__init__(message.strip())
        self.code = code.strip()
        self.message = message.strip()


def _required_string(value: Any, name: str, *, token: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{name} must be a non-empty string")
    value = value.strip()
    if token and not _TOKEN.fullmatch(value):
        raise ProtocolValidationError(f"{name} is not a valid protocol token")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, name)


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _string_tuple(value: Any, name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolValidationError(f"{name} must be an array of strings")
    result = tuple(_required_string(item, name, token=True) for item in value)
    if not allow_empty and not result:
        raise ProtocolValidationError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ProtocolValidationError(f"{name} must not contain duplicates")
    return result


def _validate_json(value: Any, name: str, *, depth: int = 0) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise ProtocolValidationError(f"{name} exceeds maximum JSON depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolValidationError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_validate_json(item, name, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return [_validate_json(item, name, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolValidationError(f"{name} object keys must be strings")
            result[key] = _validate_json(item, name, depth=depth + 1)
        return result
    raise ProtocolValidationError(f"{name} contains a non-JSON value")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"{name} must be an object")
    return value


def _check_keys(
    data: Mapping[str, Any],
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = sorted(required - data.keys())
    unknown = sorted(data.keys() - required - optional)
    if missing:
        raise ProtocolValidationError(f"missing frame fields: {missing}")
    if unknown:
        raise ProtocolValidationError(f"unknown frame fields: {unknown}")


@dataclass(frozen=True, slots=True)
class ProtocolVersion:
    major: int = PROTOCOL_MAJOR
    minor: int = PROTOCOL_MINOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "major", _strict_int(self.major, "protocol.major"))
        object.__setattr__(self, "minor", _strict_int(self.minor, "protocol.minor"))

    def to_dict(self) -> dict[str, int]:
        return {"major": self.major, "minor": self.minor}

    @classmethod
    def from_dict(cls, value: Any) -> "ProtocolVersion":
        data = _mapping(value, "protocol")
        _check_keys(data, {"major", "minor"})
        return cls(data["major"], data["minor"])


@dataclass(frozen=True, slots=True)
class ClientContext:
    """Trusted connection identity; requests cannot override these fields."""

    client_id: str
    surface: str
    agent_id: str
    session_id: str
    parent_agent_id: str | None = None
    project_root: str | None = None
    repository: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    access_scopes: tuple[str, ...] = ("private",)

    def __post_init__(self) -> None:
        for name in ("client_id", "surface", "agent_id", "session_id"):
            object.__setattr__(self, name, _required_string(getattr(self, name), name, token=True))
        for name in ("parent_agent_id", "project_root", "repository", "branch", "commit_sha"):
            object.__setattr__(self, name, _optional_string(getattr(self, name), name))
        object.__setattr__(
            self,
            "access_scopes",
            _string_tuple(self.access_scopes, "access_scopes", allow_empty=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "surface": self.surface,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "parent_agent_id": self.parent_agent_id,
            "project_root": self.project_root,
            "repository": self.repository,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "access_scopes": list(self.access_scopes),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ClientContext":
        data = _mapping(value, "context")
        required = {"client_id", "surface", "agent_id", "session_id", "access_scopes"}
        optional = {"parent_agent_id", "project_root", "repository", "branch", "commit_sha"}
        _check_keys(data, required, optional)
        return cls(
            client_id=data["client_id"],
            surface=data["surface"],
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            parent_agent_id=data.get("parent_agent_id"),
            project_root=data.get("project_root"),
            repository=data.get("repository"),
            branch=data.get("branch"),
            commit_sha=data.get("commit_sha"),
            access_scopes=_string_tuple(data["access_scopes"], "access_scopes", allow_empty=False),
        )


@dataclass(frozen=True, slots=True)
class ProtocolError:
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _required_string(self.code, "error.code", token=True))
        object.__setattr__(self, "message", _required_string(self.message, "error.message"))
        if not isinstance(self.retryable, bool):
            raise ProtocolValidationError("error.retryable must be a boolean")
        object.__setattr__(self, "details", _mapping(_validate_json(self.details, "error.details"), "error.details"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ProtocolError":
        data = _mapping(value, "error")
        _check_keys(data, {"code", "message", "retryable", "details"})
        return cls(data["code"], data["message"], data["retryable"], data["details"])


@dataclass(frozen=True, slots=True)
class Handshake:
    context: ClientContext
    capabilities: tuple[str, ...] = SUPPORTED_CAPABILITIES
    protocol: ProtocolVersion = field(default_factory=ProtocolVersion)
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.context, ClientContext):
            raise ProtocolValidationError("context must be ClientContext")
        object.__setattr__(self, "capabilities", _string_tuple(self.capabilities, "capabilities"))
        if not isinstance(self.protocol, ProtocolVersion):
            raise ProtocolValidationError("protocol must be ProtocolVersion")
        object.__setattr__(self, "schema_version", _strict_int(self.schema_version, "schema_version"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "handshake",
            "protocol": self.protocol.to_dict(),
            "schema_version": self.schema_version,
            "capabilities": list(self.capabilities),
            "context": self.context.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Handshake":
        _check_keys(data, {"type", "protocol", "schema_version", "capabilities", "context"})
        if data["type"] != "handshake":
            raise ProtocolValidationError("frame type must be handshake")
        return cls(
            context=ClientContext.from_dict(data["context"]),
            capabilities=_string_tuple(data["capabilities"], "capabilities"),
            protocol=ProtocolVersion.from_dict(data["protocol"]),
            schema_version=_strict_int(data["schema_version"], "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class HandshakeResponse:
    accepted: bool
    capabilities: tuple[str, ...]
    service_version: str
    error: ProtocolError | None = None
    protocol: ProtocolVersion = field(default_factory=ProtocolVersion)
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise ProtocolValidationError("accepted must be a boolean")
        object.__setattr__(self, "capabilities", _string_tuple(self.capabilities, "capabilities"))
        object.__setattr__(self, "service_version", _required_string(self.service_version, "service_version"))
        if not isinstance(self.protocol, ProtocolVersion):
            raise ProtocolValidationError("protocol must be ProtocolVersion")
        object.__setattr__(self, "schema_version", _strict_int(self.schema_version, "schema_version"))
        if self.accepted and self.error is not None:
            raise ProtocolValidationError("accepted handshake cannot include an error")
        if not self.accepted and not isinstance(self.error, ProtocolError):
            raise ProtocolValidationError("refused handshake must include a typed error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "handshake_response",
            "protocol": self.protocol.to_dict(),
            "schema_version": self.schema_version,
            "capabilities": list(self.capabilities),
            "service_version": self.service_version,
            "accepted": self.accepted,
            "error": None if self.error is None else self.error.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HandshakeResponse":
        required = {"type", "protocol", "schema_version", "capabilities", "service_version", "accepted", "error"}
        _check_keys(data, required)
        if data["type"] != "handshake_response":
            raise ProtocolValidationError("frame type must be handshake_response")
        error = None if data["error"] is None else ProtocolError.from_dict(data["error"])
        return cls(
            accepted=data["accepted"],
            capabilities=_string_tuple(data["capabilities"], "capabilities"),
            service_version=data["service_version"],
            error=error,
            protocol=ProtocolVersion.from_dict(data["protocol"]),
            schema_version=_strict_int(data["schema_version"], "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class Request:
    request_id: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    protocol: ProtocolVersion = field(default_factory=ProtocolVersion)
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required_string(self.request_id, "request_id", token=True))
        method = _required_string(self.method, "method", token=True)
        if method not in METHOD_CAPABILITIES:
            raise ProtocolValidationError(f"unsupported method: {method}")
        object.__setattr__(self, "method", method)
        params = _mapping(_validate_json(self.params, "params"), "params")
        spoofed = sorted(IMMUTABLE_CONTEXT_FIELDS & params.keys())
        if spoofed:
            raise ProtocolValidationError(
                f"request params cannot override immutable connection context: {spoofed}"
            )
        object.__setattr__(self, "params", params)
        if not isinstance(self.protocol, ProtocolVersion):
            raise ProtocolValidationError("protocol must be ProtocolVersion")
        object.__setattr__(self, "schema_version", _strict_int(self.schema_version, "schema_version"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "request",
            "protocol": self.protocol.to_dict(),
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "method": self.method,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Request":
        _check_keys(data, {"type", "protocol", "schema_version", "request_id", "method", "params"})
        if data["type"] != "request":
            raise ProtocolValidationError("frame type must be request")
        return cls(
            request_id=data["request_id"],
            method=data["method"],
            params=data["params"],
            protocol=ProtocolVersion.from_dict(data["protocol"]),
            schema_version=_strict_int(data["schema_version"], "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class Response:
    request_id: str
    ok: bool
    result: Any = None
    error: ProtocolError | None = None
    protocol: ProtocolVersion = field(default_factory=ProtocolVersion)
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required_string(self.request_id, "request_id", token=True))
        if not isinstance(self.ok, bool):
            raise ProtocolValidationError("ok must be a boolean")
        object.__setattr__(self, "result", _validate_json(self.result, "result"))
        if not isinstance(self.protocol, ProtocolVersion):
            raise ProtocolValidationError("protocol must be ProtocolVersion")
        object.__setattr__(self, "schema_version", _strict_int(self.schema_version, "schema_version"))
        if self.ok and self.error is not None:
            raise ProtocolValidationError("successful response cannot include an error")
        if not self.ok:
            if not isinstance(self.error, ProtocolError):
                raise ProtocolValidationError("failed response must include a typed error")
            if self.result is not None:
                raise ProtocolValidationError("failed response cannot include a result")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "response",
            "protocol": self.protocol.to_dict(),
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "ok": self.ok,
            "result": self.result,
            "error": None if self.error is None else self.error.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Response":
        required = {"type", "protocol", "schema_version", "request_id", "ok", "result", "error"}
        _check_keys(data, required)
        if data["type"] != "response":
            raise ProtocolValidationError("frame type must be response")
        error = None if data["error"] is None else ProtocolError.from_dict(data["error"])
        return cls(
            request_id=data["request_id"],
            ok=data["ok"],
            result=data["result"],
            error=error,
            protocol=ProtocolVersion.from_dict(data["protocol"]),
            schema_version=_strict_int(data["schema_version"], "schema_version"),
        )


Frame = Union[Handshake, HandshakeResponse, Request, Response]


def required_capability(method: str) -> str:
    method = _required_string(method, "method", token=True)
    try:
        return METHOD_CAPABILITIES[method]
    except KeyError as exc:
        raise ProtocolValidationError(f"unsupported method: {method}") from exc


def require_compatible_major(version: ProtocolVersion, *, supported_major: int = PROTOCOL_MAJOR) -> None:
    if version.major != supported_major:
        raise ProtocolVersionMismatch(
            f"protocol major {version.major} is incompatible with supported major {supported_major}"
        )


def negotiate_handshake(
    handshake: Handshake,
    *,
    service_version: str,
    server_capabilities: tuple[str, ...] = SUPPORTED_CAPABILITIES,
    schema_version: int = SUPPORTED_SCHEMA_VERSION,
) -> HandshakeResponse:
    """Return an explicit acceptance or refusal without raising on peer major mismatch."""
    service_version = _required_string(service_version, "service_version")
    server_capabilities = _string_tuple(server_capabilities, "server_capabilities")
    schema_version = _strict_int(schema_version, "schema_version")
    if handshake.protocol.major != PROTOCOL_MAJOR:
        return HandshakeResponse(
            accepted=False,
            capabilities=(),
            service_version=service_version,
            error=ProtocolError(
                "incompatible_protocol_major",
                f"client major {handshake.protocol.major}; server major {PROTOCOL_MAJOR}",
                retryable=False,
            ),
            schema_version=schema_version,
        )
    negotiated = tuple(cap for cap in server_capabilities if cap in handshake.capabilities)
    return HandshakeResponse(
        accepted=True,
        capabilities=negotiated,
        service_version=service_version,
        schema_version=schema_version,
    )


def encode_frame(frame: Frame, *, max_frame_size: int = MAX_FRAME_SIZE) -> bytes:
    """Canonically serialize exactly one JSON-lines frame."""
    if not isinstance(frame, (Handshake, HandshakeResponse, Request, Response)):
        raise ProtocolValidationError("unsupported frame object")
    if max_frame_size <= 0:
        raise ProtocolValidationError("max_frame_size must be positive")
    try:
        encoded = json.dumps(
            frame.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError, RecursionError) as exc:
        raise ProtocolValidationError(f"frame is not safely serializable: {exc}") from exc
    if len(encoded) > max_frame_size:
        raise FrameTooLargeError(f"frame is {len(encoded)} bytes; maximum is {max_frame_size}")
    return encoded


def _reject_constant(value: str) -> None:
    raise ProtocolValidationError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def decode_frame(
    frame: bytes | str,
    *,
    max_frame_size: int = MAX_FRAME_SIZE,
    refuse_major_mismatch: bool = True,
) -> Frame:
    """Safely decode one bounded JSON-lines frame with strict field validation."""
    if max_frame_size <= 0:
        raise ProtocolValidationError("max_frame_size must be positive")
    if isinstance(frame, str):
        try:
            raw = frame.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ProtocolValidationError("frame is not valid Unicode") from exc
    elif isinstance(frame, bytes):
        raw = frame
    else:
        raise ProtocolValidationError("frame must be bytes or string")
    if len(raw) > max_frame_size:
        raise FrameTooLargeError(f"frame is {len(raw)} bytes; maximum is {max_frame_size}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolValidationError("frame is not valid UTF-8") from exc
    if text.endswith("\r\n"):
        text = text[:-2]
    elif text.endswith("\n"):
        text = text[:-1]
    if not text or "\n" in text or "\r" in text:
        raise ProtocolValidationError("input must contain exactly one JSON-lines frame")
    try:
        data = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ProtocolValidationError(f"invalid JSON frame: {exc}") from exc
    data = _mapping(_validate_json(data, "frame"), "frame")
    frame_type = data.get("type", _MISSING)
    factories = {
        "handshake": Handshake.from_dict,
        "handshake_response": HandshakeResponse.from_dict,
        "request": Request.from_dict,
        "response": Response.from_dict,
    }
    factory = factories.get(frame_type)
    if factory is None:
        raise ProtocolValidationError(f"unknown frame type: {frame_type!r}")
    decoded = factory(data)
    if refuse_major_mismatch:
        require_compatible_major(decoded.protocol)
    return decoded
