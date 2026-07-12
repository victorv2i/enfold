"""Typed, short-lived client for Enfold's local Unix-socket protocol.

Each request opens a connection, negotiates immutable client context, sends one
request, and closes the connection.  This deliberately avoids keeping a socket
past the daemon's idle timeout and makes reconnect behavior explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
from typing import Any
from uuid import uuid4

from .protocol import (
    MAX_FRAME_SIZE,
    SUPPORTED_CAPABILITIES,
    ClientContext,
    Frame,
    Handshake,
    HandshakeResponse,
    ProtocolError,
    ProtocolValidationError,
    ProtocolVersion,
    Request,
    Response,
    decode_frame,
    encode_frame,
    required_capability,
)
from .schema import SUPPORTED_SCHEMA_VERSION


class EnfoldClientError(RuntimeError):
    """Base class for safe, public client failures."""


class EnfoldTransportError(EnfoldClientError):
    """The local daemon could not be reached or completed the exchange."""


class EnfoldProtocolError(EnfoldClientError):
    """The daemon returned an invalid or unexpected protocol exchange."""


class EnfoldRemoteError(EnfoldClientError):
    """A typed error returned by the Enfold daemon."""

    def __init__(self, error: ProtocolError, *, request_id: str | None = None):
        super().__init__(error.message)
        self.code = error.code
        self.message = error.message
        self.retryable = error.retryable
        self.details = dict(error.details)
        self.request_id = request_id


class EnfoldHandshakeError(EnfoldRemoteError):
    """The daemon explicitly refused the client handshake."""


@dataclass(frozen=True, slots=True)
class ClientConfig:
    socket_path: Path
    context: ClientContext
    capabilities: tuple[str, ...] = SUPPORTED_CAPABILITIES
    protocol: ProtocolVersion = ProtocolVersion()
    schema_version: int = SUPPORTED_SCHEMA_VERSION
    connect_timeout: float = 2.0
    request_timeout: float = 5.0
    max_frame_bytes: int = MAX_FRAME_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(self, "socket_path", Path(self.socket_path))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        if not self.socket_path.is_absolute():
            raise ValueError("socket_path must be absolute")
        if not isinstance(self.context, ClientContext):
            raise TypeError("context must be ClientContext")
        if not isinstance(self.protocol, ProtocolVersion):
            raise TypeError("protocol must be ProtocolVersion")
        if self.schema_version < 0:
            raise ValueError("schema_version must not be negative")
        if self.connect_timeout <= 0 or self.request_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_frame_bytes < 512:
            raise ValueError("max_frame_bytes must be at least 512")
        # Reuse canonical validation for duplicate or malformed capabilities.
        validated = Handshake(
            self.context,
            capabilities=self.capabilities,
            protocol=self.protocol,
            schema_version=self.schema_version,
        )
        object.__setattr__(self, "capabilities", validated.capabilities)


class EnfoldClient:
    """Reconnect-per-request Unix client with typed negotiation and errors."""

    def __init__(self, config: ClientConfig):
        self.config = config

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> Any:
        request = Request(
            request_id or self._new_request_id(),
            method,
            {} if params is None else params,
            protocol=self.config.protocol,
            schema_version=self.config.schema_version,
        )
        capability = required_capability(request.method)
        if capability not in self.config.capabilities:
            raise EnfoldProtocolError(
                f"client did not request capability required by {method}: {capability}"
            )

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(self.config.connect_timeout)
            client.connect(os.fspath(self.config.socket_path))
            client.settimeout(self.config.request_timeout)
            self._send(client, self._handshake())
            hello = self._receive(client)
            if not isinstance(hello, HandshakeResponse):
                raise EnfoldProtocolError("expected handshake response from daemon")
            self._validate_handshake(hello, capability)

            self._send(client, request)
            response = self._receive(client)
            if not isinstance(response, Response):
                raise EnfoldProtocolError("expected response frame from daemon")
            self._validate_response(response, request)
            if not response.ok:
                assert response.error is not None
                raise EnfoldRemoteError(response.error, request_id=response.request_id)
            return response.result
        except EnfoldClientError:
            raise
        except (OSError, socket.timeout) as exc:
            raise EnfoldTransportError(f"Enfold daemon transport failed: {exc}") from exc
        except ProtocolValidationError as exc:
            raise EnfoldProtocolError(f"invalid daemon protocol frame: {exc}") from exc
        finally:
            client.close()

    def _handshake(self) -> Handshake:
        return Handshake(
            context=self.config.context,
            capabilities=self.config.capabilities,
            protocol=self.config.protocol,
            schema_version=self.config.schema_version,
        )

    def _send(self, client: socket.socket, frame: Frame) -> None:
        client.sendall(
            encode_frame(frame, max_frame_size=self.config.max_frame_bytes)
        )

    def _receive(self, client: socket.socket) -> Frame:
        data = bytearray()
        while True:
            chunk = client.recv(min(65536, self.config.max_frame_bytes + 1))
            if not chunk:
                raise EnfoldTransportError("Enfold daemon closed the connection")
            data.extend(chunk)
            newline = data.find(b"\n")
            if newline >= 0:
                if newline + 1 != len(data):
                    raise EnfoldProtocolError("daemon sent multiple frames unexpectedly")
                return decode_frame(
                    bytes(data), max_frame_size=self.config.max_frame_bytes
                )
            if len(data) >= self.config.max_frame_bytes:
                raise EnfoldProtocolError("daemon frame exceeds configured limit")

    def _validate_handshake(
        self, response: HandshakeResponse, required: str
    ) -> None:
        if not response.accepted:
            assert response.error is not None
            raise EnfoldHandshakeError(response.error)
        if response.protocol != self.config.protocol:
            raise EnfoldProtocolError("daemon negotiated an unexpected protocol version")
        if response.schema_version != self.config.schema_version:
            raise EnfoldProtocolError("daemon negotiated an unexpected schema version")
        if required not in response.capabilities:
            raise EnfoldProtocolError(
                f"daemon did not negotiate required capability: {required}"
            )

    def _validate_response(self, response: Response, request: Request) -> None:
        if response.request_id != request.request_id:
            raise EnfoldProtocolError("daemon response request_id does not match request")
        if response.protocol != self.config.protocol:
            raise EnfoldProtocolError("daemon response protocol differs from handshake")
        if response.schema_version != self.config.schema_version:
            raise EnfoldProtocolError("daemon response schema differs from handshake")

    @staticmethod
    def _new_request_id() -> str:
        return f"req-{uuid4().hex}"
