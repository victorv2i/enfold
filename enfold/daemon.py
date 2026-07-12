"""Hermes-independent Unix-socket JSON-lines daemon shell.

The daemon owns transport and lifecycle only. Storage, authorization, and
protocol routing are injected through a request handler. Mutating handler
execution is serialized into one in-process SQLite writer lane while reads
run concurrently on connections owned by the application.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import errno
import logging
import os
from pathlib import Path
import socket
import stat
import struct
import threading
import time
from typing import Any, Optional, Protocol

from .protocol import (
    MAX_FRAME_SIZE,
    PROTOCOL_MINOR,
    SUPPORTED_CAPABILITIES,
    ClientContext,
    Frame,
    FrameTooLargeError,
    Handshake,
    HandshakeResponse,
    ProtocolError,
    RequestHandlingError,
    ProtocolValidationError,
    ProtocolVersion,
    Request,
    Response,
    decode_frame,
    encode_frame,
    negotiate_handshake,
    required_capability,
)
from .schema import SUPPORTED_SCHEMA_VERSION


JsonObject = dict[str, Any]
RequestHandler = Callable[[ClientContext, Request], Any]
HealthHook = Callable[[ClientContext], Mapping[str, Any]]
PeerHook = Callable[["PeerCredentials"], None]


READ_ONLY_METHODS = frozenset({
    "memory.search",
    "memory.context",
    "memory.evidence",
    "memory.history",
    "memory.changes",
    "memory.timeline",
    "memory.entities",
    "memory.entity",
    "memory.conflicts",
})


logger = logging.getLogger(__name__)


class DaemonError(RuntimeError):
    """Base daemon lifecycle error."""


class SocketPathError(DaemonError):
    """The configured socket path cannot be safely claimed."""


class DaemonAlreadyRunning(SocketPathError):
    """Another process is accepting connections on the socket."""


class RequestError(RequestHandlingError):
    """A public, structured request failure raised by a router."""

    def __init__(self, code: str, message: str):
        typed = ProtocolError(code, message)
        super().__init__(typed.code, typed.message)


@dataclass(frozen=True, slots=True)
class PeerCredentials:
    """Kernel-authenticated identity for a local Unix-socket peer."""

    pid: int
    uid: int
    gid: int


def inspect_peer_credentials(client: socket.socket) -> PeerCredentials | None:
    """Return Linux ``SO_PEERCRED`` data, or ``None`` when unavailable.

    The value comes from the kernel, not the protocol handshake.  It is an
    additional local-process audit signal; agent attribution remains the
    handshake's job.
    """

    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        return None
    try:
        raw = client.getsockopt(socket.SOL_SOCKET, option, struct.calcsize("3i"))
    except OSError:
        return None
    pid, uid, gid = struct.unpack("3i", raw)
    return PeerCredentials(pid=pid, uid=uid, gid=gid)


class FrameCodec(Protocol):
    """Narrow boundary for later integration with the shared protocol module."""

    def decode(self, frame: bytes) -> Frame: ...

    def encode(self, message: Frame) -> bytes: ...


class ProtocolCodec:
    """Adapter around Enfold's canonical typed JSON-lines protocol."""

    def __init__(self, max_frame_bytes: int):
        self._max_frame_bytes = max_frame_bytes

    def decode(self, frame: bytes) -> Frame:
        return decode_frame(
            frame,
            max_frame_size=self._max_frame_bytes,
            # Handshake major mismatches must receive an explicit refusal.
            refuse_major_mismatch=False,
        )

    def encode(self, message: Frame) -> bytes:
        return encode_frame(message, max_frame_size=self._max_frame_bytes)


@dataclass(frozen=True, slots=True)
class DaemonConfig:
    """Explicit transport configuration; there is intentionally no DB path."""

    socket_path: Path
    server_version: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION
    server_capabilities: tuple[str, ...] = SUPPORTED_CAPABILITIES
    max_frame_bytes: int = MAX_FRAME_SIZE
    client_timeout: float = 5.0
    shutdown_timeout: float = 5.0
    backlog: int = 16
    cleanup_stale_socket: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "socket_path", Path(self.socket_path))
        if not self.server_version.strip():
            raise ValueError("server_version must not be empty")
        if self.schema_version < 0:
            raise ValueError("schema_version must not be negative")
        object.__setattr__(self, "server_capabilities", tuple(self.server_capabilities))
        if self.max_frame_bytes < 512:
            raise ValueError("max_frame_bytes must be at least 512")
        if self.client_timeout <= 0 or self.shutdown_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if self.backlog <= 0:
            raise ValueError("backlog must be positive")


@dataclass(frozen=True, slots=True)
class _NegotiatedConnection:
    context: ClientContext
    capabilities: tuple[str, ...]
    protocol: ProtocolVersion
    schema_version: int


class UnixJsonDaemon:
    """Bounded local JSON-lines server with serialized mutation execution."""

    def __init__(
        self,
        config: DaemonConfig,
        handler: RequestHandler,
        *,
        health_hook: Optional[HealthHook] = None,
        peer_hook: Optional[PeerHook] = None,
        codec: Optional[FrameCodec] = None,
    ):
        self.config = config
        self._handler = handler
        self._health_hook = health_hook
        self._peer_hook = peer_hook
        self._codec = codec or ProtocolCodec(config.max_frame_bytes)
        self._listener: Optional[socket.socket] = None
        self._bound_identity: Optional[tuple[int, int]] = None
        self._stop = threading.Event()
        self._handler_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._clients_lock = threading.Lock()
        self._clients: set[socket.socket] = set()
        self._client_threads: set[threading.Thread] = set()
        self._serve_thread: Optional[threading.Thread] = None

    @property
    def socket_path(self) -> Path:
        return self.config.socket_path

    def start(self) -> None:
        """Bind the socket and serve in a background thread."""

        with self._state_lock:
            if self._listener is not None or self._serve_thread is not None:
                raise DaemonError("daemon is already started")
            self._bind()
            thread = threading.Thread(
                target=self._accept_loop,
                name="enfold-daemon",
                daemon=True,
            )
            self._serve_thread = thread
            thread.start()

    def serve_forever(self) -> None:
        """Bind and serve on the current thread until shutdown."""

        with self._state_lock:
            if self._listener is not None or self._serve_thread is not None:
                raise DaemonError("daemon is already started")
            self._bind()
        try:
            self._accept_loop()
        finally:
            self._close_listener()
            self._unlink_owned_socket()

    def shutdown(self) -> None:
        """Stop accepting, close clients, and remove only our own socket inode."""

        self._stop.set()
        self._close_listener()
        with self._clients_lock:
            clients = tuple(self._clients)
            threads = tuple(self._client_threads)
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass

        deadline = time.monotonic() + self.config.shutdown_timeout
        serve_thread = self._serve_thread
        if serve_thread is not None and serve_thread is not threading.current_thread():
            serve_thread.join(max(0.0, deadline - time.monotonic()))
        for thread in threads:
            if thread is threading.current_thread():
                continue
            thread.join(max(0.0, deadline - time.monotonic()))
        self._unlink_owned_socket()

    def request_shutdown(self) -> None:
        """Signal the serve loop without taking locks or joining threads.

        This tiny operation is safe to call from a Python signal handler.  The
        accept loop's bounded timeout observes the event and normal teardown
        happens outside the signal handler.
        """

        self._stop.set()

    def __enter__(self) -> "UnixJsonDaemon":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()

    def _bind(self) -> None:
        path = self.socket_path
        if not path.is_absolute():
            raise SocketPathError("socket_path must be absolute")
        if not path.parent.is_dir():
            raise SocketPathError("socket parent directory does not exist")
        self._prepare_socket_path(path)

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound_identity: Optional[tuple[int, int]] = None
        try:
            listener.bind(os.fspath(path))
            # Capture ownership immediately after bind. If a later setup step
            # fails, remove only this inode, never a path another process won
            # in a bind race.
            bound_info = path.lstat()
            bound_identity = (bound_info.st_dev, bound_info.st_ino)
            os.chmod(path, 0o600)
            listener.listen(self.config.backlog)
            listener.settimeout(0.2)
            info = path.lstat()
            if (info.st_dev, info.st_ino) != bound_identity:
                raise SocketPathError("socket path changed during bind")
        except BaseException:
            listener.close()
            if bound_identity is not None:
                self._unlink_socket_identity(path, bound_identity)
            raise
        self._listener = listener
        self._bound_identity = bound_identity

    def _prepare_socket_path(self, path: Path) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(info.st_mode):
            raise SocketPathError("socket path exists and is not a Unix socket")

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(min(0.25, self.config.client_timeout))
        try:
            probe.connect(os.fspath(path))
        except OSError as exc:
            stale = exc.errno in {errno.ECONNREFUSED, errno.ENOENT}
            if not stale:
                raise SocketPathError(
                    f"cannot safely inspect existing socket: {exc}"
                ) from exc
            if not self.config.cleanup_stale_socket:
                raise SocketPathError(
                    "stale socket exists; explicit cleanup_stale_socket is required"
                ) from exc
            # Refuse a path-swap race: unlink only the inode inspected above.
            current = path.lstat()
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                raise SocketPathError("socket path changed during stale check")
            path.unlink()
        else:
            raise DaemonAlreadyRunning("another daemon is accepting on socket")
        finally:
            probe.close()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                client, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set() or self._listener is None:
                    return
                raise
            peer = inspect_peer_credentials(client)
            if peer is not None:
                if peer.uid != os.getuid():
                    logger.warning(
                        "refusing Enfold socket peer pid=%s uid=%s gid=%s",
                        peer.pid,
                        peer.uid,
                        peer.gid,
                    )
                    client.close()
                    continue
                logger.debug(
                    "accepted Enfold socket peer pid=%s uid=%s gid=%s",
                    peer.pid,
                    peer.uid,
                    peer.gid,
                )
                if self._peer_hook is not None:
                    try:
                        self._peer_hook(peer)
                    except Exception:
                        logger.exception("Enfold peer audit hook failed")
            client.settimeout(self.config.client_timeout)
            thread = threading.Thread(
                target=self._serve_client,
                args=(client,),
                name="enfold-daemon-client",
                daemon=True,
            )
            with self._clients_lock:
                if self._stop.is_set():
                    client.close()
                    return
                self._clients.add(client)
                self._client_threads.add(thread)
                thread.start()

    def _serve_client(self, client: socket.socket) -> None:
        buffer = bytearray()
        connection: Optional[_NegotiatedConnection] = None
        try:
            while not self._stop.is_set():
                try:
                    chunk = client.recv(min(65536, self.config.max_frame_bytes + 1))
                except socket.timeout:
                    self._send_protocol_error(
                        client,
                        connection,
                        "request_timeout",
                        "request timed out",
                    )
                    return
                except OSError:
                    return
                if not chunk:
                    return
                buffer.extend(chunk)
                while True:
                    newline = buffer.find(b"\n")
                    if newline < 0:
                        if len(buffer) > self.config.max_frame_bytes:
                            self._send_protocol_error(
                                client,
                                connection,
                                "frame_too_large",
                                "request frame exceeds configured limit",
                            )
                            return
                        break
                    frame = bytes(buffer[:newline])
                    del buffer[: newline + 1]
                    if len(frame) > self.config.max_frame_bytes:
                        self._send_protocol_error(
                            client,
                            connection,
                            "frame_too_large",
                            "request frame exceeds configured limit",
                        )
                        return
                    keep_open, connection = self._handle_frame(
                        client, frame, connection
                    )
                    if not keep_open:
                        return
        finally:
            try:
                client.close()
            except OSError:
                pass
            with self._clients_lock:
                self._clients.discard(client)
                self._client_threads.discard(threading.current_thread())

    def _handle_frame(
        self,
        client: socket.socket,
        frame: bytes,
        connection: Optional[_NegotiatedConnection],
    ) -> tuple[bool, Optional[_NegotiatedConnection]]:
        try:
            decoded = self._codec.decode(frame)
        except (ProtocolValidationError, FrameTooLargeError) as exc:
            self._send_protocol_error(
                client,
                connection,
                "invalid_frame",
                str(exc),
            )
            return False, connection

        if connection is None:
            if not isinstance(decoded, Handshake):
                self._send_handshake_refusal(
                    client,
                    "handshake_required",
                    "the first connection frame must be a handshake",
                )
                return False, None
            response, negotiated = self._negotiate(decoded)
            sent = self._send(client, response)
            return bool(sent and response.accepted), negotiated

        if not isinstance(decoded, Request):
            self._send_protocol_error(
                client,
                connection,
                "request_required",
                "only request frames are allowed after handshake",
            )
            return False, connection

        incompatibility = self._request_incompatibility(connection, decoded)
        if incompatibility is not None:
            code, message = incompatibility
            self._send_response_error(client, decoded.request_id, code, message)
            return False, connection

        capability = required_capability(decoded.method)
        if capability not in connection.capabilities:
            self._send_response_error(
                client,
                decoded.request_id,
                "capability_not_negotiated",
                f"method requires negotiated capability: {capability}",
            )
            return True, connection

        try:
            if decoded.method == "health":
                result = self._health(connection.context)
            else:
                # Unknown/future methods fail safe into the exclusive lane;
                # only explicitly classified reads may execute concurrently.
                if decoded.method in READ_ONLY_METHODS:
                    result = self._handler(connection.context, decoded)
                else:
                    with self._handler_lock:
                        result = self._handler(connection.context, decoded)
            response = Response(
                decoded.request_id,
                True,
                result=result,
                schema_version=self.config.schema_version,
            )
        except RequestHandlingError as exc:
            response = Response(
                decoded.request_id,
                False,
                error=ProtocolError(exc.code, exc.message),
                schema_version=self.config.schema_version,
            )
        except Exception:
            response = Response(
                decoded.request_id,
                False,
                error=ProtocolError("internal_error", "request handler failed"),
                schema_version=self.config.schema_version,
            )
        return self._send(client, response), connection

    def _negotiate(
        self, handshake: Handshake
    ) -> tuple[HandshakeResponse, Optional[_NegotiatedConnection]]:
        response = negotiate_handshake(
            handshake,
            service_version=self.config.server_version,
            server_capabilities=self.config.server_capabilities,
            schema_version=self.config.schema_version,
        )
        if not response.accepted:
            return response, None
        if handshake.protocol.minor > PROTOCOL_MINOR:
            return self._handshake_refusal(
                "incompatible_protocol_minor",
                f"client minor {handshake.protocol.minor}; server minor {PROTOCOL_MINOR}",
            ), None
        if handshake.schema_version != self.config.schema_version:
            return self._handshake_refusal(
                "incompatible_schema",
                f"client schema {handshake.schema_version}; server schema "
                f"{self.config.schema_version}",
            ), None
        return response, _NegotiatedConnection(
            context=handshake.context,
            capabilities=response.capabilities,
            protocol=handshake.protocol,
            schema_version=handshake.schema_version,
        )

    @staticmethod
    def _request_incompatibility(
        connection: _NegotiatedConnection, request: Request
    ) -> Optional[tuple[str, str]]:
        if request.protocol != connection.protocol:
            return (
                "connection_protocol_changed",
                "request protocol differs from the accepted handshake",
            )
        if request.schema_version != connection.schema_version:
            return (
                "connection_schema_changed",
                "request schema differs from the accepted handshake",
            )
        return None

    def _health(self, context: ClientContext) -> JsonObject:
        result: JsonObject = {
            "status": "ok",
            "service_version": self.config.server_version,
            "protocol": ProtocolVersion().to_dict(),
            "schema_version": self.config.schema_version,
        }
        if self._health_hook is not None:
            extension = self._health_hook(context)
            if not isinstance(extension, Mapping):
                raise TypeError("health hook must return a mapping")
            result.update(extension)
            # Handshake identity cannot be overridden by a hook.
            result["service_version"] = self.config.server_version
            result["protocol"] = ProtocolVersion().to_dict()
            result["schema_version"] = self.config.schema_version
        return result

    def _handshake_refusal(self, code: str, message: str) -> HandshakeResponse:
        return HandshakeResponse(
            accepted=False,
            capabilities=(),
            service_version=self.config.server_version,
            schema_version=self.config.schema_version,
            error=ProtocolError(code, message),
        )

    def _send_handshake_refusal(
        self, client: socket.socket, code: str, message: str
    ) -> bool:
        return self._send(client, self._handshake_refusal(code, message))

    def _send_response_error(
        self,
        client: socket.socket,
        request_id: str,
        code: str,
        message: str,
    ) -> bool:
        return self._send(
            client,
            Response(
                request_id,
                False,
                error=ProtocolError(code, message),
                schema_version=self.config.schema_version,
            ),
        )

    def _send_protocol_error(
        self,
        client: socket.socket,
        connection: Optional[_NegotiatedConnection],
        code: str,
        message: str,
    ) -> bool:
        if connection is None:
            return self._send_handshake_refusal(client, code, message)
        return self._send_response_error(client, "invalid", code, message)

    def _send(self, client: socket.socket, message: Frame) -> bool:
        try:
            encoded = self._codec.encode(message)
            client.sendall(encoded)
            return True
        except FrameTooLargeError:
            if isinstance(message, Response):
                fallback: Frame = Response(
                    message.request_id,
                    False,
                    error=ProtocolError("response_too_large", "response too large"),
                    schema_version=self.config.schema_version,
                )
            else:
                fallback = self._handshake_refusal(
                    "response_too_large", "response too large"
                )
            try:
                client.sendall(self._codec.encode(fallback))
                return True
            except (OSError, ProtocolValidationError):
                return False
        except (OSError, ProtocolValidationError):
            return False

    def _close_listener(self) -> None:
        with self._state_lock:
            listener = self._listener
            self._listener = None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

    def _unlink_owned_socket(self) -> None:
        identity = self._bound_identity
        if identity is None:
            return
        path = self.socket_path
        try:
            info = path.lstat()
        except FileNotFoundError:
            self._bound_identity = None
            return
        if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == identity:
            path.unlink()
        self._bound_identity = None

    @staticmethod
    def _unlink_socket_identity(path: Path, identity: tuple[int, int]) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == identity:
            path.unlink()
