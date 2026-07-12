"""Repository-packaged Enfold daemon application and lifecycle CLI.

The server deliberately never creates or migrates a database.  Operators must
provide an explicit JSON configuration and an existing, schema-v1 store.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import importlib.metadata
import json
import hashlib
import math
import os
from pathlib import Path
import signal
import socket
import sqlite3
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

from .daemon import DaemonConfig, UnixJsonDaemon
from .extraction_enqueue import ExtractionEnqueuer, ExtractionQueueUnavailable
from .extraction_processor import ExtractionProcessor
from .extraction_worker import SupervisedExtractionWorker
from .host_extractor import HostExtractorConfig, SubprocessHostExtractor
from .embedding_jobs import (
    EmbeddingJobProcessor, EmbeddingOutbox, EmbeddingSpec,
    SupervisedEmbeddingWorker,
)
from .embeddings import FastEmbedder, OllamaEmbedder
from .ollama_artifact import (
    DEFAULT_OLLAMA_BASE_URL,
    ArtifactAttestation,
    ArtifactAttestationError,
    ArtifactAttestor,
    LocalOllamaArtifactAttestor,
    require_sha256_digest,
)
from .hybrid_retrieval import (
    HybridRetriever,
    RetrieverFactory,
    SQLiteVersionedEmbeddingBackend,
    SQLiteStoredEmbeddingWriter,
    StoredEmbeddingError,
    VersionedStoredEmbeddingAdapter,
    deterministic_retriever_factory,
)
from .policy import MemoryPolicy, validate_scope
from .protocol import MAX_FRAME_SIZE, SUPPORTED_CAPABILITIES
from .read_pool import LRUQueryEmbedder, PerRequestReadHandler
from .schema import SUPPORTED_SCHEMA_VERSION, SchemaError, require_compatible_schema
from .service import EnfoldService


class ServerConfigError(ValueError):
    """The daemon configuration or one of its paths is unsafe or invalid."""


class DatabaseOwnershipError(ServerConfigError):
    """The database writer sidecar is already owned or unsafe."""


class DatabaseOwnership:
    """Exclusive, process-wide ownership of one database writer sidecar.

    The sidecar deliberately persists after shutdown.  Reusing one stable
    inode avoids the split-lock race caused by unlinking a lock file while
    another process may already have it open.
    """

    def __init__(self, database_path: Path):
        self.path = database_path.with_name(database_path.name + ".enfold.lock")
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            raise DatabaseOwnershipError("database ownership is already acquired")
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise DatabaseOwnershipError(f"cannot open database lock sidecar: {exc}") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise DatabaseOwnershipError(
                    "database lock sidecar must be a regular file owned by this user"
                )
            if info.st_mode & 0o022:
                raise DatabaseOwnershipError(
                    "database lock sidecar must not be group/world writable"
                )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DatabaseOwnershipError(
                    "another Enfold server owns this database"
                ) from exc
            payload = json.dumps(
                {"pid": os.getpid(), "database": str(self.path.name)},
                sort_keys=True,
            ).encode("utf-8")
            os.ftruncate(fd, 0)
            os.write(fd, payload)
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _version() -> str:
    try:
        return importlib.metadata.version("enfold")
    except importlib.metadata.PackageNotFoundError:
        return "0.7.0"


def _keys(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ServerConfigError(f"unknown {where} fields: {unknown}")


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ServerConfigError(f"{name} must be a positive number")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ServerConfigError(f"{name} must be a positive integer")
    return value


def _absolute_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ServerConfigError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ServerConfigError(f"{name} must be absolute")
    return path


@dataclass(frozen=True, slots=True)
class ServerConfig:
    database_path: Path
    socket_path: Path
    grants: Mapping[str, tuple[str, ...]]
    retrieval: "RetrievalConfig"
    extraction: "ExtractionConfig"
    browse_scopes: tuple[str, ...] = ("private",)
    correction_authorities: tuple[str, ...] = ()
    conflict_resolution_authorities: tuple[str, ...] = ()
    busy_timeout_ms: int = 5000
    client_timeout: float = 5.0
    shutdown_timeout: float = 5.0
    max_frame_bytes: int = MAX_FRAME_SIZE
    backlog: int = 16
    cleanup_stale_socket: bool = False
    synchronous_full: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Explicit retrieval activation configuration."""

    mode: str
    allow_nonproduction: bool = False
    provider: str | None = None
    model: str | None = None
    dimensions: int = 256
    query_identity: str | None = None
    document_identity: str | None = None
    embedding_version: str | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    prefix_policy: str | None = None
    model_fingerprint: str | None = None
    base_url: str | None = None
    timeout: float = 30.0
    keep_alive: int | str = -1
    cache_dir: str | None = None
    processor: Mapping[str, Any] | None = None
    vector_backend: str = "auto"
    near_dedup_enabled: bool = True


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Explicit automatic-extraction activation configuration."""

    mode: str = "disabled"
    host: HostExtractorConfig | None = None
    poll_seconds: float = 1.0
    drain_limit: int = 4
    lease_seconds: float = 300.0
    heartbeat_seconds: float = 30.0
    retry_delay_seconds: float = 1.0
    max_attempts: int = 3
    heartbeat_stale_seconds: float = 240.0
    pending_stale_seconds: float = 900.0


def _extraction_config(value: Any) -> ExtractionConfig:
    if value is None:
        return ExtractionConfig()
    if not isinstance(value, dict):
        raise ServerConfigError("extraction must be an object")
    allowed = {
        "mode", "host", "poll_seconds", "drain_limit", "lease_seconds",
        "heartbeat_seconds", "retry_delay_seconds", "max_attempts",
        "heartbeat_stale_seconds", "pending_stale_seconds",
    }
    _keys(value, allowed, "extraction")
    mode = value.get("mode")
    if mode == "disabled":
        if set(value) != {"mode"}:
            raise ServerConfigError("disabled extraction accepts only mode")
        return ExtractionConfig()
    if mode != "daemon-supervised":
        raise ServerConfigError(
            "extraction.mode must be 'disabled' or 'daemon-supervised'"
        )
    host = value.get("host")
    if not isinstance(host, dict):
        raise ServerConfigError("daemon-supervised extraction requires host")
    host_allowed = {
        "type", "argv", "model_identity", "prompt_identity",
        "timeout_seconds", "terminate_grace_seconds", "max_input_bytes",
        "max_output_bytes", "max_error_bytes", "environment",
    }
    _keys(host, host_allowed, "extraction.host")
    required = {"type", "argv", "model_identity", "prompt_identity"}
    missing = sorted(required - set(host))
    if missing:
        raise ServerConfigError(f"missing extraction.host fields: {missing}")
    if host["type"] != "subprocess":
        raise ServerConfigError("extraction.host.type must be 'subprocess'")
    argv = host["argv"]
    if not isinstance(argv, list) or not argv or not all(
        isinstance(item, str) for item in argv
    ):
        raise ServerConfigError("extraction.host.argv must be a non-empty string array")
    if not Path(argv[0]).is_absolute():
        raise ServerConfigError("extraction.host.argv[0] must be absolute")
    environment = host.get("environment", {})
    if not isinstance(environment, dict):
        raise ServerConfigError("extraction.host.environment must be an object")

    def number(name: str, default: float, *, allow_zero: bool = False) -> float:
        item = value.get(name, default)
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < (0 if allow_zero else 0.0)
            or (not allow_zero and float(item) == 0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ServerConfigError(f"extraction.{name} must be {qualifier} and finite")
        return float(item)

    def integer(name: str, default: int) -> int:
        item = value.get(name, default)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ServerConfigError(f"extraction.{name} must be a positive integer")
        return item

    try:
        host_config = HostExtractorConfig(
            argv=tuple(argv),
            model_identity=host["model_identity"],
            prompt_identity=host["prompt_identity"],
            timeout_seconds=_positive_number(
                host.get("timeout_seconds", 180.0),
                "extraction.host.timeout_seconds",
            ),
            terminate_grace_seconds=_positive_number(
                host.get("terminate_grace_seconds", 2.0),
                "extraction.host.terminate_grace_seconds",
            ),
            max_input_bytes=_positive_int(
                host.get("max_input_bytes", 16 * 1024),
                "extraction.host.max_input_bytes",
            ),
            max_output_bytes=_positive_int(
                host.get("max_output_bytes", 64 * 1024),
                "extraction.host.max_output_bytes",
            ),
            max_error_bytes=_positive_int(
                host.get("max_error_bytes", 16 * 1024),
                "extraction.host.max_error_bytes",
            ),
            environment=environment,
        )
    except (TypeError, ValueError) as exc:
        raise ServerConfigError(f"invalid extraction host configuration: {exc}") from exc
    lease_seconds = number("lease_seconds", 300.0)
    heartbeat_seconds = number("heartbeat_seconds", min(30.0, lease_seconds / 3))
    if heartbeat_seconds >= lease_seconds:
        raise ServerConfigError("extraction.heartbeat_seconds must be shorter than lease_seconds")
    stale_seconds = number(
        "heartbeat_stale_seconds",
        host_config.timeout_seconds + host_config.terminate_grace_seconds + 10.0,
    )
    if stale_seconds <= host_config.timeout_seconds + host_config.terminate_grace_seconds:
        raise ServerConfigError(
            "extraction.heartbeat_stale_seconds must exceed the host timeout and cleanup grace"
        )
    return ExtractionConfig(
        mode=mode,
        host=host_config,
        poll_seconds=number("poll_seconds", 1.0),
        drain_limit=integer("drain_limit", 4),
        lease_seconds=lease_seconds,
        heartbeat_seconds=heartbeat_seconds,
        retry_delay_seconds=number("retry_delay_seconds", 1.0, allow_zero=True),
        max_attempts=integer("max_attempts", 3),
        heartbeat_stale_seconds=stale_seconds,
        pending_stale_seconds=number("pending_stale_seconds", 900.0),
    )


def _retrieval_config(value: Any) -> RetrievalConfig:
    if not isinstance(value, dict):
        raise ServerConfigError("retrieval must be an object")
    allowed = {
        "mode", "allow_nonproduction", "provider", "model", "dimensions",
        "query_identity", "document_identity", "embedding_version",
        "query_prefix", "document_prefix", "prefix_policy", "model_fingerprint",
        "base_url", "timeout", "keep_alive", "cache_dir",
        "processor", "vector_backend", "near_dedup_enabled",
    }
    _keys(value, allowed, "retrieval")
    mode = value.get("mode")
    if mode not in {"ci", "stored"}:
        raise ServerConfigError("retrieval.mode must be 'ci' or 'stored'")
    allow_nonproduction = value.get("allow_nonproduction", False)
    if not isinstance(allow_nonproduction, bool):
        raise ServerConfigError("retrieval.allow_nonproduction must be a boolean")
    dimensions = _positive_int(value.get("dimensions", 256), "retrieval.dimensions")
    vector_backend = value.get("vector_backend", "auto")
    if vector_backend not in {"auto", "sqlite-vec", "brute"}:
        raise ServerConfigError(
            "retrieval.vector_backend must be auto, sqlite-vec, or brute"
        )
    near_dedup_enabled = value.get("near_dedup_enabled", True)
    if not isinstance(near_dedup_enabled, bool):
        raise ServerConfigError("retrieval.near_dedup_enabled must be a boolean")
    if mode == "ci":
        if not allow_nonproduction:
            raise ServerConfigError(
                "CI retrieval is non-production and requires "
                "retrieval.allow_nonproduction=true"
            )
        unexpected = sorted(
            set(value) - {
                "mode", "allow_nonproduction", "dimensions", "vector_backend",
                "near_dedup_enabled",
            }
        )
        if unexpected:
            raise ServerConfigError(f"CI retrieval has unsupported fields: {unexpected}")
        return RetrievalConfig(
            mode=mode,
            allow_nonproduction=True,
            dimensions=dimensions,
            vector_backend=vector_backend,
            near_dedup_enabled=near_dedup_enabled,
        )

    if allow_nonproduction:
        raise ServerConfigError("stored retrieval cannot set allow_nonproduction")
    required = {
        "provider", "model", "dimensions", "query_identity",
        "document_identity", "embedding_version", "model_fingerprint",
        "prefix_policy",
        "processor",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ServerConfigError(f"missing stored retrieval fields: {missing}")
    provider = value["provider"]
    if provider not in {"ollama", "fastembed"}:
        raise ServerConfigError("retrieval.provider must be 'ollama' or 'fastembed'")
    if provider == "fastembed":
        raise ServerConfigError(
            "daemon-supervised stored retrieval currently requires ollama; "
            "FastEmbed needs killable process isolation before activation"
        )

    def required_text(name: str) -> str:
        item = value[name]
        if not isinstance(item, str) or not item.strip():
            raise ServerConfigError(f"retrieval.{name} must be a non-empty string")
        return item

    query_identity = required_text("query_identity")
    document_identity = required_text("document_identity")
    embedding_version = required_text("embedding_version")
    model_fingerprint = required_text("model_fingerprint")
    prefix_policy = required_text("prefix_policy")
    processor = value["processor"]
    if not isinstance(processor, dict):
        raise ServerConfigError("retrieval.processor must be an object")
    _keys(processor, {"mode", "poll_seconds", "drain_limit", "lease_seconds",
                      "max_attempts", "heartbeat_stale_seconds",
                      "pending_stale_seconds"}, "retrieval.processor")
    if processor.get("mode") != "daemon-supervised":
        raise ServerConfigError("stored retrieval requires daemon-supervised processor")
    processor_defaults = {
        "poll_seconds": 1.0,
        "drain_limit": 8,
        "lease_seconds": 60,
        "max_attempts": 5,
        "heartbeat_stale_seconds": 10.0,
        "pending_stale_seconds": 300.0,
    }
    normalized_processor: dict[str, Any] = {"mode": "daemon-supervised"}
    for name, default in processor_defaults.items():
        item = processor.get(name, default)
        integer = name in {"drain_limit", "lease_seconds", "max_attempts"}
        if isinstance(item, bool) or not isinstance(item, int if integer else (int, float)):
            raise ServerConfigError(f"retrieval.processor.{name} has an invalid type")
        if float(item) <= 0 or not math.isfinite(float(item)):
            raise ServerConfigError(f"retrieval.processor.{name} must be positive and finite")
        normalized_processor[name] = int(item) if integer else float(item)
    try:
        model_fingerprint = require_sha256_digest(
            model_fingerprint, name="retrieval.model_fingerprint"
        )
        # Validate identity role mapping without opening a model or database.
        if query_identity.count(":query:") != 1:
            raise ValueError("query_identity must contain exactly one ':query:' role")
        if document_identity != query_identity.replace(":query:", ":document:"):
            raise ValueError(
                "document_identity must exactly match query_identity with its role "
                "changed from query to document"
            )
        if not query_identity.endswith(f":{embedding_version}"):
            raise ValueError(
                "embedding_version must be the final component of both stored "
                "embedding identities"
            )
        if model_fingerprint != embedding_version:
            raise ValueError("model_fingerprint must equal embedding_version")
    except (ArtifactAttestationError, ValueError) as exc:
        raise ServerConfigError(str(exc)) from exc
    query_prefix = value.get("query_prefix", "")
    document_prefix = value.get("document_prefix", "")
    if not isinstance(query_prefix, str) or not isinstance(document_prefix, str):
        raise ServerConfigError("retrieval query/document prefixes must be strings")
    if prefix_policy == "none":
        if query_prefix or document_prefix:
            raise ServerConfigError("none prefix policy requires empty prefixes")
    elif prefix_policy.startswith("sha256-"):
        digest = hashlib.sha256(
            f"{query_prefix}\0{document_prefix}".encode("utf-8")
        ).hexdigest()
        if prefix_policy != f"sha256-{digest}":
            raise ServerConfigError("prefix_policy does not match configured prefixes")
    else:
        raise ServerConfigError("prefix_policy must be none or sha256-<full digest>")
    expected_query = (
        f"{provider}:{required_text('model')}:query:{prefix_policy}:"
        f"{embedding_version}"
    )
    if query_identity != expected_query:
        raise ServerConfigError(
            "query/document identities must be canonically derived from provider, "
            "model, prefix policy, and model fingerprint"
        )
    base_url = value.get("base_url")
    cache_dir = value.get("cache_dir")
    for name, item in (("base_url", base_url), ("cache_dir", cache_dir)):
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise ServerConfigError(f"retrieval.{name} must be a non-empty string")
    if provider == "ollama" and cache_dir is not None:
        raise ServerConfigError("retrieval.cache_dir is only valid for fastembed")
    if provider == "fastembed" and base_url is not None:
        raise ServerConfigError("retrieval.base_url is only valid for ollama")
    keep_alive = value.get("keep_alive", -1)
    if isinstance(keep_alive, bool) or not isinstance(keep_alive, int | str):
        raise ServerConfigError("retrieval.keep_alive must be an integer or string")
    if provider == "fastembed" and "keep_alive" in value:
        raise ServerConfigError("retrieval.keep_alive is only valid for ollama")
    return RetrievalConfig(
        mode=mode,
        provider=provider,
        model=required_text("model"),
        dimensions=dimensions,
        query_identity=query_identity,
        document_identity=document_identity,
        embedding_version=embedding_version,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        prefix_policy=prefix_policy,
        model_fingerprint=model_fingerprint,
        base_url=base_url,
        timeout=_positive_number(value.get("timeout", 30.0), "retrieval.timeout"),
        keep_alive=keep_alive,
        cache_dir=cache_dir,
        processor=normalized_processor,
        vector_backend=vector_backend,
        near_dedup_enabled=near_dedup_enabled,
    )


def load_config(path: str | Path, *, allow_live: bool = False) -> ServerConfig:
    """Load and validate a strict JSON grants/configuration document."""

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        raise ServerConfigError("config path must be absolute")
    _guard_live_path(config_path, allow_live=allow_live)
    try:
        info = config_path.lstat()
    except FileNotFoundError as exc:
        raise ServerConfigError("config file does not exist") from exc
    if not stat.S_ISREG(info.st_mode) or config_path.is_symlink():
        raise ServerConfigError("config path must be a regular, non-symlink file")
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ServerConfigError("config file must be owned by this user and not group/world writable")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ServerConfigError(f"cannot read config JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ServerConfigError("config must be a JSON object")
    allowed = {
        "database_path", "socket_path", "grants", "busy_timeout_ms",
        "client_timeout", "shutdown_timeout", "max_frame_bytes", "backlog",
        "cleanup_stale_socket",
        "correction_authorities", "conflict_resolution_authorities",
        "retrieval", "extraction", "synchronous_full", "browse_scopes",
    }
    _keys(raw, allowed, "config")
    missing = sorted(
        {"database_path", "socket_path", "grants", "retrieval"} - set(raw)
    )
    if missing:
        raise ServerConfigError(f"missing config fields: {missing}")

    database = _absolute_path(raw["database_path"], "database_path")
    socket_path = _absolute_path(raw["socket_path"], "socket_path")
    _guard_live_path(database, allow_live=allow_live)
    _guard_live_path(socket_path, allow_live=allow_live)
    grants_raw = raw["grants"]
    if not isinstance(grants_raw, dict) or not grants_raw:
        raise ServerConfigError("grants must be a non-empty object")
    grants: dict[str, tuple[str, ...]] = {}
    for client_id, scopes in grants_raw.items():
        if not isinstance(client_id, str) or not client_id.strip():
            raise ServerConfigError("grant client ids must be non-empty strings")
        if not isinstance(scopes, list) or not all(isinstance(x, str) for x in scopes):
            raise ServerConfigError(f"grant {client_id!r} must be an array of scopes")
        grants[client_id] = tuple(scopes)
    def authorities(name: str) -> tuple[str, ...]:
        value = raw.get(name, [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ServerConfigError(f"{name} must be an array of client ids")
        result = tuple(dict.fromkeys(item.strip() for item in value))
        unknown = sorted(set(result) - set(grants))
        if unknown:
            raise ServerConfigError(f"{name} contains clients without grants: {unknown}")
        return result

    correction_authorities = authorities("correction_authorities")
    conflict_resolution_authorities = authorities(
        "conflict_resolution_authorities"
    )
    browse_raw = raw.get("browse_scopes", ["private"])
    if not isinstance(browse_raw, list) or not browse_raw:
        raise ServerConfigError("browse_scopes must be a non-empty array of scopes")
    try:
        browse_scopes = tuple(dict.fromkeys(validate_scope(scope) for scope in browse_raw))
    except (TypeError, ValueError) as exc:
        raise ServerConfigError("browse_scopes must contain supported scopes") from exc
    # Canonical policy validation catches invalid/empty scopes and client ids.
    try:
        MemoryPolicy(
            grants,
            correction_authorities=correction_authorities,
            conflict_resolution_authorities=conflict_resolution_authorities,
        )
    except ValueError as exc:
        raise ServerConfigError(str(exc)) from exc

    cleanup = raw.get("cleanup_stale_socket", False)
    if not isinstance(cleanup, bool):
        raise ServerConfigError("cleanup_stale_socket must be a boolean")
    synchronous_full = raw.get("synchronous_full", False)
    if not isinstance(synchronous_full, bool):
        raise ServerConfigError("synchronous_full must be a boolean")
    busy = raw.get("busy_timeout_ms", 5000)
    if isinstance(busy, bool) or not isinstance(busy, int) or busy < 0:
        raise ServerConfigError("busy_timeout_ms must be a non-negative integer")
    maximum = _positive_int(raw.get("max_frame_bytes", MAX_FRAME_SIZE), "max_frame_bytes")
    if maximum < 512:
        raise ServerConfigError("max_frame_bytes must be at least 512")
    return ServerConfig(
        database_path=database,
        socket_path=socket_path,
        grants=grants,
        retrieval=_retrieval_config(raw["retrieval"]),
        extraction=_extraction_config(raw.get("extraction")),
        browse_scopes=browse_scopes,
        correction_authorities=correction_authorities,
        conflict_resolution_authorities=conflict_resolution_authorities,
        busy_timeout_ms=busy,
        client_timeout=_positive_number(raw.get("client_timeout", 5.0), "client_timeout"),
        shutdown_timeout=_positive_number(raw.get("shutdown_timeout", 5.0), "shutdown_timeout"),
        max_frame_bytes=maximum,
        backlog=_positive_int(raw.get("backlog", 16), "backlog"),
        cleanup_stale_socket=cleanup,
        synchronous_full=synchronous_full,
    )


def _guard_live_path(path: Path, *, allow_live: bool) -> None:
    home_live = Path.home() / ".hermes"
    try:
        path.absolute().relative_to(home_live.absolute())
    except ValueError:
        return
    if not allow_live:
        raise ServerConfigError(
            f"refusing .hermes path without explicit --allow-live: {path}"
        )


def _validate_runtime_paths(config: ServerConfig) -> None:
    try:
        db_info = config.database_path.lstat()
    except FileNotFoundError as exc:
        raise ServerConfigError("database does not exist; the server never creates it") from exc
    if not stat.S_ISREG(db_info.st_mode) or config.database_path.is_symlink():
        raise ServerConfigError("database must be a regular, non-symlink file")
    parent = config.socket_path.parent
    try:
        parent_info = parent.lstat()
    except FileNotFoundError as exc:
        raise ServerConfigError("socket parent directory does not exist") from exc
    if not stat.S_ISDIR(parent_info.st_mode) or parent.is_symlink():
        raise ServerConfigError("socket parent must be a non-symlink directory")
    if parent_info.st_uid != os.getuid():
        raise ServerConfigError("socket parent must be owned by this user")
    if parent_info.st_mode & 0o022:
        raise ServerConfigError("socket parent must not be group/world writable")


def _open_existing_v1(
    config: ServerConfig, *, read_only: bool = False
) -> sqlite3.Connection:
    _validate_runtime_paths(config)
    mode = "ro" if read_only else "rw"
    uri = f"{config.database_path.as_uri()}?mode={mode}"
    conn = sqlite3.connect(
        uri,
        uri=True,
        timeout=config.busy_timeout_ms / 1000,
        check_same_thread=False,
    )
    try:
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={config.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
            raise ServerConfigError("database connection cannot enable foreign keys")
        version = require_compatible_schema(conn)
        if version != SUPPORTED_SCHEMA_VERSION:
            raise ServerConfigError(
                f"database must already be schema v{SUPPORTED_SCHEMA_VERSION}; found v{version}"
            )
        if not read_only:
            # The daemon is the sole intended writer. WAL permits concurrent
            # read snapshots while the handler lane serializes mutations.
            journal_mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0])
            if journal_mode.lower() != "wal":
                raise ServerConfigError("database cannot enable WAL journal mode")
            conn.execute(
                "PRAGMA synchronous=FULL"
                if config.synchronous_full
                else "PRAGMA synchronous=NORMAL"
            )
        return conn
    except BaseException:
        conn.close()
        raise


def _retriever_factory(
    config: RetrievalConfig, conn: sqlite3.Connection, query_embedder: Any = None
) -> RetrieverFactory:
    """Build the explicitly selected retrieval stack without probing a model."""

    if config.mode == "ci":
        # Parsing already required the conspicuous non-production opt-in.
        return deterministic_retriever_factory(
            dimensions=config.dimensions, vector_backend=config.vector_backend
        )

    if query_embedder is not None:
        pass
    elif config.provider == "ollama":
        kwargs: dict[str, Any] = {
            "model": config.model,
            "timeout": config.timeout,
            "keep_alive": config.keep_alive,
        }
        if config.base_url is not None:
            kwargs["base_url"] = config.base_url
        query_embedder = OllamaEmbedder(**kwargs)
    elif config.provider == "fastembed":
        query_embedder = FastEmbedder(
            model=str(config.model),
            cache_dir=config.cache_dir,
        )
    else:  # Defensive: RetrievalConfig can also be constructed directly.
        raise ServerConfigError("stored retrieval provider is unsupported")

    def make_backend(connection: sqlite3.Connection) -> SQLiteVersionedEmbeddingBackend:
        return SQLiteVersionedEmbeddingBackend(
            connection,
            query_embedder,
            query_identity=str(config.query_identity),
            document_identity=str(config.document_identity),
            embedding_version=str(config.embedding_version),
            dimensions=config.dimensions,
            query_prefix=config.query_prefix,
        )

    try:
        backend = make_backend(conn)
    except (StoredEmbeddingError, ValueError) as exc:
        raise ServerConfigError(f"stored retrieval is not ready: {exc}") from exc
    adapter = VersionedStoredEmbeddingAdapter(backend)

    def build(
        connection: sqlite3.Connection, scopes: Sequence[str]
    ) -> HybridRetriever:
        selected = adapter
        if connection is not conn:
            try:
                selected = VersionedStoredEmbeddingAdapter(make_backend(connection))
            except (StoredEmbeddingError, ValueError) as exc:
                raise RuntimeError(f"stored retrieval is not ready: {exc}") from exc
        return HybridRetriever(
            connection,
            selected,
            allowed_scopes=scopes,
            vector_backend=config.vector_backend,
        )

    return build


def _outbox(config: RetrievalConfig, conn: sqlite3.Connection) -> EmbeddingOutbox | None:
    if config.mode != "stored":
        return None
    return EmbeddingOutbox(
        conn,
        EmbeddingSpec(
            document_identity=str(config.document_identity),
            embedding_version=str(config.embedding_version),
            dimensions=config.dimensions,
            model_fingerprint=str(config.model_fingerprint),
            prefix_policy=str(config.prefix_policy),
            query_prefix=config.query_prefix,
            document_prefix=config.document_prefix,
        ),
    )


def _attest_artifact(
    config: RetrievalConfig, attestor: ArtifactAttestor | None = None
) -> ArtifactAttestation | None:
    """Fail closed before stored-mode startup or readiness is reported."""

    if config.mode != "stored":
        return None
    if config.provider != "ollama":
        raise ServerConfigError("stored artifact attestation requires Ollama")
    try:
        expected_digest = require_sha256_digest(
            config.model_fingerprint, name="retrieval.model_fingerprint"
        )
        active_attestor = attestor or LocalOllamaArtifactAttestor(
            base_url=config.base_url or DEFAULT_OLLAMA_BASE_URL,
            timeout=config.timeout,
        )
        result = active_attestor.attest(
            model=str(config.model), expected_digest=expected_digest
        )
    except ArtifactAttestationError as exc:
        raise ServerConfigError("Ollama artifact attestation failed") from exc
    except Exception as exc:
        raise ServerConfigError("Ollama artifact attestation failed") from exc
    if not isinstance(result, ArtifactAttestation) or result.provider != "ollama":
        raise ServerConfigError("Ollama artifact attestation failed")
    return result


def _artifact_state(attestation: ArtifactAttestation | None) -> dict[str, str]:
    return (
        {"status": "not_required"}
        if attestation is None
        else attestation.safe_state()
    )


class ServerApplication:
    """Own the database connection, service, and Unix daemon as one unit."""

    def __init__(
        self,
        config: ServerConfig,
        *,
        document_embedder: Any = None,
        artifact_attestor: ArtifactAttestor | None = None,
        extraction_extractor_factory: Callable[[HostExtractorConfig], Any] = (
            SubprocessHostExtractor
        ),
    ):
        self.config = config
        # Verify a mutable configured tag before opening the database, acquiring
        # a writer lifecycle, or scheduling any embedding backfill.
        self.artifact_attestation = _attest_artifact(
            config.retrieval, artifact_attestor
        )
        self.ownership = DatabaseOwnership(config.database_path)
        self.ownership.acquire()
        try:
            self.connection = _open_existing_v1(config)
            self.extraction_connection = None
            self.extraction_worker = None
            try:
                extraction_enqueuer = ExtractionEnqueuer(self.connection)
            except ExtractionQueueUnavailable:
                extraction_enqueuer = None
            embedding_outbox = _outbox(config.retrieval, self.connection)
            if embedding_outbox is not None:
                embedding_outbox.enqueue_backfill()
                outbox_health = embedding_outbox.health()
                if not outbox_health["activation_safe"]:
                    raise ServerConfigError(
                        "stored retrieval outbox is not activation-safe: "
                        f"{outbox_health}"
                    )
            query_embedder = None
            document_query_embedder = None
            if config.retrieval.mode == "stored":
                probe_factory = _retriever_factory(config.retrieval, self.connection)
                document_query_embedder = probe_factory(
                    self.connection, ("private",)
                )._embedder.backend._query_embedder
                query_embedder = LRUQueryEmbedder(
                    document_query_embedder,
                    model=str(config.retrieval.model),
                )
            if query_embedder is None:
                write_query_embedder = None
            else:
                def embed_write_content(content: str) -> Any:
                    return query_embedder.embed(
                        f"{config.retrieval.document_prefix}{content}"
                    )
                write_query_embedder = embed_write_content
            retriever_factory = _retriever_factory(
                config.retrieval, self.connection, query_embedder
            )
            self.embedding_connection = None
            self.embedding_worker = None
            if embedding_outbox is not None:
                self.embedding_connection = _open_existing_v1(config)
                worker_outbox = _outbox(config.retrieval, self.embedding_connection)
                embedder = document_embedder
                if embedder is None:
                    embedder = document_query_embedder
                writer = SQLiteStoredEmbeddingWriter(
                    self.embedding_connection, embedder,
                    document_identity=str(config.retrieval.document_identity),
                    embedding_version=str(config.retrieval.embedding_version),
                    model_fingerprint=str(config.retrieval.model_fingerprint),
                    prefix_policy=str(config.retrieval.prefix_policy),
                    dimensions=config.retrieval.dimensions,
                    document_prefix=config.retrieval.document_prefix,
                    query_prefix=config.retrieval.query_prefix,
                )
                p = dict(config.retrieval.processor or {})
                processor = EmbeddingJobProcessor(
                    worker_outbox, writer, worker_id="daemon",
                    lease_seconds=int(p.get("lease_seconds", 60)),
                    max_attempts=int(p.get("max_attempts", 5)),
                )
                self.embedding_worker = SupervisedEmbeddingWorker(
                    processor, poll_seconds=float(p.get("poll_seconds", 1.0)),
                    drain_limit=int(p.get("drain_limit", 8)),
                )
            policy = MemoryPolicy(
                config.grants,
                correction_authorities=config.correction_authorities,
                conflict_resolution_authorities=(
                    config.conflict_resolution_authorities
                ),
            )
            self.service = EnfoldService(
                self.connection,
                policy,
                retriever_factory=retriever_factory,
                embedding_outbox=embedding_outbox,
                extraction_enqueuer=extraction_enqueuer,
                extraction_processing_mode=config.extraction.mode,
                embedding_identity=config.retrieval.document_identity,
                query_embedder=write_query_embedder,
                near_dedup_enabled=config.retrieval.near_dedup_enabled,
            )
            request_handler = PerRequestReadHandler(
                self.service,
                lambda: _open_existing_v1(config, read_only=True),
                lambda read_connection: EnfoldService(
                    read_connection,
                    policy,
                    retriever_factory=_retriever_factory(
                        config.retrieval, read_connection, query_embedder
                    ),
                    extraction_enqueuer=None,
                    extraction_processing_mode=config.extraction.mode,
                ),
            )
            if config.extraction.mode == "daemon-supervised":
                if extraction_enqueuer is None or config.extraction.host is None:
                    raise ServerConfigError("automatic extraction queue is unavailable")
                self.extraction_connection = _open_existing_v1(config)
                extraction_retriever_factory = _retriever_factory(
                    config.retrieval, self.extraction_connection
                )
                extraction_outbox = _outbox(
                    config.retrieval, self.extraction_connection
                )
                extraction_service = EnfoldService(
                    self.extraction_connection,
                    policy,
                    retriever_factory=extraction_retriever_factory,
                    embedding_outbox=extraction_outbox,
                    extraction_enqueuer=None,
                    embedding_identity=config.retrieval.document_identity,
                    query_embedder=write_query_embedder,
                    near_dedup_enabled=config.retrieval.near_dedup_enabled,
                )
                try:
                    extractor = extraction_extractor_factory(config.extraction.host)
                    processor = ExtractionProcessor(
                        self.extraction_connection,
                        extraction_service,
                        extractor,
                        worker_id="daemon-extraction",
                        max_attempts=config.extraction.max_attempts,
                        lease_seconds=config.extraction.lease_seconds,
                        heartbeat_seconds=config.extraction.heartbeat_seconds,
                        retry_delay_seconds=config.extraction.retry_delay_seconds,
                    )
                    self.extraction_worker = SupervisedExtractionWorker(
                        processor,
                        poll_seconds=config.extraction.poll_seconds,
                        drain_limit=config.extraction.drain_limit,
                    )
                except Exception as exc:
                    raise ServerConfigError(
                        "automatic extraction initialization failed"
                    ) from exc
            def embedding_health() -> dict[str, Any]:
                if embedding_outbox is None or self.embedding_worker is None:
                    return {"mode": "disabled"}
                p = dict(config.retrieval.processor or {})
                outbox_state = embedding_outbox.health()
                worker_state = self.embedding_worker.health(
                    stale_after=float(p.get("heartbeat_stale_seconds", 10.0))
                )
                age = outbox_state.get("oldest_pending_age_seconds")
                pending_stale = age is not None and float(age) > float(
                    p.get("pending_stale_seconds", 300.0)
                )
                return {
                    **outbox_state,
                    "worker": worker_state,
                    "pending_stale": pending_stale,
                    "degraded": bool(
                        worker_state["heartbeat_stale"] or pending_stale
                        or worker_state["last_error"] is not None
                        or outbox_state["dead_letter"]
                    ),
                }

            def service_health(_context: Any) -> dict[str, Any]:
                embedding_state = embedding_health()
                if self.extraction_worker is None:
                    extraction_state: dict[str, Any] = {"status": "disabled"}
                else:
                    worker_state = self.extraction_worker.health(
                        stale_after=config.extraction.heartbeat_stale_seconds
                    )
                    rows = self.connection.execute(
                        "SELECT status, count(*) FROM extract_queue GROUP BY status"
                    ).fetchall()
                    counts = {str(row[0]): int(row[1]) for row in rows}
                    oldest = self.connection.execute(
                        "SELECT max(0, (julianday('now') - julianday(min(created_at))) "
                        "* 86400.0) FROM extract_queue "
                        "WHERE status IN ('pending', 'processing')"
                    ).fetchone()[0]
                    pending_stale = oldest is not None and float(oldest) > (
                        config.extraction.pending_stale_seconds
                    )
                    degraded = bool(
                        not worker_state["running"]
                        or worker_state["heartbeat_stale"]
                        or worker_state["last_error"] is not None
                        or counts.get("dead", 0)
                        or pending_stale
                    )
                    extraction_state = {
                        "status": "degraded" if degraded else "ready",
                        "worker": worker_state,
                        "queue": {
                            "pending": counts.get("pending", 0),
                            "processing": counts.get("processing", 0),
                            "dead": counts.get("dead", 0),
                            "oldest_active_age_seconds": oldest,
                            "pending_stale": pending_stale,
                        },
                    }
                return {
                    "status": (
                        "degraded" if (
                            embedding_state.get("degraded")
                            or extraction_state["status"] == "degraded"
                        ) else "ok"
                    ),
                    "storage": "sqlite",
                    "database": "ready",
                    "retrieval": self.service.retrieval_metadata,
                    "artifact_attestation": _artifact_state(
                        self.artifact_attestation
                    ),
                    "embedding_outbox": embedding_state,
                    "automatic_llm_extraction": extraction_state,
                    "extraction_enqueue": (
                        "ready" if extraction_enqueuer is not None else "unavailable"
                    ),
                }

            self.daemon = UnixJsonDaemon(
                DaemonConfig(
                    socket_path=config.socket_path,
                    server_version=_version(),
                    schema_version=SUPPORTED_SCHEMA_VERSION,
                    server_capabilities=SUPPORTED_CAPABILITIES,
                    max_frame_bytes=config.max_frame_bytes,
                    client_timeout=config.client_timeout,
                    shutdown_timeout=config.shutdown_timeout,
                    backlog=config.backlog,
                    cleanup_stale_socket=config.cleanup_stale_socket,
                ),
                request_handler,
                health_hook=service_health,
            )
        except BaseException:
            extraction_connection = getattr(self, "extraction_connection", None)
            if extraction_connection is not None:
                extraction_connection.close()
            embedding_connection = getattr(self, "embedding_connection", None)
            if embedding_connection is not None:
                embedding_connection.close()
            connection = getattr(self, "connection", None)
            if connection is not None:
                connection.close()
            self.ownership.release()
            raise
        self._closed = False

    def serve_forever(self) -> None:
        if self.embedding_worker is not None:
            self.embedding_worker.start()
        if self.extraction_worker is not None:
            self.extraction_worker.start()
        self.daemon.serve_forever()

    def close(self) -> None:
        if self._closed:
            return
        self.daemon.shutdown()
        if self.extraction_worker is not None:
            self.extraction_worker.stop(self.config.shutdown_timeout)
        if self.embedding_worker is not None:
            self.embedding_worker.stop(self.config.shutdown_timeout)
        if self.extraction_connection is not None:
            self.extraction_connection.close()
        if self.embedding_connection is not None:
            self.embedding_connection.close()
        self.connection.close()
        self.ownership.release()
        self._closed = True

    def __enter__(self) -> "ServerApplication":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def inspect_config(
    config: ServerConfig,
    *,
    probe_socket: bool = False,
    artifact_attestor: ArtifactAttestor | None = None,
) -> dict[str, Any]:
    """Validate the store and report non-sensitive readiness information."""

    artifact_attestation = _attest_artifact(config.retrieval, artifact_attestor)
    conn = _open_existing_v1(config, read_only=True)
    try:
        embedding_outbox = _outbox(config.retrieval, conn)
        outbox_health = (
            embedding_outbox.health()
            if embedding_outbox is not None else {"activation_safe": True}
        )
        if outbox_health["activation_safe"]:
            retriever_factory = _retriever_factory(config.retrieval, conn)
            retrieval = retriever_factory(conn, ("private",))
            retrieval_metadata = dict(retrieval.metadata)
        else:
            retrieval_metadata = {
                "embedder_production_ready": False,
                "activation_blocked": "embedding_outbox_unsafe",
            }
    finally:
        conn.close()
    socket_state = "absent"
    if config.socket_path.exists():
        socket_state = "present"
        if probe_socket:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.25)
            try:
                probe.connect(os.fspath(config.socket_path))
            except OSError:
                socket_state = "stale-or-unreachable"
            else:
                socket_state = "accepting"
            finally:
                probe.close()
    live_worker_unverified = bool(
        probe_socket
        and config.retrieval.mode == "stored"
        and socket_state == "accepting"
    )
    activation_blocked = (
        not bool(outbox_health["activation_safe"]) or live_worker_unverified
    )
    return {
        "status": "blocked" if activation_blocked else "ready",
        "service_version": _version(),
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "database": "compatible",
        "socket": socket_state,
        "grant_count": len(config.grants),
        "retrieval": retrieval_metadata,
        "artifact_attestation": _artifact_state(artifact_attestation),
        "embedding_outbox": outbox_health,
        "embedding_worker": {
            "state": (
                "live-health-unverified"
                if live_worker_unverified
                else "configured-ready-to-start"
            ) if config.retrieval.mode == "stored" else "disabled"
        },
        "automatic_llm_extraction": {
            "status": (
                "configured-ready-to-start"
                if config.extraction.mode == "daemon-supervised"
                else "disabled"
            )
        },
        "activation_blocker": (
            "CLI socket probing cannot attest the live embedding worker; "
            "query protocol health"
            if live_worker_unverified
            else "embedding outbox is unsafe" if activation_blocked else None
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="enfold-server")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="allow paths below ~/.hermes (required for an intentional live deployment)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate configuration and schema without binding")
    sub.add_parser("status", help="validate and probe the configured socket")
    sub.add_parser("run", help="run the foreground daemon")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config, allow_live=args.allow_live)
        if args.command in {"check", "status"}:
            report = inspect_config(
                config, probe_socket=args.command == "status"
            )
            print(json.dumps(report, sort_keys=True))
            return 0 if report["status"] == "ready" else 2
        application = ServerApplication(config)
    except (ServerConfigError, SchemaError, sqlite3.Error) as exc:
        print(f"enfold-server: {exc}", file=sys.stderr)
        return 2

    previous: dict[int, Any] = {}

    def stop(_signum: int, _frame: Any) -> None:
        application.daemon.request_shutdown()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
        application.serve_forever()
        return 0
    finally:
        application.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
