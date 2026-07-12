"""Additive provenance schema and immutable write-boundary value objects.

This module deliberately has no dependency on the Hermes provider.  It is the
small persistence contract shared by future in-process and daemon writers.
Schema installation is explicit; importing the module never mutates a store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sqlite3
from typing import Optional


PROVENANCE_SCHEMA_VERSION = 1


def _required(value: str, name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _canonical_json_object(value: str, name: str) -> str:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be a JSON object") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object")
    return json.dumps(decoded, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ConnectionContext:
    """Identity and development context established by a trusted adapter."""

    client_id: str
    surface: str
    agent_id: str
    session_id: str
    display_name: Optional[str] = None
    parent_agent_id: Optional[str] = None
    project_root: Optional[str] = None
    repository: Optional[str] = None
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    access_scopes: tuple[str, ...] = ("private",)
    started_at: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("client_id", "surface", "agent_id", "session_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        capabilities = tuple(sorted({
            _required(value, "capability") for value in self.capabilities
        }))
        scopes = tuple(sorted({
            _required(value, "access_scope") for value in self.access_scopes
        }))
        if not scopes:
            raise ValueError("access_scopes must not be empty")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "access_scopes", scopes)


@dataclass(frozen=True, slots=True)
class WriteRequest:
    """One durable fact write and the observation supporting it.

    Attribution such as ``recorded_by`` is intentionally absent: the service
    derives it from :class:`ConnectionContext`.
    """

    idempotency_key: str
    content: str
    source_type: str
    category: str = "general"
    tags: str = ""
    trust_score: float = 0.5
    source_authority: float = 0.5
    source_uri: Optional[str] = None
    observation_content: Optional[str] = None
    asserted_by: Optional[str] = None
    performed_by: Optional[str] = None
    observed_at: Optional[str] = None
    scope: str = "private"
    sensitivity: str = "normal"
    correction_status: Optional[str] = None
    evidence_excerpt: Optional[str] = None
    relation: str = "supports"
    metadata_json: str = "{}"
    supersede_fact_id: Optional[int] = None
    operation: str = "add_fact"

    def __post_init__(self) -> None:
        for name in (
            "idempotency_key",
            "content",
            "source_type",
            "category",
            "scope",
            "sensitivity",
            "relation",
            "operation",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        if not 0.0 <= self.trust_score <= 1.0:
            raise ValueError("trust_score must be between 0 and 1")
        if not 0.0 <= self.source_authority <= 1.0:
            raise ValueError("source_authority must be between 0 and 1")
        if self.supersede_fact_id is not None and self.supersede_fact_id <= 0:
            raise ValueError("supersede_fact_id must be positive")
        object.__setattr__(
            self,
            "metadata_json",
            _canonical_json_object(self.metadata_json, "metadata_json"),
        )


@dataclass(frozen=True, slots=True)
class WriteOutcome:
    """Stable result persisted for idempotent replay."""

    write_id: str
    outcome: str
    fact_id: Optional[int]
    existing_fact_id: Optional[int] = None
    observation_id: Optional[int] = None
    replayed: bool = False
    detail_json: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "write_id", _required(self.write_id, "write_id"))
        object.__setattr__(self, "outcome", _required(self.outcome, "outcome"))
        for name in ("fact_id", "existing_fact_id", "observation_id"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        object.__setattr__(
            self,
            "detail_json",
            _canonical_json_object(self.detail_json, "detail_json"),
        )


_PROVENANCE_SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS memory_clients (
        client_id TEXT PRIMARY KEY,
        surface TEXT NOT NULL,
        display_name TEXT,
        created_at TEXT NOT NULL,
        disabled_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS memory_sessions (
        session_id TEXT NOT NULL,
        client_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        parent_agent_id TEXT,
        project_root TEXT,
        repository TEXT,
        branch TEXT,
        commit_sha TEXT,
        capabilities_json TEXT NOT NULL DEFAULT '[]',
        access_scopes_json TEXT NOT NULL DEFAULT '["private"]',
        started_at TEXT,
        ended_at TEXT,
        PRIMARY KEY (client_id, session_id),
        FOREIGN KEY (client_id) REFERENCES memory_clients(client_id)
    )""",
    """CREATE TABLE IF NOT EXISTS observations (
        observation_id INTEGER PRIMARY KEY,
        client_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        source_type TEXT NOT NULL,
        source_uri TEXT,
        project_root TEXT,
        repository TEXT,
        branch TEXT,
        commit_sha TEXT,
        content TEXT,
        content_sha256 TEXT NOT NULL,
        asserted_by TEXT,
        performed_by TEXT,
        observed_at TEXT,
        recorded_at TEXT NOT NULL,
        scope TEXT NOT NULL DEFAULT 'private',
        sensitivity TEXT NOT NULL DEFAULT 'normal',
        redacted_at TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE (client_id, content_sha256, session_id, source_type),
        FOREIGN KEY (client_id, session_id)
            REFERENCES memory_sessions(client_id, session_id)
    )""",
    """CREATE TABLE IF NOT EXISTS fact_provenance (
        fact_id INTEGER NOT NULL,
        observation_id INTEGER NOT NULL,
        relation TEXT NOT NULL DEFAULT 'supports',
        evidence_excerpt TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (fact_id, observation_id, relation),
        FOREIGN KEY (fact_id) REFERENCES facts(fact_id),
        FOREIGN KEY (observation_id) REFERENCES observations(observation_id)
    )""",
    """CREATE TABLE IF NOT EXISTS memory_write_log (
        write_id TEXT PRIMARY KEY,
        idempotency_key TEXT NOT NULL,
        client_id TEXT NOT NULL,
        session_id TEXT,
        operation TEXT NOT NULL,
        outcome TEXT NOT NULL,
        fact_id INTEGER,
        existing_fact_id INTEGER,
        observation_id INTEGER,
        recorded_at TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        detail_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE (client_id, idempotency_key),
        FOREIGN KEY (client_id) REFERENCES memory_clients(client_id),
        FOREIGN KEY (fact_id) REFERENCES facts(fact_id),
        FOREIGN KEY (existing_fact_id) REFERENCES facts(fact_id),
        FOREIGN KEY (observation_id) REFERENCES observations(observation_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_observations_session
       ON observations(client_id, session_id, recorded_at)""",
    """CREATE INDEX IF NOT EXISTS idx_fact_provenance_observation
       ON fact_provenance(observation_id)""",
    """CREATE INDEX IF NOT EXISTS idx_memory_write_log_session
       ON memory_write_log(client_id, session_id, recorded_at)""",
    """CREATE TABLE IF NOT EXISTS privacy_erasure_log (
        erasure_id TEXT PRIMARY KEY,
        fact_id INTEGER NOT NULL,
        requested_by TEXT NOT NULL,
        reason TEXT NOT NULL,
        erased_at TEXT NOT NULL,
        affected_observations INTEGER NOT NULL,
        affected_embeddings INTEGER NOT NULL,
        affected_queue_rows INTEGER NOT NULL,
        FOREIGN KEY (fact_id) REFERENCES facts(fact_id)
    )""",
    """CREATE TABLE IF NOT EXISTS embedding_jobs (
        job_id INTEGER PRIMARY KEY,
        fact_id INTEGER NOT NULL,
        document_identity TEXT NOT NULL,
        embedding_version TEXT NOT NULL,
        dimensions INTEGER NOT NULL,
        content_sha256 TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK(status IN ('pending', 'processing', 'completed', 'dead_letter')),
        attempts INTEGER NOT NULL DEFAULT 0,
        available_at TEXT NOT NULL,
        lease_token TEXT,
        lease_owner TEXT,
        lease_expires_at TEXT,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        UNIQUE(fact_id, document_identity),
        FOREIGN KEY (fact_id) REFERENCES facts(fact_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_embedding_jobs_claim
       ON embedding_jobs(status, available_at, lease_expires_at, job_id)""",
    """CREATE TABLE IF NOT EXISTS fact_embeddings (
        fact_id INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        dim INTEGER NOT NULL,
        embedding_identity TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(fact_id, embedding_identity),
        FOREIGN KEY (fact_id) REFERENCES facts(fact_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_fact_embeddings_fact_id
       ON fact_embeddings(fact_id)""",
    """CREATE INDEX IF NOT EXISTS idx_fact_embeddings_identity_dim
       ON fact_embeddings(embedding_identity, dim)""",
)


def ensure_provenance_schema(
    conn: sqlite3.Connection, *, manage_transaction: bool = True
) -> None:
    """Install the version-1 additive provenance tables.

    By default the function owns an idle connection and wraps all DDL in one
    ``BEGIN IMMEDIATE`` transaction for backwards compatibility.  The schema
    migration coordinator passes ``manage_transaction=False`` so provenance
    becomes part of the larger atomic v1 migration.  In either mode ``facts``
    must already exist because ``fact_provenance`` references it.
    """

    if manage_transaction and conn.in_transaction:
        raise RuntimeError("provenance schema setup requires an idle connection")
    if not manage_transaction and not conn.in_transaction:
        raise RuntimeError("caller-managed provenance setup requires a transaction")

    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts'"
    ).fetchone() is None:
        raise RuntimeError("facts table must exist before provenance schema")

    try:
        if manage_transaction:
            conn.execute("BEGIN IMMEDIATE")
        for statement in _PROVENANCE_SCHEMA_STATEMENTS:
            conn.execute(statement)
        if manage_transaction:
            conn.commit()
    except BaseException:
        if manage_transaction and conn.in_transaction:
            conn.rollback()
        raise
