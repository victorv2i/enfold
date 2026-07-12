"""Deterministic typed-state decisions and durable conflict records.

State slots are explicit structured identity, never inferred here from prose.
The module has no LLM or similarity-search dependency and performs no commits;
callers compose its mutations into their write transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import sqlite3
from typing import Literal, Optional
import uuid


DecisionAction = Literal["add", "dedup", "supersede", "conflict"]


_SUBJECT_KEY = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,127}$")
_PREDICATE_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class StateSlotInvariantError(RuntimeError):
    """The persisted state violates exact-slot invariants."""


def _required(value: str, name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def normalize_subject_key(value: str) -> str:
    """Canonicalize a model-produced slot subject to Enfold key style."""

    if not isinstance(value, str):
        raise ValueError("subject_key must be text")
    normalized = re.sub(r"\s+", "_", value.strip().casefold())
    if not _SUBJECT_KEY.fullmatch(normalized):
        raise ValueError("subject_key is not a canonical slot key")
    return normalized


def normalize_predicate_key(value: str) -> str:
    """Canonicalize a model-produced predicate to lowercase snake case."""

    if not isinstance(value, str):
        raise ValueError("predicate_key must be text")
    normalized = re.sub(r"[\s-]+", "_", value.strip().casefold())
    if not _PREDICATE_KEY.fullmatch(normalized):
        raise ValueError("predicate_key is not a canonical slot key")
    return normalized


def _json_object(value: str, name: str = "detail_json") -> str:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be a JSON object") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object")
    return json.dumps(decoded, sort_keys=True, separators=(",", ":"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class StateCandidate:
    content: str
    subject_key: str
    predicate_key: str
    object_value: Optional[str] = None
    source_authority: float = 0.5
    valid_from: Optional[str] = None
    memory_kind: str = "state"
    scope: str = "private"

    def __post_init__(self) -> None:
        for name in (
            "content", "scope", "subject_key", "predicate_key", "memory_kind"
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        if not 0.0 <= self.source_authority <= 1.0:
            raise ValueError("source_authority must be between 0 and 1")
        _parse_timestamp(self.valid_from)


@dataclass(frozen=True, slots=True)
class CurrentStateFact:
    fact_id: int
    content: str
    subject_key: str
    predicate_key: str
    scope: str
    object_value: Optional[str]
    source_authority: float
    valid_from: Optional[str]
    conflict_group: Optional[str]


@dataclass(frozen=True, slots=True)
class SlotDecision:
    action: DecisionAction
    scope: str
    subject_key: str
    predicate_key: str
    current_fact_ids: tuple[int, ...] = ()
    target_fact_id: Optional[int] = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ConflictRecord:
    conflict_id: str
    scope: str
    subject_key: str
    predicate_key: str
    member_fact_ids: tuple[int, ...]
    detected_at: str


@dataclass(frozen=True, slots=True)
class ConflictResolution:
    conflict_id: str
    resolution_fact_id: int
    superseded_fact_ids: tuple[int, ...]
    resolved_by: str
    reason: str
    resolved_at: str


_TYPED_COLUMNS = (
    ("memory_kind", "TEXT"),
    ("subject_key", "TEXT"),
    ("predicate_key", "TEXT"),
    ("object_value", "TEXT"),
    ("object_entity_id", "INTEGER"),
    ("confidence", "REAL"),
    ("source_authority", "REAL"),
    ("scope", "TEXT NOT NULL DEFAULT 'private'"),
    ("sensitivity", "TEXT"),
    ("correction_status", "TEXT"),
    ("schema_version", "INTEGER"),
    ("conflict_group", "TEXT"),
)


def ensure_state_slot_schema(conn: sqlite3.Connection) -> bool:
    """Add typed-fact columns, conflict tables, and the strict-slot index.

    Returns ``True`` when the partial uniqueness invariant is installed.
    It is skipped when temporal columns are absent or legacy duplicate current
    slots already exist; those stores require an explicit cleanup migration.
    No commit is performed.
    """

    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts'"
    ).fetchone() is None:
        raise RuntimeError("facts table must exist before state-slot schema")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
    for name, column_type in _TYPED_COLUMNS:
        if name not in columns:
            conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {column_type}")
            columns.add(name)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_conflicts (
            conflict_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL DEFAULT 'private',
            subject_key TEXT NOT NULL,
            predicate_key TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            resolved_at TEXT,
            resolution_fact_id INTEGER,
            resolved_by TEXT,
            resolution_reason TEXT,
            detail_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (resolution_fact_id) REFERENCES facts(fact_id)
        )
        """
    )
    conflict_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(fact_conflicts)")
    }
    for name in ("resolved_by", "resolution_reason"):
        if name not in conflict_columns:
            conn.execute(f"ALTER TABLE fact_conflicts ADD COLUMN {name} TEXT")
    if "scope" not in conflict_columns:
        conn.execute(
            "ALTER TABLE fact_conflicts ADD COLUMN scope TEXT NOT NULL DEFAULT 'private'"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_conflict_members (
            conflict_id TEXT NOT NULL,
            fact_id INTEGER NOT NULL,
            PRIMARY KEY (conflict_id, fact_id),
            FOREIGN KEY (conflict_id) REFERENCES fact_conflicts(conflict_id),
            FOREIGN KEY (fact_id) REFERENCES facts(fact_id)
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_fact_conflict_members_fact
           ON fact_conflict_members(fact_id)"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_conflict_resolutions (
            conflict_id TEXT PRIMARY KEY,
            resolution_fact_id INTEGER NOT NULL,
            resolver_client_id TEXT NOT NULL,
            resolver_session_id TEXT NOT NULL,
            resolver_agent_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            resolved_at TEXT NOT NULL,
            FOREIGN KEY (conflict_id) REFERENCES fact_conflicts(conflict_id),
            FOREIGN KEY (resolution_fact_id) REFERENCES facts(fact_id),
            FOREIGN KEY (resolver_client_id, resolver_session_id)
                REFERENCES memory_sessions(client_id, session_id)
        )
        """
    )

    if not {"invalid_at", "superseded_by"}.issubset(columns):
        return False
    duplicate = conn.execute(
        """
        SELECT 1 FROM facts
        WHERE memory_kind = 'state'
          AND subject_key IS NOT NULL AND predicate_key IS NOT NULL
          AND invalid_at IS NULL AND superseded_by IS NULL
          AND conflict_group IS NULL
        GROUP BY scope, subject_key, predicate_key
        HAVING COUNT(*) > 1
        LIMIT 1
        """
    ).fetchone()
    if duplicate is not None:
        return False
    index_columns = tuple(
        str(row[2])
        for row in conn.execute(
            'PRAGMA index_info("uq_facts_current_state_slot")'
        )
    )
    if index_columns and index_columns != (
        "scope", "subject_key", "predicate_key"
    ):
        conn.execute("DROP INDEX uq_facts_current_state_slot")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_facts_current_state_slot
        ON facts(scope, subject_key, predicate_key)
        WHERE memory_kind = 'state'
          AND subject_key IS NOT NULL AND predicate_key IS NOT NULL
          AND invalid_at IS NULL AND superseded_by IS NULL
          AND conflict_group IS NULL
        """
    )
    return True


def current_state_facts(
    conn: sqlite3.Connection,
    subject_key: str,
    predicate_key: str,
    scope: str = "private",
) -> tuple[CurrentStateFact, ...]:
    """Return active facts for exactly one structured state slot."""

    subject_key = _required(subject_key, "subject_key")
    predicate_key = _required(predicate_key, "predicate_key")
    scope = _required(scope, "scope")
    rows = conn.execute(
        """
        SELECT fact_id, content, subject_key, predicate_key, scope, object_value,
               COALESCE(source_authority, 0.5), valid_from, conflict_group
        FROM facts
        WHERE memory_kind = 'state'
          AND scope = ? AND subject_key = ? AND predicate_key = ?
          AND invalid_at IS NULL AND superseded_by IS NULL
        ORDER BY fact_id
        """,
        (scope, subject_key, predicate_key),
    ).fetchall()
    return tuple(
        CurrentStateFact(
            fact_id=int(row[0]),
            content=row[1],
            subject_key=row[2],
            predicate_key=row[3],
            scope=row[4],
            object_value=row[5],
            source_authority=float(row[6]),
            valid_from=row[7],
            conflict_group=row[8],
        )
        for row in rows
    )


def read_current_state(
    conn: sqlite3.Connection,
    subject_key: str,
    predicate_key: str,
    scope: str = "private",
) -> Optional[CurrentStateFact]:
    """Return an unambiguous current value, abstaining on conflicts."""

    facts = current_state_facts(conn, subject_key, predicate_key, scope)
    if not facts or any(fact.conflict_group for fact in facts):
        return None
    if len(facts) != 1:
        raise StateSlotInvariantError(
            "strict state slot has multiple non-conflicted current facts"
        )
    return facts[0]


def list_state_conflicts(
    conn: sqlite3.Connection,
    scope: str = "private",
    *,
    unresolved_only: bool = True,
) -> tuple[ConflictRecord, ...]:
    """List durable conflicts visible in one exact memory scope."""

    scope = _required(scope, "scope")
    predicate = "AND c.resolved_at IS NULL" if unresolved_only else ""
    rows = conn.execute(
        f"""
        SELECT c.conflict_id, c.scope, c.subject_key, c.predicate_key,
               c.detected_at, m.fact_id
        FROM fact_conflicts c
        JOIN fact_conflict_members m ON m.conflict_id = c.conflict_id
        WHERE c.scope = ? {predicate}
        ORDER BY c.detected_at, c.conflict_id, m.fact_id
        """,
        (scope,),
    ).fetchall()
    grouped: dict[str, tuple[str, str, str, str, list[int]]] = {}
    for row in rows:
        conflict_id = str(row[0])
        entry = grouped.get(conflict_id)
        if entry is None:
            entry = (str(row[1]), str(row[2]), str(row[3]), str(row[4]), [])
            grouped[conflict_id] = entry
        entry[4].append(int(row[5]))
    return tuple(
        ConflictRecord(
            conflict_id,
            values[0],
            values[1],
            values[2],
            tuple(values[4]),
            values[3],
        )
        for conflict_id, values in grouped.items()
    )


def decide_state_write(
    conn: sqlite3.Connection, candidate: StateCandidate
) -> SlotDecision:
    """Choose an exact-slot action without mutating the database.

    Authority wins only when the candidate is not demonstrably older. Equal
    authority requires a strictly newer timestamp. Conflicting authority and
    freshness signals remain visible instead of being guessed away.
    Non-state kinds always add and never supersede slot members.
    """

    if candidate.memory_kind != "state":
        return SlotDecision(
            "add",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            reason="non-state memories coexist",
        )

    current = current_state_facts(
        conn, candidate.subject_key, candidate.predicate_key, candidate.scope
    )
    if not current:
        return SlotDecision(
            "add",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            reason="slot has no current value",
        )

    exact = next(
        (
            fact
            for fact in current
            if fact.content == candidate.content
            or (
                candidate.object_value is not None
                and fact.object_value == candidate.object_value
            )
        ),
        None,
    )
    # An unresolved conflict is the truth of the slot.  A matching candidate
    # may add evidence to one member, but must never make an ordinary write
    # look like an unambiguous deduplication.
    if len(current) > 1 or any(fact.conflict_group for fact in current):
        return SlotDecision(
            "conflict",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            tuple(fact.fact_id for fact in current),
            target_fact_id=exact.fact_id if exact is not None else None,
            reason="slot already has an unresolved conflict",
        )

    if exact is not None:
        return SlotDecision(
            "dedup",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            tuple(fact.fact_id for fact in current),
            target_fact_id=exact.fact_id,
            reason="identical current content or structured value",
        )

    existing = current[0]
    candidate_time = _parse_timestamp(candidate.valid_from)
    existing_time = _parse_timestamp(existing.valid_from)
    candidate_is_not_older = existing_time is None or (
        candidate_time is not None and candidate_time >= existing_time
    )
    authority_higher = candidate.source_authority > existing.source_authority
    authority_equal = candidate.source_authority == existing.source_authority
    freshness_higher = candidate_time is not None and (
        existing_time is None or candidate_time > existing_time
    )

    if authority_higher and candidate_is_not_older:
        return SlotDecision(
            "supersede",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            (existing.fact_id,),
            target_fact_id=existing.fact_id,
            reason="higher authority and not older",
        )
    if authority_equal and freshness_higher:
        return SlotDecision(
            "supersede",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            (existing.fact_id,),
            target_fact_id=existing.fact_id,
            reason="same authority with newer validity time",
        )
    return SlotDecision(
        "conflict",
        candidate.scope,
        candidate.subject_key,
        candidate.predicate_key,
        (existing.fact_id,),
        reason="authority and freshness do not establish a clear winner",
    )


def open_state_conflict(
    conn: sqlite3.Connection,
    subject_key: str,
    predicate_key: str,
    existing_fact_ids: tuple[int, ...],
    *,
    scope: str = "private",
    detected_at: Optional[str] = None,
    detail_json: str = "{}",
) -> ConflictRecord:
    """Open a conflict and move existing members outside the unique projection.

    The caller can then insert the candidate with the returned ``conflict_id``
    as its ``conflict_group`` and call :func:`add_conflict_member`, all within
    the same transaction.
    """

    subject_key = _required(subject_key, "subject_key")
    predicate_key = _required(predicate_key, "predicate_key")
    scope = _required(scope, "scope")
    if not existing_fact_ids:
        raise ValueError("a conflict requires at least one existing fact")
    if len(set(existing_fact_ids)) != len(existing_fact_ids):
        raise ValueError("conflict member ids must be unique")
    detail_json = _json_object(detail_json)
    detected_at = detected_at or _utc_now()
    _parse_timestamp(detected_at)

    placeholders = ",".join("?" for _ in existing_fact_ids)
    rows = conn.execute(
        f"""
        SELECT fact_id FROM facts
        WHERE fact_id IN ({placeholders})
          AND memory_kind = 'state'
          AND scope = ? AND subject_key = ? AND predicate_key = ?
          AND invalid_at IS NULL AND superseded_by IS NULL
          AND conflict_group IS NULL
        """,
        (*existing_fact_ids, scope, subject_key, predicate_key),
    ).fetchall()
    found = {int(row[0]) for row in rows}
    if found != set(existing_fact_ids):
        raise StateSlotInvariantError(
            "every conflict member must be an active state fact in the exact slot"
        )

    conflict_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO fact_conflicts (
            conflict_id, scope, subject_key, predicate_key, detected_at, detail_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (conflict_id, scope, subject_key, predicate_key, detected_at, detail_json),
    )
    conn.execute(
        f"UPDATE facts SET conflict_group = ? WHERE fact_id IN ({placeholders})",
        (conflict_id, *existing_fact_ids),
    )
    conn.executemany(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES (?, ?)",
        ((conflict_id, fact_id) for fact_id in existing_fact_ids),
    )
    return ConflictRecord(
        conflict_id,
        scope,
        subject_key,
        predicate_key,
        tuple(existing_fact_ids),
        detected_at,
    )


def add_conflict_member(
    conn: sqlite3.Connection, conflict_id: str, fact_id: int
) -> None:
    """Attach a newly inserted, exact-slot fact to an open conflict."""

    row = conn.execute(
        """
        SELECT c.scope, c.subject_key, c.predicate_key
        FROM fact_conflicts c
        WHERE c.conflict_id = ? AND c.resolved_at IS NULL
        """,
        (conflict_id,),
    ).fetchone()
    if row is None:
        raise ValueError("conflict does not exist or is already resolved")
    fact = conn.execute(
        """
        SELECT scope, subject_key, predicate_key, conflict_group, memory_kind,
               invalid_at, superseded_by
        FROM facts WHERE fact_id = ?
        """,
        (fact_id,),
    ).fetchone()
    if fact is None or fact[0:5] != (
        row[0], row[1], row[2], conflict_id, "state"
    ):
        raise StateSlotInvariantError(
            "new conflict member must be a state fact in the exact slot and group"
        )
    if fact[5] is not None or fact[6] is not None:
        raise StateSlotInvariantError("new conflict member must be current")
    conn.execute(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES (?, ?)",
        (conflict_id, fact_id),
    )


def resolve_state_conflict(
    conn: sqlite3.Connection,
    conflict_id: str,
    resolution_fact_id: int,
    *,
    resolved_by: str,
    reason: str,
    resolved_at: Optional[str] = None,
    resolver_client_id: Optional[str] = None,
    resolver_session_id: Optional[str] = None,
    resolver_agent_id: Optional[str] = None,
) -> ConflictResolution:
    """Resolve a conflict to one member and retain an audited history."""

    resolved_by = _required(resolved_by, "resolved_by")
    reason = _required(reason, "reason")
    resolved_at = resolved_at or _utc_now()
    _parse_timestamp(resolved_at)
    conflict = conn.execute(
        "SELECT resolved_at FROM fact_conflicts WHERE conflict_id = ?",
        (conflict_id,),
    ).fetchone()
    if conflict is None:
        raise ValueError("conflict does not exist")
    if conflict[0] is not None:
        raise ValueError("conflict is already resolved")
    members = tuple(
        int(row[0])
        for row in conn.execute(
            """SELECT fact_id FROM fact_conflict_members
               WHERE conflict_id = ? ORDER BY fact_id""",
            (conflict_id,),
        ).fetchall()
    )
    if resolution_fact_id not in members:
        raise ValueError("resolution fact must be a conflict member")
    losers = tuple(fact_id for fact_id in members if fact_id != resolution_fact_id)
    if losers:
        placeholders = ",".join("?" for _ in losers)
        cursor = conn.execute(
            f"""
            UPDATE facts SET invalid_at = ?, superseded_by = ?
            WHERE fact_id IN ({placeholders})
              AND conflict_group = ? AND invalid_at IS NULL
            """,
            (resolved_at, resolution_fact_id, *losers, conflict_id),
        )
        if cursor.rowcount != len(losers):
            raise StateSlotInvariantError("not all losing conflict members are current")
    cursor = conn.execute(
        """
        UPDATE facts SET conflict_group = NULL
        WHERE fact_id = ? AND conflict_group = ?
          AND invalid_at IS NULL AND superseded_by IS NULL
        """,
        (resolution_fact_id, conflict_id),
    )
    if cursor.rowcount != 1:
        raise StateSlotInvariantError("resolution fact is not a current member")
    conn.execute(
        """
        UPDATE fact_conflicts
        SET resolved_at = ?, resolution_fact_id = ?, resolved_by = ?,
            resolution_reason = ?
        WHERE conflict_id = ? AND resolved_at IS NULL
        """,
        (resolved_at, resolution_fact_id, resolved_by, reason, conflict_id),
    )
    if any(
        value is not None
        for value in (resolver_client_id, resolver_session_id, resolver_agent_id)
    ):
        resolver_client_id = _required(resolver_client_id or "", "resolver_client_id")
        resolver_session_id = _required(resolver_session_id or "", "resolver_session_id")
        resolver_agent_id = _required(resolver_agent_id or "", "resolver_agent_id")
        conn.execute(
            """
            INSERT INTO fact_conflict_resolutions (
                conflict_id, resolution_fact_id, resolver_client_id,
                resolver_session_id, resolver_agent_id, reason, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conflict_id,
                resolution_fact_id,
                resolver_client_id,
                resolver_session_id,
                resolver_agent_id,
                reason,
                resolved_at,
            ),
        )
    return ConflictResolution(
        conflict_id,
        resolution_fact_id,
        losers,
        resolved_by,
        reason,
        resolved_at,
    )
