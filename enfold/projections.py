"""Bounded time and entity projections over the Enfold fact store."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
import json
import sqlite3
from typing import Any

from .core_store import active_facts, historical_facts_by_id
from .state_slots import list_state_conflicts


DEFAULT_LIMIT = 100
MAX_LIMIT = 200
MAX_OUTPUT_CHARS = 60_000
MAX_QUERY_CHARS = 16_000
MAX_ENTITY_CHARS = 2_000
PROJECTION_SCAN_LIMIT = 10_000
_ITEMS_CHAR_BUDGET = MAX_OUTPUT_CHARS - 256
_FACT_FIELDS = (
    "fact_id", "content", "category", "tags", "trust_score", "created_at",
    "updated_at", "valid_from", "invalid_at", "superseded_by", "memory_kind",
    "subject_key", "predicate_key", "object_value", "confidence",
    "source_authority", "scope", "sensitivity", "correction_status",
    "schema_version", "conflict_group",
)


def _scopes(scope: str | Sequence[str]) -> tuple[str, ...]:
    values = (scope,) if isinstance(scope, str) else tuple(scope)
    cleaned = tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
    if not cleaned:
        raise ValueError("scope must not be empty")
    return cleaned


def _limit(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_LIMIT
    ):
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    return value


def _timestamp(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty ISO-8601 timestamp")
    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _fact(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _FACT_FIELDS if key in row.keys()}


def _json_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _bounded(items: Iterable[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], bool]:
    result: list[dict[str, Any]] = []
    used = 2
    truncated = False
    for item in items:
        if len(result) >= limit:
            truncated = True
            break
        item_size = _json_chars(item) + (1 if result else 0)
        if used + item_size > _ITEMS_CHAR_BUDGET:
            truncated = True
            break
        result.append(item)
        used += item_size
    return result, truncated


def _event_rows(
    conn: sqlite3.Connection,
    scopes: tuple[str, ...],
    *,
    since: str | None = None,
    until: str | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    placeholders = ", ".join("?" for _ in scopes)
    scope_sql = f"f.scope IN ({placeholders})"
    window_sql = ""
    window_params: tuple[str, ...] = ()
    if since is not None and until is not None:
        window_sql = (
            "AND julianday(e.changed_at) >= julianday(?) "
            "AND julianday(e.changed_at) < julianday(?)"
        )
        window_params = (since, until)
    selected = ", ".join(f"f.{name}" for name in _FACT_FIELDS)
    rows = conn.execute(
        f"""
        WITH fact_events AS (
            SELECT 'created' AS kind, f.created_at AS changed_at, f.fact_id
            FROM facts f
            WHERE {scope_sql} AND f.conflict_group IS NULL
            UNION ALL
            SELECT 'superseded', f.invalid_at, f.fact_id
            FROM facts f
            WHERE {scope_sql} AND f.invalid_at IS NOT NULL
              AND f.superseded_by IS NOT NULL
            UNION ALL
            SELECT 'resolved', c.resolved_at, c.resolution_fact_id
            FROM fact_conflicts c
            JOIN facts f ON f.fact_id = c.resolution_fact_id
            WHERE {scope_sql} AND c.resolved_at IS NOT NULL
        ), newest_events AS (
            SELECT e.kind, e.changed_at, {selected}
            FROM fact_events e
            JOIN facts f ON f.fact_id = e.fact_id
            WHERE e.changed_at IS NOT NULL {window_sql}
            ORDER BY julianday(e.changed_at) DESC, e.changed_at DESC,
                     CASE e.kind
                         WHEN 'created' THEN 0
                         WHEN 'superseded' THEN 1
                         ELSE 2
                     END DESC,
                     f.fact_id DESC
            LIMIT ?
        )
        SELECT * FROM newest_events
        ORDER BY julianday(changed_at), changed_at,
                 CASE kind
                     WHEN 'created' THEN 0
                     WHEN 'superseded' THEN 1
                     ELSE 2
                 END,
                 fact_id
        """,
        (*scopes, *scopes, *scopes, *window_params, PROJECTION_SCAN_LIMIT),
    ).fetchall()
    events = [
        {
            "kind": str(row["kind"]),
            "changed_at": str(row["changed_at"]),
            "fact": _fact(row),
        }
        for row in rows
    ]
    return events, len(rows) == PROJECTION_SCAN_LIMIT


def changes(
    conn: sqlite3.Connection,
    since: str,
    until: str,
    scope: str | Sequence[str],
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Return settled fact changes in the half-open interval ``[since, until)``."""

    since_value = _timestamp(since, "since")
    until_value = _timestamp(until, "until")
    if since_value >= until_value:
        raise ValueError("since must be earlier than until")
    scanned, scan_truncated = _event_rows(
        conn, _scopes(scope), since=since_value, until=until_value
    )
    events, output_truncated = _bounded(scanned, _limit(limit))
    return {"changes": events, "truncated": scan_truncated or output_truncated}


def _entity_names(fact: dict[str, Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    subject = fact.get("subject_key")
    if isinstance(subject, str) and subject.strip():
        names[subject.strip().casefold()] = subject.strip()
    tags = fact.get("tags")
    if isinstance(tags, str):
        for tag in tags.split(","):
            if tag.strip():
                names.setdefault(tag.strip().casefold(), tag.strip())
    return names


def _matches_entity(fact: dict[str, Any], name: str) -> bool:
    return name.casefold() in _entity_names(fact)


def _entity_events(
    conn: sqlite3.Connection,
    entity: str,
    scopes: tuple[str, ...],
    limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    scanned, scan_truncated = _event_rows(conn, scopes)
    matched = [
        event
        for event in scanned
        if _matches_entity(event["fact"], entity)
    ]
    selected = matched[-limit:]
    events, chars_truncated = _bounded(selected, limit)
    return events, scan_truncated or len(matched) > limit or chars_truncated


def timeline(
    conn: sqlite3.Connection,
    subject_or_query: str,
    scope: str | Sequence[str],
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Return matching events from the newest ``PROJECTION_SCAN_LIMIT`` events."""

    if not isinstance(subject_or_query, str) or not subject_or_query.strip():
        raise ValueError("subject_or_query must not be empty")
    if len(subject_or_query) > MAX_QUERY_CHARS:
        raise ValueError(
            f"subject_or_query must not exceed {MAX_QUERY_CHARS} characters"
        )
    cap = _limit(limit)
    query = subject_or_query.strip().casefold()
    scanned, scan_truncated = _event_rows(conn, _scopes(scope))
    matched = []
    for event in scanned:
        fact = event["fact"]
        text = " ".join(
            str(fact.get(field) or "")
            for field in ("content", "subject_key", "predicate_key", "object_value", "tags")
        ).casefold()
        if query in text:
            matched.append(event)
    selected = matched[-cap:]
    events, chars_truncated = _bounded(selected, cap)
    return {
        "events": events,
        "truncated": scan_truncated or len(matched) > cap or chars_truncated,
    }


def entities(
    conn: sqlite3.Connection,
    scope: str | Sequence[str],
    min_facts: int = 1,
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Rank entities across the newest ``PROJECTION_SCAN_LIMIT`` current facts."""

    if (
        isinstance(min_facts, bool)
        or not isinstance(min_facts, int)
        or min_facts < 1
    ):
        raise ValueError("min_facts must be a positive integer")
    counts: dict[str, set[int]] = defaultdict(set)
    display: dict[str, str] = {}
    sources: dict[str, set[str]] = defaultdict(set)
    scanned = active_facts(
        conn, allowed_scopes=_scopes(scope), limit=PROJECTION_SCAN_LIMIT
    )
    for fact in scanned:
        fact_id = int(fact["fact_id"])
        subject = fact.get("subject_key")
        if isinstance(subject, str) and subject.strip():
            key = subject.strip().casefold()
            display.setdefault(key, subject.strip())
            counts[key].add(fact_id)
            sources[key].add("subject")
        tags = fact.get("tags")
        if isinstance(tags, str):
            for raw in tags.split(","):
                if not raw.strip():
                    continue
                key = raw.strip().casefold()
                display.setdefault(key, raw.strip())
                counts[key].add(fact_id)
                sources[key].add("tag")
    ranked = [
        {
            "name": display[key],
            "fact_count": len(fact_ids),
            "derived_from": sorted(sources[key]),
        }
        for key, fact_ids in counts.items()
        if len(fact_ids) >= min_facts
    ]
    ranked.sort(
        key=lambda item: (
            -item["fact_count"],
            item["name"].casefold(),
            item["name"],
        )
    )
    result, output_truncated = _bounded(ranked, _limit(limit))
    return {
        "entities": result,
        "truncated": len(scanned) == PROJECTION_SCAN_LIMIT or output_truncated,
    }


def entity_dossier(
    conn: sqlite3.Connection,
    name: str,
    scope: str | Sequence[str],
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Return an entity dossier, scanning at most ``PROJECTION_SCAN_LIMIT`` facts/events."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must not be empty")
    if len(name) > MAX_ENTITY_CHARS:
        raise ValueError(f"name must not exceed {MAX_ENTITY_CHARS} characters")
    entity = name.strip()
    scopes = _scopes(scope)
    cap = _limit(limit)
    scanned_current = active_facts(
        conn, allowed_scopes=scopes, limit=PROJECTION_SCAN_LIMIT
    )
    all_current = [
        _fact(fact)
        for fact in scanned_current
        if _matches_entity(fact, entity)
    ]
    current = all_current[:cap]
    recent, recent_truncated = _entity_events(conn, entity, scopes, cap)
    conflicts: list[dict[str, Any]] = []
    for one_scope in scopes:
        for conflict in list_state_conflicts(conn, one_scope):
            member_facts = [
                _fact(row)
                for row in historical_facts_by_id(
                    conn,
                    conflict.member_fact_ids,
                    allowed_scopes=(one_scope,),
                )
            ]
            if conflict.subject_key.casefold() != entity.casefold() and not any(
                _matches_entity(fact, entity) for fact in member_facts
            ):
                continue
            conflicts.append(
                {
                    "conflict_id": conflict.conflict_id,
                    "scope": conflict.scope,
                    "subject_key": conflict.subject_key,
                    "predicate_key": conflict.predicate_key,
                    "detected_at": conflict.detected_at,
                    "members": member_facts,
                }
            )
    all_conflicts = conflicts
    conflicts = all_conflicts[:cap]
    result: dict[str, Any] = {
        "entity": entity,
        "current_facts": current,
        "recent_changes": recent,
        "open_conflicts": conflicts,
        "truncated": (
            len(all_current) > cap
            or len(scanned_current) == PROJECTION_SCAN_LIMIT
            or len(all_conflicts) > cap
            or recent_truncated
        ),
    }
    sections = ("recent_changes", "current_facts", "open_conflicts")
    while _json_chars(result) > MAX_OUTPUT_CHARS:
        for section in sections:
            if result[section]:
                result[section].pop()
                result["truncated"] = True
                break
        else:
            break
    return result
