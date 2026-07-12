from __future__ import annotations

import sqlite3

import pytest

from enfold.state_slots import (
    StateCandidate,
    add_conflict_member,
    current_state_facts,
    decide_state_write,
    ensure_state_slot_schema,
    open_state_conflict,
    resolve_state_conflict,
)


def _connection(*, temporal: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    temporal_columns = ", valid_from TEXT, invalid_at TEXT, superseded_by INTEGER" if temporal else ""
    conn.execute(
        f"CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT NOT NULL{temporal_columns})"
    )
    return conn


def _insert_state(
    conn,
    content,
    *,
    authority=0.5,
    valid_from="2026-07-01T00:00:00+00:00",
    subject="cron:briefing",
    predicate="model",
    conflict_group=None,
):
    cursor = conn.execute(
        """
        INSERT INTO facts (
            content, memory_kind, subject_key, predicate_key, object_value,
            source_authority, valid_from, conflict_group
        ) VALUES (?, 'state', ?, ?, ?, ?, ?, ?)
        """,
        (
            content,
            subject,
            predicate,
            content.rsplit(" ", 1)[-1],
            authority,
            valid_from,
            conflict_group,
        ),
    )
    return cursor.lastrowid


def test_schema_adds_typed_columns_conflicts_and_partial_unique_index():
    conn = _connection()
    assert ensure_state_slot_schema(conn) is True
    assert ensure_state_slot_schema(conn) is True

    columns = {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
    assert {
        "memory_kind",
        "subject_key",
        "predicate_key",
        "object_value",
        "source_authority",
        "conflict_group",
    }.issubset(columns)
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(facts)")}
    assert "uq_facts_current_state_slot" in indexes
    assert tuple(
        row[2]
        for row in conn.execute(
            'PRAGMA index_info("uq_facts_current_state_slot")'
        )
    ) == ("scope", "subject_key", "predicate_key")


def test_schema_replaces_pre_scope_slot_index():
    conn = _connection()
    ensure_state_slot_schema(conn)
    conn.execute("DROP INDEX uq_facts_current_state_slot")
    conn.execute(
        """
        CREATE UNIQUE INDEX uq_facts_current_state_slot
        ON facts(subject_key, predicate_key)
        WHERE memory_kind = 'state'
          AND subject_key IS NOT NULL AND predicate_key IS NOT NULL
          AND invalid_at IS NULL AND superseded_by IS NULL
          AND conflict_group IS NULL
        """
    )

    assert ensure_state_slot_schema(conn) is True
    assert tuple(
        row[2]
        for row in conn.execute(
            'PRAGMA index_info("uq_facts_current_state_slot")'
        )
    ) == ("scope", "subject_key", "predicate_key")


def test_partial_invariant_is_skipped_without_temporal_columns():
    conn = _connection(temporal=False)
    assert ensure_state_slot_schema(conn) is False
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(facts)")}
    assert "uq_facts_current_state_slot" not in indexes


def test_exact_slot_decisions_cover_add_dedup_supersede_and_conflict():
    conn = _connection()
    ensure_state_slot_schema(conn)
    candidate = StateCandidate(
        "Morning Briefing uses Terra 5.6",
        "cron:briefing",
        "model",
        "terra-5.6",
        0.8,
        "2026-07-11T00:00:00+00:00",
    )
    assert decide_state_write(conn, candidate).action == "add"

    old_id = _insert_state(
        conn,
        "Morning Briefing uses Terra 5.5",
        authority=0.7,
        valid_from="2026-07-01T00:00:00+00:00",
    )
    decision = decide_state_write(conn, candidate)
    assert decision.action == "supersede"
    assert decision.target_fact_id == old_id

    identical = StateCandidate(
        "Morning Briefing uses Terra 5.5",
        "cron:briefing",
        "model",
        source_authority=0.1,
    )
    assert decide_state_write(conn, identical).action == "dedup"

    same_value_rephrased = StateCandidate(
        "The scheduled briefing currently runs model Terra 5.5",
        "cron:briefing",
        "model",
        object_value="5.5",
        source_authority=0.1,
    )
    assert decide_state_write(conn, same_value_rephrased).action == "dedup"

    weaker_but_newer = StateCandidate(
        "Morning Briefing uses an unverified model",
        "cron:briefing",
        "model",
        source_authority=0.2,
        valid_from="2026-07-12T00:00:00+00:00",
    )
    assert decide_state_write(conn, weaker_but_newer).action == "conflict"


def test_same_authority_newer_wins_but_older_higher_authority_conflicts():
    conn = _connection()
    ensure_state_slot_schema(conn)
    _insert_state(conn, "Uses v2", authority=0.7, valid_from="2026-07-10T00:00:00Z")

    newer = StateCandidate(
        "Uses v3", "cron:briefing", "model", source_authority=0.7,
        valid_from="2026-07-11T00:00:00Z"
    )
    assert decide_state_write(conn, newer).action == "supersede"
    older_stronger = StateCandidate(
        "Uses v1", "cron:briefing", "model", source_authority=1.0,
        valid_from="2026-07-01T00:00:00Z"
    )
    assert decide_state_write(conn, older_stronger).action == "conflict"

    undated_stronger = StateCandidate(
        "Uses an undated value",
        "cron:briefing",
        "model",
        source_authority=1.0,
    )
    assert decide_state_write(conn, undated_stronger).action == "conflict"


def test_events_never_participate_in_slot_supersession():
    conn = _connection()
    ensure_state_slot_schema(conn)
    _insert_state(conn, "Uses v2")
    event = StateCandidate(
        "Model v1 failed yesterday",
        "cron:briefing",
        "model",
        memory_kind="event",
        source_authority=1.0,
        valid_from="2026-07-11T00:00:00Z",
    )
    decision = decide_state_write(conn, event)
    assert decision.action == "add"
    assert decision.current_fact_ids == ()


def test_conflict_lifecycle_is_visible_and_resolution_is_audited():
    conn = _connection()
    ensure_state_slot_schema(conn)
    old_id = _insert_state(conn, "Uses v1")
    conflict = open_state_conflict(
        conn,
        "cron:briefing",
        "model",
        (old_id,),
        detected_at="2026-07-11T12:00:00Z",
        detail_json='{"cause":"authority-freshness-disagreement"}',
    )
    new_id = _insert_state(
        conn,
        "Uses v2",
        authority=0.4,
        valid_from="2026-07-11T00:00:00Z",
        conflict_group=conflict.conflict_id,
    )
    add_conflict_member(conn, conflict.conflict_id, new_id)

    active = current_state_facts(conn, "cron:briefing", "model")
    assert {fact.fact_id for fact in active} == {old_id, new_id}
    assert all(fact.conflict_group == conflict.conflict_id for fact in active)
    assert decide_state_write(
        conn,
        StateCandidate("Uses v3", "cron:briefing", "model"),
    ).action == "conflict"

    resolution = resolve_state_conflict(
        conn,
        conflict.conflict_id,
        new_id,
        resolved_by="victor",
        reason="confirmed from inspected config",
        resolved_at="2026-07-11T13:00:00Z",
    )
    assert resolution.superseded_fact_ids == (old_id,)
    assert current_state_facts(conn, "cron:briefing", "model")[0].fact_id == new_id
    audit = conn.execute(
        """SELECT resolution_fact_id, resolved_by, resolution_reason
           FROM fact_conflicts WHERE conflict_id = ?""",
        (conflict.conflict_id,),
    ).fetchone()
    assert audit == (new_id, "victor", "confirmed from inspected config")
    assert conn.execute(
        "SELECT superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone()[0] == new_id


def test_unique_projection_rejects_two_nonconflicted_current_values():
    conn = _connection()
    ensure_state_slot_schema(conn)
    _insert_state(conn, "Uses v1")
    with pytest.raises(sqlite3.IntegrityError):
        _insert_state(conn, "Uses v2")
