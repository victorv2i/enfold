from __future__ import annotations

import sqlite3

from enfold.policy import MemoryPolicy
from enfold.provenance import ConnectionContext, WriteRequest
from enfold.schema import migrate
from enfold.state_slots import (
    StateCandidate,
    current_state_facts,
    list_state_conflicts,
    read_current_state,
)
from enfold.write_service import FactWriteResult, MemoryWriteService


def _store() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn


def _writer(conn, request, observation_id):
    cursor = conn.execute(
        """
        INSERT INTO facts (
            content, category, tags, trust_score, source_authority,
            scope, sensitivity, correction_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request.content,
            request.category,
            request.tags,
            request.trust_score,
            request.source_authority,
            request.scope,
            request.sensitivity,
            request.correction_status,
        ),
    )
    return FactWriteResult(int(cursor.lastrowid))


def _context() -> ConnectionContext:
    return ConnectionContext(
        client_id="client-a-state-tests",
        surface="client-a",
        agent_id="client-a",
        session_id="state-thread",
        access_scopes=("private", "work"),
    )


def _write(
    service: MemoryWriteService,
    key: str,
    value: str,
    *,
    authority: float = 0.8,
    valid_from: str = "2026-07-11T12:00:00Z",
    scope: str = "private",
    correction_status: str | None = None,
):
    content = f"Morning briefing uses {value}"
    request = WriteRequest(
        idempotency_key=key,
        content=content,
        source_type="inspected_config",
        source_authority=authority,
        scope=scope,
        correction_status=correction_status,
    )
    candidate = StateCandidate(
        content=content,
        subject_key="cron:morning-briefing",
        predicate_key="model",
        object_value=value,
        source_authority=authority,
        valid_from=valid_from,
        scope=scope,
    )
    return service.write(_context(), request, state_candidate=candidate)


def test_migrated_store_typed_state_add_dedup_supersede_and_replay():
    conn = _store()
    service = MemoryWriteService(
        conn,
        _writer,
        MemoryPolicy(
            {"client-a-state-tests": ("private", "work")},
            correction_authorities=("client-a-state-tests",),
        ),
    )

    first = _write(service, "state-1", "terra-5.5")
    assert first.outcome == "add"
    dedup = _write(service, "state-2", "terra-5.5")
    assert dedup.outcome == "dedup"
    assert dedup.fact_id == first.fact_id
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1

    replacement = _write(
        service,
        "state-3",
        "terra-5.6",
        valid_from="2026-07-12T12:00:00Z",
    )
    assert replacement.outcome == "supersede"
    assert read_current_state(
        conn, "cron:morning-briefing", "model"
    ).fact_id == replacement.fact_id
    assert conn.execute(
        "SELECT superseded_by FROM facts WHERE fact_id = ?", (first.fact_id,)
    ).fetchone()[0] == replacement.fact_id

    replay = _write(
        service,
        "state-3",
        "terra-5.6",
        valid_from="2026-07-12T12:00:00Z",
    )
    assert replay.replayed is True
    assert replay.write_id == replacement.write_id
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 2


def test_migrated_store_conflicts_abstain_and_remain_durable_on_matching_write():
    conn = _store()
    service = MemoryWriteService(
        conn,
        _writer,
        MemoryPolicy({"client-a-state-tests": ("private", "work")}),
    )
    _write(service, "conflict-1", "terra-5.6", authority=0.8)
    conflicting = _write(
        service,
        "conflict-2",
        "unknown-model",
        authority=0.2,
        valid_from="2026-07-13T12:00:00Z",
    )
    assert conflicting.outcome == "conflict"
    assert read_current_state(conn, "cron:morning-briefing", "model") is None
    conflicts = list_state_conflicts(conn)
    assert len(conflicts) == 1
    assert set(conflicts[0].member_fact_ids) == set(
        fact.fact_id
        for fact in current_state_facts(
            conn, "cron:morning-briefing", "model"
        )
    )

    same_member = _write(
        service,
        "conflict-3",
        "unknown-model",
        authority=0.2,
        valid_from="2026-07-13T12:00:00Z",
    )
    assert same_member.outcome == "conflict"
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 2
    assert len(list_state_conflicts(conn)) == 1


def test_migrated_store_state_slot_identity_isolated_by_scope():
    conn = _store()
    service = MemoryWriteService(
        conn,
        _writer,
        MemoryPolicy({"client-a-state-tests": ("private", "work")}),
    )
    private = _write(service, "scope-1", "private-model")
    work = _write(
        service,
        "scope-2",
        "work-model",
        scope="work",
        authority=0.1,
    )

    assert private.outcome == "add"
    assert work.outcome == "add"
    assert read_current_state(
        conn, "cron:morning-briefing", "model", "private"
    ).fact_id == private.fact_id
    assert read_current_state(
        conn, "cron:morning-briefing", "model", "work"
    ).fact_id == work.fact_id
    assert list_state_conflicts(conn, "private") == ()
    assert list_state_conflicts(conn, "work") == ()


def test_typed_state_cannot_silently_replace_human_corrected_truth():
    conn = _store()
    service = MemoryWriteService(
        conn,
        _writer,
        MemoryPolicy(
            {"client-a-state-tests": ("private", "work")},
            correction_authorities=("client-a-state-tests",),
        ),
    )
    protected = _write(
        service,
        "protected-1",
        "human-confirmed-model",
        authority=0.6,
        correction_status="human_corrected",
    )
    observations_before = conn.execute(
        "SELECT count(*) FROM observations"
    ).fetchone()[0]

    rejected = _write(
        service,
        "protected-2",
        "agent-proposed-model",
        authority=0.9,
        valid_from="2026-07-12T12:00:00Z",
    )
    assert rejected.outcome == "needs_review"
    assert rejected.fact_id is None
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == (
        observations_before
    )
    assert read_current_state(
        conn, "cron:morning-briefing", "model"
    ).fact_id == protected.fact_id
