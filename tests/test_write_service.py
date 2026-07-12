from __future__ import annotations

import sqlite3

import pytest

from enfold.provenance import (
    ConnectionContext,
    WriteRequest,
    ensure_provenance_schema,
)
from enfold.policy import MemoryPolicy
from enfold.write_service import (
    ClientIdentityConflict,
    FactWriteResult,
    IdempotencyConflict,
    MemoryWriteService,
    SessionContextConflict,
)


def _connection(*, temporal: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    temporal_sql = ", invalid_at TEXT, superseded_by INTEGER" if temporal else ""
    conn.execute(
        f"""CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT,
            tags TEXT,
            trust_score REAL,
            source_authority REAL,
            scope TEXT NOT NULL DEFAULT 'private'
            , correction_status TEXT
            {temporal_sql}
        )"""
    )
    ensure_provenance_schema(conn)
    conn.commit()
    return conn


def _writer(conn, request, observation_id):
    cursor = conn.execute(
        """INSERT INTO facts (
               content, category, tags, trust_score, source_authority,
               correction_status, scope
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            request.content,
            request.category,
            request.tags,
            request.trust_score,
            request.source_authority,
            request.correction_status,
            request.scope,
        ),
    )
    return FactWriteResult(cursor.lastrowid)


def _context(**changes):
    values = {
        "client_id": "client-a-install-1",
        "surface": "client-a",
        "agent_id": "client-a",
        "session_id": "thread-123",
        "repository": "enfold",
    }
    values.update(changes)
    return ConnectionContext(**values)


def _request(**changes):
    values = {
        "idempotency_key": "write-123",
        "content": "Client A implemented Enfold provenance.",
        "source_type": "agent_report",
        "performed_by": "client-a",
        "evidence_excerpt": "Focused tests passed.",
    }
    values.update(changes)
    return WriteRequest(**values)


def _service(conn, writer=_writer, *, grants=None):
    grants = grants or {
        "client-a-install-1": ("private", "work"),
        "client-b-install-1": ("private", "work"),
    }
    return MemoryWriteService(conn, writer, MemoryPolicy(grants))


def test_write_atomically_records_identity_observation_fact_and_provenance():
    conn = _connection()
    result = _service(conn).write(_context(), _request())

    assert result.outcome == "inserted"
    assert result.replayed is False
    observation = conn.execute(
        """SELECT client_id, session_id, performed_by, content
           FROM observations WHERE observation_id = ?""",
        (result.observation_id,),
    ).fetchone()
    assert observation == (
        "client-a-install-1",
        "thread-123",
        "client-a",
        "Client A implemented Enfold provenance.",
    )
    assert conn.execute(
        "SELECT fact_id, relation FROM fact_provenance"
    ).fetchone() == (result.fact_id, "supports")
    assert conn.execute("SELECT count(*) FROM memory_write_log").fetchone()[0] == 1


def test_same_client_idempotency_key_replays_without_calling_writer_again():
    conn = _connection()
    calls = 0

    def counting_writer(conn, request, observation_id):
        nonlocal calls
        calls += 1
        return _writer(conn, request, observation_id)

    service = _service(conn, counting_writer)
    first = service.write(_context(), _request())
    replay = service.write(_context(), _request())

    assert calls == 1
    assert replay.replayed is True
    assert replay.write_id == first.write_id
    assert replay.fact_id == first.fact_id
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1


def test_existing_fact_result_attaches_new_evidence_without_reinserting():
    conn = _connection()
    existing_id = conn.execute(
        "INSERT INTO facts (content) VALUES ('Already known')"
    ).lastrowid
    conn.commit()

    def dedup_writer(conn, request, observation_id):
        return FactWriteResult(
            fact_id=existing_id,
            outcome="existing",
            existing_fact_id=existing_id,
        )

    outcome = _service(conn, dedup_writer).write(
        _context(), _request(content="Already known")
    )

    assert outcome.outcome == "existing"
    assert outcome.fact_id == existing_id
    assert outcome.existing_fact_id == existing_id
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
    assert conn.execute(
        "SELECT fact_id, observation_id FROM fact_provenance"
    ).fetchone() == (existing_id, outcome.observation_id)


def test_same_text_from_distinct_source_uris_creates_distinct_observations():
    conn = _connection()

    def existing_writer(conn, request, observation_id):
        row = conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?", (request.content,)
        ).fetchone()
        if row is not None:
            return FactWriteResult(row[0], outcome="existing", existing_fact_id=row[0])
        return _writer(conn, request, observation_id)

    service = _service(conn, existing_writer)
    first = service.write(_context(), _request(source_uri="commit:a"))
    second = service.write(
        _context(),
        _request(idempotency_key="write-456", source_uri="commit:b"),
    )

    assert first.fact_id == second.fact_id
    assert first.observation_id != second.observation_id
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 2


def test_observation_evidence_can_differ_from_normalized_fact_claim():
    conn = _connection()
    outcome = _service(conn).write(
        _context(),
        _request(
            content="The build passed.",
            observation_content="pytest: 483 passed in 8.5 seconds",
        ),
    )

    assert conn.execute(
        "SELECT content FROM observations WHERE observation_id = ?",
        (outcome.observation_id,),
    ).fetchone()[0] == "pytest: 483 passed in 8.5 seconds"
    assert conn.execute(
        "SELECT content FROM facts WHERE fact_id = ?", (outcome.fact_id,)
    ).fetchone()[0] == "The build passed."


def test_same_evidence_at_a_new_commit_is_a_distinct_observation():
    conn = _connection()

    def existing_writer(conn, request, observation_id):
        row = conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?", (request.content,)
        ).fetchone()
        if row is not None:
            return FactWriteResult(row[0], outcome="existing", existing_fact_id=row[0])
        return _writer(conn, request, observation_id)

    service = _service(conn, existing_writer)
    first = service.write(_context(commit_sha="aaa"), _request())
    second = service.write(
        _context(commit_sha="bbb"),
        _request(idempotency_key="write-456"),
    )

    assert first.fact_id == second.fact_id
    assert first.observation_id != second.observation_id
    assert conn.execute(
        "SELECT commit_sha FROM observations ORDER BY observation_id"
    ).fetchall() == [("aaa",), ("bbb",)]


def test_idempotency_is_per_client_and_changed_payload_conflicts():
    conn = _connection()
    service = _service(conn)
    service.write(_context(), _request())

    with pytest.raises(IdempotencyConflict):
        service.write(_context(), _request(content="A different fact"))

    other = _context(client_id="client-b-install-1", surface="client-b", agent_id="client-b")
    outcome = service.write(other, _request(content="Client B recorded this fact"))
    assert outcome.fact_id != 1


def test_stable_client_identity_cannot_be_rebound():
    conn = _connection()
    service = _service(conn)
    service.write(_context(), _request())

    rebound = _context(surface="client-b", agent_id="client-b", session_id="other")
    with pytest.raises(ClientIdentityConflict):
        service.write(rebound, _request(idempotency_key="write-456", content="Other"))


def test_one_client_install_can_host_multiple_agents_in_distinct_sessions():
    conn = _connection()
    service = _service(conn)
    service.write(_context(agent_id="wonny", session_id="main"), _request())
    service.write(
        _context(agent_id="delegate-1", session_id="delegate"),
        _request(idempotency_key="write-456", content="Delegate observation"),
    )

    assert conn.execute(
        "SELECT session_id, agent_id FROM memory_sessions ORDER BY session_id"
    ).fetchall() == [("delegate", "delegate-1"), ("main", "wonny")]


def test_session_branch_and_commit_can_advance_with_observation_provenance():
    conn = _connection()
    service = _service(conn)
    service.write(_context(branch="main", commit_sha="aaa"), _request())

    service.write(
        _context(branch="feature", commit_sha="bbb"),
        _request(idempotency_key="write-456", content="Other"),
    )

    assert conn.execute(
        "SELECT branch, commit_sha FROM memory_sessions"
    ).fetchone() == ("feature", "bbb")
    assert conn.execute(
        "SELECT branch, commit_sha FROM observations ORDER BY observation_id"
    ).fetchall() == [("main", "aaa"), ("feature", "bbb")]


def test_session_stable_context_cannot_drift_after_registration():
    conn = _connection()
    service = _service(conn)
    service.write(_context(repository="enfold"), _request())

    with pytest.raises(SessionContextConflict, match="different connection context"):
        service.write(
            _context(repository="different"),
            _request(idempotency_key="write-456", content="Other"),
        )

    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1


def test_write_scope_must_be_granted_by_connection_context():
    conn = _connection()

    outcome = _service(conn).write(
        _context(access_scopes=("private",)),
        _request(scope="work"),
    )

    assert outcome.outcome == "rejected"
    assert outcome.fact_id is None
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 0


def test_fact_writer_cannot_persist_a_different_scope():
    conn = _connection()

    def wrong_scope_writer(conn, request, observation_id):
        cursor = conn.execute(
            "INSERT INTO facts (content, scope) VALUES (?, 'private')",
            (request.content,),
        )
        return FactWriteResult(cursor.lastrowid)

    with pytest.raises(PermissionError, match="different memory scope"):
        _service(conn, wrong_scope_writer).write(
            _context(access_scopes=("private", "work")),
            _request(scope="work"),
        )

    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0


def test_writer_failure_rolls_back_entire_envelope():
    conn = _connection()

    def failing_writer(conn, request, observation_id):
        conn.execute(
            "INSERT INTO facts (content) VALUES (?)", (request.content,)
        )
        raise RuntimeError("fact writer failed")

    with pytest.raises(RuntimeError, match="fact writer failed"):
        _service(conn, failing_writer).write(_context(), _request())

    for table in (
        "memory_clients",
        "memory_sessions",
        "observations",
        "facts",
        "fact_provenance",
        "memory_write_log",
    ):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_structural_supersession_commits_with_provenance_and_outcome():
    conn = _connection()
    old_id = conn.execute(
        "INSERT INTO facts (content) VALUES ('The job uses model v1')"
    ).lastrowid
    conn.commit()

    result = _service(conn).write(
        _context(),
        _request(
            content="The job uses model v2",
            supersede_fact_id=old_id,
        ),
    )

    invalid_at, replacement = conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone()
    assert invalid_at is not None
    assert replacement == result.fact_id
    assert f'"superseded_fact_id":{old_id}' in result.detail_json


def test_missing_temporal_columns_rolls_back_requested_supersession():
    conn = _connection(temporal=False)
    old_id = conn.execute("INSERT INTO facts (content) VALUES ('old')").lastrowid
    conn.commit()

    with pytest.raises(RuntimeError, match="temporal facts columns"):
        _service(conn).write(
            _context(), _request(content="new", supersede_fact_id=old_id)
        )

    assert conn.execute("SELECT content FROM facts ORDER BY fact_id").fetchall() == [
        ("old",)
    ]
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM memory_write_log").fetchone()[0] == 0


def test_supersession_cross_scope_target_is_uniformly_unavailable():
    conn = _connection()
    old_id = conn.execute(
        "INSERT INTO facts (content, scope) VALUES ('work-only state', 'work')"
    ).lastrowid
    conn.commit()

    outcome = _service(conn).write(
        _context(access_scopes=("private", "work")),
        _request(content="private replacement", supersede_fact_id=old_id),
    )
    assert outcome.outcome == "needs_review"
    assert outcome.fact_id is None
    assert "unavailable" in outcome.detail_json
    missing = _service(conn).write(
        _context(access_scopes=("private", "work")),
        _request(
            idempotency_key="missing-target",
            content="private replacement two",
            supersede_fact_id=999999,
        ),
    )
    assert missing.outcome == outcome.outcome
    assert missing.detail_json == outcome.detail_json

    assert conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone() == (None, None)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("memory_kind", "state", "state-slot supersession"),
        ("conflict_group", "open-conflict", "open-conflict members"),
    ],
)
def test_explicit_untyped_supersession_rejects_typed_or_conflicted_target(
    column, value, message
):
    conn = _connection()
    conn.execute("ALTER TABLE facts ADD COLUMN memory_kind TEXT")
    conn.execute("ALTER TABLE facts ADD COLUMN conflict_group TEXT")
    old_id = conn.execute(
        f"INSERT INTO facts(content, {column}) VALUES ('protected truth', ?)",
        (value,),
    ).lastrowid
    conn.commit()

    outcome = _service(conn).write(
        _context(), _request(content="untyped replacement", supersede_fact_id=old_id)
    )

    assert outcome.outcome == "needs_review"
    assert message in outcome.detail_json
    assert conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone() == (None, None)
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1


def test_defensive_supersession_guard_rolls_back_if_policy_is_bypassed():
    conn = _connection()
    conn.execute("ALTER TABLE facts ADD COLUMN memory_kind TEXT")
    conn.execute("ALTER TABLE facts ADD COLUMN conflict_group TEXT")
    old_id = conn.execute(
        "INSERT INTO facts(content, memory_kind) VALUES ('typed truth', 'state')"
    ).lastrowid
    conn.commit()
    service = _service(conn)
    service._supersession_policy = lambda *args, **kwargs: None

    with pytest.raises(ValueError, match="dedicated resolution path"):
        service.write(
            _context(), _request(content="must roll back", supersede_fact_id=old_id)
        )

    assert conn.execute("SELECT content FROM facts").fetchall() == [("typed truth",)]
    assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0


def test_factless_rejection_is_safe_and_idempotently_replayed():
    conn = _connection()
    service = _service(conn)
    request = _request(content="password = hunter-hunter-123")

    first = service.write(_context(), request)
    replay = service.write(_context(), request)

    assert first.outcome == "rejected"
    assert first.fact_id is first.observation_id is None
    assert replay.replayed is True
    assert replay.write_id == first.write_id
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 0
    logged = conn.execute(
        "SELECT outcome, fact_id, observation_id, detail_json FROM memory_write_log"
    ).fetchone()
    assert logged[:3] == ("rejected", None, None)
    assert "hunter" not in logged[3]


def test_human_correction_is_not_silently_superseded_by_automation():
    conn = _connection()
    old_id = conn.execute(
        """INSERT INTO facts (
               content, source_authority, correction_status
           ) VALUES ('Victor corrected this', 1.0, 'human_corrected')"""
    ).lastrowid
    conn.commit()

    outcome = _service(conn).write(
        _context(),
        _request(content="Automated replacement", supersede_fact_id=old_id),
    )

    assert outcome.outcome == "needs_review"
    assert outcome.fact_id is None
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 0
    assert conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone() == (None, None)


def test_higher_authority_target_requires_review_instead_of_overwrite():
    conn = _connection()
    old_id = conn.execute(
        "INSERT INTO facts (content, source_authority) VALUES ('Trusted value', 0.9)"
    ).lastrowid
    conn.commit()

    outcome = _service(conn).write(
        _context(),
        _request(
            content="Lower authority replacement",
            source_authority=0.4,
            supersede_fact_id=old_id,
        ),
    )

    assert outcome.outcome == "needs_review"
    assert outcome.fact_id is None
    assert conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone() == (None, None)
