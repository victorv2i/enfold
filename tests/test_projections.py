from __future__ import annotations

import sqlite3

import enfold.projections as projections_module
from enfold.core_store import insert_fact
from enfold.policy import MemoryPolicy
from enfold.projections import changes, entities, entity_dossier, timeline
from enfold.protocol import ClientContext, Request
from enfold.schema import migrate
from enfold.service import EnfoldService
from enfold.state_slots import open_state_conflict, resolve_state_conflict


def _store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "projections.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _fact(
    conn: sqlite3.Connection,
    content: str,
    *,
    created_at: str,
    scope: str = "private",
    subject: str | None = None,
    predicate: str | None = None,
    tags: str = "",
) -> int:
    fact_id = insert_fact(
        conn,
        content,
        scope=scope,
        memory_kind="state" if predicate else None,
        subject_key=subject,
        predicate_key=predicate,
        tags=tags,
    )
    conn.execute(
        "UPDATE facts SET created_at = ?, updated_at = ? WHERE fact_id = ?",
        (created_at, created_at, fact_id),
    )
    return fact_id


def test_changes_uses_half_open_window_and_reports_supersession(tmp_path):
    conn = _store(tmp_path)
    first = _fact(
        conn,
        "Victor uses laptop A",
        created_at="2026-07-10T00:00:00Z",
        subject="Victor",
    )
    replacement = _fact(
        conn,
        "Victor uses laptop B",
        created_at="2026-07-11T00:00:00Z",
        subject="Victor",
    )
    conn.execute(
        "UPDATE facts SET invalid_at = ?, superseded_by = ? WHERE fact_id = ?",
        ("2026-07-12T00:00:00Z", replacement, first),
    )
    conn.commit()

    result = changes(
        conn,
        "2026-07-11T00:00:00Z",
        "2026-07-12T00:00:00Z",
        "private",
    )

    assert [(item["kind"], item["fact"]["fact_id"]) for item in result["changes"]] == [
        ("created", replacement),
    ]
    assert changes(
        conn,
        "2026-07-12T00:00:00Z",
        "2026-07-13T00:00:00Z",
        "private",
    )["changes"][0]["kind"] == "superseded"
    conn.close()


def test_changes_empty_results_and_scope_isolation(tmp_path):
    conn = _store(tmp_path)
    _fact(
        conn,
        "Work-only launch note",
        created_at="2026-07-11T12:00:00Z",
        scope="work",
        subject="Project Sol",
    )
    conn.commit()

    assert changes(
        conn,
        "2026-07-11T00:00:00Z",
        "2026-07-12T00:00:00Z",
        "private",
    )["changes"] == []
    assert len(changes(
        conn,
        "2026-07-11T00:00:00Z",
        "2026-07-12T00:00:00Z",
        "work",
    )["changes"]) == 1
    conn.close()


def test_changes_reports_conflict_resolution_for_winner_and_loser(tmp_path):
    conn = _store(tmp_path)
    winner = _fact(
        conn,
        "Victor uses Enfold",
        created_at="2026-07-10T00:00:00Z",
        subject="Victor",
        predicate="memory_service",
    )
    conflict = open_state_conflict(
        conn,
        "Victor",
        "memory_service",
        (winner,),
        detected_at="2026-07-11T00:00:00Z",
    )
    loser = _fact(
        conn,
        "Victor uses another service",
        created_at="2026-07-11T00:00:00Z",
        subject="Victor",
        predicate="memory_service",
    )
    conn.execute(
        "UPDATE facts SET conflict_group = ? WHERE fact_id = ?",
        (conflict.conflict_id, loser),
    )
    conn.execute(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES (?, ?)",
        (conflict.conflict_id, loser),
    )
    resolve_state_conflict(
        conn,
        conflict.conflict_id,
        winner,
        resolved_by="wonny",
        reason="Victor confirmed Enfold",
        resolved_at="2026-07-12T00:00:00Z",
    )
    conn.commit()

    result = changes(
        conn,
        "2026-07-12T00:00:00Z",
        "2026-07-13T00:00:00Z",
        "private",
    )

    assert [(item["kind"], item["fact"]["fact_id"]) for item in result["changes"]] == [
        ("superseded", loser),
        ("resolved", winner),
    ]
    conn.close()


def test_timeline_is_chronological_and_excludes_unsettled_conflicts(tmp_path):
    conn = _store(tmp_path)
    old = _fact(
        conn,
        "Victor prefers tea",
        created_at="2026-07-10T09:00:00Z",
        subject="Victor",
    )
    new = _fact(
        conn,
        "Victor prefers coffee",
        created_at="2026-07-11T09:00:00Z",
        subject="Victor",
        predicate="preferred_drink",
    )
    conn.execute(
        "UPDATE facts SET invalid_at = ?, superseded_by = ? WHERE fact_id = ?",
        ("2026-07-11T09:00:00Z", new, old),
    )
    conflict = open_state_conflict(
        conn,
        "Victor",
        "preferred_drink",
        (new,),
        detected_at="2026-07-12T09:00:00Z",
    )
    disputed = _fact(
        conn,
        "Victor prefers soda",
        created_at="2026-07-12T09:00:00Z",
        subject="Victor",
        predicate="preferred_drink",
    )
    conn.execute(
        "UPDATE facts SET conflict_group = ? WHERE fact_id = ?",
        (conflict.conflict_id, disputed),
    )
    conn.execute(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES (?, ?)",
        (conflict.conflict_id, disputed),
    )
    conn.commit()

    result = timeline(conn, "Victor", "private", limit=20)

    assert [event["changed_at"] for event in result["events"]] == sorted(
        event["changed_at"] for event in result["events"]
    )
    assert {event["fact"]["fact_id"] for event in result["events"]} == {old}
    conn.close()


def test_timeline_limit_keeps_most_recent_events_in_chronological_order(tmp_path):
    conn = _store(tmp_path)
    for day in (10, 11, 12):
        _fact(
            conn,
            f"Victor event {day}",
            created_at=f"2026-07-{day}T00:00:00Z",
            subject="Victor",
        )
    conn.commit()

    result = timeline(conn, "Victor", "private", limit=2)

    assert [event["changed_at"] for event in result["events"]] == [
        "2026-07-11T00:00:00Z",
        "2026-07-12T00:00:00Z",
    ]
    assert result["truncated"] is True
    conn.close()


def test_entities_rank_subjects_and_tags_from_current_facts(tmp_path):
    conn = _store(tmp_path)
    _fact(
        conn,
        "Victor owns Enfold",
        created_at="2026-07-10T00:00:00Z",
        subject="Victor",
        tags="Enfold, memory",
    )
    _fact(
        conn,
        "Victor works on Sol",
        created_at="2026-07-11T00:00:00Z",
        subject="Victor",
        tags="Sol, Enfold",
    )
    _fact(
        conn,
        "Hidden work fact",
        created_at="2026-07-11T00:00:00Z",
        scope="work",
        subject="Victor",
        tags="Enfold",
    )
    conn.commit()

    result = entities(conn, "private", min_facts=2)

    assert [(item["name"], item["fact_count"]) for item in result["entities"]] == [
        ("Enfold", 2),
        ("Victor", 2),
    ]
    conn.close()


def test_entities_scan_is_capped_to_newest_current_facts(tmp_path, monkeypatch):
    monkeypatch.setattr(projections_module, "PROJECTION_SCAN_LIMIT", 3)
    conn = _store(tmp_path)
    for index in range(5):
        _fact(
            conn,
            f"fact {index}",
            created_at=f"2026-07-{10 + index}T00:00:00Z",
            subject=f"Entity {index}",
        )
    conn.commit()

    result = entities(conn, "private")

    assert {item["name"] for item in result["entities"]} == {
        "Entity 2", "Entity 3", "Entity 4"
    }
    conn.close()


def test_timeline_and_dossier_scan_only_newest_history(tmp_path, monkeypatch):
    monkeypatch.setattr(projections_module, "PROJECTION_SCAN_LIMIT", 3)
    conn = _store(tmp_path)
    ids = []
    for index in range(5):
        ids.append(_fact(
            conn,
            f"Victor event {index}",
            created_at=f"2026-07-{10 + index}T00:00:00Z",
            subject="Victor",
        ))
    conn.commit()

    timeline_result = timeline(conn, "Victor", "private", limit=10)
    dossier_result = entity_dossier(conn, "Victor", "private", limit=10)

    assert [event["fact"]["fact_id"] for event in timeline_result["events"]] == ids[-3:]
    assert {fact["fact_id"] for fact in dossier_result["current_facts"]} == set(ids[-3:])
    assert [event["fact"]["fact_id"] for event in dossier_result["recent_changes"]] == ids[-3:]
    assert timeline_result["truncated"] is True
    assert dossier_result["truncated"] is True
    conn.close()


def test_entity_dossier_combines_current_changes_and_open_conflicts(tmp_path):
    conn = _store(tmp_path)
    current = _fact(
        conn,
        "Victor uses Enfold",
        created_at="2026-07-10T00:00:00Z",
        subject="Victor",
        predicate="memory_service",
        tags="Enfold",
    )
    conflict = open_state_conflict(
        conn,
        "Victor",
        "memory_service",
        (current,),
        detected_at="2026-07-11T00:00:00Z",
    )
    challenger = _fact(
        conn,
        "Victor uses another memory service",
        created_at="2026-07-11T00:00:00Z",
        subject="Victor",
        predicate="memory_service",
    )
    conn.execute(
        "UPDATE facts SET conflict_group = ? WHERE fact_id = ?",
        (conflict.conflict_id, challenger),
    )
    conn.execute(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES (?, ?)",
        (conflict.conflict_id, challenger),
    )
    conn.commit()

    result = entity_dossier(conn, "Victor", "private")

    assert result["entity"] == "Victor"
    assert result["current_facts"] == []
    assert [item["conflict_id"] for item in result["open_conflicts"]] == [
        conflict.conflict_id
    ]
    assert result["recent_changes"] == []
    conn.close()


def test_entity_dossier_uses_exact_derived_entity_names(tmp_path):
    conn = _store(tmp_path)
    exact = _fact(
        conn,
        "Project Sol is active",
        created_at="2026-07-10T00:00:00Z",
        subject="Project Sol",
        tags="Sol",
    )
    unrelated = _fact(
        conn,
        "The console is configured",
        created_at="2026-07-11T00:00:00Z",
        subject="Terminal",
    )
    conn.commit()

    result = entity_dossier(conn, "Sol", "private")

    assert [fact["fact_id"] for fact in result["current_facts"]] == [exact]
    recent_ids = {
        event["fact"]["fact_id"] for event in result["recent_changes"]
    }
    assert recent_ids == {exact}
    assert unrelated not in recent_ids
    conn.close()


def test_service_dispatches_projection_reads_with_authorized_scopes(tmp_path):
    conn = _store(tmp_path)
    _fact(
        conn,
        "Victor uses Enfold",
        created_at="2026-07-11T00:00:00Z",
        subject="Victor",
        tags="Enfold",
    )
    _fact(
        conn,
        "Work-only Victor note",
        created_at="2026-07-11T00:00:00Z",
        scope="work",
        subject="Victor",
    )
    conn.commit()
    service = EnfoldService(conn, MemoryPolicy({"client-b": ("private",)}))
    context = ClientContext("client-b", "client-b", "client-b", "session")

    change_result = service.handle(
        context,
        Request(
            "changes",
            "memory.changes",
            {
                "since": "2026-07-11T00:00:00Z",
                "until": "2026-07-12T00:00:00Z",
            },
        ),
    )
    assert [item["fact"]["scope"] for item in change_result["changes"]] == ["private"]
    assert service.handle(
        context, Request("entities", "memory.entities", {"min_facts": 1})
    )["entities"][0]["name"] == "Enfold"
    assert service.handle(
        context, Request("timeline", "memory.timeline", {"subject_or_query": "Victor"})
    )["events"][0]["fact"]["scope"] == "private"
    assert service.handle(
        context, Request("entity", "memory.entity", {"name": "Victor"})
    )["current_facts"][0]["scope"] == "private"
    conn.close()
