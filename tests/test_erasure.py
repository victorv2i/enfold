from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np
import pytest

from enfold.embeddings import embedding_to_bytes
from enfold.erasure import ErasureError, erase_fact
from enfold.embedding_jobs import EmbeddingOutbox, EmbeddingSpec
from enfold.policy import MemoryPolicy
from enfold.protocol import ClientContext, Request
from enfold.schema import migrate
from enfold.service import EnfoldService
from enfold.sqlite_vec_index import SQLiteVecIndex, rebuild_sqlite_vec_index


def _store() -> tuple[sqlite3.Connection, EnfoldService, ClientContext]:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    context = ClientContext(
        client_id="privacy-admin-tests",
        surface="client-a",
        agent_id="client-a",
        session_id="privacy-session",
        access_scopes=("private",),
    )
    service = EnfoldService(
        conn, MemoryPolicy({"privacy-admin-tests": ("private",)})
    )
    return conn, service, context


def _write(service, context, key, content, **params):
    return service.handle(
        context,
        Request(
            f"req-{key}",
            "memory.write",
            {
                "idempotency_key": key,
                "content": content,
                "source_type": "privacy_test",
                **params,
            },
        ),
    )


def test_privacy_erasure_scrubs_known_materialized_content_copies():
    conn, service, context = _store()
    secret_text = "Fictional private medical appointment is Tuesday"
    written = _write(
        service,
        context,
        "erase-1",
        secret_text,
        observation_content=secret_text,
        evidence_excerpt=secret_text,
        metadata={"note": secret_text},
    )
    fact_id = written["fact_id"]
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, 2, 'fixture')",
        (fact_id, embedding_to_bytes(np.asarray((1.0, 0.0), dtype=np.float32))),
    )
    outbox = EmbeddingOutbox(
        conn, EmbeddingSpec("fake:model:document:none:v1", "v1", 2)
    )
    outbox.enqueue_in_transaction(fact_id)
    conn.execute(
        "UPDATE embedding_jobs SET status = 'processing', lease_token = 'secret-lease', "
        "lease_owner = 'worker', lease_expires_at = '2099-01-01T00:00:00Z'"
    )
    conn.execute(
        "INSERT INTO extract_queue(id, payload, status, payload_hash, last_error) "
        "VALUES (1, ?, 'pending', 'old', NULL)",
        (f'transcript says: {secret_text}',),
    )
    insight_id = conn.execute(
        "INSERT INTO facts(content, category, tags) VALUES (?, 'insight', ?)",
        ("Derived private insight", f"source_facts:{fact_id}"),
    ).lastrowid
    conn.commit()
    rebuild_sqlite_vec_index(conn, "fixture", 2)

    report = erase_fact(
        conn,
        fact_id,
        requested_by="victor",
        reason="privacy request",
    )

    assert report.affected_observations == 1
    assert report.affected_embeddings == 1
    assert report.affected_queue_rows == 1
    assert report.invalidated_insights == 1
    fact = conn.execute(
        "SELECT content, tags, invalid_at, hrr_vector, object_value FROM facts "
        "WHERE fact_id = ?",
        (fact_id,),
    ).fetchone()
    assert fact[0] == f"[PRIVACY ERASED fact:{fact_id}]"
    assert fact[1] == ""
    assert fact[2] is not None
    assert fact[3:] == (None, None)
    observation = conn.execute(
        "SELECT source_uri, content, asserted_by, metadata_json, redacted_at "
        "FROM observations"
    ).fetchone()
    assert observation[:4] == (None, "[PRIVACY ERASED]", None, "{}")
    assert observation[4] is not None
    assert conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0] == 0
    index = SQLiteVecIndex.open(conn, "fixture", 2)
    assert index is not None and index.count() == 0
    job = conn.execute(
        "SELECT content_sha256, status, lease_token, lease_owner, last_error "
        "FROM embedding_jobs WHERE fact_id = ?",
        (fact_id,),
    ).fetchone()
    assert job[0].startswith("erased:")
    assert job[1:] == ("completed", None, None, "privacy_erased")
    assert "privacy_erased" in conn.execute(
        "SELECT payload FROM extract_queue"
    ).fetchone()[0]
    assert conn.execute(
        "SELECT invalid_at FROM facts WHERE fact_id = ?", (insight_id,)
    ).fetchone()[0] is not None
    assert conn.execute(
        "SELECT evidence_excerpt FROM fact_provenance WHERE fact_id = ?", (fact_id,)
    ).fetchone()[0] is None
    assert conn.execute("SELECT COUNT(*) FROM privacy_erasure_log").fetchone()[0] == 1
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_erasing_one_conflict_member_releases_the_remaining_truth():
    conn, service, context = _store()
    slot = {"subject_key": "person:alex", "predicate_key": "timezone"}
    first = _write(
        service,
        context,
        "state-1",
        "Alex is in UTC-5",
        source_authority=0.8,
        state={**slot, "object_value": "UTC-5", "valid_from": "2026-01-01T00:00:00Z"},
    )
    second = _write(
        service,
        context,
        "state-2",
        "Alex is in UTC-8",
        source_authority=0.2,
        state={**slot, "object_value": "UTC-8", "valid_from": "2026-01-02T00:00:00Z"},
    )
    assert second["outcome"] == "conflict"

    report = erase_fact(
        conn, second["fact_id"], requested_by="victor", reason="remove bad source"
    )

    assert report.resolved_conflicts == 1
    assert conn.execute(
        "SELECT conflict_group FROM facts WHERE fact_id = ?", (first["fact_id"],)
    ).fetchone()[0] is None
    assert conn.execute(
        "SELECT resolved_at, resolution_fact_id FROM fact_conflicts"
    ).fetchone()[1] == first["fact_id"]


def test_erasure_requires_idle_schema_v1_and_preserves_missing_fact_behavior():
    conn = sqlite3.connect(":memory:")
    with pytest.raises(ErasureError, match="schema v1"):
        erase_fact(conn, 1, requested_by="victor", reason="privacy")

    conn, _service, _context = _store()
    with pytest.raises(ErasureError, match="not found"):
        erase_fact(conn, 999, requested_by="victor", reason="privacy")


def test_erasure_follows_extraction_link_when_queue_text_is_a_paraphrase():
    conn, service, context = _store()
    written = _write(
        service,
        context,
        "erase-linked",
        "Victor's exact private preference",
        state={
            "subject_key": "person:victor-private",
            "predicate_key": "private_preference",
            "object_value": "exact private preference",
        },
    )
    fact_id = written["fact_id"]
    queue_payload = '{"transcript":"He favors the alternate option."}'
    digest = hashlib.sha256(queue_payload.encode()).hexdigest()
    conn.execute(
        "INSERT INTO extract_queue(id, payload, payload_hash) VALUES (77, ?, ?)",
        (queue_payload, digest),
    )
    observation_id = conn.execute(
        "SELECT observation_id FROM fact_provenance WHERE fact_id = ?", (fact_id,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE observations SET metadata_json = ? WHERE observation_id = ?",
        (json.dumps({"extraction_queue_id": 77, "extraction_payload_sha256": digest}), observation_id),
    )
    conn.execute(
        "UPDATE memory_write_log SET detail_json = ? WHERE fact_id = ?",
        ('{"subject_key":"person:victor-private"}', fact_id),
    )
    conn.commit()

    report = erase_fact(conn, fact_id, requested_by="victor", reason="privacy")

    assert report.affected_queue_rows == 1
    assert "privacy_erased" in conn.execute(
        "SELECT payload FROM extract_queue WHERE id = 77"
    ).fetchone()[0]
    assert tuple(conn.execute(
        "SELECT subject_key, predicate_key, object_value, memory_kind "
        "FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()) == (None, None, None, "fact")
    assert conn.execute(
        "SELECT detail_json FROM memory_write_log WHERE fact_id = ?", (fact_id,)
    ).fetchone()[0] == '{"privacy_erased":true}'
