from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from enfold.policy import MemoryPolicy
from enfold.extraction_enqueue import ExtractionEnqueuer
from enfold.embeddings import embedding_to_bytes
from enfold.hybrid_retrieval import HybridRetriever
from enfold.protocol import ClientContext, Request
from enfold.schema import migrate
from enfold.service import EnfoldService, ServiceRequestError


def _store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "enfold-service.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _context(
    client: str,
    surface: str,
    agent: str,
    *,
    scopes: tuple[str, ...] = ("private", "work"),
    session: str | None = None,
) -> ClientContext:
    return ClientContext(
        client_id=client,
        surface=surface,
        agent_id=agent,
        session_id=session or f"{agent}-session",
        repository="enfold",
        branch="service-layer",
        commit_sha="abc123",
        access_scopes=scopes,
    )


def _request(request_id: str, method: str, **params) -> Request:
    return Request(request_id, method, params)


def _write(
    service: EnfoldService,
    context: ClientContext,
    key: str,
    content: str,
    **params,
):
    return service.handle(
        context,
        _request(
            f"req-{key}",
            "memory.write",
            idempotency_key=key,
            content=content,
            source_type="agent_report",
            **params,
        ),
    )


@pytest.mark.parametrize(
    ("near_dedup_enabled", "expected_outcome", "expected_history_size"),
    [(True, "near_dedup", 2), (False, "inserted", 1)],
)
def test_service_near_duplicate_merge_and_off_switch(
    tmp_path, near_dedup_enabled, expected_outcome, expected_history_size
):
    conn = _store(tmp_path)
    identity = "fake:service:document:none:v1"
    service = EnfoldService(
        conn,
        MemoryPolicy({"codex-install": ("private",)}),
        embedding_identity=identity,
        query_embedder=lambda _content: np.asarray((1.0, 0.0), dtype=np.float32),
        near_dedup_enabled=near_dedup_enabled,
    )
    context = _context("codex-install", "codex", "codex", scopes=("private",))
    existing = _write(
        service, context, "existing", "The build uses port 3100.", trust_score=0.8
    )
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, 2, ?)",
        (
            existing["fact_id"],
            embedding_to_bytes(np.asarray((1.0, 0.0), dtype=np.float32)),
            identity,
        ),
    )
    conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
    conn.commit()

    result = _write(
        service,
        context,
        "paraphrase",
        "Build service listens on port 3100.",
        trust_score=0.4,
    )
    history = service.handle(
        context,
        _request("history-near-dedup", "memory.history", fact_id=result["fact_id"]),
    )["facts"]

    assert result["outcome"] == expected_outcome
    assert len(history) == expected_history_size
    if near_dedup_enabled:
        assert result["fact_id"] == existing["fact_id"]
        assert history[1]["superseded_by"] == existing["fact_id"]
    conn.close()


@pytest.fixture
def setup(tmp_path):
    conn = _store(tmp_path)
    grants = {
        "client-a-install": ("private", "work", "secret"),
        "client-b-install": ("private",),
        "hermes-install": ("private", "work"),
    }
    service = EnfoldService(
        conn,
        MemoryPolicy(
            grants,
            correction_authorities=("hermes-install",),
            conflict_resolution_authorities=("hermes-install",),
        ),
    )
    contexts = {
        "client-a": _context("client-a-install", "client-a", "client-a"),
        "client-b": _context("client-b-install", "client-b", "client-b"),
        "hermes": _context("hermes-install", "hermes", "wonny"),
    }
    yield conn, service, contexts
    conn.close()


def test_cross_agent_writes_have_trusted_client_and_hermes_provenance(setup):
    conn, service, contexts = setup
    outcomes = {
        name: _write(service, context, f"{name}-1", f"{name} built the memory bridge")
        for name, context in contexts.items()
    }

    rows = conn.execute(
        """SELECT client_id, session_id, performed_by, repository, branch, commit_sha
           FROM observations ORDER BY observation_id"""
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("client-a-install", "client-a-session", "client-a", "enfold", "service-layer", "abc123"),
        ("client-b-install", "client-b-session", "client-b", "enfold", "service-layer", "abc123"),
        ("hermes-install", "wonny-session", "wonny", "enfold", "service-layer", "abc123"),
    ]
    for name, result in outcomes.items():
        evidence = service.handle(
            contexts[name],
            _request(f"evidence-{name}", "memory.evidence", fact_id=result["fact_id"]),
        )
        assert evidence["evidence"][0]["client_id"] == f"{name}-install"
        assert evidence["evidence"][0]["performed_by"] == contexts[name].agent_id


def test_server_grants_narrow_reads_and_writes_without_cross_scope_oracles(setup):
    conn, service, contexts = setup
    work = _write(
        service,
        contexts["client-a"],
        "work-1",
        "Project Zephyr deployment token rotation completed",
        scope="work",
    )
    private = _write(
        service,
        contexts["client-b"],
        "private-1",
        "Project Zephyr private notes are indexed",
    )
    client_b_requested_too_much = _context(
        "client-b-install", "client-b", "client-b", scopes=("private", "work")
    )

    results = service.handle(
        client_b_requested_too_much,
        _request("search-1", "memory.search", query="Zephyr"),
    )["facts"]
    assert [fact["fact_id"] for fact in results] == [private["fact_id"]]
    rejected = _write(
        service,
        client_b_requested_too_much,
        "work-denied",
        "Client B cannot self-grant a work memory",
        scope="work",
    )
    assert rejected["outcome"] == "rejected"
    assert rejected["fact_id"] is None
    assert conn.execute(
        "SELECT COUNT(*) FROM facts WHERE fact_id = ?", (work["fact_id"],)
    ).fetchone()[0] == 1

    with pytest.raises(ServiceRequestError, match="not found") as hidden:
        service.handle(
            client_b_requested_too_much,
            _request("evidence-hidden", "memory.evidence", fact_id=work["fact_id"]),
        )
    assert hidden.value.code == "not_found"


def test_search_answers_who_did_or_learned_with_visible_only_attribution(setup):
    conn, service, contexts = setup
    private = _write(
        service,
        contexts["client-b"],
        "attribution-private",
        "Client B learned that Zephyr deploys on Tuesday",
    )
    work = _write(
        service,
        contexts["client-a"],
        "attribution-work",
        "Client A independently verified the Zephyr Tuesday deployment",
        scope="work",
    )
    work_observation = conn.execute(
        "SELECT observation_id FROM memory_write_log WHERE fact_id = ?",
        (work["fact_id"],),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO fact_provenance(fact_id, observation_id, relation, created_at) "
        "VALUES (?, ?, 'supports', '2099-01-01T00:00:00Z')",
        (private["fact_id"], work_observation),
    )
    conn.commit()

    client_b = service.handle(
        contexts["client-b"],
        _request("who-client-b", "memory.search", query="Zephyr deploys Tuesday"),
    )["facts"][0]["attribution"]
    assert client_b == {
        "performed_by": "client-b",
        "agent_id": "client-b",
        "session_id": "client-b-session",
        "source_type": "agent_report",
        "repository": "enfold",
        "branch": "service-layer",
        "commit_sha": "abc123",
        "evidence_count": 1,
    }

    hermes = service.handle(
        contexts["hermes"],
        _request("who-hermes", "memory.search", query="Zephyr deploys Tuesday"),
    )["facts"][0]["attribution"]
    assert hermes["performed_by"] == "client-a"
    assert hermes["agent_id"] == "client-a"
    assert hermes["evidence_count"] == 2


def test_secret_and_credential_writes_are_factless_and_replayable(setup):
    conn, service, contexts = setup
    client_a_secret = _context(
        "client-a-install",
        "client-a",
        "client-a",
        scopes=("private", "secret"),
        session="client-a-secret-session",
    )
    before = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    secret = _write(
        service,
        client_a_secret,
        "secret-1",
        "Do not persist this material",
        scope="secret",
        sensitivity="secret",
    )
    credential = _write(
        service,
        contexts["hermes"],
        "secret-2",
        "api_key=supersecretcredentialvalue",
    )
    replay = _write(
        service,
        client_a_secret,
        "secret-1",
        "Do not persist this material",
        scope="secret",
        sensitivity="secret",
    )

    assert secret["outcome"] == credential["outcome"] == "rejected"
    assert secret["fact_id"] is credential["fact_id"] is None
    assert replay["replayed"] is True
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == before
    assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0


def test_state_and_attribution_fields_are_credential_screened(setup):
    conn, service, contexts = setup
    before = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    result = _write(
        service,
        contexts["client-a"],
        "screen-state",
        "An otherwise harmless claim",
        asserted_by="Victor",
        state={
            "subject_key": "service:database",
            "predicate_key": "connection",
            "object_value": "postgresql://user:password123@example.test/db",
        },
    )
    assert result["outcome"] == "rejected"
    assert result["fact_id"] is None
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == before
    assert "password123" not in conn.execute(
        "SELECT detail_json FROM memory_write_log WHERE idempotency_key = 'screen-state'"
    ).fetchone()[0]


def test_state_supersession_conflict_history_evidence_and_settled_search(setup):
    _conn, service, contexts = setup
    state = {"subject_key": "agent:wonny", "predicate_key": "preferred_model"}
    first = _write(
        service,
        contexts["hermes"],
        "state-1",
        "Wonny prefers model Terra 5.5",
        source_authority=0.8,
        state={**state, "object_value": "terra-5.5", "valid_from": "2026-07-11T10:00:00Z"},
        evidence_excerpt="Observed in the Hermes configuration.",
    )
    replacement = _write(
        service,
        contexts["client-a"],
        "state-2",
        "Wonny prefers model Terra 5.6",
        source_authority=0.8,
        state={**state, "object_value": "terra-5.6", "valid_from": "2026-07-12T10:00:00Z"},
    )
    conflicting = _write(
        service,
        contexts["client-b"],
        "state-3",
        "Wonny prefers model Unknown 1",
        source_authority=0.2,
        state={**state, "object_value": "unknown-1", "valid_from": "2026-07-13T10:00:00Z"},
    )

    assert first["outcome"] == "add"
    assert replacement["outcome"] == "supersede"
    assert conflicting["outcome"] == "conflict"
    history = service.handle(
        contexts["hermes"],
        _request("history-1", "memory.history", **state, scope="private"),
    )["facts"]
    assert [fact["fact_id"] for fact in history] == [
        first["fact_id"], replacement["fact_id"], conflicting["fact_id"]
    ]
    assert history[0]["superseded_by"] == replacement["fact_id"]

    evidence = service.handle(
        contexts["hermes"],
        _request("evidence-1", "memory.evidence", fact_id=first["fact_id"]),
    )
    assert evidence["evidence"][0]["evidence_excerpt"] == (
        "Observed in the Hermes configuration."
    )
    assert evidence["evidence"][0]["client_id"] == "hermes-install"

    conflicts = service.handle(
        contexts["hermes"], _request("conflicts-1", "memory.conflicts")
    )["conflicts"]
    assert len(conflicts) == 1
    assert set(conflicts[0]["member_fact_ids"]) == {
        replacement["fact_id"], conflicting["fact_id"]
    }
    assert {member["fact_id"] for member in conflicts[0]["members"]} == set(
        conflicts[0]["member_fact_ids"]
    )
    assert service.handle(
        contexts["hermes"],
        _request("search-conflict", "memory.search", query="prefers model"),
    )["facts"] == []


def test_authorized_conflict_resolution_restores_settled_truth_and_audits(setup):
    conn, service, contexts = setup
    slot = {"subject_key": "agent:wonny", "predicate_key": "model"}
    first = _write(
        service, contexts["hermes"], "resolve-1", "Wonny uses Terra",
        source_authority=0.8,
        state={**slot, "object_value": "terra", "valid_from": "2026-07-11T10:00:00Z"},
    )
    other = _write(
        service, contexts["client-b"], "resolve-2", "Wonny uses Model Z",
        source_authority=0.2,
        state={**slot, "object_value": "model-z", "valid_from": "2026-07-12T10:00:00Z"},
    )
    conflict_id = other["detail"]["conflict_id"]

    with pytest.raises(ServiceRequestError) as denied:
        service.handle(
            contexts["client-b"],
            _request(
                "resolve-denied", "memory.resolve_conflict",
                conflict_id=conflict_id,
                resolution_fact_id=first["fact_id"],
                reason="self-claimed authority",
            ),
        )
    assert denied.value.code == "access_denied"

    result = service.handle(
        contexts["hermes"],
        _request(
            "resolve-ok", "memory.resolve_conflict",
            conflict_id=conflict_id,
            resolution_fact_id=first["fact_id"],
            reason="Victor confirmed Terra",
        ),
    )["resolution"]
    assert result["resolution_fact_id"] == first["fact_id"]
    assert result["superseded_fact_ids"] == [other["fact_id"]]
    assert service.handle(
        contexts["hermes"], _request("settled", "memory.search", query="Terra")
    )["facts"][0]["fact_id"] == first["fact_id"]
    audit = conn.execute(
        """SELECT resolver_client_id, resolver_session_id, resolver_agent_id, reason
           FROM fact_conflict_resolutions WHERE conflict_id = ?""",
        (conflict_id,),
    ).fetchone()
    assert tuple(audit) == (
        "hermes-install", "wonny-session", "wonny", "Victor confirmed Terra"
    )


def test_undated_state_uses_observed_time_and_untyped_dedup_is_scope_local(setup):
    conn, service, contexts = setup
    slot = {"subject_key": "project:enfold", "predicate_key": "phase"}
    first = _write(
        service, contexts["client-a"], "time-1", "Enfold is in alpha",
        observed_at="2026-07-11T10:00:00Z", source_authority=0.5,
        state={**slot, "object_value": "alpha"},
    )
    second = _write(
        service, contexts["client-a"], "time-2", "Enfold is in beta",
        observed_at="2026-07-12T10:00:00Z", source_authority=0.5,
        state={**slot, "object_value": "beta"},
    )
    assert first["outcome"] == "add"
    assert second["outcome"] == "supersede"
    assert conn.execute(
        "SELECT valid_from FROM facts WHERE fact_id = ?", (second["fact_id"],)
    ).fetchone()[0] == "2026-07-12T10:00:00Z"

    private = _write(service, contexts["client-a"], "dedup-1", "Exact shared text")
    duplicate = _write(service, contexts["hermes"], "dedup-2", "Exact shared text")
    work = _write(
        service, contexts["client-a"], "dedup-3", "Exact shared text", scope="work"
    )
    assert duplicate["outcome"] == "dedup"
    assert duplicate["fact_id"] == private["fact_id"]
    assert work["fact_id"] != private["fact_id"]


def test_context_is_scoped_cited_current_and_conflict_safe(setup):
    _conn, service, contexts = setup
    private = _write(
        service,
        contexts["client-b"],
        "context-private",
        "CedarContext private registry has the private endpoint.",
    )
    work = _write(
        service,
        contexts["client-a"],
        "context-work",
        "CedarContext work registry has the work endpoint.",
        scope="work",
    )
    slot = {"subject_key": "atlas", "predicate_key": "backup_schedule"}
    stale = _write(
        service,
        contexts["hermes"],
        "context-stale",
        "Atlas backup schedule runs Monday.",
        source_authority=0.8,
        state={**slot, "object_value": "monday", "valid_from": "2026-07-10T10:00:00Z"},
    )
    current = _write(
        service,
        contexts["client-a"],
        "context-current",
        "Atlas backup schedule runs Tuesday.",
        source_authority=0.8,
        state={**slot, "object_value": "tuesday", "valid_from": "2026-07-11T10:00:00Z"},
    )
    conflict_slot = {"subject_key": "atlas", "predicate_key": "unstable_model"}
    _write(
        service,
        contexts["hermes"],
        "context-conflict-first",
        "UnstableAtlas model is Terra.",
        source_authority=0.8,
        state={**conflict_slot, "object_value": "terra", "valid_from": "2026-07-10T10:00:00Z"},
    )
    _write(
        service,
        contexts["client-b"],
        "context-conflict-second",
        "UnstableAtlas model is Model Z.",
        source_authority=0.2,
        state={**conflict_slot, "object_value": "model-z", "valid_from": "2026-07-11T10:00:00Z"},
    )

    private_pack = service.handle(
        contexts["client-a"],
        _request(
            "context-private-request",
            "memory.context",
            query="CedarContext registry",
            scope="private",
            token_budget=256,
        ),
    )
    assert private["fact_id"] in [fact["fact_id"] for fact in private_pack["facts"]]
    assert work["fact_id"] not in [fact["fact_id"] for fact in private_pack["facts"]]
    assert private_pack["facts"][0]["attribution"]["agent_id"] == "client-b"
    assert "fact:" + str(private["fact_id"]) in private_pack["markdown"]
    assert private_pack["token_estimate"]["used"] <= 256

    current_pack = service.handle(
        contexts["client-a"],
        _request(
            "context-current-request",
            "memory.context",
            query="Atlas backup schedule",
            token_budget=256,
        ),
    )
    current_ids = [fact["fact_id"] for fact in current_pack["facts"]]
    assert current["fact_id"] in current_ids
    assert stale["fact_id"] not in current_ids

    conflict_pack = service.handle(
        contexts["client-a"],
        _request(
            "context-conflict-request",
            "memory.context",
            query="UnstableAtlas model",
            token_budget=256,
        ),
    )
    assert conflict_pack["abstained"] is True
    assert conflict_pack["facts"] == []


def test_context_validates_scope_and_token_budget(setup):
    _conn, service, contexts = setup
    _write(service, contexts["client-a"], "context-validation", "Orchid context note")

    with pytest.raises(ServiceRequestError, match="token_budget"):
        service.handle(
            contexts["client-a"],
            _request("context-budget", "memory.context", query="Orchid", token_budget=15),
        )
    with pytest.raises(ServiceRequestError) as denied:
        service.handle(
            contexts["client-b"],
            _request(
                "context-scope", "memory.context", query="Orchid",
                token_budget=64, scope="work",
            ),
        )
    assert denied.value.code == "access_denied"


def test_unknown_clients_and_nested_identity_spoofing_fail_closed(setup):
    _conn, service, contexts = setup
    unknown = _context("unknown-install", "client-a", "client-a")
    with pytest.raises(ServiceRequestError) as denied:
        service.handle(unknown, _request("search-denied", "memory.search", query="anything"))
    assert denied.value.code == "access_denied"

    with pytest.raises(ServiceRequestError) as spoofed:
        service.handle(
            contexts["client-a"],
            _request(
                "write-spoof",
                "memory.write",
                idempotency_key="spoof",
                content="Nested identity must not be trusted",
                source_type="agent_report",
                metadata={"audit": {"client_id": "forged"}},
            ),
        )
    assert spoofed.value.code == "invalid_params"


def test_search_accepts_natural_language_and_reports_hybrid_capabilities(setup):
    _conn, service, contexts = setup
    written = _write(
        service,
        contexts["client-a"],
        "natural-search",
        "Orchid backups run every Tuesday",
    )

    response = service.handle(
        contexts["client-a"],
        _request(
            "natural-search-request",
            "memory.search",
            query='When does Orchid backup run? (current) + "schedule"',
        ),
    )

    assert response["facts"][0]["fact_id"] == written["fact_id"]
    assert "score" in response["facts"][0]
    assert response["retrieval"]["filter_before_dense_ranking"] is True
    assert response["retrieval"]["embedder_production_ready"] is False
    assert response["retrieval"]["natural_language_query_parser"] == "quoted_token_or_v1"


def test_service_search_serializes_dense_scores_as_builtin_json_numbers(tmp_path):
    class NumpyEmbedder:
        identity = "numpy-service-regression"
        production_ready = False

        def embed_query(self, _text):
            return np.asarray((1.0, 0.0), dtype=np.float32)

        def embed_documents(self, texts):
            return tuple(
                np.asarray((1.0, 0.0), dtype=np.float32)
                for _text in texts
            )

    conn = _store(tmp_path)
    service = EnfoldService(
        conn,
        MemoryPolicy({"client-a-install": ("private",)}),
        retriever_factory=lambda connection, scopes: HybridRetriever(
            connection, NumpyEmbedder(), allowed_scopes=scopes
        ),
    )
    context = _context("client-a-install", "client-a", "client-a", scopes=("private",))
    _write(
        service,
        context,
        "numpy-score",
        "The service must serialize dense scores through JSON.",
    )

    response = service.handle(
        context,
        _request("numpy-search", "memory.search", query="dense scores"),
    )

    assert type(response["facts"][0]["dense_score"]) is float
    assert type(response["facts"][0]["score"]) is float
    json.dumps(response, allow_nan=False)
    conn.close()


def test_daemon_owned_extraction_surface_enqueues_without_model_call(tmp_path):
    conn = _store(tmp_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS extract_queue (id INTEGER PRIMARY KEY, payload TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', payload_hash TEXT)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS extraction_active_hash ON extract_queue(payload_hash) "
        "WHERE status IN ('pending', 'processing')"
    )
    conn.commit()
    service = EnfoldService(
        conn,
        MemoryPolicy({"client-a-install": ("private",)}),
        extraction_enqueuer=ExtractionEnqueuer(conn),
    )
    context = _context("client-a-install", "client-a", "client-a", scopes=("private",))
    request = _request(
        "extract-1",
        "memory.extraction.enqueue",
        transcript="Victor wants a shared local second brain.",
        source="session_end",
    )

    first = service.handle(context, request)
    second = service.handle(context, request)

    assert first["outcome"] == "queued"
    assert first["automatic_llm_extraction"] == "deferred"
    assert second["replayed"] is True
    assert conn.execute("SELECT count(*) FROM extract_queue").fetchone()[0] == 1
    conn.close()
