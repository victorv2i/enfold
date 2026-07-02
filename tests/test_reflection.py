"""Sleep-time reflection: connection-drawing insight layer (reflection.py).

Covers cluster selection (shared-entity + cosine-neighborhood paths, hub
damping), the grounding gate (an ungrounded/citation-less insight is
rejected), NONE handling, dedup on insights, interval persistence across
restarts, invalidation cascade when a cited source fact is superseded, and
fully-inert behaviour when reflection is disabled.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import types

import numpy as np
import pytest

import fake_hermes

from enfold.reflection import (
    ensure_reflection_schema,
    get_last_run_at,
    set_last_run_at,
    select_clusters,
    _parse_reflection_response,
    reflect_on_cluster,
    run_reflection,
    invalidate_insights_citing,
)
from enfold.embed_store import EmbedStore
from enfold.temporal import ensure_temporal_schema, supersede


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conn(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "facts.db"), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(fake_hermes._SCHEMA)
    ensure_temporal_schema(conn)
    conn.commit()
    return conn


def _add_fact(conn, content, category="general", tags=""):
    cur = conn.execute(
        "INSERT INTO facts (content, category, tags) VALUES (?, ?, ?)",
        (content, category, tags),
    )
    conn.commit()
    return int(cur.lastrowid)


def _link_entity(conn, fact_id, name):
    row = conn.execute("SELECT entity_id FROM entities WHERE name = ?", (name,)).fetchone()
    if row is None:
        cur = conn.execute("INSERT INTO entities (name) VALUES (?)", (name,))
        conn.commit()
        entity_id = int(cur.lastrowid)
    else:
        entity_id = int(row["entity_id"])
    conn.execute(
        "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
        (fact_id, entity_id),
    )
    conn.commit()


def _embed(conn, fact_id, vec, identity="fake:test:document:none:v1"):
    store = EmbedStore(conn, embedding_identity=identity)
    store.upsert(fact_id, np.asarray(vec, dtype=np.float32), embedding_identity=identity)


def _llm_response(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


def _stop_background_worker(provider) -> None:
    """Halt the provider's background queue/reflection worker.

    These tests call ``provider.run_reflection`` directly to assert on its
    return value and call count deterministically; the background worker
    (started by ``initialize()``) would otherwise race the same opportunistic
    call from its own loop tick and double-count LLM invocations.
    """
    if provider._queue_stop is not None:
        provider._queue_stop.set()
    if provider._queue_wake is not None:
        provider._queue_wake.set()
    worker = provider._queue_worker
    if worker is not None:
        worker.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Interval persistence
# ---------------------------------------------------------------------------

def test_last_run_at_defaults_to_none(tmp_path):
    conn = _conn(tmp_path)
    ensure_reflection_schema(conn)
    assert get_last_run_at(conn) is None


def test_set_and_get_last_run_at_round_trips(tmp_path):
    conn = _conn(tmp_path)
    ensure_reflection_schema(conn)
    set_last_run_at(conn, 1000.0)
    assert get_last_run_at(conn) == 1000.0


def test_last_run_at_persists_across_a_new_connection_to_the_same_db(tmp_path):
    conn = _conn(tmp_path)
    ensure_reflection_schema(conn)
    set_last_run_at(conn, 500.0)
    conn.close()

    conn2 = sqlite3.connect(str(tmp_path / "facts.db"))
    conn2.row_factory = sqlite3.Row
    ensure_reflection_schema(conn2)
    assert get_last_run_at(conn2) == 500.0


def test_schema_migration_is_idempotent(tmp_path):
    conn = _conn(tmp_path)
    ensure_reflection_schema(conn)
    ensure_reflection_schema(conn)  # must not raise


# ---------------------------------------------------------------------------
# Cluster selection: shared-entity path
# ---------------------------------------------------------------------------

def test_select_clusters_groups_facts_sharing_an_entity(tmp_path):
    conn = _conn(tmp_path)
    a = _add_fact(conn, "Alex Rivera moved to Springfield in March.")
    b = _add_fact(conn, "Alex Rivera started a new job at Skylark.")
    c = _add_fact(conn, "The Skylark dashboard port is 3100.")
    _link_entity(conn, a, "Alex Rivera")
    _link_entity(conn, b, "Alex Rivera")

    clusters = select_clusters(conn, max_clusters=3)
    ids = [set(cl) for cl in clusters]
    assert {a, b} in ids
    assert not any(c in cl for cl in ids)


def test_select_clusters_caps_at_max_clusters(tmp_path):
    conn = _conn(tmp_path)
    for name in ("Rivera", "Blackwood", "Cho", "Nakamura"):
        f1 = _add_fact(conn, f"{name} works remotely from Springfield.")
        f2 = _add_fact(conn, f"{name} prefers async standups.")
        _link_entity(conn, f1, name)
        _link_entity(conn, f2, name)

    clusters = select_clusters(conn, max_clusters=2)
    assert len(clusters) <= 2


def test_select_clusters_damps_hub_entities(tmp_path):
    conn = _conn(tmp_path)
    # A hub entity linked to many unrelated facts should not group them all
    # into one giant cluster.
    ids = []
    for i in range(30):
        fid = _add_fact(conn, f"Fact number {i} about the Skylark project.")
        _link_entity(conn, fid, "Skylark")
        ids.append(fid)

    clusters = select_clusters(conn, max_clusters=5, entity_hub_degree_limit=10)
    for cl in clusters:
        assert not (set(cl) <= set(ids) and len(cl) > 10)


# ---------------------------------------------------------------------------
# Cluster selection: cosine-neighborhood path (related but not duplicates)
# ---------------------------------------------------------------------------

def test_select_clusters_groups_by_cosine_neighborhood(tmp_path):
    conn = _conn(tmp_path)
    a = _add_fact(conn, "The user works best in the early morning.")
    b = _add_fact(conn, "The user schedules deep-focus work before noon.")
    c = _add_fact(conn, "The user's favorite lunch spot is a taco truck.")

    # a/b share a cosine-neighborhood (0.75-0.92): related but not duplicates.
    _embed(conn, a, [1.0, 0.0, 0.0])
    _embed(conn, b, [0.8, 0.2, 0.0])  # cosine ~0.97 -> renormalized below
    _embed(conn, c, [0.0, 1.0, 0.0])  # unrelated, cosine ~0

    embed_store = EmbedStore(conn, embedding_identity="fake:test:document:none:v1")
    clusters = select_clusters(
        conn, max_clusters=3, embed_store=embed_store,
        embedding_identity="fake:test:document:none:v1",
        cosine_low=0.5, cosine_high=0.999,
    )
    ids = [set(cl) for cl in clusters]
    assert {a, b} in ids
    assert not any(c in cl for cl in ids)


def test_select_clusters_excludes_near_duplicates_above_cosine_high(tmp_path):
    conn = _conn(tmp_path)
    a = _add_fact(conn, "The Skylark dashboard port is 3100.")
    b = _add_fact(conn, "The Skylark dashboard port is 3100 exactly.")
    _embed(conn, a, [1.0, 0.0, 0.0])
    _embed(conn, b, [1.0, 0.0001, 0.0])  # near-identical, cosine > 0.92

    embed_store = EmbedStore(conn, embedding_identity="fake:test:document:none:v1")
    clusters = select_clusters(
        conn, max_clusters=3, embed_store=embed_store,
        embedding_identity="fake:test:document:none:v1",
        cosine_low=0.75, cosine_high=0.92,
    )
    assert not any({a, b} <= set(cl) for cl in clusters)


def test_select_clusters_excludes_invalid_facts(tmp_path):
    conn = _conn(tmp_path)
    a = _add_fact(conn, "Alex Rivera moved to Springfield in March.")
    b = _add_fact(conn, "Alex Rivera started a new job at Skylark.")
    _link_entity(conn, a, "Alex Rivera")
    _link_entity(conn, b, "Alex Rivera")
    supersede(conn, a, b)

    clusters = select_clusters(conn, max_clusters=3)
    assert not any(a in cl for cl in clusters)


def test_select_clusters_excludes_legacy_superseded_facts(tmp_path):
    conn = _conn(tmp_path)
    legacy = _add_fact(conn, "SUPERSEDED 2026-06-01: Alex Rivera moved to Springfield.")
    active = _add_fact(conn, "Alex Rivera started a new job at Skylark.")
    _link_entity(conn, legacy, "Alex Rivera")
    _link_entity(conn, active, "Alex Rivera")

    clusters = select_clusters(conn, max_clusters=3)
    assert not any(legacy in cl for cl in clusters)


def test_select_clusters_excludes_insight_facts_as_sources(tmp_path):
    conn = _conn(tmp_path)
    source = _add_fact(conn, "Alex Rivera moved to Springfield in March.")
    insight = _add_fact(
        conn,
        "Alex Rivera relocation timing is tied to the Skylark role.",
        category="insight",
        tags=f"source_facts:{source}",
    )
    _link_entity(conn, source, "Alex Rivera")
    _link_entity(conn, insight, "Alex Rivera")

    clusters = select_clusters(conn, max_clusters=3)
    assert not any(insight in cl for cl in clusters)


# ---------------------------------------------------------------------------
# Grounding gate: citations required, no invented facts
# ---------------------------------------------------------------------------

def _cluster_facts():
    return [
        {"fact_id": 1, "content": "Alex Rivera moved to Springfield in March."},
        {"fact_id": 2, "content": "Alex Rivera started a new job at Skylark."},
    ]


def test_parse_reflection_accepts_a_grounded_insight():
    raw = json.dumps({
        "insight": "Alex Rivera relocated to Springfield for a new role at Skylark.",
        "source_fact_ids": [1, 2],
    })
    result = _parse_reflection_response(raw, valid_ids={1, 2})
    assert result is not None
    assert result["source_fact_ids"] == [1, 2]


def test_parse_reflection_rejects_missing_citations():
    raw = json.dumps({
        "insight": "Alex Rivera relocated to Springfield for a new role at Skylark.",
        "source_fact_ids": [],
    })
    assert _parse_reflection_response(raw, valid_ids={1, 2}) is None


def test_parse_reflection_rejects_citations_outside_the_cluster():
    raw = json.dumps({
        "insight": "Alex Rivera relocated to Springfield for a new role at Skylark.",
        "source_fact_ids": [1, 999],
    })
    assert _parse_reflection_response(raw, valid_ids={1, 2}) is None


def test_parse_reflection_handles_none_response():
    assert _parse_reflection_response("NONE", valid_ids={1, 2}) is None
    assert _parse_reflection_response('{"insight": "NONE", "source_fact_ids": []}', valid_ids={1, 2}) is None


def test_parse_reflection_rejects_garbage():
    assert _parse_reflection_response("not json at all", valid_ids={1, 2}) is None
    assert _parse_reflection_response("", valid_ids={1, 2}) is None


def test_reflect_on_cluster_rejects_ungrounded_llm_output(monkeypatch):
    import agent.auxiliary_client as aux

    # LLM returns an insight but cites no sources: must be rejected regardless
    # of how plausible the text sounds.
    aux.call_llm = lambda **kwargs: _llm_response(json.dumps({
        "insight": "Alex Rivera is planning to leave Skylark soon.",
        "source_fact_ids": [],
    }))

    result = reflect_on_cluster(
        _cluster_facts(), provider="testprov", model="testmodel",
    )
    assert result is None


def test_reflect_on_cluster_accepts_grounded_output():
    import agent.auxiliary_client as aux

    aux.call_llm = lambda **kwargs: _llm_response(json.dumps({
        "insight": "Alex Rivera relocated to Springfield around the same time as starting at Skylark.",
        "source_fact_ids": [1, 2],
    }))

    result = reflect_on_cluster(
        _cluster_facts(), provider="testprov", model="testmodel",
    )
    assert result is not None
    assert result["source_fact_ids"] == [1, 2]


def test_reflect_on_cluster_returns_none_for_explicit_none(monkeypatch):
    import agent.auxiliary_client as aux

    aux.call_llm = lambda **kwargs: _llm_response("NONE")

    result = reflect_on_cluster(
        _cluster_facts(), provider="testprov", model="testmodel",
    )
    assert result is None


# ---------------------------------------------------------------------------
# Provider-level run_reflection: interval gating, dedup, disabled-by-default
# ---------------------------------------------------------------------------

def test_reflection_disabled_by_default(make_provider):
    provider = make_provider(extraction_provider="testprov", extraction_model="testmodel")
    assert provider._reflection_enabled is False


def test_run_reflection_inert_when_disabled(make_provider, aux_module):
    calls = []
    aux_module.call_llm = lambda **kwargs: calls.append(1) or _llm_response("NONE")
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel",
        reflection_enabled=False,
    )
    _stop_background_worker(provider)
    _seed_related_pair(provider)

    provider.run_reflection(now=time.time())

    assert calls == []
    facts = provider._store.list_facts(min_trust=0.0, limit=50, category="insight")
    assert facts == []


def _seed_related_pair(provider):
    # The parent store auto-extracts + links multi-word capitalized entities
    # on add_fact, so "Alex Rivera" links both facts without any manual
    # fact_entities bookkeeping.
    a = provider._store.add_fact("Alex Rivera moved to Springfield in March.")
    b = provider._store.add_fact("Alex Rivera started a new job at Skylark.")
    return a, b


def test_run_reflection_inserts_a_grounded_insight_as_a_fact(make_provider, aux_module):
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel",
        reflection_enabled=True,
    )
    _stop_background_worker(provider)
    a, b = _seed_related_pair(provider)

    aux_module.call_llm = lambda **kwargs: _llm_response(json.dumps({
        "insight": "Alex Rivera relocated to Springfield around the same time as starting at Skylark.",
        "source_fact_ids": [a, b],
    }))

    provider.run_reflection(now=time.time())

    facts = provider._store.list_facts(min_trust=0.0, limit=50, category="insight")
    assert len(facts) == 1
    assert f"source_facts:{a},{b}" in facts[0]["tags"] or f"source_facts:{b},{a}" in facts[0]["tags"]


def test_run_reflection_respects_dedup_gate(make_provider, aux_module):
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel",
        reflection_enabled=True,
    )
    _stop_background_worker(provider)
    a, b = _seed_related_pair(provider)
    existing = (
        "Alex Rivera relocated to Springfield around the same time as "
        "starting at Skylark."
    )
    provider._store.add_fact(existing, category="insight", tags=f"source_facts:{a},{b}")

    aux_module.call_llm = lambda **kwargs: _llm_response(json.dumps({
        "insight": existing,
        "source_fact_ids": [a, b],
    }))

    provider.run_reflection(now=time.time())

    facts = provider._store.list_facts(min_trust=0.0, limit=50, category="insight")
    assert len(facts) == 1  # the reflection pass never flooded a duplicate in


def test_run_reflection_respects_min_interval(make_provider, aux_module):
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel",
        reflection_enabled=True, reflection_interval_hours=24,
    )
    _stop_background_worker(provider)
    a, b = _seed_related_pair(provider)
    calls = []
    aux_module.call_llm = lambda **kwargs: calls.append(1) or _llm_response("NONE")

    now = time.time()
    provider.run_reflection(now=now)
    assert len(calls) == 1

    provider.run_reflection(now=now + 60)  # 1 minute later: too soon
    assert len(calls) == 1

    provider.run_reflection(now=now + 25 * 3600)  # >24h later: due again
    assert len(calls) == 2


def test_run_reflection_interval_persists_across_reinitialize(make_provider, aux_module, tmp_path):
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel",
        reflection_enabled=True, reflection_interval_hours=24,
    )
    _stop_background_worker(provider)
    _seed_related_pair(provider)
    aux_module.call_llm = lambda **kwargs: _llm_response("NONE")

    now = time.time()
    provider.run_reflection(now=now)

    provider.initialize("test-session-2")  # simulate a restart against same db
    _stop_background_worker(provider)
    calls = []
    aux_module.call_llm = lambda **kwargs: calls.append(1) or _llm_response("NONE")
    provider.run_reflection(now=now + 60)  # still within the 24h window
    assert calls == []


def test_run_reflection_caps_clusters_per_run(make_provider, aux_module):
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel",
        reflection_enabled=True, reflection_max_clusters=1,
    )
    _stop_background_worker(provider)
    for name in ("Rivera Delacroix", "Blackwood Nakamura"):
        provider._store.add_fact(f"{name} works remotely from Springfield.")
        provider._store.add_fact(f"{name} prefers async standups.")

    calls = []
    aux_module.call_llm = lambda **kwargs: calls.append(1) or _llm_response("NONE")
    provider.run_reflection(now=time.time())
    assert len(calls) == 1


def test_run_reflection_revalidates_cited_sources_before_insert(tmp_path, monkeypatch, caplog):
    import enfold.reflection as reflection

    conn = _conn(tmp_path)
    a = _add_fact(conn, "Alex Rivera moved to Springfield in March.")
    b = _add_fact(conn, "Alex Rivera started a new job at Skylark.")
    _link_entity(conn, a, "Alex Rivera")
    _link_entity(conn, b, "Alex Rivera")
    inserted = []

    def reflect_and_stale_source(facts, **kwargs):
        conn.execute(
            "UPDATE facts SET invalid_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
            (a,),
        )
        conn.commit()
        return {
            "insight": "Alex Rivera relocation timing is tied to the Skylark role.",
            "source_fact_ids": [a, b],
        }

    monkeypatch.setattr(reflection, "reflect_on_cluster", reflect_and_stale_source)
    caplog.set_level(logging.DEBUG, logger="enfold.reflection")

    count = run_reflection(
        conn,
        now=time.time(),
        enabled=True,
        interval_hours=24,
        max_clusters=3,
        provider="testprov",
        model="testmodel",
        insert_fact=lambda content, category, tags: inserted.append((content, category, tags)) or 999,
    )

    assert count == 0
    assert inserted == []
    assert "skipped insight with stale source facts" in caplog.text


# ---------------------------------------------------------------------------
# Invalidation cascade: superseding a source fact stales its insight
# ---------------------------------------------------------------------------

def test_invalidate_insights_citing_marks_stale(tmp_path):
    conn = _conn(tmp_path)
    a = _add_fact(conn, "Alex Rivera moved to Springfield in March.")
    b = _add_fact(conn, "Alex Rivera started a new job at Skylark.")
    insight_id = _add_fact(
        conn,
        "Alex Rivera relocated to Springfield around the same time as starting at Skylark.",
        category="insight", tags=f"source_facts:{a},{b}",
    )

    invalidate_insights_citing(conn, a)

    row = conn.execute(
        "SELECT invalid_at FROM facts WHERE fact_id = ?", (insight_id,)
    ).fetchone()
    assert row["invalid_at"] is not None


def test_invalidate_insights_citing_ignores_unrelated_insights(tmp_path):
    conn = _conn(tmp_path)
    a = _add_fact(conn, "Alex Rivera moved to Springfield in March.")
    b = _add_fact(conn, "Alex Rivera started a new job at Skylark.")
    c = _add_fact(conn, "Unrelated fact about a different topic.")
    insight_id = _add_fact(
        conn,
        "Alex Rivera relocated to Springfield around the same time as starting at Skylark.",
        category="insight", tags=f"source_facts:{a},{b}",
    )

    invalidate_insights_citing(conn, c)

    row = conn.execute(
        "SELECT invalid_at FROM facts WHERE fact_id = ?", (insight_id,)
    ).fetchone()
    assert row["invalid_at"] is None


def test_provider_supersede_cascades_to_dependent_insight(make_provider, aux_module):
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel",
        reflection_enabled=True,
    )
    _stop_background_worker(provider)
    a, b = _seed_related_pair(provider)

    aux_module.call_llm = lambda **kwargs: _llm_response(json.dumps({
        "insight": "Alex Rivera relocated to Springfield around the same time as starting at Skylark.",
        "source_fact_ids": [a, b],
    }))
    provider.run_reflection(now=time.time())

    insight = provider._store.list_facts(min_trust=0.0, limit=50, category="insight")[0]

    new_id = provider._store.add_fact("Alex Rivera moved to Portland instead.")
    provider._supersede_fact(a, new_id)

    row = provider._store._conn.execute(
        "SELECT invalid_at FROM facts WHERE fact_id = ?", (insight["fact_id"],)
    ).fetchone()
    assert row["invalid_at"] is not None
