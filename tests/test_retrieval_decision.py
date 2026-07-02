"""Calibrated retrieval decision gates for Enfold search."""

import numpy as np


def _result(fact_id: int, score: float, *, trust: float = 0.9, content: str | None = None) -> dict:
    return {
        "fact_id": fact_id,
        "content": content or f"fact {fact_id}",
        "category": "general",
        "tags": "",
        "trust_score": trust,
        "retrieval_count": 0,
        "helpful_count": 0,
        "created_at": "2026-01-01 00:00:00",
        "updated_at": "2026-01-01 00:00:00",
        "score": score,
    }


def _force_holographic_only(provider, monkeypatch, rows):
    provider._embedder_available = False
    monkeypatch.setattr(provider._retriever, "search", lambda *args, **kwargs: list(rows))


def test_retrieval_decision_is_off_by_default(make_provider, monkeypatch):
    provider = make_provider()
    _force_holographic_only(provider, monkeypatch, [_result(1, 0.2)])

    results = provider.search("ambiguous", limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [1]


def test_retrieval_decision_thresholds_are_ignored_until_enabled(make_provider, monkeypatch):
    provider = make_provider(retrieval_decision_min_score=0.9, retrieval_decision_min_trust=0.9)
    _force_holographic_only(provider, monkeypatch, [_result(1, 0.2, trust=0.2)])

    results = provider.search("ambiguous", limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [1]


def test_retrieval_decision_enabled_without_thresholds_preserves_results(make_provider, monkeypatch):
    provider = make_provider(retrieval_decision_enabled=True)
    _force_holographic_only(provider, monkeypatch, [_result(1, 0.2), _result(2, 0.1)])

    results = provider.search("ambiguous", limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [1, 2]


def test_retrieval_decision_zero_score_threshold_is_explicit_noop(make_provider, monkeypatch):
    provider = make_provider(retrieval_decision_enabled=True, retrieval_decision_min_score=0)
    _force_holographic_only(provider, monkeypatch, [_result(1, 0.0), _result(2, 0.1)])

    results = provider.search("ambiguous", limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [1, 2]


def test_retrieval_decision_config_helpers_parse_false_and_zero(hp):
    assert hp._cfg_bool("false", True) is False
    assert hp._cfg_bool("off", True) is False
    assert hp._cfg_float("0") == 0.0


def test_retrieval_decision_min_score_filters_low_confidence_results(make_provider, monkeypatch):
    provider = make_provider(retrieval_decision_enabled=True, retrieval_decision_min_score=0.5)
    _force_holographic_only(provider, monkeypatch, [
        _result(1, 0.49),
        _result(2, 0.5),
        _result(3, 0.51),
    ])

    results = provider.search("ambiguous", limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [2, 3]


def test_retrieval_decision_min_trust_filters_low_trust_results(make_provider, monkeypatch):
    provider = make_provider(retrieval_decision_enabled=True, retrieval_decision_min_trust=0.5)
    _force_holographic_only(provider, monkeypatch, [
        _result(1, 0.9, trust=0.49),
        _result(2, 0.8, trust=0.51),
    ])

    results = provider.search("ambiguous", limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [2]


def test_retrieval_decision_min_margin_abstains_on_ambiguous_top_scores(make_provider, monkeypatch):
    provider = make_provider(retrieval_decision_enabled=True, retrieval_decision_min_margin=0.02)
    _force_holographic_only(provider, monkeypatch, [
        _result(1, 0.71),
        _result(2, 0.70),
    ])

    assert provider.search("ambiguous", limit=5, bump=False) == []


def test_retrieval_decision_margin_uses_filtered_candidate_pool(make_provider, monkeypatch):
    provider = make_provider(
        retrieval_decision_enabled=True,
        retrieval_decision_min_margin=0.02,
        retrieval_decision_min_trust=0.5,
    )
    _force_holographic_only(provider, monkeypatch, [
        _result(1, 0.71, trust=0.1),
        _result(2, 0.70, trust=0.9),
    ])

    results = provider.search("ambiguous", limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [2]


def test_retrieval_decision_score_filter_runs_before_margin_gate(make_provider, monkeypatch):
    provider = make_provider(
        retrieval_decision_enabled=True,
        retrieval_decision_min_score=0.5,
        retrieval_decision_min_margin=0.02,
    )
    _force_holographic_only(provider, monkeypatch, [
        _result(1, 0.51),
        _result(2, 0.49),
    ])

    results = provider.search("ambiguous", limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [1]


def test_retrieval_decision_all_gates_can_apply_together(make_provider, monkeypatch):
    provider = make_provider(
        retrieval_decision_enabled=True,
        retrieval_decision_min_score=0.5,
        retrieval_decision_min_trust=0.5,
        retrieval_decision_min_margin=0.02,
    )
    _force_holographic_only(provider, monkeypatch, [
        _result(1, 0.9, trust=0.1),
        _result(2, 0.6, trust=0.9),
        _result(3, 0.55, trust=0.9),
        _result(4, 0.49, trust=0.9),
    ])

    results = provider.search("ambiguous", limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [2, 3]


def test_retrieval_decision_single_result_survives_margin_gate(make_provider, monkeypatch):
    provider = make_provider(retrieval_decision_enabled=True, retrieval_decision_min_margin=0.02)
    _force_holographic_only(provider, monkeypatch, [_result(1, 0.71)])

    results = provider.search("ambiguous", limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [1]


def test_search_pre_filters_superseded_rows_before_decision_gate(make_provider, monkeypatch):
    provider = make_provider(retrieval_decision_enabled=True)
    _force_holographic_only(provider, monkeypatch, [
        _result(1, 0.9, content="Superseded 2026-06-28: old value"),
        _result(2, 0.8, content="current value"),
    ])

    results = provider.search("value", limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [2]


def test_retrieval_decision_gate_suppresses_stale_prefixed_rows(make_provider):
    provider = make_provider(retrieval_decision_enabled=True)

    results = provider._apply_retrieval_decision([
        _result(1, 0.9, content="Superseded 2026-06-28: old value"),
        _result(2, 0.8, content="current value"),
    ])

    assert [r["fact_id"] for r in results] == [2]


def test_retrieval_decision_does_not_bump_filtered_rows(make_provider, monkeypatch):
    provider = make_provider(retrieval_decision_enabled=True, retrieval_decision_min_score=0.5)
    fact_id = provider._store.add_fact("low confidence memory", category="general")
    _force_holographic_only(provider, monkeypatch, [
        _result(fact_id, 0.1, content="low confidence memory"),
    ])

    assert provider.search("ambiguous", limit=5, bump=True) == []
    row = provider._store._conn.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id = ?",
        (fact_id,),
    ).fetchone()
    assert row["retrieval_count"] == 0


def test_retrieval_decision_filters_hybrid_embedding_results(make_provider):
    provider = make_provider(
        embedding_weight=1.0,
        retrieval_decision_enabled=True,
        retrieval_decision_min_score=0.5,
    )
    query_vec = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    weak_vec = np.array([-1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    provider._fake_embedder.table["semantic probe"] = query_vec

    strong_id = provider._store.add_fact("strong semantic memory", category="general")
    weak_id = provider._store.add_fact("weak semantic memory", category="general")
    provider._fake_embedder.table["strong semantic memory"] = query_vec
    provider._fake_embedder.table["weak semantic memory"] = weak_vec
    provider._embed_store.upsert(
        strong_id,
        query_vec,
        embedding_identity=provider._embedding_identity("document"),
    )
    provider._embed_store.upsert(
        weak_id,
        weak_vec,
        embedding_identity=provider._embedding_identity("document"),
    )

    results = provider.search("semantic probe", min_trust=0.0, limit=5, bump=False)

    assert [r["fact_id"] for r in results] == [strong_id]
