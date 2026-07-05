from __future__ import annotations

from memory_eval.autotune import (
    TrialScore,
    _base_eval_config,
    _is_better,
    _normalize_inactive_knobs,
    _parse_scalar,
)


def test_parse_scalar_matches_flat_yaml_fallback_needs():
    assert _parse_scalar("true") is True
    assert _parse_scalar("False") is False
    assert _parse_scalar("null") is None
    assert _parse_scalar("42") == 42
    assert _parse_scalar("0.45") == 0.45
    assert _parse_scalar('"embeddinggemma"') == "embeddinggemma"


def test_base_eval_config_uses_live_retrieval_values_but_forces_safe_eval_flags(tmp_path):
    db = tmp_path / "scratch.db"
    config = _base_eval_config(
        {
            "embedding_weight": 0.9,
            "embed_on_add": True,
            "dedup_on_add": True,
            "reflection_enabled": True,
            "extract_drain_batch": 10,
        },
        db,
    )

    assert config["db_path"] == str(db)
    assert config["embedding_weight"] == 0.9
    assert config["embed_on_add"] is False
    assert config["dedup_on_add"] is False
    assert config["reflection_enabled"] is False
    assert config["extract_drain_batch"] == 0


def test_is_better_rejects_any_stale_leak_increase_over_baseline():
    baseline = TrialScore(0.6, 0, 0.0, 20.0)
    incumbent = TrialScore(0.6, 0, 0.0, 20.0)
    challenger = TrialScore(0.8, 1, 0.1, 10.0)

    accepted, decision = _is_better(challenger, incumbent, baseline)

    assert accepted is False
    assert "stale_leak@1 increased" in decision


def test_normalize_inactive_knobs_restores_values_that_cannot_affect_trial():
    baseline = {
        "embedding_weight": 0.45,
        "entity_hub_degree_limit": 25,
        "retrieval_decision_min_score": None,
        "retrieval_decision_min_margin": None,
        "retrieval_decision_min_trust": None,
    }
    config = {
        "embedding_weight": 0.9,
        "entity_expansion": False,
        "entity_hub_degree_limit": 5,
        "retrieval_decision_enabled": False,
        "retrieval_decision_min_score": 0.7,
        "retrieval_decision_min_margin": 0.02,
        "retrieval_decision_min_trust": 0.6,
    }

    normalized = _normalize_inactive_knobs(
        config,
        baseline,
        active_backend={"dense_embeddings": False},
    )

    assert normalized["embedding_weight"] == 0.45
    assert normalized["entity_hub_degree_limit"] == 25
    assert normalized["retrieval_decision_min_score"] is None
    assert normalized["retrieval_decision_min_margin"] is None
    assert normalized["retrieval_decision_min_trust"] is None
