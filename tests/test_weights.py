"""Weight rescale invariants and blend partition."""

import pytest


@pytest.mark.parametrize("ew", [0.3, 0.45, 0.7])
def test_holographic_weights_sum_to_one_for_any_embedding_weight(make_provider, ew):
    provider = make_provider(embedding_weight=ew)
    r = provider._retriever
    assert r.fts_weight + r.jaccard_weight + r.hrr_weight == pytest.approx(1.0)
    # Rescale divides by the parent weights' own sum (0.7), not (1 - ew)
    assert r.fts_weight == pytest.approx(0.3 / 0.7)
    assert r.jaccard_weight == pytest.approx(0.2 / 0.7)
    assert r.hrr_weight == pytest.approx(0.2 / 0.7)


@pytest.mark.parametrize("ew", [0.3, 0.45, 0.7])
def test_blend_score_partitions_the_budget(hp, ew):
    # Max holographic score (relevance 1.0 at trust 1.0) plus max embedding
    # similarity must exactly exhaust the budget.
    assert hp._blend_score(1.0, 1.0, 1.0, ew) == pytest.approx(1.0)
    # The holographic side alone is capped at its (1 - ew) share
    assert hp._blend_score(1.0, None, 1.0, ew) == pytest.approx(1.0 - ew)
    # Cosine -1 maps to a zero embedding contribution, not a penalty
    assert hp._blend_score(0.5, -1.0, 1.0, ew) == pytest.approx((1.0 - ew) * 0.5)


def test_blend_score_trust_scales_embedding_term(hp):
    ew = 0.4
    trust = 0.5
    # holo 0, sim 1.0 at trust 0.5: embedding term is ew * 1.0 * trust
    assert hp._blend_score(0.0, 1.0, trust, ew) == pytest.approx(ew * trust)


def test_facts_without_embeddings_cannot_earn_the_embedding_slice(hp):
    ew = 0.3
    with_emb = hp._blend_score(0.4, 0.2, 0.8, ew)
    without_emb = hp._blend_score(0.4, None, 0.8, ew)
    assert with_emb > without_emb
    assert without_emb == pytest.approx(0.7 * 0.4)
