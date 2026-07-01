"""Instruction-prefix support and configurable holographic weights.

These cover the two code changes that let the benchmark-optimal config be
expressed purely through config: applying each model's documented query/passage
prefixes, and being able to down-weight or disable the HRR signal.
"""

import pytest


# ---------------------------------------------------------------------------
# Instruction prefixes
# ---------------------------------------------------------------------------

def test_prefix_policy_defaults_to_none_and_applies_nothing(make_provider):
    p = make_provider(fastembed_model="BAAI/bge-large-en-v1.5")
    assert p._embedding_prefix_policy == "none"
    p._store.add_fact("the user prefers dark mode", category="general")
    p.rebuild_embeddings()
    fake = p._fake_embedder
    fake.embed_calls.clear()
    p.search("does the user like a low light theme", min_trust=0.0, limit=5)
    assert fake.embed_calls, "query should have been embedded"
    assert all("Represent this sentence" not in c for c in fake.embed_calls)


def test_auto_policy_applies_bge_query_instruction(make_provider):
    p = make_provider(embedding_prefix_policy="auto",
                      fastembed_model="BAAI/bge-large-en-v1.5")
    p._store.add_fact("the user prefers dark mode", category="general")
    p.rebuild_embeddings()
    fake = p._fake_embedder
    # bge document prefix is empty: the stored fact is embedded verbatim
    assert any(c == "the user prefers dark mode" for c in fake.embed_calls)
    fake.embed_calls.clear()
    p.search("does the user like a low light theme", min_trust=0.0, limit=5)
    assert any(
        c.startswith("Represent this sentence for searching relevant passages: ")
        for c in fake.embed_calls
    )


def test_auto_policy_applies_embeddinggemma_doc_and_query(make_provider):
    p = make_provider(embedding_prefix_policy="auto",
                      embedding_backend="ollama", ollama_model="embeddinggemma")
    p._store.add_fact("database backups run at 2am", category="general")
    p.rebuild_embeddings()
    fake = p._fake_embedder
    assert any(c.startswith("title: none | text: ") for c in fake.embed_calls)
    fake.embed_calls.clear()
    p.search("when do snapshots happen", min_trust=0.0, limit=5)
    assert any(c.startswith("task: search result | query: ") for c in fake.embed_calls)


def test_explicit_prefix_overrides_registry(make_provider):
    p = make_provider(embedding_prefix_policy="auto",
                      fastembed_model="BAAI/bge-large-en-v1.5",
                      embedding_query_prefix="QQ: ",
                      embedding_document_prefix="DD: ")
    p._store.add_fact("alpha beta", category="general")
    p.rebuild_embeddings()
    fake = p._fake_embedder
    assert any(c.startswith("DD: ") for c in fake.embed_calls)
    fake.embed_calls.clear()
    p.search("gamma delta", min_trust=0.0, limit=5)
    assert any(c.startswith("QQ: ") for c in fake.embed_calls)


def test_prefix_policy_is_part_of_embedding_identity(make_provider):
    p_none = make_provider(embedding_prefix_policy="none")
    p_auto = make_provider(embedding_prefix_policy="auto")
    assert p_none._embedding_identity("document") != p_auto._embedding_identity("document")


# ---------------------------------------------------------------------------
# Configurable holographic weights (disable / down-weight HRR)
# ---------------------------------------------------------------------------

def test_default_weights_unchanged(make_provider):
    r = make_provider()._retriever
    assert r.fts_weight == pytest.approx(0.3 / 0.7)
    assert r.jaccard_weight == pytest.approx(0.2 / 0.7)
    assert r.hrr_weight == pytest.approx(0.2 / 0.7)


def test_hrr_weight_zero_disables_hrr_and_rescales(make_provider):
    r = make_provider(hrr_weight=0.0)._retriever
    assert r.hrr_weight == 0.0
    assert r.fts_weight + r.jaccard_weight == pytest.approx(1.0)
    assert r.fts_weight == pytest.approx(0.3 / 0.5)
    assert r.jaccard_weight == pytest.approx(0.2 / 0.5)


def test_custom_weights_rescale_to_one(make_provider):
    r = make_provider(fts_weight=0.5, jaccard_weight=0.5, hrr_weight=0.0)._retriever
    assert r.fts_weight + r.jaccard_weight + r.hrr_weight == pytest.approx(1.0)
    assert r.fts_weight == pytest.approx(0.5)


def test_zero_total_weight_falls_back_to_defaults(make_provider):
    r = make_provider(fts_weight=0.0, jaccard_weight=0.0, hrr_weight=0.0)._retriever
    assert r.fts_weight == pytest.approx(0.3 / 0.7)
