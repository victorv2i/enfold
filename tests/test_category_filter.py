"""Category filter on embedding-only search candidates."""

import numpy as np


def _setup_embedded_facts(provider):
    """Insert facts that FTS cannot match for the probe query, with embeddings
    identical to the query vector so they surface purely via embeddings."""
    same_vec = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    provider._fake_embedder.table["zzqq probe"] = same_vec
    facts = [
        ("alpha widget configuration detail", "tool"),
        ("beta widget configuration detail", "tool"),
        ("gamma project roadmap detail", "project"),
    ]
    ids = {}
    for content, category in facts:
        fid = provider._store.add_fact(content, category=category)
        ids[content] = fid
        provider._fake_embedder.table[content] = same_vec
        provider._embed_store.upsert(
            fid,
            same_vec,
            embedding_identity=provider._embedding_identity("document"),
        )
    return ids


def test_embedding_only_results_respect_category(make_provider):
    provider = make_provider()
    _setup_embedded_facts(provider)

    results = provider.search("zzqq probe", category="tool", min_trust=0.0, limit=10)
    assert results, "embedding-only candidates should surface"
    assert all(r["category"] == "tool" for r in results)

    results_project = provider.search("zzqq probe", category="project", min_trust=0.0, limit=10)
    assert results_project
    assert all(r["category"] == "project" for r in results_project)


def test_embedding_only_results_unfiltered_without_category(make_provider):
    provider = make_provider()
    _setup_embedded_facts(provider)
    results = provider.search("zzqq probe", min_trust=0.0, limit=10)
    assert {r["category"] for r in results} == {"tool", "project"}


def test_embedding_only_results_respect_min_trust(make_provider):
    provider = make_provider()
    ids = _setup_embedded_facts(provider)
    # Default trust is 0.5; a min_trust above it must exclude everything
    results = provider.search("zzqq probe", min_trust=0.9, limit=10)
    assert results == []
    assert ids  # facts exist, they were filtered, not missing
