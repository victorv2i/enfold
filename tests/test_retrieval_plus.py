"""PlusFactRetriever: encode-once hot path and parent equivalence."""

import fake_hermes
import pytest

CONTENTS = [
    ("The user prefers pnpm for all node projects", "tool", "pnpm,node"),
    ("The deploy target for web projects is vercel", "tool", "deploy,vercel"),
    ("The tracker app uses sqlite for the fact store", "project", "tracker,sqlite"),
    ("The gateway restarts are scheduled overnight", "general", ""),
    ("The user keeps projects under the home projects directory", "project", "projects"),
    ("Node version is managed with mise for projects", "tool", "node,mise"),
    ("The memory plugin stores facts in sqlite", "project", "memory,sqlite"),
    ("The user likes dark themed dashboards for projects", "user_pref", "ui,dark"),
]


@pytest.fixture()
def populated_store(tmp_path):
    store = fake_hermes.MemoryStore(db_path=tmp_path / "facts.db", hrr_dim=64)
    for content, category, tags in CONTENTS:
        store.add_fact(content, category=category, tags=tags)
    yield store
    store.close()


def _spy_encode_text(monkeypatch):
    calls = []
    original = fake_hermes.hrr.encode_text

    def spy(text, dim=1024):
        calls.append(text)
        return original(text, dim)

    monkeypatch.setattr(fake_hermes.hrr, "encode_text", spy)
    return calls


def test_query_encoded_exactly_once_per_search(hp, populated_store, monkeypatch):
    retriever = hp.retrieval_plus.PlusFactRetriever(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    calls = _spy_encode_text(monkeypatch)
    results = retriever.search("projects", min_trust=0.0, limit=3)
    assert results, "expected FTS matches for 'projects'"
    assert calls == ["projects"], "query must be HRR-encoded exactly once"


def test_parent_encodes_per_candidate_baseline(populated_store, monkeypatch):
    # Baseline check that the parent really does re-encode per candidate,
    # so the test above is meaningful.
    retriever = fake_hermes.FactRetriever(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    calls = _spy_encode_text(monkeypatch)
    results = retriever.search("projects", min_trust=0.0, limit=3)
    assert results
    assert len(calls) > 1


def test_single_token_queries_match_parent_exactly(hp, populated_store):
    """Single-token queries: byte-identical to the parent.

    With one token there is no AND-vs-OR difference (the parent's raw
    ``facts_fts MATCH 'projects'`` and the sanitised ``MATCH '"projects"'``
    select the same rows), so the hot-path optimisations must reproduce the
    parent's ids and scores exactly.
    """
    kwargs = dict(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    parent = fake_hermes.FactRetriever(**kwargs)
    plus = hp.retrieval_plus.PlusFactRetriever(**kwargs)

    for query in ["projects", "user", "sqlite", "node"]:
        expected = parent.search(query, min_trust=0.0, limit=5)
        actual = plus.search(query, min_trust=0.0, limit=5)
        assert [f["fact_id"] for f in actual] == [f["fact_id"] for f in expected], query
        for a, e in zip(actual, expected):
            assert a["score"] == pytest.approx(e["score"])
            assert "hrr_vector" not in a


def test_multi_token_recall_is_a_parent_superset(hp, populated_store):
    """Multi-token queries: a strict recall improvement over the parent.

    The parent feeds the raw query to FTS5, which ANDs every token, so a
    fact must contain ALL tokens to be a candidate (and any hyphen/punctuation
    silently errors the whole match out). The sanitiser ORs the significant
    tokens, so PlusFactRetriever finds a SUPERSET of the parent's candidates:
    every fact the parent returned is still present, plus additional genuine
    lexical matches the parent's AND-semantics missed. The shared facts keep
    the parent's relative order.
    """
    kwargs = dict(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    parent = fake_hermes.FactRetriever(**kwargs)
    plus = hp.retrieval_plus.PlusFactRetriever(**kwargs)

    saw_strict_superset = False
    for query in ["sqlite store", "node deploy", "memory facts", "dark dashboards"]:
        expected = parent.search(query, min_trust=0.0, limit=10)
        actual = plus.search(query, min_trust=0.0, limit=10)
        exp_ids = [f["fact_id"] for f in expected]
        act_ids = [f["fact_id"] for f in actual]
        # Superset: every parent hit is still retrieved.
        assert set(exp_ids).issubset(set(act_ids)), f"{query}: {exp_ids} !subset {act_ids}"
        # Shared facts keep the parent's relative order.
        shared = [fid for fid in act_ids if fid in set(exp_ids)]
        assert shared == exp_ids, f"{query}: shared order {shared} != parent {exp_ids}"
        for a in actual:
            assert "hrr_vector" not in a
        if len(act_ids) > len(exp_ids):
            saw_strict_superset = True
    assert saw_strict_superset, (
        "expected at least one query where the sanitised OR recovers facts the "
        "parent's AND-semantics missed"
    )


def test_category_and_trust_filters_match_parent(hp, populated_store):
    kwargs = dict(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    parent = fake_hermes.FactRetriever(**kwargs)
    plus = hp.retrieval_plus.PlusFactRetriever(**kwargs)
    expected = parent.search("projects", category="project", min_trust=0.0, limit=5)
    actual = plus.search("projects", category="project", min_trust=0.0, limit=5)
    assert [f["fact_id"] for f in actual] == [f["fact_id"] for f in expected]
    assert all(f["category"] == "project" for f in actual)


def test_fts_candidates_exclude_hrr_blob(hp, populated_store):
    plus = hp.retrieval_plus.PlusFactRetriever(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    candidates = plus._fts_candidates("projects", None, 0.0, 30)
    assert candidates
    for fact in candidates:
        assert "hrr_vector" not in fact
        assert "fts_rank" in fact


def test_no_blob_loads_when_hrr_disabled(hp, populated_store, monkeypatch):
    plus = hp.retrieval_plus.PlusFactRetriever(
        store=populated_store, hrr_dim=64,
        fts_weight=0.6, jaccard_weight=0.4, hrr_weight=0.0,
    )
    calls = _spy_encode_text(monkeypatch)
    loads = []
    original = hp.retrieval_plus.PlusFactRetriever._load_hrr_vectors
    monkeypatch.setattr(
        hp.retrieval_plus.PlusFactRetriever,
        "_load_hrr_vectors",
        lambda self, ids: loads.append(ids) or original(self, ids),
    )
    results = plus.search("projects", min_trust=0.0, limit=3)
    assert results
    assert calls == []
    assert loads == []


def test_malformed_fts_query_returns_empty(hp, populated_store):
    plus = hp.retrieval_plus.PlusFactRetriever(store=populated_store, hrr_dim=64)
    assert plus.search('"unbalanced AND (', min_trust=0.0, limit=5) == []
