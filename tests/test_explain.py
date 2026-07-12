"""Recall diagnostics: explain_search breakdown and CLI (explain.py).

explain_search() must reuse the real search() scoring path (retriever +
_blend_score + temporal/superseded filtering) rather than reimplementing it,
so a breakdown can never drift from what search() actually returns. These
tests pin that the top-N fact_ids and final scores from explain_search()
match search() exactly on the same query, and that filtered facts still
appear in the breakdown tagged with why they were excluded.
"""

import json
import subprocess
import sys

import numpy as np
import pytest


def test_explain_search_top_n_matches_search_ranking(make_provider):
    provider = make_provider()
    provider._store.add_fact("Alex Rivera prefers Postgres over MySQL", category="tool")
    provider._store.add_fact("The Skylark dashboard runs on port 3100", category="project")
    provider._store.add_fact("Springfield office closes at 6pm", category="general")

    query = "Alex Rivera database preference"
    searched = provider.search(query, min_trust=0.0, limit=10)
    explained = provider.explain_search(query, limit=10)

    assert [r["fact_id"] for r in explained] == [r["fact_id"] for r in searched]
    for e, s in zip(explained, searched):
        assert e["final_score"] == pytest.approx(s["score"])


def test_explain_search_breakdown_has_component_scores(make_provider):
    provider = make_provider()
    provider._store.add_fact("Alex Rivera prefers Postgres over MySQL", category="tool")

    explained = provider.explain_search("Alex Rivera database preference", limit=5)
    assert explained
    row = explained[0]
    for key in (
        "fact_id", "content", "rank", "fts_score", "jaccard_score",
        "hrr_score", "entity_boost", "trust_score", "raw_embedding_cosine",
        "embedding_contribution", "holo_score", "final_score", "excluded",
    ):
        assert key in row
    assert row["rank"] == 1
    assert row["excluded"] is None


def test_explain_search_marks_superseded_fact_with_reason(make_provider):
    provider = make_provider()
    provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark dashboard port is 3100.",
        "category": "project",
    })
    provider._handle_fact_store({
        "action": "add",
        "content": "The Skylark dashboard port is 3200.",
        "category": "project",
    })

    explained = provider.explain_search("Skylark dashboard port", limit=10)
    contents = {r["content"]: r for r in explained}
    old = contents["The Skylark dashboard port is 3100."]
    new = contents["The Skylark dashboard port is 3200."]

    assert old["excluded"] == "temporally_invalid"
    assert new["excluded"] is None


def test_explain_search_respects_limit_on_included_results(make_provider):
    provider = make_provider()
    for i in range(5):
        provider._store.add_fact(f"Fact number {i} about Skylark testing", category="general")

    explained = provider.explain_search("Skylark testing", limit=2)
    included = [r for r in explained if r["excluded"] is None]
    assert len(included) <= 2


def test_dense_candidates_filter_before_truncation(make_provider):
    provider = make_provider(embedding_weight=1.0)
    query = "needleprobe"
    qvec = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    provider._fake_embedder.table[query] = qvec

    invalid_ids = []
    for i, vec in enumerate((
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )):
        fact_id = provider._store.add_fact(
            f"Retired vector candidate number {i}.", category="general"
        )
        invalid_ids.append(fact_id)
        provider._embed_store.upsert(
            fact_id,
            np.asarray(vec, dtype=np.float32),
            embedding_identity=provider._embedding_identity("document"),
        )
        provider._store._conn.execute(
            "UPDATE facts SET invalid_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
            (fact_id,),
        )

    valid_id = provider._store.add_fact(
        "Live vector candidate survives the embedding window.", category="general"
    )
    provider._embed_store.upsert(
        valid_id,
        np.asarray([0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        embedding_identity=provider._embedding_identity("document"),
    )
    provider._store._conn.commit()

    results = provider.search(query, min_trust=0.0, limit=1, bump=False)
    assert [r["fact_id"] for r in results] == [valid_id]

    explained = provider.search(query, min_trust=0.0, limit=1, bump=False, explain=True)
    included = [r for r in explained if r["excluded"] is None]
    excluded = {r["fact_id"]: r["excluded"] for r in explained if r["excluded"]}
    assert [r["fact_id"] for r in included] == [valid_id]
    assert excluded[invalid_ids[0]] == "temporally_invalid"
    assert excluded[invalid_ids[1]] == "temporally_invalid"


def test_cli_smoke_prints_breakdown(tmp_path, hp):
    import fake_hermes

    bridge = subprocess.run(
        [
            sys.executable,
            "-c",
            "from plugins.memory.holographic import HolographicMemoryProvider",
        ],
        capture_output=True,
        text=True,
    )
    if bridge.returncode:
        pytest.skip(
            "enfold.explain standalone smoke test requires an importable Hermes "
            "holographic bridge in the subprocess environment"
        )

    db_path = tmp_path / "facts.db"
    store = fake_hermes.MemoryStore(db_path=db_path, hrr_dim=64)
    store.add_fact("Alex Rivera prefers Postgres over MySQL", category="tool")
    store.close()

    result = subprocess.run(
        [
            sys.executable, "-m", "enfold.explain", str(db_path), "Postgres preference",
            "--ollama-url", "http://127.0.0.1:1", "--hrr-dim", "64",
        ],
        cwd=str(hp.__file__.rsplit("/enfold/", 1)[0]),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["content"] == "Alex Rivera prefers Postgres over MySQL"


def test_cli_refuses_hermes_path(tmp_path):
    from enfold.explain import main

    fake_db = tmp_path / ".hermes" / "facts.db"
    fake_db.parent.mkdir(parents=True)
    fake_db.touch()

    with pytest.raises(SystemExit):
        main([str(fake_db), "some query"])
