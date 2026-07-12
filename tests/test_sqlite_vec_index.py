from __future__ import annotations

import logging

import numpy as np
import pytest

from enfold.core_store import connect_database
from enfold.context import pack_context
from enfold.embeddings import embedding_to_bytes
from enfold.embed_store import EmbedStore
from enfold.hybrid_retrieval import (
    HybridRetriever,
    RankingConfig,
    SQLiteVersionedEmbeddingBackend,
    VersionedStoredEmbeddingAdapter,
)
from enfold.schema import migrate
from enfold.sqlite_vec_index import (
    IDENTITY_KEY,
    SQLiteVecIndex,
    load_sqlite_vec,
    rebuild_sqlite_vec_index,
)


IDENTITY = "fake:model:document:none:v1"


class QueryEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, text):
        return self.vectors[text]


def _database(tmp_path):
    conn = connect_database(tmp_path / "vectors.db")
    migrate(conn)
    return conn


def _fact(conn, fact_id, content, vector):
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, trust_score, schema_version) "
        "VALUES (?, ?, 'private', 0.8, 1)",
        (fact_id, content),
    )
    array = np.asarray(vector, dtype=np.float32)
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (fact_id, embedding_to_bytes(array), len(array), IDENTITY),
    )


def _retriever(conn, query_embedder, vector_backend):
    backend = SQLiteVersionedEmbeddingBackend(
        conn,
        query_embedder,
        query_identity="fake:model:query:none:v1",
        document_identity=IDENTITY,
        embedding_version="v1",
        dimensions=3,
    )
    return HybridRetriever(
        conn,
        VersionedStoredEmbeddingAdapter(backend),
        vector_backend=vector_backend,
        min_score=0.0,
        ranking_config=RankingConfig(
            trust_weight=0.0,
            memory_kind_weight=0.0,
            recency_weight=0.0,
            ambiguity_margin=0.0,
        ),
    )


def test_sqlite_vec_dense_scores_match_brute_real_retrieval_path(tmp_path):
    conn = _database(tmp_path)
    _fact(conn, 1, "alpha one", (1.0, 0.0, 0.0))
    _fact(conn, 2, "beta two", (0.5, 0.5, 0.0))
    _fact(conn, 3, "gamma three", (-1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    query_embedder = QueryEmbedder({"unmatched": (1.0, 0.0, 0.0)})

    brute = _retriever(conn, query_embedder, "brute").search("unmatched")
    indexed = _retriever(conn, query_embedder, "sqlite-vec").search("unmatched")

    assert [row["fact_id"] for row in indexed] == [row["fact_id"] for row in brute]
    assert [row["dense_score"] for row in indexed] == pytest.approx(
        [row["dense_score"] for row in brute], abs=1e-6
    )
    assert _retriever(conn, query_embedder, "auto").metadata["vector_backend"] == "sqlite-vec"


def test_sqlite_vec_mmr_matches_brute_when_token_and_embedding_diversity_disagree(
    tmp_path,
):
    conn = _database(tmp_path)
    _fact(conn, 1, "shared alpha", (1.0, 0.0, 0.0))
    _fact(conn, 2, "unique beta", (1.0, 0.0, 0.0))
    _fact(conn, 3, "shared alpha gamma", (0.0, 1.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    query_embedder = QueryEmbedder({"what is relevant": (1.0, 0.0, 0.0)})

    def selected(vector_backend):
        rows = _retriever(conn, query_embedder, vector_backend).search(
            "what is relevant", limit=3
        )
        return [
            fact["fact_id"]
            for fact in pack_context(
                rows, token_budget=512, max_facts=2, mmr_lambda=0.2
            ).facts
        ]

    assert selected("brute") == [1, 3]
    assert selected("sqlite-vec") == selected("brute")


def test_auto_falls_back_honestly_when_extension_is_missing(
    tmp_path, monkeypatch, caplog
):
    conn = _database(tmp_path)
    _fact(conn, 1, "only memory", (1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    query_embedder = QueryEmbedder({"unmatched": (1.0, 0.0, 0.0)})
    expected = _retriever(conn, query_embedder, "brute").search("unmatched")

    def unavailable(_conn):
        raise RuntimeError("extension missing")

    monkeypatch.setattr("enfold.sqlite_vec_index.load_sqlite_vec", unavailable)
    with caplog.at_level(logging.WARNING, logger="enfold.sqlite_vec_index"):
        actual = _retriever(conn, query_embedder, "auto").search("unmatched")

    assert actual == expected
    assert "falling back to brute" in caplog.text


@pytest.mark.parametrize("corruption", ["identity", "population"])
def test_auto_falls_back_on_invalid_index_with_identical_results(
    tmp_path, caplog, corruption
):
    conn = _database(tmp_path)
    _fact(conn, 1, "only memory", (1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None
    if corruption == "identity":
        conn.execute(
            "UPDATE enfold_meta SET value='wrong' WHERE key=?", (IDENTITY_KEY,)
        )
    else:
        conn.execute(f'DELETE FROM "{index.table_name}" WHERE rowid=1')
    conn.commit()
    query_embedder = QueryEmbedder({"unmatched": (1.0, 0.0, 0.0)})
    expected = _retriever(conn, query_embedder, "brute").search("unmatched")

    with caplog.at_level(logging.WARNING, logger="enfold.sqlite_vec_index"):
        actual = _retriever(conn, query_embedder, "auto").search("unmatched")

    assert actual == expected
    assert "falling back to brute" in caplog.text


def test_query_time_index_problem_falls_back_with_identical_results(
    tmp_path, caplog
):
    conn = _database(tmp_path)
    _fact(conn, 1, "zero vector is valid canonical data", (0.0, 0.0, 0.0))
    _fact(conn, 2, "ordinary vector", (1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    query_embedder = QueryEmbedder({"unmatched": (1.0, 0.0, 0.0)})
    expected = _retriever(conn, query_embedder, "brute").search("unmatched")

    with caplog.at_level(logging.WARNING, logger="enfold.hybrid_retrieval"):
        retriever = _retriever(conn, query_embedder, "sqlite-vec")
        actual = retriever.search("unmatched")

    assert actual == expected
    assert retriever.metadata["vector_backend"] == "brute"
    assert "falling back to brute" in caplog.text


def test_extension_loading_is_disabled_immediately_after_load(tmp_path):
    conn = _database(tmp_path)

    load_sqlite_vec(conn)

    with pytest.raises(Exception):
        conn.load_extension("definitely-not-an-extension")


def test_transactional_upsert_and_delete_hooks(tmp_path):
    conn = _database(tmp_path)
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None

    vector = embedding_to_bytes(np.asarray((1.0, 0.0, 0.0), dtype=np.float32))
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, trust_score, schema_version) "
        "VALUES (1, 'atomic', 'private', 0.8, 1)"
    )
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (1, ?, 3, ?)",
        (vector, IDENTITY),
    )
    index.upsert_in_transaction(1, vector)
    conn.rollback()
    assert index.count() == 0

    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, trust_score, schema_version) "
        "VALUES (1, 'atomic', 'private', 0.8, 1)"
    )
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (1, ?, 3, ?)",
        (vector, IDENTITY),
    )
    index.upsert_in_transaction(1, vector)
    conn.commit()
    assert index.count() == 1

    conn.execute("BEGIN IMMEDIATE")
    conn.execute("DELETE FROM fact_embeddings WHERE fact_id = 1")
    index.delete_in_transaction(1)
    conn.commit()
    assert index.count() == 0


def test_embed_store_write_and_delete_paths_keep_vec0_in_sync(tmp_path):
    conn = _database(tmp_path)
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, trust_score, schema_version) "
        "VALUES (1, 'legacy path', 'private', 0.8, 1)"
    )
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    store = EmbedStore(conn, embedding_identity=IDENTITY)

    store.upsert(1, np.asarray((1.0, 0.0, 0.0), dtype=np.float32))
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None and index.count() == 1

    store.delete(1)
    assert index.count() == 0


def test_rebuild_is_idempotent_and_replaces_corrupt_population(tmp_path):
    conn = _database(tmp_path)
    _fact(conn, 1, "one", (1.0, 0.0, 0.0))
    _fact(conn, 2, "two", (0.0, 1.0, 0.0))
    conn.commit()

    first = rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    second = rebuild_sqlite_vec_index(conn, IDENTITY, 3)

    assert first.indexed_count == second.indexed_count == 2
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None and index.count() == 2
