import sqlite3

import numpy as np

from holographic_plus.embed_store import EmbedStore

_DOC = "test:model:document:none:v1"
_QUERY = "test:model:query:none:v1"


def _store():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return EmbedStore(conn, embedding_identity=_DOC)


def _count(es):
    return int(es._conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0])


def test_upsert_roundtrip_and_idempotence():
    es = _store()
    v = np.array([0.6, 0.8], dtype=np.float32)
    es.upsert(1, v)
    results = es.score_all(v, embedding_identity=_QUERY)
    assert results and results[0][0] == 1
    assert results[0][1] > 0.99  # the stored vector matches itself
    assert _count(es) == 1
    # upsert is idempotent on (fact_id, identity)
    es.upsert(1, v)
    assert _count(es) == 1


def test_score_all_ranks_most_similar_first():
    es = _store()
    es.upsert(1, np.array([1.0, 0.0], dtype=np.float32))
    es.upsert(2, np.array([0.0, 1.0], dtype=np.float32))
    results = es.score_all(np.array([1.0, 0.0], dtype=np.float32), embedding_identity=_QUERY)
    assert results[0][0] == 1  # fact 1 wins
    assert results[0][1] > results[1][1]


def test_ids_without_embeddings():
    es = _store()
    es.upsert(1, np.array([1.0, 0.0], dtype=np.float32))
    missing = es.ids_without_embeddings([1, 2, 3], embedding_identity=_DOC)
    assert set(missing) == {2, 3}


def test_delete_removes_and_invalidates_cache():
    es = _store()
    es.upsert(1, np.array([1.0, 0.0], dtype=np.float32))
    # prime the cache
    es.score_all(np.array([1.0, 0.0], dtype=np.float32), embedding_identity=_QUERY)
    es.delete(1)
    assert _count(es) == 0
    assert es.score_all(np.array([1.0, 0.0], dtype=np.float32), embedding_identity=_QUERY) == []


def test_score_all_empty_store():
    es = _store()
    assert es.score_all(np.array([1.0, 0.0], dtype=np.float32), embedding_identity=_QUERY) == []
