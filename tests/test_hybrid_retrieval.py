from __future__ import annotations

from collections.abc import Sequence

from enfold.core_store import connect_database
import pytest

from enfold.hybrid_retrieval import (
    DeterministicFeatureHashEmbedder,
    HybridRetriever,
    RankingConfig,
    SQLiteVersionedEmbeddingBackend,
    SQLiteStoredEmbeddingWriter,
    StoredEmbeddingError,
    VersionedStoredEmbeddingAdapter,
    deterministic_retriever_factory,
)
from enfold.embeddings import embedding_to_bytes
import numpy as np
from enfold.schema import migrate
from enfold.sqlite_vec_index import SQLiteVecIndex, rebuild_sqlite_vec_index


class TableEmbedder:
    identity = "test-table-v1"
    production_ready = False

    def __init__(self, table: dict[str, Sequence[float]]):
        self.table = table
        self.document_calls: list[tuple[str, ...]] = []

    def embed_query(self, text: str) -> Sequence[float]:
        return self.table[text]

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.document_calls.append(tuple(texts))
        return tuple(self.table[text] for text in texts)


def _store(tmp_path):
    conn = connect_database(tmp_path / "hybrid.db")
    migrate(conn)
    return conn


def _fact(conn, fact_id: int, content: str, **fields):
    values = {
        "category": "general",
        "tags": "",
        "trust_score": 0.8,
        "scope": "private",
        "sensitivity": "normal",
        "schema_version": 1,
        **fields,
    }
    columns = ("fact_id", "content", *values)
    conn.execute(
        f"INSERT INTO facts({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
        (fact_id, content, *values.values()),
    )


def test_dense_signal_recovers_semantic_only_candidate(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "The vehicle is parked inside the garage")
    _fact(conn, 2, "A recipe calls for toasted walnuts")
    conn.commit()
    embedder = TableEmbedder({
        "automobile location": (1.0, 0.0),
        "The vehicle is parked inside the garage": (1.0, 0.0),
        "A recipe calls for toasted walnuts": (0.0, 1.0),
    })

    rows = HybridRetriever(conn, embedder).search("automobile location")

    assert rows[0]["fact_id"] == 1
    assert rows[0]["fts_score"] == 0.0
    assert rows[0]["jaccard_score"] == 0.0
    assert rows[0]["dense_score"] == 1.0
    conn.close()


def test_negative_dense_cosine_is_clamped_to_zero(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "A document pointing away from the lookup")
    conn.commit()
    embedder = TableEmbedder({
        "needleterm": (1.0, 0.0),
        "A document pointing away from the lookup": (-1.0, 0.0),
    })

    base_only = RankingConfig(trust_weight=0.0, memory_kind_weight=0.0, recency_weight=0.0)
    rows = HybridRetriever(
        conn, embedder, min_score=0.0, ranking_config=base_only
    ).search("needleterm")

    assert len(rows) == 1
    assert rows[0]["dense_score"] == 0.0
    assert rows[0]["score"] == 0.0
    conn.close()


def test_combined_score_orders_lexical_and_dense_signals_together(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "Unrelated embedding-only candidate")
    _fact(conn, 2, "ranking query lexical candidate")
    conn.commit()
    embedder = TableEmbedder({
        "ranking query": (1.0, 0.0),
        "Unrelated embedding-only candidate": (1.0, 0.0),
        "ranking query lexical candidate": (0.5, 0.8660254),
    })

    base_only = RankingConfig(trust_weight=0.0, memory_kind_weight=0.0, recency_weight=0.0)
    rows = HybridRetriever(conn, embedder, ranking_config=base_only).search("ranking query")

    assert [row["fact_id"] for row in rows] == [2, 1]
    assert rows[0]["score"] == pytest.approx(0.35 + 0.25 * 0.5 + 0.4 * 0.5)
    assert rows[1]["score"] == pytest.approx(0.4)
    conn.close()


def test_scope_current_conflict_and_trust_filters_run_before_dense_embedding(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "eligible current private memory")
    _fact(conn, 2, "forbidden team memory", scope="team")
    _fact(conn, 3, "invalid historical memory", invalid_at="2026-01-01T00:00:00Z")
    _fact(conn, 4, "superseded historical memory", superseded_by=1)
    _fact(conn, 5, "unsettled conflict memory", conflict_group="conflict-1")
    _fact(conn, 6, "low trust memory", trust_score=0.1)
    conn.commit()
    texts = [row[0] for row in conn.execute("SELECT content FROM facts")]
    embedder = TableEmbedder({"memory lookup": (1.0, 0.0), **{text: (1.0, 0.0) for text in texts}})

    rows = HybridRetriever(conn, embedder, allowed_scopes=("private",)).search(
        "memory lookup", min_trust=0.3
    )

    assert [row["fact_id"] for row in rows] == [1]
    assert embedder.document_calls == [("eligible current private memory",)]
    conn.close()


def test_named_anchor_abstains_before_calling_dense_embedder(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "The approved initiative budget is twelve thousand dollars")
    conn.commit()
    embedder = TableEmbedder({})

    rows = HybridRetriever(conn, embedder).search(
        "What budget was approved for Project Ember?"
    )

    assert rows == []
    assert embedder.document_calls == []
    conn.close()


def test_ci_feature_hash_embedder_and_ranking_are_deterministic(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "Orchid backup runs every Tuesday")
    _fact(conn, 2, "Quartz deployment uses a blue environment")
    conn.commit()
    first = HybridRetriever(conn, DeterministicFeatureHashEmbedder()).search(
        "When does Orchid backup run?"
    )
    second = HybridRetriever(conn, DeterministicFeatureHashEmbedder()).search(
        "When does Orchid backup run?"
    )

    assert [(row["fact_id"], row["score"]) for row in first] == [
        (row["fact_id"], row["score"]) for row in second
    ]
    assert first[0]["fact_id"] == 1
    conn.close()


class StoredBackend:
    identity = "local-fastembed"
    embedding_version = "bge-small-en-v1.5@sha256:fixture"
    dimensions = 2

    def __init__(self):
        self.documents = []

    def embed_query(self, text):
        return (1.0, 0.0)

    def load_documents(self, documents):
        self.documents.append(tuple(documents))
        return tuple((1.0, 0.0) if fact_id == 1 else (0.0, 1.0) for fact_id, _ in documents)


def test_versioned_backend_receives_only_eligible_candidate_ids(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "The vehicle is parked inside the garage")
    _fact(conn, 2, "forbidden team memory", scope="team")
    conn.commit()
    backend = StoredBackend()
    adapter = VersionedStoredEmbeddingAdapter(backend)

    rows = HybridRetriever(conn, adapter).search("automobile location")

    assert rows[0]["fact_id"] == 1
    assert backend.documents == [((1, "The vehicle is parked inside the garage"),)]
    assert adapter.identity == "local-fastembed@bge-small-en-v1.5@sha256:fixture:2"
    conn.close()


def test_versioned_backend_rejects_invalid_vectors(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "eligible memory")
    conn.commit()
    backend = StoredBackend()
    backend.dimensions = 3

    with pytest.raises(ValueError, match="dimensions"):
        HybridRetriever(conn, VersionedStoredEmbeddingAdapter(backend)).search("memory")
    conn.close()


def test_deterministic_factory_reports_nonproduction_metadata(tmp_path):
    conn = _store(tmp_path)
    retriever = deterministic_retriever_factory(dimensions=64)(conn, ("private",))

    assert retriever.metadata["embedder_identity"] == "ci-feature-hash-v1:64"
    assert retriever.metadata["embedder_production_ready"] is False
    assert retriever.metadata["filter_before_dense_ranking"] is True
    conn.close()


class FakeQueryEmbedder:
    def __init__(self, vector=(1.0, 0.0)):
        self.vector = vector
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return self.vector


def _embedding_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_embeddings(
            fact_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            embedding_identity TEXT NOT NULL,
            PRIMARY KEY(fact_id, embedding_identity)
        )
        """
    )


def _stored(conn, fact_id, vector, identity="fake:model:document:prefix:v1"):
    array = np.asarray(vector, dtype=np.float32)
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (fact_id, embedding_to_bytes(array), len(array), identity),
    )


def _sqlite_backend(conn, query_embedder):
    return SQLiteVersionedEmbeddingBackend(
        conn,
        query_embedder,
        query_identity="fake:model:query:prefix:v1",
        document_identity="fake:model:document:prefix:v1",
        embedding_version="v1",
        dimensions=2,
        query_prefix="Represent this query: ",
    )


def test_sqlite_backend_loads_only_requested_candidate_ids_in_input_order(tmp_path):
    conn = _store(tmp_path)
    _embedding_table(conn)
    _fact(conn, 1, "stored one")
    _fact(conn, 2, "stored two")
    _stored(conn, 1, (1.0, 0.0))
    _stored(conn, 2, (0.0, 1.0))
    conn.commit()
    query_embedder = FakeQueryEmbedder()
    backend = _sqlite_backend(conn, query_embedder)
    statements = []
    conn.set_trace_callback(statements.append)

    vectors = backend.load_documents(((2, "second"),))
    query = backend.embed_query("where")

    assert [tuple(vector) for vector in vectors] == [(0.0, 1.0)]
    assert query == (1.0, 0.0)
    assert query_embedder.calls == ["Represent this query: where"]
    candidate_selects = [sql for sql in statements if "FROM fact_embeddings" in sql]
    assert len(candidate_selects) == 1
    assert "fact_id IN (2)" in candidate_selects[0]
    assert backend.metadata["missing_embedding_behavior"] == "fail-closed"
    conn.close()


def test_stored_dense_scores_are_protocol_json_scalars(tmp_path):
    conn = _store(tmp_path)
    _embedding_table(conn)
    _fact(conn, 1, "Tuesday preference")
    _stored(conn, 1, (1.0, 0.0))
    conn.commit()
    retriever = HybridRetriever(
        conn,
        VersionedStoredEmbeddingAdapter(
            _sqlite_backend(conn, FakeQueryEmbedder((1.0, 0.0)))
        ),
        allowed_scopes=("private",),
    )

    row = retriever.search("Tuesday", limit=1)[0]

    assert type(row["dense_score"]) is float
    assert type(row["score"]) is float
    conn.close()


def test_sqlite_backend_fails_closed_on_missing_candidate_coverage(tmp_path):
    conn = _store(tmp_path)
    _embedding_table(conn)
    _fact(conn, 1, "stored one")
    _stored(conn, 1, (1.0, 0.0))
    conn.commit()
    backend = _sqlite_backend(conn, FakeQueryEmbedder())

    with pytest.raises(StoredEmbeddingError, match="missing 1 required"):
        backend.load_documents(((1, "present"), (2, "missing")))
    conn.close()


def test_sqlite_backend_validates_identity_dimension_and_query_availability(tmp_path):
    conn = _store(tmp_path)
    _embedding_table(conn)
    _fact(conn, 1, "stored one")
    _stored(conn, 1, (1.0, 0.0))
    conn.commit()

    with pytest.raises(ValueError, match="exactly match"):
        SQLiteVersionedEmbeddingBackend(
            conn,
            FakeQueryEmbedder(),
            query_identity="fake:model:query:prefix:v1",
            document_identity="fake:other:document:prefix:v1",
            embedding_version="v1",
            dimensions=2,
        )
    with pytest.raises(StoredEmbeddingError, match="unexpected dimension"):
        SQLiteVersionedEmbeddingBackend(
            conn,
            FakeQueryEmbedder(),
            query_identity="fake:model:query:prefix:v1",
            document_identity="fake:model:document:prefix:v1",
            embedding_version="v1",
            dimensions=3,
        )
    backend = _sqlite_backend(conn, FakeQueryEmbedder(None))
    with pytest.raises(StoredEmbeddingError, match="query embedding is unavailable"):
        backend.embed_query("test")
    conn.close()


def test_explicit_stored_embedding_writer_is_idempotent_and_not_service_wired(tmp_path):
    conn = _store(tmp_path)
    _embedding_table(conn)
    _fact(conn, 1, "document to embed")
    conn.commit()
    rebuild_sqlite_vec_index(conn, "fake:model:document:none:v1", 2)
    embedder = FakeQueryEmbedder((0.25, 0.75))
    writer = SQLiteStoredEmbeddingWriter(
        conn,
        embedder,
        document_identity="fake:model:document:none:v1",
        embedding_version="v1",
        model_fingerprint="v1",
        prefix_policy="none",
        dimensions=2,
    )

    assert writer.ensure_fact(1) is True
    assert writer.ensure_fact(1) is False
    assert embedder.calls == ["document to embed"]
    row = conn.execute(
        "SELECT dim, embedding_identity FROM fact_embeddings WHERE fact_id = 1"
    ).fetchone()
    assert tuple(row) == (2, "fake:model:document:none:v1")
    index = SQLiteVecIndex.open(conn, "fake:model:document:none:v1", 2)
    assert index is not None and index.count() == 1
    conn.close()
