from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from enfold.core_store import connect_database
import pytest

from enfold.hybrid_retrieval import HybridRetriever, RankingConfig
from enfold.schema import migrate


class TableEmbedder:
    identity = "ranking-quality-fixture"
    production_ready = False

    def __init__(self, table: dict[str, Sequence[float]]):
        self._table = table

    def embed_query(self, text: str) -> Sequence[float]:
        return self._table[text]

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._table[text] for text in texts)


def _store(tmp_path):
    conn = connect_database(tmp_path / "ranking-quality.db")
    migrate(conn)
    return conn


def _fact(conn, fact_id: int, content: str, **fields):
    values = {
        "category": "general",
        "tags": "",
        "trust_score": 0.8,
        "created_at": "2026-07-12 12:00:00",
        "updated_at": "2026-07-12 12:00:00",
        "memory_kind": None,
        "scope": "private",
        "sensitivity": "normal",
        "schema_version": 1,
        **fields,
    }
    columns = ("fact_id", "content", *values)
    conn.execute(
        f"INSERT INTO facts({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        (fact_id, content, *values.values()),
    )


def test_fresh_high_trust_state_paraphrase_outranks_old_low_trust_match(tmp_path):
    conn = _store(tmp_path)
    old = "Archived replicas live in the west storage vault"
    fresh = "Current backup snapshots are retained in the western repository"
    _fact(
        conn,
        1,
        old,
        trust_score=0.31,
        created_at="2020-01-01 00:00:00",
        updated_at="2020-01-01 00:00:00",
        memory_kind="event",
    )
    _fact(conn, 2, fresh, trust_score=0.95, memory_kind="state")
    conn.commit()
    query = "Where are nightly copies kept?"
    embedder = TableEmbedder({
        query: (1.0, 0.0),
        old: (0.98, 0.2),
        fresh: (0.95, 0.1),
    })

    rows = HybridRetriever(
        conn,
        embedder,
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0)

    assert [row["fact_id"] for row in rows[:2]] == [2, 1]
    assert rows[0]["trust_score_component"] > rows[1]["trust_score_component"]
    assert rows[0]["recency_score"] > rows[1]["recency_score"]
    assert rows[0]["memory_kind_score"] > rows[1]["memory_kind_score"]
    conn.close()


def test_state_kind_breaks_an_otherwise_identical_relevance_tie(tmp_path):
    conn = _store(tmp_path)
    event = "The deployment target is the cedar cluster"
    state = "The active deployment target remains the cedar cluster"
    _fact(conn, 1, event, memory_kind="event")
    _fact(conn, 2, state, memory_kind="state")
    conn.commit()
    query = "active target location"
    embedder = TableEmbedder({query: (1.0, 0.0), event: (1.0, 0.0), state: (1.0, 0.0)})

    rows = HybridRetriever(
        conn,
        embedder,
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0)

    assert [row["fact_id"] for row in rows[:2]] == [2, 1]
    conn.close()


def test_tiny_top_margin_abstains_instead_of_exposing_tie_breaker(tmp_path):
    conn = _store(tmp_path)
    first = "Archived replicas use the west vault"
    second = "Stored copies use the western vault"
    _fact(conn, 1, first)
    _fact(conn, 2, second)
    conn.commit()
    query = "nightly retention location"
    embedder = TableEmbedder({query: (1.0, 0.0), first: (1.0, 0.0), second: (1.0, 0.0)})

    rows = HybridRetriever(
        conn,
        embedder,
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0, limit=10)

    assert rows == []
    conn.close()


def test_top_candidate_below_score_floor_abstains(tmp_path):
    conn = _store(tmp_path)
    content = "Unrelated archived observation"
    _fact(
        conn,
        1,
        content,
        trust_score=0,
        created_at="2020-01-01 00:00:00",
        updated_at="2020-01-01 00:00:00",
        memory_kind="event",
    )
    conn.commit()
    query = "nightly retention location"
    embedder = TableEmbedder({query: (1.0, 0.0), content: (0.0, 1.0)})

    rows = HybridRetriever(
        conn,
        embedder,
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0)

    assert rows == []
    conn.close()


def test_default_formula_components_sum_to_reported_score(tmp_path):
    conn = _store(tmp_path)
    content = "Atlas backups run Tuesday"
    _fact(conn, 1, content, trust_score=0.8, memory_kind="insight")
    conn.commit()
    query = "Atlas backups"
    row = HybridRetriever(
        conn,
        TableEmbedder({query: (1.0, 0.0), content: (0.6, 0.8)}),
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0)[0]

    expected = 0.90 * (
        0.35 * row["fts_score"]
        + 0.25 * row["jaccard_score"]
        + 0.40 * row["dense_score"]
    ) + 0.05 * 0.8 + 0.02 * 0.75 + 0.03 * 1.0
    assert row["score"] == pytest.approx(expected)
    conn.close()


@pytest.mark.parametrize("field", ["trust_weight", "score_floor", "recency_half_life_days"])
def test_ranking_config_rejects_non_finite_values(field):
    with pytest.raises(ValueError):
        RankingConfig(**{field: float("nan")})
