"""Offline near-duplicate cluster merge tool (cluster_merge.py).

Covers union-find clustering by embedding cosine, survivor selection (pre-
existing-fact bias for clusters spanning a flood cutoff, else
trust*retrieval/earliest), the merge plan, and the guard rails on execution
(live-path refusal, backup-file requirement, drop-count band, dry-run
default).
"""

import os
import sqlite3

import numpy as np
import pytest

from enfold.cluster_merge import (
    build_clusters,
    choose_survivor,
    plan_merge,
    execute_merge,
    GuardRailError,
)

_SCHEMA = """
CREATE TABLE facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    invalid_at      TIMESTAMP,
    hrr_vector      BLOB
);
CREATE TABLE fact_entities (
    fact_id   INTEGER,
    entity_id INTEGER,
    PRIMARY KEY (fact_id, entity_id)
);
CREATE VIRTUAL TABLE facts_fts USING fts5(content, tags, content=facts, content_rowid=fact_id);
CREATE TABLE fact_embeddings (
    fact_id    INTEGER NOT NULL,
    embedding  BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    embedding_identity TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fact_id, embedding_identity)
);
"""

_ID = "test:model:document:none:v1"


def _vec(*xs):
    return np.array(xs, dtype=np.float32)


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    c.commit()
    return c


def _add_fact(conn, content, created_at, trust=0.5, retrieval=0, helpful=0,
              tags="", category="general", invalid_at=None):
    cur = conn.execute(
        "INSERT INTO facts (content, category, tags, trust_score, retrieval_count, "
        "helpful_count, created_at, updated_at, invalid_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            content,
            category,
            tags,
            trust,
            retrieval,
            helpful,
            created_at,
            created_at,
            invalid_at,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def _embed(conn, fact_id, vec, identity=_ID):
    conn.execute(
        "INSERT INTO fact_embeddings (fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (fact_id, np.asarray(vec, dtype="<f4").tobytes(), len(vec), identity),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# build_clusters
# ---------------------------------------------------------------------------

def test_build_clusters_groups_near_identical_vectors():
    conn = _conn()
    a = _add_fact(conn, "a", "2026-06-01 00:00:00")
    b = _add_fact(conn, "b", "2026-06-01 00:00:01")
    c = _add_fact(conn, "c", "2026-06-01 00:00:02")
    _embed(conn, a, _vec(1.0, 0.0))
    _embed(conn, b, _vec(0.999, 0.001))
    _embed(conn, c, _vec(0.0, 1.0))
    clusters = build_clusters(conn, threshold=0.92, embedding_identity=_ID)
    assert sorted(map(sorted, clusters)) == [sorted([a, b])]


def test_build_clusters_transitive_chain_merges():
    conn = _conn()
    a = _add_fact(conn, "a", "2026-06-01 00:00:00")
    b = _add_fact(conn, "b", "2026-06-01 00:00:01")
    c = _add_fact(conn, "c", "2026-06-01 00:00:02")
    # a~b close, b~c close, a~c not directly close enough alone -> still one cluster via union-find
    _embed(conn, a, _vec(1.0, 0.0, 0.0))
    _embed(conn, b, _vec(0.95, 0.31, 0.0))
    _embed(conn, c, _vec(0.85, 0.5, 0.17))
    clusters = build_clusters(conn, threshold=0.9, embedding_identity=_ID)
    assert len(clusters) == 1
    assert set(clusters[0]) == {a, b, c}


def test_build_clusters_ignores_singletons():
    conn = _conn()
    a = _add_fact(conn, "a", "2026-06-01 00:00:00")
    b = _add_fact(conn, "b", "2026-06-01 00:00:01")
    _embed(conn, a, _vec(1.0, 0.0))
    _embed(conn, b, _vec(0.0, 1.0))
    assert build_clusters(conn, threshold=0.92, embedding_identity=_ID) == []


def test_build_clusters_excludes_structurally_invalid_and_legacy_superseded_facts():
    conn = _conn()
    active = _add_fact(conn, "current routing fact", "2026-06-01 00:00:00")
    invalid = _add_fact(
        conn,
        "old routing fact",
        "2026-05-01 00:00:00",
        invalid_at="2026-06-02 00:00:00",
    )
    legacy = _add_fact(
        conn,
        "SUPERSEDED 2026-06-02: older routing fact",
        "2026-04-01 00:00:00",
    )
    _embed(conn, active, _vec(1.0, 0.0))
    _embed(conn, invalid, _vec(0.999, 0.001))
    _embed(conn, legacy, _vec(0.999, 0.002))

    assert build_clusters(conn, threshold=0.92, embedding_identity=_ID) == []


# ---------------------------------------------------------------------------
# choose_survivor
# ---------------------------------------------------------------------------

def test_choose_survivor_prefers_pre_existing_when_cluster_spans_cutoff():
    conn = _conn()
    pre = _add_fact(conn, "pre-existing statement", "2026-06-20 00:00:00", trust=0.5, retrieval=1)
    flood1 = _add_fact(conn, "flood restatement one", "2026-06-30 05:01:00", trust=0.9, retrieval=9)
    flood2 = _add_fact(conn, "flood restatement two", "2026-06-30 05:01:04", trust=0.9, retrieval=9)
    survivor, losers = choose_survivor(
        conn, [pre, flood1, flood2], flood_cutoff="2026-06-29 18:00:00"
    )
    assert survivor == pre
    assert set(losers) == {flood1, flood2}


def test_choose_survivor_all_flood_picks_highest_trust_then_earliest():
    conn = _conn()
    a = _add_fact(conn, "a", "2026-06-30 05:01:00", trust=0.5, retrieval=2)
    b = _add_fact(conn, "b", "2026-06-30 05:01:04", trust=0.7, retrieval=2)
    c = _add_fact(conn, "c", "2026-06-30 05:01:08", trust=0.7, retrieval=2)
    survivor, losers = choose_survivor(conn, [a, b, c], flood_cutoff="2026-06-29 18:00:00")
    # b and c tie on trust*retrieval (1.4) and both beat a (1.0) -> earliest of the tie wins
    assert survivor == b
    assert set(losers) == {a, c}


def test_choose_survivor_all_pre_existing_picks_by_trust_times_retrieval():
    conn = _conn()
    a = _add_fact(conn, "a", "2026-06-01 00:00:00", trust=0.5, retrieval=10)
    b = _add_fact(conn, "b", "2026-06-02 00:00:00", trust=0.9, retrieval=10)
    survivor, losers = choose_survivor(conn, [a, b], flood_cutoff="2026-06-29 18:00:00")
    assert survivor == b
    assert losers == [a]


def test_plan_merge_does_not_delete_active_replacement_for_invalid_high_score_fact():
    conn = _conn()
    invalid = _add_fact(
        conn,
        "old dashboard port is 3000",
        "2026-06-01 00:00:00",
        trust=0.99,
        retrieval=100,
        invalid_at="2026-06-15 00:00:00",
    )
    active = _add_fact(
        conn,
        "dashboard port is 3100",
        "2026-06-20 00:00:00",
        trust=0.5,
        retrieval=1,
    )
    _embed(conn, invalid, _vec(1.0, 0.0))
    _embed(conn, active, _vec(0.999, 0.001))

    plan = plan_merge(
        conn,
        threshold=0.92,
        flood_cutoff="2026-06-29 18:00:00",
        embedding_identity=_ID,
    )

    assert plan.clusters == []


# ---------------------------------------------------------------------------
# plan_merge
# ---------------------------------------------------------------------------

def test_plan_merge_merges_counts_and_tags():
    conn = _conn()
    pre = _add_fact(conn, "pre-existing statement", "2026-06-20 00:00:00",
                     trust=0.5, retrieval=3, helpful=1, tags="alpha")
    flood = _add_fact(conn, "flood restatement", "2026-06-30 05:01:00",
                       trust=0.9, retrieval=5, helpful=2, tags="beta")
    _embed(conn, pre, _vec(1.0, 0.0))
    _embed(conn, flood, _vec(0.999, 0.001))
    plan = plan_merge(conn, threshold=0.92, flood_cutoff="2026-06-29 18:00:00",
                       embedding_identity=_ID)
    assert len(plan.clusters) == 1
    m = plan.clusters[0]
    assert m.survivor_id == pre
    assert m.loser_ids == [flood]
    assert m.merged_retrieval_count == 8
    assert m.merged_helpful_count == 3
    assert set(m.merged_tags.split(",")) == {"alpha", "beta"}
    assert plan.drop_count == 1
    assert plan.projected_final_count == 1


def test_plan_merge_flags_suspicious_oversized_cluster():
    conn = _conn()
    ids = []
    for i in range(30):
        fid = _add_fact(conn, f"restatement {i}", f"2026-06-30 05:0{i%6}:0{i%9}")
        _embed(conn, fid, _vec(1.0, 0.0001 * i))
        ids.append(fid)
    plan = plan_merge(conn, threshold=0.9, flood_cutoff="2026-06-29 18:00:00",
                       embedding_identity=_ID, suspicious_cluster_size=20)
    assert plan.clusters[0].suspicious is True


# ---------------------------------------------------------------------------
# execute_merge guard rails
# ---------------------------------------------------------------------------

def _seed_two_dup_facts(conn):
    a = _add_fact(conn, "a restatement", "2026-06-30 05:01:00", trust=0.5, retrieval=1)
    b = _add_fact(conn, "a restatement again", "2026-06-30 05:01:04", trust=0.5, retrieval=1)
    _embed(conn, a, _vec(1.0, 0.0))
    _embed(conn, b, _vec(0.999, 0.001))
    return a, b


def test_execute_merge_defaults_to_dry_run(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    a, b = _seed_two_dup_facts(conn)
    conn.close()
    (tmp_path / "store.db.backup").write_bytes(b"x")

    result = execute_merge(str(db_path), threshold=0.92,
                            flood_cutoff="2026-06-29 18:00:00",
                            embedding_identity=_ID,
                            backup_path=str(tmp_path / "store.db.backup"),
                            expected_drop_min=0, expected_drop_max=10)
    assert result.dry_run is True
    assert result.drop_count == 1

    conn2 = sqlite3.connect(str(db_path))
    remaining = conn2.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert remaining == 2  # nothing actually deleted in dry-run


def test_execute_merge_refuses_live_hermes_path(tmp_path):
    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    db_path = fake_home / "memory_store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    with pytest.raises(GuardRailError, match="hermes"):
        execute_merge(str(db_path), threshold=0.92,
                      flood_cutoff="2026-06-29 18:00:00",
                      embedding_identity=_ID,
                      backup_path=str(tmp_path / "nope.backup"),
                      dry_run=False,
                      expected_drop_min=0, expected_drop_max=10)


def test_execute_merge_refuses_live_hermes_path_via_symlink(tmp_path):
    # A symlink whose own literal path does NOT contain ".hermes" but which
    # resolves into a real .hermes directory must be refused just like a
    # direct literal path would be.
    real_hermes_dir = tmp_path / "real_store" / ".hermes"
    real_hermes_dir.mkdir(parents=True)
    real_db = real_hermes_dir / "memory_store.db"
    conn = sqlite3.connect(str(real_db))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()

    sneaky_link = tmp_path / "sneaky_link"
    os.symlink(real_hermes_dir, sneaky_link)
    db_path = sneaky_link / "memory_store.db"
    assert ".hermes" not in str(db_path).split("/")

    with pytest.raises(GuardRailError, match="hermes"):
        execute_merge(str(db_path), threshold=0.92,
                      flood_cutoff="2026-06-29 18:00:00",
                      embedding_identity=_ID,
                      backup_path=str(tmp_path / "nope.backup"),
                      dry_run=False,
                      expected_drop_min=0, expected_drop_max=10)


def test_execute_merge_refuses_without_backup_file(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    _seed_two_dup_facts(conn)
    conn.close()

    with pytest.raises(GuardRailError, match="backup"):
        execute_merge(str(db_path), threshold=0.92,
                      flood_cutoff="2026-06-29 18:00:00",
                      embedding_identity=_ID,
                      backup_path=str(tmp_path / "missing.backup"),
                      dry_run=False,
                      expected_drop_min=0, expected_drop_max=10)


def test_execute_merge_refuses_when_drop_count_outside_band(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    _seed_two_dup_facts(conn)
    conn.close()
    (tmp_path / "store.db.backup").write_bytes(b"x")

    with pytest.raises(GuardRailError, match="drop count"):
        execute_merge(str(db_path), threshold=0.92,
                      flood_cutoff="2026-06-29 18:00:00",
                      embedding_identity=_ID,
                      backup_path=str(tmp_path / "store.db.backup"),
                      dry_run=False,
                      expected_drop_min=5, expected_drop_max=10)


def test_execute_merge_refuses_when_drop_count_exceeds_relative_cap(tmp_path):
    # 10 active facts total: one cluster of 7 near-identical facts (6 losers,
    # 1 survivor) plus 3 standalone facts. 6 drops / 10 active facts = 60%,
    # under the absolute band [0, 10000] but over the default 0.5 relative
    # cap, so this must be refused even though the absolute check would pass.
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()

    for i in range(7):
        fid = _add_fact(conn, f"restatement {i}", f"2026-06-30 05:0{i}:00")
        _embed(conn, fid, _vec(1.0, 0.0001 * i), identity=_ID)
    for i in range(3):
        _add_fact(conn, f"standalone fact {i}", f"2026-06-30 05:1{i}:00")
    conn.close()
    (tmp_path / "store.db.backup").write_bytes(b"x")

    with pytest.raises(GuardRailError, match="relative"):
        execute_merge(str(db_path), threshold=0.9,
                      flood_cutoff="2026-06-29 18:00:00",
                      embedding_identity=_ID,
                      backup_path=str(tmp_path / "store.db.backup"),
                      dry_run=False,
                      expected_drop_min=0, expected_drop_max=10_000)


def test_execute_merge_real_run_deletes_losers_and_checkpoints(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    a, b = _seed_two_dup_facts(conn)
    conn.close()
    (tmp_path / "store.db.backup").write_bytes(b"x")

    result = execute_merge(str(db_path), threshold=0.92,
                            flood_cutoff="2026-06-29 18:00:00",
                            embedding_identity=_ID,
                            backup_path=str(tmp_path / "store.db.backup"),
                            dry_run=False,
                            expected_drop_min=0, expected_drop_max=10)
    assert result.dry_run is False
    assert result.drop_count == 1
    assert result.integrity_ok is True

    conn2 = sqlite3.connect(str(db_path))
    conn2.row_factory = sqlite3.Row
    remaining_ids = {r[0] for r in conn2.execute("SELECT fact_id FROM facts").fetchall()}
    assert remaining_ids == {a}
    embedded_ids = {r[0] for r in conn2.execute("SELECT fact_id FROM fact_embeddings").fetchall()}
    assert embedded_ids == {a}
    survivor = conn2.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id = ?", (a,)
    ).fetchone()[0]
    assert survivor == 2  # 1 + 1 merged from the loser
