"""Stale-embedding garbage collection: EmbedStore.identity_counts / prune_identities
and the provider-level vacuum_embeddings + rebuild_embeddings(prune_stale=)."""

import sqlite3

import numpy as np
import pytest

from holographic_plus.embed_store import EmbedStore

_CUR = "fastembed:bge-large:document:none:v1"
_OLD = "fastembed:bge-base:document:none:v1"
_QRY = "fastembed:bge-large:query:none:v1"


def _store(identity=_CUR):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return EmbedStore(conn, embedding_identity=identity)


def _v(*xs):
    return np.array(xs, dtype=np.float32)


# ---------------------------------------------------------------------------
# identity_counts
# ---------------------------------------------------------------------------

def test_identity_counts_groups_by_identity():
    es = _store()
    es.upsert(1, _v(1.0, 0.0), embedding_identity=_CUR)
    es.upsert(2, _v(0.0, 1.0), embedding_identity=_CUR)
    es.upsert(1, _v(1.0, 0.0), embedding_identity=_OLD)
    assert es.identity_counts() == {_CUR: 2, _OLD: 1}


# ---------------------------------------------------------------------------
# prune_identities
# ---------------------------------------------------------------------------

def test_prune_identities_removes_unkept_and_returns_count():
    es = _store()
    es.upsert(1, _v(1.0, 0.0), embedding_identity=_CUR)
    es.upsert(2, _v(0.0, 1.0), embedding_identity=_CUR)
    es.upsert(1, _v(1.0, 0.0), embedding_identity=_OLD)
    es.upsert(2, _v(0.0, 1.0), embedding_identity=_OLD)
    deleted = es.prune_identities({_CUR})
    assert deleted == 2
    assert es.identity_counts() == {_CUR: 2}


def test_prune_identities_keeps_all_listed():
    es = _store()
    es.upsert(1, _v(1.0, 0.0), embedding_identity=_CUR)
    es.upsert(1, _v(1.0, 0.0), embedding_identity=_OLD)
    es.upsert(1, _v(1.0, 0.0), embedding_identity="third:id:document:none:v1")
    deleted = es.prune_identities({_CUR, _OLD})
    assert deleted == 1
    assert set(es.identity_counts()) == {_CUR, _OLD}


def test_prune_identities_empty_keep_raises_and_deletes_nothing():
    es = _store()
    es.upsert(1, _v(1.0, 0.0), embedding_identity=_CUR)
    with pytest.raises(ValueError):
        es.prune_identities(set())
    assert es.identity_counts() == {_CUR: 1}


def test_prune_identities_invalidates_cache():
    es = _store()
    es.upsert(1, _v(1.0, 0.0), embedding_identity=_CUR)
    es.upsert(2, _v(0.0, 1.0), embedding_identity=_OLD)
    # Prime the cache for the current identity (the _QRY identity maps to _CUR).
    es.score_all(_v(1.0, 0.0), embedding_identity=_QRY)
    # Prune everything except _OLD, which removes the cached _CUR vector.
    es.prune_identities({_OLD})
    # The cache must reflect the deletion: the current identity now has nothing.
    assert es.score_all(_v(1.0, 0.0), embedding_identity=_QRY) == []


# ---------------------------------------------------------------------------
# provider: vacuum_embeddings + rebuild_embeddings(prune_stale=)
# ---------------------------------------------------------------------------

_STALE = "fastembed:superseded-model:document:none:v1"


def _seed_facts(p, contents):
    for c in contents:
        p._store.add_fact(c, category="general")


def _u8(i=0):
    """A fixed dim-8 unit vector (matches the test FakeEmbedder's dimension)."""
    v = np.zeros(8, dtype=np.float32)
    v[i % 8] = 1.0
    return v


def test_vacuum_embeddings_prunes_superseded_identity(make_provider):
    p = make_provider()
    _seed_facts(p, ["alpha fact", "beta fact", "gamma fact"])
    p.rebuild_embeddings()  # current-identity coverage for all three
    current = p._embedding_identity("document")
    for fid in (1, 2, 3):  # inject vectors from a superseded model
        p._embed_store.upsert(fid, _u8(0), embedding_identity=_STALE)
    assert p._embed_store.identity_counts().get(_STALE) == 3

    stats = p.vacuum_embeddings()

    assert stats["pruned"] == 3
    after = p._embed_store.identity_counts()
    assert after.get(current) == 3
    assert _STALE not in after


def test_vacuum_embeddings_keeps_extra_canary(make_provider):
    p = make_provider()
    _seed_facts(p, ["alpha fact", "beta fact"])
    p.rebuild_embeddings()
    canary = "fastembed:canary-model:document:none:v1"
    for fid in (1, 2):
        p._embed_store.upsert(fid, _u8(1), embedding_identity=canary)
        p._embed_store.upsert(fid, _u8(0), embedding_identity=_STALE)

    stats = p.vacuum_embeddings(extra_keep=[canary])

    counts = p._embed_store.identity_counts()
    assert _STALE not in counts          # superseded model pruned
    assert counts.get(canary) == 2       # canary preserved
    assert stats["pruned"] == 2          # only the two stale rows


def test_rebuild_embeddings_prunes_stale_by_default(make_provider):
    p = make_provider()
    _seed_facts(p, ["alpha fact", "beta fact"])
    for fid in (1, 2):  # an old model's vectors already present
        p._embed_store.upsert(fid, _u8(0), embedding_identity=_STALE)

    p.rebuild_embeddings()  # re-embeds current, prunes stale by default

    counts = p._embed_store.identity_counts()
    assert _STALE not in counts
    assert counts.get(p._embedding_identity("document")) == 2


def test_rebuild_embeddings_prune_stale_false_keeps(make_provider):
    p = make_provider()
    _seed_facts(p, ["alpha fact", "beta fact"])
    for fid in (1, 2):
        p._embed_store.upsert(fid, _u8(0), embedding_identity=_STALE)

    p.rebuild_embeddings(prune_stale=False)

    counts = p._embed_store.identity_counts()
    assert counts.get(_STALE) == 2
    assert counts.get(p._embedding_identity("document")) == 2
