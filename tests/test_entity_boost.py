"""Entity-graph retrieval: boost facts linked to a query-mentioned entity, and
optionally expand one hop across shared entities, damping high-degree hubs.

The DB already maintains entities + fact_entities tables (populated by the
parent store's entity extraction on every add_fact) that retrieval never
reads. These tests wire that graph into PlusFactRetriever.search(), off by
default so existing installs see byte-identical ranking until it is enabled.
"""

import fake_hermes
import pytest


@pytest.fixture()
def store(tmp_path):
    s = fake_hermes.MemoryStore(db_path=tmp_path / "facts.db", hrr_dim=64)
    yield s
    s.close()


def _retriever(hp, store, **kwargs):
    kwargs.setdefault("fts_weight", 3 / 7)
    kwargs.setdefault("jaccard_weight", 2 / 7)
    kwargs.setdefault("hrr_weight", 2 / 7)
    return hp.retrieval_plus.PlusFactRetriever(store=store, hrr_dim=64, **kwargs)


# ---------------------------------------------------------------------------
# (d) off by default: byte-identical ranking to current behaviour
# ---------------------------------------------------------------------------

def test_entity_boost_weight_defaults_to_zero(hp, store):
    retriever = _retriever(hp, store)
    assert retriever.entity_boost_weight == 0.0
    assert retriever.entity_expansion is False


def test_entity_boost_off_by_default_matches_explicit_zero_weight(hp, store):
    """Constructing PlusFactRetriever without entity kwargs at all (the real
    call site's shape whenever entity_boost_weight/entity_expansion are left
    at their config defaults) must rank identically to passing them
    explicitly at 0.0/False, i.e. the defaults really are inert."""
    store.add_fact("Victor Iglesias prefers pnpm for node projects", category="tool")
    store.add_fact("The deploy target for web projects is vercel", category="tool")
    store.add_fact("Victor Iglesias keeps projects under the home directory", category="project")

    defaulted = _retriever(hp, store)
    explicit_off = _retriever(hp, store, entity_boost_weight=0.0, entity_expansion=False)

    expected = explicit_off.search("Victor Iglesias projects", min_trust=0.0, limit=10)
    actual = defaulted.search("Victor Iglesias projects", min_trust=0.0, limit=10)

    assert [f["fact_id"] for f in actual] == [f["fact_id"] for f in expected]
    for a, e in zip(actual, expected):
        assert a["score"] == pytest.approx(e["score"])


def test_entity_boost_off_by_default_matches_parent_lexical_recall(hp, store):
    """With the entity feature untouched, PlusFactRetriever's own documented
    lexical-recall difference from the base parent (OR vs AND FTS semantics,
    see retrieval_plus.py) is unchanged: no entity-only rows sneak into the
    result set beyond what that superset recall already produces."""
    store.add_fact("Victor Iglesias prefers pnpm for node projects", category="tool")
    store.add_fact("The deploy target for web projects is vercel", category="tool")
    third_id = store.add_fact(
        "Victor Iglesias keeps projects under the home directory", category="project"
    )

    plus = _retriever(hp, store)
    actual = plus.search("Victor Iglesias projects", min_trust=0.0, limit=10)

    # third_id matches "projects" lexically (OR semantics), so it is present,
    # but strictly ranked below the two-token match and carries no
    # expansion marker (the feature never ran).
    assert third_id in [f["fact_id"] for f in actual]
    assert all("expanded_from_entity" not in f for f in actual)


# ---------------------------------------------------------------------------
# (a) query mentioning an entity boosts facts linked to that entity
# ---------------------------------------------------------------------------

def test_query_entity_mention_boosts_linked_facts(hp, store):
    # Two facts equally relevant lexically to "projects", but only one is
    # linked to the "Victor Iglesias" entity mentioned in the query.
    linked_id = store.add_fact(
        "Victor Iglesias runs several side projects", category="project"
    )
    unlinked_id = store.add_fact(
        "The team runs several side projects", category="project"
    )

    plus = _retriever(hp, store, entity_boost_weight=0.5)
    results = plus.search("Victor Iglesias side projects", min_trust=0.0, limit=10)

    by_id = {f["fact_id"]: f for f in results}
    assert by_id[linked_id]["score"] > by_id[unlinked_id]["score"]


def test_entity_boost_weight_zero_gives_no_boost_even_with_entity_mention(hp, store):
    linked_id = store.add_fact(
        "Victor Iglesias runs several side projects", category="project"
    )

    plus_off = _retriever(hp, store, entity_boost_weight=0.0)
    plus_on = _retriever(hp, store, entity_boost_weight=0.5)

    score_off = next(
        f["score"] for f in plus_off.search("Victor Iglesias side projects", min_trust=0.0, limit=10)
        if f["fact_id"] == linked_id
    )
    score_on = next(
        f["score"] for f in plus_on.search("Victor Iglesias side projects", min_trust=0.0, limit=10)
        if f["fact_id"] == linked_id
    )
    assert score_on > score_off


# ---------------------------------------------------------------------------
# (b) 1-hop expansion surfaces a related fact via shared entity, ranked below
#     direct hits and clearly marked
# ---------------------------------------------------------------------------

def test_one_hop_expansion_surfaces_related_fact_below_direct_hits(hp, store):
    direct_id = store.add_fact(
        "Victor Iglesias prefers pnpm for node projects", category="tool"
    )
    # Shares the "Victor Iglesias" entity but has no lexical overlap with the
    # query at all, so plain FTS/Jaccard/HRR would never surface it.
    related_id = store.add_fact(
        "Victor Iglesias lives in Miami", category="general"
    )
    # No entity relationship at all: must not be pulled in by expansion.
    unrelated_id = store.add_fact(
        "The gateway restarts are scheduled overnight", category="general"
    )

    plus = _retriever(hp, store, entity_boost_weight=0.3, entity_expansion=True)
    results = plus.search("pnpm preference for node", min_trust=0.0, limit=10)

    ids = [f["fact_id"] for f in results]
    assert direct_id in ids
    assert related_id in ids
    assert unrelated_id not in ids

    direct_rank = ids.index(direct_id)
    related_rank = ids.index(related_id)
    assert direct_rank < related_rank, "expanded fact must rank below the direct hit"

    related_fact = next(f for f in results if f["fact_id"] == related_id)
    assert related_fact.get("expanded_from_entity"), (
        "expansion-only results must be clearly marked"
    )
    direct_fact = next(f for f in results if f["fact_id"] == direct_id)
    assert not direct_fact.get("expanded_from_entity")


def test_expansion_is_off_by_default_even_with_boost_weight_set(hp, store):
    store.add_fact("Victor Iglesias prefers pnpm for node projects", category="tool")
    related_id = store.add_fact("Victor Iglesias lives in Miami", category="general")

    plus = _retriever(hp, store, entity_boost_weight=0.3)  # entity_expansion not set
    results = plus.search("pnpm preference for node", min_trust=0.0, limit=10)

    assert related_id not in [f["fact_id"] for f in results]


# ---------------------------------------------------------------------------
# (c) hub entities (high degree) are damped/excluded from expansion
# ---------------------------------------------------------------------------

def test_hub_entity_is_excluded_from_expansion(hp, store):
    direct_id = store.add_fact(
        "Victor Iglesias prefers pnpm for node projects", category="tool"
    )
    # "Victor Iglesias" becomes a hub: linked to > entity_hub_degree_limit facts.
    hub_limit = 5
    for i in range(hub_limit + 5):
        store.add_fact(f"Victor Iglesias noted item number {i} today", category="general")

    plus = _retriever(
        hp, store, entity_boost_weight=0.3, entity_expansion=True,
        entity_hub_degree_limit=hub_limit,
    )
    results = plus.search("pnpm preference for node", min_trust=0.0, limit=20)

    ids = [f["fact_id"] for f in results]
    # None of the "noted item" filler facts should be pulled in purely via the
    # now-hub "Victor Iglesias" entity.
    assert not any("noted item number" in f["content"] for f in results if f["fact_id"] != direct_id)
    assert direct_id in ids


def test_non_hub_entity_still_expands_when_hub_limit_set_low(hp, store):
    direct_id = store.add_fact(
        "Victor Iglesias prefers pnpm for node projects", category="tool"
    )
    related_id = store.add_fact("Victor Iglesias lives in Miami", category="general")

    plus = _retriever(
        hp, store, entity_boost_weight=0.3, entity_expansion=True,
        entity_hub_degree_limit=25,
    )
    results = plus.search("Victor Iglesias pnpm preference", min_trust=0.0, limit=10)

    ids = [f["fact_id"] for f in results]
    assert direct_id in ids
    assert related_id in ids


def test_entity_hub_degree_limit_defaults_to_25(hp, store):
    retriever = _retriever(hp, store)
    assert retriever.entity_hub_degree_limit == 25


# ---------------------------------------------------------------------------
# Provider config wiring
# ---------------------------------------------------------------------------

def test_provider_defaults_leave_entity_feature_off(make_provider):
    provider = make_provider()
    r = provider._retriever
    assert r.entity_boost_weight == 0.0
    assert r.entity_expansion is False
    assert r.entity_hub_degree_limit == 25


def test_provider_config_wires_entity_settings_to_retriever(make_provider):
    provider = make_provider(
        entity_boost_weight=0.4,
        entity_expansion=True,
        entity_hub_degree_limit=10,
    )
    r = provider._retriever
    assert r.entity_boost_weight == pytest.approx(0.4)
    assert r.entity_expansion is True
    assert r.entity_hub_degree_limit == 10
