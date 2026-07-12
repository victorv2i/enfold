"""Extraction-path dedup gate: the primary flood root cause.

The interactive fact_store "add" action runs every new fact through
_find_near_duplicate before insert. The extraction path (insert_facts, called
from the queue drain) bypassed that gate entirely and called store.add_fact()
directly, so LLM extraction could flood the store with near-duplicate facts
that the interactive path would have caught. These tests pin:

  - extraction inserts are near-duplicate-checked (same gate, not a copy)
  - a paraphrase with high cosine but Jaccard just under threshold is caught
  - a genuine value update is NEVER dropped
  - _existing_summary includes topic-similar facts, not just top-trust/recent
  - the queue drain loop caps facts processed per tick
"""
import json
import types

import numpy as np

from enfold.llm_extract import insert_facts


def _llm_response(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


def _extraction_cfg():
    return {"extraction_provider": "testprov", "extraction_model": "testmodel"}


# ---------------------------------------------------------------------------
# (a) extraction insert path is near-duplicate-checked
# ---------------------------------------------------------------------------

def test_insert_facts_skips_near_duplicate_of_existing_fact(make_provider):
    provider = make_provider()
    provider._store.add_fact(
        "The user prefers pnpm as their package manager for node projects.",
        category="tool",
    )

    result = insert_facts(
        provider._store,
        [{
            "content": "The user prefers pnpm as their package manager for node projects.",
            "category": "tool",
            "tags": "pnpm",
        }],
        dedup_check=provider._find_near_duplicate,
    )

    assert result.inserted == 0
    assert result.skipped == 1
    assert result.failed == 0
    facts = provider._store.list_facts(min_trust=0.0, limit=50)
    assert len(facts) == 1, "the verbatim restatement must not be stored again"


def test_queue_drain_dedupes_extracted_facts_against_existing_store(
    make_provider, aux_module, waiter
):
    """End-to-end: the queue worker must not flood duplicates past the gate."""
    provider = make_provider(**_extraction_cfg())
    provider._store.add_fact(
        "The user deploys web apps to vercel.", category="tool"
    )

    aux_module.call_llm = lambda **kwargs: _llm_response(json.dumps([
        {"content": "The user deploys web apps to vercel.",
         "category": "tool", "tags": "deploy,vercel"},
    ]))

    provider.on_session_end([
        {"role": "user", "content": "Also the deploy target for my web apps is vercel."},
        {"role": "assistant", "content": "Got it, vercel is the deploy target."},
    ])

    assert waiter(lambda: provider._extract_queue.pending_count() == 0)
    facts = provider._store.list_facts(min_trust=0.0, limit=50)
    assert len(facts) == 1, "extraction must not duplicate a fact already in the store"


def test_queue_drain_retries_when_fact_insert_fails(make_provider, aux_module):
    provider = make_provider(**_extraction_cfg())
    # This test drives one drain synchronously.  Quiesce the daemon worker
    # first so it cannot reclaim the retry row between our assertion reads.
    provider._queue_stop.set()
    provider._queue_wake.set()
    provider._queue_worker.join(timeout=1)
    assert not provider._queue_worker.is_alive()
    provider._queue_max_attempts = 3
    provider._extract_drain_batch = 1
    aux_module.call_llm = lambda **kwargs: _llm_response(json.dumps([
        {"content": "The storage failure should keep the queue item retryable.",
         "category": "general", "tags": "storage"},
    ]))

    def locked_add_fact(*args, **kwargs):
        raise RuntimeError("database is locked")

    provider._store.add_fact = locked_add_fact
    row_id = provider._extract_queue.enqueue("USER: a transcript that extracts one fact")

    import threading
    stop = threading.Event()
    original_mark_failed = provider._extract_queue.mark_failed

    def mark_failed_once(*args, **kwargs):
        attempts = original_mark_failed(*args, **kwargs)
        stop.set()
        return attempts

    provider._extract_queue.mark_failed = mark_failed_once
    provider._drain_extract_queue(stop, provider._extract_queue)

    row = provider._store._conn.execute(
        "SELECT status, attempts, last_error FROM extract_queue WHERE id = ?",
        (row_id,),
    ).fetchone()
    assert row["status"] == "pending"
    assert row["attempts"] == 1
    assert "extracted fact insert" in row["last_error"]


# ---------------------------------------------------------------------------
# (b) paraphrase duplicate: high cosine, lower Jaccard
# ---------------------------------------------------------------------------

def _add_with_embedding(provider, content, category, vec):
    """Insert *content* and give it a stored embedding, mirroring the real
    write path's end state without depending on the async embed pool."""
    fid = provider._store.add_fact(content, category=category)
    provider._fake_embedder.table[content] = np.asarray(vec, dtype=np.float32)
    provider._embed_store.upsert(
        fid, np.asarray(vec, dtype=np.float32),
        embedding_identity=provider._embedding_identity("document"),
    )
    return fid


def test_semantic_gate_catches_paraphrase_with_low_jaccard(make_provider):
    """A reworded restatement shares almost no words but is the same fact.

    Jaccard alone (word-overlap) misses this; cosine similarity from the
    embedding already computed in the write path catches it.
    """
    a = "The user prefers Postgres over MySQL for new projects."
    b = "For new work the user always reaches for Postgres instead of MySQL."

    # Same vector for both -> cosine 1.0, but the wording is different enough
    # that Jaccard is low.
    shared_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    provider = make_provider(dedup_cosine=0.92)
    _add_with_embedding(provider, a, "user_pref", shared_vec)
    provider._fake_embedder.table[b] = np.asarray(shared_vec, dtype=np.float32)

    from enfold import _jaccard, _norm_tokens
    assert _jaccard(_norm_tokens(a), _norm_tokens(b)) < 0.5, "test fixture must be a low-Jaccard paraphrase"

    dup = provider._find_near_duplicate(b, category="user_pref")
    assert dup is not None, "high-cosine paraphrase must be caught by the semantic gate"


def test_low_cosine_and_low_jaccard_is_not_a_duplicate(make_provider):
    a = "The user prefers Postgres over MySQL for new projects."
    b = "Skylark launched publicly at skylark.example.com this month."

    provider = make_provider(dedup_cosine=0.92)
    _add_with_embedding(provider, a, "user_pref", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    provider._fake_embedder.table[b] = np.asarray(
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )

    dup = provider._find_near_duplicate(b, category="user_pref")
    assert dup is None


# ---------------------------------------------------------------------------
# (c) a genuine value update is never dropped, even under the semantic gate
# ---------------------------------------------------------------------------

def test_semantic_gate_never_drops_a_changed_value(make_provider):
    a = "The Skylark dashboard port is 3100."
    b = "The Skylark dashboard port is 3200."

    # Force near-identical embeddings (as a real embedder would give two
    # sentences differing by one digit) to prove the value-token guard, not
    # a low cosine, is what keeps the update.
    shared_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    provider = make_provider(dedup_cosine=0.92)
    _add_with_embedding(provider, a, "project", shared_vec)
    provider._fake_embedder.table[b] = np.asarray(shared_vec, dtype=np.float32)

    dup = provider._find_near_duplicate(b, category="project")
    assert dup is None, "a changed concrete value must always be kept as an update"


def test_semantic_gate_falls_back_to_jaccard_when_embedder_unavailable(make_provider):
    """If the embedder is down, the gate must still behave (Jaccard-only)."""
    provider = make_provider(dedup_cosine=0.92)
    provider._embedder_available = False

    provider._store.add_fact(
        "The user prefers pnpm as their package manager for node projects.",
        category="tool",
    )
    dup = provider._find_near_duplicate(
        "The user prefers pnpm as their package manager for node projects.",
        category="tool",
    )
    assert dup is not None, "exact restatement must still be caught via Jaccard alone"


# ---------------------------------------------------------------------------
# _existing_summary widened with topic-similar facts
# ---------------------------------------------------------------------------

def test_existing_summary_includes_topic_similar_facts_not_just_top_trust_recent(hp):
    """_existing_summary must accept a search_fn and include its hits even when
    they are neither top-trust nor most-recent (the top-40 trust/recent window
    misses paraphrase targets on a large store)."""
    similar_hit = {"content": "The user just adopted the uv package manager for a new project"}

    def fake_search_fn(topic, limit):
        return [similar_hit]

    class _StubStore:
        def __init__(self):
            self._conn = self

        def execute(self, *a, **k):
            return self

        def fetchall(self):
            return []

    summary = hp.llm_extract._existing_summary(
        _StubStore(), topic="node package manager", search_fn=fake_search_fn
    )
    assert "uv package manager" in summary


# ---------------------------------------------------------------------------
# (d) drain loop respects a per-tick cap
# ---------------------------------------------------------------------------

def test_drain_extract_queue_caps_items_processed_per_tick(make_provider, aux_module):
    provider = make_provider(**_extraction_cfg())
    provider._extract_drain_batch = 2
    provider._queue_stop.set()
    if provider._queue_worker:
        provider._queue_worker.join(timeout=2.0)

    aux_module.call_llm = lambda **kwargs: _llm_response("[]")

    import threading
    for i in range(5):
        provider._extract_queue.enqueue(f"USER: transcript number {i}")

    stop = threading.Event()
    provider._drain_extract_queue(stop, provider._extract_queue)

    assert provider._extract_queue.pending_count() == 3, (
        "only extract_drain_batch items may be drained per tick"
    )
