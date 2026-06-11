"""holographic_plus: Holographic memory with dense embedding retrieval.

Extends HolographicMemoryProvider with a 4th retrieval signal: dense cosine
similarity. The embedder is pluggable, FastEmbed (local CPU, the default) or
Ollama, and durable facts are extracted at session end by a configurable LLM
(the host agent's own model by default).

Config (same ``plugins.hermes-memory-store`` block, extra keys):

    plugins:
      hermes-memory-store:
        # ... all existing holographic keys ...
        embedding_weight: 0.3            # weight for embedding similarity (default 0.3)
        embedding_backend: fastembed     # "fastembed" (local CPU, default) or "ollama"
        fastembed_model: BAAI/bge-base-en-v1.5
        embed_on_add: true               # embed immediately when a fact is added (default true)
        # extraction_provider / extraction_model: default to the host agent's model

Retrieval weights:
    The holographic signals (defaults FTS=0.3, Jaccard=0.2, HRR=0.2) are
    rescaled by their own sum (0.7) so they total exactly 1.0 inside the
    retriever. The blend then gives that holographic score a
    (1 - embedding_weight) share of the final budget and the dense cosine
    similarity the remaining embedding_weight share (see _blend_score), so
    all four signals genuinely partition 1.0 for any embedding_weight.

If the embedding backend is unreachable the plugin falls back silently to
holographic-only scoring (embedding weight redistributed to the other three).

First-run behaviour:
    On initialize(), any fact that lacks an embedding is queued for batch
    embedding in a background thread so startup is non-blocking, and any
    extraction transcripts left in the persistent queue by a previous run
    are drained by the background worker.

Usage: change config.yaml::

    memory:
      provider: holographic_plus
"""

from __future__ import annotations

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from plugins.memory.holographic import HolographicMemoryProvider
from .embeddings import FastEmbedder, OllamaEmbedder
from .embed_store import EmbedStore
from .extract_queue import ExtractQueue
from .llm_extract import _format_conversation, extract_facts_from_transcript, insert_facts
from .retrieval_plus import PlusFactRetriever

logger = logging.getLogger(__name__)

# Default weights
_FTS_W     = 0.3
_JACCARD_W = 0.2
_HRR_W     = 0.2
_EMBED_W   = 0.3


def _blend_score(holo_score: float, raw_emb_sim: Optional[float], trust: float, ew: float) -> float:
    """Combine the holographic score and the dense-embedding similarity on one scale.

    ``holo_score`` is the parent's relevance × trust (range ``[0, trust]``), with the
    parent's FTS/Jaccard/HRR weights rescaled to sum to 1.0. The holographic signals
    share a ``(1 - ew)`` slice of the budget and the embedding gets ``ew``. The cosine
    similarity is mapped ``[-1, 1] → [0, 1]`` and likewise trust-weighted, so both
    terms live in ``[0, trust]`` and the weights genuinely partition the budget. A
    fact with no embedding simply cannot earn the ``ew`` slice.
    """
    base = (1.0 - ew) * holo_score
    if raw_emb_sim is None:
        return base
    emb_norm = (raw_emb_sim + 1.0) / 2.0  # cosine [-1,1] → [0,1]
    return base + ew * emb_norm * trust


class HolographicPlusProvider(HolographicMemoryProvider):
    """Holographic memory + dense embedding retrieval (FastEmbed or Ollama)."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        cfg = self._config
        self._embed_weight: float = float(cfg.get("embedding_weight", _EMBED_W))
        self._embedding_backend: str = str(cfg.get("embedding_backend", "fastembed")).lower()
        self._embedding_prefix_policy: str = str(cfg.get("embedding_prefix_policy", "none"))
        self._ollama_url: str     = str(cfg.get("ollama_url", "http://localhost:11434"))
        self._ollama_model: str   = str(cfg.get("ollama_model", "qwen3-embedding:8b"))
        self._fastembed_model: str = str(cfg.get("fastembed_model", "BAAI/bge-base-en-v1.5"))
        self._fastembed_cache_dir: Optional[str] = cfg.get("fastembed_cache_dir")
        self._embed_on_add: bool  = bool(cfg.get("embed_on_add", True))

        # Fact-extraction LLM: explicit override, else the host agent's own model,
        # else disabled. Never hardcodes a provider: works with whatever the user runs.
        _host_model = _host_model_config()
        self._extract_provider: Optional[str] = cfg.get("extraction_provider") or _host_model.get("provider")
        self._extract_model: Optional[str]    = cfg.get("extraction_model") or _host_model.get("default")
        self._extract_effort: Optional[str]   = cfg.get("extraction_effort")

        self._embedder: Optional[Any] = None
        self._embed_store: Optional[EmbedStore]  = None
        self._embedder_available: bool             = False
        self._backfill_thread: Optional[threading.Thread] = None
        self._backfill_stop: Optional[threading.Event] = None
        self._embed_pool: Optional[ThreadPoolExecutor] = None

        # Persistent extraction queue + its single worker thread
        self._extract_queue: Optional[ExtractQueue] = None
        self._queue_worker: Optional[threading.Thread] = None
        self._queue_stop: Optional[threading.Event] = None
        self._queue_wake: Optional[threading.Event] = None
        # Worker tunables (instance attributes so tests can tighten them)
        self._queue_max_attempts: int = 5
        # In-memory attempt counts per queue row id: a fallback bound so a
        # row whose failure cannot even be recorded in the DB (mark_failed
        # itself failing) can never re-run the LLM indefinitely.
        self._queue_mem_attempts: Dict[int, int] = {}
        self._queue_backoff_base: float = 2.0     # seconds
        self._queue_backoff_cap: float = 300.0    # seconds
        self._queue_poll_interval: float = 60.0   # seconds between idle wakeups
        self._backfill_interval: float = 900.0    # seconds between embed backfill ticks

    # ------------------------------------------------------------------
    # Embedding backend helpers
    # ------------------------------------------------------------------

    def _embedding_model_name(self) -> str:
        if self._embedding_backend == "fastembed":
            return self._fastembed_model
        return self._ollama_model

    def _embedding_identity(self, role: str = "document") -> str:
        role = role if role in {"query", "document"} else "document"
        model = self._embedding_model_name()
        prefix_policy = getattr(self, "_embedding_prefix_policy", "none")
        return f"{self._embedding_backend}:{model}:{role}:{prefix_policy}:v1"

    def _create_embedder(self):
        if self._embedding_backend == "fastembed":
            return FastEmbedder(
                model=self._fastembed_model,
                cache_dir=self._fastembed_cache_dir,
            )
        return OllamaEmbedder(
            base_url=self._ollama_url,
            model=self._ollama_model,
        )

    # ------------------------------------------------------------------
    # MemoryProvider identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "holographic_plus"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize parent store + embedding layer.

        Re-initialization is safe: any worker thread, embed pool, and store
        connection from a previous initialize() are shut down first so nothing
        leaks across gateway session re-inits.
        """
        prev_store = self._store
        prev_quiescent = self._teardown_background()
        super().initialize(session_id, **kwargs)
        if prev_store is not None and prev_store is not self._store:
            if prev_quiescent:
                try:
                    prev_store.close()
                except Exception as exc:
                    logger.debug("holographic_plus: closing previous store failed: %s", exc)
            else:
                # A previous worker, backfill thread, or pool task may still be
                # mid-write on that connection. Leaking it beats closing it
                # under a writer (and the parent always leaked it anyway).
                logger.warning(
                    "holographic_plus: previous store connection left open, "
                    "background work from the prior session may still be using it"
                )

        # ---- Re-build FactRetriever so FTS/Jaccard/HRR genuinely sum to 1.0:
        #      each default weight is divided by their combined sum (0.7), so the
        #      rescale is independent of embedding_weight. The embedding signal is
        #      folded in at merge time (see search()/_blend_score), where the
        #      (1-ew)/ew split keeps every signal on a consistent scale.
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        holo_sum  = _FTS_W + _JACCARD_W + _HRR_W
        fts_w     = _FTS_W / holo_sum
        jaccard_w = _JACCARD_W / holo_sum
        hrr_w     = _HRR_W / holo_sum

        self._retriever = PlusFactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            fts_weight=fts_w,
            jaccard_weight=jaccard_w,
            hrr_weight=hrr_w,
            hrr_dim=hrr_dim,
        )

        # ---- Embedding layer
        self._embedder = self._create_embedder()
        self._embed_store = EmbedStore(
            conn=self._store._conn,
            embedding_identity=self._embedding_identity("document"),
            lock=getattr(self._store, "_lock", None),
        )
        self._embed_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hp-embed")
        self._embedder_available = self._embedder.is_available()

        if self._embedder_available:
            logger.info(
                "holographic_plus: embedding backend available (%s, model=%s)",
                self._embedding_backend, self._embedding_model_name(),
            )
            # Kick off background backfill for facts without embeddings
            self._backfill_stop = threading.Event()
            self._backfill_thread = threading.Thread(
                target=self._backfill_embeddings,
                args=(self._backfill_stop,),
                daemon=True,
                name="holographic_plus_backfill",
            )
            self._backfill_thread.start()
        else:
            logger.warning(
                "holographic_plus: embedding backend %s not available, "
                "falling back to holographic-only retrieval",
                self._embedding_backend,
            )

        # Reclaim any WAL growth left by the previous run before new work starts.
        self._wal_checkpoint()

        # ---- Persistent extraction queue + worker. Started after the store so
        #      transcripts queued by a previous run (crash, restart) are drained
        #      as soon as the provider comes up.
        self._extract_queue = ExtractQueue(
            conn=self._store._conn,
            lock=getattr(self._store, "_lock", None),
        )
        try:
            pending = self._extract_queue.pending_count()
            if pending:
                logger.info(
                    "holographic_plus: draining %d queued extraction(s) from a previous run",
                    pending,
                )
        except Exception:
            pass
        self._queue_mem_attempts = {}
        self._queue_stop = threading.Event()
        self._queue_wake = threading.Event()
        self._queue_worker = threading.Thread(
            target=self._queue_worker_loop,
            args=(self._queue_stop, self._queue_wake, self._extract_queue),
            daemon=True,
            name="holographic_plus_extract_queue",
        )
        self._queue_worker.start()
        self._queue_wake.set()

    def shutdown(self) -> None:
        self._teardown_background()
        super().shutdown()
        self._embedder    = None
        self._embed_store = None

    def _teardown_background(self) -> bool:
        """Stop the worker, backfill thread, and embed pool from a previous initialize().

        The worker and backfill thread are asked to stop and joined briefly
        (0.5s each: this runs on the synchronous agent-init path, so a busy
        worker must not stall session start). If one is mid LLM call or mid
        chunk it keeps running as a daemon and exits at its next stop check;
        a row the worker was processing stays pending and is drained by the
        next worker (add_fact deduplicates by content, so a rare overlap is
        harmless).

        Returns True when background work is quiescent: the worker join
        succeeded, the backfill thread is not alive, and the pool was shut
        down. Only then is the previous store connection safe to close; the
        caller leaks it otherwise.
        """
        quiescent = True
        if self._queue_stop is not None:
            self._queue_stop.set()
        if self._backfill_stop is not None:
            self._backfill_stop.set()
        if self._queue_wake is not None:
            self._queue_wake.set()
        worker = self._queue_worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=0.5)
            if worker.is_alive():
                quiescent = False
                logger.warning(
                    "holographic_plus: extraction worker still busy after 0.5s, "
                    "it will exit at its next stop check"
                )
        backfill = self._backfill_thread
        if backfill is not None and backfill.is_alive():
            backfill.join(timeout=0.5)
            if backfill.is_alive():
                quiescent = False
                logger.warning(
                    "holographic_plus: backfill thread still busy after 0.5s, "
                    "it will exit at its next chunk"
                )
        self._queue_worker = None
        self._queue_stop = None
        self._queue_wake = None
        self._backfill_thread = None
        self._backfill_stop = None
        self._extract_queue = None
        if self._embed_pool is not None:
            # cancel_futures drops queued (not yet started) embeds so they
            # never write to the old connection; backfill re-embeds them.
            self._embed_pool.shutdown(wait=False, cancel_futures=True)
            self._embed_pool = None
        return quiescent

    # ------------------------------------------------------------------
    # Session end: LLM-based fact extraction (configurable model)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pre-compression hook: save facts BEFORE context window is trimmed
    # ------------------------------------------------------------------

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Queue fact extraction for messages about to be compressed.

        Called by MemoryManager before Hermes compresses the context window.
        The formatted transcript is enqueued on the persistent extraction
        queue and this returns immediately, so compression is never blocked.

        Returns an empty string: nothing is injected into the compression
        prompt itself; the facts land in the store from the background worker.
        """
        if not self._store or not messages:
            return ""

        # Only act when there is meaningful content about to be discarded
        # (at least 4 turns of real dialogue).
        real_messages = [
            m for m in messages
            if m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
            and len(m["content"].strip()) > 20
        ]
        if len(real_messages) < 4:
            return ""

        self._enqueue_extraction(messages, source="pre_compress")
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """At session end: run regex auto-extract (parent), then queue LLM extraction.

        Parent handles cheap regex patterns (I prefer X, we decided Y).
        The LLM pass is enqueued on the persistent extraction queue, so session
        teardown is never blocked and a crash before extraction completes does
        not lose the transcript: it is drained on the next initialize().
        """
        # Run parent regex extraction first (fast, cheap, synchronous)
        super().on_session_end(messages)

        if not self._store or not messages:
            return

        self._enqueue_extraction(messages, source="session_end")

    def _enqueue_extraction(self, messages: List[Dict[str, Any]], source: str) -> bool:
        """Format and persist a transcript for the extraction worker.

        Returns True when a row was enqueued. Skips quietly when no extraction
        model is configured (queued rows could never be processed).
        """
        if self._extract_queue is None or not messages:
            return False
        if not self._extract_provider or not self._extract_model:
            return False
        try:
            transcript = _format_conversation(messages)
            if not transcript.strip():
                return False
            self._extract_queue.enqueue(transcript)
            if self._queue_wake is not None:
                self._queue_wake.set()
            logger.debug("holographic_plus: queued extraction from %s", source)
            return True
        except Exception as exc:
            logger.warning(
                "holographic_plus: failed to enqueue extraction (%s): %s", source, exc
            )
            return False

    # ------------------------------------------------------------------
    # Extraction queue worker
    # ------------------------------------------------------------------

    def _queue_worker_loop(
        self,
        stop: threading.Event,
        wake: threading.Event,
        queue: ExtractQueue,
    ) -> None:
        """Single daemon worker: drain the extraction queue, tick embed backfill.

        Receives its events and queue as arguments so a re-initialized provider
        (new events, new queue) never races a worker from a previous run.
        """
        last_backfill = time.monotonic()
        while not stop.is_set():
            wake.wait(timeout=self._queue_poll_interval)
            if stop.is_set():
                break
            wake.clear()
            try:
                self._drain_extract_queue(stop, queue)
            except Exception as exc:
                logger.warning("holographic_plus: extraction queue drain failed: %s", exc)
            # Periodic backfill: re-embeds facts whose per-fact embed attempt
            # failed (transient backend errors are otherwise silently dropped).
            if time.monotonic() - last_backfill >= self._backfill_interval:
                last_backfill = time.monotonic()
                if self._embedder_available:
                    try:
                        self._backfill_embeddings(stop)
                    except Exception as exc:
                        logger.debug("holographic_plus: backfill tick failed: %s", exc)

    def _drain_extract_queue(self, stop: threading.Event, queue: ExtractQueue) -> None:
        """Process pending rows until the queue is empty or stop is set."""
        if not self._store:
            return
        if not self._extract_provider or not self._extract_model:
            return

        processed = 0
        while not stop.is_set():
            # Rows that hit the in-memory attempt cap are dropped from this
            # process's consideration until restart, even when the DB-side
            # mark_failed bookkeeping is broken and cannot record failures.
            blocked = {
                rid for rid, n in self._queue_mem_attempts.items()
                if n >= self._queue_max_attempts
            }
            row = queue.next_pending(
                max_attempts=self._queue_max_attempts, exclude_ids=blocked
            )
            if row is None:
                break
            try:
                facts = extract_facts_from_transcript(
                    row["payload"],
                    self._store,
                    provider=self._extract_provider,
                    model=self._extract_model,
                    effort=self._extract_effort,
                )
                inserted = 0
                if facts:
                    with self._deferred_bank_rebuild():
                        inserted = insert_facts(
                            self._store, facts, embed_callback=self._embed_cb
                        )
                queue.mark_done(row["id"])
                self._queue_mem_attempts.pop(row["id"], None)
                processed += 1
                if inserted:
                    logger.info(
                        "holographic_plus: extraction queue item %d stored %d facts",
                        row["id"], inserted,
                    )
            except Exception as exc:
                mem_attempts = self._queue_mem_attempts.get(row["id"], 0) + 1
                self._queue_mem_attempts[row["id"]] = mem_attempts
                if mem_attempts >= self._queue_max_attempts:
                    logger.warning(
                        "holographic_plus: queue item %d dropped in-process after "
                        "%d attempts",
                        row["id"], mem_attempts,
                    )
                try:
                    attempts = queue.mark_failed(
                        row["id"], str(exc), max_attempts=self._queue_max_attempts
                    )
                except Exception as mark_exc:
                    logger.debug(
                        "holographic_plus: could not record queue failure: %s", mark_exc
                    )
                    break
                logger.warning(
                    "holographic_plus: extraction attempt %d for queue item %d failed: %s",
                    attempts, row["id"], exc,
                )
                if attempts >= self._queue_max_attempts:
                    logger.warning(
                        "holographic_plus: queue item %d marked dead after %d attempts",
                        row["id"], attempts,
                    )
                    continue  # dead rows are never retried, skip the backoff wait
                # Exponential backoff with jitter before the next attempt
                delay = min(
                    self._queue_backoff_base * (2 ** attempts), self._queue_backoff_cap
                ) + random.uniform(0, self._queue_backoff_base)
                if stop.wait(delay):
                    break

        if processed:
            self._wal_checkpoint()

    def _wal_checkpoint(self) -> Optional[tuple]:
        """Run PRAGMA wal_checkpoint(TRUNCATE) on the fact store database.

        Keeps the -wal sidecar from growing without bound under the extraction
        and embedding write load. Returns the (busy, wal_pages, checkpointed)
        pragma row, or None when the store is missing or the pragma fails.
        """
        if not self._store:
            return None
        try:
            with self._store._lock:
                row = self._store._conn.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
            if row is not None:
                logger.info(
                    "holographic_plus: wal_checkpoint(TRUNCATE) busy=%s wal_pages=%s checkpointed=%s",
                    row[0], row[1], row[2],
                )
                return tuple(row)
            return None
        except Exception as exc:
            logger.debug("holographic_plus: wal_checkpoint failed: %s", exc)
            return None

    def _embed_cb(self, fact_id: int, content: str) -> None:
        """Embed callback for newly inserted facts (used by the queue worker)."""
        if self._embedder_available and self._embedder and self._embed_store:
            self._submit_embed(self._embed_and_store, fact_id, content)

    @contextmanager
    def _deferred_bank_rebuild(self):
        """Batch seam: suppress the parent store's per-add category bank rebuild.

        MemoryStore.add_fact() ends with self._rebuild_bank(category), a FULL
        rebuild of the category's memory bank (store.py reads every hrr_vector
        in the category), so inserting an extraction batch of N facts rebuilds
        banks N times. The parent exposes no API to defer this, so the seam is:
        shadow _rebuild_bank with a category collector on the store INSTANCE
        (the class method is untouched), insert the batch, then pop the shadow
        and run ONE real rebuild per affected category.

        The store lock is held only for the batch adds and the shadow pop, so
        adds from other threads serialize behind the batch and can never
        observe the shadowed method. The real rebuilds run AFTER the lock is
        released (each parent _rebuild_bank re-acquires it), so turn-thread
        prefetch and tool adds are not blocked for the whole adds+rebuilds
        stretch. Benign worst case: a concurrent same-category add lands
        between the pop and our rebuild and triggers its own immediate
        rebuild, making ours redundant. Only the extraction worker uses this;
        single facts added via tools keep the parent's immediate rebuild
        behavior.
        """
        store = self._store
        if store is None:
            yield
            return
        pending: set = set()
        try:
            with store._lock:
                store._rebuild_bank = pending.add  # instance attr shadows the class method
                try:
                    yield
                finally:
                    store.__dict__.pop("_rebuild_bank", None)
        finally:
            # Outside the lock: rebuilds for categories the batch touched.
            for category in sorted(pending):
                try:
                    store._rebuild_bank(category)
                except Exception as exc:
                    logger.warning(
                        "holographic_plus: deferred bank rebuild for %r failed: %s",
                        category, exc,
                    )

    # ------------------------------------------------------------------
    # Tool handler: intercept 'add' to embed new facts
    # ------------------------------------------------------------------

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Intercept fact_store mutations to keep dense embeddings in sync."""
        result_json = super().handle_tool_call(tool_name, args, **kwargs)

        if tool_name == "fact_store":
            self._sync_embedding_after_fact_store_result(args, result_json)

        return result_json

    def _sync_embedding_after_fact_store_result(self, args: Dict[str, Any], result_json: str) -> None:
        """Maintain the embedding sidecar after fact_store add/update/remove."""
        import json as _json

        action = args.get("action")
        try:
            result = _json.loads(result_json)
        except Exception as exc:
            logger.debug("holographic_plus: failed to parse fact_store result: %s", exc)
            return

        try:
            if action == "add" and self._embed_on_add:
                fact_id = result.get("fact_id")
                if fact_id and self._embedder_available and self._embed_store:
                    content = args.get("content", "")
                    self._submit_embed(self._embed_and_store, fact_id, content)

            elif action == "update" and result.get("updated") and args.get("content"):
                fact_id = int(args["fact_id"])
                content = args.get("content", "")
                if self._embedder_available and self._embedder and self._embed_store:
                    self._submit_embed(self._embed_and_store, fact_id, content)
                elif self._embed_store:
                    # Content changed but embeddings are unavailable; remove the stale vector.
                    self._embed_store.delete(fact_id)

            elif action == "remove" and result.get("removed") and self._embed_store:
                self._embed_store.delete(int(args["fact_id"]))

        except Exception as exc:
            logger.debug("holographic_plus: embedding sidecar sync failed: %s", exc)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes + embed them."""
        super().on_memory_write(action, target, content)

        if action == "add" and self._embed_on_add and content and self._embedder_available:
            # Find the fact_id that was just inserted (content is UNIQUE in the
            # parent schema, so this resolves to exactly one row).
            try:
                with self._store._lock:
                    row = self._store._conn.execute(
                        "SELECT fact_id FROM facts WHERE content = ?", (content.strip(),)
                    ).fetchone()
                if row:
                    self._submit_embed(self._embed_and_store, int(row["fact_id"]), content)
            except Exception as exc:
                logger.debug("holographic_plus: on_memory_write embed failed: %s", exc)

    # ------------------------------------------------------------------
    # Prefetch: merge holographic + embedding scores
    # ------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""

        try:
            results = self.search(query, min_trust=self._min_trust, limit=5)
            if not results:
                return ""
            lines = []
            for r in results:
                trust = r.get("trust_score", r.get("trust", 0))
                embed_score = r.get("embedding_score")
                embed_note = f", emb={embed_score:.3f}" if embed_score is not None else ""
                lines.append(f"- [{trust:.1f}{embed_note}] {r.get('content', '')}")
            return "## Holographic+ Memory\n" + "\n".join(lines)
        except Exception as exc:
            logger.debug("holographic_plus prefetch failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Public search API (used by prefetch + tool handler)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: holographic pipeline + embedding similarity.

        1. Run parent holographic search (FTS + Jaccard + HRR).
        2. If Ollama is available: embed query, score all stored embeddings,
           merge scores.
        3. Re-rank and return top *limit* results.
        """
        # --- Step 1: holographic candidates (get more for re-ranking headroom)
        holo_results = self._retriever.search(
            query,
            category=category,
            min_trust=min_trust,
            limit=limit * 3,
        )

        if not self._embedder_available or not self._embed_store or not self._embedder:
            # Pure holographic fallback
            for r in holo_results[:limit]:
                r["embedding_score"] = None
            self._bump_retrieval_counts(holo_results[:limit])
            return holo_results[:limit]

        # --- Step 2: embed query + score all facts
        try:
            query_vec = self._embedder.embed(query)
        except Exception as exc:
            logger.debug("holographic_plus: query embed failed: %s", exc)
            query_vec = None

        if query_vec is None:
            for r in holo_results[:limit]:
                r["embedding_score"] = None
            self._bump_retrieval_counts(holo_results[:limit])
            return holo_results[:limit]

        # Get all embedding scores as a dict
        try:
            emb_pairs = self._embed_store.score_all(
                query_vec,
                embedding_identity=self._embedding_identity("query"),
            )
        except Exception as exc:
            logger.debug("holographic_plus: score_all failed: %s", exc)
            emb_pairs = []

        emb_scores: Dict[int, float] = {fid: sim for fid, sim in emb_pairs}

        # --- Step 3: merge
        ew = self._embed_weight
        # holo_score already covers (1 - ew) budget via parent FactRetriever
        # (trust-weighted). We scale the holographic score back by (1-ew) so
        # total budget is preserved, then add embedding contribution.

        merged: List[Dict[str, Any]] = []

        # Build a set of candidate fact_ids (holographic + top-K embedding)
        holo_ids = {r["fact_id"] for r in holo_results}
        # Include top embedding candidates not caught by holographic FTS
        top_emb_ids = {fid for fid, _ in emb_pairs[:limit * 2]}
        extra_ids = top_emb_ids - holo_ids

        # Fetch extra facts by ID if needed (same trust and category filters
        # as the holographic candidates)
        extra_facts: List[Dict[str, Any]] = []
        if extra_ids and self._store:
            placeholders = ",".join("?" * len(extra_ids))
            params: List[Any] = list(extra_ids) + [min_trust]
            category_clause = ""
            if category:
                category_clause = " AND category = ?"
                params.append(category)
            with self._store._lock:
                rows = self._store._conn.execute(
                    f"""
                    SELECT fact_id, content, category, tags, trust_score,
                           retrieval_count, helpful_count, created_at, updated_at
                    FROM facts
                    WHERE fact_id IN ({placeholders})
                      AND trust_score >= ?{category_clause}
                    """,
                    params,
                ).fetchall()
            for row in rows:
                d = dict(row)
                d["score"] = 0.0  # no holographic score
                extra_facts.append(d)

        all_candidates = list(holo_results) + extra_facts

        for fact in all_candidates:
            fid = fact["fact_id"]
            holo_score = fact.get("score", 0.0)  # parent relevance × trust, in [0, trust]
            trust = float(fact.get("trust_score", fact.get("trust", 0.0)) or 0.0)
            raw_emb_sim = emb_scores.get(fid)

            fact["score"] = _blend_score(holo_score, raw_emb_sim, trust, ew)
            fact["embedding_score"] = round(raw_emb_sim, 4) if raw_emb_sim is not None else None

            merged.append(fact)

        merged.sort(key=lambda x: x["score"], reverse=True)
        # Deduplicate by fact_id (extra_ids might overlap holo)
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for f in merged:
            if f["fact_id"] not in seen:
                seen.add(f["fact_id"])
                unique.append(f)

        self._bump_retrieval_counts(unique[:limit])
        return unique[:limit]

    def _bump_retrieval_counts(self, results: List[Dict[str, Any]]) -> None:
        """Increment facts.retrieval_count for the returned facts (parent-store idiom).

        The parent MemoryStore.search_facts() bumps retrieval_count itself, but our
        search() goes through FactRetriever which doesn't, so we do it here for the
        final ranked results only. Never fails the search.
        """
        if not results or not self._store:
            return
        try:
            ids = [(int(r["fact_id"]),) for r in results]
            with self._store._lock:
                self._store._conn.executemany(
                    "UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id = ?",
                    ids,
                )
                self._store._conn.commit()
        except Exception as exc:
            logger.debug("holographic_plus: retrieval_count bump failed: %s", exc)

    # ------------------------------------------------------------------
    # Tool handler override: expose search with embeddings
    # ------------------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        import json as _json
        action = args.get("action")
        if action == "search":
            try:
                results = self.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return _json.dumps({"results": results, "count": len(results)})
            except KeyError as exc:
                return _json.dumps({"error": f"Missing required argument: {exc}"})
            except Exception as exc:
                return _json.dumps({"error": str(exc)})

        # For all other actions delegate to parent
        return super()._handle_fact_store(args)

    # ------------------------------------------------------------------
    # Maintenance: rebuild_embeddings
    # ------------------------------------------------------------------

    def rebuild_embeddings(self, batch_size: int = 20) -> Dict[str, Any]:
        """Recompute embeddings for all facts. Similar to rebuild_all_vectors().

        Returns stats dict with: total, embedded, skipped, elapsed_sec.
        """
        if not self._embedder_available or not self._embedder or not self._embed_store:
            return {"error": "embedding backend not available", "total": 0, "embedded": 0}

        if not self._store:
            return {"error": "Store not initialized", "total": 0, "embedded": 0}

        with self._store._lock:
            rows = self._store._conn.execute(
                "SELECT fact_id, content FROM facts ORDER BY fact_id"
            ).fetchall()

        total = len(rows)
        embedded = 0
        skipped = 0
        t0 = time.perf_counter()

        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]
            contents = [row["content"] for row in batch]
            try:
                vecs = self._embedder.embed_batch(contents)
            except Exception as exc:
                logger.debug("rebuild_embeddings: batch embed failed at %d: %s", i, exc)
                skipped += len(batch)
                continue
            for row, vec in zip(batch, vecs):
                if vec is not None:
                    try:
                        self._embed_store.upsert(
                            int(row["fact_id"]),
                            vec,
                            embedding_identity=self._embedding_identity("document"),
                        )
                        embedded += 1
                    except Exception as exc:
                        logger.debug("rebuild_embeddings: fact %s upsert failed: %s", row["fact_id"], exc)
                        skipped += 1
                else:
                    skipped += 1

        elapsed = round(time.perf_counter() - t0, 2)
        logger.info(
            "holographic_plus: rebuild_embeddings complete, %d/%d embedded in %.1fs",
            embedded, total, elapsed,
        )
        return {
            "total": total,
            "embedded": embedded,
            "skipped": skipped,
            "elapsed_sec": elapsed,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit_embed(self, fn, *args) -> None:
        """Run an embedding task on the bounded pool (caps concurrent embeds).

        Falls back to running inline if the pool is absent or already shutting
        down, so a fact is never silently lost.
        """
        pool = self._embed_pool
        if pool is not None:
            try:
                pool.submit(fn, *args)
                return
            except RuntimeError:
                pass  # pool shut down mid-flight; fall through to inline
        try:
            fn(*args)
        except Exception as exc:
            logger.debug("holographic_plus: inline embed fallback failed: %s", exc)

    def _embed_and_store(self, fact_id: int, content: str) -> None:
        """Compute embedding for one fact and persist it (runs in a thread)."""
        try:
            if not self._embedder or not self._embed_store:
                return
            vec = self._embedder.embed(content)
            if vec is not None:
                self._embed_store.upsert(
                    fact_id,
                    vec,
                    embedding_identity=self._embedding_identity("document"),
                )
                logger.debug("holographic_plus: embedded fact %d", fact_id)
        except Exception as exc:
            logger.debug("holographic_plus: _embed_and_store(%d) failed: %s", fact_id, exc)

    def _backfill_embeddings(self, stop: Optional[threading.Event] = None) -> None:
        """Background thread: embed all facts that don't have embeddings yet.

        Checks *stop* between chunks so teardown can halt a long backfill
        instead of leaving it writing to a connection about to be replaced.
        """
        try:
            if not self._store or not self._embed_store or not self._embedder:
                return

            with self._store._lock:
                rows = self._store._conn.execute(
                    "SELECT fact_id, content FROM facts ORDER BY fact_id"
                ).fetchall()

            all_ids = [int(r["fact_id"]) for r in rows]
            missing_ids = self._embed_store.ids_without_embeddings(
                all_ids,
                embedding_identity=self._embedding_identity("document"),
            )

            if not missing_ids:
                logger.debug("holographic_plus: all %d facts already have embeddings", len(all_ids))
                return

            logger.info(
                "holographic_plus: backfilling embeddings for %d/%d facts",
                len(missing_ids), len(all_ids),
            )

            id_to_content = {int(r["fact_id"]): r["content"] for r in rows}
            count = 0

            # Batch-embed in chunks. embed_batch() is provably identical to per-fact
            # embed() (verified bit-for-bit) but issues one model/HTTP call per chunk
            # instead of one per fact. Empty-content facts are skipped.
            _BATCH = 32
            pending = [
                (fid, id_to_content.get(fid, ""))
                for fid in missing_ids
                if id_to_content.get(fid, "")
            ]
            for i in range(0, len(pending), _BATCH):
                if stop is not None and stop.is_set():
                    logger.debug(
                        "holographic_plus: backfill stopped early (%d embeddings added)",
                        count,
                    )
                    return
                chunk = pending[i:i + _BATCH]
                contents = [c for _, c in chunk]
                try:
                    vecs = self._embedder.embed_batch(contents)
                except Exception as exc:
                    logger.debug("holographic_plus backfill: batch embed failed: %s", exc)
                    continue
                for (fid, _content), vec in zip(chunk, vecs):
                    if vec is None:
                        continue
                    try:
                        self._embed_store.upsert(
                            fid,
                            vec,
                            embedding_identity=self._embedding_identity("document"),
                        )
                        count += 1
                    except Exception as exc:
                        logger.debug("holographic_plus backfill: fact %d upsert failed: %s", fid, exc)

            logger.info("holographic_plus: backfill complete, %d embeddings added", count)

        except Exception as exc:
            logger.warning("holographic_plus: backfill thread failed: %s", exc)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def _host_model_config() -> dict:
    """Read the host agent's model config (``model.provider`` + ``model.default``)
    so fact extraction can default to whatever model the user's Hermes already
    runs on, never a hardcoded provider. Returns {} if unavailable."""
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return {}
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("model", {}) or {}
    except Exception:
        return {}


def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("plugins", {}).get("hermes-memory-store", {}) or {}
    except Exception:
        return {}


def register(ctx) -> None:
    """Register holographic_plus memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicPlusProvider(config=config)
    ctx.register_memory_provider(provider)
