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
        embedding_prefix_policy: none    # "none" (default) or "auto" (apply the model's query/passage prefixes)
        hrr_weight: 0.2                  # set 0 to disable the HRR signal (fts/jaccard/hrr rescale to sum 1.0)
        embed_on_add: true               # embed immediately when a fact is added (default true)
        dedup_on_add: true               # skip near-verbatim restatements on add (default true)
        dedup_jaccard: 0.9               # word-overlap needed to call it a duplicate (same values required)
        dedup_cosine: 0.92               # dense cosine needed to call a paraphrase a duplicate (same values required)
        temporal_filter: true            # exclude structurally superseded facts from search (default true)
        extract_drain_batch: 5           # extraction queue rows processed per drain tick (default 5)
        # extraction_provider / extraction_model: default to the host agent's model
        entity_boost_weight: 0.0         # additive boost for facts linked to a query-mentioned entity (default off)
        entity_expansion: false          # 1-hop expansion to facts sharing an entity with a top hit (default off)
        entity_hub_degree_limit: 25      # entities linked to more facts than this are excluded from expansion
        reflection_enabled: false        # sleep-time reflection: synthesize insights from related facts (default off)
        reflection_interval_hours: 24    # minimum hours between reflection passes, persisted across restarts
        reflection_max_clusters: 3       # candidate clusters considered per reflection pass
        reflection_cosine_low: 0.75      # lower cosine bound for a "related" (non-duplicate) cluster
        reflection_cosine_high: 0.92     # upper cosine bound; at/above this the dedup gate already owns it

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
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from plugins.memory.holographic import HolographicMemoryProvider
from .embeddings import FastEmbedder, OllamaEmbedder


# --- Near-duplicate detection (write-time dedup guard) -----------------------
_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm_tokens(text: str) -> set:
    """Lowercased alphanumeric token set."""
    return set(_WORD_RE.findall((text or "").lower()))


def _value_tokens(text: str) -> set:
    """Tokens carrying a concrete value (numbers, ports, SHAs, ids, versions).

    A changed value means an UPDATE, not a duplicate, so these must match for a
    write to be treated as a near-duplicate and skipped.
    """
    return {t for t in _norm_tokens(text) if any(c.isdigit() for c in t)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_STOPWORDS = frozenset(
    "a an the is are was were be been being am of to in on at for and or with by "
    "as it its this that these those from into over under has have had do does did "
    "will would can could should may might must not no".split()
)


def _content_tokens(text: str) -> set:
    """Significant (non-function) words."""
    return _norm_tokens(text) - _STOPWORDS


def _is_near_duplicate(content: str, other: str, threshold: float) -> bool:
    """True only if *content* is a near-identical RESTATEMENT of *other*: the same
    concrete values AND the same content words (ignoring function words and word
    order), with token Jaccard >= *threshold* as a fast pre-filter.

    Deliberately conservative: any differing content word (active->archived,
    enabled->disabled, joining->declining) or any differing value means a possible
    UPDATE, so the fact is KEPT, never skipped. Reworded-but-equivalent duplicates
    are left to the value-aware, reviewable consolidation pass, not dropped here.
    """
    if _value_tokens(content) != _value_tokens(other):
        return False
    if _jaccard(_norm_tokens(content), _norm_tokens(other)) < threshold:
        return False
    return _content_tokens(content) == _content_tokens(other)


_RESTATEMENT_JACCARD = 0.7


def _is_semantic_duplicate(
    content: str, other: str, cosine: Optional[float], threshold: float
) -> bool:
    """True if *content* is a paraphrase of *other*: same concrete values (the
    value-token guard that keeps genuine updates from being dropped) AND a dense
    cosine similarity >= *threshold*.

    Catches reworded restatements that share few surface words (low Jaccard) but
    the same meaning, e.g. "prefers Postgres over MySQL" vs "always reaches for
    Postgres instead of MySQL". Returns False when *cosine* is None (embedder
    unavailable), so callers fall back to the Jaccard check alone.

    Guards against a gap the Jaccard path already closes: when *content* and
    *other* share almost all of their wording (raw Jaccard >= the near-restatement
    bar) but differ in a content word, that is a state-word flip (active ->
    archived, enabled -> disabled), i.e. an UPDATE, not a duplicate, even if the
    embedder's cosine similarity is high. A pair with mostly different wording is
    left to the cosine check alone, so genuine low-Jaccard paraphrases are still
    caught.
    """
    if cosine is None:
        return False
    if _value_tokens(content) != _value_tokens(other):
        return False
    if _content_tokens(content) != _content_tokens(other):
        if _jaccard(_norm_tokens(content), _norm_tokens(other)) >= _RESTATEMENT_JACCARD:
            return False
    return cosine >= threshold


_SUPERSEDED_PREFIXES = ("superseded", "stale/disabled", "historical/superseded")


def _is_superseded(content: str) -> bool:
    """True if *content* is explicitly marked as a retired/superseded fact, so
    retrieval can exclude it. Demote-not-delete only helps if reads skip these;
    trust demotion alone proved unreliable (markers landed at/above the floor)."""
    return (content or "").lstrip().lower().startswith(_SUPERSEDED_PREFIXES)


from .embed_store import EmbedStore
from .extract_queue import ExtractQueue, is_quota_error, quota_retry_delay
from .llm_extract import _format_conversation, extract_facts_from_transcript, insert_facts
from .reflection import ensure_reflection_schema, invalidate_insights_citing, run_reflection
from .retrieval_plus import PlusFactRetriever
from .temporal import ensure_temporal_schema, fact_history, find_value_update_target, supersede

logger = logging.getLogger(__name__)

# Default weights
_FTS_W     = 0.3
_JACCARD_W = 0.2
_HRR_W     = 0.2
_EMBED_W   = 0.3


# Per-model instruction prefixes, applied when embedding_prefix_policy="auto".
# Matched by case-insensitive substring on the embedding model name, so versioned
# ids resolve (e.g. "BAAI/bge-large-en-v1.5"). Each entry is
# (model_key, query_prefix, document_prefix) from the model's documented
# retrieval usage. Order matters: specific keys first, the broad "e5" key last.
# Models with no known prefix (e.g. bge-m3) fall through and embed verbatim.
_MODEL_PREFIXES = (
    ("embeddinggemma", "task: search result | query: ", "title: none | text: "),
    ("qwen3-embedding",
     "Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: ",
     ""),
    ("nomic-embed", "search_query: ", "search_document: "),
    ("arctic-embed2", "query: ", ""),
    ("arctic-embed", "Represent this sentence for searching relevant passages: ", ""),
    ("mxbai-embed", "Represent this sentence for searching relevant passages: ", ""),
    ("bge-large", "Represent this sentence for searching relevant passages: ", ""),
    ("bge-base", "Represent this sentence for searching relevant passages: ", ""),
    ("bge-small", "Represent this sentence for searching relevant passages: ", ""),
    ("e5", "query: ", "passage: "),
)


def _registry_prefix(model: str, role: str) -> str:
    """Return the documented query/document prefix for *model*, or "" if unknown."""
    m = (model or "").lower()
    for key, qpfx, dpfx in _MODEL_PREFIXES:
        if key in m:
            return qpfx if role == "query" else dpfx
    return ""


def _blend_score(holo_score: float, raw_emb_sim: Optional[float], ew: float) -> float:
    """Combine the holographic score and the dense-embedding similarity on one scale.

    ``holo_score`` is the parent's relevance × trust (range ``[0, trust]``), with the
    parent's FTS/Jaccard/HRR weights rescaled to sum to 1.0, and it carries the trust
    signal. It gets a ``(1 - ew)`` slice; the dense cosine gets ``ew``. The cosine is
    mapped ``[-1, 1] → [0, 1]`` and is deliberately NOT trust-weighted: multiplying it
    by trust let high-trust distractors outrank the correct default-trust fact and
    measured ~23 points lower recall@1, so trust influences ranking only via the
    holographic term. A fact with no embedding cannot earn the ``ew`` slice.
    """
    base = (1.0 - ew) * holo_score
    if raw_emb_sim is None:
        return base
    emb_norm = (raw_emb_sim + 1.0) / 2.0  # cosine [-1,1] → [0,1]
    return base + ew * emb_norm


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _cfg_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class HolographicPlusProvider(HolographicMemoryProvider):
    """Holographic memory + dense embedding retrieval (FastEmbed or Ollama)."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        cfg = self._config
        self._embed_weight: float = float(cfg.get("embedding_weight", _EMBED_W))
        self._embedding_backend: str = str(cfg.get("embedding_backend", "fastembed")).lower()
        self._embedding_prefix_policy: str = str(cfg.get("embedding_prefix_policy", "none")).lower()
        # Explicit prefix overrides win over the per-model registry when the
        # policy is not "none"; either may be set on its own.
        self._embedding_query_prefix: Optional[str] = cfg.get("embedding_query_prefix")
        self._embedding_document_prefix: Optional[str] = cfg.get("embedding_document_prefix")
        self._ollama_url: str     = str(cfg.get("ollama_url", "http://localhost:11434"))
        self._ollama_model: str   = str(cfg.get("ollama_model", "qwen3-embedding:8b"))
        self._fastembed_model: str = str(cfg.get("fastembed_model", "BAAI/bge-base-en-v1.5"))
        self._fastembed_cache_dir: Optional[str] = cfg.get("fastembed_cache_dir")
        self._embed_on_add: bool  = bool(cfg.get("embed_on_add", True))
        # Write-time dedup: skip a new fact that merely restates an existing one
        # (heavy word overlap AND identical concrete values, so updates are kept).
        self._dedup_on_add: bool   = bool(cfg.get("dedup_on_add", True))
        self._dedup_jaccard: float = float(cfg.get("dedup_jaccard", 0.9))
        # Semantic dedup: catches paraphrases that share few surface words (low
        # Jaccard) but the same meaning, via the dense cosine already computed
        # in the write path. ORed with the Jaccard check, both still gated by
        # the value-token guard so a changed number/SHA/state word is always
        # kept as an update.
        self._dedup_cosine: float  = float(cfg.get("dedup_cosine", 0.92))

        # Temporal validity: search excludes structurally superseded facts
        # (invalid_at set) unless disabled, reproducing pre-temporal ranking
        # exactly when turned off.
        self._temporal_filter: bool = _cfg_bool(cfg.get("temporal_filter"), True)

        # Optional final-stage retrieval decision gates. Disabled by default so
        # existing installs preserve current recall until a MemoryArena-calibrated
        # rule is explicitly enabled in config.
        self._retrieval_decision_enabled: bool = _cfg_bool(
            cfg.get("retrieval_decision_enabled"), False
        )
        self._retrieval_decision_min_score: Optional[float] = _cfg_float(
            cfg.get("retrieval_decision_min_score")
        )
        self._retrieval_decision_min_margin: Optional[float] = _cfg_float(
            cfg.get("retrieval_decision_min_margin")
        )
        self._retrieval_decision_min_trust: Optional[float] = _cfg_float(
            cfg.get("retrieval_decision_min_trust")
        )

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
        # Cap on rows processed per drain tick: a large backlog (e.g. after
        # extraction was down) drains over hours at the poll interval instead
        # of firing the extraction LLM back-to-back for the whole backlog.
        self._extract_drain_batch: int = int(cfg.get("extract_drain_batch", 5))

        # Sleep-time reflection: opt-in connection-drawing insight layer.
        # Off by default; a deployer enables it explicitly in config.
        self._reflection_enabled: bool = _cfg_bool(cfg.get("reflection_enabled"), False)
        self._reflection_interval_hours: float = float(cfg.get("reflection_interval_hours", 24))
        self._reflection_max_clusters: int = int(cfg.get("reflection_max_clusters", 3))
        self._reflection_cosine_low: float = float(cfg.get("reflection_cosine_low", 0.75))
        self._reflection_cosine_high: float = float(cfg.get("reflection_cosine_high", 0.92))

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

    def _prefix_for(self, role: str) -> str:
        """Resolve the instruction prefix for *role* ("query" or "document").

        Policy "none" (default) applies nothing. Otherwise an explicit config
        override wins, falling back to the per-model registry under "auto".
        """
        policy = getattr(self, "_embedding_prefix_policy", "none")
        if policy == "none":
            return ""
        if role == "query" and self._embedding_query_prefix is not None:
            return self._embedding_query_prefix
        if role == "document" and self._embedding_document_prefix is not None:
            return self._embedding_document_prefix
        if policy == "auto":
            return _registry_prefix(self._embedding_model_name(), role)
        return ""

    def _embed_text(self, text: str, role: str) -> str:
        """Prepend the role-appropriate instruction prefix (if any) to *text*."""
        prefix = self._prefix_for(role)
        return f"{prefix}{text}" if prefix else text

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

        # Additive schema migration for temporal validity: idempotent, so
        # safe to run on every initialize() including a re-init against the
        # same store.
        ensure_temporal_schema(self._store._conn)
        ensure_reflection_schema(self._store._conn)

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

        # Holographic signal weights are config-overridable so the HRR signal can
        # be down-weighted or disabled (hrr_weight: 0). Whatever the raw values,
        # they are rescaled to sum to 1.0 so the (1-ew)/ew blend budget holds.
        fts_cfg = float(self._config.get("fts_weight", _FTS_W))
        jac_cfg = float(self._config.get("jaccard_weight", _JACCARD_W))
        hrr_cfg = float(self._config.get("hrr_weight", _HRR_W))
        holo_sum = fts_cfg + jac_cfg + hrr_cfg
        if holo_sum <= 0:
            fts_cfg, jac_cfg, hrr_cfg = _FTS_W, _JACCARD_W, _HRR_W
            holo_sum = _FTS_W + _JACCARD_W + _HRR_W
        fts_w     = fts_cfg / holo_sum
        jaccard_w = jac_cfg / holo_sum
        hrr_w     = hrr_cfg / holo_sum

        self._retriever = PlusFactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            fts_weight=fts_w,
            jaccard_weight=jaccard_w,
            hrr_weight=hrr_w,
            hrr_dim=hrr_dim,
            entity_boost_weight=float(self._config.get("entity_boost_weight", 0.0)),
            entity_expansion=_cfg_bool(self._config.get("entity_expansion"), False),
            entity_hub_degree_limit=int(self._config.get("entity_hub_degree_limit", 25)),
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
            revived = self._extract_queue.revive_recent_quota_dead()
            if revived:
                logger.info(
                    "holographic_plus: auto-revived %d quota-dead extraction(s) "
                    "younger than the 48h age cap",
                    revived,
                )
        except Exception as exc:
            logger.debug("holographic_plus: quota-dead revival failed: %s", exc)
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
            if self._reflection_enabled:
                try:
                    self.run_reflection(time.time())
                except Exception as exc:
                    logger.warning("holographic_plus: reflection pass failed: %s", exc)
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
        """Process up to _extract_drain_batch pending rows, or until stop is set.

        A large backlog (extraction was down, or a burst of sessions ended at
        once) drains a few items per poll tick instead of firing the
        extraction LLM back-to-back for the whole backlog in one go.
        """
        if not self._store:
            return
        if not self._extract_provider or not self._extract_model:
            return

        processed = 0
        while not stop.is_set() and processed < self._extract_drain_batch:
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
                    search_fn=lambda topic, limit: self.search(
                        topic, min_trust=self._min_trust, limit=limit, bump=False
                    ),
                )
                inserted = 0
                if facts:
                    with self._deferred_bank_rebuild():
                        inserted = insert_facts(
                            self._store, facts, embed_callback=self._embed_cb,
                            dedup_check=self._find_near_duplicate if self._dedup_on_add else None,
                            update_check=self._find_update_target if self._dedup_on_add else None,
                            supersede=self._supersede_fact if self._dedup_on_add else None,
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
                if stop.is_set():
                    # Interrupted by shutdown/teardown: the DB connection is
                    # being closed under the worker (surfaces as a bad file
                    # descriptor). This is not a genuine extraction failure, so
                    # leave the row pending with its attempt count untouched and
                    # exit. The next worker drains it cleanly on startup, instead
                    # of burning a retry attempt and logging an error on every
                    # gateway restart.
                    logger.debug(
                        "holographic_plus: extraction interrupted by shutdown; "
                        "leaving queue item %s pending", row["id"],
                    )
                    break
                err = str(exc)
                if is_quota_error(err):
                    # Plan-limit windows reset on the provider's schedule, not
                    # ours: reschedule via not_before without consuming an
                    # attempt, so the row survives until the window reopens
                    # (bounded by the queue's 48h age cap). next_pending()
                    # skips the row until due, so no busy-spin: the worker
                    # falls back to its normal poll wait.
                    delay = quota_retry_delay(err)
                    try:
                        rescheduled = queue.mark_quota_failed(
                            row["id"], err, time.time() + delay
                        )
                    except Exception as mark_exc:
                        # Same safety bound as below: a row whose failure
                        # cannot be recorded must not re-run the LLM forever.
                        self._queue_mem_attempts[row["id"]] = (
                            self._queue_mem_attempts.get(row["id"], 0) + 1
                        )
                        logger.debug(
                            "holographic_plus: could not record queue quota "
                            "failure: %s",
                            mark_exc,
                        )
                        break
                    if rescheduled:
                        logger.warning(
                            "holographic_plus: queue item %d hit a provider "
                            "quota limit, next attempt in ~%ds: %s",
                            row["id"], int(delay), exc,
                        )
                    else:
                        logger.warning(
                            "holographic_plus: queue item %d marked dead by "
                            "the 48h age cap",
                            row["id"],
                        )
                    continue
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
            if action == "add" and self._embed_on_add and result.get("status") == "added":
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
        bump: bool = True,
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
        # Exclude explicitly-superseded facts from retrieval (demote-not-delete
        # only works if reads skip them; trust demotion alone proved unreliable).
        holo_results = [r for r in holo_results if not _is_superseded(r.get("content", ""))]
        if self._temporal_filter:
            holo_results = self._exclude_temporally_invalid(holo_results)

        if not self._embedder_available or not self._embed_store or not self._embedder:
            # Pure holographic fallback
            selected = self._apply_retrieval_decision(holo_results)[:limit]
            for r in selected:
                r["embedding_score"] = None
            if bump:
                self._bump_retrieval_counts(selected)
            return selected

        # --- Step 2: embed query + score all facts
        try:
            query_vec = self._embedder.embed(self._embed_text(query, "query"))
        except Exception as exc:
            logger.debug("holographic_plus: query embed failed: %s", exc)
            query_vec = None

        if query_vec is None:
            selected = self._apply_retrieval_decision(holo_results)[:limit]
            for r in selected:
                r["embedding_score"] = None
            if bump:
                self._bump_retrieval_counts(selected)
            return selected

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
                           retrieval_count, helpful_count, created_at, updated_at,
                           invalid_at
                    FROM facts
                    WHERE fact_id IN ({placeholders})
                      AND trust_score >= ?{category_clause}
                    """,
                    params,
                ).fetchall()
            for row in rows:
                d = dict(row)
                if _is_superseded(d.get("content", "")):
                    continue
                if self._temporal_filter and d.pop("invalid_at", None) is not None:
                    continue
                d.pop("invalid_at", None)
                d["score"] = 0.0  # no holographic score
                extra_facts.append(d)

        all_candidates = list(holo_results) + extra_facts

        for fact in all_candidates:
            fid = fact["fact_id"]
            holo_score = fact.get("score", 0.0)  # parent relevance × trust, in [0, trust]
            raw_emb_sim = emb_scores.get(fid)

            fact["score"] = _blend_score(holo_score, raw_emb_sim, ew)
            fact["embedding_score"] = round(raw_emb_sim, 4) if raw_emb_sim is not None else None

            merged.append(fact)

        # Primary key: blended relevance. Tie-break: more-recent first, so recency
        # only decides exact score ties and never displaces a clearly better match
        # (normalize the mixed "YYYY-MM-DD HH:MM" / "...THH:MM" timestamp formats).
        merged.sort(
            key=lambda x: (
                x["score"],
                (x.get("updated_at") or x.get("created_at") or "").replace(" ", "T"),
            ),
            reverse=True,
        )
        # Deduplicate by fact_id (extra_ids might overlap holo)
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for f in merged:
            if f["fact_id"] not in seen:
                seen.add(f["fact_id"])
                unique.append(f)

        selected = self._apply_retrieval_decision(unique)[:limit]
        if bump:
            self._bump_retrieval_counts(selected)
        return selected

    def _exclude_temporally_invalid(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop rows whose fact has been structurally superseded (invalid_at set).

        ``_retriever.search()`` candidate rows don't carry ``invalid_at`` (see
        ``PlusFactRetriever._fts_candidates``), so this looks it up in one
        query over the candidate fact_ids rather than changing the parent's
        candidate SQL. Degrades to returning *results* unchanged on error, so
        a lookup failure never breaks search.
        """
        if not results or not self._store:
            return results
        ids = [int(r["fact_id"]) for r in results]
        try:
            placeholders = ",".join("?" * len(ids))
            with self._store._lock:
                rows = self._store._conn.execute(
                    f"SELECT fact_id FROM facts "
                    f"WHERE fact_id IN ({placeholders}) AND invalid_at IS NOT NULL",
                    ids,
                ).fetchall()
        except Exception as exc:
            logger.debug("holographic_plus: temporal filter lookup failed: %s", exc)
            return results
        invalid_ids = {int(r["fact_id"]) for r in rows}
        if not invalid_ids:
            return results
        return [r for r in results if r["fact_id"] not in invalid_ids]

    def _apply_retrieval_decision(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply optional final-stage confidence gates to ranked search results.

        This is the production hook for MemoryArena-calibrated abstention: keep
        ranking untouched when disabled; otherwise remove low-confidence rows
        and, when configured, abstain from the whole query if the top two scores
        are too close to distinguish. Callers pass rows sorted by descending score.
        """
        if not self._retrieval_decision_enabled or not results:
            return results

        filtered: List[Dict[str, Any]] = []
        for row in results:
            if _is_superseded(row.get("content", "")):
                continue
            if self._retrieval_decision_min_score is not None:
                try:
                    score = float(row.get("score", 0.0))
                except (TypeError, ValueError):
                    continue
                if score < self._retrieval_decision_min_score:
                    continue
            if self._retrieval_decision_min_trust is not None:
                try:
                    trust = float(row.get("trust_score", 0.0))
                except (TypeError, ValueError):
                    continue
                if trust < self._retrieval_decision_min_trust:
                    continue
            filtered.append(row)

        min_margin = self._retrieval_decision_min_margin
        if min_margin is not None and len(filtered) > 1:
            try:
                margin = float(filtered[0].get("score", 0.0)) - float(filtered[1].get("score", 0.0))
            except (TypeError, ValueError):
                margin = None
            if margin is not None and margin < min_margin:
                return []
        return filtered

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

        update_target: Optional[Dict[str, Any]] = None
        if action == "add" and self._dedup_on_add:
            content = (args.get("content") or "").strip()
            category = args.get("category")
            dup = self._find_near_duplicate(content, category=category)
            if dup is not None:
                return _json.dumps({
                    "fact_id": dup.get("fact_id"),
                    "status": "deduped",
                    "note": (
                        f"near-duplicate of existing fact {dup.get('fact_id')}; "
                        "not stored again"
                    ),
                })
            update_target = self._find_update_target(content, category=category)

        # For all other actions delegate to parent
        result_json = super()._handle_fact_store(args)

        if action == "add" and update_target is not None:
            try:
                result = _json.loads(result_json)
            except Exception:
                result = {}
            new_fact_id = result.get("fact_id")
            if new_fact_id and result.get("status") == "added":
                self._supersede_fact(int(update_target["fact_id"]), int(new_fact_id))

        return result_json

    def _find_near_duplicate(
        self, content: str, category: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return an existing active fact that *content* merely restates, else None.

        Uses the hybrid search to fetch the closest existing facts, then keeps
        only a match that either:
          - overlaps heavily in wording (token Jaccard >= ``self._dedup_jaccard``), or
          - is a semantic paraphrase (dense cosine >= ``self._dedup_cosine``,
            using the embedding ``search()`` already computed for these
            candidates),
        AND carries the same concrete values in both cases, so a genuine value
        update is never skipped. Degrades to None (normal add) on error.
        """
        if not content:
            return None
        try:
            results = self.search(
                content,
                category=category,
                min_trust=self._min_trust,
                limit=3,
                bump=False,
            )
        except Exception as exc:
            logger.debug("holographic_plus: dedup search failed: %s", exc)
            return None
        for r in results:
            other = r.get("content", "")
            if _is_near_duplicate(content, other, self._dedup_jaccard):
                return r
            if _is_semantic_duplicate(
                content, other, r.get("embedding_score"), self._dedup_cosine
            ):
                return r
        return None

    def _find_update_target(
        self, content: str, category: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the existing active fact that *content* is a VALUE UPDATE of, else None.

        Only called after ``_find_near_duplicate`` finds no restatement: a
        value update (same content words, a changed concrete value, e.g.
        "port is 3100" -> "port is 3200") is exactly the case the dedup gate
        deliberately lets through as a new insert. Reuses the same candidate
        search so this costs no extra query beyond the dedup check already run.
        Degrades to None (plain add, no supersession) on error.
        """
        if not content:
            return None
        try:
            results = self.search(
                content, category=category, min_trust=self._min_trust,
                limit=3, bump=False,
            )
        except Exception as exc:
            logger.debug("holographic_plus: update-target search failed: %s", exc)
            return None
        return find_value_update_target(content, results)

    def _supersede_fact(self, old_fact_id: int, new_fact_id: int) -> None:
        """Structurally supersede *old_fact_id* with *new_fact_id* (invalidate-not-delete).

        Never fails the caller's insert: any error here is logged and
        swallowed, leaving both rows live rather than losing the new fact.
        """
        if not self._store or old_fact_id == new_fact_id:
            return
        try:
            with self._store._lock:
                supersede(self._store._conn, old_fact_id, new_fact_id)
                invalidate_insights_citing(self._store._conn, old_fact_id)
        except Exception as exc:
            logger.debug(
                "holographic_plus: supersede(%s -> %s) failed: %s",
                old_fact_id, new_fact_id, exc,
            )

    def fact_history(self, fact_id: int) -> List[Dict[str, Any]]:
        """Return the full supersession chain containing *fact_id*, oldest first.

        Public read API over ``temporal.fact_history``: walks
        ``superseded_by`` both directions from *fact_id* so callers can pass
        any fact in a chain and get the same complete history back.
        """
        if not self._store:
            return []
        with self._store._lock:
            return fact_history(self._store._conn, fact_id)

    # ------------------------------------------------------------------
    # Sleep-time reflection: connection-drawing insight layer
    # ------------------------------------------------------------------

    def run_reflection(self, now: float) -> int:
        """Run one opportunistic reflection pass; returns insights inserted.

        Called from the same background loop that drains the extraction
        queue. Inert when ``reflection_enabled`` is false (the default) or no
        extraction model is configured (reflection reuses the same host-LLM
        plumbing). The min-interval clock is persisted in the store, so a
        restart never resets it.
        """
        if not self._store or not self._reflection_enabled:
            return 0
        if not self._extract_provider or not self._extract_model:
            return 0
        return run_reflection(
            self._store._conn,
            now=now,
            enabled=self._reflection_enabled,
            interval_hours=self._reflection_interval_hours,
            max_clusters=self._reflection_max_clusters,
            provider=self._extract_provider,
            model=self._extract_model,
            effort=self._extract_effort,
            embed_store=self._embed_store if self._embedder_available else None,
            embedding_identity=self._embedding_identity("document"),
            cosine_low=self._reflection_cosine_low,
            cosine_high=self._reflection_cosine_high,
            entity_hub_degree_limit=int(self._config.get("entity_hub_degree_limit", 25)),
            dedup_check=self._find_near_duplicate if self._dedup_on_add else None,
            insert_fact=self._insert_reflection_fact,
            lock=getattr(self._store, "_lock", None),
        )

    def _insert_reflection_fact(self, content: str, category: str, tags: str) -> int:
        """Insert one accepted insight through the normal fact-add path.

        Reuses ``store.add_fact`` (so entity linking runs exactly as for any
        other fact) and triggers the same embed callback the extraction
        queue uses, so an insight is retrievable by dense similarity like
        everything else.
        """
        fact_id = self._store.add_fact(content, category=category, tags=tags)
        if fact_id:
            self._embed_cb(fact_id, content)
        return fact_id

    # ------------------------------------------------------------------
    # Maintenance: rebuild_embeddings
    # ------------------------------------------------------------------

    def rebuild_embeddings(self, batch_size: int = 20, prune_stale: bool = True) -> Dict[str, Any]:
        """Recompute embeddings for all facts. Similar to rebuild_all_vectors().

        When *prune_stale* is true (the default), vectors left behind by a
        superseded embedding model are dropped after the rebuild: every
        embeddable fact has just been re-embedded under the current identity,
        so any other identity is redundant. Pass ``prune_stale=False`` to keep
        them (for example while canarying a second model).

        Returns stats dict with: total, embedded, skipped, elapsed_sec,
        pruned_stale.
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
                vecs = self._embedder.embed_batch(
                    [self._embed_text(c, "document") for c in contents]
                )
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

        pruned_stale = 0
        if prune_stale and embedded > 0:
            # Every embeddable fact now has a current-identity vector, so any
            # vector under a different (superseded) identity is redundant.
            try:
                pruned_stale = self._embed_store.prune_identities(
                    {self._embedding_identity("document")}
                )
            except Exception as exc:
                logger.debug("rebuild_embeddings: prune_stale failed: %s", exc)

        logger.info(
            "holographic_plus: rebuild_embeddings complete, %d/%d embedded in "
            "%.1fs (%d superseded pruned)",
            embedded, total, elapsed, pruned_stale,
        )
        return {
            "total": total,
            "embedded": embedded,
            "skipped": skipped,
            "elapsed_sec": elapsed,
            "pruned_stale": pruned_stale,
        }

    def vacuum_embeddings(self, extra_keep=()) -> Dict[str, Any]:
        """Reclaim vectors left behind by superseded embedding models.

        Deletes every stored vector whose identity is not the current document
        identity, optionally preserving *extra_keep* identities (for instance a
        canary model running side by side). Unlike ``rebuild_embeddings`` this
        does NOT re-embed: a fact left without a current-identity vector is
        healed by the background backfill on its next pass, so callers reclaim
        space without paying for a full re-embed. Returns stats: ``pruned``,
        ``kept_identities``, and the per-identity counts ``before`` and ``after``.
        """
        if not self._embed_store:
            return {"error": "embedding store not initialized", "pruned": 0}
        current = self._embedding_identity("document")
        keep = {current, *(str(k) for k in extra_keep)}
        before = self._embed_store.identity_counts()
        try:
            pruned = self._embed_store.prune_identities(keep)
        except ValueError as exc:
            return {"error": str(exc), "pruned": 0}
        after = self._embed_store.identity_counts()
        if pruned:
            logger.info(
                "holographic_plus: vacuum_embeddings pruned %d superseded vector(s)",
                pruned,
            )
        return {
            "pruned": pruned,
            "kept_identities": sorted(keep),
            "before": before,
            "after": after,
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
            vec = self._embedder.embed(self._embed_text(content, "document"))
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
                    vecs = self._embedder.embed_batch(
                        [self._embed_text(c, "document") for c in contents]
                    )
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
