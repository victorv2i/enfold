"""holographic_plus — Holographic memory with dense embedding retrieval.

Extends HolographicMemoryProvider with a 4th retrieval signal: dense cosine
similarity. The embedder is pluggable — FastEmbed (local CPU, the default) or
Ollama — and durable facts are extracted at session end by a configurable LLM
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

Retrieval weights (must sum to ≤ 1.0; remainder goes to trust scaling):
    FTS=0.3, Jaccard=0.2, HRR=0.2, Embedding=0.3

If the embedding backend is unreachable the plugin falls back silently to
holographic-only scoring (embedding weight redistributed to the other three).

First-run behaviour:
    On initialize(), any fact that lacks an embedding is queued for batch
    embedding in a background thread so startup is non-blocking.

Usage — change config.yaml::

    memory:
      provider: holographic_plus
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from plugins.memory.holographic import HolographicMemoryProvider
from .embeddings import FastEmbedder, OllamaEmbedder
from .embed_store import EmbedStore
from .llm_extract import extract_facts_from_session

logger = logging.getLogger(__name__)

# Default weights
_FTS_W     = 0.3
_JACCARD_W = 0.2
_HRR_W     = 0.2
_EMBED_W   = 0.3

# Holographic base weights (without embedding) must also sum correctly.
# The parent FactRetriever is constructed with fts=0.4, jaccard=0.3, hrr=0.3
# but we override via config so the parent uses our reduced weights.
_PARENT_FTS_W     = _FTS_W     / (1.0 - _EMBED_W)   # ≈ 0.4286
_PARENT_JACCARD_W = _JACCARD_W / (1.0 - _EMBED_W)   # ≈ 0.2857
_PARENT_HRR_W     = _HRR_W    / (1.0 - _EMBED_W)    # ≈ 0.2857


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
        # else disabled. Never hardcodes a provider — works with whatever the user runs.
        _host_model = _host_model_config()
        self._extract_provider: Optional[str] = cfg.get("extraction_provider") or _host_model.get("provider")
        self._extract_model: Optional[str]    = cfg.get("extraction_model") or _host_model.get("default")
        self._extract_effort: Optional[str]   = cfg.get("extraction_effort")

        self._embedder: Optional[Any] = None
        self._embed_store: Optional[EmbedStore]  = None
        self._ollama_available: bool             = False
        self._backfill_thread: Optional[threading.Thread] = None

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
        """Initialize parent store + embedding layer."""
        # ---- Adjust parent retrieval weights so HRR+FTS+Jaccard use remaining budget
        # We pass our reduced weights so the parent FactRetriever's combined score
        # represents (1 - embed_weight) of the total budget.
        ew = self._embed_weight
        remaining = max(0.0, 1.0 - ew)
        # Scale parent weights proportionally so they fill the remaining budget
        cfg_override = dict(self._config)
        # FactRetriever picks these up via hrr_weight key in config... but actually
        # the parent passes them directly to FactRetriever. We set them in config so
        # our overridden initialize() can forward them.
        cfg_override.setdefault("hrr_weight", round(_HRR_W / remaining, 6) if remaining else 0.0)

        # Store the original config reference, swap temporarily
        _orig_config = self._config
        self._config = cfg_override
        super().initialize(session_id, **kwargs)
        self._config = _orig_config  # restore

        # ---- Re-create FactRetriever with correct scaled weights
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        if remaining > 0:
            fts_w     = round(_FTS_W     / remaining, 6)
            jaccard_w = round(_JACCARD_W / remaining, 6)
            hrr_w     = round(_HRR_W     / remaining, 6)
        else:
            fts_w = jaccard_w = hrr_w = 0.0

        from plugins.memory.holographic.retrieval import FactRetriever
        self._retriever = FactRetriever(
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
        )
        self._ollama_available = self._embedder.is_available()

        if self._ollama_available:
            logger.info(
                "holographic_plus: embedding backend available (%s, model=%s)",
                self._embedding_backend, self._embedding_model_name(),
            )
            # Kick off background backfill for facts without embeddings
            self._backfill_thread = threading.Thread(
                target=self._backfill_embeddings,
                daemon=True,
                name="holographic_plus_backfill",
            )
            self._backfill_thread.start()
        else:
            logger.warning(
                "holographic_plus: embedding backend %s not available — "
                "falling back to holographic-only retrieval",
                self._embedding_backend,
            )

    def shutdown(self) -> None:
        super().shutdown()
        self._embedder    = None
        self._embed_store = None

    # ------------------------------------------------------------------
    # Session end — LLM-based fact extraction (configurable model)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pre-compression hook — save facts BEFORE context window is trimmed
    # ------------------------------------------------------------------

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract and save durable facts from messages about to be compressed.

        Called by MemoryManager before Hermes compresses the context window.
        We run a lightweight synchronous extraction (no daemon thread — we need
        to return before compression proceeds) using the same GPT-5.5 pipeline
        as on_session_end but with a tighter budget (3 facts max, 30s timeout).

        Returns an empty string — we don't inject anything into the compression
        prompt itself; we just side-effect into the fact store so nothing is lost.
        """
        if not self._store or not messages:
            return ""

        # Only act on messages that are about to be discarded.
        # Compression typically trims the oldest messages — we only care if
        # there's meaningful content (at least 4 turns of real dialogue).
        real_messages = [
            m for m in messages
            if m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
            and len(m["content"].strip()) > 20
        ]
        if len(real_messages) < 4:
            return ""

        try:
            self._extract_before_compress(messages)
        except Exception as exc:
            logger.debug("holographic_plus: on_pre_compress failed: %s", exc)

        return ""  # don't inject into compression prompt

    def _extract_before_compress(self, messages: List[Dict[str, Any]]) -> None:
        """Synchronous lightweight extraction before compression. 3 facts max."""
        try:
            from agent.auxiliary_client import call_llm
        except ImportError:
            return
        if not self._extract_provider or not self._extract_model:
            return

        from .llm_extract import _format_conversation, _existing_summary, _parse_response

        _COMPRESS_SYSTEM = """\
You are a memory preservation assistant. A conversation is about to be compressed
and older messages may be lost. Extract up to 3 facts that would be lost after
compression but are worth remembering long-term.

RULES:
1. Extract 0-3 facts. Only facts that survive past this session.
2. Each fact: one atomic statement, under 400 characters.
3. No ephemeral details, no task instructions, no meta-commentary.
4. Assign category: user_pref | project | tool | general
5. Assign 2-5 comma-separated tags (lowercase, no spaces).
6. JSON array only. Output [] if nothing is worth saving.
"""

        conversation = _format_conversation(messages, max_chars=8000)
        if not conversation.strip():
            return

        existing = _existing_summary(self._store, limit=30)
        user_msg = (
            f"Existing facts (do not re-extract):\n{existing}\n\n"
            f"---CONVERSATION ABOUT TO BE COMPRESSED---\n{conversation}\n---END---\n\n"
            "Extract up to 3 facts that would otherwise be lost."
        )

        try:
            resp = call_llm(
                provider=self._extract_provider,
                model=self._extract_model,
                messages=[
                    {"role": "system", "content": _COMPRESS_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=512,
                timeout=30,
                extra_body=({"reasoning": {"effort": self._extract_effort}} if self._extract_effort else None),
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.debug("holographic_plus: pre-compress LLM call failed: %s", exc)
            return

        facts = _parse_response(raw)
        if not facts:
            return

        def _embed_cb(fact_id: int, content: str) -> None:
            if self._ollama_available and self._embedder and self._embed_store:
                self._embed_and_store(fact_id, content)

        inserted = 0
        for fact in facts:
            try:
                fact_id = self._store.add_fact(
                    fact["content"],
                    category=fact["category"],
                    tags=fact["tags"],
                )
                inserted += 1
                if fact_id:
                    threading.Thread(
                        target=_embed_cb,
                        args=(fact_id, fact["content"]),
                        daemon=True,
                    ).start()
            except Exception as exc:
                logger.debug("holographic_plus: pre-compress add_fact failed: %s", exc)

        if inserted:
            logger.info(
                "holographic_plus: on_pre_compress saved %d/%d facts before compression",
                inserted, len(facts),
            )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """At session end: run regex auto-extract (parent) then LLM extraction.

        Parent handles cheap regex patterns (I prefer X, we decided Y).
        We then run an LLM pass (the host agent's model by default) to catch everything else.
        Runs in a daemon thread — never blocks session teardown.
        """
        # Run parent regex extraction first (fast, cheap, synchronous)
        super().on_session_end(messages)

        # Then kick off LLM extraction in background
        if not self._store or not messages:
            return

        # Build embed callback so new facts get embedded immediately
        def _embed_cb(fact_id: int, content: str) -> None:
            if self._ollama_available and self._embedder and self._embed_store:
                self._embed_and_store(fact_id, content)

        extract_facts_from_session(
            messages=messages,
            store=self._store,
            embed_callback=_embed_cb,
            blocking=False,
            provider=self._extract_provider,
            model=self._extract_model,
            effort=self._extract_effort,
        )

    # ------------------------------------------------------------------
    # Tool handler — intercept 'add' to embed new facts
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
                if fact_id and self._ollama_available and self._embed_store:
                    content = args.get("content", "")
                    threading.Thread(
                        target=self._embed_and_store,
                        args=(fact_id, content),
                        daemon=True,
                        name=f"embed_fact_{fact_id}",
                    ).start()

            elif action == "update" and result.get("updated") and args.get("content"):
                fact_id = int(args["fact_id"])
                content = args.get("content", "")
                if self._ollama_available and self._embedder and self._embed_store:
                    self._embed_and_store(fact_id, content)
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

        if action == "add" and self._embed_on_add and content and self._ollama_available:
            # Find the fact_id that was just inserted
            try:
                row = self._store._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content.strip(),)
                ).fetchone()
                if row:
                    threading.Thread(
                        target=self._embed_and_store,
                        args=(int(row["fact_id"]), content),
                        daemon=True,
                    ).start()
            except Exception as exc:
                logger.debug("holographic_plus: on_memory_write embed failed: %s", exc)

    # ------------------------------------------------------------------
    # Prefetch — merge holographic + embedding scores
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

        if not self._ollama_available or not self._embed_store or not self._embedder:
            # Pure holographic fallback
            for r in holo_results[:limit]:
                r["embedding_score"] = None
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

        # Fetch extra facts by ID if needed
        extra_facts: List[Dict[str, Any]] = []
        if extra_ids and self._store:
            placeholders = ",".join("?" * len(extra_ids))
            rows = self._store._conn.execute(
                f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE fact_id IN ({placeholders})
                  AND trust_score >= ?
                """,
                list(extra_ids) + [min_trust],
            ).fetchall()
            for row in rows:
                d = dict(row)
                d["score"] = 0.0  # no holographic score
                extra_facts.append(d)

        all_candidates = list(holo_results) + extra_facts

        for fact in all_candidates:
            fid = fact["fact_id"]
            holo_score = fact.get("score", 0.0)   # trust-weighted holographic score
            raw_emb_sim = emb_scores.get(fid)

            if raw_emb_sim is not None:
                # Shift [-1,1] → [0,1] then weight
                emb_contribution = ew * (raw_emb_sim + 1.0) / 2.0
                # Holographic contribution (already scaled by 1-ew inside FactRetriever)
                # We add the embedding on top of the existing trust-weighted score
                fact["score"] = holo_score + emb_contribution
                fact["embedding_score"] = round(raw_emb_sim, 4)
            else:
                fact["embedding_score"] = None
                # No embedding — don't penalise, just keep holographic score

            merged.append(fact)

        merged.sort(key=lambda x: x["score"], reverse=True)
        # Deduplicate by fact_id (extra_ids might overlap holo)
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for f in merged:
            if f["fact_id"] not in seen:
                seen.add(f["fact_id"])
                unique.append(f)

        return unique[:limit]

    # ------------------------------------------------------------------
    # Tool handler override — expose search with embeddings
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
        if not self._ollama_available or not self._embedder or not self._embed_store:
            return {"error": "Ollama not available", "total": 0, "embedded": 0}

        if not self._store:
            return {"error": "Store not initialized", "total": 0, "embedded": 0}

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
            "holographic_plus: rebuild_embeddings complete — %d/%d embedded in %.1fs",
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

    def _backfill_embeddings(self) -> None:
        """Background thread: embed all facts that don't have embeddings yet."""
        try:
            if not self._store or not self._embed_store or not self._embedder:
                return

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

            logger.info("holographic_plus: backfill complete — %d embeddings added", count)

        except Exception as exc:
            logger.warning("holographic_plus: backfill thread failed: %s", exc)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def _host_model_config() -> dict:
    """Read the host agent's model config (``model.provider`` + ``model.default``)
    so fact extraction can default to whatever model the user's Hermes already
    runs on — never a hardcoded provider. Returns {} if unavailable."""
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
