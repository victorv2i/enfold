# holographic_plus

A hybrid long-term memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

It extends the bundled **holographic** (HRR) fact store with **dense semantic embeddings** and **LLM-based fact extraction**, then merges every signal in a single hybrid retrieval pass, so the agent recalls facts by *meaning*, not just keyword overlap, while keeping the holographic store's symbolic strengths.

It does not replace the holographic provider, it subclasses it. All the holographic foundation (the SQLite fact store, FTS search, trust scoring, entity resolution, and HRR compositional retrieval) is dusterbloom's work; this plugin adds a layer on top. See [NOTICE](NOTICE).

## What it adds over base `holographic`

- **Dense embedding retrieval.** Each fact is embedded into a sidecar `fact_embeddings` table; queries are scored by cosine similarity. This catches paraphrases and fuzzy matches that pure keyword/FTS search misses.
- **Hybrid scoring.** Four signals merged per query: FTS keyword, Jaccard overlap, HRR compositional, and dense embedding, all on one trust-weighted scale. The weights genuinely partition the budget: the three holographic weights (defaults FTS 0.3, Jaccard 0.2, HRR 0.2) are rescaled by their own sum so they total exactly 1.0 inside the retriever, that holographic score then gets a `1 - embedding_weight` share of the final score and the dense cosine similarity gets the remaining `embedding_weight` share (default 0.3, so 70% holographic / 30% embedding). A fact with no stored embedding simply cannot earn the embedding share.
- **LLM fact extraction.** At session end (and just before context compression) it extracts durable, atomic facts from the conversation so nothing important is lost when the window rolls over. Extraction uses your agent's own model by default, no hardcoded provider, and is configurable.
- **Reliable extraction pipeline.** Transcripts are persisted to an on-disk queue and processed by a background worker, so extraction survives crashes and restarts and never blocks the session (see Reliability below).
- **Graceful fallback.** If the embedding backend is unreachable, it silently falls back to holographic-only scoring, never a hard failure.

## Requirements

- A working Hermes Agent install (this is a plugin for it, and it subclasses the bundled `holographic` provider).
- `numpy`.
- One embedding backend:
  - **FastEmbed**: local, CPU-only, recommended for privacy. `pip install fastembed`. The code default is `BAAI/bge-base-en-v1.5` (768-dim); set `fastembed_model` in config to run a larger model.
  - **Ollama**: point `ollama_url` at a running server and pick an embedding `ollama_model`.

## Install

Hermes auto-discovers any memory-provider directory under `~/.hermes/plugins/`, so installation is a copy:

```bash
git clone https://github.com/victorv2i/holographic-plus.git
cp -r holographic-plus/holographic_plus ~/.hermes/plugins/
```

Then select it as your provider:

```bash
hermes config set memory.provider holographic_plus
```

## Configuration

Under `plugins.hermes-memory-store` in `config.yaml`. Every base holographic key still applies, plus these:

```yaml
plugins:
  hermes-memory-store:
    embedding_backend: fastembed            # "fastembed" (local CPU, default) or "ollama"
    embedding_weight: 0.3                   # weight of dense similarity in the hybrid score
    embed_on_add: true                      # embed a fact immediately when it is added
    fastembed_model: BAAI/bge-base-en-v1.5  # 768-dim, local
    ollama_url: http://localhost:11434      # only if embedding_backend: ollama
    ollama_model: qwen3-embedding:8b        # only if embedding_backend: ollama
    # Fact extraction is optional and defaults to your agent's own model
    # (config.yaml `model.provider` + `model.default`). Override only to use a
    # different model than the agent runs on:
    # extraction_provider: <provider>
    # extraction_model: <model>
    # extraction_effort: high               # only if your provider supports reasoning effort
```

Embeddings are **identity-versioned** (`backend:model:role:vN`), so switching models never corrupts existing vectors: stale ones are simply re-embedded on the next backfill. On startup, any fact missing an embedding is backfilled in a non-blocking background thread (in batches), and the background worker re-runs the backfill periodically so transient embedding failures heal on their own.

## Tools

It inherits the base `fact_store` tool (add / search / probe / related / reason / contradict / update / remove / list) and `fact_feedback` (rate facts helpful/unhelpful to train trust). The `search` action is overridden to merge dense-embedding similarity into the ranked results.

## How it works

- `prefetch(query)` runs the hybrid search and injects the top matches into the agent's context each turn.
- `handle_tool_call` / `on_memory_write` keep the embedding sidecar in sync on every add/update/remove.
- `on_session_end` runs the base regex extraction, then enqueues an LLM pass for the facts regex misses; `on_pre_compress` enqueues an extraction before the context window is trimmed and returns immediately, so compression is never blocked. Both feed the persistent extraction queue below.
- A shared, write-invalidated cache holds the normalized embedding matrix so repeated queries within a session stay fast.

## Reliability

- **Persistent extraction queue.** Transcripts to extract are written to an `extract_queue` table in the same SQLite database as the facts, so queued work survives crashes, gateway restarts, and LLM timeouts. Rows left behind by a previous run are drained automatically on the next startup.
- **Retries with backoff, dead-letter rows.** A failed extraction attempt is retried with exponential backoff and jitter (5 attempts by default). Rows that keep failing are kept with status `dead` for inspection instead of being silently dropped, and an in-memory attempt bound stops the LLM from being re-run indefinitely even when the failure bookkeeping itself cannot be written.
- **Non-blocking hooks.** The session-end and pre-compression hooks only format and enqueue the transcript; the LLM call happens on a single background worker thread, so session teardown and context compression never wait on a model.
- **Safe re-initialization.** Re-initializing the provider tears down the previous worker, backfill thread, and embedding pool first, and only closes the previous database connection once background work is quiescent (otherwise it is deliberately left open rather than closed under a writer).

## Performance

- The query is HRR-encoded exactly once per search. The parent retriever re-encodes it per candidate, which dominated prefetch latency.
- FTS candidate rows are fetched without the HRR vector blob; blobs are loaded in a single targeted query only for the candidates that get HRR-scored, and not at all when HRR is disabled. Scoring stays bit-identical to the parent (covered by an equivalence test against a real Hermes checkout).
- An extraction batch rebuilds each affected category memory bank once, not once per inserted fact, and the rebuilds run outside the store lock so they never stall concurrent reads and adds.
- The SQLite WAL is truncated at startup and after each extraction batch, so the `-wal` sidecar cannot grow without bound under the extraction and embedding write load.

## Evaluation

`tests/eval.py` is a self-contained recall benchmark: it seeds a synthetic corpus into the real `EmbedStore`, runs paraphrased queries (deliberately low keyword overlap with their target), and reports recall@k for a keyword baseline vs the dense-embedding retrieval this plugin adds. Reproduce it with `python tests/eval.py`.

On the bundled 24-fact / 16-query paraphrase set:

```
retriever     recall@1  recall@3  recall@5     MRR
--------------------------------------------------
keyword           0.38      0.56      0.62    0.51
embedding         0.81      0.88      0.88    0.86
blend*            0.81      0.88      0.88    0.86
```

The dense-embedding layer roughly doubles recall@1 over keyword (0.38 → 0.81) on paraphrased queries, the exact case where keyword search fails. `*` The `blend` row is an *illustrative* keyword+embedding mix for this isolated comparison; it is **not** the formula the plugin ships. The real scorer adds the trust-weighted embedding onto the trust-weighted holographic score across FTS+Jaccard+HRR (see `_blend_score`). The keyword and HRR signals earn their weight on literal and compositional queries, which this benchmark intentionally underweights. It is a synthetic, illustrative benchmark, not a production guarantee, but every number is reproducible from the script.

## Credits

Built on the **holographic** memory provider by **dusterbloom** (NousResearch/hermes-agent, PR #2351), which implements **Holographic Reduced Representations**: Tony A. Plate, *"Holographic Reduced Representations,"* IEEE Transactions on Neural Networks 6(3):623-641, 1995. Full attribution in [NOTICE](NOTICE).

The dense-embedding retrieval, LLM fact extraction, and hybrid scoring are this project's contribution.

## License

MIT, see [LICENSE](LICENSE).

## Release notes

### 0.3.0

- Fact extraction moved from fire-and-forget threads to a persistent SQLite-backed queue with a single background worker: retries with backoff, dead-letter rows for inspection, and automatic draining of work left by a previous run. Pre-compression extraction no longer blocks.
- Hybrid weights now genuinely partition the score: the holographic weights are rescaled to sum to 1.0 and the blend splits the budget `1 - embedding_weight` / `embedding_weight`.
- Faster retrieval: the query is HRR-encoded once per search and FTS candidate fetches skip the HRR vector blobs (bit-identical scoring, equivalence-tested).
- Category and trust filters now also apply to candidates surfaced only by embedding similarity.
- The extraction dedup context blends top-trust facts with the most recently created ones, so freshly stored facts are not re-extracted as paraphrases.
- WAL truncation at startup and after extraction batches; one bank rebuild per category per batch, run outside the store lock; safer re-initialization and shutdown.
- Search results bump `retrieval_count`, matching the parent store's idiom.
- Standalone pytest suite under `tests/`: runs with no Hermes install, no network, and no real embedding backend.

### 0.2.0

- Portable extraction (defaults to the host agent's own model) and FastEmbed as the default embedding backend.
- Robust JSON parsing for extraction output, the corrected hybrid blend, and the recall benchmark in `tests/eval.py`.
