<p align="center">
  <img src="assets/logo.png" alt="Enfold logo: folded planes of light converging on a single bright point" width="280">
</p>

<h1 align="center">Enfold</h1>

<p align="center"><strong>Agent memory for builders who want recall with receipts.</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/MCP-stdio%20server-1f6feb" alt="MCP stdio server">
  <img src="https://img.shields.io/badge/hermes--agent-memory%20plugin-8A2BE2" alt="Hermes Agent memory plugin">
  <img src="https://img.shields.io/badge/local--first-SQLite-044a10" alt="local-first SQLite">
</p>

Enfold is a local SQLite memory layer for AI agents. It gives MCP-capable agents and Hermes Agent a shared fact store with temporal supersession, write-time dedup gates, hybrid holographic plus dense retrieval, and a reproducible eval harness.

It is for people building agents that run across sessions, terminals, and tools, where stale facts and duplicate memories are bugs. Try it in 60 seconds with the MCP server below; no hosted service or container is required.

The name is the design. In a hologram, every fragment of the plate reconstructs the whole scene. Enfold treats memory the same way: facts are stored once, atomically, and a partial cue (a paraphrase, an entity, a stray keyword) can bring back full context.

## 60-second Quickstart

```bash
git clone https://github.com/victorv2i/enfold.git
cd enfold
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[mcp,fastembed]"
python enfold/mcp_server.py \
    --db-path /tmp/enfold-memory.db \
    --embedding-backend fastembed \
    --hrr-dim 256
```

That last command is the stdio MCP server. In a real client config, use the same file-path launch and arguments:

```toml
[mcp_servers.enfold]
command = "/absolute/path/to/enfold/.venv/bin/python"
args = ["/absolute/path/to/enfold/enfold/mcp_server.py",
        "--db-path", "/home/you/enfold-memory.db",
        "--embedding-backend", "fastembed",
        "--hrr-dim", "256"]
```

Verified smoke behavior in this review: `python3 enfold/mcp_server.py --db-path /tmp/enfold-review-host.db --embedding-backend fastembed --hrr-dim 256` reached the stdio server loop with no startup error. In this sandbox, public `git clone` and `pip install` could not be fully verified because DNS/PyPI access was unavailable.

Two launch details matter:

- Run the server by file path, not `python -m enfold.mcp_server` (details in [MCP server](#mcp-server)).
- For a new standalone MCP store, `--hrr-dim 256` is the documented quickstart geometry. When sharing a store with a live Hermes gateway, use the gateway's configured `hrr_dim` instead.

## Feature Map

- **Recall by meaning, not keywords.** Dense embeddings, BM25 keyword search, token overlap, and holographic (HRR) compositional retrieval are blended into one trust-weighted score. Paraphrases hit; so do exact ports, SHAs, and hostnames.
- **One store, many agents.** The MCP server and the Hermes gateway operate on the same SQLite file concurrently, with WAL plus a cross-process write lock. Every write carries a `source` tag, so you always know which agent learned what.
- **Writes are gated, so the store stays clean.** Near-duplicate restatements are rejected at write time. A genuine value update (same fact, new number or state) supersedes the old fact instead of coexisting with it, and the history chain is preserved: invalidate, never delete.
- **Local-first, nothing leaves your machine.** SQLite for storage, local embeddings via Ollama (GPU) or FastEmbed (CPU-only, zero infrastructure). If the embedder is down, recall degrades gracefully to keyword and symbolic search instead of failing.
- **Benchmarked behavior.** Ships a reproducible recall benchmark; every number in this README comes from a script in the repo. When a cross-encoder reranker regressed recall on this workload, it was dropped and the negative result documented.
- **Built for unattended operation.** Fact extraction runs through a crash-safe SQLite queue with retries, backoff, and dead-letter rows. Sleep-time reflection distills grounded insights from clusters of related facts, with mandatory citations back to sources.
- **Self-tuning loop.** `memory_eval/autotune.py` copies the live SQLite DB, runs bounded retrieval trials on snapshots, rejects configs that leak stale facts, and writes a proposal report without mutating the live config.

## Numbers

Reports land under `$HERMES_HOME/reports/memory-eval/` (one JSON plus a short summary per run).

Environment notes: single-user local SQLite store; Python provider path; 50 exact-fact smoke cases; live DB copied with the SQLite backup API before provider load. The first run had the Ollama backend unavailable and fell back to holographic-only retrieval. A same-day rerun used Ollama with `embeddinggemma` confirmed in metadata. This is a smoke test for current-fact recall and stale suppression, not a hard semantic benchmark.

| Run | Recall@1 | MRR | Stale leak@1 | Stale leak@3 | Latency |
| --- | ---: | ---: | ---: | ---: | --- |
| Holographic fallback | 1.0000 | 1.0000 | 0/50 | 0/50 | mean 3.885 ms, p95 6.819 ms |
| Dense path, Ollama `embeddinggemma` | 1.0000 | 1.0000 | 0/50 | 0/50 | mean 81.4 ms, p95 92.5 ms |

What this says honestly: the exact-fact smoke set passed, temporal stale suppression held, and dense retrieval added about 77 ms mean latency without changing outcomes on this easy set. The next eval investment is a semantic case set with paraphrases and distractors.

## Quickstart: Claude Code

```bash
git clone https://github.com/victorv2i/enfold.git
cd enfold
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[mcp,fastembed]"

claude mcp add enfold -- /absolute/path/to/enfold/.venv/bin/python /absolute/path/to/enfold/enfold/mcp_server.py \
    --db-path ~/enfold-memory.db \
    --embedding-backend fastembed \
    --hrr-dim 256
```

That's the whole setup. Your agent now has `memory_search`, `memory_add`, `memory_supersede`, `memory_explain`, and `memory_history`. No server to babysit, no container, no cloud.

Two things to know:

- Run the server **by file path**, never `python -m` (details in [MCP server](#mcp-server)).
- For a new standalone MCP store, pass `--hrr-dim 256`. For a shared Hermes store, pass the gateway's configured `hrr_dim`.
- FastEmbed downloads its model on first run; without an embedding backend, the server still works with keyword and symbolic recall only.
- Without a Hermes checkout, the server runs on a bundled lightweight engine (real SQLite, FTS5, trust scores, and entity links; the same harness the 300-test suite runs against). For the full holographic engine, add `--hermes-src /path/to/hermes-agent/src`. A plain `git clone` of hermes-agent is enough: no install, no running gateway.

## Quickstart: Codex CLI

In `~/.codex/config.toml`:

```toml
[mcp_servers.enfold]
command = "/absolute/path/to/enfold/.venv/bin/python"
args = ["/absolute/path/to/enfold/enfold/mcp_server.py",
        "--db-path", "/home/you/enfold-memory.db",
        "--embedding-backend", "fastembed",
        "--hrr-dim", "256"]
```

Any other MCP-capable agent wires up the same way: it is a plain stdio MCP server.

## Quickstart: Hermes Agent (native provider)

Hermes auto-discovers memory-provider directories under `~/.hermes/plugins/`, so installation is a copy:

```bash
git clone https://github.com/victorv2i/enfold.git
cp -r enfold/enfold ~/.hermes/plugins/
hermes config set memory.provider enfold
```

As a native provider, Enfold does more than serve tools: it prefetches relevant facts into context every turn, extracts durable facts from conversations at session end and before context compression, and keeps the embedding sidecar in sync on every write.

To share the gateway's store with your other agents, point the MCP server at the same file and matching geometry:

```bash
python3 enfold/mcp_server.py \
    --db-path ~/.hermes/memory_store.db \
    --hermes-src ~/hermes-agent/src \
    --hrr-dim <your gateway's hrr_dim>
```

## What it adds over base `holographic`

Enfold does not replace the bundled Hermes holographic provider, it subclasses it. The foundation (SQLite fact store, FTS search, trust scoring, entity resolution, HRR compositional retrieval) is dusterbloom's work; see [NOTICE](NOTICE). On top of it:

- **Dense embedding retrieval.** Each fact is embedded into a sidecar `fact_embeddings` table; queries are scored by cosine similarity. This catches paraphrases and fuzzy matches that pure keyword/FTS search misses.
- **Hybrid scoring.** Four signals merged per query: FTS keyword, Jaccard overlap, HRR compositional, and dense embedding, all on one trust-weighted scale. The weights genuinely partition the budget: the three holographic weights (defaults FTS 0.3, Jaccard 0.2, HRR 0.2) are rescaled by their own sum so they total exactly 1.0 inside the retriever, that holographic score then gets a `1 - embedding_weight` share of the final score and the dense cosine similarity gets the remaining `embedding_weight` share (default 0.3, so 70% holographic / 30% embedding). A fact with no stored embedding simply cannot earn the embedding share.
- **Write-time dedup.** Adds are checked against existing facts with a token-Jaccard gate and a semantic cosine gate. A changed number, id, or state word is never treated as a duplicate (an update must always land), and an antonym-flip guard keeps opposite-meaning paraphrases apart.
- **Temporal validity.** Value updates supersede the prior fact (`valid_from` / `invalid_at` / `superseded_by`), superseded facts stop surfacing in search, and `memory_history` walks the chain. Invalidate, never delete.
- **LLM fact extraction.** At session end (and just before context compression) it extracts durable, atomic facts from the conversation so nothing important is lost when the window rolls over. Extraction uses your agent's own model by default, no hardcoded provider, and is configurable.
- **Reliable extraction pipeline.** Transcripts are persisted to an on-disk queue and processed by a background worker, so extraction survives crashes and restarts and never blocks the session (see [Reliability](#reliability)).
- **Sleep-time reflection.** On a configurable interval, clusters of related facts are distilled into grounded insight facts with mandatory citations to their sources; an insight is invalidated automatically when its sources are superseded.
- **Entity-graph boost (off by default).** The base store links facts to entities it recognises. When enabled, a query mentioning an entity boosts facts linked to it, and can expand one hop to related facts with no lexical overlap with the query. High-degree hub entities (e.g. the user's own name) are excluded so they can't flood results.
- **Graceful fallback.** If the embedding backend is unreachable, it silently falls back to holographic-only scoring, never a hard failure.

## Benchmarks

A reproducible recall benchmark (`tests/eval.py` for the dense layer, plus the `memory_eval/` end-to-end harness for provider sweeps on SQLite snapshots) measures single-user self-hosted stores. One earlier 120-fact / 72-query sweep with hard distractor clusters produced:

```
bge-large, no prefix, HRR on (typical default)    r@1 0.69   MRR 0.79
bge-large, +prefix, HRR on                        r@1 0.78   MRR 0.85
embeddinggemma, +prefix, HRR off                  r@1 0.90   MRR 0.94
```

What the sweep taught us:

- The dense embedding model is the dominant lever. The embedding weight and the HRR signal do not move recall on semantic / paraphrase queries, where keyword search has little to grip; the large per-fact HRR vector is effectively dead weight there.
- Instruction prefixes matter: `embedding_prefix_policy: auto` applies each model's documented query/passage prefixes and is worth several recall points on models tuned for them.
- A final-stage cross-encoder reranker did not help on short atomic facts paired with a strong embedder; it regressed recall, so it is not recommended here.

Recommended configuration (GPU via Ollama):

```yaml
plugins:
  hermes-memory-store:
    embedding_backend: ollama
    ollama_model: embeddinggemma            # 768-dim, Apache-2.0
    embedding_prefix_policy: auto
    hrr_weight: 0                           # HRR adds no measured recall here; frees storage
```

CPU-only alternative: `embedding_backend: fastembed`, `fastembed_model: snowflake/snowflake-arctic-embed-l`, `embedding_prefix_policy: auto`.

These numbers are from one synthetic set, so treat small gaps as noise and re-run `tests/eval.py` on your own data before committing to a change.

The autotune loop uses the same harness:

```bash
python -m memory_eval.autotune \
    --max-experiments 200 \
    --max-minutes 480 \
    --db ~/.hermes/memory_store.db \
    --repo-root /absolute/path/to/enfold
```

It writes `experiments.jsonl`, `summary.json`, and `RECOMMENDATION.md` under `~/.hermes/reports/memory-eval/autotune-*`. It is proposal-only: each trial runs on a fresh SQLite backup, and the live DB/config are not changed.

## Configuration

Under `plugins.hermes-memory-store` in `config.yaml`. Every base holographic key still applies, plus these:

```yaml
plugins:
  hermes-memory-store:
    embedding_backend: fastembed            # "fastembed" (local CPU, default) or "ollama"
    embedding_weight: 0.3                   # weight of dense similarity in the hybrid score
    embedding_prefix_policy: none           # "none" (default) or "auto": apply the model's documented query/passage prefixes
    # fts_weight / jaccard_weight / hrr_weight default 0.3 / 0.2 / 0.2 and are
    # rescaled to sum to 1.0; set hrr_weight: 0 to disable the HRR signal.
    dedup_on_add: true                      # write-time near-duplicate gate
    dedup_jaccard: 0.9                      # token-overlap threshold for the lexical gate
    dedup_cosine: 0.92                      # cosine threshold for the semantic gate
    retrieval_decision_enabled: false       # optional calibrated final-stage filter/abstention gates
    # retrieval_decision_min_score: 0.5      # drop candidates below this final score when enabled
    # retrieval_decision_min_margin: 0.02    # abstain when top-2 filtered scores are too close
    # retrieval_decision_min_trust: 0.5      # drop candidates below this trust when enabled
    entity_boost_weight: 0.0                # additive boost for facts linked to a query-mentioned entity (default off)
    entity_expansion: false                 # 1-hop expansion to facts sharing an entity with a top hit (default off)
    entity_hub_degree_limit: 25             # entities linked to more facts than this are excluded from expansion
    reflection_enabled: false               # sleep-time reflection insights (default off)
    reflection_interval_hours: 24
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

Embeddings are **identity-versioned** (`backend:model:role:vN`), so switching models never corrupts existing vectors: each model's vectors live under their own identity, and the new model's are backfilled in the background. On startup, any fact missing a current-identity embedding is backfilled in a non-blocking background thread (in batches), and the worker re-runs the backfill periodically so transient embedding failures heal on their own. Vectors from a superseded model are kept until you reclaim them: call `provider.vacuum_embeddings()` to drop them without re-embedding, or `rebuild_embeddings()`, which prunes them after re-embedding.

## Tools

As a Hermes provider it inherits the base `fact_store` tool (add / search / probe / related / reason / contradict / update / remove / list) and `fact_feedback` (rate facts helpful/unhelpful to train trust). The `search` action is overridden to merge dense-embedding similarity into the ranked results.

## MCP server

`enfold/mcp_server.py` exposes the fact store over the Model Context Protocol (stdio transport). It registers `memory_search`, `memory_add`, `memory_supersede`, `memory_explain`, and `memory_history`. `--read-only` registers only search, explain, and history (the two writes are never registered at all in this mode, not just blocked at call time).

The `mcp` package (FastMCP) is an optional dependency, only needed to run this server, never to import `enfold` as a Hermes plugin.

Run the server **by file path**, not with `-m`:

```bash
python3 enfold/mcp_server.py \
    --db-path ~/.hermes/memory_store.db \
    --ollama-url http://localhost:11434 \
    --ollama-model embeddinggemma:latest \
    --hrr-dim 256
```

`python -m enfold.mcp_server` (or anything that imports `enfold` as a package before this module resolves its own parent Hermes checkout) triggers `enfold/__init__.py`'s unconditional `plugins.memory.holographic` import at module load time. On a host with a second, unrelated Hermes install already on `sys.path`, that import can win silently and this module never gets to point at the checkout you actually meant (`ENFOLD_HERMES_SRC`, see `mcp_provider.py`). Running the file directly sidesteps `__init__.py` entirely, which is why every example here uses the file path.

When sharing a store with a live Hermes gateway, `--hrr-dim` must match the `hrr_dim` the gateway is configured with. The standalone quickstart uses `--hrr-dim 256`; do not reuse that value for a shared gateway store unless the gateway also uses 256. There is no auto-detection: a mismatched dimension does not error at startup, it crashes later, the first time HRR scoring runs against a vector encoded at the other dimension.

Every write must carry a `source` tag, one of `claude-code`, `codex`, or `other`; `memory_add` and `memory_supersede` reject a call without one. The tag is appended to the fact's `tags` as `source:<agent>`, so it's visible alongside the fact everywhere tags already show up. Writes from the MCP server go through the exact same dedup and value-update-supersession gates as a live Hermes write: a near-duplicate is rejected and the existing fact's id returned instead of storing again, and a genuine value update (same wording, a changed number/id/state word) supersedes the prior fact automatically.

Concurrency: the store runs in SQLite WAL mode with `busy_timeout` set on the connection, and an Enfold write is several separate short transactions (dedup search, insert, bank rebuild, optional supersession), not one, so `busy_timeout` alone cannot make that whole sequence atomic across two processes. Every MCP write additionally takes an OS advisory file lock (`flock` on a `<db_path>.mcp-write.lock` sidecar) for its whole duration, so writes from separate MCP server processes (e.g. one for Claude Code, one for Codex) are fully serialized against each other rather than racing.

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

The dense-embedding layer roughly doubles recall@1 over keyword (0.38 to 0.81) on paraphrased queries, the exact case where keyword search fails. `*` The `blend` row is an *illustrative* keyword+embedding mix for this isolated comparison; it is **not** the formula the plugin ships. The real scorer adds the trust-weighted embedding onto the trust-weighted holographic score across FTS+Jaccard+HRR (see `_blend_score`). The keyword and HRR signals earn their weight on literal and compositional queries, which this benchmark intentionally underweights. It is a synthetic, illustrative benchmark, not a production guarantee, but every number is reproducible from the script.

There is also a larger harness under `memory_eval/` for end-to-end sweeps against a real database snapshot.

## Requirements

- Python 3 (developed and tested on 3.13) and `numpy`.
- For the MCP server: `pip install mcp`, plus one embedding backend (`fastembed` for CPU-only, or a running Ollama for GPU).
- For the Hermes provider: a working Hermes Agent install.

## Operations

By default, Hermes stores Enfold data in `~/.hermes/memory_store.db`. MCP users can choose any file with `--db-path`.

SQLite may create `-wal` and `-shm` sidecars next to the database, and MCP writes use a `.mcp-write.lock` sidecar. For backups, copy the database after a checkpoint or while the server is idle so WAL contents are not missed.

To uninstall MCP use, remove the MCP registration and delete the database file if you no longer need the facts. To uninstall Hermes use, remove the copied plugin directory from `~/.hermes/plugins/` and reset `memory.provider`.

Troubleshooting:

- Missing `mcp` package: install with `pip install -e .[mcp]` or `pip install mcp`.
- FastEmbed download failure: retry with network access or switch to an Ollama backend.
- Ollama down: recall still runs, but dense embedding recall is degraded.
- `hrr-dim` mismatch crash: restart with the same `--hrr-dim` as the Hermes gateway.
- Explicit `--hermes-src` that fails now exits instead of silently falling back.

## Credits

Built on the **holographic** memory provider by **dusterbloom** (NousResearch/hermes-agent, PR #2351), which implements **Holographic Reduced Representations**: Tony A. Plate, *"Holographic Reduced Representations,"* IEEE Transactions on Neural Networks 6(3):623-641, 1995. Full attribution in [NOTICE](NOTICE).

The dense-embedding retrieval, LLM fact extraction, write gates, temporal supersession, reflection, MCP server, and hybrid scoring are this project's contribution.

## License

MIT, see [LICENSE](LICENSE).

## Release notes

### 0.6.0

- Stdio MCP server (`enfold/mcp_server.py` + `mcp_provider.py`) shares
  the fact store with other coding agents (Claude Code, Codex CLI) over the
  Model Context Protocol, while the Hermes gateway keeps using it in-process.
  Optional `mcp` dependency, source-tagged writes through the same dedup and
  supersession gates, `--read-only` mode, and cross-process write locking on
  top of WAL + `busy_timeout`. See MCP server above.

### 0.5.1

- Graceful extraction-queue shutdown: when the gateway restarts mid-extraction,
  the in-flight row is now left pending with its attempt count untouched instead
  of burning a retry and logging a "bad file descriptor" error. The next worker
  drains it cleanly on startup. Fixes the recurring restart-time warning.

### 0.5.0

- Instruction-prefix support: `embedding_prefix_policy: auto` applies each model's
  documented query/passage prefixes (bge, e5, nomic, arctic, embeddinggemma,
  qwen3), with optional `embedding_query_prefix` / `embedding_document_prefix`
  overrides. The policy is part of the embedding identity, so switching it
  re-embeds in the background and never corrupts existing vectors.
- Configurable holographic weights: `fts_weight` / `jaccard_weight` /
  `hrr_weight`, rescaled to sum to 1.0. Set `hrr_weight: 0` to disable the HRR
  signal, which the benchmark shows contributes no recall on semantic queries.
- Added a benchmark-backed recommended configuration (see above).

### 0.4.0

- Reclaim vectors left behind by superseded embedding models. `provider.vacuum_embeddings()` drops every vector whose identity is not the current model's (optionally keeping a canary model), without re-embedding; facts left bare are healed by the background backfill. `rebuild_embeddings()` now prunes superseded vectors after re-embedding by default (`prune_stale=False` keeps them). `EmbedStore` gains `identity_counts()` and `prune_identities(keep)`.
- Fixes unbounded growth of the `fact_embeddings` table across model swaps: each swap previously left the old model's vectors behind indefinitely.

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
