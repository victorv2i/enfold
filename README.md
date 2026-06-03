# holographic_plus

A hybrid long-term memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

It extends the bundled **holographic** (HRR) fact store with **dense semantic embeddings** and **LLM-based fact extraction**, then merges every signal in a single hybrid retrieval pass — so the agent recalls facts by *meaning*, not just keyword overlap, while keeping the holographic store's symbolic strengths.

It does not replace the holographic provider — it subclasses it. All the holographic foundation (the SQLite fact store, FTS search, trust scoring, entity resolution, and HRR compositional retrieval) is dusterbloom's work; this plugin adds a layer on top. See [NOTICE](NOTICE).

## What it adds over base `holographic`

- **Dense embedding retrieval.** Each fact is embedded into a sidecar `fact_embeddings` table; queries are scored by cosine similarity. This catches paraphrases and fuzzy matches that pure keyword/FTS search misses.
- **Hybrid scoring.** Four signals merged per query — FTS keyword, Jaccard overlap, HRR compositional, and dense embedding — with the remaining budget going to trust weighting. Default split: FTS 0.3, Jaccard 0.2, HRR 0.2, embedding 0.3.
- **LLM fact extraction.** At session end (and just before context compression) it extracts durable, atomic facts from the conversation so nothing important is lost when the window rolls over. Extraction uses your agent's own model by default — no hardcoded provider — and is configurable.
- **Graceful fallback.** If the embedding backend is unreachable, it silently falls back to holographic-only scoring with the embedding weight redistributed — never a hard failure.

## Requirements

- A working Hermes Agent install (this is a plugin for it, and it subclasses the bundled `holographic` provider).
- `numpy`.
- One embedding backend:
  - **FastEmbed** — local, CPU-only, recommended for privacy. `pip install fastembed` (default model `BAAI/bge-base-en-v1.5`, 768-dim).
  - **Ollama** — point `ollama_url` at a running server and pick an embedding `ollama_model`.

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

Under `plugins.hermes-memory-store` in `config.yaml` — every base holographic key still applies, plus these:

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

Embeddings are **identity-versioned** (`backend:model:role:vN`), so switching models never corrupts existing vectors — stale ones are simply re-embedded on the next backfill. On startup, any fact missing an embedding is backfilled in a non-blocking background thread (in batches).

## Tools

It inherits the base `fact_store` tool (add / search / probe / related / reason / contradict / update / remove / list) and `fact_feedback` (rate facts helpful/unhelpful to train trust). The `search` action is overridden to merge dense-embedding similarity into the ranked results.

## How it works

- `prefetch(query)` runs the hybrid search and injects the top matches into the agent's context each turn.
- `handle_tool_call` / `on_memory_write` keep the embedding sidecar in sync on every add/update/remove.
- `on_session_end` runs the base regex extraction, then an LLM pass for the facts regex misses; `on_pre_compress` does a tight extraction before the context window is trimmed.
- A shared, write-invalidated cache holds the normalized embedding matrix so repeated queries within a session stay fast.

## Evaluation

`tests/eval.py` is a self-contained recall benchmark: it seeds a synthetic corpus into the real `EmbedStore`, runs paraphrased queries (deliberately low keyword overlap with their target), and reports recall@k for a keyword baseline vs the dense-embedding retrieval this plugin adds. Reproduce it with `python tests/eval.py`.

On the bundled 24-fact / 16-query paraphrase set:

```
retriever     recall@1  recall@3  recall@5     MRR
--------------------------------------------------
keyword           0.38      0.56      0.62    0.51
embedding         0.81      0.88      0.88    0.86
hybrid            0.81      0.88      0.88    0.86
```

The dense-embedding layer roughly doubles recall@1 over keyword (0.38 → 0.81) on paraphrased queries — the exact case where keyword search fails. On this paraphrase-heavy set the hybrid matches pure embedding (keyword adds no extra lift here); the keyword and HRR signals earn their weight on literal and compositional queries, which this benchmark intentionally underweights. It is a synthetic, illustrative benchmark, not a production guarantee — but every number is reproducible from the script.

## Credits

Built on the **holographic** memory provider by **dusterbloom** (NousResearch/hermes-agent, PR #2351), which implements **Holographic Reduced Representations** — Tony A. Plate, *"Holographic Reduced Representations,"* IEEE Transactions on Neural Networks 6(3):623-641, 1995. Full attribution in [NOTICE](NOTICE).

The dense-embedding retrieval, LLM fact extraction, and hybrid scoring are this project's contribution.

## License

MIT — see [LICENSE](LICENSE).
