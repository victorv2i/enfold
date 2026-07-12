# MemoryArena fixtures

`personal_arena.py` is the private, offline retrieval benchmark. Run it with:

```bash
python -m memory_eval.personal_arena \
  --cases ~/.config/enfold/private-arena/cases-v0.jsonl \
  --db ~/.hermes/memory_store.db --seed 0
```

It opens the live database read-only and makes a SQLite backup in a temporary
directory before every run. Retrieval is then `HybridRetriever` with the
deterministic feature-hash embedder: active/current/scope/trust filters, FTS,
Jaccard, the shipped hybrid weights, and ranking all run for real. It does not
measure production embedding-model quality, stored-vector coverage, or MCP
transport.

Each private JSONL row has `id`, `query`, and `category`; it has either
`expected_fact_ids` and/or `expected_content_regexes`, or neither for an
abstention case. `forbidden_content_regexes` marks text that must not appear in
the top three and contributes to stale-leak rate. `asof` is a human-auditable
temporal annotation; current-truth filtering comes from the snapshot schema.
Expected ids/regexes are alternatives, so a case passes recall when any one
matches. The top result is considered a confident answer at score >= 0.35
(configurable with `--abstention-min-score`).

The repository contains only `fixtures/personal_arena_sample.jsonl`, a
synthetic format sample. It intentionally does not match any real database.
Keep real cases, results, and source facts under
`~/.config/enfold/private-arena/`; do not add them to git. Add a case only
after checking its expected fact against a read-only snapshot, choose a stable
id, and record a precise stale regex whenever the question has a known prior
answer.
