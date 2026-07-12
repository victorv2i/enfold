<p align="center">
  <img src="assets/logo.png" alt="Enfold logo" width="280">
</p>

<h1 align="center">Enfold</h1>

<p align="center"><strong>Local SQLite memory with provenance, current-state rules, and a small daemon.</strong></p>

Enfold is a local-first memory service for durable facts. It stores evidence and
history in SQLite, keeps current state separate from superseded and conflicted
facts, and exposes a Unix-socket daemon with an optional stdio MCP bridge.
There is no hosted service.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[mcp,sqlite-vec]'
```

`sqlite-vec` is optional and pinned to `0.1.9`. The canonical
`fact_embeddings` table remains the source of truth. If the extension, its
index, or its metadata is unavailable, retrieval automatically uses brute-force
cosine scoring instead. Rebuild an index only in a maintenance window:

```bash
python -m enfold.ops rebuild-vector-index /absolute/path/to/memory.db \
  --embedding-identity 'provider:model:document:policy:version' \
  --dimensions 768
```

## Run

Create a user-owned JSON configuration with explicit database, socket,
retrieval, and client grants. `ci` retrieval is deterministic plumbing only,
not a semantic embedding model. See [server deployment](docs/SERVER_DEPLOYMENT.md)
for the full configuration contract.

```bash
enfold-server --config /absolute/path/to/server.json check
enfold-server --config /absolute/path/to/server.json run
```

The optional MCP bridge connects to an already-running daemon and never opens
SQLite itself:

```bash
enfold-mcp-proxy --socket-path /absolute/private/enfold.sock \
  --client-id workstation-1 --surface local --agent-id worker-1 \
  --session-id session-1
```

## Retrieval and context

Candidates are authorized and filtered for current, settled facts before dense
ranking. The default ranking is:

```text
score = 0.90 × (0.35 × FTS + 0.25 × Jaccard + 0.40 × cosine)
      + 0.05 × trust + 0.02 × kind + 0.03 × recency
```

The kind prior is state `1.00`, insight `0.75`, untyped `0.50`, and event
`0.25`. Recency uses an exponential 365-day half-life. These values and the
confidence gates are `RankingConfig` defaults in `enfold/hybrid_retrieval.py`:
`score_floor: 0.12` rejects weak candidates and `ambiguity_margin: 0.005`
abstains when the top two results are too close. A named anchor in the query
also requires that anchor in a candidate.

`memory_context` produces a bounded, cited Markdown block. It estimates tokens
as Unicode characters divided by four, truncates individual facts to fit, omits
unsafe or duplicate state slots, and can use maximal marginal relevance (MMR)
to choose diverse context. MMR uses dense similarity when present and token
overlap otherwise.

## Writes and typed state

Writes are idempotent and carry provenance. Exact duplicates reuse the current
fact. For untyped near duplicates, the service finds an FTS-bounded semantic
candidate, writes the incoming observation, and merges the two by retaining one
surviving fact while superseding the other. This is not a claim that every
similar sentence is rejected.

Extraction accepts typed proposals with `state`, `preference`, `commitment`, or
`event` labels when confidence is at least `0.8`. Only `state` is routed into a
structured `(scope, subject_key, predicate_key)` state slot. A changed state can
supersede the prior slot value; competing current state is recorded as a
conflict. The other labels remain attributed facts with their extracted type in
metadata.

## MCP tools

The bridge provides `memory_write`, `memory_search`, `memory_context`,
`memory_evidence`, `memory_history`, `memory_conflicts`,
`memory_resolve_conflict`, and `memory_extraction_enqueue`.

It also provides change and entity views:

- `memory_changes(since, until)` lists created, superseded, and resolved facts
  in a half-open time window.
- `memory_timeline(subject_or_query)` returns chronological settled events.
- `memory_entities()` ranks visible entities from current subjects and tags.
- `memory_entity(name)` returns that entity's current facts, recent changes,
  and open conflicts.

## Operations

Use SQLite's backup API rather than copying a live database file:

```bash
python -m enfold.ops backup SOURCE.sqlite BACKUP.sqlite \
  --secondary-directory /mounted/offsite \
  --age-recipient-path /secure/recipients.txt
```

The primary backup is verified. A secondary destination is best effort: failure
does not invalidate a completed primary backup. When `age` is available,
Enfold uses `age -R` and places only an encrypted `.age` artifact in the
secondary destination. Without `age`, it warns and makes a private plain copy.
Keep identities and recipient files outside the repository and server config.

Rehearse the newest `*.sqlite` backup without changing the live database:

```bash
python -m enfold.backup_rehearsal LIVE_DB BACKUP_DIR STATE_DIR \
  --fact-count-tolerance 100
```

The rehearsal restores to a temporary directory, runs `quick_check`, compares
fact counts, writes a dated JSON pass/fail report, and exits nonzero on failure.

For a read-only local browser, configure `browse_scopes`, then create a filtered
snapshot. It includes only current, settled, normal-sensitivity facts in those
scopes and a small `metadata.json` for Datasette. Serve the resulting immutable
SQLite file with a local Datasette installation:

```bash
python -m enfold.ops browse-snapshot /absolute/path/to/server.json
```

Regenerate the snapshot when browser-visible facts change. It does not serve or
modify the live database.

## Evaluation

The public Arena is a synthetic regression harness. It does not read a live
store or measure production embedding quality:

```bash
python -m memory_eval.public_arena --provider core-fts-current --limit 5
```

The personal Arena harness is public, but its corpus remains private by design.
Keep real cases and reports outside the repository, then run:

```bash
python -m memory_eval.personal_arena \
  --cases ~/.config/enfold/private-arena/cases-v0.jsonl \
  --db ~/.hermes/memory_store.db --seed 0
```

Each JSONL line has `id`, `query`, and `category`. Add either or both of
`expected_fact_ids` and `expected_content_regexes` for an answerable case; omit
both for an abstention case. `forbidden_content_regexes` identifies text that
must not appear in the top three, and `asof` is an optional human-auditable
time annotation. See the synthetic
[format sample](memory_eval/fixtures/personal_arena_sample.jsonl).

## License

MIT, see [LICENSE](LICENSE).
