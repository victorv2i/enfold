# Enfold Shared Second Brain Specification

Status: Draft for council review
Target: Enfold 1.0
Compatibility baseline: Enfold 0.7.x

## 1. Purpose

Enfold is a local-first memory service shared by multiple AI-agent surfaces.
Its primary deployment target is one machine running Hermes and other MCP
clients against one canonical memory store.

Enfold must provide:

- fast, relevant recall on every agent turn;
- reliable writes from every participating agent;
- immutable attribution and evidence for each memory;
- current-state tracking without losing history;
- grounded connections across people, projects, organizations, and events;
- human-readable explanations and correction paths;
- safe handling of personal and work information;
- portable installation without machine-specific assumptions;
- reproducible evaluation of recall, freshness, provenance, and latency.

Enfold is not merely a vector database. It is a temporal, provenance-aware
memory service whose derived knowledge always remains traceable to evidence.

## 2. Design principles

1. **One writer, many clients.** One daemon owns schema migration and durable
   writes. Hermes and MCP clients use adapters rather than opening SQLite for
   writes independently.
2. **Evidence before inference.** Observations are retained before facts or
   insights are derived from them.
3. **Append history, project current truth.** Events and observations are
   append-only. Current facts are projections with explicit supersession.
4. **Identity is connection metadata.** A caller cannot establish provenance
   merely by supplying a source value in tool arguments.
5. **Writer is not actor.** The agent recording a memory may differ from the
   person or agent who performed the work, made the claim, or verified it.
6. **Confidence is not freshness.** Epistemic confidence, source authority,
   current applicability, retrieval relevance, and usefulness are separate.
7. **Fast and deep recall are different products.** Turn prefetch must remain
   small and predictable; deliberate research may perform graph traversal and
   synthesis.
8. **Derived knowledge carries receipts.** A fact or insight without evidence
   is either explicitly user-authored or marked unverified.
9. **Corrections outrank automation.** Human correction is durable, auditable,
   and protected from being silently overwritten by extraction.
10. **Local-first is the default, not a limitation.** Network services and
    hosted models are optional adapters.
11. **Attribution is not authentication.** Enfold 1.0 assumes cooperative
    clients under one local OS account. Connection identity prevents accidental
    misattribution; it is not a boundary against another same-user process.
12. **No model on synchronous writes.** Writes classify deterministically or
    heuristically, then enqueue optional model refinement.

## 3. Process architecture

```text
Hermes native adapter ----------+
MCP client A stdio proxy -------+--> Enfold daemon --> SQLite + indexes
MCP client B stdio proxy -------+
Other MCP/SDK clients ----------+
```

The daemon is a standalone Enfold package with no Hermes imports. Enfold owns
its storage engine; the Hermes provider is an adapter that depends on Enfold,
never the reverse.

The daemon listens on a local Unix socket by default. A loopback HTTP transport
may be added later for platforms that cannot use Unix sockets. Remote network
exposure is out of scope for 1.0.

Each adapter establishes immutable connection context:

- `client_id`: stable installation identity;
- `surface`: `hermes`, `mcp-client-a`, `mcp-client-b`, `mcp`, or another registered
  surface;
- `agent_id`: `wonny`, a Hermes cron/delegate identifier, or client agent;
- `session_id`: conversation or task identifier;
- `parent_agent_id`: optional delegation parent;
- `project_root`, `repository`, `branch`, and `commit`: optional development
  context;
- `capabilities`: negotiated API and schema features;
- `access_scopes`: scopes the connection may read or write.

The MCP proxy adds this context server-side. Memory-write tools do not accept
the caller identity as authoritative user input.

## 4. Memory layers

### 4.1 Observations

An observation is source material Enfold received: a conversation excerpt,
agent report, file observation, tool result summary, human correction, imported
document passage, or external event.

Observations are append-only. Redaction may replace protected content with a
tombstone while retaining the audit record.

### 4.2 Facts

A fact is one atomic claim derived from or directly supplied by one or more
observations. Facts remain text-first for portability and inspectability, with
optional structured identity for temporal maintenance.

Memory kinds:

- `state`: a current value that may change;
- `preference`: a durable but revisable choice;
- `decision`: a dated choice and its rationale;
- `event`: something that happened and should coexist with other events;
- `relationship`: a typed connection between entities;
- `reference`: a path, procedure, identifier, or durable lookup;
- `instruction`: a standing behavior or safety rule;
- `insight`: a derived multi-source conclusion;
- `note`: unclassified durable information retained without stronger claims.

### 4.3 Current-state slots

State-like facts may define:

- `subject_key`, such as `cron:bdad4c3a47d5`;
- `predicate_key`, such as `model`;
- `object_value` or `object_entity_id`;
- `valid_from`, `invalid_at`, and `superseded_by`.

Scope is part of strict-slot identity: `(scope, subject_key, predicate_key)`.
This prevents a client authorized for one compartment from learning about or
mutating a slot in another. A reader authorized for multiple scopes may still
surface cross-scope disagreement explicitly.

At most one non-conflicted fact is current for a strict state slot. A new value
either supersedes the previous fact or creates a visible conflict. Events never
supersede merely because they share a subject.

### 4.4 Entities and relationships

Entities represent people, agents, projects, organizations, places, systems,
and concepts. Aliases are first-class. Relationships are facts with temporal
validity and provenance; the graph is a secondary view over facts, not an
independent source of truth.

### 4.5 Profiles and insights

Profiles are bounded materialized views assembled from pinned identity facts,
current state, preferences, relationships, and recent decisions. Insights are
derived facts citing at least two active source facts or observations.

Profiles and insights are invalidated or rebuilt when their dependencies
change. They never erase their supporting evidence.

## 5. Provenance model

Every durable write records separate roles:

- `recorded_by`: connection/agent that submitted the write;
- `asserted_by`: person or agent that made the underlying claim;
- `performed_by`: actor that performed a reported action;
- `source_type`: conversation, file, commit, test run, import, correction,
  reflection, or manual entry;
- `source_uri`: stable local or external reference where appropriate;
- `observed_at`: when the evidence was observed;
- `recorded_at`: when Enfold received it;
- `extractor`: extraction implementation and model identity;
- `scope`: access and sensitivity classification.

An agent report is not automatically verified merely because it was recorded
by a client or Hermes. Later verification is a new observation connected
with a `verifies` provenance relation; correction uses `corrects`.

## 6. Proposed additive schema

Existing `facts` and `fact_embeddings` remain readable throughout migration.
The first implementation uses additive tables and columns.

```sql
CREATE TABLE memory_clients (
    client_id TEXT PRIMARY KEY,
    surface TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    display_name TEXT,
    created_at TEXT NOT NULL,
    disabled_at TEXT
);

CREATE TABLE memory_sessions (
    session_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    parent_agent_id TEXT,
    project_root TEXT,
    repository TEXT,
    branch TEXT,
    commit_sha TEXT,
    started_at TEXT,
    ended_at TEXT,
    PRIMARY KEY (client_id, session_id),
    FOREIGN KEY (client_id) REFERENCES memory_clients(client_id)
);

CREATE TABLE observations (
    observation_id INTEGER PRIMARY KEY,
    client_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_uri TEXT,
    content TEXT,
    content_sha256 TEXT NOT NULL,
    asserted_by TEXT,
    performed_by TEXT,
    observed_at TEXT,
    recorded_at TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'private',
    sensitivity TEXT NOT NULL DEFAULT 'normal',
    redacted_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (client_id, content_sha256, session_id, source_type)
);

CREATE TABLE fact_provenance (
    fact_id INTEGER NOT NULL,
    observation_id INTEGER NOT NULL,
    relation TEXT NOT NULL DEFAULT 'supports',
    evidence_excerpt TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (fact_id, observation_id, relation),
    FOREIGN KEY (fact_id) REFERENCES facts(fact_id),
    FOREIGN KEY (observation_id) REFERENCES observations(observation_id)
);

CREATE TABLE memory_write_log (
    write_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL,
    client_id TEXT NOT NULL,
    session_id TEXT,
    operation TEXT NOT NULL,
    outcome TEXT NOT NULL,
    fact_id INTEGER,
    existing_fact_id INTEGER,
    recorded_at TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (client_id, idempotency_key)
);

CREATE TABLE fact_conflicts (
    conflict_id TEXT PRIMARY KEY,
    subject_key TEXT NOT NULL,
    predicate_key TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution_fact_id INTEGER,
    detail_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE fact_conflict_members (
    conflict_id TEXT NOT NULL,
    fact_id INTEGER NOT NULL,
    PRIMARY KEY (conflict_id, fact_id)
);

CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE enfold_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

Additive `facts` columns:

```text
memory_kind, subject_key, predicate_key, object_value,
object_entity_id, confidence, source_authority, scope,
sensitivity, correction_status, schema_version
conflict_group
```

`enfold_meta` stores schema/protocol compatibility, active embedding identity,
and HRR geometry. Openers fail closed on unknown newer schemas or incompatible
geometry.

For strict state facts, a partial unique index enforces one non-conflicted
current value per `(scope, subject_key, predicate_key)`. Conflict members are
excluded from settled-truth retrieval but remain visible through the conflict
API until explicitly resolved.

The existing global `facts.content UNIQUE` constraint prevents legitimate
value flip-flops such as X → Y → X. Removing it requires a rehearsed table
rebuild during controlled activation, not an opportunistic startup migration.

Exact migrations may split scope and identity tables further after council
review. JSON is metadata only; fields needed for correctness or filtering use
typed columns.

## 7. Write protocol

All durable mutations use one daemon transaction coordinator and an
idempotency key.

1. Authenticate connection context and intersect requested scopes with a
   server-side grant policy keyed by stable client identity. Handshake scope
   claims are never grants by themselves.
2. Validate and redact prohibited content such as credentials.
3. Append or resolve the source observation.
4. Normalize candidate facts into atomic claims using deterministic rules.
5. Classify kind and optional state slot heuristically; enqueue optional model
   refinement rather than blocking the write.
6. For typed state, query the exact active
   `(scope, subject_key, predicate_key)` slot.
   For untyped text, search for duplicates, updates, and contradictions without
   excluding low-trust facts.
7. Produce a write decision: `added`, `deduped`, `superseded`, `conflicted`,
   `rejected`, or `needs_review`.
8. Insert fact, provenance edges, temporal changes, and write log atomically.
9. Enqueue embeddings, entity resolution, profile refresh, and optional
   reflection after commit.
10. Return fact identity, decision, provenance summary, and warnings.

If optional post-commit work fails, the canonical write remains valid and the
queue retries idempotently.

## 8. Read protocols

### Fast lane

`memory_context(query, scope, token_budget)` returns a compact, diversified
set of current facts suitable for turn prefetch. It must support abstention and
must not inject unresolved conflicts as settled truth.

Target: p95 below 150 ms with local dense embeddings at 10,000 active facts.

### Search and evidence

- `memory_search(query, filters, limit)`
- `memory_explain(query, limit)`
- `memory_evidence(fact_id)`
- `memory_history(fact_id)`

### Structured views

- `memory_profile(entity, scope)`
- `memory_timeline(entity, from, to)`
- `memory_connections(entity_or_query, depth, limit)`
- `memory_conflicts(filters)`

### Deep lane

`memory_research(query, scope, budget)` may retrieve observations, documents,
relationships, and facts before producing a cited synthesis. It is explicit,
asynchronous-capable, rate-limited, and never part of automatic turn prefetch.

## 9. Corrections

`memory_correct` records a new correction observation and creates an audited
replacement or invalidation. Human corrections receive the highest default
source authority but remain attributable and reversible.

Automation cannot silently supersede a protected human correction or a fact
with higher source authority. It records `needs_review` or opens a visible
conflict instead.

Deleting a fact is reserved for privacy or legal erasure. Ordinary correction
uses invalidation and history.

## 10. Privacy and access

Initial scopes:

- `private`: general personal memory;
- `work`: work-related information;
- `project:<id>`: project-specific context;
- `public`: safe for broadly configured clients;
- `sensitive`: explicit retrieval only;
- `secret`: rejected from durable memory by default.

Scope is enforced by the daemon before candidate retrieval. Post-filtering is
insufficient because unauthorized facts must not enter ranking or synthesis.
FTS candidate SQL includes allowed scopes, while dense scoring masks
unauthorized fact ids before top-k selection.

No non-private write path may be enabled until every active read surface,
including history, evidence, dedup lookup, and dense candidate generation,
enforces effective server-granted scopes. Dedup never searches outside those
effective scopes and cannot become a cross-scope existence oracle.

Exports and public test fixtures exclude personal data. Logs avoid raw content
unless diagnostic logging is explicitly enabled.

Observation retention is configurable. Long-lived evidence retains hashes,
source references, and minimal excerpts; raw conversation content may be
compacted or redacted after a policy-defined interval. Provenance and audit
outcomes remain after compaction.

Privacy erasure covers every materialized copy: observation content and
excerpts, fact content, FTS rows, HRR vectors, dense embeddings, provenance
excerpts, queued extraction/reflection payloads, and raw-content diagnostic
artifacts. Audit rows retain only non-reversible identifiers and the erasure
event. Erasure is a maintenance transaction followed by index verification.

## 11. Compatibility

The 0.7 MCP tools remain available during the transition. Legacy
`source:<agent>` tags are imported as low-resolution provenance and preserved.
Legacy clients can read and write through a compatibility proxy, but new
structured provenance fields require protocol capability negotiation.

The daemon publishes:

- service version;
- schema version;
- protocol version;
- enabled optional capabilities;
- active embedding identity;
- migration and queue health.

Adapters refuse unsafe major-version mismatches rather than silently falling
back to direct SQLite writes.

When the daemon is unavailable, adapters may offer degraded direct read-only
access. Writes fail explicitly. Adapters never spool writes locally because a
second queue would become an unordered second source of truth.

## 12. Explicit non-goals for 1.0

- remote multi-tenant SaaS operation;
- a general graph database;
- autonomous psychological profiling by default;
- mandatory LLM inference on the read hot path;
- automatic ingestion of arbitrary tool output;
- storing credentials or secret material;
- replacing the human-readable Vault;
- distributed consensus across machines;
- ANN indexing before measured scale requires it.
- authentication boundaries between processes owned by the same OS user;
- adapter-side write spooling;
- synchronous model calls during writes.

## 13. Success criteria

Enfold 1.0 is ready when:

- Hermes and multiple MCP clients write through the same service;
- every new fact has immutable client/session provenance;
- duplicate concurrent writes are idempotent;
- state updates do not leave obsolete active truth in benchmark cases;
- conflicts are visible and never silently presented as settled;
- evidence and history are inspectable from every client;
- scoped facts cannot leak into unauthorized retrieval;
- fast-lane latency meets its p95 target;
- migration and rollback are tested on realistic database snapshots;
- public and private evaluation gates pass.
