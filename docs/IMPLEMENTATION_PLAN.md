# Enfold 1.0 Implementation and Migration Plan

Status: Draft pending council approval

## Safety contract

Until the activation phase, all development uses repository fixtures and
SQLite backups outside `~/.hermes`. Development must not:

- restart Hermes or its gateway;
- modify active MCP registrations;
- replace the installed Hermes plugin;
- migrate, checkpoint, vacuum, or write to the live memory database;
- start a second production daemon on the live database;
- run heavyweight embedding sweeps while other interactive agents rely on
  Ollama.

## Phase 0: Freeze the contract

Deliverables:

- council-approved `SECOND_BRAIN_SPEC.md`;
- protocol and schema versioning policy;
- threat model for local clients and sensitive scopes;
- representative migration snapshot inventory;
- benchmark case taxonomy.

Exit gate: no unresolved disagreement about ownership of writes, provenance
roles, temporal semantics, or compatibility guarantees.

## Phase 0.5: Version and migration gate

Before introducing new schema, ship a compatibility release to every current
writer with:

- explicit schema and protocol version checks;
- `schema_migrations` and `enfold_meta` ownership;
- fail-closed behavior on unknown newer schemas;
- an explicit `enfold migrate` command;
- no automatic new-schema migration during provider initialization;
- backup and verified restore commands.

Exit gate: all installed writers understand the version gate before Phase 1
DDL can reach the live database.

## Phase 1: Provenance foundation in-process

Implement additive schemas and a `MemoryWriteService` behind the current
provider and MCP server, without introducing a daemon yet.

Deliverables:

- migrations for clients, sessions, observations, provenance, and write log;
- structured connection context;
- server-side client scope grants; handshake scopes only narrow those grants;
- idempotent write envelope;
- compatibility mapping from existing MCP `source` values;
- evidence and write-log read APIs;
- one `BEGIN IMMEDIATE` transaction covering fact, provenance, supersession,
  and write-log state;
- per-client idempotency with replay of the original outcome;
- idempotent extraction enqueue by payload hash;
- backup verification using integrity, foreign-key, FTS, and row-count checks;
- tests for migrations, retries, rollback, and old database compatibility.

All tests use temporary databases or copied snapshots.

Exit gate: existing MCP behavior remains compatible and every new structured
write can be traced to an observation and client. Every installed legacy
writer refuses a schema it does not understand before v1 migration is allowed.

## Phase 2: Typed memory and temporal truth

Deliverables:

- memory-kind classifier contract;
- optional subject/predicate state slots;
- conflict records and conflict query;
- exact `(scope, subject, predicate)` slot-keyed supersession with a partial
  uniqueness invariant;
- separate confidence, authority, freshness, and usefulness fields;
- correction API;
- protected corrections and higher-authority supersession review;
- conflict listing shipped in the same slice that hides conflict members from
  settled-truth retrieval;
- initial pre-retrieval scopes: `private`, `work`, and secret rejection;
- regression cases for differently worded updates, negation, changed
  preferences, event coexistence, and false supersession.

Exit gate: the Morning Briefing and Wonny-model stale-truth cases select or
flag current truth correctly without regressing event history. No non-private
write is enabled until all search, history, evidence, dense, and dedup reads
enforce server-granted scopes. Privacy-erasure tests remove every materialized
content copy and rebuild affected indexes.

## Phase 3: Central daemon

Deliverables:

- single-writer Unix-socket daemon;
- standalone storage engine with no Hermes import or checkout dependency;
- health, version, capability, and queue endpoints;
- thin MCP stdio proxy;
- Hermes adapter using the same protocol;
- crash-safe post-commit job queue;
- daemon lifecycle CLI and systemd user unit template;
- graceful behavior when the daemon is unavailable.
- one shared protocol client library used by Hermes and MCP adapters;
- socket permissions and local peer-credential inspection;

Direct SQLite writes become an explicit offline maintenance mode, never an
automatic runtime fallback.

The daemon owns migrations and holds the exclusive database fence for its
lifetime. Adapters never migrate. Degraded direct access is read-only; writes
fail rather than spool.

Exit gate: concurrent adapter tests demonstrate idempotent writes, consistent
read-after-write, clean restart recovery, and no database corruption.

## Phase 4: Entities and retrieval-only deep recall

Deliverables:

- entity aliases and typed temporal relationships;
- timeline and connections APIs;
- deep retrieval returning evidence and relationships without mandatory LLM
  synthesis;
- retrieval diversification and contradiction-aware context assembly;
- strict scope enforcement before retrieval.

Exit gate: people/project connection cases improve without unsupported
inferences or privacy leakage.

Profile materialization and LLM synthesis remain deferred until evaluation
shows that on-demand assembly is insufficient.

## Phase 5: Evaluation and productization

Private evaluation:

- Victor Memory Arena with direct, paraphrase, state-update, preference,
  decision, relationship, timeline, contradiction, multi-hop, abstention, and
  privacy cases.

Public evaluation:

- sanitized temporal/provenance suite;
- selected LoCoMo, LongMemEval, or BEAM-compatible evaluation;
- concurrent-write and crash-recovery benchmark;
- latency and token-budget report;
- reproducible configuration and raw result artifacts.

Product deliverables:

- one-command local quickstart;
- packaged console entry points and elimination of the `python -m` import trap;
- no production fallback to test/fake Hermes modules;
- platform-neutral paths and configuration;
- import/export and backup commands;
- upgrade and rollback documentation;
- synthetic demo data;
- CI coverage across fresh, legacy, and migrated stores.

Exit gate: public claims are limited to reproduced evidence.

## Phase 6: Controlled live activation

This is the first phase requiring a quiet maintenance window.

Before activation:

1. Stop or disconnect Hermes and other memory-writing clients that can
   write Enfold memory.
2. Confirm no Enfold MCP process has the live database open for writes.
3. Create and verify a SQLite backup using the backup API.
4. Record current config, plugin revision, database checksum, schema, and
   embedding identities.
5. Run migration rehearsal and the full evaluation suite on a fresh copy,
   including a restore rehearsal back to the previous version.
6. Install version-matched daemon and adapters.
7. Migrate the live database once.
8. Start only the daemon, verify health and read-only smoke cases, then test a
   reversible write.
9. Reconnect Hermes and other clients one at a time.
10. Monitor queue health, write attribution, retrieval correctness, and SQLite
    integrity before declaring completion.

Rollback restores the previous plugin/config and verified database backup.
No downgrade process may open a newer schema unless compatibility is tested.

## Workstream boundaries

After council approval, implementation can be parallelized safely:

- Workstream A: schema, migrations, provenance, and write service;
- Workstream B: protocol, daemon, MCP proxy, and client identity;
- Workstream C: temporal slots, conflicts, correction, and evaluation cases;
- Integrator: architecture consistency, compatibility, tests, and release
  documentation.

Workstreams use separate branches or worktrees and never modify live state.

## Deferred until measured need

- profile materialization and dependency graphs;
- LLM-generated deep synthesis;
- HTTP or remote transports;
- ANN indexing;
- scopes beyond the initial private/work model;
- destructive entity merges;
- retrieval-scoring changes during migration.
