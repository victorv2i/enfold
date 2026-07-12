# Architecture Review Synthesis

Date: 2026-07-11
Status: Approved direction with required revisions

Three read-only council seats reviewed the Enfold repository and draft 1.0
specification: architecture, product/evaluation/privacy, and implementation
safety/migration.

## Consensus

- Ship provenance and transactional writes before advanced reasoning.
- Introduce schema/version gates before new DDL reaches a shared store.
- Make state updates exact-slot operations rather than retrieval guesses.
- Store conflicts durably and expose them instead of silently ranking one.
- Preserve human corrections and distinguish authority from freshness.
- Enforce minimal scopes before retrieval.
- Make Enfold standalone; Hermes becomes an adapter.
- Build public temporal/provenance evaluations and a private Memory Arena.
- Defer profiles, LLM synthesis, HTTP, ANN, and feature-parity work.

## Resolved disagreement: daemon timing

The product seat recommended deferring the daemon for a one-maintainer,
single-machine project. The architecture and safety seats recommended retaining
it because the current system already has multiple writer processes,
multi-transaction writes, deployment drift, and no central migration owner.

Decision: retain the daemon as the 1.0 convergence point, but first implement
the write-service contract in-process. The daemon changes transport and
ownership only after semantics, idempotency, provenance, and versioning are
tested.

## Required gates

1. Compatibility/version release reaches every current writer.
2. Backup and restore commands are tested before live schema changes.
3. Provenance, write log, supersession, and fact insertion are one transaction.
4. Idempotency is scoped by client and covers extraction and supersession.
5. Public synthetic Arena fixtures cover updates, conflicts, abstention, and
   stale leakage.
6. Live activation occurs only in a quiet window after a restore rehearsal.

## Maintenance-window boundary

Repository development, temporary databases, snapshots, and side-by-side
binary installation do not require other sessions to stop. A quiet window
begins immediately before the authoritative pre-migration backup. All Hermes
and MCP writers must then stop, registrations and plugin revisions are swapped,
the database is migrated once, and clients reconnect one at a time.
