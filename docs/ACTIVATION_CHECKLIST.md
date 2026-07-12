# Enfold v1 activation checklist

Status date: YYYY-MM-DD

This is the authoritative release-state checklist for the provenance-first,
sole-writer v1 path. Historical review documents explain how the design
arrived here; this file is a reusable activation template.

## Evidence levels

- **Implemented**: the capability exists in this source tree.
- **Repository-verified**: checked-in tests exercise the capability without
  using live memory. This does not prove the operator's model, database, or
  service configuration.
- **Rehearsed on copy**: the exact deployment procedure has passed against an
  explicit SQLite backup copy, including restore.
- **Live-activated**: the v1 daemon and clients have been deliberately cut
  over and checked during a quiet maintenance window.

Passing a lower level never implies a higher one.

## Current state

| Capability or gate | Implemented | Repository-verified | Rehearsed on copy | Live-activated |
| --- | :---: | :---: | :---: | :---: |
| Schema migration and backup/restore | Record outcome | Record outcome | Record outcome | Record outcome |
| Sole-writer lock, scoped provenance, and conflict-safe writes | Record outcome | Record outcome | Record outcome | Record outcome |
| Durable embedding outbox, leased processor, daemon supervision, and health | Record outcome | Record outcome | Record outcome | Record outcome |
| Fail-closed artifact attestation before database open | Record outcome | Record outcome | Record outcome | Record outcome |
| Scoped, attributed, token-bounded `memory.context` | Record outcome | Record outcome | Record outcome | Record outcome |
| Offline synthetic evaluation | Record outcome | Record outcome | Not applicable | Not applicable |
| Bounded process-group-safe extraction adapter | Record outcome | Record outcome | Record outcome | Record outcome |
| Opt-in daemon configuration and automatic host-model extraction | Record outcome | Record outcome | Record outcome | Record outcome |
| Migration, search/evidence smoke, and rollback using a fresh database copy | Record outcome | Record outcome | Record outcome | Record outcome |
| Configured extraction, embedding drain, attestation, and daemon health | Record outcome | Record outcome | Record outcome | Record outcome |
| Quiet client cutover | Record outcome | Not applicable | Record outcome | Record outcome |

“Repository-verified” above is supported by checked-in focused tests, including
`tests/test_embedding_jobs.py`, `tests/test_ollama_artifact.py`,
`tests/test_context.py`, `tests/test_context_arena.py`,
`tests/test_host_extractor.py`, and the server/integration suites. These focused
checks do not replace an authoritative full-suite run on the frozen activation
candidate and are not evidence about the live host.

Record activation evidence in the operator's private change-management system,
not in the public repository.

## Completed activation procedure and rollback-safe reference

1. Freeze the intended package and deployment configuration. Run the full test
   suite, focused lint, configuration `check`, and the synthetic Context Arena.
2. Create and verify an authoritative SQLite backup. Work only on a fresh copy:
   migrate once, start the daemon, attest the configured artifact digest, drain
   embedding and extraction work, exercise scoped clients, inspect health, then
   restore and verify the pre-v1 schema/data path.
3. Record rehearsal inputs and results in private operator records. Do not
   include secrets, provider payloads, host identifiers, exact checksums, or
   database-specific identifiers in this repository.
4. Schedule a quiet maintenance window. Stop every legacy writer, take a new
   verified backup, migrate the live store once, start the sole-writer daemon,
   require healthy attestation/workers, then reconnect each registered client
   one at a time and verify attribution and scope after each connection.
5. If any gate fails, disconnect v1 clients, stop the daemon, restore the
   verified pre-migration database and prior client configuration, verify the
   old schema, and only then reconnect legacy readers and writers.

No future migration is authorized by repository-only tests or a synthetic
Arena run; it must repeat the evidence-graded procedure above.
