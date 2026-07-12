# Changelog

## 0.8.0

- Added optional `sqlite-vec==0.1.9` indexing with validated automatic
  brute-force fallback and an explicit `rebuild-vector-index` operation.
- Added standalone hybrid ranking priors, confidence abstention, bounded MMR
  context packing, and typed state-slot handling.
- Added near-duplicate merge handling for untyped writes, preserving one fact
  and superseding the other after recording the incoming observation.
- Added MCP change, timeline, entity-list, and entity-detail tools.
- Added verified backup secondary destinations, optional age encryption, restore
  rehearsal reports, and filtered browse snapshots for Datasette.
- Added public synthetic and private-corpus personal Arena evaluation surfaces.
