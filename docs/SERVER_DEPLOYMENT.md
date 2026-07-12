# Standalone daemon packaging

`enfold-server` is the foreground entry point for the shared Enfold service.
It never creates or migrates a database. Run
`python -m enfold.ops migrate /absolute/path/to/memory.db` during an explicit
maintenance window before starting it against an older store.

Example configuration (store as a user-owned, non-group/world-writable file):

```json
{
  "database_path": "/absolute/path/to/memory.db",
  "socket_path": "/absolute/private/directory/enfold.sock",
  "retrieval": {
    "mode": "ci",
    "allow_nonproduction": true,
    "dimensions": 256
  },
  "grants": {
    "client-a-install-1": ["private", "work", "project:enfold"],
    "client-b-install-1": ["private", "work", "project:enfold"],
    "hermes-native": ["private", "work", "project:enfold"]
  }
}
```

The `retrieval` object is required. `ci` mode is an offline plumbing test, not
a semantic production retriever, and therefore requires the conspicuous
`allow_nonproduction: true` opt-in. Health output preserves
`embedder_production_ready: false`.

The staged stored retrieval stack has a concrete SQLite backend and validates all
of the following before use: exact query-to-document identity-role mapping,
embedding version, dimension, active-fact coverage, candidate-only vector
loading, finite vectors, and complete per-search coverage. Missing or malformed
vectors fail closed. A representative configuration is:

```json
{
  "retrieval": {
    "mode": "stored",
    "provider": "ollama",
    "model": "embeddinggemma",
    "dimensions": 768,
    "query_identity": "ollama:embeddinggemma:query:none:sha256:<64-lowercase-hex>",
    "document_identity": "ollama:embeddinggemma:document:none:sha256:<64-lowercase-hex>",
    "embedding_version": "sha256:<64-lowercase-hex>",
    "model_fingerprint": "sha256:<64-lowercase-hex>",
    "prefix_policy": "none",
    "processor": {"mode": "daemon-supervised"},
    "query_prefix": ""
  }
}
```

`model_fingerprint` is mandatory and must be an immutable
`sha256:<64-lowercase-hex>` Ollama artifact digest. `embedding_version` must
equal that digest and remain the final component of both identities.

Before stored-mode `check`, `status`, or startup can proceed, Enfold resolves
the configured local Ollama tag through `/api/tags` and fails closed unless its
reported digest matches exactly. A mutable tag is therefore never an
attestation. Safe check and authenticated health output exposes only
`artifact_attestation: {"provider": "ollama", "status": "verified"}`; it
does not expose provider payloads or artifact details.

The model fingerprint equals the final identity version component, and the
identity is derived from provider, model, role, prefix policy, and fingerprint.
Non-empty prefixes require a full SHA-256 prefix policy.

Daemon-supervised stored mode currently permits Ollama only because its request
timeout bounds shutdown. FastEmbed remains blocked until embedding inference is
isolated in a killable worker process; an unbounded in-process ONNX call must
not be able to strand the sole-writer daemon during shutdown.

Stored mode provisions a durable `embedding_jobs` outbox. Every new fact and
its exact identity/version/dimension job commit in one transaction; no model
call occurs on `memory.write`. `EmbeddingJobProcessor.process_one()` is an
explicit lease/retry/dead-letter worker run by the daemon on a dedicated SQLite
connection. `check` verifies the configured artifact and reports safe
attestation state; live health reports the
worker heartbeat, last success/error, and outbox state. The worker stops and
joins before its connection closes. No model work occurs on `memory.write`.

An explicit maintenance flow can call `EmbeddingOutbox.enqueue_backfill()` for
preexisting active facts; the daemon supervisor also owns this at startup.
`check` remains read-only. A missing vector is temporarily eligible only when
an exact pending/processing job exists; that
candidate remains searchable lexically with zero dense contribution. Missing
work without a viable job, malformed vectors, identity mismatch, and dead
letters make health unsafe and block activation or fail retrieval. Expired
leases are reclaimable, retries are bounded, and completion revalidates the
lease, content hash, and current/non-erased fact state after the model call.

The socket directory must already exist, be owned by the current user, and
must not be group/world writable. Validate without binding a socket:

```bash
enfold-server --config /absolute/path/to/server.json check
enfold-server --config /absolute/path/to/server.json status
```

For stored mode, `status` deliberately does not call a bare socket connection
“healthy”: socket probing cannot attest the worker heartbeat. It reports live
health as unverified and exits nonzero; query the authenticated protocol
`health` method to evaluate heartbeat, errors, dead letters, and pending age.

Run the foreground process only after validation:

```bash
enfold-server --config /absolute/path/to/server.json run
```

Any configuration, database, or socket path below `~/.hermes` is refused
unless `--allow-live` is supplied explicitly. The flag acknowledges a live
deployment; it does not migrate, back up, install, or register anything.

Optional top-level configuration fields are `busy_timeout_ms`, `client_timeout`,
`shutdown_timeout`, `max_frame_bytes`, `backlog`, and
`cleanup_stale_socket`. The optional `extraction` object is described below.
Unknown fields fail validation.

## Automatic host-model extraction

Automatic model extraction is opt-in and defaults to
`{"mode": "disabled"}`. The implemented subprocess boundary uses explicit argv,
does not invoke a shell or inherit the daemon environment, bounds JSON input,
output, errors, and execution time, validates structured proposals, and
terminates the child process group on failure. A representative configuration
is:

```json
{
  "extraction": {
    "mode": "daemon-supervised",
    "host": {
      "type": "subprocess",
      "argv": ["/absolute/path/to/extractor-command"],
      "model_identity": "host-model-v1",
      "prompt_identity": "durable-memory-v1",
      "timeout_seconds": 180,
      "terminate_grace_seconds": 2,
      "max_input_bytes": 16384,
      "max_output_bytes": 65536,
      "max_error_bytes": 16384,
      "environment": {}
    },
    "poll_seconds": 1,
    "drain_limit": 4,
    "lease_seconds": 300,
    "heartbeat_seconds": 30,
    "retry_delay_seconds": 1,
    "max_attempts": 3,
    "heartbeat_stale_seconds": 240,
    "pending_stale_seconds": 900
  }
}
```

The executable path must be absolute. Environment inheritance is disabled;
only explicitly listed values reach the child. Keep the user-owned server
configuration private and do not commit credentials. The daemon gives the
worker a dedicated SQLite connection, exposes worker and queue state through
authenticated health, stops new claims during shutdown, and stops the worker
before closing its connection. Repository verification is not a real-store
rehearsal or permission to activate; follow
[`ACTIVATION_CHECKLIST.md`](ACTIVATION_CHECKLIST.md).

### Bundled local Ollama child

`enfold-ollama-extractor` implements the subprocess contract for a local
Ollama `/api/chat` endpoint. It accepts only loopback HTTP URLs, disables
ambient proxies, supplies a strict system prompt and JSON Schema `format`,
bounds and validates the response, and never writes transcript or model text
to stderr. The model is configurable; `qwen3:30b` is only the example default.

```json
{
  "type": "subprocess",
  "argv": [
    "/absolute/path/to/enfold-ollama-extractor",
    "--endpoint", "http://127.0.0.1:11434/api/chat",
    "--model", "qwen3:30b",
    "--model-identity", "ollama:qwen3-30b",
    "--prompt-identity", "durable-memory-v1"
  ],
  "model_identity": "ollama:qwen3-30b",
  "prompt_identity": "durable-memory-v1",
  "environment": {}
}
```

The child verifies both request identities. Alternatively, the only non-secret
variables it reads are `ENFOLD_OLLAMA_ENDPOINT`, `ENFOLD_OLLAMA_MODEL`,
`ENFOLD_OLLAMA_TIMEOUT_SECONDS`, and `ENFOLD_OLLAMA_MAX_RESPONSE_BYTES`.
Authenticated and non-local endpoints are intentionally unsupported. Success
emits exactly one canonical `{"proposals": [...], "version": 1}` object.
Configuration, invalid-data, unavailable-service, and unexpected failures use
stable statuses 64, 65, 69, and 70 without a diagnostic body.
