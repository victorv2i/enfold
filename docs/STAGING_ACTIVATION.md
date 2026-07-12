# Enfold v1 staging and activation boundary

The repository now contains a concrete Hermes provider bridge at
`integrations/hermes_enfold_v1`. It is deliberately separate from the legacy
`enfold` Hermes provider. Repository tests may import it, but nothing copies,
registers, starts, or connects it to live memory.

The authoritative current gate state is
[`ACTIVATION_CHECKLIST.md`](ACTIVATION_CHECKLIST.md). In particular,
“implemented,” “repository-verified,” “rehearsed on a real-store copy,” and
“live-activated” are separate evidence levels.

## Runtime ownership

`enfold-server` acquires an exclusive advisory `flock` on
`<database>.enfold.lock` before opening the writable SQLite connection. A
second server refuses startup even if it is configured with another socket.
The 0600 sidecar persists after shutdown so every process coordinates on one
stable inode; unlinking a lock file during handoff can create split ownership.

On Linux the daemon also records kernel-provided `SO_PEERCRED` PID/UID/GID for
accepted local peers and refuses a peer from another UID. Protocol attribution
still comes from the immutable handshake because OS credentials identify the
client process, not the agent or session inside it.

Signal handlers only set the daemon stop event. They do not acquire lifecycle
locks, join worker threads, close SQLite, or unlink sockets. Normal teardown
runs after the bounded accept timeout observes the event.

## Hermes bridge behavior

The staged `enfold_v1` provider:

- creates immutable daemon sessions from Hermes session, agent, project, Git,
  scope, and parent-agent context;
- prefetches current scoped facts before turns;
- exposes explicit search/add/evidence/history/conflicts operations;
- mirrors explicit built-in memory additions with stable idempotency keys;
- records delegation results under the child session and agent with parent
  agent attribution;
- enqueues attributed session-end and pre-compression transcripts through the
  daemon protocol, without running a model in the Hermes hook;
- returns empty prefetch on daemon outage and a visible retryable tool error;
- never spools, redirects, or writes directly to SQLite when the daemon is
  unavailable.

## Extraction processor boundary

`enfold.extraction_processor.ExtractionProcessor` is the staged processing
contract. It claims provisioned queue rows with leases, accepts structured
proposals from a pluggable extractor, screens the full batch before writing,
and applies each proposal through the authoritative `EnfoldService` write
route. Payload-derived idempotency keys make an expired lease safe to replay
after a crash. Transient failures retry; malformed, secret, or repeatedly
failing jobs remain as dead-letter rows.

The processor is fail-closed and host-driven: importing Enfold does not start
a worker, and the server does not invent or select a model. The repository now
includes a reviewed, bounded `SubprocessHostExtractor` adapter. It launches an
explicit argv without a shell or ambient environment, exchanges bounded JSON,
and cleans up the child process group. Explicit `daemon-supervised` server
configuration now constructs it with a dedicated SQLite connection and a
daemon-owned `ExtractionProcessor` worker. The worker reports bounded health,
stops future claims during shutdown, and stops before its connection closes.
Automatic extraction remains disabled by default and is repository-verified,
not rehearsed against a copy of the real store or live-activated.

## Dense embedding boundary

Stored semantic retrieval validates an exact model identity, version,
dimension, and complete active-fact vector coverage. It never substitutes the
deterministic CI embedder or compares mismatched vectors. The repository has
an explicit idempotent backfill primitive, but synchronous memory writes never
call an embedding model. The repository implements a durable asynchronous
write-to-embedding outbox, leased processor, daemon-owned worker lifecycle,
and health reporting. Every fact and its exact embedding job commit atomically;
model calls occur outside the write transaction. This is repository-verified,
not rehearsed against a copy of the real store and not live-activated.

Stored Ollama mode additionally requires a configured immutable
`sha256:<64-lowercase-hex>` artifact digest. Before a server opens the database
or schedules backfill, it resolves the configured local model tag and fails
closed unless the reported artifact digest matches. Health and `check` expose
only the verified/unverified attestation state, never provider payloads.

Environment selected during activation:

- `ENFOLD_SOCKET_PATH`: absolute daemon socket path;
- `ENFOLD_HERMES_CLIENT_ID`: grant-matched installation identity;
- `ENFOLD_HERMES_SCOPES`: comma-separated requested scopes, narrowed by the
  server grant;
- optional connect/request timeout variables.

## Maintenance window

Follow [`ACTIVATION_CHECKLIST.md`](ACTIVATION_CHECKLIST.md) and
the private activation record. Stop all memory-writing sessions only after
the copied-real-store rehearsal succeeds. Back up and migrate once, start and
health-check the sole-writer daemon, then connect Hermes and each MCP client
individually. The legacy v0 provider remains unchanged and must never open a
schema-v1 database.
