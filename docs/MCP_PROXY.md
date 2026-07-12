# Enfold v1 MCP stdio proxy

`enfold-mcp-proxy` is the thin MCP adapter for an already-running standalone
Enfold v1 daemon. It never opens SQLite and does not import Hermes. Every tool
call uses `EnfoldClient`, which opens a fresh Unix-socket connection, negotiates
the fixed client context, performs one request, and disconnects.

Install the optional MCP dependency and inspect all startup options:

```bash
python -m pip install -e '.[mcp]'
enfold-mcp-proxy --help
```

For a static MCP client registration, use `enfold-mcp-launch`. It creates
a fresh cryptographically random session ID for every proxy process and safely
captures the process CWD, Git root, credential-free origin, branch, and commit
when available. Client, surface, agent, socket, and requested scopes remain
explicit immutable registration arguments; environment variables and tool
parameters cannot override them. Git discovery uses bounded subprocess argv
without a shell, prompt, pager, or inherited environment.

MCP client A registration (the `client-a-install-1` grant must exist in the
daemon configuration with the same or broader scopes):

```bash
mcp-client-a add enfold -- /path/to/enfold/.venv/bin/python \
  -m enfold.mcp_launcher \
  --socket-path /path/to/enfold.sock \
  --client-id client-a-install-1 \
  --surface mcp-client-a \
  --agent-id client-a \
  --access-scope private \
  --access-scope work \
  --access-scope project:enfold
```

MCP client B registration (use a distinct server grant):

```bash
mcp-client-b add enfold -- \
  /path/to/enfold/.venv/bin/python -m enfold.mcp_launcher \
  --socket-path /path/to/enfold.sock \
  --client-id client-b-install-1 \
  --surface mcp-client-b \
  --agent-id client-b \
  --access-scope private \
  --access-scope work \
  --access-scope project:enfold
```

Do not put `--session-id` in a static registration: omission is what gives
each proxy process a fresh session. An explicit session is available only for
a trusted supervisor that already owns a stable session identifier.

Direct proxy launch remains useful for diagnostics (only after an Enfold
daemon is running):

```bash
enfold-mcp-proxy \
  --socket-path /path/to/enfold.sock \
  --client-id client-a-install-1 \
  --surface mcp-client-a \
  --agent-id client-a \
  --session-id client-a-thread-123 \
  --project-root /path/to/project \
  --repository owner/project \
  --branch main \
  --access-scope private \
  --access-scope work
```

The required values also accept `ENFOLD_SOCKET_PATH`, `ENFOLD_CLIENT_ID`,
`ENFOLD_SURFACE`, `ENFOLD_AGENT_ID`, and `ENFOLD_SESSION_ID`. Optional
provenance and scope variables are listed by `--help`.

The tools are `memory_write`, `memory_search`, `memory_context`,
`memory_evidence`, `memory_history`, `memory_conflicts`,
`memory_resolve_conflict`, and `memory_extraction_enqueue`.
Writer/session/project identity is not present in their schemas: it comes only
from startup context. The `asserted_by` write field records the subject who
made a claim; it does not replace the connection's writer identity.

Daemon application failures become MCP tool errors whose message is compact
JSON containing `code`, `message`, `retryable`, `details`, and `request_id`.
Transport outages use `code="daemon_unavailable"` and are retryable. Successful
results are checked and normalized to JSON before they cross the MCP boundary.

This adapter establishes reliable attribution between cooperative local
clients. It is not an authentication boundary against another process running
as the same operating-system user.
