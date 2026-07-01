"""Stdio MCP server exposing holographic_plus as a shared-memory tool set.

Lets other coding agents (Claude Code, Codex CLI) read and write the same
fact store the Hermes gateway uses in-process, over the Model Context
Protocol, instead of each agent keeping its own disconnected memory.

Tools:
    memory_search(query, limit)                          -- hybrid search
    memory_add(content, category, tags, source)           -- write, dedup-gated
    memory_supersede(old_fact_id, new_content, source)     -- explicit update
    memory_explain(query, limit)                           -- scoring breakdown
    memory_history(fact_id)                                -- supersession chain

memory_search/memory_explain/memory_history are read-only and always
registered. memory_add/memory_supersede are writes and are omitted entirely
in --read-only mode (registered but return errors is NOT the read-only
contract here: the tools simply do not exist, so a read-only client can
never even attempt one).

Run directly (run the file by path, NOT `python -m holographic_plus.mcp_server`;
see the warning below for why):

    python holographic_plus/mcp_server.py \\
        --db-path ~/.hermes/memory_store.db \\
        --ollama-url http://localhost:11434 \\
        --ollama-model embeddinggemma:latest

See mcp_provider.py for how the parent hermes modules and db connection are
resolved and configured.

IMPORTANT -- run this file by path, never via `-m`: importing
holographic_plus as a package (``python -m holographic_plus.mcp_server``, or
any ``import holographic_plus`` before this module has resolved its parent)
runs holographic_plus/__init__.py first, Python's own package-import
semantics, and that file does its own unconditional
``from plugins.memory.holographic import HolographicMemoryProvider`` at
module level. On a host with a *separate* Hermes install already on
sys.path (e.g. a pip-installed hermes-agent), that import silently wins the
race and this module's own HOLOPLUS_HERMES_SRC resolution never gets a
chance to run. Executing this file directly (``python
holographic_plus/mcp_server.py ...``) sidesteps the package __init__.py
entirely, which is why every example here uses the file path.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover - posix-only; this server targets Linux
    fcntl = None

_THIS_DIR = Path(__file__).resolve().parent

# mcp_provider decides which parent hermes modules to load (real checkout vs
# the bundled fake_hermes stubs) and must run BEFORE holographic_plus itself
# is imported as a package, since `import holographic_plus` runs its
# __init__.py, which does its own unconditional parent import at module
# level. Load it by file path so this module never triggers that.


def _load_mcp_provider():
    name = "_holographic_plus_mcp_provider"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _THIS_DIR / "mcp_provider.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mcp_provider = _load_mcp_provider()

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without the dep
    raise ImportError(
        "The 'mcp' package is required to run the holographic_plus MCP server "
        "(pip install mcp). It is an optional dependency of this repo, only "
        "needed for mcp_server.py / mcp_provider.py, not for the Hermes plugin "
        "itself."
    ) from exc


VALID_SOURCES = ("claude-code", "codex", "other")

_T = TypeVar("_T")


@contextlib.contextmanager
def _cross_process_write_lock(db_path: str) -> Iterator[None]:
    """Serialize writers across separate MCP server processes sharing db_path.

    A holographic_plus write (dedup search, insert, HRR bank rebuild,
    optional supersession) is several separate short SQLite transactions,
    not one. SQLite's own busy_timeout retries a single blocked statement,
    but it cannot make that whole multi-statement sequence atomic across two
    processes: one process can complete its INSERT, then have its bank
    rebuild collide with another process's INSERT, and so on, so contention
    compounds instead of just queueing once. An OS advisory lock (flock) on
    a sidecar file next to db_path makes each MCP write fully serialized
    with any other process's write to the *same* db, which is what actually
    eliminates SQLITE_BUSY here (verified: without this lock, two processes
    each adding 25 facts to a fresh db occasionally exhaust a 30-attempt
    exponential backoff and still fail).

    A no-op within a single process across threads (RLock in fake_hermes/
    the real store already serializes those); this only matters once there
    are two separate OS processes, which is the deployment shape multiple
    MCP server instances (Claude Code, Codex CLI, ...) actually have.
    """
    if fcntl is None:  # pragma: no cover - posix-only
        yield
        return
    lock_path = f"{db_path}.mcp-write.lock"
    fh = open(lock_path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def _retry_on_locked(fn: Callable[[], _T], attempts: int = 30, base_delay: float = 0.02) -> _T:
    """Retry *fn* through transient SQLITE_BUSY from another writer.

    Belt-and-suspenders on top of _cross_process_write_lock and
    PRAGMA busy_timeout (see mcp_provider._apply_busy_timeout): the
    background embed/backfill/extraction threads inside the provider itself
    (not the cross-process write lock's concern) can still occasionally
    collide with a foreground write. Exponential backoff, capped, never
    masking a non-lock failure.
    """
    delay = base_delay
    for attempt in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 1.0)
    raise AssertionError("unreachable")  # pragma: no cover


def _require_source(args: Dict[str, Any]) -> Optional[str]:
    """Validate the required `source` tag; returns an error string or None."""
    source = args.get("source")
    if not source:
        return "missing required argument: source (one of claude-code, codex, other)"
    if source not in VALID_SOURCES:
        return f"invalid source {source!r}; must be one of {VALID_SOURCES}"
    return None


def _json_safe_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    """Drop non-JSON-serializable columns (the raw hrr_vector BLOB) from a fact row.

    temporal.fact_history() does ``SELECT *``, which includes hrr_vector;
    every other read path in this package (search, explain_search) already
    excludes it before returning.
    """
    return {k: v for k, v in fact.items() if k != "hrr_vector"}


def _tag_source(tags: str, source: str) -> str:
    """Append a `source:<agent>` marker to *tags*, comma-separated."""
    marker = f"source:{source}"
    existing = [t for t in (tags or "").split(",") if t.strip()]
    if marker not in existing:
        existing.append(marker)
    return ",".join(existing)


def build_server(provider, read_only: bool = False) -> "FastMCP":
    """Register holographic_plus tools against *provider* and return the FastMCP app.

    *provider* must already be initialized (see mcp_provider.build_provider).
    When *read_only* is true, memory_add and memory_supersede are never
    registered at all.
    """
    server = FastMCP("holographic-plus-memory")
    db_path = str(provider._store.db_path)

    @server.tool()
    def memory_search(query: str, limit: int = 10) -> Dict[str, Any]:
        """Hybrid search (FTS + Jaccard + HRR + dense embedding) over the shared fact store."""
        results = provider.search(query, limit=limit, bump=False)
        return {"results": results, "count": len(results)}

    @server.tool()
    def memory_explain(query: str, limit: int = 10) -> Dict[str, Any]:
        """Per-candidate scoring breakdown for *query* (same pass memory_search uses)."""
        breakdown = provider.explain_search(query, limit=limit)
        return {"breakdown": breakdown}

    @server.tool()
    def memory_history(fact_id: int) -> Dict[str, Any]:
        """Full supersession chain containing *fact_id*, oldest first."""
        return {"history": [_json_safe_fact(f) for f in provider.fact_history(int(fact_id))]}

    if read_only:
        return server

    @server.tool()
    def memory_add(
        content: str,
        source: str,
        category: str = "general",
        tags: str = "",
    ) -> Dict[str, Any]:
        """Add a fact, tagged with its originating agent.

        Routes through the same near-duplicate dedup gate and value-update
        supersession as the live Hermes write path: a near-verbatim
        restatement is rejected (status "deduped", the existing fact_id is
        returned instead), and a genuine value update (same wording, a
        changed number/id/state word) supersedes the prior fact rather than
        leaving both live.

        source must be one of: claude-code, codex, other.
        """
        args = {"content": content, "category": category, "source": source}
        error = _require_source(args)
        if error:
            return {"error": error}

        tagged = _tag_source(tags, source)

        def _do_add() -> Dict[str, Any]:
            dup = provider._find_near_duplicate(content, category=category)
            if dup is not None:
                return {
                    "fact_id": dup.get("fact_id"),
                    "status": "deduped",
                    "note": (
                        f"near-duplicate of existing fact {dup.get('fact_id')}; "
                        "not stored again"
                    ),
                }

            update_target = provider._find_update_target(content, category=category)
            fact_id = provider._store.add_fact(content, category=category, tags=tagged)
            provider._embed_cb(fact_id, content)

            if update_target is not None:
                provider._supersede_fact(int(update_target["fact_id"]), int(fact_id))

            return {"fact_id": fact_id, "status": "added"}

        with _cross_process_write_lock(db_path):
            return _retry_on_locked(_do_add)

    @server.tool()
    def memory_supersede(
        old_fact_id: int,
        new_content: str,
        source: str,
        category: str = "general",
        tags: str = "",
    ) -> Dict[str, Any]:
        """Explicitly supersede old_fact_id with a new fact (invalidate-not-delete).

        Unlike memory_add's implicit value-update detection, this always
        inserts new_content as a new fact and marks old_fact_id invalid,
        regardless of how similar the wording is. source must be one of:
        claude-code, codex, other.
        """
        args = {"content": new_content, "source": source}
        error = _require_source(args)
        if error:
            return {"error": error}

        tagged = _tag_source(tags, source)

        def _do_supersede() -> Dict[str, Any]:
            new_fact_id = provider._store.add_fact(new_content, category=category, tags=tagged)
            provider._embed_cb(new_fact_id, new_content)
            provider._supersede_fact(int(old_fact_id), int(new_fact_id))
            return {
                "fact_id": new_fact_id,
                "status": "superseded",
                "superseded": int(old_fact_id),
            }

        with _cross_process_write_lock(db_path):
            return _retry_on_locked(_do_supersede)

    return server


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default=None,
        help="path to memory_store.db (default: live ~/.hermes/memory_store.db)",
    )
    parser.add_argument(
        "--hermes-src",
        default=None,
        help="path to a hermes source checkout providing plugins.memory.holographic "
        "(default: $HOLOPLUS_HERMES_SRC or ~/hermes-migration-stage/src)",
    )
    parser.add_argument("--embedding-backend", default="ollama",
                         help="ollama or fastembed (default ollama)")
    parser.add_argument("--ollama-url", default=mcp_provider.DEFAULT_OLLAMA_URL)
    parser.add_argument("--ollama-model", default=mcp_provider.DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--embedding-prefix-policy", default=mcp_provider.DEFAULT_PREFIX_POLICY)
    parser.add_argument("--hrr-dim", type=int, default=1024)
    parser.add_argument("--dedup-jaccard", type=float, default=0.9)
    parser.add_argument("--dedup-cosine", type=float, default=0.92)
    parser.add_argument(
        "--busy-timeout-ms", type=int, default=mcp_provider.DEFAULT_BUSY_TIMEOUT_MS,
        help="sqlite busy_timeout in ms for concurrent writers (default 5000)",
    )
    parser.add_argument(
        "--read-only", action="store_true",
        help="register only memory_search/memory_explain/memory_history",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    db_path = args.db_path
    if db_path is None:
        try:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        except Exception:
            db_path = str(Path.home() / ".hermes" / "memory_store.db")

    provider = mcp_provider.build_provider(
        db_path=db_path,
        hermes_src=args.hermes_src,
        embedding_backend=args.embedding_backend,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        embedding_prefix_policy=args.embedding_prefix_policy,
        hrr_dim=args.hrr_dim,
        dedup_jaccard=args.dedup_jaccard,
        dedup_cosine=args.dedup_cosine,
        busy_timeout_ms=args.busy_timeout_ms,
        session_id="mcp-server",
    )
    try:
        server = build_server(provider, read_only=args.read_only)
        server.run(transport="stdio")
    finally:
        provider.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
