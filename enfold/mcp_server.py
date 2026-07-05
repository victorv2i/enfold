"""Stdio MCP server exposing enfold as a shared-memory tool set.

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

In --read-only mode the provider is also opened read-only. Startup skips all
mutating initialization work: schema migrations, WAL checkpoints, embedding
backfill, extraction queue workers, and reflection passes. The server can
search existing stores but will not create or repair database objects.

Run directly (run the file by path, NOT `python -m enfold.mcp_server`;
see the warning below for why):

    python enfold/mcp_server.py \\
        --db-path ~/.hermes/memory_store.db \\
        --ollama-url http://localhost:11434 \\
        --ollama-model embeddinggemma:latest

See mcp_provider.py for how the parent hermes modules and db connection are
resolved and configured.

IMPORTANT -- run this file by path, never via `-m`: importing
enfold as a package (``python -m enfold.mcp_server``, or
any ``import enfold`` before this module has resolved its parent)
runs enfold/__init__.py first, Python's own package-import
semantics, and that file does its own unconditional
``from plugins.memory.holographic import HolographicMemoryProvider`` at
module level. On a host with a *separate* Hermes install already on
sys.path (e.g. a pip-installed hermes-agent), that import silently wins the
race and this module's own ENFOLD_HERMES_SRC resolution never gets a
chance to run. Executing this file directly (``python
enfold/mcp_server.py ...``) sidesteps the package __init__.py
entirely, which is why every example here uses the file path.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

_THIS_DIR = Path(__file__).resolve().parent

# mcp_provider decides which parent hermes modules to load (real checkout vs
# the bundled fake_hermes stubs) and must run BEFORE enfold itself
# is imported as a package, since `import enfold` runs its
# __init__.py, which does its own unconditional parent import at module
# level. Load it by file path so this module never triggers that.


def _load_mcp_provider():
    name = "_enfold_mcp_provider"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _THIS_DIR / "mcp_provider.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mcp_provider = _load_mcp_provider()


def _load_write_lock():
    name = "_enfold_write_lock"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _THIS_DIR / "write_lock.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_cross_process_write_lock = _load_write_lock().cross_process_write_lock

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without the dep
    raise ImportError(
        "The 'mcp' package is required to run the enfold MCP server "
        "(pip install mcp). It is an optional dependency of this repo, only "
        "needed for mcp_server.py / mcp_provider.py, not for the Hermes plugin "
        "itself."
    ) from exc


VALID_SOURCES = ("claude-code", "codex", "other")
MAX_CONTENT_CHARS = 16_000
MAX_QUERY_CHARS = 16_000
MAX_TAGS_CHARS = 2_000
MAX_CATEGORY_CHARS = 128
MIN_LIMIT = 1
MAX_LIMIT = 50

_T = TypeVar("_T")
logger = logging.getLogger(__name__)


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
    if not isinstance(source, str):
        return "invalid source: must be a string"
    if source not in VALID_SOURCES:
        return f"invalid source {source!r}; must be one of {VALID_SOURCES}"
    return None


def _require_string(name: str, value: Any, max_chars: int, *, non_blank: bool = False) -> Optional[str]:
    if not isinstance(value, str):
        return f"invalid {name}: must be a string"
    if len(value) > max_chars:
        return f"invalid {name}: exceeds {max_chars} characters"
    if non_blank and not value.strip():
        return f"invalid {name}: must not be blank"
    return None


def _validated_limit(limit: Any) -> tuple[Optional[int], Optional[str]]:
    if isinstance(limit, bool):
        return None, "invalid limit: must be an integer"
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return None, "invalid limit: must be an integer"
    return max(MIN_LIMIT, min(MAX_LIMIT, value)), None


def _validate_fact_args(content: Any, category: Any, tags: Any) -> Optional[str]:
    for name, value, max_chars, non_blank in (
        ("content", content, MAX_CONTENT_CHARS, True),
        ("category", category, MAX_CATEGORY_CHARS, False),
        ("tags", tags, MAX_TAGS_CHARS, False),
    ):
        error = _require_string(name, value, max_chars, non_blank=non_blank)
        if error:
            return error
    return None


def _json_safe_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    """Drop non-JSON-serializable columns (the raw hrr_vector BLOB) from a fact row.

    temporal.fact_history() does ``SELECT *``, which includes hrr_vector;
    every other read path in this package (search, explain_search) already
    excludes it before returning.
    """
    return {k: v for k, v in fact.items() if k != "hrr_vector"}


def _tag_source(tags: str, source: str) -> str:
    """Replace inbound source markers with exactly one canonical marker."""
    marker = f"source:{source}"
    existing = [
        t.strip()
        for t in (tags or "").split(",")
        if t.strip() and not t.strip().lower().startswith("source:")
    ]
    existing.append(marker)
    return ",".join(existing)


def _active_fact_exists(provider, fact_id: int) -> bool:
    store = getattr(provider, "_store", None)
    conn = getattr(store, "_conn", None)
    if conn is None:
        return False
    row = conn.execute(
        "SELECT fact_id FROM facts WHERE fact_id = ? AND invalid_at IS NULL",
        (fact_id,),
    ).fetchone()
    return row is not None


def _supersede_with_rowcount(provider, old_fact_id: int, new_fact_id: int) -> bool:
    store = getattr(provider, "_store", None)
    conn = getattr(store, "_conn", None)
    if conn is None:
        return False
    lock = getattr(store, "_lock", None)

    def _update() -> bool:
        cur = conn.execute(
            """
            UPDATE facts
               SET invalid_at = CURRENT_TIMESTAMP,
                   superseded_by = ?
             WHERE fact_id = ? AND invalid_at IS NULL
            """,
            (new_fact_id, old_fact_id),
        )
        conn.commit()
        if int(cur.rowcount) != 1:
            return False
        try:
            reflection = importlib.import_module("enfold.reflection")
            reflection.invalidate_insights_citing(conn, old_fact_id)
        except Exception as exc:
            logger.debug("enfold MCP: insight invalidation failed: %s", exc)
        return True

    if lock is not None:
        with lock:
            return _update()
    return _update()


def build_server(provider, read_only: bool = False) -> "FastMCP":
    """Register enfold tools against *provider* and return the FastMCP app.

    *provider* must already be initialized (see mcp_provider.build_provider).
    When *read_only* is true, memory_add and memory_supersede are never
    registered at all.
    """
    server = FastMCP("enfold-memory")
    db_path = str(Path(provider._store.db_path).expanduser().resolve())

    @server.tool()
    def memory_search(query: Any, limit: Any = 10) -> Dict[str, Any]:
        """Hybrid search (FTS + Jaccard + HRR + dense embedding) over the shared fact store."""
        error = _require_string("query", query, MAX_QUERY_CHARS)
        if error:
            return {"error": error}
        safe_limit, error = _validated_limit(limit)
        if error:
            return {"error": error}
        results = provider.search(query, limit=safe_limit, bump=False)
        return {"results": results, "count": len(results)}

    @server.tool()
    def memory_explain(query: Any, limit: Any = 10) -> Dict[str, Any]:
        """Per-candidate scoring breakdown for *query* (same pass memory_search uses)."""
        error = _require_string("query", query, MAX_QUERY_CHARS)
        if error:
            return {"error": error}
        safe_limit, error = _validated_limit(limit)
        if error:
            return {"error": error}
        breakdown = provider.explain_search(query, limit=safe_limit)
        return {"breakdown": breakdown}

    @server.tool()
    def memory_history(fact_id: Any) -> Dict[str, Any]:
        """Full supersession chain containing *fact_id*, oldest first."""
        try:
            safe_fact_id = int(fact_id)
        except (TypeError, ValueError):
            return {"error": "invalid fact_id: must be an integer"}
        return {"history": [_json_safe_fact(f) for f in provider.fact_history(safe_fact_id)]}

    if read_only:
        return server

    @server.tool()
    def memory_add(
        content: Any,
        source: Any,
        category: Any = "general",
        tags: Any = "",
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
        error = _validate_fact_args(content, category, tags)
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
        old_fact_id: Any,
        new_content: Any,
        source: Any,
        category: Any = "general",
        tags: Any = "",
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
        error = _validate_fact_args(new_content, category, tags)
        if error:
            return {"error": error}
        try:
            safe_old_fact_id = int(old_fact_id)
        except (TypeError, ValueError):
            return {"error": "invalid old_fact_id: must be a positive integer"}
        if safe_old_fact_id <= 0:
            return {"error": "invalid old_fact_id: must be a positive integer"}

        tagged = _tag_source(tags, source)

        def _do_supersede() -> Dict[str, Any]:
            if not _active_fact_exists(provider, safe_old_fact_id):
                return {"error": "invalid old_fact_id: active fact not found"}
            new_fact_id = provider._store.add_fact(new_content, category=category, tags=tagged)
            provider._embed_cb(new_fact_id, new_content)
            if not _supersede_with_rowcount(provider, safe_old_fact_id, int(new_fact_id)):
                return {
                    "fact_id": new_fact_id,
                    "status": "failed",
                    "error": "supersede failed: old fact was not updated",
                }
            return {
                "fact_id": new_fact_id,
                "status": "superseded",
                "superseded": safe_old_fact_id,
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
        "(default: $ENFOLD_HERMES_SRC or ~/hermes-migration-stage/src)",
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

    try:
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
            read_only=args.read_only,
        )
    except RuntimeError as exc:
        print(f"enfold MCP startup failed: {exc}", file=sys.stderr)
        return 1
    try:
        server = build_server(provider, read_only=args.read_only)
        server.run(transport="stdio")
    finally:
        provider.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
