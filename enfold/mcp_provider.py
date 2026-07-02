"""Provider factory for the enfold MCP server.

Builds a real ``EnfoldProvider`` against a configurable db_path, the
same way the offline ``explain.py`` CLI does, but resolves the parent
``plugins.memory.holographic`` modules from one of two sources:

  1. A real Hermes checkout, pointed at by the ``ENFOLD_HERMES_SRC`` env var
     or ``--hermes-src``. This is how the server shares the *exact* live
     store the Hermes gateway writes to.
  2. The repo's bundled ``tests/fake_hermes`` stubs, as a documented fallback
     for a host with no Hermes install and no explicit source (matches the
     test suite's own harness).

If a source is explicit and cannot be used, startup fails closed. Silent
fallback to the stubs is only for the default auto-discovery path.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sqlite3
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

DEFAULT_HERMES_SRC = str(Path.home() / "hermes-migration-stage" / "src")
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "embeddinggemma:latest"
DEFAULT_PREFIX_POLICY = "auto"
DEFAULT_BUSY_TIMEOUT_MS = 5000

_PARENT_PKG = "plugins.memory.holographic"
_MODULE_RESTORE_PREFIXES = ("agent", "plugins", "hermes_state")


def _canonical_db_path(db_path: str) -> str:
    return str(Path(db_path).expanduser().resolve())


def _sqlite_readonly_uri(db_path: str) -> str:
    return f"file:{quote(_canonical_db_path(db_path), safe='/')}?mode=ro"


def _module_snapshot() -> dict[str, Any]:
    return {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _MODULE_RESTORE_PREFIXES)
    }


def _restore_modules(snapshot: dict[str, Any]) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _MODULE_RESTORE_PREFIXES):
            if name not in snapshot:
                sys.modules.pop(name, None)
    for name, module in snapshot.items():
        sys.modules[name] = module


def _real_parent_available(hermes_src: str) -> bool:
    holo_dir = Path(hermes_src) / "plugins" / "memory" / "holographic"
    return (holo_dir / "retrieval.py").exists() and (holo_dir / "store.py").exists()


def _install_real_parent(hermes_src: str) -> None:
    """Register the real plugin.memory.holographic package from *hermes_src*."""
    if hermes_src not in sys.path:
        sys.path.insert(0, hermes_src)

    holo_dir = Path(hermes_src) / "plugins" / "memory" / "holographic"

    # The real store lazily imports hermes_state, which needs
    # agent.memory_manager; stub the one function it uses if not present.
    if "agent.memory_manager" not in sys.modules:
        agent_pkg = sys.modules.get("agent")
        if agent_pkg is None:
            agent_pkg = types.ModuleType("agent")
            agent_pkg.__path__ = []
            sys.modules["agent"] = agent_pkg
        mem_mgr = types.ModuleType("agent.memory_manager")
        mem_mgr.sanitize_context = lambda value: value
        sys.modules["agent.memory_manager"] = mem_mgr
        agent_pkg.memory_manager = mem_mgr

    pkg = types.ModuleType(_PARENT_PKG)
    pkg.__path__ = [str(holo_dir)]
    sys.modules[_PARENT_PKG] = pkg

    plugins_pkg = sys.modules.get("plugins") or types.ModuleType("plugins")
    plugins_pkg.__path__ = getattr(plugins_pkg, "__path__", [])
    sys.modules["plugins"] = plugins_pkg
    memory_pkg = sys.modules.get("plugins.memory") or types.ModuleType("plugins.memory")
    memory_pkg.__path__ = getattr(memory_pkg, "__path__", [])
    sys.modules["plugins.memory"] = memory_pkg
    plugins_pkg.memory = memory_pkg
    memory_pkg.holographic = pkg

    holographic_mod = importlib.import_module(_PARENT_PKG + ".holographic")
    retrieval_mod = importlib.import_module(_PARENT_PKG + ".retrieval")
    store_mod = importlib.import_module(_PARENT_PKG + ".store")

    pkg.holographic = holographic_mod
    pkg.retrieval = retrieval_mod
    pkg.store = store_mod

    # The real parent's own HolographicMemoryProvider lives in its
    # __init__.py, but *pkg* (the package module) is already registered as
    # this placeholder object in sys.modules, so importlib.import_module()
    # on the package name would just return it unexecuted. Exec the real
    # __init__.py's code directly into it instead.
    init_spec = importlib.util.spec_from_file_location(
        _PARENT_PKG, holo_dir / "__init__.py", submodule_search_locations=[str(holo_dir)]
    )
    init_spec.loader.exec_module(pkg)

    # Surface a broken hermes_state / env problem now, as an import error the
    # caller can catch and fall back on, rather than a later runtime crash.
    importlib.import_module("hermes_state")

    if not getattr(holographic_mod, "_HAS_NUMPY", True):
        raise RuntimeError("real holographic module reports numpy unavailable")


def _install_fake_parent() -> None:
    """Register the repo's bundled fake_hermes stubs as the parent modules.

    ``fake_hermes.install_stubs()`` is a no-op once the parent package is
    already in ``sys.modules`` (e.g. installed earlier by the test suite's
    own ``conftest.py``), so this only needs to guarantee the package is
    present, not that ``install_stubs`` runs again.
    """
    tests_dir = Path(__file__).resolve().parent.parent / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    import fake_hermes  # type: ignore

    fake_hermes.install_stubs()


def resolve_parent_modules(hermes_src: Optional[str] = None) -> str:
    """Ensure ``plugins.memory.holographic`` is importable; return which source was used.

    Returns ``"real"`` or ``"fake_hermes"``. Tries the real checkout first
    (explicit arg, env var ``ENFOLD_HERMES_SRC``, else ``DEFAULT_HERMES_SRC``).
    Explicit sources fail closed on missing or broken imports. The bundled
    stubs are used only when no source was specified.
    """
    env_src = os.environ.get("ENFOLD_HERMES_SRC")
    explicit_src = hermes_src is not None or env_src is not None
    src = hermes_src or env_src or DEFAULT_HERMES_SRC

    if explicit_src and not _real_parent_available(src):
        raise RuntimeError(
            f"explicit Hermes source {src!r} is missing plugins.memory.holographic "
            "retrieval.py or store.py"
        )

    if (
        not explicit_src
        and _PARENT_PKG in sys.modules
        and hasattr(sys.modules[_PARENT_PKG], "HolographicMemoryProvider")
    ):
        # Already resolved in this process (real, fake_hermes, or installed by
        # something else, e.g. the test suite's own conftest.py); reuse it
        # rather than re-importing. Anything installed without going through
        # this module (no source marker) is assumed to be the fake stubs,
        # since that is the only other installer in this codebase.
        used = getattr(sys.modules[_PARENT_PKG], "_enfold_mcp_source", "fake_hermes")
        logger.info("enfold MCP: resolved parent engine %s", used)
        return used

    if _real_parent_available(src):
        snapshot = _module_snapshot()
        try:
            _install_real_parent(src)
            sys.modules[_PARENT_PKG]._enfold_mcp_source = "real"
            logger.info("enfold MCP: using real hermes parent at %s", src)
            logger.info("enfold MCP: resolved parent engine real")
            return "real"
        except Exception as exc:
            _restore_modules(snapshot)
            if explicit_src:
                raise RuntimeError(
                    f"explicit Hermes source {src!r} failed to import: {exc}"
                ) from exc
            logger.warning(
                "enfold MCP: real hermes parent at %s failed to import (%s), "
                "falling back to fake_hermes stubs",
                src, exc,
            )

    _install_fake_parent()
    sys.modules[_PARENT_PKG]._enfold_mcp_source = "fake_hermes"
    logger.info(
        "enfold MCP: real hermes parent not found at %s, "
        "using tests/fake_hermes stubs",
        src,
    )
    logger.info("enfold MCP: resolved parent engine fake_hermes")
    return "fake_hermes"


def _load_enfold_module():
    """Import the enfold package fresh, bound to whichever parent
    modules resolve_parent_modules() just installed in sys.modules."""
    name = "enfold"
    if name in sys.modules and hasattr(sys.modules[name], "EnfoldProvider"):
        return sys.modules[name]
    pkg_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        name, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_journal_mode(db_path: str) -> str:
    """Read-only check of an existing sqlite file's journal_mode.

    Opens a short-lived connection, reads the pragma, and closes; does not
    modify the database. Returns the mode as reported by sqlite (e.g. "wal",
    "delete").
    """
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0])
    finally:
        conn.close()


class _ReadOnlyStore:
    def __init__(self, db_path: str, hrr_dim: int) -> None:
        self.db_path = _canonical_db_path(db_path)
        self.hrr_dim = hrr_dim
        self._hrr_available = True
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            _sqlite_readonly_uri(self.db_path),
            uri=True,
            check_same_thread=False,
            timeout=10.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA query_only = ON")

    def close(self) -> None:
        self._conn.close()


def _readonly_embed_store(hp, conn: sqlite3.Connection, identity: str, lock):
    embed_store = object.__new__(hp.EmbedStore)
    embed_store._conn = conn
    embed_store._lock = lock if lock is not None else threading.RLock()
    embed_store._embedding_identity = identity
    embed_store._cache_ids = None
    embed_store._cache_matrix = None
    embed_store._cache_dim = None
    embed_store._cache_identity = None
    return embed_store


def _configure_read_only_provider(provider, hp, session_id: str, hrr_dim: int) -> None:
    provider._store = _ReadOnlyStore(provider._config["db_path"], hrr_dim)
    provider._session_id = session_id

    temporal_decay = int(provider._config.get("temporal_decay_half_life", 0))
    fts_cfg = float(provider._config.get("fts_weight", hp._FTS_W))
    jac_cfg = float(provider._config.get("jaccard_weight", hp._JACCARD_W))
    hrr_cfg = float(provider._config.get("hrr_weight", hp._HRR_W))
    holo_sum = fts_cfg + jac_cfg + hrr_cfg
    if holo_sum <= 0:
        fts_cfg, jac_cfg, hrr_cfg = hp._FTS_W, hp._JACCARD_W, hp._HRR_W
        holo_sum = hp._FTS_W + hp._JACCARD_W + hp._HRR_W

    provider._retriever = hp.PlusFactRetriever(
        store=provider._store,
        temporal_decay_half_life=temporal_decay,
        fts_weight=fts_cfg / holo_sum,
        jaccard_weight=jac_cfg / holo_sum,
        hrr_weight=hrr_cfg / holo_sum,
        hrr_dim=hrr_dim,
        entity_boost_weight=float(provider._config.get("entity_boost_weight", 0.0)),
        entity_expansion=hp._cfg_bool(provider._config.get("entity_expansion"), False),
        entity_hub_degree_limit=int(provider._config.get("entity_hub_degree_limit", 25)),
    )

    try:
        provider._embedder = provider._create_embedder()
        provider._embedder_available = bool(provider._embedder.is_available())
    except Exception as exc:
        logger.debug("enfold MCP: read-only embedder unavailable: %s", exc)
        provider._embedder = None
        provider._embedder_available = False

    provider._embed_store = _readonly_embed_store(
        hp,
        provider._store._conn,
        provider._embedding_identity("document"),
        getattr(provider._store, "_lock", None),
    )
    provider._embed_pool = None
    provider._backfill_thread = None
    provider._backfill_stop = None
    provider._extract_queue = None
    provider._queue_worker = None
    provider._queue_stop = None
    provider._queue_wake = None


def build_provider(
    *,
    db_path: str,
    hermes_src: Optional[str] = None,
    embedding_backend: str = "ollama",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    embedding_prefix_policy: str = DEFAULT_PREFIX_POLICY,
    hrr_dim: int = 1024,
    dedup_jaccard: float = 0.9,
    dedup_cosine: float = 0.92,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    session_id: str = "mcp-server",
    read_only: bool = False,
):
    """Construct and initialize a EnfoldProvider against *db_path*.

    Resolves the parent hermes modules (real checkout or fake_hermes stubs),
    loads the enfold package against them, and initializes a
    provider with the given config. ``embedding_backend="fake"`` builds an
    in-process deterministic embedder instead of a network backend, for
    tests only.

    The caller owns the returned provider's lifecycle (call ``.shutdown()``
    when done); this mirrors the ``explain.py`` CLI's own usage.
    """
    parent_source = resolve_parent_modules(hermes_src)
    logger.info("enfold MCP: startup parent engine %s", parent_source)
    hp = _load_enfold_module()
    canonical_path = _canonical_db_path(db_path)

    config = {
        "db_path": canonical_path,
        "embedding_backend": embedding_backend if embedding_backend != "fake" else "ollama",
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
        "embedding_prefix_policy": embedding_prefix_policy,
        "hrr_dim": hrr_dim,
        "dedup_jaccard": dedup_jaccard,
        "dedup_cosine": dedup_cosine,
    }

    if embedding_backend == "fake":
        tests_dir = Path(__file__).resolve().parent.parent / "tests"
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        import fake_hermes  # type: ignore

        fake_embedder = fake_hermes.FakeEmbedder()

        class _FakeEmbedProvider(hp.EnfoldProvider):
            def _create_embedder(self):
                return fake_embedder

        provider = _FakeEmbedProvider(config=config)
    else:
        provider = hp.EnfoldProvider(config=config)

    if read_only:
        _configure_read_only_provider(provider, hp, session_id, hrr_dim)
    else:
        _initialize_with_retry(provider, session_id)
    _apply_busy_timeout(provider, busy_timeout_ms)
    return provider


def _initialize_with_retry(provider, session_id: str, attempts: int = 20, delay: float = 0.05) -> None:
    """Retry provider.initialize() through transient SQLITE_BUSY at connection open.

    Two processes racing to open a *fresh* db_path for the first time can
    both hit ``sqlite3.OperationalError: database is locked`` while the
    store sets ``PRAGMA journal_mode=WAL`` (a schema-level operation that
    briefly needs an exclusive lock even before any busy_timeout has been
    configured on this connection). This is a one-time startup race, not an
    ongoing write-contention issue (that is handled separately by
    busy_timeout once the connection is open), so a short bounded retry with
    backoff is sufficient and never masks a genuine non-lock failure.
    """
    for attempt in range(attempts):
        try:
            provider.initialize(session_id)
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == attempts - 1:
                raise
            time.sleep(delay)


def _apply_busy_timeout(provider, busy_timeout_ms: int) -> None:
    """Set PRAGMA busy_timeout on the provider's live store connection.

    A short (default 5000ms) busy timeout makes SQLITE_BUSY retries automatic
    across concurrent writers on WAL, instead of surfacing immediately as an
    error; short transactions elsewhere keep any single wait bounded.
    """
    store = getattr(provider, "_store", None)
    conn = getattr(store, "_conn", None)
    if conn is None:
        return
    lock = getattr(store, "_lock", None)
    if lock is not None:
        with lock:
            conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    else:
        conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
