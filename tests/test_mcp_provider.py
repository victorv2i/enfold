"""Tests for the MCP server's provider factory (enfold.mcp_provider).

Covers: resolving the parent hermes modules (real checkout via
ENFOLD_HERMES_SRC, else the bundled fake_hermes stubs), building a
EnfoldProvider against a configurable db_path/config, and the
journal_mode/busy_timeout concurrency settings the shared-DB use case needs.
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

import pytest

# mcp_provider.py decides which parent hermes modules to load (real checkout
# vs fake_hermes stubs) BEFORE enfold itself is ever imported. A
# plain `from enfold import mcp_provider` would run
# enfold/__init__.py first (Python package import semantics),
# which does its own unconditional `from plugins.memory.holographic import
# ...` at module level, resolving to whatever is already importable (e.g. a
# separate hermes-agent install on sys.path) before mcp_provider gets a say.
# Load it directly from its file instead, same technique conftest.py and
# test_real_parent_equivalence.py already use for the plugin package itself.
_MCP_PROVIDER_PATH = Path(__file__).resolve().parents[1] / "enfold" / "mcp_provider.py"


def _load_mcp_provider():
    name = "_hp_mcp_provider_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _MCP_PROVIDER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mcp_provider = _load_mcp_provider()


def test_resolve_parent_modules_falls_back_to_fake_hermes(monkeypatch):
    """With no real hermes checkout configured, the fake_hermes stubs are used."""
    monkeypatch.delenv("ENFOLD_HERMES_SRC", raising=False)
    used = mcp_provider.resolve_parent_modules()
    assert used == "fake_hermes"


def test_resolve_parent_modules_real_src_missing_falls_back(monkeypatch, tmp_path):
    """The default missing checkout falls back when no source is explicit."""
    monkeypatch.delenv("ENFOLD_HERMES_SRC", raising=False)
    monkeypatch.setattr(mcp_provider, "DEFAULT_HERMES_SRC", str(tmp_path / "does-not-exist"))
    used = mcp_provider.resolve_parent_modules()
    assert used == "fake_hermes"


def test_explicit_missing_hermes_src_fails_closed(tmp_path):
    import subprocess

    missing_src = tmp_path / "does-not-exist"
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('mcpp', {str(_MCP_PROVIDER_PATH)!r})\n"
        "mcpp = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mcpp)\n"
        "mcpp.resolve_parent_modules(sys.argv[1])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(missing_src)],
        capture_output=True, text=True, timeout=15,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode != 0
    assert "explicit Hermes source" in result.stderr


def test_real_parent_failure_restores_touched_sys_modules(monkeypatch, tmp_path):
    src = tmp_path / "src"
    holo_dir = src / "plugins" / "memory" / "holographic"
    holo_dir.mkdir(parents=True)
    (holo_dir / "holographic.py").write_text("raise RuntimeError('broken real parent')\n")
    (holo_dir / "retrieval.py").write_text("class FactRetriever:\n    pass\n")
    (holo_dir / "store.py").write_text("class MemoryStore:\n    pass\n")
    (holo_dir / "__init__.py").write_text("class HolographicMemoryProvider:\n    pass\n")

    removed = [
        "agent", "agent.memory_manager", "plugins", "plugins.memory",
        "plugins.memory.holographic", "plugins.memory.holographic.holographic",
        "plugins.memory.holographic.retrieval", "plugins.memory.holographic.store",
    ]
    for name in removed:
        monkeypatch.delitem(sys.modules, name, raising=False)

    with pytest.raises(RuntimeError, match="explicit Hermes source"):
        mcp_provider.resolve_parent_modules(str(src))

    for name in removed:
        assert name not in sys.modules


def test_build_provider_canonicalizes_db_path(tmp_path, monkeypatch):
    monkeypatch.delenv("ENFOLD_HERMES_SRC", raising=False)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    os.symlink(real_dir, link_dir)

    provider = mcp_provider.build_provider(
        db_path=str(link_dir / "facts.db"),
        embedding_backend="fake",
        hrr_dim=64,
    )
    try:
        assert os.path.realpath(str(provider._store.db_path)) == os.path.realpath(
            str(real_dir / "facts.db")
        )
        assert str(provider._store.db_path) == os.path.realpath(str(real_dir / "facts.db"))
    finally:
        provider.shutdown()


def test_read_only_provider_skips_mutating_initialize(tmp_path, monkeypatch):
    monkeypatch.delenv("ENFOLD_HERMES_SRC", raising=False)
    db_path = tmp_path / "facts.db"
    writer = mcp_provider.build_provider(
        db_path=str(db_path),
        embedding_backend="fake",
        hrr_dim=64,
    )
    try:
        writer._store.add_fact("Alex Rivera keeps the Springfield runbook in Git", category="tool")
    finally:
        writer.shutdown()

    def fail_initialize(*args, **kwargs):
        raise AssertionError("read-only startup must not call provider.initialize")

    monkeypatch.setattr(mcp_provider, "_initialize_with_retry", fail_initialize)
    reader = mcp_provider.build_provider(
        db_path=str(db_path),
        embedding_backend="fake",
        hrr_dim=64,
        read_only=True,
    )
    try:
        assert reader._queue_worker is None
        assert reader._backfill_thread is None
        assert reader.search("Springfield runbook", limit=5, bump=False)
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            reader._store._conn.execute(
                "INSERT INTO facts (content, category) VALUES (?, ?)",
                ("write should fail", "general"),
            )
    finally:
        reader.shutdown()


def test_explicit_env_hermes_src_import_failure_fails_closed(tmp_path):
    import subprocess

    src = tmp_path / "src"
    holo_dir = src / "plugins" / "memory" / "holographic"
    holo_dir.mkdir(parents=True)
    (holo_dir / "holographic.py").write_text("raise RuntimeError('boom')\n")
    (holo_dir / "retrieval.py").write_text("class FactRetriever:\n    pass\n")
    (holo_dir / "store.py").write_text("class MemoryStore:\n    pass\n")
    (holo_dir / "__init__.py").write_text("class HolographicMemoryProvider:\n    pass\n")

    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('mcpp', {str(_MCP_PROVIDER_PATH)!r})\n"
        "mcpp = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mcpp)\n"
        "mcpp.resolve_parent_modules()\n"
    )
    env = os.environ.copy()
    env["ENFOLD_HERMES_SRC"] = str(src)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=15,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
    )
    assert result.returncode != 0
    assert "explicit Hermes source" in result.stderr


def test_resolve_parent_modules_explicit_real_src_missing_fails(monkeypatch, tmp_path):
    """A configured nonexistent ENFOLD_HERMES_SRC fails closed."""
    monkeypatch.setenv("ENFOLD_HERMES_SRC", str(tmp_path / "does-not-exist"))
    with pytest.raises(RuntimeError, match="explicit Hermes source"):
        mcp_provider.resolve_parent_modules()


def test_build_provider_uses_configurable_db_path(tmp_path, monkeypatch):
    monkeypatch.delenv("ENFOLD_HERMES_SRC", raising=False)
    db_path = tmp_path / "facts.db"
    provider = mcp_provider.build_provider(
        db_path=str(db_path),
        embedding_backend="fake",
        hrr_dim=64,
    )
    try:
        assert db_path.exists()
    finally:
        provider.shutdown()


def test_build_provider_defaults_match_live_ollama_identity(tmp_path, monkeypatch):
    """Defaults mirror the live box: ollama backend, embeddinggemma model, auto prefix."""
    monkeypatch.delenv("ENFOLD_HERMES_SRC", raising=False)
    provider = mcp_provider.build_provider(
        db_path=str(tmp_path / "facts.db"),
        embedding_backend="fake",
        hrr_dim=64,
    )
    try:
        # "fake" swaps only the embedder implementation for tests; the
        # backend name stays "ollama" so the embedding identity string still
        # matches production (see mcp_provider.build_provider).
        assert provider._embedding_backend == "ollama"
    finally:
        provider.shutdown()

    # The real defaults (when embedding_backend is left at its own default)
    # match the live box's identity components.
    assert mcp_provider.DEFAULT_OLLAMA_MODEL == "embeddinggemma:latest"
    assert mcp_provider.DEFAULT_OLLAMA_URL == "http://localhost:11434"
    assert mcp_provider.DEFAULT_PREFIX_POLICY == "auto"


def test_build_provider_sets_busy_timeout(tmp_path, monkeypatch):
    """The store connection must have a busy_timeout for concurrent writers."""
    monkeypatch.delenv("ENFOLD_HERMES_SRC", raising=False)
    db_path = tmp_path / "facts.db"
    provider = mcp_provider.build_provider(
        db_path=str(db_path),
        embedding_backend="fake",
        hrr_dim=64,
        busy_timeout_ms=5000,
    )
    try:
        row = provider._store._conn.execute("PRAGMA busy_timeout").fetchone()
        assert row[0] == 5000
    finally:
        provider.shutdown()


def test_build_provider_journal_mode_is_wal(tmp_path, monkeypatch):
    monkeypatch.delenv("ENFOLD_HERMES_SRC", raising=False)
    db_path = tmp_path / "facts.db"
    provider = mcp_provider.build_provider(
        db_path=str(db_path),
        embedding_backend="fake",
        hrr_dim=64,
    )
    try:
        row = provider._store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0].lower() == "wal"
    finally:
        provider.shutdown()


def test_check_journal_mode_reads_existing_db(tmp_path):
    """Read-only journal-mode check against an arbitrary sqlite file."""
    db_path = tmp_path / "existing.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.commit()
    conn.close()

    mode = mcp_provider.check_journal_mode(str(db_path))
    assert mode.lower() == "wal"


_REAL_HERMES_SRC = os.environ.get(
    "ENFOLD_HERMES_SRC", str(Path.home() / "hermes-migration-stage" / "src")
)
_REAL_HOLO_DIR = Path(_REAL_HERMES_SRC) / "plugins" / "memory" / "holographic"


@pytest.mark.skipif(
    not (_REAL_HOLO_DIR / "store.py").exists(),
    reason=f"real hermes checkout not found at {_REAL_HERMES_SRC}",
)
def test_resolve_parent_modules_real_checkout_resolves_to_real(monkeypatch):
    """When a real checkout is configured and present, it wins over the fallback.

    Runs in-process against whatever the module-level cache currently holds;
    the important assertion is that it names "real" the source it found,
    not "fake_hermes", when ENFOLD_HERMES_SRC points at a genuine checkout.
    This exercises the same code path used on the live box.
    """
    monkeypatch.setenv("ENFOLD_HERMES_SRC", _REAL_HERMES_SRC)
    # Run in a subprocess so the module-level sys.modules cache from earlier
    # tests in this file (which install the fake stubs) cannot mask a real
    # resolution failure.
    import subprocess

    code = (
        "import sys, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('mcpp', {str(_MCP_PROVIDER_PATH)!r})\n"
        "mcpp = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mcpp)\n"
        "print(mcpp.resolve_parent_modules())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "real"
