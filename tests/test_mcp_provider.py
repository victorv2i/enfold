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
    """A configured but nonexistent ENFOLD_HERMES_SRC falls back, doesn't raise."""
    monkeypatch.setenv("ENFOLD_HERMES_SRC", str(tmp_path / "does-not-exist"))
    used = mcp_provider.resolve_parent_modules()
    assert used == "fake_hermes"


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
