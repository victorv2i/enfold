"""Tests for the enfold MCP tool server (enfold.mcp_server).

Exercises tool registration, search parity with provider.search(), the add
path routing through the same dedup gate + supersession as the live write
path, source tagging, supersession, explain/history, concurrent writers, and
read-only mode.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MCP_PROVIDER_PATH = _REPO_ROOT / "enfold" / "mcp_provider.py"
_MCP_SERVER_PATH = _REPO_ROOT / "enfold" / "mcp_server.py"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mcp_provider(monkeypatch):
    monkeypatch.delenv("ENFOLD_HERMES_SRC", raising=False)
    return _load("_hp_test_mcp_provider", _MCP_PROVIDER_PATH)


@pytest.fixture()
def mcp_server_mod(mcp_provider):
    return _load("_hp_test_mcp_server", _MCP_SERVER_PATH)


@pytest.fixture()
def provider(mcp_provider, tmp_path):
    p = mcp_provider.build_provider(
        db_path=str(tmp_path / "facts.db"),
        embedding_backend="fake",
        hrr_dim=64,
    )
    yield p
    p.shutdown()


def run(coro):
    return asyncio.run(coro)


def _call(server, name, args):
    return run(server.call_tool(name, args))


def _tool_names(server):
    tools = run(server.list_tools())
    return {t.name for t in tools}


def test_direct_invocation_help_does_not_crash():
    """`python enfold/mcp_server.py --help` must work standalone.

    This is the documented launch command; it must be run by file path, not
    `python -m enfold.mcp_server` (see the module docstring): a
    package-style import runs enfold/__init__.py first, which can
    silently resolve plugins.memory.holographic against an unrelated Hermes
    install already on sys.path (this box has one), before mcp_server's own
    ENFOLD_HERMES_SRC resolution ever gets a chance to run.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, str(_MCP_SERVER_PATH), "--help"],
        capture_output=True, text=True, timeout=15, cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "memory_store.db" in result.stdout


def test_direct_invocation_resolves_configured_hermes_src(tmp_path):
    """The file-path launch resolves ENFOLD_HERMES_SRC, unlike `-m` import.

    Regression guard: builds a minimal fake checkout at a custom path,
    points ENFOLD_HERMES_SRC at it, and confirms mcp_server.py (run by
    file path, as documented) picks it up rather than any hermes install
    that happens to already be importable on this box.
    """
    import subprocess

    holo_dir = tmp_path / "src" / "plugins" / "memory" / "holographic"
    holo_dir.mkdir(parents=True)
    (holo_dir / "store.py").write_text("class MemoryStore:\n    pass\n")
    (holo_dir / "retrieval.py").write_text("class FactRetriever:\n    pass\n")

    code = (
        "import sys, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('mcpp', {str(_MCP_PROVIDER_PATH)!r})\n"
        "mcpp = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mcpp)\n"
        "print(mcpp._real_parent_available(sys.argv[1]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "src")],
        capture_output=True, text=True, timeout=15, cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_registers_all_tools_in_readwrite_mode(mcp_server_mod, provider):
    server = mcp_server_mod.build_server(provider, read_only=False)
    names = _tool_names(server)
    assert names == {
        "memory_search",
        "memory_add",
        "memory_supersede",
        "memory_explain",
        "memory_history",
    }


def test_read_only_mode_registers_only_read_tools(mcp_server_mod, provider):
    server = mcp_server_mod.build_server(provider, read_only=True)
    names = _tool_names(server)
    assert names == {"memory_search", "memory_explain", "memory_history"}
    assert "memory_add" not in names
    assert "memory_supersede" not in names


def _content_json(result):
    """Normalize a FastMCP call_tool() return value to the tool's plain dict.

    call_tool() returns (content_blocks, structured_result). structured_result
    is the tool's dict wrapped under a "result" key when the return type
    annotation is a bare ``Dict[str, Any]`` (no named schema), so unwrap that;
    otherwise fall back to parsing the first text content block.
    """
    if isinstance(result, tuple):
        blocks, structured = result
        if isinstance(structured, dict) and "result" in structured and len(structured) == 1:
            return structured["result"]
        if structured is not None:
            return structured
        result = blocks
    if isinstance(result, dict):
        return result
    # Sequence of content blocks: take the first text block.
    text = result[0].text
    return json.loads(text)


def test_memory_search_parity_with_provider_search(mcp_server_mod, provider):
    provider._store.add_fact(
        "Alex Rivera prefers Postgres over MySQL for Springfield projects",
        category="tool",
    )
    provider._store.add_fact(
        "The Springfield deploy pipeline runs nightly", category="general"
    )
    server = mcp_server_mod.build_server(provider, read_only=False)

    direct = provider.search("Postgres preference", limit=5, bump=False)
    tool_result = _content_json(_call(server, "memory_search", {
        "query": "Postgres preference", "limit": 5,
    }))

    assert [r["fact_id"] for r in tool_result["results"]] == [
        r["fact_id"] for r in direct
    ]
    assert [r["content"] for r in tool_result["results"]] == [
        r["content"] for r in direct
    ]


def test_memory_search_rejects_non_string_query_before_provider_call(mcp_server_mod, provider):
    def fail_search(*args, **kwargs):
        raise AssertionError("invalid query reached provider.search")

    provider.search = fail_search
    server = mcp_server_mod.build_server(provider, read_only=False)
    result = _content_json(_call(server, "memory_search", {
        "query": 123, "limit": 5,
    }))
    assert result["error"] == "invalid query: must be a string"


def test_memory_search_clamps_limit_before_provider_call(mcp_server_mod, provider):
    seen = {}

    def fake_search(query, limit=10, bump=True, **kwargs):
        seen["limit"] = limit
        return []

    provider.search = fake_search
    server = mcp_server_mod.build_server(provider, read_only=False)
    result = _content_json(_call(server, "memory_search", {
        "query": "Springfield", "limit": 500,
    }))
    assert result == {"results": [], "count": 0}
    assert seen["limit"] == 50


def test_memory_add_requires_source(mcp_server_mod, provider):
    """source has no default, so a missing value is rejected before the tool
    body even runs (FastMCP validates required args against the signature)."""
    server = mcp_server_mod.build_server(provider, read_only=False)
    with pytest.raises(Exception, match="(?i)source"):
        _call(server, "memory_add", {
            "content": "Alex Rivera's team uses Skylark CI for Springfield builds",
            "category": "tool",
        })


def test_memory_add_tags_source_and_persists(mcp_server_mod, provider):
    server = mcp_server_mod.build_server(provider, read_only=False)
    result = _content_json(_call(server, "memory_add", {
        "content": "Alex Rivera's team uses Skylark CI for Springfield builds",
        "category": "tool",
        "source": "claude-code",
    }))
    assert result["status"] == "added"
    fact_id = result["fact_id"]
    row = provider._store._conn.execute(
        "SELECT tags FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    assert "source:claude-code" in row["tags"]


def test_memory_add_strips_spoofed_source_tags(mcp_server_mod, provider):
    server = mcp_server_mod.build_server(provider, read_only=False)
    result = _content_json(_call(server, "memory_add", {
        "content": "Alex Rivera's Springfield deploy owner is Morgan",
        "category": "tool",
        "source": "codex",
        "tags": "owner, source:claude-code,source:other",
    }))
    assert result["status"] == "added"
    row = provider._store._conn.execute(
        "SELECT tags FROM facts WHERE fact_id = ?", (result["fact_id"],)
    ).fetchone()
    assert row["tags"] == "owner,source:codex"


def test_memory_add_rejects_invalid_args_before_provider_call(mcp_server_mod, provider):
    def fail_find(*args, **kwargs):
        raise AssertionError("invalid content reached provider")

    provider._find_near_duplicate = fail_find
    server = mcp_server_mod.build_server(provider, read_only=False)
    result = _content_json(_call(server, "memory_add", {
        "content": "   ",
        "source": "codex",
    }))
    assert result["error"] == "invalid content: must not be blank"

    result = _content_json(_call(server, "memory_add", {
        "content": "valid content",
        "category": "x" * 129,
        "source": "codex",
    }))
    assert result["error"] == "invalid category: exceeds 128 characters"


def test_memory_add_rejects_invalid_source(mcp_server_mod, provider):
    server = mcp_server_mod.build_server(provider, read_only=False)
    result = _content_json(_call(server, "memory_add", {
        "content": "Alex Rivera's team uses Skylark CI for Springfield builds",
        "source": "not-a-real-source",
    }))
    assert "error" in result


def test_memory_add_routes_through_dedup_gate(mcp_provider, mcp_server_mod, tmp_path):
    """A low-lexical-overlap paraphrase is caught by the semantic dedup path
    (dense cosine), same gate the live Hermes write path uses. Force the
    fake embedder to treat the paraphrase as identical in meaning (a real
    embedder would do this on its own; FakeEmbedder needs a table since it
    otherwise hashes distinct strings to unrelated vectors).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fake_hermes

    original = "Alex Rivera prefers Postgres over MySQL for Springfield"
    paraphrase = "Alex Rivera always reaches for Postgres instead of MySQL for Springfield"
    shared_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    embedder = fake_hermes.FakeEmbedder(table={original: shared_vec, paraphrase: shared_vec})

    provider = mcp_provider.build_provider(
        db_path=str(tmp_path / "facts.db"), embedding_backend="fake", hrr_dim=64,
        embedding_prefix_policy="none",
    )
    provider._embedder = embedder  # swap in the shared-vector embedder post-init
    provider._fake_embedder = embedder

    server = mcp_server_mod.build_server(provider, read_only=False)
    first = _content_json(_call(server, "memory_add", {
        "content": original, "category": "tool", "source": "codex",
    }))
    assert first["status"] == "added"

    duplicate = _content_json(_call(server, "memory_add", {
        "content": paraphrase, "category": "tool", "source": "codex",
    }))
    assert duplicate["status"] == "deduped"
    assert duplicate["fact_id"] == first["fact_id"]

    count = provider._store._conn.execute(
        "SELECT COUNT(*) AS c FROM facts"
    ).fetchone()["c"]
    assert count == 1
    provider.shutdown()


def test_memory_add_value_update_supersedes(mcp_server_mod, provider):
    server = mcp_server_mod.build_server(provider, read_only=False)
    first = _content_json(_call(server, "memory_add", {
        "content": "The Springfield API rate limit is 100 requests per minute",
        "category": "general",
        "source": "claude-code",
    }))
    updated = _content_json(_call(server, "memory_add", {
        "content": "The Springfield API rate limit is 200 requests per minute",
        "category": "general",
        "source": "claude-code",
    }))
    assert updated["status"] == "added"
    assert updated["fact_id"] != first["fact_id"]

    history = _content_json(_call(server, "memory_history", {
        "fact_id": updated["fact_id"],
    }))
    fact_ids = [h["fact_id"] for h in history["history"]]
    assert first["fact_id"] in fact_ids
    assert updated["fact_id"] in fact_ids


def test_memory_supersede_tool(mcp_server_mod, provider):
    server = mcp_server_mod.build_server(provider, read_only=False)
    first = _content_json(_call(server, "memory_add", {
        "content": "Alex Rivera's Springfield office is on the third floor",
        "category": "general",
        "source": "other",
    }))
    result = _content_json(_call(server, "memory_supersede", {
        "old_fact_id": first["fact_id"],
        "new_content": "Alex Rivera's Springfield office is on the fifth floor",
        "source": "other",
    }))
    assert result["status"] == "superseded"
    new_id = result["fact_id"]

    history = _content_json(_call(server, "memory_history", {"fact_id": new_id}))
    fact_ids = {h["fact_id"] for h in history["history"]}
    assert first["fact_id"] in fact_ids
    assert new_id in fact_ids

    active = provider.search("Springfield office floor", limit=5, bump=False)
    active_ids = {r["fact_id"] for r in active}
    assert first["fact_id"] not in active_ids
    assert new_id in active_ids


def test_memory_supersede_rejects_unknown_old_fact_before_insert(mcp_server_mod, provider):
    server = mcp_server_mod.build_server(provider, read_only=False)
    result = _content_json(_call(server, "memory_supersede", {
        "old_fact_id": 999999,
        "new_content": "Alex Rivera's Springfield office is on the ninth floor",
        "source": "other",
    }))
    assert result["error"] == "invalid old_fact_id: active fact not found"
    count = provider._store._conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"]
    assert count == 0


def test_memory_supersede_rejects_non_positive_old_fact_id(mcp_server_mod, provider):
    server = mcp_server_mod.build_server(provider, read_only=False)
    result = _content_json(_call(server, "memory_supersede", {
        "old_fact_id": 0,
        "new_content": "Alex Rivera's Springfield office is on the ninth floor",
        "source": "other",
    }))
    assert result["error"] == "invalid old_fact_id: must be a positive integer"


def test_memory_explain_matches_provider_explain_search(mcp_server_mod, provider):
    provider._store.add_fact(
        "Alex Rivera's Springfield build uses Skylark packaging", category="tool"
    )
    server = mcp_server_mod.build_server(provider, read_only=False)
    direct = provider.explain_search("Skylark packaging", limit=5)
    via_tool = _content_json(_call(server, "memory_explain", {
        "query": "Skylark packaging", "limit": 5,
    }))
    assert via_tool["breakdown"] == direct


def test_memory_history_empty_for_unknown_fact(mcp_server_mod, provider):
    server = mcp_server_mod.build_server(provider, read_only=False)
    result = _content_json(_call(server, "memory_history", {"fact_id": 999999}))
    assert result["history"] == []


def test_readonly_server_rejects_write_tool_call(mcp_server_mod, provider):
    server = mcp_server_mod.build_server(provider, read_only=True)
    with pytest.raises(Exception):
        _call(server, "memory_add", {
            "content": "Alex Rivera's Springfield office is remote-first",
            "source": "codex",
        })


def test_write_lock_uses_canonical_db_path(mcp_server_mod, tmp_path):
    real_db = tmp_path / "real-facts.db"
    real_db.touch()
    db_path = tmp_path / "link-facts.db"
    os.symlink(real_db, db_path)

    with mcp_server_mod._cross_process_write_lock(str(db_path)):
        pass

    assert (tmp_path / "real-facts.db.mcp-write.lock").exists()
    assert not (tmp_path / "link-facts.db.mcp-write.lock").exists()


def test_concurrent_writers_no_lost_writes(mcp_provider, tmp_path):
    """Two separate OS processes hammering memory_add against the same
    scratch DB: every distinct fact must land, and PRAGMA quick_check must
    stay clean.

    Runs each writer in its own subprocess (see _concurrent_writer_worker.py)
    rather than a thread in this process: the parent-module resolution
    (real hermes checkout vs fake_hermes) is cached in a single shared
    sys.modules slot for the whole pytest session (test_real_parent_
    equivalence.py deliberately does the same real-checkout install), so an
    in-process thread here could silently inherit whatever another test file
    in this session already installed instead of the fake stubs this test
    asks for. Separate processes also more faithfully model the real
    deployment shape: two independent MCP server processes sharing one DB.
    """
    import subprocess

    db_path = str(tmp_path / "concurrent.db")
    n_per_thread = 25
    worker_script = str(Path(__file__).resolve().parent / "_concurrent_writer_worker.py")

    procs = [
        subprocess.Popen(
            [sys.executable, worker_script, db_path, str(tid), str(n_per_thread)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(_REPO_ROOT),
        )
        for tid in range(2)
    ]
    results = [p.communicate(timeout=90) for p in procs]

    for (stdout, stderr), p in zip(results, procs):
        assert p.returncode == 0 and "OK" in stdout, (stdout, stderr)

    check_provider = mcp_provider.build_provider(
        db_path=db_path, embedding_backend="fake", hrr_dim=64, busy_timeout_ms=5000,
    )
    try:
        rows = check_provider._store._conn.execute(
            "SELECT COUNT(*) AS c FROM facts WHERE content LIKE 'Alex Rivera fact%'"
        ).fetchone()
        assert rows["c"] == 2 * n_per_thread

        quick_check = check_provider._store._conn.execute(
            "PRAGMA quick_check"
        ).fetchone()[0]
        assert quick_check == "ok"
    finally:
        check_provider.shutdown()
