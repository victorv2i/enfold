"""Subprocess worker for test_concurrent_writers_no_lost_writes.

Runs in its own process (own sys.modules) so it can never share parent-module
state (real hermes checkout vs fake_hermes) with any other test module in the
same pytest session; this is the same isolation concern
test_real_parent_equivalence.py already documents for the plugin package.

Usage: python _concurrent_writer_worker.py <db_path> <thread_id> <n_facts>
Writes one line "OK" to stdout on success, or "ERROR: <msg>" on failure.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    db_path, thread_id, n_facts = sys.argv[1], sys.argv[2], int(sys.argv[3])

    mcp_provider = _load("_worker_mcp_provider", _REPO_ROOT / "enfold" / "mcp_provider.py")
    mcp_server = _load("_worker_mcp_server", _REPO_ROOT / "enfold" / "mcp_server.py")

    provider = mcp_provider.build_provider(
        db_path=db_path, embedding_backend="fake", hrr_dim=64, busy_timeout_ms=5000,
    )
    server = mcp_server.build_server(provider, read_only=False)

    async def run_all():
        for i in range(n_facts):
            content = f"Alex Rivera fact thread{thread_id} item{i} about Springfield build {thread_id}-{i}"
            blocks, structured = await server.call_tool("memory_add", {
                "content": content, "category": "general", "source": "codex",
            })
            result = structured.get("result", structured) if isinstance(structured, dict) else json.loads(blocks[0].text)
            if "error" in result:
                print(f"ERROR: {result['error']}")
                return 1
        return 0

    try:
        rc = asyncio.run(run_all())
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        # _teardown_background() gives the extraction/backfill daemon
        # threads only a brief 0.5s join before shutdown() proceeds to close
        # the connection (documented tradeoff: a leaked connection beats
        # closing one under an active writer). A short settle wait here
        # lets those threads reach their next stop-check on their own
        # before the interpreter tears down, avoiding a use-after-free style
        # crash in CPython's sqlite3 statement-cache finalizer if a daemon
        # thread is still mid-statement when the process exits.
        time.sleep(0.2)
        provider.shutdown()
        time.sleep(0.1)

    if rc == 0:
        print("OK")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
