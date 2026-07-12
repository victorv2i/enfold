"""Black-box MCP stdio coverage against the packaged Unix-socket daemon."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import socket
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from enfold.client import ClientConfig, EnfoldClient
from enfold.protocol import ClientContext
from enfold.schema import migrate


_ROOT = Path(__file__).resolve().parents[1]


def _server_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    database = tmp_path / "memory.db"
    connection = sqlite3.connect(database)
    migrate(connection)
    connection.close()
    socket_path = tmp_path / "enfold.sock"
    config = tmp_path / "server.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(database),
                "socket_path": str(socket_path),
                "grants": {"e2e-client": ["private"]},
                "cleanup_stale_socket": True,
                "retrieval": {
                    "mode": "ci",
                    "allow_nonproduction": True,
                    "dimensions": 64,
                },
                "client_timeout": 0.5,
                "shutdown_timeout": 1.0,
            }
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    return database, socket_path, config


def _wait_for_socket(process: subprocess.Popen[str], socket_path: Path) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if socket_path.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.settimeout(0.1)
                probe.connect(os.fspath(socket_path))
            except OSError:
                pass
            else:
                return
            finally:
                probe.close()
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            pytest.fail(f"enfold-server exited before binding its socket: {stderr}")
        time.sleep(0.02)
    pytest.fail("enfold-server did not bind its socket within 5 seconds")


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.fixture
def daemon(tmp_path):
    database, socket_path, config = _server_config(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "enfold.server", "--config", str(config), "run"],
        cwd=_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_for_socket(process, socket_path)
    try:
        yield database, socket_path, config, process
    finally:
        _stop(process)


def _mcp_process(socket_path: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "enfold.mcp_stdio",
            "--socket-path",
            str(socket_path),
            "--client-id",
            "e2e-client",
            "--surface",
            "pytest",
            "--agent-id",
            "pytest",
            "--session-id",
            "mcp-e2e",
        ],
        cwd=_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _send(process: subprocess.Popen[str], message: dict[str, object]) -> None:
    assert process.stdin is not None
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()


def _receive(process: subprocess.Popen[str]) -> dict[str, object]:
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], 5)
    if not ready:
        stderr = (
            process.stderr.read()
            if process.stderr is not None and process.poll() is not None
            else ""
        )
        pytest.fail(f"MCP stdio server did not respond within 5 seconds: {stderr}")
    line = process.stdout.readline()
    assert line, "MCP stdio server closed stdout before responding"
    return json.loads(line)


def _initialize(process: subprocess.Popen[str]) -> None:
    from mcp.types import LATEST_PROTOCOL_VERSION

    _send(
        process,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "enfold-test", "version": "1"},
            },
        },
    )
    response = _receive(process)
    assert response["id"] == 1
    assert "result" in response
    _send(
        process,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    )


def _tool_result(response: dict[str, object]) -> dict[str, object]:
    result = response["result"]
    assert isinstance(result, dict)
    assert result.get("isError") is not True
    content = result["content"]
    assert isinstance(content, list) and content
    text = content[0]["text"]
    assert isinstance(text, str)
    return json.loads(text)


def test_mcp_stdio_subprocess_writes_and_searches_through_real_daemon(daemon):
    _database, socket_path, _config, _server = daemon
    process = _mcp_process(socket_path)
    try:
        _initialize(process)
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "memory_write",
                    "arguments": {
                        "idempotency_key": "mcp-e2e-1",
                        "content": "The MCP subprocess wrote this durable memory.",
                        "source_type": "integration_test",
                    },
                },
            },
        )
        written = _tool_result(_receive(process))
        assert written["outcome"] == "inserted"

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "memory_search",
                    "arguments": {"query": "MCP subprocess durable memory"},
                },
            },
        )
        found = _tool_result(_receive(process))
        assert [row["fact_id"] for row in found["facts"]] == [written["fact_id"]]
    finally:
        _stop(process)


def test_mcp_stdio_subprocess_returns_parse_error_for_malformed_frame(daemon):
    _database, socket_path, _config, _server = daemon
    process = _mcp_process(socket_path)
    try:
        assert process.stdin is not None
        process.stdin.write("not valid json\n")
        process.stdin.flush()
        response = _receive(process)
        assert response["method"] == "notifications/message"
        assert response["params"]["level"] == "error"
        assert process.poll() is None
        _initialize(process)
    finally:
        _stop(process)


def test_mcp_stdio_subprocess_exits_when_client_disconnects(daemon):
    _database, socket_path, _config, _server = daemon
    process = _mcp_process(socket_path)
    _initialize(process)
    assert process.stdin is not None
    process.stdin.close()
    assert process.wait(timeout=5) == 0


def _socket_client(socket_path: Path, session_id: str) -> EnfoldClient:
    return EnfoldClient(
        ClientConfig(
            socket_path=socket_path,
            context=ClientContext(
                client_id="e2e-client",
                surface="pytest",
                agent_id="pytest",
                session_id=session_id,
                access_scopes=("private",),
            ),
        )
    )


def test_daemon_recovers_durable_writes_after_kill_during_load(daemon):
    database, socket_path, config, first = daemon
    completed: list[int] = []
    started = threading.Event()
    stop = threading.Event()

    def write_load() -> None:
        client = _socket_client(socket_path, "before-restart")
        for index in range(100):
            if stop.is_set():
                return
            started.set()
            try:
                result = client.request(
                    "memory.write",
                    {
                        "idempotency_key": f"restart-load-{index}",
                        "content": f"Crash durability load memory {index}",
                        "source_type": "integration_test",
                    },
                )
            except Exception:
                return
            completed.append(int(result["fact_id"]))
            if len(completed) >= 8:
                return

    writer = threading.Thread(target=write_load)
    writer.start()
    assert started.wait(2)
    deadline = time.monotonic() + 3
    while len(completed) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert completed, "write load did not reach the daemon before it was killed"
    first.kill()
    assert first.wait(timeout=5) != 0
    stop.set()
    writer.join(timeout=5)
    assert not writer.is_alive()

    restarted = subprocess.Popen(
        [sys.executable, "-m", "enfold.server", "--config", str(config), "run"],
        cwd=_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_for_socket(restarted, socket_path)
    try:
        new_client = _socket_client(socket_path, "after-restart")
        health = new_client.request("health")
        assert health["status"] == "ok"
        found = new_client.request(
            "memory.search", {"query": "Crash durability load memory", "limit": 100}
        )
        returned_ids = {int(row["fact_id"]) for row in found["facts"]}
        assert set(completed) <= returned_ids
        connection = sqlite3.connect(database)
        try:
            assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        finally:
            connection.close()
    finally:
        _stop(restarted)
