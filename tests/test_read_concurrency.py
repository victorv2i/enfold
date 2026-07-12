from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

import enfold.server as server_module
from enfold.client import ClientConfig, EnfoldClient
from enfold.protocol import ClientContext
from enfold.read_pool import LRUQueryEmbedder, PerRequestReadHandler
from enfold.protocol import Request
from enfold.schema import migrate
from enfold.server import (
    ExtractionConfig,
    RetrievalConfig,
    ServerApplication,
    ServerConfig,
    _open_existing_v1,
    load_config,
)


def _client(socket_path):
    return EnfoldClient(
        ClientConfig(
            socket_path,
            ClientContext(
                client_id="concurrency-test",
                surface="client-a",
                agent_id="client-a",
                session_id="read-concurrency",
                access_scopes=("private",),
            ),
            request_timeout=2.0,
        )
    )


def test_slow_query_embedding_does_not_block_write_or_second_read(
    tmp_path, monkeypatch
):
    socket_path = tmp_path / "enfold.sock"
    database_path = tmp_path / "memory.db"
    conn = sqlite3.connect(database_path)
    migrate(conn)
    conn.close()
    config_path = tmp_path / "server.json"
    config_path.write_text(
        json.dumps(
            {
                "database_path": str(database_path),
                "socket_path": str(socket_path),
                "grants": {"concurrency-test": ["private"]},
                "retrieval": {
                    "mode": "ci",
                    "allow_nonproduction": True,
                    "dimensions": 64,
                },
                "client_timeout": 2.0,
                "shutdown_timeout": 2.0,
            }
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)
    embedding_started = threading.Event()
    release_embedding = threading.Event()
    completed: dict[str, tuple[float, object]] = {}
    opened_reads = []
    opened_reads_lock = threading.Lock()

    class SlowEmbedder:
        def embed(self, query):
            embedding_started.set()
            assert release_embedding.wait(2.0)
            return (query,)

    embedder = SlowEmbedder()

    class SlowRetriever:
        metadata = {"test": "slow-embedding"}

        def search(self, query, **_kwargs):
            embedder.embed(query)
            return []

    def retriever_factory(_config, _connection, _query_embedder=None):
        return lambda _conn, _scopes: SlowRetriever()

    original_open = server_module._open_existing_v1

    def tracked_open(config, *, read_only=False):
        connection = original_open(config, read_only=read_only)
        if read_only:
            with opened_reads_lock:
                opened_reads.append(connection)
        return connection

    monkeypatch.setattr(server_module, "_retriever_factory", retriever_factory)
    monkeypatch.setattr(server_module, "_open_existing_v1", tracked_open)
    application = ServerApplication(load_config(config_path))
    application.daemon.start()

    def request(name, method, params=None):
        started = time.monotonic()
        result = _client(socket_path).request(method, params)
        completed[name] = (time.monotonic() - started, result)

    slow = threading.Thread(
        target=request,
        args=("slow", "memory.search", {"query": "deliberately slow"}),
    )
    slow.start()
    try:
        assert embedding_started.wait(1.0)
        write = threading.Thread(
            target=request,
            args=(
                "write",
                "memory.write",
                {
                    "idempotency_key": "concurrent-write",
                    "content": "A write concurrent with slow embedding",
                    "source_type": "test",
                },
            ),
        )
        read = threading.Thread(
            target=request,
            args=("read", "memory.conflicts", {}),
        )
        write.start()
        read.start()
        write.join(0.75)
        read.join(0.75)

        assert not write.is_alive(), "a slow read blocked a concurrent write"
        assert not read.is_alive(), "a slow read blocked a second read"
        assert completed["write"][0] < 0.75
        assert completed["read"][0] < 0.75
        assert slow.is_alive()
        assert len(opened_reads) == 2
        assert opened_reads[0] is not opened_reads[1]
    finally:
        release_embedding.set()
        slow.join(2.0)
        application.close()


def test_read_only_connection_rejects_insert(tmp_path):
    database_path = tmp_path / "memory.db"
    conn = sqlite3.connect(database_path)
    migrate(conn)
    conn.close()
    config = ServerConfig(
        database_path=database_path,
        socket_path=tmp_path / "enfold.sock",
        grants={"concurrency-test": ("private",)},
        retrieval=RetrievalConfig(mode="ci", allow_nonproduction=True),
        extraction=ExtractionConfig(),
    )

    read_conn = _open_existing_v1(config, read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            read_conn.execute(
                "INSERT INTO facts (content) VALUES (?)",
                ("must fail",),
            )
    finally:
        read_conn.close()


def test_query_embedding_cache_uses_exact_model_input_and_evicts_lru():
    class RecordingEmbedder:
        def __init__(self):
            self.calls = []

        def embed(self, text):
            self.calls.append(text)
            return (len(self.calls), 2.0)

    backend = RecordingEmbedder()
    cached = LRUQueryEmbedder(backend, model="nomic-v1", maxsize=2)

    assert cached.embed("  Shared   Memory ") == (1.0, 2.0)
    assert cached.embed("shared memory") == (2.0, 2.0)
    assert cached.embed("US") == (3.0, 2.0)
    assert cached.embed("us") == (4.0, 2.0)
    assert cached.embed("US") == (3.0, 2.0)
    assert backend.calls == ["  Shared   Memory ", "shared memory", "US", "us"]


@pytest.mark.parametrize(
    "method",
    ("memory.changes", "memory.timeline", "memory.entities", "memory.entity"),
)
def test_every_projection_method_routes_through_read_pool(method):
    opened = []
    handled = []

    class ReadConnection:
        def close(self):
            pass

    def mutation_handler(_context, _request):
        raise AssertionError("projection routed through mutation handler")

    def open_read_connection():
        connection = ReadConnection()
        opened.append(connection)
        return connection

    def build_read_handler(connection):
        def handler(_context, request):
            handled.append((connection, request.method))
            return request.method

        return handler

    router = PerRequestReadHandler(
        mutation_handler, open_read_connection, build_read_handler
    )
    context = ClientContext("test", "codex", "codex", "session")

    assert router(context, Request("request", method, {})) == method
    assert handled == [(opened[0], method)]
