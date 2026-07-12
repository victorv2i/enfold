from __future__ import annotations

import sqlite3

from memory_eval.daemon_arena import DaemonProtocolArenaProvider
from memory_eval.public_arena import EnfoldOfflineHybridProvider, load_public_arena, run_public_arena
from enfold.client import ClientConfig, EnfoldClient
from enfold.daemon import DaemonConfig, UnixJsonDaemon
from enfold.policy import MemoryPolicy
from enfold.protocol import ClientContext, Request
from enfold.schema import migrate
from enfold.service import EnfoldService


class ProviderClient:
    def __init__(self, provider):
        self.provider = provider
        self.calls = []

    def request(self, method, params=None):
        self.calls.append((method, dict(params or {})))
        if method == "memory.conflicts":
            conflicts = []
            for record in self.provider.search_conflicts("support launch", min_trust=0.0):
                conflict = next(
                    (item for item in conflicts if item["conflict_id"] == record["conflict_id"]),
                    None,
                )
                if conflict is None:
                    conflict = {"conflict_id": record["conflict_id"], "members": []}
                    conflicts.append(conflict)
                conflict["members"].append(record)
            return {"conflicts": conflicts}
        return {"facts": self.provider.search(**params)}


def test_daemon_protocol_arena_path_pins_quality_and_latency():
    arena = load_public_arena()
    with EnfoldOfflineHybridProvider(arena) as core:
        client = ProviderClient(core)
        daemon_provider = DaemonProtocolArenaProvider(client)
        run = run_public_arena(daemon_provider, arena=arena)

    assert run.summary["answerable"]["recall@1"] >= 0.8333
    assert run.summary["set_f1"] == 1.0
    assert run.summary["stale_leak@3"]["leaks"] == 0
    assert run.summary["abstention"]["true_abstain"] == 2
    assert len(client.calls) == 14
    assert {method for method, _ in client.calls} == {"memory.search", "memory.conflicts"}
    # A deliberately generous offline gate catches hangs/regressions without
    # pretending CI timing predicts production embedding latency.
    assert daemon_provider.metadata["latency_ms_p95"] < 250.0
    assert daemon_provider.metadata["transport_path"] == "versioned-daemon-protocol"


def test_real_unix_daemon_protocol_arena_path(tmp_path):
    arena = load_public_arena()
    with EnfoldOfflineHybridProvider(arena) as core:
        service = EnfoldService(
            core.connection,
            MemoryPolicy({"arena-client": ("private",)}),
        )
        daemon = UnixJsonDaemon(
            DaemonConfig(
                socket_path=tmp_path / "arena.sock",
                server_version="arena-test",
                client_timeout=1.0,
                shutdown_timeout=1.0,
            ),
            service,
        )
        daemon.start()
        try:
            client = EnfoldClient(ClientConfig(
                tmp_path / "arena.sock",
                ClientContext(
                    client_id="arena-client",
                    surface="arena",
                    agent_id="arena",
                    session_id="public-arena",
                ),
            ))
            provider = DaemonProtocolArenaProvider(client)
            run = run_public_arena(provider, arena=arena)
        finally:
            daemon.shutdown()

    assert run.summary["answerable"]["recall@1"] >= 0.8333
    assert run.summary["set_f1"] == 1.0
    assert run.summary["stale_leak@3"]["leaks"] == 0
    assert run.summary["abstention"]["true_abstain"] == 2
    assert provider.metadata["latency_ms_p95"] < 500.0


def test_real_unix_daemon_context_round_trip_is_bounded_and_cited(tmp_path):
    conn = sqlite3.connect(tmp_path / "context-arena.db", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    service = EnfoldService(conn, MemoryPolicy({"context-client": ("private",)}))
    context = ClientContext(
        client_id="context-client",
        surface="arena",
        agent_id="arena-agent",
        session_id="context-round-trip",
    )
    write = service.handle(
        context,
        Request(
            "context-write",
            "memory.write",
            {
                "idempotency_key": "context-write",
                "content": "Cedar runbook is the current fast lane reference.",
                "source_type": "synthetic_fixture",
            },
        ),
    )
    daemon = UnixJsonDaemon(
        DaemonConfig(
            socket_path=tmp_path / "context.sock",
            server_version="context-arena-test",
            client_timeout=1.0,
            shutdown_timeout=1.0,
        ),
        service,
    )
    daemon.start()
    try:
        client = EnfoldClient(ClientConfig(tmp_path / "context.sock", context))
        result = client.request(
            "memory.context",
            {"query": "Cedar fast lane reference", "token_budget": 128},
        )
    finally:
        daemon.shutdown()
        conn.close()

    assert result["abstained"] is False
    assert [fact["fact_id"] for fact in result["facts"]] == [write["fact_id"]]
    assert result["facts"][0]["attribution"]["agent_id"] == "arena-agent"
    assert result["token_estimate"]["used"] <= 128
    assert "fact:" + str(write["fact_id"]) in result["markdown"]
