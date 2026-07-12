from __future__ import annotations

import json
from pathlib import Path
import socket
import sqlite3
import time

import numpy as np
import pytest

from enfold.client import ClientConfig, EnfoldClient
from enfold.embeddings import embedding_to_bytes
from enfold.extraction_processor import ExtractedMemory
from enfold.ollama_artifact import ArtifactAttestation, ArtifactAttestationError
from enfold.protocol import ClientContext
from enfold.schema import migrate
from enfold.server import (
    DatabaseOwnershipError,
    ServerApplication,
    ServerConfigError,
    inspect_config,
    load_config,
    main,
)


_ARTIFACT_DIGEST = "sha256:" + "a" * 64


class FakeArtifactAttestor:
    def __init__(self, *, failure: Exception | None = None):
        self.failure = failure
        self.calls: list[tuple[str, str]] = []

    def attest(self, *, model: str, expected_digest: str) -> ArtifactAttestation:
        self.calls.append((model, expected_digest))
        if self.failure is not None:
            raise self.failure
        return ArtifactAttestation()


def _database(path: Path) -> Path:
    conn = sqlite3.connect(path)
    migrate(conn)
    conn.close()
    return path


def _config(tmp_path: Path, **changes) -> Path:
    data = {
        "database_path": str(_database(tmp_path / "memory.db")),
        "socket_path": str(tmp_path / "enfold.sock"),
        "grants": {
            "client-a-install": ["private", "work"],
            "client-b-install": ["private"],
            "hermes-install": ["private", "work"],
        },
        "retrieval": {
            "mode": "ci",
            "allow_nonproduction": True,
            "dimensions": 64,
        },
        "client_timeout": 0.5,
        "shutdown_timeout": 1.0,
    }
    data.update(changes)
    path = tmp_path / "server.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    path.chmod(0o600)
    return path


def _client(socket_path: Path) -> EnfoldClient:
    context = ClientContext(
        client_id="client-a-install",
        surface="client-a",
        agent_id="client-a",
        session_id="server-integration",
        access_scopes=("private", "work"),
    )
    return EnfoldClient(ClientConfig(socket_path, context))


def test_application_composes_service_health_and_write_in_temp_directory(tmp_path):
    config = load_config(_config(tmp_path))
    with ServerApplication(config) as application:
        application.daemon.start()
        client = _client(config.socket_path)
        health = client.request("health")
        assert health["status"] == "ok"
        assert health["schema_version"] == 1
        assert health["storage"] == "sqlite"
        assert health["retrieval"]["filter_before_dense_ranking"] is True
        assert health["retrieval"]["embedder_production_ready"] is False
        assert health["automatic_llm_extraction"] == {"status": "disabled"}
        result = client.request(
            "memory.write",
            {
                "idempotency_key": "server-test-1",
                "content": "Client A exercised the packaged Enfold daemon",
                "source_type": "integration_test",
                "scope": "private",
            },
        )
        assert result["outcome"] == "inserted"
        found = client.request(
            "memory.search", {"query": "packaged Enfold daemon"}
        )
        assert [row["fact_id"] for row in found["facts"]] == [result["fact_id"]]
    assert not config.socket_path.exists()


def test_database_sidecar_refuses_second_server_and_allows_clean_reacquire(tmp_path):
    config = load_config(_config(tmp_path))
    first = ServerApplication(config)
    lock_path = config.database_path.with_name(config.database_path.name + ".enfold.lock")
    try:
        assert lock_path.is_file()
        assert lock_path.stat().st_mode & 0o777 == 0o600
        with pytest.raises(DatabaseOwnershipError, match="another Enfold server"):
            ServerApplication(config)
    finally:
        first.close()

    # The stable sidecar inode persists, but releasing flock permits the next
    # server to acquire sole ownership without a split-lock unlink race.
    with ServerApplication(config):
        pass


def test_database_sidecar_symlink_is_refused(tmp_path):
    config = load_config(_config(tmp_path))
    target = tmp_path / "elsewhere"
    target.write_text("", encoding="utf-8")
    lock_path = config.database_path.with_name(config.database_path.name + ".enfold.lock")
    lock_path.symlink_to(target)
    with pytest.raises(DatabaseOwnershipError, match="cannot open database lock"):
        ServerApplication(config)


def test_check_reports_version_schema_and_grants_without_binding(tmp_path, capsys):
    config_path = _config(tmp_path)
    assert main(["--config", str(config_path), "check"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ready"
    assert report["schema_version"] == 1
    assert report["database"] == "compatible"
    assert report["socket"] == "absent"
    assert report["grant_count"] == 3
    assert report["retrieval"]["embedder_production_ready"] is False


def test_privileged_memory_actions_require_named_granted_clients(tmp_path):
    config = load_config(
        _config(
            tmp_path,
            correction_authorities=["hermes-install"],
            conflict_resolution_authorities=["hermes-install"],
        )
    )
    assert config.correction_authorities == ("hermes-install",)
    assert config.conflict_resolution_authorities == ("hermes-install",)

    with pytest.raises(ServerConfigError, match="clients without grants"):
        load_config(
            _config(
                tmp_path,
                correction_authorities=["self-claimed-client"],
            )
        )


def test_missing_database_is_not_created_or_migrated(tmp_path):
    missing = tmp_path / "missing.db"
    config_path = _config(tmp_path, database_path=str(missing))
    config = load_config(config_path)
    with pytest.raises(ServerConfigError, match="never creates"):
        ServerApplication(config)
    assert not missing.exists()


def test_unmigrated_database_is_rejected_without_schema_changes(tmp_path):
    database = tmp_path / "empty.db"
    sqlite3.connect(database).close()
    config = load_config(_config(tmp_path, database_path=str(database)))
    with pytest.raises(ServerConfigError, match="must already be schema v1; found v0"):
        inspect_config(config)
    conn = sqlite3.connect(database)
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == 0
    conn.close()


def test_live_paths_require_explicit_allow_live():
    path = Path.home() / ".hermes" / "enfold-server.json"
    with pytest.raises(ServerConfigError, match="--allow-live"):
        load_config(path)


def test_config_and_socket_parent_permissions_fail_closed(tmp_path):
    config_path = _config(tmp_path)
    config_path.chmod(0o622)
    with pytest.raises(ServerConfigError, match="group/world writable"):
        load_config(config_path)

    config_path.chmod(0o600)
    config = load_config(config_path)
    tmp_path.chmod(0o777)
    try:
        with pytest.raises(ServerConfigError, match="socket parent"):
            ServerApplication(config)
    finally:
        tmp_path.chmod(0o700)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"unknown": True}, "unknown config fields"),
        ({"grants": {}}, "non-empty object"),
        ({"grants": {"client-a": ["made-up"]}}, "unsupported memory scope"),
        ({"cleanup_stale_socket": "yes"}, "must be a boolean"),
        ({"retrieval": {"mode": "ci"}}, "allow_nonproduction=true"),
        ({"retrieval": {"mode": "mystery"}}, "must be 'ci' or 'stored'"),
    ],
)
def test_strict_config_validation(tmp_path, change, message):
    with pytest.raises(ServerConfigError, match=message):
        load_config(_config(tmp_path, **change))


def test_near_dedup_off_switch_is_strictly_parsed(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["retrieval"]["near_dedup_enabled"] = False
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert load_config(path).retrieval.near_dedup_enabled is False

    raw["retrieval"]["near_dedup_enabled"] = "no"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ServerConfigError, match="near_dedup_enabled must be a boolean"):
        load_config(path)


def test_retrieval_selection_is_required_and_ci_needs_explicit_opt_in(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["retrieval"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ServerConfigError, match="missing config fields.*retrieval"):
        load_config(path)


def test_stored_retrieval_readiness_checks_identity_without_model_call(tmp_path):
    database = _database(tmp_path / "memory.db")
    conn = sqlite3.connect(database)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_embeddings(
            fact_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            embedding_identity TEXT NOT NULL,
            PRIMARY KEY(fact_id, embedding_identity)
        )
        """
    )
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (
            1,
            embedding_to_bytes(np.asarray((1.0, 0.0), dtype=np.float32)),
            2,
            f"ollama:fixture:document:none:{_ARTIFACT_DIGEST}",
        ),
    )
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope) VALUES (1, 'fixture fact', 'private')"
    )
    conn.commit()
    conn.close()
    retrieval = {
        "mode": "stored",
        "provider": "ollama",
        "model": "fixture",
        "dimensions": 2,
        "query_identity": f"ollama:fixture:query:none:{_ARTIFACT_DIGEST}",
        "document_identity": f"ollama:fixture:document:none:{_ARTIFACT_DIGEST}",
        "embedding_version": _ARTIFACT_DIGEST,
        "model_fingerprint": _ARTIFACT_DIGEST,
        "prefix_policy": "none",
        "processor": {"mode": "daemon-supervised", "poll_seconds": 0.01},
    }
    config = load_config(_config(tmp_path, retrieval=retrieval))

    attestor = FakeArtifactAttestor()
    report = inspect_config(config, artifact_attestor=attestor)

    assert report["status"] == "ready"
    assert report["activation_blocker"] is None
    assert report["retrieval"]["embedder_production_ready"] is True
    assert report["retrieval"]["document_embedding_identity"] == (
        f"ollama:fixture:document:none:{_ARTIFACT_DIGEST}"
    )
    assert report["retrieval"]["missing_embedding_behavior"] == "fail-closed"
    assert report["artifact_attestation"] == {
        "provider": "ollama", "status": "verified"
    }
    assert attestor.calls == [("fixture", _ARTIFACT_DIGEST)]

    with ServerApplication(
        config, artifact_attestor=FakeArtifactAttestor()
    ) as application:
        application.daemon.start()
        health = _client(config.socket_path).request("health")
        assert health["artifact_attestation"] == {
            "provider": "ollama", "status": "verified"
        }

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(config.socket_path))
    listener.listen(1)
    try:
        live = inspect_config(
            config, probe_socket=True, artifact_attestor=FakeArtifactAttestor()
        )
    finally:
        listener.close()
        config.socket_path.unlink()
    assert live["status"] == "blocked"
    assert live["embedding_worker"]["state"] == "live-health-unverified"
    assert "protocol health" in live["activation_blocker"]

    missing = dict(retrieval)
    missing["query_identity"] = f"ollama:missing:query:none:{_ARTIFACT_DIGEST}"
    missing["document_identity"] = (
        f"ollama:missing:document:none:{_ARTIFACT_DIGEST}"
    )
    with pytest.raises(ServerConfigError, match="canonically derived"):
        load_config(_config(tmp_path, retrieval=missing))


def test_stored_retrieval_requires_an_immutable_artifact_digest(tmp_path):
    retrieval = {
        "mode": "stored",
        "provider": "ollama",
        "model": "fixture",
        "dimensions": 2,
        "query_identity": "ollama:fixture:query:none:v1",
        "document_identity": "ollama:fixture:document:none:v1",
        "embedding_version": "v1",
        "model_fingerprint": "v1",
        "prefix_policy": "none",
        "processor": {"mode": "daemon-supervised"},
    }
    with pytest.raises(ServerConfigError, match="64 lowercase hexadecimal"):
        load_config(_config(tmp_path, retrieval=retrieval))

    missing = dict(retrieval)
    missing.pop("model_fingerprint")
    with pytest.raises(ServerConfigError, match="model_fingerprint"):
        load_config(_config(tmp_path, retrieval=missing))


def test_stored_startup_fails_before_backfill_when_attestation_fails(tmp_path):
    database = _database(tmp_path / "memory.db")
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO facts(content, scope) VALUES ('must not be backfilled', 'private')"
        )
        conn.commit()
    retrieval = {
        "mode": "stored",
        "provider": "ollama",
        "model": "fixture",
        "dimensions": 2,
        "query_identity": f"ollama:fixture:query:none:{_ARTIFACT_DIGEST}",
        "document_identity": f"ollama:fixture:document:none:{_ARTIFACT_DIGEST}",
        "embedding_version": _ARTIFACT_DIGEST,
        "model_fingerprint": _ARTIFACT_DIGEST,
        "prefix_policy": "none",
        "processor": {"mode": "daemon-supervised"},
    }
    config = load_config(_config(tmp_path, retrieval=retrieval))
    attestor = FakeArtifactAttestor(
        failure=ArtifactAttestationError("fixture attestation failure")
    )

    with pytest.raises(ServerConfigError, match="artifact attestation failed"):
        ServerApplication(config, artifact_attestor=attestor)

    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM embedding_jobs").fetchone() == (0,)
    assert not (tmp_path / "memory.db.enfold.lock").exists()
    assert attestor.calls == [("fixture", _ARTIFACT_DIGEST)]


@pytest.mark.parametrize(
    ("field", "value"),
    [("poll_seconds", 0), ("drain_limit", 1.5), ("lease_seconds", True),
     ("max_attempts", -1), ("heartbeat_stale_seconds", float("inf"))],
)
def test_stored_processor_numeric_configuration_fails_early(tmp_path, field, value):
    retrieval = {
        "mode": "stored", "provider": "ollama", "model": "fixture",
        "dimensions": 2,
        "query_identity": f"ollama:fixture:query:none:{_ARTIFACT_DIGEST}",
        "document_identity": f"ollama:fixture:document:none:{_ARTIFACT_DIGEST}",
        "embedding_version": _ARTIFACT_DIGEST,
        "model_fingerprint": _ARTIFACT_DIGEST,
        "prefix_policy": "none",
        "processor": {"mode": "daemon-supervised", field: value},
    }
    with pytest.raises(ServerConfigError, match=f"processor.{field}"):
        load_config(_config(tmp_path, retrieval=retrieval))


def test_stored_fastembed_is_blocked_until_worker_process_isolation(tmp_path):
    retrieval = {
        "mode": "stored", "provider": "fastembed", "model": "fixture",
        "dimensions": 2,
        "query_identity": "fastembed:fixture:query:none:v1",
        "document_identity": "fastembed:fixture:document:none:v1",
        "embedding_version": "v1", "model_fingerprint": "v1",
        "prefix_policy": "none",
        "processor": {"mode": "daemon-supervised"},
    }
    with pytest.raises(ServerConfigError, match="killable process isolation"):
        load_config(_config(tmp_path, retrieval=retrieval))


def test_close_is_retryable_when_worker_join_temporarily_fails(tmp_path):
    application = ServerApplication(load_config(_config(tmp_path)))

    class FlakyWorker:
        calls = 0

        def stop(self, _timeout):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("join timeout")

    worker = FlakyWorker()
    application.embedding_worker = worker
    with pytest.raises(RuntimeError, match="join timeout"):
        application.close()
    assert application._closed is False
    assert application.ownership._fd is not None

    application.close()
    assert application._closed is True
    assert worker.calls == 2


def _extraction_config(**changes):
    host = {
        "type": "subprocess",
        "argv": ["/usr/bin/enfold-extractor-fixture"],
        "model_identity": "fixture-model",
        "prompt_identity": "fixture-prompt-v1",
        "timeout_seconds": 0.2,
        "terminate_grace_seconds": 0.1,
    }
    config = {
        "mode": "daemon-supervised",
        "host": host,
        "poll_seconds": 0.01,
        "lease_seconds": 1.0,
        "heartbeat_seconds": 0.1,
        "heartbeat_stale_seconds": 0.5,
        "pending_stale_seconds": 30.0,
    }
    config.update(changes)
    return config


class FakeExtractionAdapter:
    identity = "fake:fixture-model:fixture-prompt-v1"

    def __init__(self):
        self.calls = 0

    def extract(self, _envelope):
        self.calls += 1
        return [ExtractedMemory("Victor uses supervised shared memory.")]


def test_supervised_extraction_uses_dedicated_connection_and_reports_health(tmp_path):
    adapter = FakeExtractionAdapter()
    factory_calls = []

    def factory(config):
        factory_calls.append(config)
        return adapter

    config = load_config(_config(tmp_path, extraction=_extraction_config()))
    with ServerApplication(config, extraction_extractor_factory=factory) as application:
        assert application.extraction_connection is not application.connection
        assert factory_calls == [config.extraction.host]
        application.extraction_worker.start()
        application.daemon.start()
        client = _client(config.socket_path)
        queued = client.request(
            "memory.extraction.enqueue",
            {
                "transcript": "Victor uses supervised shared memory.",
                "source": "integration_test",
            },
        )
        assert queued["outcome"] == "queued"
        assert queued["automatic_llm_extraction"] == "daemon-supervised"
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            health = client.request("health")
            extraction_health = health["automatic_llm_extraction"]
            queue_health = extraction_health.get("queue", {})
            if (
                extraction_health["status"] == "ready"
                and adapter.calls
                and queue_health.get("pending") == 0
                and queue_health.get("processing") == 0
            ):
                break
            time.sleep(0.01)
        assert health["status"] == "ok"
        extraction = health["automatic_llm_extraction"]
        assert extraction["status"] == "ready"
        assert extraction["worker"]["last_error"] is None
        assert extraction["queue"] == {
            "pending": 0,
            "processing": 0,
            "dead": 0,
            "oldest_active_age_seconds": None,
            "pending_stale": False,
        }
        assert adapter.calls == 1


@pytest.mark.parametrize(
    ("extraction", "message"),
    [
        ({"mode": "daemon-supervised"}, "requires host"),
        ({"mode": "disabled", "poll_seconds": 1}, "accepts only mode"),
        (
            _extraction_config(host={
                "type": "subprocess", "argv": ["relative-command"],
                "model_identity": "fixture", "prompt_identity": "v1",
            }),
            r"argv\[0\] must be absolute",
        ),
        (
            _extraction_config(heartbeat_stale_seconds=0.3),
            "must exceed the host timeout",
        ),
    ],
)
def test_extraction_configuration_fails_closed(tmp_path, extraction, message):
    with pytest.raises(ServerConfigError, match=message):
        load_config(_config(tmp_path, extraction=extraction))


def test_check_reports_configured_extraction_without_starting_adapter(tmp_path):
    config = load_config(_config(tmp_path, extraction=_extraction_config()))
    report = inspect_config(config)
    assert report["automatic_llm_extraction"] == {
        "status": "configured-ready-to-start"
    }
