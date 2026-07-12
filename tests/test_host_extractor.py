from __future__ import annotations

import os
import sys
import time

import pytest

from enfold.extraction_processor import ExtractionEnvelope
from enfold.host_extractor import (
    HostExtractorConfig,
    HostExtractorError,
    SubprocessHostExtractor,
)
from enfold.protocol import ClientContext


def _envelope() -> ExtractionEnvelope:
    return ExtractionEnvelope(
        transcript="USER: Victor prefers local tools.",
        source="session_end",
        scope="private",
        context=ClientContext(
            client_id="client-a-install",
            surface="client-a",
            agent_id="client-a",
            session_id="thread-1",
            access_scopes=("private",),
        ),
    )


def _config(script: str, **changes) -> HostExtractorConfig:
    values = {
        "argv": (sys.executable, "-c", script),
        "model_identity": "local-extractor:v1",
        "prompt_identity": "extract-v1",
        "timeout_seconds": 1.0,
        "terminate_grace_seconds": 0.05,
        "environment": {},
    }
    values.update(changes)
    return HostExtractorConfig(**values)


def test_subprocess_adapter_uses_argv_json_and_allowlisted_environment(monkeypatch):
    monkeypatch.setenv("ENFOLD_SHOULD_NOT_LEAK", "secret")
    script = """
import json, os, sys
request = json.load(sys.stdin)
assert request[\"version\"] == 1
assert request[\"envelope\"][\"scope\"] == \"private\"
assert os.environ[\"ONLY_THIS\"] == \"allowed\"
assert \"ENFOLD_SHOULD_NOT_LEAK\" not in os.environ
json.dump({\"version\": 1, \"proposals\": [{\"content\": \"Victor prefers local tools.\"}]}, sys.stdout)
"""
    extractor = SubprocessHostExtractor(
        _config(script, environment={"ONLY_THIS": "allowed"})
    )

    result = extractor.extract(_envelope())

    assert extractor.identity == "subprocess:local-extractor:v1:extract-v1"
    assert [proposal.content for proposal in result] == ["Victor prefers local tools."]


def test_subprocess_adapter_streams_and_limits_stdout_without_waiting_for_timeout():
    oversized = """
import sys, time
sys.stdout.buffer.write(b'x' * 65536)
sys.stdout.buffer.flush()
time.sleep(20)
"""
    started = time.monotonic()
    with pytest.raises(HostExtractorError, match="adapter_output_too_large"):
        SubprocessHostExtractor(
            _config(oversized, max_output_bytes=1024, timeout_seconds=2.0)
        ).extract(_envelope())
    assert time.monotonic() - started < 1.0


def test_subprocess_adapter_streams_and_limits_stderr_independently():
    oversized = """
import sys, time
sys.stderr.buffer.write(b'x' * 65536)
sys.stderr.buffer.flush()
time.sleep(20)
"""
    started = time.monotonic()
    with pytest.raises(HostExtractorError, match="adapter_output_too_large"):
        SubprocessHostExtractor(
            _config(oversized, max_error_bytes=1024, timeout_seconds=2.0)
        ).extract(_envelope())
    assert time.monotonic() - started < 1.0

    malformed = "import sys; sys.stdout.write('not json')"
    with pytest.raises(HostExtractorError, match="adapter_invalid_output"):
        SubprocessHostExtractor(_config(malformed)).extract(_envelope())


def test_subprocess_adapter_timeout_terminates_and_reaps_without_error_text():
    stubborn = """
import signal, time
signal.signal(signal.SIGTERM, lambda _signum, _frame: None)
time.sleep(20)
"""
    started = time.monotonic()
    with pytest.raises(HostExtractorError, match="adapter_timeout"):
        SubprocessHostExtractor(
            _config(stubborn, timeout_seconds=0.05, terminate_grace_seconds=0.05),
        ).extract(_envelope())
    assert time.monotonic() - started < 1.0


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_timeout_kills_descendants_in_the_dedicated_process_group(tmp_path):
    pid_path = tmp_path / "descendant.pid"
    script = f"""
import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
pathlib.Path({str(pid_path)!r}).write_text(str(child.pid), encoding="utf-8")
time.sleep(20)
"""
    with pytest.raises(HostExtractorError, match="adapter_timeout"):
        SubprocessHostExtractor(
            _config(script, timeout_seconds=0.2, terminate_grace_seconds=0.05)
        ).extract(_envelope())
    assert pid_path.exists()
    descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 1.0
    while os.path.exists(f"/proc/{descendant_pid}") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not os.path.exists(f"/proc/{descendant_pid}")


def test_host_identity_rejects_secret_shaped_values():
    with pytest.raises(ValueError, match="must not contain secret"):
        _config("pass", model_identity="sk-live-secret")
    with pytest.raises(ValueError, match="must not contain secret"):
        _config("pass", prompt_identity="token-v1")
