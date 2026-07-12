from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from enfold import mcp_launcher


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--socket-path", str(tmp_path / "enfold.sock"),
        "--client-id", "client-a-install-1",
        "--surface", "client-a",
        "--agent-id", "client-a",
        "--access-scope", "private",
        "--access-scope", "work",
    ]


def test_launcher_generates_fresh_session_and_ignores_identity_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        mcp_launcher,
        "discover_provenance",
        lambda *args, **kwargs: mcp_launcher.ProjectProvenance(
            "/workspace/project", "github.com/example/project", "main", "a" * 40
        ),
    )
    hostile = {
        "ENFOLD_CLIENT_ID": "attacker",
        "ENFOLD_SURFACE": "attacker",
        "ENFOLD_AGENT_ID": "attacker",
        "ENFOLD_SESSION_ID": "attacker",
        "ENFOLD_ACCESS_SCOPES": "secret",
    }

    first = mcp_launcher.parse_config(_arguments(tmp_path), environ=hostile)
    second = mcp_launcher.parse_config(_arguments(tmp_path), environ=hostile)

    assert first.context.client_id == "client-a-install-1"
    assert first.context.surface == first.context.agent_id == "client-a"
    assert first.context.access_scopes == ("private", "work")
    assert first.context.session_id != second.context.session_id
    assert re.fullmatch(r"client-a-[A-Za-z0-9_-]{32}", first.context.session_id)
    assert first.context.repository == "github.com/example/project"


def test_explicit_session_is_supported_for_a_trusted_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mcp_launcher,
        "discover_provenance",
        lambda *args, **kwargs: mcp_launcher.ProjectProvenance(),
    )
    config = mcp_launcher.parse_config(
        [*_arguments(tmp_path), "--session-id", "client-a-thread-7"], environ={}
    )
    assert config.context.session_id == "client-a-thread-7"


def test_discovery_uses_argv_without_shell_and_strips_remote_credentials(
    tmp_path, monkeypatch
):
    root = tmp_path / "project;touch-pwned"
    root.mkdir()
    responses = {
        ("rev-parse", "--show-toplevel"): (0, str(root)),
        ("config", "--get", "remote.origin.url"): (
            0,
            "https://token:secret@example.com/owner/project.git?credential=bad#fragment",
        ),
        ("symbolic-ref", "--quiet", "--short", "HEAD"): (0, "feature/safe"),
        ("rev-parse", "--verify", "HEAD"): (0, "A" * 40),
    }
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        returncode, stdout = responses[tuple(argv[3:])]
        return SimpleNamespace(returncode=returncode, stdout=stdout + "\n")

    monkeypatch.setattr(mcp_launcher.subprocess, "run", fake_run)
    result = mcp_launcher.discover_provenance(root, environ={"PATH": "/bin"})

    assert result == mcp_launcher.ProjectProvenance(
        str(root), "example.com/owner/project", "feature/safe", "a" * 40
    )
    assert len(calls) == 4
    assert all(call[1].get("shell") is None for call in calls)
    assert all(Path(call[0][0]).name == "git" for call in calls)
    assert all(call[0][1:3] == ["-C", str(root)] for call in calls)
    assert all(call[1]["timeout"] == 1.0 for call in calls)
    assert all("token" not in str(call[1]["env"]) for call in calls)


def test_non_repository_and_git_failures_keep_only_safe_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mcp_launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=128, stdout=""),
    )
    result = mcp_launcher.discover_provenance(tmp_path, environ={})
    assert result == mcp_launcher.ProjectProvenance(
        str(tmp_path.resolve()), None, None, None
    )


def test_malformed_git_outputs_are_omitted(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    responses = iter(
        [str(root), "git@example.com:owner/repo.git", "bad\x00branch", "not-a-sha"]
    )
    monkeypatch.setattr(
        mcp_launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=next(responses)),
    )
    result = mcp_launcher.discover_provenance(root, environ={})
    assert result.repository == "example.com/owner/repo"
    assert result.branch is None
    assert result.commit_sha is None


def test_requires_static_registration_identity_scope_and_absolute_socket(tmp_path):
    with pytest.raises(SystemExit):
        mcp_launcher.parse_config(
            [
                "--socket-path", str(tmp_path / "enfold.sock"),
                "--client-id", "client-a",
                "--surface", "client-a",
                "--agent-id", "client-a",
            ],
            environ={},
        )
    with pytest.raises(SystemExit):
        mcp_launcher.parse_config(
            [
                "--socket-path", "relative.sock",
                "--client-id", "client-a",
                "--surface", "client-a",
                "--agent-id", "client-a",
                "--access-scope", "private",
            ],
            environ={},
        )


def test_main_starts_stdio_proxy_in_process(monkeypatch):
    config = object()
    server = SimpleNamespace(run=lambda **kwargs: calls.append(kwargs))
    calls = []
    monkeypatch.setattr(mcp_launcher, "parse_config", lambda argv: config)
    monkeypatch.setattr(
        mcp_launcher, "build_server", lambda received: server if received is config else None
    )

    assert mcp_launcher.main(["ignored-by-test-parser"]) == 0
    assert calls == [{"transport": "stdio"}]
