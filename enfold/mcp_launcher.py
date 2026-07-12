"""Safe process launcher for the Enfold v1 MCP stdio proxy.

MCP clients have static registrations but sessions and development provenance
are dynamic.  This module binds the grant-matched identity supplied by the
registration, generates a fresh process session, discovers only bounded local
Git metadata, and starts :mod:`enfold.mcp_stdio` in-process.

No identity or scope is accepted from the environment, no shell is invoked,
and discovery failures simply omit optional provenance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
from typing import Mapping, Sequence
from urllib.parse import urlsplit

from .client import ClientConfig
from .mcp_stdio import MEMORY_CAPABILITIES, build_server
from .protocol import ClientContext


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_GIT_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class ProjectProvenance:
    project_root: str | None = None
    repository: str | None = None
    branch: str | None = None
    commit_sha: str | None = None


def new_session_id(surface: str) -> str:
    """Return an unguessable protocol-safe ID unique to this proxy process."""

    return f"{surface}-{secrets.token_urlsafe(24)}"


def _safe_value(value: str, *, maximum: int = 512) -> str | None:
    value = value.strip()
    if not value or len(value) > maximum or _CONTROL_RE.search(value):
        return None
    return value


def _git(
    cwd: Path,
    arguments: Sequence[str],
    *,
    environ: Mapping[str, str],
) -> str | None:
    path = environ.get("PATH") or os.defpath
    executable = shutil.which("git", path=path)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "-C", os.fspath(cwd), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
            env={
                "PATH": path,
                "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_PAGER": "cat",
                "GIT_TERMINAL_PROMPT": "0",
            },
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _safe_value(result.stdout)


def _safe_repository(remote: str | None, fallback: str) -> str | None:
    """Normalize an origin without retaining credentials or URL parameters."""

    if remote:
        remote = _safe_value(remote)
    if remote:
        # Git's common SCP-style form: git@host:owner/repository.git.
        scp_match = re.fullmatch(r"(?:[^@/:]+@)?([^/:]+):(.+)", remote)
        if scp_match and "://" not in remote:
            host, path = scp_match.groups()
            candidate = f"{host}/{path.lstrip('/')}"
        else:
            parsed = urlsplit(remote)
            if parsed.scheme and parsed.hostname:
                candidate = f"{parsed.hostname}{parsed.path}"
            elif parsed.scheme == "file":
                candidate = Path(parsed.path).name
            else:
                candidate = Path(remote).name
        candidate = candidate.removesuffix(".git").strip("/")
        safe = _safe_value(candidate)
        if safe:
            return safe
    return _safe_value(fallback)


def discover_provenance(
    cwd: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ProjectProvenance:
    """Discover safe CWD/Git provenance without executing a shell."""

    # Resolve Git from the platform's system path by default, rather than an
    # inherited caller-controlled PATH. Embedded callers may pass a path.
    env = {"PATH": os.defpath} if environ is None else environ
    try:
        working_directory = Path.cwd() if cwd is None else Path(cwd)
        working_directory = working_directory.resolve()
    except OSError:
        return ProjectProvenance()

    root_value = _git(
        working_directory, ("rev-parse", "--show-toplevel"), environ=env
    )
    if root_value:
        root = Path(root_value)
        project_root = _safe_value(os.fspath(root))
    else:
        root = working_directory
        project_root = _safe_value(os.fspath(working_directory))

    repository = None
    branch = None
    commit_sha = None
    if root_value:
        remote = _git(root, ("config", "--get", "remote.origin.url"), environ=env)
        repository = _safe_repository(remote, root.name)
        branch = _git(
            root, ("symbolic-ref", "--quiet", "--short", "HEAD"), environ=env
        )
        commit_sha = _git(root, ("rev-parse", "--verify", "HEAD"), environ=env)
    if branch is not None:
        branch = _safe_value(branch, maximum=255)
    if commit_sha is None or not _COMMIT_RE.fullmatch(commit_sha):
        commit_sha = None
    else:
        commit_sha = commit_sha.lower()
    return ProjectProvenance(project_root, repository, branch, commit_sha)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enfold-mcp-launch",
        description=(
            "Launch the Enfold v1 MCP proxy with static registration identity, "
            "a fresh process session, and safely discovered project provenance."
        ),
    )
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument(
        "--access-scope",
        action="append",
        dest="access_scopes",
        required=True,
        metavar="SCOPE",
        help="requested server-granted scope; repeat for multiple scopes",
    )
    parser.add_argument(
        "--session-id",
        help="explicit process session ID; normally omit to generate a fresh ID",
    )
    parser.add_argument("--connect-timeout", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    return parser


def parse_config(
    argv: Sequence[str] | None = None,
    *,
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ClientConfig:
    parser = _parser()
    args = parser.parse_args(argv)
    scopes = tuple(args.access_scopes)
    if len(scopes) != len(set(scopes)):
        parser.error("access scopes must not contain duplicates")
    provenance = discover_provenance(cwd, environ=environ)
    try:
        context = ClientContext(
            client_id=args.client_id,
            surface=args.surface,
            agent_id=args.agent_id,
            session_id=args.session_id or new_session_id(args.surface),
            project_root=provenance.project_root,
            repository=provenance.repository,
            branch=provenance.branch,
            commit_sha=provenance.commit_sha,
            access_scopes=scopes,
        )
        return ClientConfig(
            socket_path=Path(args.socket_path).expanduser(),
            context=context,
            capabilities=MEMORY_CAPABILITIES,
            connect_timeout=args.connect_timeout,
            request_timeout=args.request_timeout,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError("argparse.error did not exit")


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    try:
        server = build_server(config)
    except RuntimeError as exc:
        print(f"enfold MCP launcher startup failed: {exc}", file=sys.stderr)
        return 2
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
