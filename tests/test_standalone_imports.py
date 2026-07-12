"""Public-install imports must work without the optional Hermes package."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _clean_python(code: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    # -S prevents globally installed site packages from accidentally supplying
    # Hermes; numpy is needed by enfold and is made available by its known path
    # through the normal interpreter, so instead explicitly block parent names.
    wrapped = """
import builtins
real_import = builtins.__import__
def isolated_import(name, *args, **kwargs):
    if name == 'plugins' or name.startswith('plugins.'):
        raise ModuleNotFoundError("No module named 'plugins'", name='plugins')
    return real_import(name, *args, **kwargs)
builtins.__import__ = isolated_import
""" + code
    return subprocess.run(
        [sys.executable, "-c", wrapped],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_core_store_import_does_not_require_hermes(tmp_path):
    result = _clean_python(
        """
import enfold
import enfold.core_store
assert not enfold._HERMES_ADAPTER_AVAILABLE
conn = enfold.core_store.connect_database('standalone.db')
enfold.core_store.ensure_core_schema(conn)
conn.commit()
conn.close()
print('ok')
""",
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_ops_module_and_cli_help_do_not_require_hermes(tmp_path):
    result = _clean_python(
        """
import runpy, sys
sys.argv = ['enfold.ops', '--help']
try:
    runpy.run_module('enfold.ops', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
""",
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "schema-status" in result.stdout


def test_optional_adapter_export_fails_only_when_instantiated(tmp_path):
    result = _clean_python(
        """
import enfold
assert enfold.EnfoldProvider
try:
    enfold.EnfoldProvider(config={})
except enfold.HermesAdapterUnavailableError as exc:
    assert 'standalone storage' in str(exc)
else:
    raise AssertionError('optional adapter unexpectedly initialized')
""",
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
