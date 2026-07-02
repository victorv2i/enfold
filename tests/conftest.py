"""Standalone test harness for enfold.

Runs without a hermes install on the path and without network access:
fake_hermes injects sys.modules stubs for the hermes internals, the plugin
package is then loaded from the enfold/ directory, and all databases
live in pytest temp dirs. Run from the repo root with:

    pytest tests/
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PLUGIN_DIR = TESTS_DIR.parent / "enfold"

sys.path.insert(0, str(TESTS_DIR))

import fake_hermes  # noqa: E402

fake_hermes.install_stubs()


def _load_plugin():
    name = "enfold"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_hp = _load_plugin()


@pytest.fixture(scope="session")
def hp():
    """The enfold package loaded against the fakes."""
    return _hp


@pytest.fixture()
def aux_module():
    """The agent.auxiliary_client stub; tests assign .call_llm on it."""
    mod = sys.modules["agent.auxiliary_client"]
    original = mod.call_llm
    yield mod
    mod.call_llm = original


class _ProviderFactory:
    def __init__(self, hp_module, tmp_path):
        self._hp = hp_module
        self._tmp = tmp_path
        self._made = []

    def __call__(self, embedder=None, init: bool = True, **cfg):
        config = {
            "db_path": str(self._tmp / "facts.db"),
            "hrr_dim": 64,
        }
        config.update(cfg)
        fake = embedder or fake_hermes.FakeEmbedder()

        class _TestProvider(self._hp.EnfoldProvider):
            def _create_embedder(self):
                return fake

        provider = _TestProvider(config=config)
        provider._fake_embedder = fake
        # Tight worker timings so tests never sit on real backoff windows
        provider._queue_backoff_base = 0.01
        provider._queue_backoff_cap = 0.05
        provider._queue_poll_interval = 0.1
        if init:
            provider.initialize("test-session")
        self._made.append(provider)
        return provider

    def shutdown_all(self):
        for provider in self._made:
            try:
                provider.shutdown()
            except Exception:
                pass


@pytest.fixture()
def make_provider(hp, tmp_path):
    """Factory for initialized providers on a temp database."""
    factory = _ProviderFactory(hp, tmp_path)
    yield factory
    factory.shutdown_all()


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """Poll *predicate* until it returns truthy or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


@pytest.fixture()
def waiter():
    return wait_until
