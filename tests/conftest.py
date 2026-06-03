"""Test bootstrap.

Puts the repo root on sys.path so `holographic_plus` imports as a package, and
stubs the Hermes base class so the package's `__init__.py` imports without a
full Hermes install — keeping the unit tests hermetic (no network, no host
dependency), per the Hermes CONTRIBUTING test conventions.
"""
import os
import sys
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Minimal stub of the bundled holographic provider that holographic_plus
# subclasses. Only needs to be a subclassable class for import to succeed.
if "plugins.memory.holographic" not in sys.modules:
    plugins = types.ModuleType("plugins")
    plugins.__path__ = []  # mark as package
    memory = types.ModuleType("plugins.memory")
    memory.__path__ = []
    holographic = types.ModuleType("plugins.memory.holographic")

    class HolographicMemoryProvider:  # pragma: no cover - import stub
        def __init__(self, config=None):
            self._config = config or {}

    holographic.HolographicMemoryProvider = HolographicMemoryProvider

    sys.modules["plugins"] = plugins
    sys.modules["plugins.memory"] = memory
    sys.modules["plugins.memory.holographic"] = holographic
