"""Equivalence against the GENUINE hermes parent retriever.

The rest of the suite exercises PlusFactRetriever against the stand-ins in
fake_hermes. This module loads the real holographic plugin modules from a
hermes checkout (~/hermes-agent by default, override with HERMES_AGENT_ROOT),
rebuilds retrieval_plus against them, and asserts bit-identical fact ids and
scores (==, not approx) versus the real FactRetriever across queries,
category filters, trust levels, and the real temporal-decay path. Skips
cleanly when the checkout is unavailable.

The real package __init__ pulls in gateway-only dependencies, so the
holographic submodules are imported through a synthetic package that points
straight at the real source files; the retrieval and store code under test
is untouched upstream code.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

HERMES_ROOT = Path(
    os.environ.get("HERMES_AGENT_ROOT", str(Path.home() / "hermes-agent"))
)
HOLO_DIR = HERMES_ROOT / "plugins" / "memory" / "holographic"
PLUGIN_DIR = Path(__file__).resolve().parents[1] / "holographic_plus"

if not (HOLO_DIR / "retrieval.py").exists():
    pytest.skip(
        f"real hermes parent not found at {HOLO_DIR}", allow_module_level=True
    )

pytest.importorskip("numpy")

_PKG = "_hp_real_parent"
_PLUS_REAL = "_hp_retrieval_plus_real"


def _ensure_stub(name: str, **attrs) -> types.ModuleType:
    """Install a minimal module stub if *name* is not already importable."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    parent_name, _, child = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], child, mod)
    return mod


def _import_real_parent():
    """Import the genuine holographic/retrieval/store modules from the checkout."""
    if _PKG in sys.modules:
        return (
            sys.modules[_PKG + ".holographic"],
            sys.modules[_PKG + ".retrieval"],
            sys.modules[_PKG + ".store"],
        )
    if str(HERMES_ROOT) not in sys.path:
        sys.path.insert(0, str(HERMES_ROOT))
    # The real store lazily imports hermes_state, which needs
    # agent.memory_manager; the fake agent package does not provide it.
    _ensure_stub("agent.memory_manager", sanitize_context=lambda value: value)
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(HOLO_DIR)]
    sys.modules[_PKG] = pkg
    holo = importlib.import_module(_PKG + ".holographic")
    retrieval = importlib.import_module(_PKG + ".retrieval")
    store = importlib.import_module(_PKG + ".store")
    importlib.import_module("hermes_state")  # surface env problems as a skip
    if not holo._HAS_NUMPY:
        raise RuntimeError("real holographic module reports numpy unavailable")
    return holo, retrieval, store


def _load_plus_against_real(real_holo, real_retrieval):
    """Exec a fresh retrieval_plus bound to the real parent modules."""
    if _PLUS_REAL in sys.modules:
        return sys.modules[_PLUS_REAL]
    shim = types.ModuleType("plugins.memory.holographic")
    shim.holographic = real_holo
    shim.retrieval = real_retrieval
    targets = {
        "plugins.memory.holographic": shim,
        "plugins.memory.holographic.holographic": real_holo,
        "plugins.memory.holographic.retrieval": real_retrieval,
    }
    saved = {key: sys.modules.get(key) for key in targets}
    sys.modules.update(targets)
    try:
        spec = importlib.util.spec_from_file_location(
            _PLUS_REAL, PLUGIN_DIR / "retrieval_plus.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PLUS_REAL] = module
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


@pytest.fixture(scope="module")
def real_parent():
    try:
        return _import_real_parent()
    except Exception as exc:
        pytest.skip(f"real hermes parent could not be imported: {exc}")


@pytest.fixture(scope="module")
def plus_real(real_parent):
    real_holo, real_retrieval, _ = real_parent
    return _load_plus_against_real(real_holo, real_retrieval)


# (content, category, tags, trust, age_days)
FACTS = [
    ("The user prefers pnpm for all node projects", "tool", "pnpm,node", 0.9, 0),
    ("The deploy target for web projects is vercel", "tool", "deploy,vercel", 0.7, 3),
    ("The tracker app uses sqlite for the fact store", "project", "tracker,sqlite", 0.8, 30),
    ("The gateway restarts are scheduled overnight", "general", "", 0.4, 60),
    ("The user keeps projects under the home projects directory", "project", "projects", 0.55, 10),
    ("Node version is managed with mise for projects", "tool", "node,mise", 0.35, 90),
    ("The memory plugin stores facts in sqlite", "project", "memory,sqlite", 0.95, 1),
    ("The user likes dark themed dashboards for projects", "user_pref", "ui,dark", 0.6, 120),
    ("The fact store search blends sqlite fts with hrr vectors", "project", "memory,search", 0.5, 45),
    ("The user runs the deploy of node projects from the terminal", "tool", "deploy,node", 0.25, 14),
]

WEIGHTS = dict(fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7, hrr_dim=64)
QUERIES = ["projects", "user", "sqlite store", "node deploy", "memory facts"]
CATEGORIES = [None, "tool", "project"]
TRUST_LEVELS = [0.0, 0.3, 0.6]


@pytest.fixture()
def real_store(real_parent, tmp_path):
    _, _, store_mod = real_parent
    store = store_mod.MemoryStore(db_path=tmp_path / "facts.db", hrr_dim=64)
    for content, category, tags, trust, age_days in FACTS:
        fact_id = store.add_fact(content, category=category, tags=tags)
        # Vary trust and backdate timestamps so the real decay path has
        # genuinely different ages to work with.
        store._conn.execute(
            """
            UPDATE facts
            SET trust_score = ?,
                created_at = datetime('now', ?),
                updated_at = datetime('now', ?)
            WHERE fact_id = ?
            """,
            (trust, f"-{age_days} days", f"-{age_days} days", fact_id),
        )
    store._conn.commit()
    yield store
    store.close()


def _assert_parent_recall_superset(parent, plus, query, category, min_trust):
    """Plus is a strict recall improvement over the parent, scores preserved.

    The parent feeds the raw query to ``facts_fts MATCH``, which ANDs every
    token (and errors out on hyphens/punctuation), so it silently misses facts
    that contain only some query tokens -- the dead-lexical-recall bug.
    PlusFactRetriever sanitises the query into an OR of the index's own tokens,
    so for any query it returns a SUPERSET of the parent's facts. Where a fact
    appears in BOTH result sets the score is byte-identical (==, not approx),
    which keeps this an exact check of the hot-path scoring and the real
    temporal-decay math on the shared facts. The hot-path no-blob invariant
    still holds.
    """
    expected = parent.search(query, category=category, min_trust=min_trust, limit=5)
    actual = plus.search(query, category=category, min_trust=min_trust, limit=5)
    label = f"query={query!r} category={category!r} min_trust={min_trust}"

    exp_scores = {f["fact_id"]: f["score"] for f in expected}
    act_scores = {f["fact_id"]: f["score"] for f in actual}

    # Recall superset: every parent hit is also retrieved by Plus.
    assert set(exp_scores).issubset(set(act_scores)), (
        f"{label}: parent {sorted(exp_scores)} not subset of plus {sorted(act_scores)}"
    )
    # Scores byte-identical for facts present in both (exact, not approx).
    for fid in set(exp_scores) & set(act_scores):
        assert act_scores[fid] == exp_scores[fid], f"{label}: score drift on fact {fid}"
    for fact in actual:
        assert "hrr_vector" not in fact
    return bool(expected)


def test_plus_is_a_real_parent_subclass(real_parent, plus_real):
    real_holo, real_retrieval, _ = real_parent
    assert issubclass(plus_real.PlusFactRetriever, real_retrieval.FactRetriever)
    assert plus_real.hrr is real_holo


def test_parent_recall_superset_without_decay(real_parent, plus_real, real_store):
    _, retrieval_mod, _ = real_parent
    parent = retrieval_mod.FactRetriever(
        store=real_store, temporal_decay_half_life=0, **WEIGHTS
    )
    plus = plus_real.PlusFactRetriever(
        store=real_store, temporal_decay_half_life=0, **WEIGHTS
    )
    matched = 0
    for query in QUERIES:
        for category in CATEGORIES:
            for min_trust in TRUST_LEVELS:
                matched += _assert_parent_recall_superset(parent, plus, query, category, min_trust)
    assert matched >= 10, "too few non-empty result sets for a meaningful check"


def test_parent_recall_superset_with_real_temporal_decay(real_parent, plus_real, real_store, monkeypatch):
    """Temporal decay NOT hardwired: old created_at drives the real decay math.

    The real _temporal_decay reads datetime.now() per call, so the wall clock
    is frozen on the real retrieval module (which both retrievers share via
    inheritance) so the shared-fact scores stay byte-identical for comparison.
    """
    _, retrieval_mod, _ = real_parent

    frozen_now = datetime.now(timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return frozen_now.replace(tzinfo=None)
            return frozen_now.astimezone(tz)

    monkeypatch.setattr(retrieval_mod, "datetime", _FrozenDatetime)

    parent = retrieval_mod.FactRetriever(
        store=real_store, temporal_decay_half_life=7, **WEIGHTS
    )
    plus = plus_real.PlusFactRetriever(
        store=real_store, temporal_decay_half_life=7, **WEIGHTS
    )

    # Sanity: the decay path genuinely engages for a backdated timestamp.
    old_ts = (frozen_now - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    assert parent._temporal_decay(old_ts) < 0.1
    assert plus._temporal_decay(old_ts) == parent._temporal_decay(old_ts)

    matched = 0
    for query in QUERIES:
        for category in CATEGORIES:
            for min_trust in TRUST_LEVELS:
                matched += _assert_parent_recall_superset(parent, plus, query, category, min_trust)
    assert matched >= 10, "too few non-empty result sets for a meaningful check"
