from __future__ import annotations

import json

import pytest

from memory_eval.context_arena import (
    CONTEXT_ARENA_PATH,
    load_context_arena,
    run_context_arena,
)


def test_context_arena_is_synthetic_complete_and_passes_all_acceptance_cases():
    arena = load_context_arena()
    run = run_context_arena(arena)

    assert arena.version == "1.0"
    assert arena.source_path == CONTEXT_ARENA_PATH.resolve()
    assert len(arena.cases) == 6
    assert run.passed is True
    assert all(result.passed for result in run.results)
    assert {result.case_id for result in run.results} == {
        "current-state-excludes-stale",
        "unresolved-conflict-abstains",
        "private-scope-excludes-work",
        "work-scope-excludes-private",
        "budget-truncates-deterministically",
        "budget-can-abstain",
    }


def test_context_arena_loader_rejects_unknown_fact_references(tmp_path):
    payload = json.loads(CONTEXT_ARENA_PATH.read_text())
    payload["cases"][0]["expected_fact_keys"] = ["missing"]
    path = tmp_path / "bad-context-arena.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unknown facts"):
        load_context_arena(path)
