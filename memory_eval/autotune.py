from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    yaml = None

from .baseline import (
    clear_pending_extract_queue_for_eval,
    load_provider,
    prepare_eval_db,
    resolve_cases,
)
from .runner import EvalCase, run_retrieval_cases, summarize_results
from .sqlite_utils import backup_sqlite_db, quick_check


HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
LIVE_CONFIG_PATH = HERMES_HOME / "config.yaml"
DEFAULT_REPORT_ROOT = HERMES_HOME / "reports" / "memory-eval"
DEFAULT_LIVE_DB = HERMES_HOME / "memory_store.db"

RETRIEVAL_KEYS = (
    "embedding_weight",
    "fts_weight",
    "jaccard_weight",
    "hrr_weight",
    "entity_boost_weight",
    "entity_expansion",
    "entity_hub_degree_limit",
    "retrieval_decision_enabled",
    "retrieval_decision_min_score",
    "retrieval_decision_min_margin",
    "retrieval_decision_min_trust",
    "temporal_decay_half_life",
)

FIXED_RETRIEVAL_KEYS = (
    "embedding_backend",
    "embedding_prefix_policy",
    "embedding_query_prefix",
    "embedding_document_prefix",
    "ollama_url",
    "ollama_model",
    "fastembed_model",
    "fastembed_cache_dir",
    "hrr_dim",
    "temporal_filter",
    "min_trust_threshold",
)


@dataclass(frozen=True)
class TrialScore:
    recall_at_1: float
    stale_leaks_at_1: int
    stale_leak_rate_at_1: float
    latency_p95_ms: float

    @classmethod
    def from_summary(cls, summary: dict[str, Any]) -> "TrialScore":
        stale = summary.get("stale_leak@1", {})
        latency = summary.get("latency_ms", {})
        return cls(
            recall_at_1=float(summary.get("recall@1", 0.0)),
            stale_leaks_at_1=int(stale.get("leaks", 0)),
            stale_leak_rate_at_1=float(stale.get("leak_rate", 0.0)),
            latency_p95_ms=float(latency.get("p95", 0.0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "recall@1": self.recall_at_1,
            "stale_leaks@1": self.stale_leaks_at_1,
            "stale_leak_rate@1": self.stale_leak_rate_at_1,
            "latency_p95_ms": self.latency_p95_ms,
        }


@dataclass(frozen=True)
class Proposal:
    knob: str | None
    before: Any
    after: Any
    config: dict[str, Any]


@dataclass(frozen=True)
class TrialOutcome:
    trial: int
    proposal: Proposal
    score: TrialScore
    summary: dict[str, Any]
    active_backend: dict[str, Any]
    accepted: bool
    decision: str
    elapsed_seconds: float
    cleared_extract_queue_rows: int


@dataclass(frozen=True)
class KnobSpec:
    key: str
    values: tuple[Any, ...]

    def propose(self, current: dict[str, Any], rng: random.Random) -> tuple[Any, Any]:
        before = current.get(self.key)
        choices = [value for value in self.values if value != before]
        if not choices:
            return before, before

        if isinstance(before, (int, float)) and not isinstance(before, bool):
            numeric = [
                value
                for value in choices
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if numeric and rng.random() < 0.65:
                after = min(numeric, key=lambda value: abs(float(value) - float(before)))
                return before, after

        return before, rng.choice(choices)


def _parse_scalar(value: str) -> Any:
    raw = value.strip()
    if raw in {"", "null", "Null", "NULL", "~"}:
        return None
    if raw in {"true", "True", "TRUE"}:
        return True
    if raw in {"false", "False", "FALSE"}:
        return False
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip("\"'")


def _fallback_live_plugin_config(path: Path) -> dict[str, Any]:
    """Tiny fallback parser for the flat hermes-memory-store block.

    PyYAML is available in Victor's Hermes environment, but Enfold's package
    deps do not require it. This parser intentionally handles only the simple
    scalar shape used by the retrieval config block.
    """
    lines = path.read_text().splitlines()
    in_plugins = False
    in_block = False
    out: dict[str, Any] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("plugins:"):
            in_plugins = True
            in_block = False
            continue
        if in_plugins and line and not line.startswith(" "):
            break
        if in_plugins and line.startswith("  hermes-memory-store:"):
            in_block = True
            continue
        if in_block:
            if not line.startswith("    "):
                break
            key, sep, value = stripped.partition(":")
            if sep:
                out[key] = _parse_scalar(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        return {"plugins": {"hermes-memory-store": _fallback_live_plugin_config(path)}}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _live_plugin_config(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    plugins = data.get("plugins", {})
    if not isinstance(plugins, dict):
        return {}
    block = plugins.get("hermes-memory-store", {})
    return dict(block) if isinstance(block, dict) else {}


def _expand_db_path(value: Any) -> Path:
    raw = str(value or DEFAULT_LIVE_DB)
    raw = raw.replace("$HERMES_HOME", str(HERMES_HOME))
    return Path(raw).expanduser()


def _base_eval_config(live: dict[str, Any], db_path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {
        "db_path": str(db_path),
        "embedding_backend": "ollama",
        "ollama_model": "embeddinggemma",
        "embedding_prefix_policy": "auto",
        "embedding_weight": 0.45,
        "fts_weight": 0.3,
        "jaccard_weight": 0.2,
        "hrr_weight": 0.0,
        "entity_boost_weight": 0.0,
        "entity_expansion": False,
        "entity_hub_degree_limit": 25,
        "retrieval_decision_enabled": False,
        "retrieval_decision_min_score": None,
        "retrieval_decision_min_margin": None,
        "retrieval_decision_min_trust": None,
        "temporal_decay_half_life": 0,
        "temporal_filter": True,
        "hrr_dim": 1024,
        "embed_on_add": False,
        "dedup_on_add": False,
        "reflection_enabled": False,
        "extract_drain_batch": 0,
    }
    for key in (*RETRIEVAL_KEYS, *FIXED_RETRIEVAL_KEYS):
        if key in live:
            config[key] = live[key]
    config["db_path"] = str(db_path)
    config["embed_on_add"] = False
    config["dedup_on_add"] = False
    config["reflection_enabled"] = False
    config["extract_drain_batch"] = 0
    return config


def _jsonable_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted((*RETRIEVAL_KEYS, *FIXED_RETRIEVAL_KEYS)):
        if key in config:
            out[key] = config[key]
    return out


def _eligible_knobs(active_backend: dict[str, Any], current: dict[str, Any]) -> list[KnobSpec]:
    dense_active = bool(active_backend.get("dense_embeddings"))
    knobs = [
        KnobSpec("fts_weight", (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)),
        KnobSpec("jaccard_weight", (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)),
        KnobSpec("hrr_weight", (0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 1.0)),
        KnobSpec("entity_boost_weight", (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)),
        KnobSpec("entity_expansion", (False, True)),
        KnobSpec("retrieval_decision_enabled", (False, True)),
        KnobSpec("temporal_decay_half_life", (0, 604800, 2592000, 7776000)),
    ]
    if dense_active:
        knobs.insert(0, KnobSpec("embedding_weight", (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)))
    if bool(current.get("entity_expansion")):
        knobs.append(KnobSpec("entity_hub_degree_limit", (5, 10, 15, 25, 40, 75)))
    if bool(current.get("retrieval_decision_enabled")):
        knobs.extend([
            KnobSpec("retrieval_decision_min_score", (None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)),
            KnobSpec("retrieval_decision_min_margin", (None, 0.0, 0.01, 0.02, 0.05, 0.1)),
            KnobSpec("retrieval_decision_min_trust", (None, 0.3, 0.4, 0.5, 0.6, 0.7)),
        ])
    return knobs


def _propose_neighbor(
    current: dict[str, Any],
    knobs: list[KnobSpec],
    rng: random.Random,
) -> Proposal:
    spec = rng.choice(knobs)
    before, after = spec.propose(current, rng)
    proposed = copy.deepcopy(current)
    proposed[spec.key] = after
    return Proposal(knob=spec.key, before=before, after=after, config=proposed)


def _active_backend(provider: Any) -> dict[str, Any]:
    backend = str(getattr(provider, "_embedding_backend", "unknown"))
    model_name = None
    model_fn = getattr(provider, "_embedding_model_name", None)
    if callable(model_fn):
        try:
            model_name = model_fn()
        except Exception:
            model_name = None
    dense = bool(getattr(provider, "_embedder_available", False))
    return {
        "configured_backend": backend,
        "model": model_name,
        "dense_embeddings": dense,
        "active": f"{backend}:{model_name}" if dense and model_name else "holographic_only",
    }


def _evaluate_trial(
    *,
    trial: int,
    proposal: Proposal,
    base_snapshot: Path,
    report_dir: Path,
    cases: list[EvalCase],
    limit: int,
    repo_root: Path,
    hermes_src: Path | None,
    test_stubs: bool,
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]]:
    trial_db = report_dir / "scratch" / f"trial-{trial:04d}.db"
    backup_sqlite_db(base_snapshot, trial_db, overwrite=True)
    cleared = clear_pending_extract_queue_for_eval(trial_db)
    config = copy.deepcopy(proposal.config)
    config["db_path"] = str(trial_db)

    provider = load_provider(
        repo_root,
        config,
        hermes_src=hermes_src,
        test_stubs=test_stubs,
    )
    active_backend = _active_backend(provider)
    try:
        results = run_retrieval_cases(provider, cases, limit=limit)
    finally:
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    summary = summarize_results(results)
    return summary, config, cleared, active_backend


def _is_better(
    challenger: TrialScore,
    incumbent: TrialScore,
    baseline: TrialScore,
) -> tuple[bool, str]:
    if challenger.stale_leaks_at_1 > baseline.stale_leaks_at_1:
        return False, "rejected: stale_leak@1 increased over baseline"
    if challenger.recall_at_1 > incumbent.recall_at_1:
        return True, "accepted: recall@1 improved"
    if challenger.recall_at_1 < incumbent.recall_at_1:
        return False, "rejected: recall@1 regressed"
    if challenger.stale_leaks_at_1 < incumbent.stale_leaks_at_1:
        return True, "accepted: stale_leak@1 improved at equal recall"
    if challenger.stale_leaks_at_1 > incumbent.stale_leaks_at_1:
        return False, "rejected: stale_leak@1 regressed"
    if challenger.latency_p95_ms < incumbent.latency_p95_ms:
        return True, "accepted: latency p95 improved at equal recall and stale leak"
    return False, "rejected: no objective improvement"


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _diff_config(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    keys = sorted(set(before) | set(after))
    for key in keys:
        if before.get(key) != after.get(key):
            diff[key] = {"current": before.get(key), "recommended": after.get(key)}
    return diff


def _normalize_inactive_knobs(
    config: dict[str, Any],
    baseline: dict[str, Any],
    active_backend: dict[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    if not active_backend.get("dense_embeddings"):
        normalized["embedding_weight"] = baseline.get("embedding_weight")
    if not normalized.get("entity_expansion"):
        normalized["entity_hub_degree_limit"] = baseline.get("entity_hub_degree_limit")
    if not normalized.get("retrieval_decision_enabled"):
        normalized["retrieval_decision_min_score"] = baseline.get("retrieval_decision_min_score")
        normalized["retrieval_decision_min_margin"] = baseline.get("retrieval_decision_min_margin")
        normalized["retrieval_decision_min_trust"] = baseline.get("retrieval_decision_min_trust")
    return normalized


def _format_config_value(value: Any) -> str:
    if value is None:
        return "null"
    return json.dumps(value)


def _write_recommendation(
    *,
    path: Path,
    live_config: dict[str, Any],
    baseline_config: dict[str, Any],
    best_config: dict[str, Any],
    baseline_score: TrialScore,
    best_score: TrialScore,
    trials: list[TrialOutcome],
    caps: dict[str, Any],
    case_count: int,
    case_notes: list[str],
    backend_notes: list[str],
) -> None:
    baseline_retrieval = {key: baseline_config.get(key) for key in RETRIEVAL_KEYS if key in baseline_config}
    best_retrieval = {key: best_config.get(key) for key in RETRIEVAL_KEYS if key in best_config}
    # Baseline config is the effective current live retrieval config: explicit
    # YAML values plus provider defaults for absent keys. Diff against that
    # instead of raw YAML so default-valued missing keys are not presented as
    # recommendations.
    diff_vs_live = _diff_config(baseline_retrieval, best_retrieval)
    diff_vs_baseline = _diff_config(baseline_retrieval, best_retrieval)

    delta_recall = best_score.recall_at_1 - baseline_score.recall_at_1
    delta_stale = best_score.stale_leaks_at_1 - baseline_score.stale_leaks_at_1
    delta_latency = best_score.latency_p95_ms - baseline_score.latency_p95_ms
    accepted_trials = [trial for trial in trials if trial.accepted]

    lines = [
        "# Enfold Retrieval Autotune Recommendation",
        "",
        "Proposal only. No changes were applied to the live Hermes config or the live memory DB.",
        "",
        "## Objective",
        "",
        "- Primary: improve `recall@1`.",
        "- Hard constraint: reject any trial that increases `stale_leak@1` over baseline.",
        "- Tiebreak: lower latency p95.",
        "",
        "## Result",
        "",
        f"- Cases: {case_count}",
        f"- Trials logged: {len(trials)}",
        f"- Accepted proposals: {len(accepted_trials)}",
        f"- Baseline recall@1: {baseline_score.recall_at_1:.4f}",
        f"- Best recall@1: {best_score.recall_at_1:.4f} ({delta_recall:+.4f})",
        f"- Baseline stale leaks@1: {baseline_score.stale_leaks_at_1}",
        f"- Best stale leaks@1: {best_score.stale_leaks_at_1} ({delta_stale:+d})",
        f"- Baseline latency p95: {baseline_score.latency_p95_ms:.2f} ms",
        f"- Best latency p95: {best_score.latency_p95_ms:.2f} ms ({delta_latency:+.2f} ms)",
        "",
        "## Recommended Diff Vs Live",
        "",
    ]
    if diff_vs_live:
        for key, values in diff_vs_live.items():
            lines.append(
                f"- `{key}`: {_format_config_value(values['current'])} -> "
                f"{_format_config_value(values['recommended'])}"
            )
    else:
        lines.append("- No retrieval config change beat the live baseline under this run.")

    lines.extend(["", "## Accepted Trial Diffs Vs Baseline", ""])
    if diff_vs_baseline:
        for key, values in diff_vs_baseline.items():
            lines.append(
                f"- `{key}`: {_format_config_value(values['current'])} -> "
                f"{_format_config_value(values['recommended'])}"
            )
    else:
        lines.append("- Best config is the baseline config.")

    lines.extend(["", "## Confidence Notes", ""])
    notes = [
        f"Caps: max_experiments={caps['max_experiments']}, max_minutes={caps['max_minutes']}.",
        "Each trial ran on a fresh SQLite backup copied from the initial snapshot.",
        "The recency tie-break in `EnfoldProvider.search()` is hardcoded; only `temporal_decay_half_life` is configurable.",
    ]
    notes.extend(case_notes)
    notes.extend(backend_notes)
    if not accepted_trials:
        notes.append("No proposal beat baseline, so confidence favors keeping current live retrieval values.")
    for note in notes:
        lines.append(f"- {note}")

    path.write_text("\n".join(lines) + "\n")


def _default_report_dir(now: datetime) -> Path:
    return DEFAULT_REPORT_ROOT / f"autotune-{now.strftime('%Y-%m-%d-%H%M%S')}"


def _trial_payload(
    outcome: TrialOutcome,
    *,
    current_best_score: TrialScore,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trial": outcome.trial,
        "proposal": {
            "knob": outcome.proposal.knob,
            "before": outcome.proposal.before,
            "after": outcome.proposal.after,
        },
        "config": _jsonable_config(config),
        "scores": outcome.score.as_dict(),
        "accepted": outcome.accepted,
        "decision": outcome.decision,
        "current_best_scores": current_best_score.as_dict(),
        "active_backend": outcome.active_backend,
        "elapsed_seconds": outcome.elapsed_seconds,
        "cleared_extract_queue_rows": outcome.cleared_extract_queue_rows,
    }


def run_autotune(
    *,
    db_path: Path,
    report_dir: Path,
    max_experiments: int,
    max_minutes: float,
    cases_path: Path | None,
    sample: int,
    limit: int,
    min_trust: float,
    repo_root: Path,
    hermes_src: Path | None,
    test_stubs: bool,
    seed: int,
) -> dict[str, Any]:
    if max_experiments <= 0:
        raise ValueError("--max-experiments must be positive")
    if max_minutes <= 0:
        raise ValueError("--max-minutes must be positive")

    rng = random.Random(seed)
    report_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = report_dir / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "experiments.jsonl"
    recommendation_path = report_dir / "RECOMMENDATION.md"

    live_config = _live_plugin_config(LIVE_CONFIG_PATH)
    base_snapshot = scratch_dir / "base-snapshot.db"
    prepared = prepare_eval_db(db_path, base_snapshot)
    snapshot_check = quick_check(prepared.path)
    cases = resolve_cases(
        db_path=prepared.path,
        cases_path=cases_path,
        sample=sample,
        min_trust=min_trust,
    )
    case_notes = [
        "No paraphrase or distractor generation was found in `memory_eval/cases.py`; this run used exact-fact cases."
    ]

    baseline_config = _base_eval_config(live_config, prepared.path)
    baseline_proposal = Proposal(knob=None, before=None, after=None, config=baseline_config)
    started = time.monotonic()
    summary, actual_baseline_config, cleared, active_backend = _evaluate_trial(
        trial=0,
        proposal=baseline_proposal,
        base_snapshot=prepared.path,
        report_dir=report_dir,
        cases=cases,
        limit=limit,
        repo_root=repo_root,
        hermes_src=hermes_src,
        test_stubs=test_stubs,
    )
    baseline_score = TrialScore.from_summary(summary)
    best_score = baseline_score
    best_config = copy.deepcopy(baseline_config)
    elapsed = time.monotonic() - started
    baseline_outcome = TrialOutcome(
        trial=0,
        proposal=baseline_proposal,
        score=baseline_score,
        summary=summary,
        active_backend=active_backend,
        accepted=True,
        decision="baseline",
        elapsed_seconds=elapsed,
        cleared_extract_queue_rows=cleared,
    )
    trials = [baseline_outcome]
    _write_jsonl(log_path, _trial_payload(
        baseline_outcome,
        current_best_score=best_score,
        config=actual_baseline_config,
    ))

    backend_notes = [
        f"Baseline active backend: `{active_backend['active']}`.",
    ]
    if not active_backend.get("dense_embeddings"):
        backend_notes.append(
            "Dense embedding backend was unavailable, so this run used the plugin's holographic-only fallback and is not comparable to Ollama-backed runs."
        )

    deadline = started + max_minutes * 60.0
    next_trial = 1
    while next_trial < max_experiments and time.monotonic() < deadline:
        knobs = _eligible_knobs(active_backend, best_config)
        proposal = _propose_neighbor(best_config, knobs, rng)
        trial_started = time.monotonic()
        try:
            summary, trial_config, cleared, trial_backend = _evaluate_trial(
                trial=next_trial,
                proposal=proposal,
                base_snapshot=prepared.path,
                report_dir=report_dir,
                cases=cases,
                limit=limit,
                repo_root=repo_root,
                hermes_src=hermes_src,
                test_stubs=test_stubs,
            )
            score = TrialScore.from_summary(summary)
            accepted, decision = _is_better(score, best_score, baseline_score)
        except Exception as exc:
            trial_config = copy.deepcopy(proposal.config)
            trial_backend = {"active": "error", "dense_embeddings": False}
            cleared = 0
            summary = {"error": f"{type(exc).__name__}: {exc}"}
            score = TrialScore(0.0, baseline_score.stale_leaks_at_1 + 1, 1.0, float("inf"))
            accepted = False
            decision = f"rejected: trial error: {type(exc).__name__}: {exc}"

        if trial_backend.get("active") != active_backend.get("active"):
            decision = f"rejected: backend changed from baseline ({trial_backend.get('active')})"
            accepted = False

        if accepted:
            best_config = copy.deepcopy(proposal.config)
            best_score = score

        outcome = TrialOutcome(
            trial=next_trial,
            proposal=proposal,
            score=score,
            summary=summary,
            active_backend=trial_backend,
            accepted=accepted,
            decision=decision,
            elapsed_seconds=time.monotonic() - trial_started,
            cleared_extract_queue_rows=cleared,
        )
        trials.append(outcome)
        _write_jsonl(log_path, _trial_payload(
            outcome,
            current_best_score=best_score,
            config=trial_config,
        ))
        next_trial += 1

    best_config = _normalize_inactive_knobs(best_config, baseline_config, active_backend)

    _write_recommendation(
        path=recommendation_path,
        live_config=live_config,
        baseline_config=baseline_config,
        best_config=best_config,
        baseline_score=baseline_score,
        best_score=best_score,
        trials=trials,
        caps={"max_experiments": max_experiments, "max_minutes": max_minutes},
        case_count=len(cases),
        case_notes=case_notes,
        backend_notes=backend_notes,
    )

    result = {
        "report_dir": str(report_dir),
        "experiments_log": str(log_path),
        "recommendation": str(recommendation_path),
        "snapshot": {
            "source": str(prepared.backup.source),
            "destination": str(prepared.backup.destination),
            "quick_check": snapshot_check,
            "bytes": prepared.backup.bytes,
        },
        "case_count": len(cases),
        "trials": len(trials),
        "baseline": baseline_score.as_dict(),
        "best": best_score.as_dict(),
        "beat_baseline": best_score != baseline_score,
        "active_backend": active_backend,
    }
    (report_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Proposal-only overnight tuner for Enfold retrieval on SQLite snapshots."
    )
    parser.add_argument("--db", help="Input SQLite memory_store.db; copied before any provider use")
    parser.add_argument("--out-dir", help="Report directory; defaults under ~/.hermes/reports/memory-eval")
    parser.add_argument("--max-experiments", type=int, required=True, help="Hard cap including baseline trial")
    parser.add_argument("--max-minutes", type=float, required=True, help="Wall-clock cap in minutes")
    parser.add_argument("--cases", help="Optional JSON eval case file")
    parser.add_argument("--sample", type=int, default=50, help="Exact-fact case count when --cases is omitted")
    parser.add_argument("--limit", type=int, default=10, help="Search result limit")
    parser.add_argument("--min-trust", type=float, default=0.3)
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--hermes-src", help="Hermes source root for real parent provider imports")
    parser.add_argument("--test-stubs", action="store_true", help="Use tests/fake_hermes stubs")
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args(argv)

    live = _live_plugin_config(LIVE_CONFIG_PATH)
    db_path = Path(args.db).expanduser() if args.db else _expand_db_path(live.get("db_path"))
    now = datetime.now()
    report_dir = Path(args.out_dir).expanduser() if args.out_dir else _default_report_dir(now)

    result = run_autotune(
        db_path=db_path,
        report_dir=report_dir,
        max_experiments=args.max_experiments,
        max_minutes=args.max_minutes,
        cases_path=Path(args.cases).expanduser() if args.cases else None,
        sample=args.sample,
        limit=args.limit,
        min_trust=args.min_trust,
        repo_root=Path(args.repo_root).expanduser(),
        hermes_src=Path(args.hermes_src).expanduser() if args.hermes_src else None,
        test_stubs=args.test_stubs,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
