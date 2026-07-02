from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cases import generate_exact_fact_cases, load_cases, write_json_report
from .runner import EvalCase, run_retrieval_cases, summarize_results
from .schema_checks import inspect_memory_schema
from .sqlite_utils import BackupResult, TERMINAL_EXTRACT_QUEUE_STATUSES, backup_sqlite_db, quick_check


@dataclass(frozen=True)
class PreparedEvalDb:
    path: Path
    backup: BackupResult


def production_like_config(db_path: str | Path) -> dict[str, Any]:
    """Return the current tested local-first Enfold config for a DB copy."""
    return {
        "db_path": str(Path(db_path)),
        "embedding_backend": "ollama",
        "ollama_model": "embeddinggemma",
        "embedding_prefix_policy": "auto",
        "embedding_weight": 0.45,
        "hrr_weight": 0.0,
        "embed_on_add": False,
        "dedup_on_add": False,
        "min_trust_threshold": 0.3,
    }


def resolve_cases(
    *,
    db_path: str | Path,
    cases_path: str | Path | None,
    sample: int,
    min_trust: float,
) -> list[EvalCase]:
    if cases_path is not None:
        return load_cases(cases_path)
    return generate_exact_fact_cases(db_path, limit=sample, min_trust=min_trust)


def prepare_eval_db(db_path: str | Path, scratch_db: str | Path) -> PreparedEvalDb:
    """Create the writable DB snapshot used by the provider during eval.

    EnfoldProvider opens a normal SQLite connection on initialize().
    Even with `bump=False`, the safest boundary is therefore: never hand the
    provider the operator-supplied path. Always run on a fresh backup-API copy.
    """
    backup = backup_sqlite_db(db_path, scratch_db, overwrite=True)
    return PreparedEvalDb(path=backup.destination, backup=backup)


def clear_pending_extract_queue_for_eval(db_path: str | Path) -> int:
    """Delete pending extraction rows from the scratch DB before loading provider.

    The eval runner snapshots first, then measures schema state. If the snapshot
    contains a pending extraction row, initializing EnfoldProvider would
    start its background worker and may drain that row on the scratch DB. Eval
    retrieval does not need extraction side effects, so clear only the scratch
    copy after measurement and before provider initialization.
    """
    db = Path(db_path)
    with closing(sqlite3.connect(db)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "extract_queue" not in tables:
            return 0
        cols = {row[1] for row in conn.execute("PRAGMA table_info(extract_queue)")}
        if "status" in cols:
            placeholders = ", ".join("?" for _ in TERMINAL_EXTRACT_QUEUE_STATUSES)
            where = f"status NOT IN ({placeholders})"
            count = int(conn.execute(
                f"SELECT COUNT(*) FROM extract_queue WHERE {where}",
                TERMINAL_EXTRACT_QUEUE_STATUSES,
            ).fetchone()[0])
            conn.execute(
                f"DELETE FROM extract_queue WHERE {where}",
                TERMINAL_EXTRACT_QUEUE_STATUSES,
            )
        else:
            count = int(conn.execute("SELECT COUNT(*) FROM extract_queue").fetchone()[0])
            conn.execute("DELETE FROM extract_queue")
        conn.commit()
        return count


def _install_test_stubs(repo_root: Path) -> None:
    tests_dir = repo_root / "tests"
    sys.path.insert(0, str(tests_dir))
    import fake_hermes  # type: ignore

    fake_hermes.install_stubs()


def load_provider(repo_root: Path, config: dict[str, Any], *, hermes_src: Path | None, test_stubs: bool):
    """Load the repo's Enfold provider with either real Hermes or test stubs."""
    sys.path.insert(0, str(repo_root))
    if hermes_src is not None:
        sys.path.insert(0, str(hermes_src))
    if test_stubs:
        _install_test_stubs(repo_root)

    from enfold import EnfoldProvider

    provider = EnfoldProvider(config=config)
    provider.initialize("memory-eval-baseline")
    return provider


def _metadata(
    db_path: Path,
    cases: list[EvalCase],
    config: dict[str, Any],
    *,
    quick: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    # Keep only non-secret operational fields; this function intentionally never
    # reports URLs with credentials or environment-derived provider/API settings.
    allowed = {
        "embedding_backend",
        "ollama_model",
        "embedding_prefix_policy",
        "embedding_weight",
        "hrr_weight",
        "embed_on_add",
        "dedup_on_add",
        "min_trust_threshold",
    }
    redacted_config = {k: v for k, v in config.items() if k in allowed}
    return {
        "db_path": str(db_path),
        "quick_check": quick,
        "case_count": len(cases),
        "case_source": "file" if any("exact-fact-smoke" not in c.tags for c in cases) else "exact-fact-smoke",
        "config": redacted_config,
        "schema": schema,
    }


def run_baseline(
    *,
    db_path: str | Path,
    out_path: str | Path,
    cases_path: str | Path | None = None,
    sample: int = 50,
    limit: int = 10,
    min_trust: float = 0.3,
    repo_root: str | Path = ".",
    hermes_src: str | Path | None = None,
    test_stubs: bool = False,
    include_text: bool = False,
    scratch_db: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(out_path)
    scratch = Path(scratch_db) if scratch_db is not None else out.with_suffix(".db")
    prepared = prepare_eval_db(db_path, scratch)
    db = prepared.path
    quick = quick_check(db)
    cases = resolve_cases(db_path=db, cases_path=cases_path, sample=sample, min_trust=min_trust)
    config = production_like_config(db)
    schema = inspect_memory_schema(
        db,
        current_embedding_identity="ollama:embeddinggemma:document:auto:v1",
    )
    cleared_queue_rows = clear_pending_extract_queue_for_eval(db)
    provider = load_provider(
        Path(repo_root),
        config,
        hermes_src=Path(hermes_src) if hermes_src else None,
        test_stubs=test_stubs,
    )
    try:
        results = run_retrieval_cases(provider, cases, limit=limit)
    finally:
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    summary = summarize_results(results)
    metadata = _metadata(db, cases, config, quick=quick, schema=schema)
    metadata["input_db_path"] = str(Path(db_path))
    metadata["snapshot"] = {
        "source": str(prepared.backup.source),
        "destination": str(prepared.backup.destination),
        "quick_check": prepared.backup.quick_check,
        "bytes": prepared.backup.bytes,
    }
    metadata["eval_safety"] = {
        "cleared_extract_queue_rows": cleared_queue_rows,
    }
    write_json_report(out_path, summary=summary, results=results, metadata=metadata, include_text=include_text)
    return {"metadata": metadata, "summary": summary, "out_path": str(out_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Enfold read-only memory baseline on a SQLite DB snapshot.")
    parser.add_argument("--db", required=True, help="Input SQLite memory_store.db; copied to a scratch DB before provider use")
    parser.add_argument("--out", required=True, help="Path to write JSON report")
    parser.add_argument("--scratch-db", help="Writable snapshot path; defaults to OUT with .db suffix")
    parser.add_argument("--cases", help="Optional JSON eval case file")
    parser.add_argument("--sample", type=int, default=50, help="Number of exact-fact smoke cases when --cases is omitted")
    parser.add_argument("--limit", type=int, default=10, help="Search result limit")
    parser.add_argument("--min-trust", type=float, default=0.3)
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--hermes-src", help="Hermes source root for real parent provider imports")
    parser.add_argument("--test-stubs", action="store_true", help="Use tests/fake_hermes stubs instead of real Hermes imports")
    parser.add_argument("--include-text", action="store_true", help="Include public-tier query/result content in the local JSON report")
    args = parser.parse_args(argv)

    result = run_baseline(
        db_path=args.db,
        out_path=args.out,
        scratch_db=args.scratch_db,
        cases_path=args.cases,
        sample=args.sample,
        limit=args.limit,
        min_trust=args.min_trust,
        repo_root=args.repo_root,
        hermes_src=args.hermes_src,
        test_stubs=args.test_stubs,
        include_text=args.include_text,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
