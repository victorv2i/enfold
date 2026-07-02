"""Offline backfill: convert legacy 'SUPERSEDED <date>:' facts to structural invalid_at.

Standalone maintenance script for a ``enfold`` SQLite fact store,
mirroring ``cluster_merge.py``'s conventions: dry-run by default, refuses a
live ``.hermes`` path, meant to run against a copy of a fact store.

Before temporal validity existed, a retired fact was marked by rewriting its
content with a ``SUPERSEDED <date>:`` / ``STALE/DISABLED <date>:`` /
``Historical/superseded <date>:`` prefix (``_is_superseded`` in
``enfold/__init__.py``), and search excluded it by string-matching
that prefix. This backfill converts each matching row to the structural
convention (``invalid_at`` set) going forward, WITHOUT rewriting content, so
the original wording is preserved for history/audit. There is no reliable way
to identify which later fact replaced a legacy row from the prefix alone, so
``superseded_by`` is left NULL: this backfill only marks the row invalid, it
never fabricates a supersession link.

The legacy string-prefix filter in ``__init__.py`` is untouched by design: it
keeps working for any row this backfill has not (yet) touched, so running it
is optional and safe to skip.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

_SUPERSEDED_PREFIXES = ("superseded", "stale/disabled", "historical/superseded")


class GuardRailError(Exception):
    """Raised when execute_backfill refuses to run for safety reasons."""


def _is_legacy_superseded(content: str) -> bool:
    """Same prefix rule as ``enfold._is_superseded``, kept in sync
    intentionally rather than imported, so this maintenance script has no
    runtime dependency on the plugin package."""
    return (content or "").lstrip().lower().startswith(_SUPERSEDED_PREFIXES)


@dataclass
class BackfillPlan:
    fact_ids: List[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.fact_ids)


def plan_backfill(conn: sqlite3.Connection) -> BackfillPlan:
    """Find every currently-valid fact whose content carries a legacy marker.

    Only rows with ``invalid_at IS NULL`` are candidates: a row already
    converted (by a prior backfill run, or by the write-time supersession
    path) is left alone.
    """
    rows = conn.execute(
        "SELECT fact_id, content FROM facts WHERE invalid_at IS NULL"
    ).fetchall()
    fact_ids = [int(r[0]) for r in rows if _is_legacy_superseded(r[1])]
    return BackfillPlan(fact_ids=fact_ids)


@dataclass
class BackfillResult:
    dry_run: bool
    count: int
    updated: Optional[int] = None
    integrity_ok: Optional[bool] = None


def _refuse_if_live_path(db_path: str) -> None:
    literal_parts = str(db_path).replace("\\", "/").split("/")
    resolved_parts = os.path.realpath(db_path).replace("\\", "/").split("/")
    if ".hermes" in literal_parts or ".hermes" in resolved_parts:
        raise GuardRailError(
            f"refusing to run against a path under .hermes ({db_path}); "
            "this tool only ever runs against a copy"
        )


def execute_backfill(
    db_path: str,
    dry_run: bool = True,
) -> BackfillResult:
    """Plan and (optionally) execute the legacy-marker backfill against *db_path*.

    Always refuses a path under ``.hermes``. A non-dry-run sets ``invalid_at``
    (structural supersession, ``superseded_by`` left NULL) on every matching
    row and ends with ``PRAGMA integrity_check``.
    """
    _refuse_if_live_path(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "invalid_at" not in cols:
            raise GuardRailError(
                "facts.invalid_at column is missing; run the temporal schema "
                "migration (ensure_temporal_schema) before backfilling"
            )

        plan = plan_backfill(conn)

        if not dry_run:
            if plan.fact_ids:
                placeholders = ",".join("?" * len(plan.fact_ids))
                conn.execute(
                    f"UPDATE facts SET invalid_at = CURRENT_TIMESTAMP "
                    f"WHERE fact_id IN ({placeholders})",
                    plan.fact_ids,
                )
                conn.commit()
            integrity_ok = _integrity_check(conn)
            return BackfillResult(
                dry_run=False,
                count=plan.count,
                updated=plan.count,
                integrity_ok=integrity_ok,
            )

        return BackfillResult(dry_run=True, count=plan.count)
    finally:
        conn.close()


def _integrity_check(conn: sqlite3.Connection) -> bool:
    row = conn.execute("PRAGMA integrity_check").fetchone()
    return bool(row) and row[0] == "ok"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", help="path to the fact store copy to backfill")
    parser.add_argument("--execute", action="store_true",
                         help="actually perform the backfill (default is dry-run)")
    args = parser.parse_args(argv)

    result = execute_backfill(args.db_path, dry_run=not args.execute)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
