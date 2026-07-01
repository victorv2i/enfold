from __future__ import annotations

from pathlib import Path
from contextlib import closing
import sqlite3
from typing import Any

from .sqlite_utils import TERMINAL_EXTRACT_QUEUE_STATUSES, connect_readonly

_REQUIRED_PROVENANCE_TABLES = {"raw_episodes", "fact_provenance"}
_REQUIRED_FACT_TEMPORAL_COLUMNS = {"valid_from", "valid_to", "superseded_by"}


def _tables(conn) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    return {str(row[0]) for row in rows}


def _columns(conn, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        return set()
    return {str(row[1]) for row in rows}


def _count(conn, table: str) -> int:
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except Exception:
        return 0


def _embedding_count(conn, identity: str | None) -> int:
    try:
        if identity:
            return int(conn.execute(
                "SELECT COUNT(*) FROM fact_embeddings WHERE embedding_identity = ?",
                (identity,),
            ).fetchone()[0])
        return int(conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0])
    except Exception:
        return 0


def _hrr_count(conn) -> int:
    try:
        return int(conn.execute("SELECT COUNT(*) FROM facts WHERE hrr_vector IS NOT NULL").fetchone()[0])
    except Exception:
        return 0


def _pending_queue_count(conn) -> int:
    cols = _columns(conn, "extract_queue")
    if not cols:
        return 0
    try:
        if "status" in cols:
            placeholders = ", ".join("?" for _ in TERMINAL_EXTRACT_QUEUE_STATUSES)
            return int(conn.execute(
                f"SELECT COUNT(*) FROM extract_queue WHERE status NOT IN ({placeholders})",
                TERMINAL_EXTRACT_QUEUE_STATUSES,
            ).fetchone()[0])
        return _count(conn, "extract_queue")
    except sqlite3.OperationalError:
        return 0


def inspect_memory_schema(
    db_path: str | Path,
    *,
    current_embedding_identity: str | None = None,
) -> dict[str, Any]:
    """Inspect a copied Holographic+ SQLite DB for SOTA memory readiness gates."""
    with closing(connect_readonly(db_path)) as conn:
        tables = _tables(conn)
        fact_columns = _columns(conn, "facts")
        fact_count = _count(conn, "facts") if "facts" in tables else 0
        embedding_count = _embedding_count(conn, current_embedding_identity) if "fact_embeddings" in tables else 0
        all_embedding_count = _count(conn, "fact_embeddings") if "fact_embeddings" in tables else 0
        hrr_count = _hrr_count(conn) if "facts" in tables and "hrr_vector" in fact_columns else 0
        queue_pending = _pending_queue_count(conn) if "extract_queue" in tables else 0

    missing_tables = sorted(_REQUIRED_PROVENANCE_TABLES - tables)
    missing_fact_columns = sorted(_REQUIRED_FACT_TEMPORAL_COLUMNS - fact_columns)
    embedding_coverage = 0.0 if fact_count == 0 else embedding_count / fact_count
    hrr_coverage = 0.0 if fact_count == 0 else hrr_count / fact_count

    temporal_ok = not missing_fact_columns or "fact_supersessions" in tables
    provenance_ok = not missing_tables
    return {
        "tables": sorted(tables),
        "counts": {
            "facts": fact_count,
            "fact_embeddings": all_embedding_count,
            "current_identity_embeddings": embedding_count,
        },
        "coverage": {
            "embedding": embedding_coverage,
            "hrr": hrr_coverage,
        },
        "extract_queue": {
            "pending": queue_pending,
            "empty": queue_pending == 0,
        },
        "sota_gates": {
            "embedding_coverage_complete": fact_count > 0 and embedding_count == fact_count,
            "extract_queue_empty": queue_pending == 0,
            "provenance_tables": provenance_ok,
            "temporal_supersession": temporal_ok,
        },
        "missing": {
            "tables": missing_tables,
            "fact_columns": missing_fact_columns,
        },
    }
