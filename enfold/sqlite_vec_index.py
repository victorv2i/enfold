"""Optional sqlite-vec index over canonical Enfold fact embeddings.

``fact_embeddings`` remains the source of truth.  This module owns only the
derived vec0 table and its binding metadata; callers may always fall back to
the canonical blobs when the extension or index is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import sqlite3
from collections.abc import Sequence

import numpy as np

from .embeddings import bytes_to_embedding, embedding_to_bytes


LOGGER = logging.getLogger(__name__)
TABLE_NAMES = ("enfold_vec_embeddings_a", "enfold_vec_embeddings_b")
ACTIVE_TABLE_KEY = "sqlite_vec_active_table"
IDENTITY_KEY = "sqlite_vec_embedding_identity"
DIMENSIONS_KEY = "sqlite_vec_dimensions"


class SQLiteVecError(RuntimeError):
    """The optional vec0 index could not be safely used or rebuilt."""


@dataclass(frozen=True, slots=True)
class SQLiteVecRebuildReport:
    embedding_identity: str
    dimensions: int
    indexed_count: int


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec while minimizing SQLite's extension-loading window."""

    try:
        import sqlite_vec
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SQLiteVecError("sqlite-vec package is not installed") from exc
    enabled = False
    try:
        conn.enable_load_extension(True)
        enabled = True
        sqlite_vec.load(conn)
    except (AttributeError, sqlite3.Error, OSError) as exc:
        raise SQLiteVecError(f"sqlite-vec extension is unavailable: {exc}") from exc
    finally:
        if enabled:
            conn.enable_load_extension(False)


def _meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM enfold_meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


class SQLiteVecIndex:
    """A validated identity/dimension-bound vec0 index on one connection."""

    def __init__(
        self, conn: sqlite3.Connection, identity: str, dimensions: int, table_name: str
    ):
        self.conn = conn
        self.identity = identity
        self.dimensions = dimensions
        self.table_name = table_name

    @classmethod
    def open_configured(
        cls, conn: sqlite3.Connection, *, warn: bool = False
    ) -> SQLiteVecIndex | None:
        try:
            identity = _meta(conn, IDENTITY_KEY)
            raw_dimensions = _meta(conn, DIMENSIONS_KEY)
        except sqlite3.Error:
            return None
        if identity is None or raw_dimensions is None:
            return None
        try:
            dimensions = int(raw_dimensions)
        except ValueError:
            if warn:
                LOGGER.warning(
                    "sqlite-vec health warning: invalid dimension metadata; "
                    "falling back to brute"
                )
            return None
        return cls.open(conn, identity, dimensions, warn=warn)

    @classmethod
    def open(
        cls,
        conn: sqlite3.Connection,
        identity: str,
        dimensions: int,
        *,
        warn: bool = False,
    ) -> SQLiteVecIndex | None:
        """Return a fully validated index, or ``None`` for honest fallback."""

        reason: str | None = None
        if not identity.strip() or dimensions < 1:
            reason = "invalid configured embedding identity or dimensions"
        else:
            try:
                load_sqlite_vec(conn)
                table_name = _meta(conn, ACTIVE_TABLE_KEY)
                if table_name not in TABLE_NAMES or not _table_exists(conn, table_name):
                    reason = "vec0 table is absent"
                elif _meta(conn, IDENTITY_KEY) != identity:
                    reason = "embedding identity does not match index metadata"
                elif _meta(conn, DIMENSIONS_KEY) != str(dimensions):
                    reason = "embedding dimensions do not match index metadata"
                else:
                    source_count = int(conn.execute(
                        "SELECT COUNT(*) FROM fact_embeddings "
                        "WHERE embedding_identity=? AND dim=?",
                        (identity, dimensions),
                    ).fetchone()[0])
                    index_count = int(conn.execute(
                        f'SELECT COUNT(*) FROM "{table_name}"'
                    ).fetchone()[0])
                    missing = conn.execute(
                        f"""
                        SELECT fact_id FROM fact_embeddings
                        WHERE embedding_identity=? AND dim=?
                        EXCEPT SELECT rowid FROM "{table_name}" LIMIT 1
                        """,
                        (identity, dimensions),
                    ).fetchone()
                    extra = conn.execute(
                        f"""
                        SELECT rowid FROM "{table_name}"
                        EXCEPT SELECT fact_id FROM fact_embeddings
                        WHERE embedding_identity=? AND dim=? LIMIT 1
                        """,
                        (identity, dimensions),
                    ).fetchone()
                    if source_count != index_count or missing is not None or extra is not None:
                        reason = "vec0 population does not match canonical embeddings"
            except Exception as exc:
                reason = str(exc)
        if reason is not None:
            if warn:
                LOGGER.warning("sqlite-vec health warning: %s; falling back to brute", reason)
            return None
        return cls(conn, identity, dimensions, table_name)

    def count(self) -> int:
        return int(self.conn.execute(
            f'SELECT COUNT(*) FROM "{self.table_name}"'
        ).fetchone()[0])

    def scores(
        self, query_vector: Sequence[float], fact_ids: Sequence[int]
    ) -> dict[int, float]:
        """Return exact cosine similarities for only the authorized fact IDs."""

        if len(query_vector) != self.dimensions or any(
            not math.isfinite(float(value)) for value in query_vector
        ):
            raise SQLiteVecError("query vector does not match vec0 index dimensions")
        if len(set(fact_ids)) != len(fact_ids):
            raise SQLiteVecError("candidate fact ids must be unique")
        if not fact_ids:
            return {}
        query = embedding_to_bytes(np.asarray(query_vector, dtype=np.float32))
        scores: dict[int, float] = {}
        for offset in range(0, len(fact_ids), 500):
            batch = tuple(int(value) for value in fact_ids[offset:offset + 500])
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"""
                SELECT rowid, distance FROM "{self.table_name}"
                WHERE embedding MATCH ? AND k = ?
                  AND rowid IN ({placeholders})
                """,
                (query, len(batch), *batch),
            ).fetchall()
            for fact_id, distance in rows:
                if distance is None:
                    raise SQLiteVecError(
                        "vec0 returned an invalid cosine distance"
                    )
                score = 1.0 - float(distance)
                if not math.isfinite(score):
                    raise SQLiteVecError(
                        "vec0 returned an invalid cosine distance"
                    )
                scores[int(fact_id)] = max(0.0, score)
        if len(scores) != len(fact_ids):
            raise SQLiteVecError("vec0 query did not return every authorized candidate")
        return scores

    def upsert_in_transaction(self, fact_id: int, embedding: bytes) -> None:
        if not self.conn.in_transaction:
            raise RuntimeError("vec0 upsert requires a caller-owned transaction")
        try:
            vector = bytes_to_embedding(embedding)
        except ValueError as exc:
            raise SQLiteVecError("embedding blob is malformed") from exc
        if len(vector) != self.dimensions or any(
            not math.isfinite(float(value)) for value in vector
        ):
            raise SQLiteVecError("embedding does not match vec0 index dimensions")
        self.conn.execute(
            f'INSERT OR REPLACE INTO "{self.table_name}"(rowid, embedding) VALUES (?, ?)',
            (fact_id, embedding),
        )

    def delete_in_transaction(self, fact_id: int) -> None:
        if not self.conn.in_transaction:
            raise RuntimeError("vec0 delete requires a caller-owned transaction")
        self.conn.execute(f'DELETE FROM "{self.table_name}" WHERE rowid=?', (fact_id,))

    def clear_in_transaction(self) -> None:
        if not self.conn.in_transaction:
            raise RuntimeError("vec0 clear requires a caller-owned transaction")
        self.conn.execute(f'DELETE FROM "{self.table_name}"')


def rebuild_sqlite_vec_index(
    conn: sqlite3.Connection, embedding_identity: str, dimensions: int
) -> SQLiteVecRebuildReport:
    """Validate canonical vectors and atomically replace the derived vec0 index."""

    if conn.in_transaction:
        raise SQLiteVecError("sqlite-vec rebuild requires an idle connection")
    if not embedding_identity.strip() or dimensions < 1:
        raise ValueError("embedding identity and dimensions must be valid")
    load_sqlite_vec(conn)
    try:
        conn.execute("BEGIN IMMEDIATE")
        active = _meta(conn, ACTIVE_TABLE_KEY)
        target = TABLE_NAMES[1] if active == TABLE_NAMES[0] else TABLE_NAMES[0]
        wrong_dim = int(conn.execute(
            "SELECT COUNT(*) FROM fact_embeddings "
            "WHERE embedding_identity=? AND dim<>?",
            (embedding_identity, dimensions),
        ).fetchone()[0])
        if wrong_dim:
            raise SQLiteVecError(
                f"configured identity has {wrong_dim} vector(s) with the wrong dimension"
            )
        rows = conn.execute(
            "SELECT fact_id, embedding FROM fact_embeddings "
            "WHERE embedding_identity=? ORDER BY fact_id",
            (embedding_identity,),
        ).fetchall()
        validated: list[tuple[int, bytes]] = []
        for fact_id, blob in rows:
            if not isinstance(blob, bytes | bytearray | memoryview):
                raise SQLiteVecError(f"embedding for fact {fact_id} is not a blob")
            raw = bytes(blob)
            try:
                vector = bytes_to_embedding(raw)
            except ValueError as exc:
                raise SQLiteVecError(
                    f"embedding for fact {fact_id} is malformed"
                ) from exc
            if len(vector) != dimensions or any(
                not math.isfinite(float(value)) for value in vector
            ):
                raise SQLiteVecError(f"embedding for fact {fact_id} is malformed")
            validated.append((int(fact_id), raw))

        if _table_exists(conn, target):
            conn.execute(f'DROP TABLE "{target}"')
        conn.execute(
            f'CREATE VIRTUAL TABLE "{target}" USING vec0('
            f'embedding float[{dimensions}] distance_metric=cosine)'
        )
        conn.executemany(
            f'INSERT INTO "{target}"(rowid, embedding) VALUES (?, ?)', validated
        )
        conn.executemany(
            "INSERT INTO enfold_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (
                (IDENTITY_KEY, embedding_identity),
                (DIMENSIONS_KEY, str(dimensions)),
                (ACTIVE_TABLE_KEY, target),
            ),
        )
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    return SQLiteVecRebuildReport(embedding_identity, dimensions, len(validated))
