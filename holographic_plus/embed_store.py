"""SQLite storage layer for dense fact embeddings.

Manages a single table ``fact_embeddings`` in the same database file used by
MemoryStore.  Intentionally kept separate from the holographic store so that:

  - No schema changes are needed in the parent plugin.
  - The table is lazily created on first use.
  - The parent plugin remains unaware of embeddings.

Table schema::

    fact_embeddings (
        fact_id   INTEGER NOT NULL,      -- FK to facts.fact_id (not enforced)
        embedding BLOB NOT NULL,         -- numpy float32 bytes
        dim       INTEGER NOT NULL,
        embedding_identity TEXT NOT NULL,-- provider:model:role:prefix/version identity
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (fact_id, embedding_identity)
    )
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import List, Optional, Tuple

import numpy as np

from .embeddings import bytes_to_embedding, embedding_to_bytes

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_id    INTEGER NOT NULL,
    embedding  BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    embedding_identity TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fact_id, embedding_identity)
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_fact_embeddings_fact_id
    ON fact_embeddings(fact_id);
"""

_CREATE_IDENTITY_DIM_INDEX = """
CREATE INDEX IF NOT EXISTS idx_fact_embeddings_identity_dim
    ON fact_embeddings(embedding_identity, dim);
"""


class EmbedStore:
    """CRUD for the fact_embeddings table.

    Attaches to the same SQLite connection used by MemoryStore by accepting
    either a path string or an existing ``sqlite3.Connection``.  Sharing the
    connection avoids locking issues on WAL-mode databases.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedding_identity: Optional[str] = None,
        lock: Optional["threading.RLock"] = None,
    ) -> None:
        self._conn = conn
        # Share the parent store's lock when provided so embedding writes and the
        # parent's fact writes serialize on the same connection (the RLock is
        # reentrant, so nesting is safe). Falls back to a private lock for tests.
        self._lock = lock if lock is not None else threading.RLock()
        self._embedding_identity = embedding_identity
        self._cache_ids: Optional[np.ndarray] = None
        self._cache_matrix: Optional[np.ndarray] = None
        self._cache_dim: Optional[int] = None
        self._cache_identity: Optional[str] = None
        self._init_table()

    def _invalidate_cache(self) -> None:
        """Drop the in-process embedding matrix cache after writes."""
        self._cache_ids = None
        self._cache_matrix = None
        self._cache_dim = None
        self._cache_identity = None

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_table(self) -> None:
        self._conn.execute(_CREATE_TABLE)
        self._ensure_schema_v2()
        self._conn.execute(_CREATE_INDEX)
        self._conn.execute(_CREATE_IDENTITY_DIM_INDEX)
        self._conn.commit()

    def _ensure_schema_v2(self) -> None:
        """Migrate legacy one-vector-per-fact tables to identity-scoped rows.

        Older Holographic+ builds used ``fact_id`` as the primary key and later
        added nullable ``embedding_identity`` metadata. That was safe for
        filtering, but not for side-by-side model shadowing because a second
        model would overwrite the first. Schema v2 uses a composite primary key
        so each fact can keep multiple vector spaces at once.
        """
        info = self._conn.execute("PRAGMA table_info(fact_embeddings)").fetchall()
        cols = {row[1] for row in info}
        if "embedding_identity" not in cols:
            self._conn.execute("ALTER TABLE fact_embeddings ADD COLUMN embedding_identity TEXT")
            info = self._conn.execute("PRAGMA table_info(fact_embeddings)").fetchall()

        pk_cols = [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5] > 0]
        if pk_cols == ["fact_id", "embedding_identity"]:
            return

        legacy_name = "fact_embeddings_legacy_v1"
        existing_tables = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if legacy_name in existing_tables:
            suffix = 1
            while f"{legacy_name}_{suffix}" in existing_tables:
                suffix += 1
            legacy_name = f"{legacy_name}_{suffix}"

        self._conn.execute(f"ALTER TABLE fact_embeddings RENAME TO {legacy_name}")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(
            f"""
            INSERT OR REPLACE INTO fact_embeddings
                (fact_id, embedding, dim, embedding_identity, created_at)
            SELECT
                fact_id,
                embedding,
                dim,
                COALESCE(embedding_identity, 'ollama:qwen3-embedding:8b:document:none:v1'),
                created_at
            FROM {legacy_name}
            """
        )
        self._conn.execute(f"DROP TABLE {legacy_name}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, fact_id: int, vec: np.ndarray, embedding_identity: Optional[str] = None) -> None:
        """Store or replace the embedding for *fact_id*."""
        with self._lock:
            blob = embedding_to_bytes(vec)
            dim = len(vec)
            identity = embedding_identity if embedding_identity is not None else self._embedding_identity
            if not identity:
                identity = "legacy:unknown:document:none:v1"
            self._conn.execute(
                """
                INSERT INTO fact_embeddings (fact_id, embedding, dim, embedding_identity)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(fact_id, embedding_identity) DO UPDATE SET
                    embedding  = excluded.embedding,
                    dim        = excluded.dim,
                    embedding_identity = excluded.embedding_identity,
                    created_at = CURRENT_TIMESTAMP
                """,
                (fact_id, blob, dim, identity),
            )
            self._conn.commit()
            self._invalidate_cache()

    def delete(self, fact_id: int) -> None:
        """Remove embedding for *fact_id* (no-op if not present)."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM fact_embeddings WHERE fact_id = ?", (fact_id,)
            )
            self._conn.commit()
            self._invalidate_cache()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def ids_without_embeddings(self, all_fact_ids: List[int], embedding_identity: Optional[str] = None) -> List[int]:
        """Return the subset of *all_fact_ids* that have no stored embedding."""
        if not all_fact_ids:
            return []
        with self._lock:
            identity = embedding_identity if embedding_identity is not None else self._embedding_identity
            placeholders = ",".join("?" * len(all_fact_ids))
            params = list(all_fact_ids)
            identity_clause = ""
            if identity:
                if self._include_legacy_null_identity(identity):
                    identity_clause = " AND (embedding_identity = ? OR embedding_identity IS NULL)"
                    params.append(identity)
                else:
                    identity_clause = " AND embedding_identity = ?"
                    params.append(identity)
            rows = self._conn.execute(
                f"SELECT fact_id FROM fact_embeddings WHERE fact_id IN ({placeholders}){identity_clause}",
                params,
            ).fetchall()
            have_embeddings = {int(r[0]) for r in rows}
            return [fid for fid in all_fact_ids if fid not in have_embeddings]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _identity_for_storage(self, identity: Optional[str]) -> Optional[str]:
        """Map query/document identities to the stored document vector identity."""
        identity = identity if identity is not None else self._embedding_identity
        if identity:
            return identity.replace(":query:", ":document:")
        return identity

    @staticmethod
    def _include_legacy_null_identity(identity: Optional[str]) -> bool:
        """Legacy rows belong to the historical qwen3-embedding:8b default only."""
        return identity == "ollama:qwen3-embedding:8b:document:none:v1"

    def _embedding_matrix(self, dim: int, embedding_identity: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return cached (fact_ids, normalised_matrix) for embeddings matching dim/identity."""
        identity = self._identity_for_storage(embedding_identity)
        with self._lock:
            if (
                self._cache_ids is not None
                and self._cache_matrix is not None
                and self._cache_dim == dim
                and self._cache_identity == identity
            ):
                return self._cache_ids, self._cache_matrix

            if identity:
                if self._include_legacy_null_identity(identity):
                    rows = self._conn.execute(
                        """
                        SELECT fact_id, embedding FROM fact_embeddings
                        WHERE dim = ?
                          AND (embedding_identity = ? OR embedding_identity IS NULL)
                        """,
                        (dim, identity),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        """
                        SELECT fact_id, embedding FROM fact_embeddings
                        WHERE dim = ? AND embedding_identity = ?
                        """,
                        (dim, identity),
                    ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT fact_id, embedding FROM fact_embeddings WHERE dim = ?",
                    (dim,),
                ).fetchall()
            if not rows:
                self._cache_ids = np.array([], dtype=np.int64)
                self._cache_matrix = np.empty((0, dim), dtype=np.float32)
                self._cache_dim = dim
                self._cache_identity = identity
                return self._cache_ids, self._cache_matrix

            fact_ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
            matrix = np.stack([bytes_to_embedding(r[1]) for r in rows]).astype(np.float32, copy=False)

            row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            row_norms = np.where(row_norms < 1e-9, 1.0, row_norms)
            matrix_normed = matrix / row_norms

            self._cache_ids = fact_ids
            self._cache_matrix = matrix_normed
            self._cache_dim = dim
            self._cache_identity = identity
            return self._cache_ids, self._cache_matrix

    def score_all(
        self, query_vec: np.ndarray, embedding_identity: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        """Compute cosine similarity between *query_vec* and every stored embedding.

        Returns a list of (fact_id, similarity) pairs sorted by similarity desc.
        Similarity is in [-1, 1] but practically [0, 1] for pre-normalised vectors.
        Only embeddings with the same dimension as *query_vec* are scored; this
        avoids crashes during canary/migration periods with mixed dimensions.
        """
        if query_vec is None or len(query_vec) == 0:
            return []

        q = query_vec.astype(np.float32, copy=False)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-9:
            q = q / q_norm

        fact_ids, matrix_normed = self._embedding_matrix(len(q), embedding_identity=embedding_identity)
        if matrix_normed.size == 0:
            return []

        sims = matrix_normed @ q  # (N,)

        result = sorted(
            zip(fact_ids.astype(int).tolist(), sims.astype(float).tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return result
