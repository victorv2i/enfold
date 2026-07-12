"""Offline near-duplicate cluster merge tool.

Standalone maintenance script for a ``enfold`` SQLite fact store.
Clusters facts by dense-embedding cosine similarity (union-find), picks one
survivor per cluster, merges retrieval/helpful counts and tags onto it, and
deletes the losers. Meant to run against a *copy* of a fact store, never the
live path, and defaults to a dry run.

Survivor selection:
    - If a cluster spans both a pre-existing fact (created before
      *flood_cutoff*) and a flood fact (created at/after it), the
      pre-existing fact survives: it already carries real trust/retrieval
      history, and the flood fact is a paraphrase restatement of it.
    - Otherwise, the fact with the highest ``trust_score * retrieval_count``
      survives; ties break on the earliest ``created_at`` (the original
      statement, not a later restatement).

Guard rails on ``execute_merge``:
    - ``dry_run`` defaults to True. A real run must be requested explicitly.
    - Refuses any *db_path* under a ``.hermes`` directory, live or not,
      checking both the literal path and its resolved (realpath) form so a
      symlink cannot bypass the check.
    - Requires *backup_path* to already exist on disk.
    - Refuses unless the computed drop count falls within
      [*expected_drop_min*, *expected_drop_max*].
    - Refuses if the computed drop count exceeds *max_drop_fraction*
      (default 0.5) of the starting active fact count, even if it is
      within the absolute band above.
    - A real run ends with ``PRAGMA integrity_check`` and an FTS5 integrity
      check, plus ``PRAGMA wal_checkpoint(TRUNCATE)``.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embed_store import EmbedStore
from .sqlite_vec_index import SQLiteVecIndex

_SUPERSEDED_PREFIXES = ("superseded", "stale/disabled", "historical/superseded")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NEGATION_WORDS = frozenset(("not", "no", "never", "without"))
_DATE_WORDS = frozenset(
    "january february march april may june july august september october november december "
    "monday tuesday wednesday thursday friday saturday sunday today tomorrow yesterday"
    .split()
)
_STATE_WORD_GROUPS = (
    frozenset(("enabled", "disabled")),
    frozenset(("active", "inactive", "archived")),
    frozenset(("on", "off")),
    frozenset(("open", "closed")),
    frozenset(("paused", "resumed")),
    frozenset(("up", "down")),
    frozenset(("alive", "dead")),
    frozenset(("started", "stopped")),
    frozenset(("pending", "running", "completed", "failed", "succeeded")),
    frozenset(("deployed", "undeployed", "installed", "uninstalled")),
    frozenset(("available", "unavailable")),
    frozenset(("approved", "rejected", "accepted", "denied")),
)
_STATE_WORDS = frozenset().union(*_STATE_WORD_GROUPS)


class GuardRailError(Exception):
    """Raised when execute_merge refuses to run for safety reasons."""


@dataclass(frozen=True, slots=True)
class NearDuplicateCandidate:
    """An active FTS-prefiltered fact whose stored vector matches a write."""

    fact_id: int
    trust_score: float
    created_at: str
    cosine: float


def _tokens(content: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall((content or "").lower()))


def _value_tokens(content: str) -> tuple[str, ...]:
    """Concrete values, including dates, versions, ids, and plain numbers."""
    return tuple(
        token for token in _tokens(content)
        if any(char.isdigit() for char in token) or token in _DATE_WORDS
    )


def _state_tokens(content: str) -> frozenset[str]:
    tokens = _tokens(content)
    states = set(tokens) & _STATE_WORDS
    # ``on`` is commonly a preposition ("listens on port 3100"), whereas
    # ``is on`` and ``turned on`` are lifecycle assertions.  Do not let the
    # preposition turn an otherwise safe paraphrase into a false state change.
    for index, token in enumerate(tokens):
        if token in {"on", "off"} and (
            index == 0 or tokens[index - 1] not in {"is", "was", "turned", "set"}
        ):
            states.discard(token)
    return frozenset(states)


def safe_to_merge_near_duplicate(content: str, other: str) -> bool:
    """Return False for textual signals that can denote a changed fact.

    Dense embeddings blur numeric, temporal, polarity, and lifecycle changes.
    A difference in any of those signals is therefore an absolute block, not a
    score penalty.  The caller still applies its cosine threshold afterwards.
    """
    if _value_tokens(content) != _value_tokens(other):
        return False
    if bool(frozenset(_tokens(content)) & _NEGATION_WORDS) != bool(
        frozenset(_tokens(other)) & _NEGATION_WORDS
    ):
        return False
    return _state_tokens(content) == _state_tokens(other)


def find_write_near_duplicates(
    conn: sqlite3.Connection,
    *,
    content: str,
    scope: str,
    query_embedding: np.ndarray,
    threshold: float,
    candidate_limit: int,
    embedding_identity: Optional[str] = None,
) -> List[NearDuplicateCandidate]:
    """Find safe write-time near duplicates without scanning a whole scope.

    FTS supplies a bounded lexical candidate set before any vector is decoded.
    If FTS or stored embeddings are unavailable, this returns no candidates so
    the write path can retain its exact-match fallback rather than blocking a
    fact on incomplete embedding work.
    """
    terms = tuple(dict.fromkeys(_tokens(content)))
    vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if not terms or vector.size == 0 or not np.isfinite(vector).all():
        return []
    if candidate_limit <= 0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("near-duplicate search configuration is invalid")
    match_query = " OR ".join(f'"{term}"' for term in terms)
    identity_clause = ""
    params: list[object] = [match_query, scope]
    if embedding_identity is not None:
        identity_clause = " AND e.embedding_identity = ?"
        params.append(embedding_identity)
    params.append(candidate_limit)
    try:
        rows = conn.execute(
            f"""
            SELECT f.fact_id, f.content, f.trust_score, f.created_at,
                   e.embedding, e.dim
            FROM facts_fts
            JOIN facts AS f ON f.fact_id = facts_fts.rowid
            JOIN fact_embeddings AS e ON e.fact_id = f.fact_id
            WHERE facts_fts MATCH ? AND f.scope = ?
              AND f.invalid_at IS NULL AND f.superseded_by IS NULL
              AND f.conflict_group IS NULL{identity_clause}
            ORDER BY bm25(facts_fts), f.fact_id
            LIMIT ?
            """,
            params,
        ).fetchall()
    except sqlite3.DatabaseError:
        return []

    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0:
        return []
    matches: List[NearDuplicateCandidate] = []
    for row in rows:
        if int(row["dim"]) != vector.size or not safe_to_merge_near_duplicate(
            content, str(row["content"])
        ):
            continue
        stored = np.frombuffer(row["embedding"], dtype="<f4")
        if stored.size != vector.size:
            continue
        stored_norm = float(np.linalg.norm(stored))
        if stored_norm == 0.0:
            continue
        cosine = float(np.dot(vector, stored) / (vector_norm * stored_norm))
        if cosine >= threshold:
            matches.append(
                NearDuplicateCandidate(
                    fact_id=int(row["fact_id"]),
                    trust_score=float(row["trust_score"]),
                    created_at=str(row["created_at"]),
                    cosine=cosine,
                )
            )
    return matches


# ---------------------------------------------------------------------------
# Union-find clustering
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, ids: Sequence[int]) -> None:
        self._parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def build_clusters(
    conn: sqlite3.Connection,
    threshold: float,
    embedding_identity: Optional[str] = None,
) -> List[List[int]]:
    """Return groups (size >= 2) of fact_ids whose embeddings are near-duplicates.

    Computes full pairwise cosine similarity over every embedded fact and
    union-finds pairs at or above *threshold*. Facts with no embedding, or
    whose only neighbor is themselves, are omitted (singletons aren't
    clusters).
    """
    embed_store = EmbedStore(conn, embedding_identity=embedding_identity)
    fact_ids_arr, matrix = embed_store._embedding_matrix(
        _dim(conn, embedding_identity), embedding_identity=embedding_identity
    )
    active_ids = _active_fact_ids(conn)
    fact_ids = [
        fid for fid in fact_ids_arr.astype(int).tolist()
        if fid in active_ids
    ]
    if len(fact_ids) != len(fact_ids_arr):
        keep = [
            i for i, fid in enumerate(fact_ids_arr.astype(int).tolist())
            if fid in active_ids
        ]
        matrix = matrix[keep]
    if len(fact_ids) < 2:
        return []

    uf = _UnionFind(fact_ids)
    n = len(fact_ids)
    sims = matrix @ matrix.T
    for i in range(n):
        row = sims[i]
        for j in range(i + 1, n):
            if row[j] >= threshold:
                uf.union(fact_ids[i], fact_ids[j])

    groups: Dict[int, List[int]] = {}
    for fid in fact_ids:
        groups.setdefault(uf.find(fid), []).append(fid)

    return [members for members in groups.values() if len(members) >= 2]


def _dim(conn: sqlite3.Connection, embedding_identity: Optional[str]) -> int:
    row = conn.execute(
        "SELECT dim FROM fact_embeddings WHERE embedding_identity = ? LIMIT 1"
        if embedding_identity
        else "SELECT dim FROM fact_embeddings LIMIT 1",
        (embedding_identity,) if embedding_identity else (),
    ).fetchone()
    if row is None:
        return 0
    return int(row[0])


def _is_legacy_superseded(content: str) -> bool:
    return (content or "").lstrip().lower().startswith(_SUPERSEDED_PREFIXES)


def _fact_table_cols(conn: sqlite3.Connection) -> set:
    return {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}


def _active_fact_ids(conn: sqlite3.Connection) -> set:
    cols = _fact_table_cols(conn)
    where = " WHERE invalid_at IS NULL" if "invalid_at" in cols else ""
    rows = conn.execute(f"SELECT fact_id, content FROM facts{where}").fetchall()
    return {
        int(row["fact_id"])
        for row in rows
        if not _is_legacy_superseded(row["content"])
    }


# ---------------------------------------------------------------------------
# Survivor selection
# ---------------------------------------------------------------------------

def _fact_rows(conn: sqlite3.Connection, fact_ids: Sequence[int]) -> Dict[int, sqlite3.Row]:
    placeholders = ",".join("?" * len(fact_ids))
    invalid_at_select = (
        "invalid_at" if "invalid_at" in _fact_table_cols(conn) else "NULL AS invalid_at"
    )
    rows = conn.execute(
        f"SELECT fact_id, content, tags, trust_score, retrieval_count, "
        f"helpful_count, created_at, {invalid_at_select} "
        f"FROM facts WHERE fact_id IN ({placeholders})",
        list(fact_ids),
    ).fetchall()
    return {int(r["fact_id"]): r for r in rows}


def choose_survivor(
    conn: sqlite3.Connection,
    fact_ids: Sequence[int],
    flood_cutoff: str,
) -> Tuple[int, List[int]]:
    """Pick the survivor fact_id for a cluster; return (survivor, losers).

    See module docstring for the selection rule.
    """
    rows = _fact_rows(conn, fact_ids)
    active_ids = [
        fid for fid in fact_ids
        if rows[fid]["invalid_at"] is None
        and not _is_legacy_superseded(rows[fid]["content"])
    ]
    if active_ids:
        fact_ids = active_ids

    pre_existing = [fid for fid in fact_ids if rows[fid]["created_at"] < flood_cutoff]
    flood = [fid for fid in fact_ids if rows[fid]["created_at"] >= flood_cutoff]

    if pre_existing and flood:
        candidates = pre_existing
    else:
        candidates = list(fact_ids)

    def _key(fid: int):
        r = rows[fid]
        score = float(r["trust_score"]) * float(r["retrieval_count"])
        # Higher score first, then earliest created_at first.
        return (-score, r["created_at"])

    survivor = min(candidates, key=_key)
    losers = [fid for fid in fact_ids if fid != survivor]
    return survivor, losers


# ---------------------------------------------------------------------------
# Merge plan
# ---------------------------------------------------------------------------

@dataclass
class ClusterMerge:
    survivor_id: int
    loser_ids: List[int]
    merged_retrieval_count: int
    merged_helpful_count: int
    merged_tags: str
    suspicious: bool = False


@dataclass
class MergePlan:
    clusters: List[ClusterMerge] = field(default_factory=list)
    starting_fact_count: int = 0

    @property
    def drop_count(self) -> int:
        return sum(len(c.loser_ids) for c in self.clusters)

    @property
    def projected_final_count(self) -> int:
        return self.starting_fact_count - self.drop_count


def _merge_tags(*tag_strings: str) -> str:
    seen: List[str] = []
    for tags in tag_strings:
        for tag in (tags or "").split(","):
            tag = tag.strip()
            if tag and tag not in seen:
                seen.append(tag)
    return ",".join(seen)


def plan_merge(
    conn: sqlite3.Connection,
    threshold: float,
    flood_cutoff: str,
    embedding_identity: Optional[str] = None,
    suspicious_cluster_size: int = 25,
) -> MergePlan:
    """Build a MergePlan: one ClusterMerge per near-duplicate cluster.

    A cluster is flagged *suspicious* when it has more members than
    *suspicious_cluster_size* -- large enough that it's worth a human
    spot-check before trusting the merge, rather than assuming every member
    is a genuine paraphrase of the same statement.
    """
    starting = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
    clusters = build_clusters(conn, threshold, embedding_identity=embedding_identity)

    plan = MergePlan(starting_fact_count=starting)
    for members in clusters:
        survivor, losers = choose_survivor(conn, members, flood_cutoff)
        rows = _fact_rows(conn, members)
        merged_retrieval = sum(int(rows[fid]["retrieval_count"]) for fid in members)
        merged_helpful = sum(int(rows[fid]["helpful_count"]) for fid in members)
        merged_tags = _merge_tags(*(rows[fid]["tags"] for fid in members))
        plan.clusters.append(
            ClusterMerge(
                survivor_id=survivor,
                loser_ids=losers,
                merged_retrieval_count=merged_retrieval,
                merged_helpful_count=merged_helpful,
                merged_tags=merged_tags,
                suspicious=len(members) > suspicious_cluster_size,
            )
        )
    return plan


# ---------------------------------------------------------------------------
# Execution (guarded)
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    dry_run: bool
    drop_count: int
    projected_final_count: int
    final_fact_count: Optional[int] = None
    integrity_ok: Optional[bool] = None
    fts_integrity_ok: Optional[bool] = None


def _refuse_if_live_path(db_path: str) -> None:
    literal_parts = str(db_path).replace("\\", "/").split("/")
    resolved_parts = os.path.realpath(db_path).replace("\\", "/").split("/")
    if ".hermes" in literal_parts or ".hermes" in resolved_parts:
        raise GuardRailError(
            f"refusing to run against a path under .hermes ({db_path}); "
            "this tool only ever runs against a copy"
        )


def execute_merge(
    db_path: str,
    threshold: float,
    flood_cutoff: str,
    embedding_identity: Optional[str],
    backup_path: str,
    expected_drop_min: int,
    expected_drop_max: int,
    dry_run: bool = True,
    suspicious_cluster_size: int = 25,
    max_drop_fraction: float = 0.5,
) -> MergeResult:
    """Plan and (optionally) execute the merge against *db_path*.

    Always refuses a path under ``.hermes``. A non-dry-run additionally
    requires *backup_path* to exist and the computed drop count to fall
    inside [*expected_drop_min*, *expected_drop_max*] AND to not exceed
    *max_drop_fraction* of the starting active fact count (default 0.5, i.e.
    refuse a plan that would drop more than half the store even if it is
    still within the absolute band), then deletes losers, merges their
    counts/tags onto each survivor, drops their embeddings, and runs an
    integrity check + wal_checkpoint.
    """
    _refuse_if_live_path(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        plan = plan_merge(
            conn, threshold, flood_cutoff,
            embedding_identity=embedding_identity,
            suspicious_cluster_size=suspicious_cluster_size,
        )

        if not dry_run:
            if not os.path.exists(backup_path):
                raise GuardRailError(
                    f"refusing to run: backup file does not exist ({backup_path})"
                )
            if not (expected_drop_min <= plan.drop_count <= expected_drop_max):
                raise GuardRailError(
                    f"refusing to run: drop count {plan.drop_count} is outside "
                    f"the expected band [{expected_drop_min}, {expected_drop_max}]"
                )
            if plan.starting_fact_count > 0 and (
                plan.drop_count > max_drop_fraction * plan.starting_fact_count
            ):
                raise GuardRailError(
                    f"refusing to run: drop count {plan.drop_count} exceeds the "
                    f"relative cap of {max_drop_fraction:.0%} of the "
                    f"{plan.starting_fact_count} active facts"
                )
            _apply_merge(conn, plan)
            integrity_ok = _integrity_check(conn)
            fts_ok = _fts_integrity_check(conn)
            _wal_checkpoint(conn)
            final_count = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
            return MergeResult(
                dry_run=False,
                drop_count=plan.drop_count,
                projected_final_count=plan.projected_final_count,
                final_fact_count=final_count,
                integrity_ok=integrity_ok,
                fts_integrity_ok=fts_ok,
            )

        return MergeResult(
            dry_run=True,
            drop_count=plan.drop_count,
            projected_final_count=plan.projected_final_count,
        )
    finally:
        conn.close()


def _apply_merge(conn: sqlite3.Connection, plan: MergePlan) -> None:
    vector_index = SQLiteVecIndex.open_configured(conn, warn=False)
    for cluster in plan.clusters:
        conn.execute(
            "UPDATE facts SET retrieval_count = ?, helpful_count = ?, tags = ? "
            "WHERE fact_id = ?",
            (
                cluster.merged_retrieval_count,
                cluster.merged_helpful_count,
                cluster.merged_tags,
                cluster.survivor_id,
            ),
        )
        for loser_id in cluster.loser_ids:
            conn.execute("DELETE FROM facts WHERE fact_id = ?", (loser_id,))
            conn.execute("DELETE FROM fact_embeddings WHERE fact_id = ?", (loser_id,))
            if vector_index is not None:
                vector_index.delete_in_transaction(loser_id)
            conn.execute("DELETE FROM fact_entities WHERE fact_id = ?", (loser_id,))
    conn.commit()


def _integrity_check(conn: sqlite3.Connection) -> bool:
    row = conn.execute("PRAGMA integrity_check").fetchone()
    return bool(row) and row[0] == "ok"


def _fts_integrity_check(conn: sqlite3.Connection) -> bool:
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'facts_fts'"
        ).fetchall()
    }
    if "facts_fts" not in tables:
        return True
    try:
        conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('integrity-check')")
        conn.commit()
        return True
    except sqlite3.DatabaseError:
        return False


def _wal_checkpoint(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.DatabaseError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", help="path to the fact store copy to clean up")
    parser.add_argument("--threshold", type=float, default=0.92,
                         help="cosine similarity threshold for clustering (default 0.92)")
    parser.add_argument("--flood-cutoff", required=True,
                         help="created_at cutoff (YYYY-MM-DD HH:MM:SS) marking pre-existing vs flood facts")
    parser.add_argument("--embedding-identity", default=None,
                         help="restrict clustering to one embedding_identity (default: any)")
    parser.add_argument("--backup-path", required=True,
                         help="path to a pre-existing backup of db_path (required for --execute)")
    parser.add_argument("--expected-drop-min", type=int, default=0)
    parser.add_argument("--expected-drop-max", type=int, default=10_000)
    parser.add_argument("--max-drop-fraction", type=float, default=0.5,
                         help="refuse if drop count exceeds this fraction of active facts (default 0.5)")
    parser.add_argument("--execute", action="store_true",
                         help="actually perform the merge (default is dry-run)")
    args = parser.parse_args(argv)

    result = execute_merge(
        args.db_path,
        threshold=args.threshold,
        flood_cutoff=args.flood_cutoff,
        embedding_identity=args.embedding_identity,
        backup_path=args.backup_path,
        expected_drop_min=args.expected_drop_min,
        expected_drop_max=args.expected_drop_max,
        max_drop_fraction=args.max_drop_fraction,
        dry_run=not args.execute,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
