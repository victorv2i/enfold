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
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .embed_store import EmbedStore

_SUPERSEDED_PREFIXES = ("superseded", "stale/disabled", "historical/superseded")


class GuardRailError(Exception):
    """Raised when execute_merge refuses to run for safety reasons."""


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
