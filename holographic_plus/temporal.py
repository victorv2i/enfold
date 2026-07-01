"""Temporal validity: invalidate-not-delete supersession for facts.

Additive-only schema on top of the parent ``facts`` table (Graphiti-inspired,
no graph DB): a fact is never deleted when it is superseded by a newer value,
it is marked ``invalid_at`` and linked to its replacement via
``superseded_by``, so the history stays queryable. Search excludes invalid
facts by default (``temporal_filter``, on by default); turning it off
reproduces the pre-temporal ranking exactly.

This module owns three things:

  - ``ensure_temporal_schema``: idempotent additive migration (safe to call
    on every ``initialize()``, including against a live-size store).
  - ``_is_value_update`` / ``find_value_update_target``: detects the case the
    write-time near-duplicate gate deliberately lets through, an incoming
    fact whose CONTENT words match an existing fact but whose VALUE tokens
    (numbers, ports, SHAs, ids, versions) differ, i.e. a value change rather
    than a restatement or an unrelated new fact.
  - ``supersede``: marks the old fact invalid and links it to the new one.
  - ``fact_history``: walks the ``superseded_by`` chain both directions from
    any fact in it.

The legacy ``SUPERSEDED <date>:`` content-prefix convention
(``_is_superseded`` in ``__init__.py``) is untouched: old facts written under
that convention keep working via the string filter, this module only governs
supersession going forward.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

# Reuse the same content-token / value-token split as the near-duplicate gate
# so a value-update decision is consistent with the dedup gate it complements.
from . import _content_tokens, _value_tokens


def ensure_temporal_schema(conn: sqlite3.Connection) -> None:
    """Add the temporal-validity columns to ``facts`` if not already present.

    Idempotent: checks ``PRAGMA table_info`` first, so re-running against an
    already-migrated database (including a fresh v2 schema) is a no-op. Each
    column is nullable with no default, so every pre-existing row reads as
    "currently valid" (``invalid_at IS NULL``) without a backfill pass, and a
    live database of a few thousand facts migrates in well under a second (3
    ``ALTER TABLE ... ADD COLUMN`` statements, no data rewrite).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
    if "valid_from" not in cols:
        conn.execute("ALTER TABLE facts ADD COLUMN valid_from TIMESTAMP")
    if "invalid_at" not in cols:
        conn.execute("ALTER TABLE facts ADD COLUMN invalid_at TIMESTAMP")
    if "superseded_by" not in cols:
        conn.execute("ALTER TABLE facts ADD COLUMN superseded_by INTEGER")
    conn.commit()


_RESTATEMENT_JACCARD = 0.6


def _is_value_update(content: str, other: str) -> bool:
    """True if *content* is a VALUE UPDATE of *other*: same content words, a
    changed concrete value.

    This is deliberately the complement of the near-duplicate gate
    (``_is_near_duplicate`` / ``_is_semantic_duplicate`` in ``__init__.py``),
    which requires *equal* value tokens. Here the value tokens must DIFFER
    (otherwise it is a duplicate, not an update, and belongs to the dedup
    gate) while the non-value content words stay the same, so "the port is
    3100" -> "the port is 3200" qualifies but an unrelated fact about a
    different topic does not.
    """
    a_values, b_values = _value_tokens(content), _value_tokens(other)
    if a_values == b_values:
        return False  # no value changed: not an update, the dedup gate owns this
    if not a_values or not b_values:
        return False  # need a concrete value on both sides to call it an update
    # Compare content words with the (differing) value tokens themselves
    # removed, so "port 3100" -> "port 3200" is judged on {"port"} == {"port"}.
    a_words = _content_tokens(content) - a_values
    b_words = _content_tokens(other) - b_values
    return a_words == b_words


def find_value_update_target(
    content: str,
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the existing fact among *candidates* that *content* updates, else None.

    *candidates* are dicts with at least ``content`` (the shape returned by
    the provider's hybrid ``search()``). Skips a candidate already superseded
    (has a truthy ``superseded_by``), so a chain never re-supersedes a link
    that already exists.
    """
    for candidate in candidates:
        if candidate.get("superseded_by"):
            continue
        if _is_value_update(content, candidate.get("content", "")):
            return candidate
    return None


def supersede(conn: sqlite3.Connection, old_fact_id: int, new_fact_id: int) -> None:
    """Mark *old_fact_id* invalid, superseded by *new_fact_id*.

    Idempotent no-op if the old fact does not exist or is already superseded.
    """
    conn.execute(
        """
        UPDATE facts
           SET invalid_at = CURRENT_TIMESTAMP,
               superseded_by = ?
         WHERE fact_id = ? AND invalid_at IS NULL
        """,
        (new_fact_id, old_fact_id),
    )
    conn.commit()


def fact_history(conn: sqlite3.Connection, fact_id: int) -> List[Dict[str, Any]]:
    """Return the full supersession chain containing *fact_id*, oldest first.

    Walks ``superseded_by`` forward (newer replacements) and backward (older
    facts this one superseded) from *fact_id*, so the caller can pass any
    fact in a chain and get the same complete history. Each row is a plain
    dict of the ``facts`` columns; ``fact_id`` itself is always included even
    when it has no history (a chain of one).
    """
    row = conn.execute(
        "SELECT * FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    if row is None:
        return []

    chain: Dict[int, Dict[str, Any]] = {int(fact_id): dict(row)}

    # Walk backward: facts whose superseded_by eventually points at something
    # already in the chain.
    changed = True
    while changed:
        changed = False
        ids = tuple(chain.keys())
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT * FROM facts WHERE superseded_by IN ({placeholders})",
            ids,
        ).fetchall()
        for r in rows:
            fid = int(r["fact_id"])
            if fid not in chain:
                chain[fid] = dict(r)
                changed = True

    # Walk forward: follow superseded_by from every fact currently in the chain.
    changed = True
    while changed:
        changed = False
        next_ids = {
            int(f["superseded_by"]) for f in chain.values() if f.get("superseded_by")
        }
        next_ids -= set(chain.keys())
        if not next_ids:
            break
        placeholders = ",".join("?" * len(next_ids))
        rows = conn.execute(
            f"SELECT * FROM facts WHERE fact_id IN ({placeholders})",
            tuple(next_ids),
        ).fetchall()
        for r in rows:
            fid = int(r["fact_id"])
            if fid not in chain:
                chain[fid] = dict(r)
                changed = True

    return sorted(chain.values(), key=lambda f: (f.get("created_at") or "", f["fact_id"]))
