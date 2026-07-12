"""Temporal validity: invalidate-not-delete supersession for facts.

Additive-only schema on top of the parent ``facts`` table (graph-inspired,
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
    fact whose concrete value tokens differ, or whose state words flip across
    the same subject context, i.e. a value change rather than a restatement
    or an unrelated new fact.
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
import logging
import re
from typing import Any, Dict, List, Optional

# Keep this module importable before package __init__ has finished executing.
# Hermes' user-plugin loader preloads sibling .py files, so importing token
# helpers from "." here can leave a half-initialized temporal module in
# sys.modules and break the later package import. These helpers intentionally
# mirror the near-duplicate gate's token/state split in __init__.py.
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the is are was were be been being am of to in on at for and or with by "
    "as it its this that these those from into over under has have had do does did "
    "will would can could should may might must not no".split()
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
)
_STATE_WORDS = frozenset().union(*_STATE_WORD_GROUPS)
_NEGATION_WORDS = frozenset(("not", "no", "never", "without"))


def _norm_token_sequence(text: str) -> tuple:
    """Lowercased alphanumeric tokens in text order."""
    return tuple(_WORD_RE.findall((text or "").lower()))


def _norm_tokens(text: str) -> set:
    """Lowercased alphanumeric token set."""
    return set(_norm_token_sequence(text))


def _value_tokens(text: str) -> tuple:
    """Tokens carrying a concrete value (numbers, ports, SHAs, ids, versions)."""
    return tuple(t for t in _norm_token_sequence(text) if any(c.isdigit() for c in t))


def _content_tokens(text: str) -> set:
    """Significant (non-function) words."""
    return _norm_tokens(text) - _STOPWORDS


def _state_words(text: str) -> set:
    return _content_tokens(text) & _STATE_WORDS


def _negation_words(text: str) -> set:
    return _norm_tokens(text) & _NEGATION_WORDS


def _subjectish_tokens(text: str) -> set:
    tokens = _content_tokens(text) - _STATE_WORDS
    expanded = set(tokens)
    for token in tokens:
        for suffix in ("ation", "ing", "ed", "ic", "ion", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                expanded.add(token[: -len(suffix)])
    return expanded


def _has_subjectish_overlap(content: str, other: str) -> bool:
    return bool(_subjectish_tokens(content) & _subjectish_tokens(other))


def _has_opposing_state_words(
    content: str, other: str, *, require_context: bool = True
) -> bool:
    """True when opposite state words appear on opposite sides."""
    if require_context and not _has_subjectish_overlap(content, other):
        return False
    a_states = _state_words(content)
    b_states = _state_words(other)
    if not a_states or not b_states:
        return False
    for group in _STATE_WORD_GROUPS:
        a_group = a_states & group
        b_group = b_states & group
        if a_group and b_group and a_group != b_group:
            return True
    return False


def _has_negation_mismatch(
    content: str, other: str, *, require_context: bool = True
) -> bool:
    """True when one side contains explicit negation and the other does not."""
    if require_context and not _has_subjectish_overlap(content, other):
        return False
    return bool(_negation_words(content)) != bool(_negation_words(other))

logger = logging.getLogger(__name__)


def ensure_temporal_schema(conn: sqlite3.Connection) -> None:
    """Add the temporal-validity columns to ``facts`` if not already present.

    Idempotent: checks ``PRAGMA table_info`` first, so re-running against an
    already-migrated database (including a fresh v2 schema) is a no-op. Each
    column is nullable with no default, so every pre-existing row reads as
    "currently valid" (``invalid_at IS NULL``) without a backfill pass, and a
    live database of a few thousand facts migrates in well under a second (3
    ``ALTER TABLE ... ADD COLUMN`` statements, no data rewrite).

    The check-then-add is racy across two separate processes migrating the
    same fresh db_path at the same moment (both see the column missing, both
    ``ALTER TABLE``, the second raises "duplicate column name"): several
    agents each starting their own MCP server against a brand-new store can
    hit this on first run. ``ADD COLUMN`` on an already-migrated table is
    swallowed as a lost race, not a real failure, so this stays a no-op for
    the loser instead of crashing initialize().
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
    for column, coltype in (
        ("valid_from", "TIMESTAMP"),
        ("invalid_at", "TIMESTAMP"),
        ("superseded_by", "INTEGER"),
    ):
        if column in cols:
            continue
        try:
            conn.execute(f"ALTER TABLE facts ADD COLUMN {column} {coltype}")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
    conn.commit()


_RESTATEMENT_JACCARD = 0.6


def _is_value_update(content: str, other: str) -> bool:
    """True if *content* is a VALUE UPDATE of *other*.

    This is deliberately the complement of the near-duplicate gate
    (``_is_near_duplicate`` / ``_is_semantic_duplicate`` in ``__init__.py``),
    which requires *equal* ordered value tokens and no opposing state words.
    Here differing value tokens with the same non-value content, or opposing
    state words with subject overlap, qualify as an update. Unrelated facts
    still do not.
    """
    a_values, b_values = _value_tokens(content), _value_tokens(other)
    if a_values == b_values:
        return _has_opposing_state_words(content, other) or _has_negation_mismatch(
            content, other
        )
    a_value_set = set(a_values)
    b_value_set = set(b_values)
    if not a_value_set or not b_value_set:
        return _has_opposing_state_words(content, other) or _has_negation_mismatch(
            content, other
        )
    # Compare content words with the differing value tokens and state words
    # removed, so "port 3100" -> "port 3200" is judged on {"port"}.
    a_words = _content_tokens(content) - a_value_set - _state_words(content)
    b_words = _content_tokens(other) - b_value_set - _state_words(other)
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


def supersede(conn: sqlite3.Connection, old_fact_id: int, new_fact_id: int) -> bool:
    """Mark *old_fact_id* invalid, superseded by *new_fact_id*.

    Returns True when the old row was invalidated. Returns False when the old
    fact does not exist or is already superseded.
    """
    cur = conn.execute(
        """
        UPDATE facts
           SET invalid_at = CURRENT_TIMESTAMP,
               superseded_by = ?
         WHERE fact_id = ? AND invalid_at IS NULL
        """,
        (new_fact_id, old_fact_id),
    )
    conn.commit()
    updated = int(cur.rowcount) == 1
    if not updated:
        logger.debug(
            "temporal: supersede no-op old_fact_id=%s new_fact_id=%s",
            old_fact_id,
            new_fact_id,
        )
    return updated


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
