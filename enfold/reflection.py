"""Sleep-time reflection: the connection-drawing insight layer.

The reflection layer opportunistically looks for related ACTIVE facts that,
taken together, imply something durable neither one states on its own (e.g.
two facts about a person's move and new job implying a life transition), and
store that synthesis as a new fact so retrieval can surface the connection
directly instead of relying on the agent to re-derive it every time.

Strictly grounded and rate-limited:

  - Candidate clusters are either facts that share an entity (via
    ``fact_entities``) or facts in each other's cosine neighborhood
    (0.75-0.92: related, but not near-duplicates the dedup gate already
    consolidates). Capped at ``reflection_max_clusters`` per run.
  - Exactly one LLM call per cluster, with a strict prompt: synthesize AT
    MOST ONE insight, MUST cite the source fact ids, MUST answer ``NONE``
    when nothing genuine emerges, forbidden to introduce facts not entailed
    by the sources.
  - The response is parsed defensively (mirrors ``llm_extract._extract_json_array``)
    and REJECTED outright if it cites no sources, or cites a fact id outside
    the cluster: an insight with no traceable grounding is worse than no
    insight.
  - Accepted insights are stored as ordinary facts (category ``insight``,
    tags carrying ``source_facts:<id>,<id>,...``) through the SAME
    near-duplicate dedup gate as everything else, so a reflection pass can
    never flood the store.
  - A durable ``reflection_meta`` table persists ``last_run_at`` so a
    restart does not reset the min-interval clock.
  - If a fact an insight cites is later superseded, ``invalidate_insights_citing``
    marks that insight stale too (the temporal layer's invalidate-not-delete,
    not a delete), so a synthesis built on outdated premises does not linger
    as if still current.

Config-gated OFF by default (``reflection_enabled: false``); a deployer opts
in explicitly.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SUPERSEDED_PREFIXES = ("superseded", "stale/disabled", "historical/superseded")


# ── Schema: durable min-interval clock ──────────────────────────────────────

def ensure_reflection_schema(conn: sqlite3.Connection) -> None:
    """Create the ``reflection_meta`` key/value table if not already present.

    Idempotent, mirrors the additive-migration style of ``extract_queue`` and
    ``embed_store``: a single small table, created lazily, the parent schema
    untouched.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reflection_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()


def get_last_run_at(conn: sqlite3.Connection) -> Optional[float]:
    """Return the epoch timestamp of the last reflection run, or None."""
    row = conn.execute(
        "SELECT value FROM reflection_meta WHERE key = 'last_run_at'"
    ).fetchone()
    if row is None or row[0] is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def set_last_run_at(conn: sqlite3.Connection, epoch: float) -> None:
    """Persist *epoch* as the last reflection run time."""
    conn.execute(
        """
        INSERT INTO reflection_meta (key, value) VALUES ('last_run_at', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(epoch),),
    )
    conn.commit()


# ── Cluster selection ────────────────────────────────────────────────────────

def _is_legacy_superseded(content: str) -> bool:
    return (content or "").lstrip().lower().startswith(_SUPERSEDED_PREFIXES)


def _active_fact_rows(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    """All currently-valid source facts as ``{fact_id: {content, category}}``."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
    where = " WHERE invalid_at IS NULL" if "invalid_at" in cols else ""
    rows = conn.execute(f"SELECT fact_id, content, category FROM facts{where}").fetchall()
    return {
        int(r["fact_id"]): dict(r)
        for r in rows
        if r["category"] != "insight" and not _is_legacy_superseded(r["content"])
    }


def _sources_still_active(conn: sqlite3.Connection, source_ids: List[int]) -> bool:
    if not source_ids:
        return False
    placeholders = ",".join("?" * len(source_ids))
    rows = conn.execute(
        f"SELECT fact_id, content, category, invalid_at FROM facts "
        f"WHERE fact_id IN ({placeholders})",
        source_ids,
    ).fetchall()
    if len(rows) != len(set(source_ids)):
        return False
    for row in rows:
        if row["invalid_at"] is not None:
            return False
        if row["category"] == "insight":
            return False
        if _is_legacy_superseded(row["content"]):
            return False
    return True


def _entity_clusters(
    conn: sqlite3.Connection,
    active_ids: set,
    entity_hub_degree_limit: int,
) -> List[List[int]]:
    """Group active facts that share a non-hub entity.

    An entity linked to more than *entity_hub_degree_limit* active facts is a
    hub (a ubiquitous name, a project everything mentions) and is excluded,
    so it cannot fold unrelated facts into one giant undifferentiated
    cluster.
    """
    try:
        rows = conn.execute(
            """
            SELECT fe.entity_id, fe.fact_id FROM fact_entities fe
            JOIN facts f ON f.fact_id = fe.fact_id
            """
        ).fetchall()
    except Exception:
        return []

    by_entity: Dict[int, List[int]] = {}
    for row in rows:
        fid = int(row["fact_id"])
        if fid not in active_ids:
            continue
        by_entity.setdefault(int(row["entity_id"]), []).append(fid)

    clusters = []
    for members in by_entity.values():
        unique = sorted(set(members))
        if len(unique) < 2 or len(unique) > entity_hub_degree_limit:
            continue
        clusters.append(unique)
    return clusters


def _cosine_clusters(
    embed_store,
    active_ids: set,
    embedding_identity: Optional[str],
    cosine_low: float,
    cosine_high: float,
) -> List[List[int]]:
    """Pairs of active facts whose dense cosine similarity is in
    ``[cosine_low, cosine_high)``: related, but not near-duplicates (the
    dedup gate already owns anything at/above *cosine_high*).
    """
    if embed_store is None:
        return []
    try:
        ids_arr, matrix = embed_store._embedding_matrix(
            _first_dim(embed_store, embedding_identity), embedding_identity=embedding_identity
        )
    except Exception:
        return []
    if matrix.size == 0:
        return []

    ids = ids_arr.astype(int).tolist()
    sims = matrix @ matrix.T
    clusters: List[List[int]] = []
    n = len(ids)
    for i in range(n):
        if ids[i] not in active_ids:
            continue
        for j in range(i + 1, n):
            if ids[j] not in active_ids:
                continue
            sim = float(sims[i][j])
            if cosine_low <= sim < cosine_high:
                clusters.append(sorted({ids[i], ids[j]}))
    return clusters


def _first_dim(embed_store, embedding_identity: Optional[str]) -> int:
    conn = embed_store._conn
    row = conn.execute(
        "SELECT dim FROM fact_embeddings WHERE embedding_identity = ? LIMIT 1"
        if embedding_identity
        else "SELECT dim FROM fact_embeddings LIMIT 1",
        (embedding_identity,) if embedding_identity else (),
    ).fetchone()
    return int(row[0]) if row is not None else 0


def select_clusters(
    conn: sqlite3.Connection,
    max_clusters: int = 3,
    embed_store=None,
    embedding_identity: Optional[str] = None,
    cosine_low: float = 0.75,
    cosine_high: float = 0.92,
    entity_hub_degree_limit: int = 25,
) -> List[List[int]]:
    """Return up to *max_clusters* candidate clusters of related ACTIVE fact ids.

    Entity-sharing clusters are tried first (cheap, no embeddings required),
    then cosine-neighborhood pairs fill any remaining slots. A fact already
    superseded (``invalid_at`` set) is never included in a cluster.
    """
    active = _active_fact_rows(conn)
    active_ids = set(active.keys())
    if len(active_ids) < 2:
        return []

    clusters = _entity_clusters(conn, active_ids, entity_hub_degree_limit)
    if len(clusters) < max_clusters:
        clusters += _cosine_clusters(
            embed_store, active_ids, embedding_identity, cosine_low, cosine_high
        )
    return clusters[:max_clusters]


# ── Prompt + defensive parse ─────────────────────────────────────────────────

_SYSTEM = """\
You are a memory reflection assistant for an AI agent called Hermes.
You will be shown a small cluster of related facts already in long-term memory.

Your job: decide whether these facts, taken TOGETHER, imply ONE durable
higher-order insight that is NOT already stated by any single fact alone.

STRICT RULES:
1. Output AT MOST ONE insight. Quality over quantity.
2. The insight MUST be entailed by the given facts. Never introduce a new
   name, date, place, or claim that is not already present in the sources.
3. You MUST cite every source fact id the insight draws on, using their
   numeric ids as given.
4. If there is no genuine higher-order insight, i.e. the facts are merely
   related but combining them adds nothing, output exactly: NONE
5. No meta-commentary. State the insight directly, as a fact.

OUTPUT: Respond with a single JSON object only, no prose, no markdown fences.
{"insight": "<the synthesized insight, or NONE>", "source_fact_ids": [<ids the insight draws on>]}

If nothing is worth synthesizing, output exactly: NONE
"""

_USER_TEMPLATE = """\
Here are the related facts (id: content):
{facts}

Decide whether a genuine higher-order insight follows from these facts together.
"""


def _format_cluster(facts: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{f['fact_id']}: {f['content']}" for f in facts)


def _parse_reflection_response(raw: str, valid_ids) -> Optional[Dict[str, Any]]:
    """Parse and grounding-check a reflection response.

    Returns ``{"insight": str, "source_fact_ids": [int, ...]}`` only when the
    response is valid JSON, the insight is non-empty and not the literal
    "NONE", and every cited id is a member of *valid_ids* with at least one
    citation present. Any other shape (garbage, missing citations, citations
    outside the cluster, an explicit NONE) returns None: reflection is
    strictly grounded, so an ungrounded or unparseable response is always
    treated as "nothing to add" rather than guessed at.
    """
    text = (raw or "").strip()
    if not text or text.upper() == "NONE":
        return None

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        block = _extract_json_object(text)
        if block is None:
            return None
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict):
        return None

    insight = str(parsed.get("insight", "")).strip()
    if not insight or insight.upper() == "NONE":
        return None

    raw_ids = parsed.get("source_fact_ids", [])
    if not isinstance(raw_ids, list) or not raw_ids:
        return None

    try:
        cited = [int(i) for i in raw_ids]
    except (TypeError, ValueError):
        return None

    valid = set(int(v) for v in valid_ids)
    if not cited or any(i not in valid for i in cited):
        return None

    return {"insight": insight, "source_fact_ids": cited}


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced top-level ``{...}`` block in *text*, or None.

    Mirrors ``llm_extract._extract_json_array``'s bracket-balanced,
    string-literal-aware scan, but for a single JSON object.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def reflect_on_cluster(
    facts: List[Dict[str, Any]],
    *,
    provider: str,
    model: str,
    effort: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """One LLM call over *facts*; returns a grounded insight dict or None.

    *facts* are dicts with at least ``fact_id`` and ``content``. Raises on
    transport/LLM failure (mirrors ``extract_facts_from_transcript``) so a
    caller integrating this into a retryable worker keeps that option; the
    provider's opportunistic ``run_reflection`` catches and logs per-cluster
    failures instead of aborting the whole run.
    """
    if not provider or not model or len(facts) < 2:
        return None

    from agent.auxiliary_client import call_llm

    user_msg = _USER_TEMPLATE.format(facts=_format_cluster(facts))
    resp = call_llm(
        provider=provider,
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=400,
        timeout=180,
        extra_body=({"reasoning": {"effort": effort}} if effort else None),
    )
    raw = resp.choices[0].message.content or ""
    valid_ids = {f["fact_id"] for f in facts}
    return _parse_reflection_response(raw, valid_ids)


# ── Invalidation cascade ─────────────────────────────────────────────────────

_SOURCE_FACTS_RE = re.compile(r"source_facts:([0-9,]+)")


def _cited_ids(tags: str) -> List[int]:
    match = _SOURCE_FACTS_RE.search(tags or "")
    if not match:
        return []
    return [int(t) for t in match.group(1).split(",") if t.strip().isdigit()]


def invalidate_insights_citing(conn: sqlite3.Connection, source_fact_id: int) -> int:
    """Mark stale any active insight that cites *source_fact_id* as a source.

    Called when a source fact is superseded: a synthesis built on a premise
    that no longer holds should not keep surfacing as current. Uses the same
    invalidate-not-delete mechanism as ``temporal.supersede`` (no
    ``superseded_by`` link, since there is no replacement insight, just
    ``invalid_at``), so the stale insight stays queryable via history.
    Returns the number of insights marked stale.
    """
    rows = conn.execute(
        "SELECT fact_id, tags FROM facts WHERE category = 'insight' AND invalid_at IS NULL"
    ).fetchall()
    stale_ids = [
        int(row["fact_id"]) for row in rows if source_fact_id in _cited_ids(row["tags"])
    ]
    if not stale_ids:
        return 0
    placeholders = ",".join("?" * len(stale_ids))
    conn.execute(
        f"UPDATE facts SET invalid_at = CURRENT_TIMESTAMP WHERE fact_id IN ({placeholders})",
        stale_ids,
    )
    conn.commit()
    return len(stale_ids)


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_reflection(
    conn: sqlite3.Connection,
    *,
    now: float,
    enabled: bool,
    interval_hours: float,
    max_clusters: int,
    provider: Optional[str],
    model: Optional[str],
    effort: Optional[str] = None,
    embed_store=None,
    embedding_identity: Optional[str] = None,
    cosine_low: float = 0.75,
    cosine_high: float = 0.92,
    entity_hub_degree_limit: int = 25,
    dedup_check=None,
    insert_fact,
    lock: Optional["threading.RLock"] = None,
) -> int:
    """Run one opportunistic reflection pass; returns the number of insights inserted.

    Inert (no-op, no LLM calls, no schema writes beyond the idempotent
    migration) when *enabled* is False. Otherwise: checks the persisted
    min-interval clock, selects clusters, makes at most one LLM call per
    cluster, and inserts each grounded, non-duplicate insight through
    *insert_fact* (the provider's own fact-add path, so embeddings and entity
    linking happen exactly as they do for any other fact). Per-cluster
    failures are logged and skipped; one bad cluster never aborts the run.

    *lock*, when given, is held only around each DB read/write, the same
    pattern the extraction queue worker uses: an LLM call (network I/O, can
    take seconds) never holds the store lock, so it never blocks a
    concurrent session-thread fact add or search.
    """
    if not enabled:
        return 0
    if not provider or not model:
        return 0

    lock = lock or threading.RLock()

    with lock:
        ensure_reflection_schema(conn)
        last_run = get_last_run_at(conn)
    if last_run is not None and (now - last_run) < interval_hours * 3600:
        return 0

    with lock:
        active = _active_fact_rows(conn)
        clusters = select_clusters(
            conn,
            max_clusters=max_clusters,
            embed_store=embed_store,
            embedding_identity=embedding_identity,
            cosine_low=cosine_low,
            cosine_high=cosine_high,
            entity_hub_degree_limit=entity_hub_degree_limit,
        )

    inserted = 0
    for fact_ids in clusters:
        facts = [
            {"fact_id": fid, "content": active[fid]["content"]}
            for fid in fact_ids
            if fid in active
        ]
        if len(facts) < 2:
            continue
        try:
            result = reflect_on_cluster(facts, provider=provider, model=model, effort=effort)
        except Exception as exc:
            logger.debug("reflection: cluster reflection failed: %s", exc)
            continue
        if result is None:
            continue

        content = result["insight"]
        source_ids = sorted(result["source_fact_ids"])
        tags = "source_facts:" + ",".join(str(i) for i in source_ids)

        with lock:
            if not _sources_still_active(conn, source_ids):
                logger.debug(
                    "reflection: skipped insight with stale source facts %s",
                    source_ids,
                )
                continue

        if dedup_check is not None:
            try:
                dup = dedup_check(content, category="insight")
            except Exception as exc:
                logger.debug("reflection: dedup check failed: %s", exc)
                dup = None
            if dup is not None:
                logger.debug(
                    "reflection: skipped near-duplicate insight of fact %s",
                    dup.get("fact_id"),
                )
                continue

        try:
            fact_id = insert_fact(content, category="insight", tags=tags)
        except Exception as exc:
            logger.debug("reflection: insert failed: %s", exc)
            continue
        if fact_id:
            inserted += 1

    with lock:
        set_last_run_at(conn, now)
    return inserted
