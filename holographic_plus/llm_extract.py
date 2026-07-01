"""LLM-based fact extraction from conversation transcripts.

Uses a configurable LLM, the host agent's own model by default, to extract
3-7 atomic, durable facts from a conversation. Transcripts are formatted by
the provider hooks (_format_conversation), enqueued on the persistent
extract_queue, and processed by the provider's background worker, which calls
extract_facts_from_transcript() and insert_facts().

Design principles:
- Only extract facts that will still be true next week
- One fact = one atomic, self-contained statement
- Deduplicate against existing facts in the store before inserting
- The LLM call raises on failure so the queue worker can retry with backoff
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from plugins.memory.holographic.store import MemoryStore

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a memory extraction assistant for an AI agent called Hermes.
Your job: read a conversation and extract atomic facts worth remembering long-term.

RULES:
1. Extract 3-7 facts. Quality beats quantity: extract 0 if nothing is truly durable.
2. Each fact must be ONE atomic statement (1-3 sentences max, under 400 characters).
3. ONLY extract facts that will still be true next week, no ephemeral task details.
4. No meta-commentary ("the user asked…"). State the fact directly.
5. Assign a category from: user_pref | project | tool | general
6. Assign 2-5 comma-separated tags (lowercase, no spaces).
7. Do NOT extract facts already obvious from names/dates (e.g. "today is Monday").
8. Skip anything that's clearly a one-off instruction to the agent for this session.
9. Emit each distinct fact EXACTLY ONCE. Never output multiple rephrasings or near-duplicates of the same fact (e.g. the same SHA, URL, port, or status stated several ways); choose the single clearest phrasing.
10. Resolve pronouns and vague references to concrete entities so each fact stands alone out of context.
11. If a fact corrects or updates something in the known facts, state the new fact with the current value (the system supersedes the old one); do not restate unchanged known facts.
12. Convert relative time references ("yesterday", "last week", "next month") to absolute dates using TODAY'S DATE (given below), so every fact is unambiguous out of context.

GOOD facts:
- "The user prefers Postgres over MySQL for new projects."
- "The production app deploys to Vercel from the main branch."
- "The user's working hours are roughly 9am-6pm Eastern."

BAD facts (don't extract):
- "The user asked me to summarize a document." (ephemeral task)
- "The conversation was about memory systems." (meta)
- "The user said hello." (trivial)
- The same fact restated several ways, e.g. "X is pinned at SHA abc" + "X uses commit abc" + "X's clone is at abc" (duplication, output it once)

OUTPUT: Respond with a JSON array only. No prose, no markdown fences.
Example:
[
  {"content": "The user prefers pnpm as their package manager for Node projects.", "category": "tool", "tags": "pnpm,node,package-manager"},
  {"content": "The legacy v2 service is archived; active development is on v3.", "category": "project", "tags": "v3,migration,project-status"}
]

If nothing is worth saving, output: []
"""

_USER_TEMPLATE = """\
Today's date is {today}. Here is the conversation to extract facts from.
Existing known facts (do not re-extract these):
{existing_summary}

---CONVERSATION---
{conversation}
---END---

Extract new durable facts not already in the known facts list.
"""


def _format_conversation(messages: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """Render messages as a readable transcript, truncated to max_chars."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if not isinstance(content, str):
            # Handle list content (tool results etc.), flatten to string
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            else:
                content = str(content)
        if not content.strip():
            continue
        # Truncate very long individual messages
        if len(content) > 2000:
            content = content[:2000] + "…[truncated]"
        lines.append(f"{role.upper()}: {content}")

    transcript = "\n\n".join(lines)
    if len(transcript) > max_chars:
        # Keep the last max_chars (most recent = most relevant)
        transcript = "…[earlier content omitted]\n\n" + transcript[-max_chars:]
    return transcript


def _existing_summary(
    store: "MemoryStore",
    limit: int = 40,
    *,
    topic: str = "",
    search_fn=None,
    similar_limit: int = 10,
) -> str:
    """Return a compact summary of existing facts for the dedup prompt.

    Blends the most trusted facts with the most recently created ones. Top
    trust alone misses freshly inserted trust-0.5 facts, so repeated
    pre-compress events in one session would re-extract paraphrases of facts
    the store already holds. The recent slice keeps just-added facts in the
    dedup context.

    When *search_fn* is given (a callable ``search_fn(topic, limit)`` such as
    the provider's hybrid ``search()``), its top hits for *topic* are also
    included: on a large store, a fact worth deduping against may be neither
    top-trust nor recent, so the transcript's own topic must be searched for
    directly.
    """
    recent_slice = 15
    top_slice = max(limit - recent_slice, 1)
    contents: List[str] = []
    try:
        rows = store._conn.execute(
            """
            SELECT content FROM (
                SELECT fact_id, content FROM (
                    SELECT fact_id, content FROM facts
                    ORDER BY trust_score DESC, updated_at DESC LIMIT ?
                )
                UNION
                SELECT fact_id, content FROM (
                    SELECT fact_id, content FROM facts
                    ORDER BY created_at DESC, fact_id DESC LIMIT ?
                )
                ORDER BY fact_id
                LIMIT ?
            )
            """,
            (top_slice, recent_slice, limit),
        ).fetchall()
        contents.extend(row["content"] for row in rows)
    except Exception:
        return "(unavailable)"

    if search_fn is not None and topic.strip():
        try:
            hits = search_fn(topic, similar_limit)
            for hit in hits:
                content = hit.get("content", "")
                if content and content not in contents:
                    contents.append(content)
        except Exception as exc:
            logger.debug("llm_extract: topic search for dedup context failed: %s", exc)

    if not contents:
        return "(none)"
    return "\n".join(f"- {content[:120]}" for content in contents)


def _extract_json_array(text: str) -> Optional[str]:
    """Return the first balanced top-level ``[...]`` block in *text*, or None.

    Bracket-balanced and string-literal aware, so it survives ```json fences and
    any leading/trailing prose (a very common LLM output shape) without being
    confused by brackets that appear inside a fact's content.
    """
    start = text.find("[")
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
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_response(raw: str) -> List[Dict[str, str]]:
    """Parse an LLM response into a list of fact dicts.

    Tolerant of ```json fences and of prose before or after the JSON array: it
    tries a direct parse, then falls back to the first balanced top-level
    ``[...]`` block so trailing/leading text never drops a valid extraction.
    """
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        block = _extract_json_array(raw)
        if block is None:
            logger.debug("llm_extract: no JSON array in response: %r", raw[:200])
            return []
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError as exc:
            logger.debug("llm_extract: JSON parse failed (%s) on: %r", exc, block[:300])
            return []

    if not isinstance(parsed, list):
        logger.debug("llm_extract: response is not a list: %r", str(parsed)[:200])
        return []
    validated = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        category = str(item.get("category", "general")).strip()
        tags = str(item.get("tags", "")).strip()
        if not content or len(content) < 10:
            continue
        if category not in ("user_pref", "project", "tool", "general"):
            category = "general"
        validated.append({"content": content[:400], "category": category, "tags": tags})
    return validated


def extract_facts_from_transcript(
    transcript: str,
    store: "MemoryStore",
    *,
    provider: str,
    model: str,
    effort: str | None = None,
    search_fn=None,
) -> List[Dict[str, str]]:
    """Call the extraction LLM on an already formatted transcript.

    Returns the validated fact dicts ([] when the model finds nothing worth
    saving). Raises on transport or LLM failure so the persistent queue
    worker can retry with backoff instead of losing the transcript.

    *search_fn*, when given, widens the dedup context (see _existing_summary)
    with facts similar to this transcript, not just top-trust/recent ones.
    """
    if not provider or not model:
        raise RuntimeError("extraction provider/model not configured")
    if not transcript or not transcript.strip():
        return []

    from agent.auxiliary_client import call_llm  # ImportError propagates: retryable

    existing = _existing_summary(store, topic=transcript, search_fn=search_fn)
    from datetime import date
    user_msg = _USER_TEMPLATE.format(
        today=date.today().isoformat(),
        existing_summary=existing,
        conversation=transcript,
    )

    resp = call_llm(
        provider=provider,
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=1024,
        timeout=180,
        extra_body=({"reasoning": {"effort": effort}} if effort else None),
    )
    raw = resp.choices[0].message.content or ""
    return _parse_response(raw)


def insert_facts(
    store: "MemoryStore",
    facts: List[Dict[str, str]],
    embed_callback=None,  # optional: callable(fact_id, content) to trigger embedding
    dedup_check=None,  # optional: callable(content, category=...) -> existing fact dict or None
) -> int:
    """Insert extracted facts into the store; returns the number stored.

    Per-fact failures are logged and skipped so one bad fact never drops the
    rest of the batch. Duplicates resolve to their existing fact_id via the
    store's content UNIQUE constraint.

    When *dedup_check* is given (the provider's near-duplicate gate, the same
    one the interactive fact_store "add" action uses), each fact is checked
    against the store before insert so extraction never floods the store with
    duplicates the interactive path would have caught.
    """
    inserted = 0
    for fact in facts:
        try:
            if dedup_check is not None:
                dup = dedup_check(fact["content"], category=fact["category"])
                if dup is not None:
                    logger.debug(
                        "llm_extract: skipped near-duplicate of fact %s: %r",
                        dup.get("fact_id"), fact["content"][:80],
                    )
                    continue
            fact_id = store.add_fact(
                fact["content"],
                category=fact["category"],
                tags=fact["tags"],
            )
            inserted += 1
            if embed_callback and fact_id:
                try:
                    embed_callback(fact_id, fact["content"])
                except Exception as emb_exc:
                    logger.debug("llm_extract: embed callback failed for %d: %s", fact_id, emb_exc)
        except Exception as exc:
            logger.debug("llm_extract: add_fact failed: %s", exc)

    if inserted:
        logger.info("llm_extract: inserted %d/%d extracted facts", inserted, len(facts))
    return inserted
