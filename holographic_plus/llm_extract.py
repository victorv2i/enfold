"""LLM-based fact extraction at session end.

Uses GPT-5.5 with xhigh reasoning to extract 3-7 atomic, durable facts
from a conversation. Called from HolographicPlusProvider.on_session_end().

Design principles:
- Only extract facts that will still be true next week
- One fact = one atomic, self-contained statement
- Deduplicate against existing facts in the store before inserting
- Fail silently so a broken extraction never crashes a session
- Run in a background thread so session teardown is non-blocking
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from plugins.memory.holographic.store import MemoryStore

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a memory extraction assistant for an AI agent called Hermes.
Your job: read a conversation and extract atomic facts worth remembering long-term.

RULES:
1. Extract 3-7 facts. Quality beats quantity — extract 0 if nothing is truly durable.
2. Each fact must be ONE atomic statement (1-3 sentences max, under 400 characters).
3. ONLY extract facts that will still be true next week — no ephemeral task details.
4. No meta-commentary ("the user asked…"). State the fact directly.
5. Assign a category from: user_pref | project | tool | general
6. Assign 2-5 comma-separated tags (lowercase, no spaces).
7. Do NOT extract facts already obvious from names/dates (e.g. "today is Monday").
8. Skip anything that's clearly a one-off instruction to the agent for this session.

GOOD facts:
- "The user prefers Postgres over MySQL for new projects."
- "The production app deploys to Vercel from the main branch."
- "The user's working hours are roughly 9am-6pm Eastern."

BAD facts (don't extract):
- "The user asked me to summarize a document." (ephemeral task)
- "The conversation was about memory systems." (meta)
- "The user said hello." (trivial)

OUTPUT: Respond with a JSON array only. No prose, no markdown fences.
Example:
[
  {"content": "The user prefers pnpm as their package manager for Node projects.", "category": "tool", "tags": "pnpm,node,package-manager"},
  {"content": "The legacy v2 service is archived; active development is on v3.", "category": "project", "tags": "v3,migration,project-status"}
]

If nothing is worth saving, output: []
"""

_USER_TEMPLATE = """\
Here is the conversation to extract facts from.
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
            # Handle list content (tool results etc.) — flatten to string
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


def _existing_summary(store: "MemoryStore", limit: int = 40) -> str:
    """Return a compact summary of existing facts for the dedup prompt."""
    try:
        rows = store._conn.execute(
            "SELECT content FROM facts ORDER BY trust_score DESC, updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            return "(none)"
        return "\n".join(f"- {row['content'][:120]}" for row in rows)
    except Exception:
        return "(unavailable)"


def _parse_response(raw: str) -> List[Dict[str, str]]:
    """Parse LLM response into a list of fact dicts. Tolerant of minor formatting issues."""
    raw = raw.strip()
    # Strip markdown fences if model added them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            logger.debug("llm_extract: response is not a list: %r", raw[:200])
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
    except json.JSONDecodeError as exc:
        logger.debug("llm_extract: JSON parse failed (%s) on: %r", exc, raw[:300])
        return []


def _run_extraction(
    messages: List[Dict[str, Any]],
    store: "MemoryStore",
    embed_callback=None,  # optional: callable(fact_id, content) to trigger embedding
) -> None:
    """Core extraction logic. Runs synchronously (called from a daemon thread)."""
    try:
        from agent.auxiliary_client import call_llm
    except ImportError:
        logger.warning("llm_extract: cannot import call_llm — skipping")
        return

    conversation = _format_conversation(messages)
    if not conversation.strip():
        logger.debug("llm_extract: empty conversation, skipping")
        return

    existing = _existing_summary(store)
    user_msg = _USER_TEMPLATE.format(
        existing_summary=existing,
        conversation=conversation,
    )

    try:
        resp = call_llm(
            provider="openai-codex",
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=1024,
            timeout=60,
            extra_body={"reasoning": {"effort": "xhigh"}},
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("llm_extract: LLM call failed: %s", exc)
        return

    facts = _parse_response(raw)
    if not facts:
        logger.debug("llm_extract: no facts extracted")
        return

    inserted = 0
    for fact in facts:
        try:
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
        logger.info("llm_extract: inserted %d/%d new facts from session", inserted, len(facts))
    else:
        logger.debug("llm_extract: all %d extracted facts were duplicates", len(facts))


def extract_facts_from_session(
    messages: List[Dict[str, Any]],
    store: "MemoryStore",
    embed_callback=None,
    blocking: bool = False,
) -> None:
    """Public entry point. Runs extraction in a daemon thread by default.

    Args:
        messages:       Full conversation history from on_session_end.
        store:          The active MemoryStore instance.
        embed_callback: Optional callable(fact_id, content) — called after each
                        successful insert to trigger embedding in holographic_plus.
        blocking:       If True, run synchronously (for testing). Default: False.
    """
    if blocking:
        _run_extraction(messages, store, embed_callback)
        return

    t = threading.Thread(
        target=_run_extraction,
        args=(messages, store, embed_callback),
        daemon=True,
        name="llm_fact_extract",
    )
    t.start()
