from __future__ import annotations

from enfold.context import TOKEN_ESTIMATE_METHOD, estimate_tokens, pack_context


def _fact(fact_id: int, content: str, **values):
    fact = {
        "fact_id": fact_id,
        "content": content,
        "score": 1.0 - fact_id / 100,
        "memory_kind": None,
        "subject_key": None,
        "predicate_key": None,
        "scope": "private",
        "invalid_at": None,
        "superseded_by": None,
        "conflict_group": None,
        "attribution": {
            "performed_by": "wonny",
            "agent_id": "wonny",
            "evidence_count": 1,
        },
    }
    fact.update(values)
    return fact


def test_pack_context_keeps_rank_order_one_current_state_per_slot_and_receipts():
    first = _fact(
        1,
        "Current backup schedule runs Tuesday at 02:00 UTC.",
        memory_kind="state",
        subject_key="atlas",
        predicate_key="backup_schedule",
    )
    duplicate_slot = _fact(
        2,
        "Another candidate for the same current backup slot.",
        memory_kind="state",
        subject_key="atlas",
        predicate_key="backup_schedule",
    )
    stale = _fact(3, "Former backup schedule ran Monday.", invalid_at="2026-07-10")
    conflict = _fact(4, "Disputed backup schedule.", conflict_group="backup-conflict")
    reference = _fact(5, "The Atlas runbook is stored in the Cedar registry.")

    pack = pack_context(
        [first, duplicate_slot, stale, conflict, reference], token_budget=512
    )

    assert pack.abstained is False
    assert [fact["fact_id"] for fact in pack.facts] == [1, 5]
    assert pack.unsafe_fact_count == 2
    assert pack.omitted_fact_count == 1
    assert "fact:1" in pack.markdown
    assert "score:0.990" in pack.markdown
    assert "by:wonny" in pack.markdown
    assert "fact:2" not in pack.markdown
    assert "fact:3" not in pack.markdown
    assert "fact:4" not in pack.markdown


def test_pack_context_uses_conservative_budget_and_deterministic_truncation():
    fact = _fact(9, "x" * 800)

    first = pack_context([fact], token_budget=128)
    second = pack_context([fact], token_budget=128)

    assert first == second
    assert first.abstained is False
    assert first.token_estimate <= 128
    assert first.facts[0]["context_truncated"] is True
    assert first.markdown.endswith("…\n")
    assert first.as_dict()["token_estimate"]["method"] == TOKEN_ESTIMATE_METHOD
    assert estimate_tokens(first.markdown) == first.token_estimate


def test_pack_context_abstains_when_no_safe_fact_can_fit():
    pack = pack_context([_fact(1, "A durable fact")], token_budget=16)

    assert pack.abstained is True
    assert pack.markdown == ""
    assert pack.facts == ()
    assert pack.token_estimate == 0


def test_pack_context_mmr_diversifies_near_duplicates_with_jaccard_fallback():
    facts = [
        _fact(1, "Atlas backup runs Tuesday at 02:00 UTC.", score=0.99),
        _fact(2, "Atlas backup runs Tuesday at 02:00 UTC exactly.", score=0.98),
        _fact(3, "Cedar registry stores the incident runbook.", score=0.80),
    ]

    first = pack_context(facts, token_budget=512, max_facts=2)
    second = pack_context(facts, token_budget=512, max_facts=2)

    assert [fact["fact_id"] for fact in first.facts] == [1, 3]
    assert first == second
    assert first.omitted_fact_count == 1


def test_pack_context_mmr_uses_embeddings_and_preserves_budget():
    facts = [
        _fact(1, "High relevance A", score=0.99, _mmr_embedding=(1.0, 0.0)),
        _fact(2, "High relevance B", score=0.98, _mmr_embedding=(1.0, 0.0)),
        _fact(3, "Alternative fact", score=0.80, _mmr_embedding=(0.0, 1.0)),
    ]

    pack = pack_context(facts, token_budget=64, max_facts=2)

    assert [fact["fact_id"] for fact in pack.facts] == [1, 3]
    assert "_mmr_embedding" not in pack.facts[0]
    assert pack.token_estimate <= 64
