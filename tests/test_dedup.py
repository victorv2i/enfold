"""Write-time near-duplicate guard, blend scoring, and FTS query sanitization."""

from holographic_plus import (
    _is_near_duplicate,
    _is_semantic_duplicate,
    _is_superseded,
    _value_tokens,
    _content_tokens,
    _blend_score,
)


def test_identical_is_duplicate():
    s = "CourseKit was described as stalled in the 2026-06-24 LDI discussion."
    assert _is_near_duplicate(s, s, 0.9) is True


def test_stopword_and_order_only_difference_is_duplicate():
    a = "the LDI Canvas port is 3100 on the host"
    b = "LDI Canvas port 3100 host"  # same content words + value, fewer function words
    assert _is_near_duplicate(a, b, 0.6) is True


def test_added_content_word_is_kept():
    # b adds a content word ("file") -> may carry new info -> NOT a duplicate
    a = "Canvas upstream clone is pinned at SHA abc123def in the lock"
    b = "Canvas upstream clone is pinned at SHA abc123def in the lock file"
    assert _is_near_duplicate(a, b, 0.8) is False


def test_antonym_flip_is_kept():
    # Regression: long facts differing only by an opposite content word were
    # wrongly skipped (Jaccard > 0.9, no numeric token). They are UPDATES -> keep.
    a = "The LDI Canvas sandbox service is currently active and serving traffic"
    b = "The LDI Canvas sandbox service is currently archived and serving traffic"
    assert _is_near_duplicate(a, b, 0.9) is False
    c = "The nightly recycle timer is enabled and configured for the gateway"
    d = "The nightly recycle timer is disabled and configured for the gateway"
    assert _is_near_duplicate(c, d, 0.9) is False


def test_changed_number_is_kept():
    assert _is_near_duplicate("LDI Canvas port is 3100", "LDI Canvas port is 3200", 0.5) is False


def test_distinct_facts_are_not_duplicates():
    assert _is_near_duplicate("Victor likes Python", "GameDeck launched on gamedeck.gg", 0.9) is False


def test_value_tokens_and_content_tokens():
    assert _value_tokens("port 3100 sha abc123") == {"3100", "abc123"}
    assert _value_tokens("no values here") == set()
    assert _content_tokens("the port is on a host") == {"port", "host"}


def test_blend_score_dense_term_not_trust_weighted():
    # dense term is pure cosine*ew, independent of any trust signal
    assert abs(_blend_score(0.0, 1.0, 0.45) - 0.45) < 1e-9          # cosine 1.0 -> emb_norm 1.0
    assert abs(_blend_score(0.0, 0.0, 0.45) - 0.45 * 0.5) < 1e-9    # cosine 0 -> emb_norm 0.5
    assert _blend_score(0.5, None, 0.45) == (1.0 - 0.45) * 0.5      # no embedding -> base only


def test_semantic_duplicate_antonym_flip_is_kept():
    # Regression: with no digit tokens, _value_tokens are both empty and "equal",
    # so a high cosine alone must not be enough to call a state-word change
    # (active -> archived) a duplicate. This is an UPDATE and must be kept.
    a = "The LDI Canvas sandbox service is currently active"
    b = "The LDI Canvas sandbox service is currently archived"
    assert _is_semantic_duplicate(a, b, 0.99, 0.92) is False


def test_semantic_duplicate_paraphrase_is_caught():
    a = "prefers Postgres over MySQL"
    b = "always reaches for Postgres instead of MySQL"
    assert _is_semantic_duplicate(a, b, 0.95, 0.92) is True


def test_semantic_duplicate_same_content_tokens_reordered_is_caught():
    # Same content words, just reordered with different filler/stopwords ->
    # still a duplicate under the cosine gate, same as the Jaccard path allows.
    a = "the LDI Canvas port is 3100 on the host"
    b = "LDI Canvas port 3100 host"
    assert _is_semantic_duplicate(a, b, 0.95, 0.92) is True


def test_is_superseded_detects_retired_markers():
    assert _is_superseded("SUPERSEDED 2026-06-22: old routing fact")
    assert _is_superseded("STALE/DISABLED as of 2026-05-25: Gemini routing")
    assert _is_superseded("Historical/superseded note about the backend")
    assert _is_superseded("  superseded 2026: leading whitespace tolerated")
    assert not _is_superseded("Victor prefers dark mode in the editor")
    assert not _is_superseded("The project supersedes the old plan")  # word mid-sentence, not a marker
    assert not _is_superseded("")
