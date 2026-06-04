"""Regression tests for the correctness fixes:

- robust JSON parsing (fences + leading/trailing prose, the load-bearing bug)
- the hybrid blend math (`_blend_score`) — previously untested
- the LLM extraction pipeline — previously untested
- blank-input batch embedding equivalence
"""
import sys
import types

from holographic_plus import _blend_score
from holographic_plus.llm_extract import (
    _extract_json_array,
    _parse_response,
    extract_facts_from_session,
)
from holographic_plus.embeddings import OllamaEmbedder


# --- robust JSON parsing ---------------------------------------------------

def test_parse_handles_trailing_prose_after_fence():
    raw = (
        '```json\n'
        '[{"content":"The user prefers dark mode in the editor.","category":"user_pref","tags":"ui"}]\n'
        '```\nHope this helps!'
    )
    out = _parse_response(raw)
    assert len(out) == 1 and out[0]["category"] == "user_pref"


def test_parse_handles_leading_prose():
    raw = 'Sure, here you go:\n[{"content":"Deploys run from the main branch.","category":"project","tags":"ci"}]'
    assert len(_parse_response(raw)) == 1


def test_parse_ignores_brackets_inside_content():
    raw = '[{"content":"Config uses a list like [1, 2, 3] for ports.","category":"tool","tags":"cfg"}] thanks'
    out = _parse_response(raw)
    assert len(out) == 1 and "[1, 2, 3]" in out[0]["content"]


def test_extract_json_array_is_balanced_and_first():
    assert _extract_json_array("noise [a] [b]") == "[a]"
    assert _extract_json_array("no array here") is None


# --- hybrid blend math -----------------------------------------------------

def test_blend_no_embedding_scales_by_one_minus_ew():
    assert abs(_blend_score(0.5, None, trust=0.8, ew=0.3) - 0.35) < 1e-9


def test_blend_perfect_match_equals_trust():
    trust = 0.9
    assert abs(_blend_score(trust, 1.0, trust, 0.3) - trust) < 1e-9


def test_blend_embedding_is_trust_weighted():
    hi = _blend_score(0.0, 1.0, trust=1.0, ew=0.3)
    lo = _blend_score(0.0, 1.0, trust=0.1, ew=0.3)
    assert abs(hi - 0.3) < 1e-9 and abs(lo - 0.03) < 1e-9 and hi > lo


def test_blend_monotonic_in_similarity():
    a = _blend_score(0.2, -1.0, 0.7, 0.3)
    b = _blend_score(0.2, 0.0, 0.7, 0.3)
    c = _blend_score(0.2, 1.0, 0.7, 0.3)
    assert a < b < c


# --- extraction pipeline ---------------------------------------------------

class _FakeStore:
    def __init__(self):
        self.added = []
        self._conn = self

    def execute(self, *a, **k):
        return self

    def fetchall(self):
        return []

    def add_fact(self, content, category="general", tags=""):
        self.added.append((content, category, tags))
        return len(self.added)


def _install_fake_call_llm(content):
    resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )
    sys.modules["agent"] = types.ModuleType("agent")
    sub = types.ModuleType("agent.auxiliary_client")
    sub.call_llm = lambda **kw: resp
    sys.modules["agent.auxiliary_client"] = sub


def test_extraction_pipeline_inserts_parsed_facts():
    _install_fake_call_llm(
        '[{"content":"The user prefers Postgres over MySQL.","category":"user_pref","tags":"db"}]'
    )
    store = _FakeStore()
    extract_facts_from_session(
        [{"role": "user", "content": "I like Postgres a lot for new projects, fyi."}],
        store,
        blocking=True,
        provider="p",
        model="m",
    )
    assert len(store.added) == 1 and store.added[0][1] == "user_pref"


def test_extraction_skips_when_no_provider():
    _install_fake_call_llm('[{"content":"this fact should never be inserted","category":"general","tags":""}]')
    store = _FakeStore()
    extract_facts_from_session(
        [{"role": "user", "content": "some content here that is long enough to format"}],
        store,
        blocking=True,
        provider=None,
        model=None,
    )
    assert store.added == []


# --- blank-input batch embedding ------------------------------------------

def test_ollama_embed_blank_returns_none():
    emb = OllamaEmbedder(base_url="http://127.0.0.1:9")  # never contacted; blanks short-circuit
    assert emb.embed("") is None and emb.embed("   ") is None


def test_ollama_embed_batch_all_blank_no_network():
    emb = OllamaEmbedder(base_url="http://127.0.0.1:9")
    assert emb.embed_batch(["", "  ", "\n"]) == [None, None, None]
