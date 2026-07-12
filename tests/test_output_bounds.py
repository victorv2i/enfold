from __future__ import annotations

import json
import sqlite3

from enfold.context import TOKEN_ESTIMATE_METHOD, estimate_tokens
from enfold.policy import MemoryPolicy
from enfold.protocol import ClientContext, Request
from enfold.schema import migrate
from enfold.service import EnfoldService, OutputBounds, TRUNCATION_MARKER


class RecordingRetriever:
    metadata = {"retrieval_stack": "bounds-fixture"}

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [dict(row) for row in self._rows[:kwargs["limit"]]]


def _service(tmp_path, rows, bounds):
    conn = sqlite3.connect(tmp_path / "output-bounds.db")
    migrate(conn)
    retriever = RecordingRetriever(rows)
    service = EnfoldService(
        conn,
        MemoryPolicy({"bounds-client": ("private",)}),
        retriever_factory=lambda _conn, _scopes: retriever,
        output_bounds=bounds,
    )
    context = ClientContext(
        client_id="bounds-client",
        surface="client-a",
        agent_id="worker",
        session_id="bounds-session",
        access_scopes=("private",),
    )
    return conn, service, context, retriever


def _row(fact_id, content, *, trust=0.9):
    return {
        "fact_id": fact_id,
        "content": content,
        "category": "general",
        "tags": "bounds",
        "trust_score": trust,
        "created_at": "2026-07-12 12:00:00",
        "updated_at": "2026-07-12 12:00:00",
        "memory_kind": None,
        "scope": "private",
        "invalid_at": None,
        "superseded_by": None,
        "conflict_group": None,
        "score": 0.9 - fact_id / 100,
    }


def _request(request_id, method, **params):
    return Request(request_id, method, params)


def test_search_defaults_filter_trust_but_explicit_zero_is_preserved(tmp_path):
    bounds = OutputBounds()
    conn, service, context, retriever = _service(
        tmp_path, [_row(1, "low trust", trust=0.1)], bounds
    )

    service.handle(context, _request("default", "memory.search", query="trust"))
    service.handle(
        context,
        _request("explicit", "memory.search", query="trust", min_trust=0),
    )
    service.handle(
        context,
        _request("context-default", "memory.context", query="trust", token_budget=64),
    )
    service.handle(
        context,
        _request("context", "memory.context", query="trust", token_budget=64, min_trust=0),
    )

    assert retriever.calls[0][1]["min_trust"] == bounds.default_min_trust
    assert retriever.calls[1][1]["min_trust"] == 0
    assert retriever.calls[2][1]["min_trust"] == bounds.default_min_trust
    assert retriever.calls[3][1]["min_trust"] == 0
    conn.close()


def test_search_caps_results_fact_content_and_total_serialized_chars(tmp_path):
    bounds = OutputBounds(
        search_max_results=2,
        max_fact_chars=48,
        search_max_total_chars=900,
    )
    rows = [_row(index, f"fact-{index} " + "x" * 400) for index in range(1, 5)]
    conn, service, context, retriever = _service(tmp_path, rows, bounds)

    result = service.handle(
        context,
        _request("bounded", "memory.search", query="fact", limit=200, min_trust=0),
    )

    assert retriever.calls[0][1]["limit"] == 3
    assert len(result["facts"]) <= 2
    assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) <= 900
    assert all(len(fact["content"]) <= 48 for fact in result["facts"])
    assert all(fact["content"].endswith(TRUNCATION_MARKER) for fact in result["facts"])
    assert all(fact["content_truncated"] is True for fact in result["facts"])
    assert result["output_truncated"] is True
    conn.close()


def test_context_uses_chars_per_four_estimate_and_caps_full_payload(tmp_path):
    bounds = OutputBounds(
        context_max_results=2,
        max_fact_chars=64,
        context_max_total_chars=1200,
    )
    rows = [_row(index, f"context-{index} " + "y" * 800) for index in range(1, 5)]
    conn, service, context, retriever = _service(tmp_path, rows, bounds)

    result = service.handle(
        context,
        _request(
            "bounded-context",
            "memory.context",
            query="context",
            token_budget=256,
            min_trust=0,
        ),
    )

    assert TOKEN_ESTIMATE_METHOD == "unicode_chars_divided_by_four"
    assert estimate_tokens("abcdefgh") == 2
    assert retriever.calls[0][1]["limit"] == 9
    assert len(result["facts"]) <= 2
    assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) <= 1200
    assert result["markdown"].endswith(TRUNCATION_MARKER + "\n")
    assert result["facts"][0]["content"].endswith(TRUNCATION_MARKER)
    assert result["facts"][0]["context_truncated"] is True
    assert result["output_truncated"] is True
    conn.close()
