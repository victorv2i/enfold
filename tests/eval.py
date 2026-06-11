#!/usr/bin/env python3
"""Recall benchmark for holographic_plus.

Question it answers: does the dense-embedding layer recover facts that a
keyword baseline misses? It seeds a synthetic corpus into the *real* EmbedStore,
runs paraphrased queries (deliberately low lexical overlap with their target),
and reports recall@k for three retrievers:

  - keyword   : Jaccard token overlap (a fair stand-in for FTS/keyword search)
  - embedding : dense cosine via the plugin's EmbedStore.score_all (the layer this plugin adds)
  - hybrid    : a normalized blend of the two

This isolates the embedding layer's contribution. In the full plugin these
scores also combine with HRR and FTS5 + trust. It is a synthetic, illustrative
benchmark, not a production guarantee. Self-contained; needs only fastembed.

    python tests/eval.py
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys

import numpy as np

# Make the package importable without a full Hermes install: reuse the test
# suite's hermes stubs (same mechanism as tests/conftest.py).
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS_DIR))
sys.path.insert(0, _TESTS_DIR)
import fake_hermes  # noqa: E402

fake_hermes.install_stubs()

from holographic_plus.embed_store import EmbedStore  # noqa: E402
from holographic_plus.embeddings import FastEmbedder  # noqa: E402

CORPUS = [
    "The user prefers dark mode in their code editor.",
    "Production deploys happen automatically from the main branch.",
    "The team uses pnpm as the package manager for Node projects.",
    "Database backups run nightly and are kept for thirty days.",
    "The staging environment mirrors production on a smaller instance.",
    "API rate limits are one hundred requests per minute per key.",
    "The user's working hours are roughly nine to six Eastern time.",
    "Unit tests must pass before any pull request can be merged.",
    "The frontend is built with React and TypeScript.",
    "Support tickets are triaged within one business day.",
    "The mobile app supports offline mode with local caching.",
    "Secrets are stored in a vault and never committed to the repo.",
    "The CI pipeline runs lint, typecheck, and tests on every push.",
    "Invoices are sent on the first of each month.",
    "Search uses fuzzy matching to tolerate typos.",
    "User passwords are hashed with bcrypt, never stored in plaintext.",
    "The analytics dashboard refreshes every five minutes.",
    "Feature flags let the team roll out changes gradually.",
    "The onboarding flow has three steps and takes under two minutes.",
    "Email notifications can be disabled in account settings.",
    "The recommendation engine retrains weekly on fresh data.",
    "Large file uploads are chunked and resumable.",
    "The admin panel requires two-factor authentication.",
    "Logs are retained for ninety days, then archived to cold storage.",
]

# (query, target_index): paraphrases, most with deliberately low keyword overlap.
QUERIES = [
    ("Is there a way to switch the IDE to a low-light theme?", 0),
    ("When does new code reach end users?", 1),
    ("Which tool installs Node packages in this project?", 2),
    ("How long are database snapshots retained?", 3),
    ("What's the ceiling on API calls per key?", 5),
    ("Can I turn off email alerts?", 19),
    ("How are account credentials protected at rest?", 15),
    ("Does the application function without an internet connection?", 10),
    ("What keeps the suggestion model up to date?", 20),
    ("How is administrator access secured?", 22),
    ("Will lookup still work if I mistype a word?", 14),
    ("How frequently does the metrics view update?", 16),
    ("What must pass before a change can be merged?", 7),
    ("Where are API keys and tokens kept?", 11),
    ("How long are server logs stored before archiving?", 23),
    ("What hours does the user typically work?", 6),
]

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOK.findall(text.lower()))


def _keyword_scores(query: str) -> dict:
    qt = _tokens(query)
    scores = {}
    for i, fact in enumerate(CORPUS):
        ft = _tokens(fact)
        union = len(qt | ft) or 1
        scores[i] = len(qt & ft) / union  # Jaccard
    return scores


def _minmax(d: dict) -> dict:
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {k: (v - lo) / rng for k, v in d.items()}


def _ranked(scores: dict) -> list:
    return [i for i, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def _recall_at_k(ranked_per_query: list, targets: list, k: int) -> float:
    hits = sum(1 for ranked, t in zip(ranked_per_query, targets) if t in ranked[:k])
    return hits / len(targets)


def _mrr(ranked_per_query: list, targets: list) -> float:
    total = 0.0
    for ranked, t in zip(ranked_per_query, targets):
        if t in ranked:
            total += 1.0 / (ranked.index(t) + 1)
    return total / len(targets)


def main() -> int:
    embedder = FastEmbedder()
    if not embedder.is_available():
        print("fastembed not installed, run `pip install fastembed` to evaluate.")
        return 1

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    doc_id = "eval:bge:document:none:v1"
    qry_id = "eval:bge:query:none:v1"
    store = EmbedStore(conn, embedding_identity=doc_id)

    for i, vec in enumerate(embedder.embed_batch(CORPUS)):
        store.upsert(i, vec, embedding_identity=doc_id)

    targets = [t for _, t in QUERIES]
    kw_ranked, emb_ranked, hyb_ranked = [], [], []

    for query, _ in QUERIES:
        kw = _keyword_scores(query)
        kw_ranked.append(_ranked(kw))

        qv = embedder.embed(query)
        emb = {fid: sim for fid, sim in store.score_all(qv, embedding_identity=qry_id)}
        emb_ranked.append(_ranked(emb))

        kw_n, emb_n = _minmax(kw), _minmax(emb)
        hybrid = {i: 0.4 * kw_n[i] + 0.6 * emb_n.get(i, 0.0) for i in range(len(CORPUS))}
        hyb_ranked.append(_ranked(hybrid))

    print(f"holographic_plus recall benchmark: {len(CORPUS)} facts, {len(QUERIES)} paraphrased queries\n")
    header = f"{'retriever':<12} {'recall@1':>9} {'recall@3':>9} {'recall@5':>9} {'MRR':>7}"
    print(header)
    print("-" * len(header))
    for label, ranked in (("keyword", kw_ranked), ("embedding", emb_ranked), ("blend*", hyb_ranked)):
        print(
            f"{label:<12} "
            f"{_recall_at_k(ranked, targets, 1):>9.2f} "
            f"{_recall_at_k(ranked, targets, 3):>9.2f} "
            f"{_recall_at_k(ranked, targets, 5):>9.2f} "
            f"{_mrr(ranked, targets):>7.2f}"
        )
    print(
        "\n* 'blend' is an illustrative keyword+embedding mix (min-max normalised, 0.4/0.6) "
        "for this isolated comparison. It is NOT the shipped scorer: the plugin adds the "
        "trust-weighted embedding onto the trust-weighted holographic score (FTS+Jaccard+HRR)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
