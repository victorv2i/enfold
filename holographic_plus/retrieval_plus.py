"""Faster drop-in FactRetriever for holographic_plus.

Two hot-path fixes over the parent retriever, with scoring semantics kept
identical:

1. The query HRR vector is encoded ONCE per search. The parent re-encodes
   the query inside the candidate loop (one encode_text call per candidate,
   measured at roughly 18 ms each at current dimensions), which dominated
   prefetch latency.
2. FTS candidates are fetched with explicit columns, excluding the large
   hrr_vector blob that the parent's ``SELECT f.*`` drags through SQLite
   overflow pages for every candidate row. Blobs are then loaded in a single
   targeted query, only for the candidates that get HRR-scored, and not at
   all when HRR scoring is disabled (hrr_weight == 0 or numpy missing).

Note on (2): the parent HRR-scores every FTS candidate, so all candidates
that have a stored vector still need their blob when HRR is enabled. The win
is that rows without vectors, and every search with HRR disabled, no longer
pay the blob read, and the candidate dicts never carry blobs around.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from plugins.memory.holographic import holographic as hrr
from plugins.memory.holographic.retrieval import FactRetriever

# Token boundary matching the FTS5 default (unicode61) tokenizer: it splits on
# every non-alphanumeric character (including '_', '-', ':', '.', '/'), so a
# literal like ``term_id`` is indexed as ``term`` + ``id`` and ``localhost:18791``
# as ``localhost`` + ``18791``. Extracting tokens the same way lets us build a
# MATCH string whose terms actually exist in the index.
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Natural-language filler that adds BM25 noise without identifying a fact. Kept
# small and conservative: only words that are almost never the distinguishing
# token of a stored fact. Dropped ONLY when at least one non-stopword survives.
_FTS_STOPWORDS = frozenset(
    """
    a an and are as at be but by do does for from has have how i if in is it its
    me my no not of on or our so that the their them then there these this to
    was we were what when where which who whom whose why will with you your
    about can could would should may might must shall into over under out up
    """.split()
)


class PlusFactRetriever(FactRetriever):
    """FactRetriever with a single query encode and lean candidate fetch."""

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> List[dict]:
        """Hybrid search mirroring the parent pipeline exactly.

        1. FTS5 candidates (limit * 3, lean columns)
        2. Jaccard + FTS + HRR relevance, trust weighted
        3. Optional temporal decay
        """
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)
        if not candidates:
            return []

        query_tokens = self._tokenize(query)

        # Load HRR blobs in one query, and encode the query vector at most
        # once, only when at least one candidate actually has a vector.
        hrr_vectors: Dict[int, bytes] = {}
        query_vec = None
        if self.hrr_weight > 0:
            hrr_vectors = self._load_hrr_vectors([f["fact_id"] for f in candidates])
            if hrr_vectors:
                query_vec = hrr.encode_text(query, self.hrr_dim)

        scored = []
        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity (same neutral 0.5 fallback as the parent)
            blob = hrr_vectors.get(fact["fact_id"])
            if query_vec is not None and blob:
                fact_vec = hrr.bytes_to_phases(blob)
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0
            else:
                hrr_sim = 0.5

            relevance = (self.fts_weight * fts_score
                         + self.jaccard_weight * jaccard
                         + self.hrr_weight * hrr_sim)

            score = relevance * fact["trust_score"]

            if self.half_life > 0:
                score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

            fact["score"] = score
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        # Candidates never carried hrr_vector, so no blob stripping is needed.
        return scored[:limit]

    def _load_hrr_vectors(self, fact_ids: List[int]) -> Dict[int, bytes]:
        """Fetch hrr_vector blobs for *fact_ids* in a single query."""
        if not fact_ids:
            return {}
        conn = self.store._conn
        placeholders = ",".join("?" * len(fact_ids))
        try:
            rows = conn.execute(
                f"""
                SELECT fact_id, hrr_vector FROM facts
                WHERE fact_id IN ({placeholders}) AND hrr_vector IS NOT NULL
                """,
                list(fact_ids),
            ).fetchall()
        except Exception:
            return {}
        return {int(r["fact_id"]): r["hrr_vector"] for r in rows}

    @staticmethod
    def _fts_match_query(query: str) -> str:
        """Build a safe FTS5 MATCH expression from a natural-language query.

        The parent feeds the raw query straight into ``facts_fts MATCH ?``.
        FTS5 then parses it as a query *expression*: a hyphen is the NOT
        operator (``agent-deck`` -> ``agent NOT deck``), and ``:`` ``.`` ``/``
        ``~`` ``'`` raise syntax errors, so almost every real query either
        errors out (caught -> empty) or, with default AND semantics, requires
        every token to co-occur in one short fact and matches nothing. Lexical
        recall is effectively dead.

        This extracts the index's own tokens, drops stopwords (only while a
        content token survives), wraps each survivor in double quotes so no
        character is treated as an operator, and ORs them. OR maximises recall;
        BM25 ``rank`` ordering plus the downstream Jaccard/dense/trust rerank
        restore precision. Returns ``""`` when nothing significant remains, so
        the caller can skip the MATCH entirely.
        """
        if not query:
            return ""
        tokens = _FTS_TOKEN_RE.findall(query.lower())
        if not tokens:
            return ""
        significant = [t for t in tokens if t not in _FTS_STOPWORDS]
        # If the query was all stopwords, fall back to the raw tokens rather
        # than returning nothing (better an over-broad match than no recall).
        chosen = significant or tokens
        return " OR ".join(f'"{t}"' for t in chosen)

    def _fts_candidates(
        self,
        query: str,
        category: Optional[str],
        min_trust: float,
        limit: int,
    ) -> List[dict]:
        """Parent's FTS5 candidate fetch with explicit columns.

        Same filtering, ordering, and rank normalisation as the parent, with
        two differences: hrr_vector is not selected (so candidate rows do not
        pull the blob through SQLite's overflow pages), and the raw query is
        sanitised into a safe MATCH expression via ``_fts_match_query`` so
        FTS5 operators/punctuation no longer silently kill lexical recall.
        """
        conn = self.store._conn

        match_query = self._fts_match_query(query)
        if not match_query:
            return []

        params: list = [match_query]
        where_clauses = ["facts_fts MATCH ?"]

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                   f.retrieval_count, f.helpful_count, f.created_at, f.updated_at,
                   facts_fts.rank AS fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # FTS5 MATCH can fail on malformed queries, same fallback as parent
            return []

        if not rows:
            return []

        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        max_rank = max(max_rank, 1e-6)

        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank
            results.append(fact)

        return results
