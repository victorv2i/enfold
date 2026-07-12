"""Standalone, scope-safe hybrid retrieval for Enfold v1 stores.

Candidate authorization and current-truth filtering happen before dense
embedding.  The embedder is deliberately pluggable; this module does not load
models, use the network, or depend on Hermes.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

from .core_store import active_facts, search_fts
from .embeddings import bytes_to_embedding, embedding_to_bytes
from .sqlite_vec_index import SQLiteVecIndex

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_SENTENCE_OPENERS = frozenset({
    "A", "An", "Does", "How", "Is", "The", "What", "When", "Where",
    "Which", "Who", "Why",
})
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """All score weights and confidence gates for hybrid retrieval.

    With the defaults, ``score = 0.90 * (0.35*fts + 0.25*jaccard +
    0.40*cosine) + 0.05*trust + 0.02*memory_kind + 0.03*recency``.  Trust is in
    ``[0, 1]``; state, insight, untyped, and event kind priors are respectively
    1.0, 0.75, 0.5, and 0.25.  Recency decays exponentially from 1.0 with the
    configured half-life and is capped in ``[0, 1]`` so it cannot dominate
    semantic relevance.
    """

    fts_weight: float = 0.35
    jaccard_weight: float = 0.25
    dense_weight: float = 0.40
    trust_weight: float = 0.05
    memory_kind_weight: float = 0.02
    recency_weight: float = 0.03
    recency_half_life_days: float = 365.0
    state_kind_score: float = 1.0
    insight_kind_score: float = 0.75
    untyped_kind_score: float = 0.5
    event_kind_score: float = 0.25
    score_floor: float = 0.12
    ambiguity_margin: float = 0.005

    def __post_init__(self) -> None:
        relevance = (self.fts_weight, self.jaccard_weight, self.dense_weight)
        priors = (self.trust_weight, self.memory_kind_weight, self.recency_weight)
        kind_scores = (
            self.state_kind_score,
            self.insight_kind_score,
            self.untyped_kind_score,
            self.event_kind_score,
        )
        numeric_values = (*relevance, *priors, *kind_scores)
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("ranking weights and scores must be finite")
        if any(weight < 0 for weight in (*relevance, *priors)):
            raise ValueError("ranking weights must be non-negative")
        if not math.isclose(sum(relevance), 1.0):
            raise ValueError("relevance weights must sum to 1")
        if sum(priors) >= 1.0:
            raise ValueError("ranking prior weights must sum to less than 1")
        if any(not 0.0 <= score <= 1.0 for score in kind_scores):
            raise ValueError("memory kind scores must be between 0 and 1")
        if not math.isfinite(self.recency_half_life_days) or self.recency_half_life_days <= 0:
            raise ValueError("recency half-life must be positive")
        if (
            not math.isfinite(self.score_floor)
            or not math.isfinite(self.ambiguity_margin)
            or self.score_floor < 0
            or self.score_floor > 1
            or self.ambiguity_margin < 0
            or self.ambiguity_margin > 1
        ):
            raise ValueError("ranking confidence gates must be finite and between 0 and 1")

    @property
    def relevance_weight(self) -> float:
        return 1.0 - self.trust_weight - self.memory_kind_weight - self.recency_weight


DEFAULT_RANKING_CONFIG = RankingConfig()


class DenseEmbedder(Protocol):
    """Minimal dense embedding interface used by the standalone retriever."""

    identity: str

    def embed_query(self, text: str) -> Sequence[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class VersionedEmbeddingBackend(Protocol):
    """Backend contract for production, versioned document embeddings.

    Implementations may read vectors from SQLite or another local index and
    backfill missing vectors outside the synchronous search path.  Enfold only
    passes candidates that already survived scope/current/conflict filtering.
    """

    identity: str
    embedding_version: str
    dimensions: int

    def embed_query(self, text: str) -> Sequence[float]: ...

    def load_documents(
        self, documents: Sequence[tuple[int, str]]
    ) -> Sequence[Sequence[float]]: ...


class QueryEmbedder(Protocol):
    """Small adapter boundary shared by Ollama, FastEmbed, and test doubles."""

    def embed(self, text: str) -> Sequence[float] | None: ...


class StoredEmbeddingError(RuntimeError):
    """Stored dense retrieval cannot safely serve the requested candidates."""


class SQLiteVersionedEmbeddingBackend:
    """Load exact-version vectors for authorized candidates from SQLite.

    The query/document identities are deliberately separate configuration
    values.  Their roles must map exactly, preventing a query vector from
    being compared with a different model, prefix policy, or vector version.
    Missing, duplicate, malformed, or dimension-mismatched candidate vectors
    fail the whole search; production never silently substitutes freshly
    embedded documents or a deterministic CI vector.
    """

    required_columns = frozenset({
        "fact_id", "embedding", "dim", "embedding_identity",
    })

    def __init__(
        self,
        conn: sqlite3.Connection,
        query_embedder: QueryEmbedder,
        *,
        query_identity: str,
        document_identity: str,
        embedding_version: str,
        dimensions: int,
        query_prefix: str = "",
        sql_batch_size: int = 500,
    ):
        if query_identity.count(":query:") != 1:
            raise ValueError("query_identity must contain exactly one ':query:' role")
        expected_document = query_identity.replace(":query:", ":document:")
        if document_identity != expected_document:
            raise ValueError(
                "document_identity must exactly match query_identity with its role "
                "changed from query to document"
            )
        if not embedding_version.strip():
            raise ValueError("embedding_version must be non-empty")
        if not query_identity.endswith(f":{embedding_version}"):
            raise ValueError(
                "embedding_version must be the final component of both stored "
                "embedding identities"
            )
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if sql_batch_size < 1:
            raise ValueError("sql_batch_size must be positive")
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(fact_embeddings)").fetchall()
        }
        if not self.required_columns <= columns:
            raise StoredEmbeddingError(
                "fact_embeddings is absent or incompatible; backfill the exact "
                "production identity before activation"
            )
        population = conn.execute(
            """
            SELECT COUNT(*), MIN(dim), MAX(dim)
            FROM fact_embeddings
            WHERE embedding_identity = ?
            """,
            (document_identity,),
        ).fetchone()
        if (
            population is not None
            and int(population[0]) > 0
            and (int(population[1]) != dimensions or int(population[2]) != dimensions)
        ):
            raise StoredEmbeddingError(
                "configured document identity contains vectors with an unexpected dimension"
            )
        missing_active = conn.execute(
            """
            SELECT COUNT(*)
            FROM facts AS f
            LEFT JOIN fact_embeddings AS e
              ON e.fact_id = f.fact_id
             AND e.embedding_identity = ?
             AND e.dim = ?
            LEFT JOIN embedding_jobs AS j
              ON j.fact_id = f.fact_id
             AND j.document_identity = ?
             AND j.embedding_version = ?
             AND j.dimensions = ?
             AND j.status IN ('pending', 'processing')
            WHERE f.invalid_at IS NULL
              AND f.superseded_by IS NULL
              AND f.conflict_group IS NULL
              AND e.fact_id IS NULL
              AND j.job_id IS NULL
            """,
            (
                document_identity, dimensions, document_identity,
                embedding_version, dimensions,
            ),
        ).fetchone()
        if missing_active is not None and int(missing_active[0]) > 0:
            raise StoredEmbeddingError(
                f"{int(missing_active[0])} active fact(s) lack the configured stored "
                "embedding identity and dimension"
            )
        self._conn = conn
        self._query_embedder = query_embedder
        self.query_identity = query_identity
        self.document_identity = document_identity
        self.identity = query_identity
        self.embedding_version = embedding_version
        self.dimensions = dimensions
        self.query_prefix = query_prefix
        self._sql_batch_size = sql_batch_size

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "query_embedding_identity": self.query_identity,
            "document_embedding_identity": self.document_identity,
            "embedding_version": self.embedding_version,
            "embedding_dimensions": self.dimensions,
            "stored_embedding_coverage": "strict-all-candidates",
            "missing_embedding_behavior": "fail-closed",
            "queued_embedding_behavior": "lexical-only-zero-dense-until-processed",
            "candidate_vector_source": "sqlite-fact_embeddings",
        }

    def embed_query(self, text: str) -> Sequence[float]:
        vector = self._query_embedder.embed(f"{self.query_prefix}{text}")
        if vector is None:
            raise StoredEmbeddingError("production query embedding is unavailable")
        return vector

    def load_documents(
        self, documents: Sequence[tuple[int, str]]
    ) -> Sequence[Sequence[float]]:
        if not documents:
            return ()
        fact_ids = [fact_id for fact_id, _content in documents]
        if len(set(fact_ids)) != len(fact_ids):
            raise StoredEmbeddingError("candidate fact ids must be unique")

        loaded: dict[int, Sequence[float]] = {}
        for offset in range(0, len(fact_ids), self._sql_batch_size):
            batch = fact_ids[offset:offset + self._sql_batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"""
                SELECT fact_id, embedding, dim
                FROM fact_embeddings
                WHERE embedding_identity = ?
                  AND fact_id IN ({placeholders})
                """,
                (self.document_identity, *batch),
            ).fetchall()
            for row in rows:
                fact_id = int(row[0])
                if fact_id in loaded:
                    raise StoredEmbeddingError(
                        f"duplicate stored embedding for candidate fact {fact_id}"
                    )
                if int(row[2]) != self.dimensions:
                    raise StoredEmbeddingError(
                        f"stored embedding dimension mismatch for candidate fact {fact_id}"
                    )
                blob = row[1]
                if not isinstance(blob, bytes | bytearray | memoryview):
                    raise StoredEmbeddingError(
                        f"malformed stored embedding for candidate fact {fact_id}"
                    )
                vector = bytes_to_embedding(bytes(blob))
                if len(vector) != self.dimensions:
                    raise StoredEmbeddingError(
                        f"stored embedding byte length mismatch for candidate fact {fact_id}"
                    )
                loaded[fact_id] = vector

        missing = [fact_id for fact_id in fact_ids if fact_id not in loaded]
        if missing:
            placeholders = ",".join("?" for _ in missing)
            queued = {
                int(row[0])
                for row in self._conn.execute(
                    f"""
                    SELECT fact_id FROM embedding_jobs
                    WHERE document_identity = ? AND embedding_version = ?
                      AND dimensions = ? AND status IN ('pending', 'processing')
                      AND fact_id IN ({placeholders})
                    """,
                    (
                        self.document_identity, self.embedding_version,
                        self.dimensions, *missing,
                    ),
                ).fetchall()
            }
            uncovered = [fact_id for fact_id in missing if fact_id not in queued]
            if uncovered:
                preview = ", ".join(str(fact_id) for fact_id in uncovered[:10])
                suffix = "..." if len(uncovered) > 10 else ""
                raise StoredEmbeddingError(
                    f"missing {len(uncovered)} required stored candidate embedding(s) "
                    f"without a viable exact-identity job: {preview}{suffix}"
                )
            zero = tuple(0.0 for _ in range(self.dimensions))
            loaded.update((fact_id, zero) for fact_id in missing)
        return tuple(loaded[fact_id] for fact_id in fact_ids)


class SQLiteStoredEmbeddingWriter:
    """Explicit idempotent maintenance/backfill API for document vectors.

    The packaged request service does not call this API: model work is
    forbidden on the synchronous memory-write path.  A future durable outbox
    processor may use it after claiming a job outside the write transaction.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        document_embedder: QueryEmbedder,
        *,
        document_identity: str,
        embedding_version: str,
        model_fingerprint: str,
        prefix_policy: str,
        dimensions: int,
        document_prefix: str = "",
        query_prefix: str = "",
    ):
        if document_identity.count(":document:") != 1:
            raise ValueError(
                "document_identity must contain exactly one ':document:' role"
            )
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if embedding_version != model_fingerprint:
            raise ValueError("model fingerprint must equal embedding version")
        if prefix_policy == "none":
            if query_prefix or document_prefix:
                raise ValueError("none prefix policy requires empty query/document prefixes")
        elif prefix_policy.startswith("sha256-"):
            digest = hashlib.sha256(
                f"{query_prefix}\0{document_prefix}".encode("utf-8")
            ).hexdigest()
            if prefix_policy != f"sha256-{digest}":
                raise ValueError("document prefix does not match prefix policy")
        else:
            raise ValueError("prefix policy must be none or sha256-<full digest>")
        if not document_identity.endswith(
            f":document:{prefix_policy}:{embedding_version}"
        ):
            raise ValueError("document identity is not bound to writer configuration")
        self._conn = conn
        self._document_embedder = document_embedder
        self.document_identity = document_identity
        self.embedding_version = embedding_version
        self.model_fingerprint = model_fingerprint
        self.prefix_policy = prefix_policy
        self.dimensions = dimensions
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self._vector_index = SQLiteVecIndex.open(
            conn, document_identity, dimensions, warn=False
        )

    def embed_document(self, content: str) -> bytes:
        """Run the configured model and return a validated portable vector blob."""

        vector = self._document_embedder.embed(f"{self.document_prefix}{content}")
        if vector is None:
            raise StoredEmbeddingError("production document embedding is unavailable")
        if len(vector) != self.dimensions or any(
            not math.isfinite(float(value)) for value in vector
        ):
            raise StoredEmbeddingError("production document embedding is invalid")
        return embedding_to_bytes(np.asarray(vector, dtype=np.float32))

    def upsert_in_transaction(self, fact_id: int, embedding: bytes) -> None:
        """Store a prepared vector inside a caller-owned validation transaction."""

        if not self._conn.in_transaction:
            raise RuntimeError("prepared embedding upsert requires a transaction")
        if len(bytes_to_embedding(embedding)) != self.dimensions:
            raise StoredEmbeddingError("prepared document embedding is invalid")
        self._conn.execute(
            """
            INSERT INTO fact_embeddings(
                fact_id, embedding, dim, embedding_identity
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(fact_id, embedding_identity) DO UPDATE SET
                embedding = excluded.embedding, dim = excluded.dim
            """,
            (fact_id, embedding, self.dimensions, self.document_identity),
        )
        if self._vector_index is not None:
            self._vector_index.upsert_in_transaction(fact_id, embedding)

    def ensure_fact(self, fact_id: int, *, force: bool = False) -> bool:
        """Ensure one committed fact has its exact production vector.

        Returns ``False`` when a valid vector already exists and ``True`` after
        writing one. Callers must invoke this only from an explicit maintenance
        flow or an asynchronous durable-job processor.
        """

        if self._conn.in_transaction:
            raise RuntimeError("stored embedding write-through must run after commit")
        row = self._conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            raise StoredEmbeddingError("committed fact disappeared before embedding")
        existing = self._conn.execute(
            """
            SELECT embedding, dim FROM fact_embeddings
            WHERE fact_id = ? AND embedding_identity = ?
            """,
            (fact_id, self.document_identity),
        ).fetchone()
        if existing is not None and not force:
            vector = bytes_to_embedding(bytes(existing[0]))
            if int(existing[1]) != self.dimensions or len(vector) != self.dimensions:
                raise StoredEmbeddingError(
                    "existing stored embedding has the wrong dimension"
                )
            return False
        content = str(row[0])
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        embedding = self.embed_document(content)
        # Serialize explicitly as portable little-endian float32 bytes without
        # requiring callers to know the storage format.
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            current = self._conn.execute(
                "SELECT content, invalid_at, superseded_by, conflict_group FROM facts "
                "WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if current is None or any(value is not None for value in current[1:]):
                raise StoredEmbeddingError("fact became ineligible during embedding")
            current_hash = hashlib.sha256(str(current[0]).encode("utf-8")).hexdigest()
            if current_hash != content_hash:
                raise StoredEmbeddingError("fact content changed during embedding")
            self.upsert_in_transaction(fact_id, embedding)
            self._conn.commit()
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
        return True


@dataclass(frozen=True, slots=True)
class VersionedStoredEmbeddingAdapter:
    """Adapt a versioned production backend to Enfold's retriever contract."""

    backend: VersionedEmbeddingBackend
    production_ready: bool = True

    @property
    def identity(self) -> str:
        return (
            f"{self.backend.identity}@{self.backend.embedding_version}"
            f":{self.backend.dimensions}"
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(getattr(self.backend, "metadata", {}))

    def embed_query(self, text: str) -> Sequence[float]:
        vector = self.backend.embed_query(text)
        self._validate_vector(vector, "query")
        return vector

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise RuntimeError(
            "versioned stored embeddings require candidate ids; use "
            "embed_candidates through HybridRetriever"
        )

    def embed_candidates(
        self, candidates: Sequence[Mapping[str, Any]]
    ) -> Sequence[Sequence[float]]:
        documents = tuple(
            (int(row["fact_id"]), str(row["content"])) for row in candidates
        )
        vectors = self.backend.load_documents(documents)
        if len(vectors) != len(documents):
            raise ValueError("embedding backend returned the wrong number of vectors")
        for vector in vectors:
            self._validate_vector(vector, "document")
        return vectors

    def _validate_vector(self, vector: Sequence[float], kind: str) -> None:
        if len(vector) != self.backend.dimensions:
            raise ValueError(f"{kind} embedding dimensions do not match backend metadata")
        if any(not math.isfinite(float(value)) for value in vector):
            raise ValueError(f"{kind} embedding contains a non-finite value")


RetrieverFactory = Callable[[sqlite3.Connection, Sequence[str]], "HybridRetriever"]


def deterministic_retriever_factory(
    *,
    dimensions: int = 256,
    fts_weight: float = DEFAULT_RANKING_CONFIG.fts_weight,
    jaccard_weight: float = DEFAULT_RANKING_CONFIG.jaccard_weight,
    dense_weight: float = DEFAULT_RANKING_CONFIG.dense_weight,
    min_score: float = DEFAULT_RANKING_CONFIG.score_floor,
    vector_backend: str = "auto",
) -> RetrieverFactory:
    """Return an offline factory suitable for tests and explicit CI config."""

    def build(conn: sqlite3.Connection, scopes: Sequence[str]) -> HybridRetriever:
        return HybridRetriever(
            conn,
            DeterministicFeatureHashEmbedder(dimensions),
            allowed_scopes=scopes,
            fts_weight=fts_weight,
            jaccard_weight=jaccard_weight,
            dense_weight=dense_weight,
            min_score=min_score,
            vector_backend=vector_backend,
        )

    return build


class DeterministicFeatureHashEmbedder:
    """Offline deterministic CI embedder, explicitly not a semantic model.

    It hashes word tokens and character trigrams into a fixed vector.  This is
    useful for exercising dense plumbing reproducibly, but production should
    inject a real versioned local embedding model.
    """

    production_ready = False

    def __init__(self, dimensions: int = 256):
        if dimensions < 16:
            raise ValueError("dimensions must be at least 16")
        self.dimensions = dimensions
        self.identity = f"ci-feature-hash-v1:{dimensions}"

    def _embed(self, text: str) -> tuple[float, ...]:
        normalized = " ".join(_TOKEN_RE.findall(text.lower()))
        features = [f"w:{token}" for token in normalized.split()]
        compact = normalized.replace(" ", "_")
        features.extend(f"c3:{compact[i:i + 3]}" for i in range(max(0, len(compact) - 2)))
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return tuple(vector)

    def embed_query(self, text: str) -> Sequence[float]:
        return self._embed(text)

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._embed(text) for text in texts)


def named_anchor_tokens(text: str) -> frozenset[str]:
    """Return explicit capitalized names that must occur in a result."""

    words = _WORD_RE.findall(text)
    return frozenset(
        word.lower()
        for index, word in enumerate(words)
        if word[0].isupper()
        and not (index == 0 and word in _SENTENCE_OPENERS)
    )


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("query and document embedding dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    # NumPy-backed stored vectors yield NumPy scalar products.  Normalize the
    # public score to a built-in float so protocol JSON serialization cannot
    # fail only after dense coverage becomes complete.
    return float(
        sum(a * b for a, b in zip(left, right, strict=True))
        / (left_norm * right_norm)
    )


def _fts_query(text: str) -> str:
    return " OR ".join(f'"{token}"' for token in sorted(_tokens(text)))


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, (str, datetime)):
        return None
    try:
        parsed = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        )
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recency_score(row: Mapping[str, Any], now: datetime, half_life_days: float) -> float:
    timestamp = _as_utc(row.get("updated_at")) or _as_utc(row.get("created_at"))
    if timestamp is None:
        return 0.0
    age_days = max(0.0, (now - timestamp).total_seconds() / 86_400.0)
    return math.exp(-math.log(2.0) * age_days / half_life_days)


def _memory_kind_score(kind: Any, config: RankingConfig) -> float:
    return {
        "state": config.state_kind_score,
        "insight": config.insight_kind_score,
        "event": config.event_kind_score,
    }.get(kind, config.untyped_kind_score)


class HybridRetriever:
    """Blend FTS, Jaccard, and pluggable dense scores over eligible facts."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: DenseEmbedder,
        *,
        allowed_scopes: Sequence[str] = ("private",),
        fts_weight: float | None = None,
        jaccard_weight: float | None = None,
        dense_weight: float | None = None,
        min_score: float | None = None,
        candidate_limit: int = 10_000,
        vector_backend: str = "auto",
        ranking_config: RankingConfig = DEFAULT_RANKING_CONFIG,
        now: datetime | None = None,
    ):
        relevance_weights = (
            ranking_config.fts_weight if fts_weight is None else fts_weight,
            ranking_config.jaccard_weight if jaccard_weight is None else jaccard_weight,
            ranking_config.dense_weight if dense_weight is None else dense_weight,
        )
        min_score = ranking_config.score_floor if min_score is None else min_score
        if any(weight < 0 for weight in relevance_weights) or not math.isclose(
            sum(relevance_weights), 1.0
        ):
            raise ValueError("retrieval weights must be non-negative and sum to 1")
        if not math.isfinite(min_score) or not 0 <= min_score <= 1 or candidate_limit < 1:
            raise ValueError("min_score must be between 0 and 1 and candidate_limit positive")
        if vector_backend not in {"auto", "sqlite-vec", "brute"}:
            raise ValueError("vector_backend must be auto, sqlite-vec, or brute")
        self._conn = conn
        self._embedder = embedder
        self._allowed_scopes = tuple(allowed_scopes)
        self._weights = relevance_weights
        self._ranking = ranking_config
        self._min_score = min_score
        self._candidate_limit = candidate_limit
        self._vector_index: SQLiteVecIndex | None = None
        if vector_backend != "brute" and isinstance(embedder, VersionedStoredEmbeddingAdapter):
            backend = embedder.backend
            document_identity = getattr(backend, "document_identity", None)
            if isinstance(document_identity, str):
                self._vector_index = SQLiteVecIndex.open(
                    conn,
                    document_identity,
                    backend.dimensions,
                    warn=True,
                )
        clock = now or datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if not isinstance(clock, datetime):
            raise ValueError("now must be a datetime")
        self._now = _as_utc(clock)
        assert self._now is not None
        self.metadata = {
            "retrieval_stack": "standalone_core_fts+jaccard+pluggable_dense",
            "embedder_identity": str(embedder.identity),
            "embedder_production_ready": bool(getattr(embedder, "production_ready", True)),
            "filter_before_dense_ranking": True,
            "explicit_named_anchor_abstention": True,
            "natural_language_query_parser": "quoted_token_or_v1",
            "stored_embedding_contract": "versioned-candidate-id-v1",
            "score_formula": (
                "relevance_weight*(fts_weight*fts+jaccard_weight*jaccard+"
                "dense_weight*cosine)+trust_weight*trust+"
                "memory_kind_weight*memory_kind+recency_weight*recency"
            ),
            "weights": {
                "relevance": ranking_config.relevance_weight,
                "fts": relevance_weights[0],
                "jaccard": relevance_weights[1],
                "dense": relevance_weights[2],
                "trust": ranking_config.trust_weight,
                "memory_kind": ranking_config.memory_kind_weight,
                "recency": ranking_config.recency_weight,
            },
            "score_floor": min_score,
            "ambiguity_margin": ranking_config.ambiguity_margin,
            "recency_half_life_days": ranking_config.recency_half_life_days,
            "vector_backend_config": vector_backend,
            "vector_backend": "sqlite-vec" if self._vector_index else "brute",
        }
        self.metadata.update(dict(getattr(embedder, "metadata", {})))

    def _stored_mmr_embeddings(
        self, fact_ids: Sequence[int]
    ) -> dict[int, tuple[float, ...]]:
        """Decode canonical vectors for the bounded vec0 result set only."""

        if not fact_ids or not isinstance(
            self._embedder, VersionedStoredEmbeddingAdapter
        ):
            return {}
        backend = self._embedder.backend
        identity = getattr(backend, "document_identity", None)
        dimensions = getattr(backend, "dimensions", None)
        if not isinstance(identity, str) or not isinstance(dimensions, int):
            return {}
        loaded: dict[int, tuple[float, ...]] = {}
        for offset in range(0, len(fact_ids), 500):
            batch = fact_ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"""
                SELECT fact_id, embedding, dim
                FROM fact_embeddings
                WHERE embedding_identity = ?
                  AND fact_id IN ({placeholders})
                """,
                (identity, *batch),
            ).fetchall()
            for row in rows:
                blob = row[1]
                if (
                    int(row[2]) != dimensions
                    or not isinstance(blob, bytes | bytearray | memoryview)
                ):
                    continue
                try:
                    vector = bytes_to_embedding(bytes(blob))
                except ValueError:
                    continue
                if len(vector) == dimensions:
                    loaded[int(row[0])] = tuple(float(value) for value in vector)
        return loaded

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        bump: bool = False,
    ) -> list[dict[str, Any]]:
        del bump
        query = query.strip()
        if not query or limit <= 0:
            return []

        # This SQL boundary excludes invalid, superseded, conflicted, and
        # out-of-scope facts.  Trust filtering and named-anchor abstention also
        # happen before document text reaches the dense embedder.
        candidates = active_facts(
            self._conn,
            allowed_scopes=self._allowed_scopes,
            category=category,
            limit=self._candidate_limit,
        )
        candidates = [row for row in candidates if float(row.get("trust_score") or 0.0) >= min_trust]
        anchors = named_anchor_tokens(query)
        if anchors:
            candidates = [
                row for row in candidates
                if anchors <= _tokens(f"{row.get('content', '')} {row.get('tags', '')}")
            ]
        if not candidates:
            return []

        fts_scores: dict[int, float] = {}
        fts_query = _fts_query(query)
        if fts_query:
            for rank, row in enumerate(search_fts(
                self._conn,
                fts_query,
                allowed_scopes=self._allowed_scopes,
                category=category,
                min_trust=min_trust,
                limit=self._candidate_limit,
            )):
                fts_scores[int(row["fact_id"])] = 1.0 / (rank + 1.0)

        query_vector = self._embedder.embed_query(query)
        dense_scores: dict[int, float] | None = None
        if self._vector_index is not None:
            try:
                dense_scores = self._vector_index.scores(
                    query_vector, tuple(int(row["fact_id"]) for row in candidates)
                )
            # The index is optional derived state.  Treat every failure inside
            # its scoring boundary (including unexpected extension result
            # types such as NULL cosine distances for zero vectors) as an
            # index health problem, then use the canonical embedding blobs.
            except Exception as exc:
                LOGGER.warning(
                    "sqlite-vec health warning: %s; falling back to brute", exc
                )
                self._vector_index = None
                self.metadata["vector_backend"] = "brute"
        document_vectors: Sequence[Sequence[float]] = ()
        if dense_scores is None:
            candidate_embedder = getattr(self._embedder, "embed_candidates", None)
            if callable(candidate_embedder):
                document_vectors = candidate_embedder(candidates)
            else:
                document_vectors = self._embedder.embed_documents(
                    tuple(str(row["content"]) for row in candidates)
                )
            if len(document_vectors) != len(candidates):
                raise ValueError("embedder returned the wrong number of document vectors")

        query_tokens = _tokens(query)
        fts_weight, jaccard_weight, dense_weight = self._weights
        ranking = self._ranking
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for position, row in enumerate(candidates):
            fact_id = int(row["fact_id"])
            fts_score = fts_scores.get(fact_id, 0.0)
            jaccard_score = _jaccard(query_tokens, _tokens(str(row["content"])))
            dense_score = (
                dense_scores[fact_id]
                if dense_scores is not None
                else max(0.0, _cosine(query_vector, document_vectors[position]))
            )
            relevance_score = (
                fts_weight * fts_score
                + jaccard_weight * jaccard_score
                + dense_weight * dense_score
            )
            trust_score = min(1.0, max(0.0, float(row.get("trust_score") or 0.0)))
            memory_kind_score = _memory_kind_score(row.get("memory_kind"), ranking)
            recency_score = _recency_score(
                row, self._now, ranking.recency_half_life_days
            )
            score = (
                ranking.relevance_weight * relevance_score
                + ranking.trust_weight * trust_score
                + ranking.memory_kind_weight * memory_kind_score
                + ranking.recency_weight * recency_score
            )
            if score < self._min_score:
                continue
            vector = (
                tuple(float(value) for value in document_vectors[position])
                if document_vectors
                else None
            )
            result = dict(row)
            result.update({
                "score": score,
                "_mmr_embedding": vector,
                "fts_score": fts_score,
                "jaccard_score": jaccard_score,
                "dense_score": dense_score,
                "trust_score_component": trust_score,
                "memory_kind_score": memory_kind_score,
                "recency_score": recency_score,
            })
            ranked.append((score, fact_id, result))
        ranked.sort(key=lambda item: (-item[0], -float(item[2].get("trust_score") or 0.0), item[1]))
        # A near-tie is ambiguous: returning the deterministic fact-id winner
        # would manufacture confidence from a tie-breaker.  Fail closed instead.
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < ranking.ambiguity_margin:
            return []
        survivors = ranked[:limit]
        if dense_scores is not None:
            mmr_vectors = self._stored_mmr_embeddings(
                [fact_id for _, fact_id, _ in survivors]
            )
            for _, fact_id, row in survivors:
                row["_mmr_embedding"] = mmr_vectors.get(fact_id)
        return [row for _, _, row in survivors]
