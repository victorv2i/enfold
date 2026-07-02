"""Stand-ins for the hermes internals that enfold imports.

Installs sys.modules stubs for:

  - plugins.memory.holographic (parent provider, store, retriever, hrr math)
  - agent.auxiliary_client (call_llm, controllable per test)
  - hermes_constants (get_hermes_home pointed at a temp dir)

The parent fakes replicate the upstream holographic plugin's behavior closely
enough that the subclass under test exercises real SQLite, real FTS5, and the
same scoring pipeline, with no hermes install, no network, and no real
embedding backend. The FactRetriever fake intentionally keeps the parent's
per-candidate encode_text call so it can serve as the semantics baseline for
the optimized PlusFactRetriever.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import sys
import tempfile
import threading
import types
from pathlib import Path

import numpy as np

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# hrr math (mirrors plugins/memory/holographic/holographic.py)
# ---------------------------------------------------------------------------

def _build_hrr_module() -> types.ModuleType:
    mod = types.ModuleType("plugins.memory.holographic.holographic")

    def encode_atom(word: str, dim: int = 1024):
        values_per_block = 16
        blocks_needed = math.ceil(dim / values_per_block)
        uint16_values = []
        for i in range(blocks_needed):
            digest = hashlib.sha256(f"{word}:{i}".encode()).digest()
            uint16_values.extend(struct.unpack("<16H", digest))
        return np.array(uint16_values[:dim], dtype=np.float64) * (_TWO_PI / 65536.0)

    def bind(a, b):
        return (a + b) % _TWO_PI

    def unbind(memory, key):
        return (memory - key) % _TWO_PI

    def bundle(*vectors):
        complex_sum = np.sum([np.exp(1j * v) for v in vectors], axis=0)
        return np.angle(complex_sum) % _TWO_PI

    def similarity(a, b):
        return float(np.mean(np.cos(a - b)))

    def encode_text(text: str, dim: int = 1024):
        tokens = [t.strip(".,!?;:\"'()[]{}") for t in text.lower().split()]
        tokens = [t for t in tokens if t]
        if not tokens:
            return encode_atom("__hrr_empty__", dim)
        return bundle(*[encode_atom(t, dim) for t in tokens])

    def encode_fact(content: str, entities, dim: int = 1024):
        role_content = encode_atom("__hrr_role_content__", dim)
        role_entity = encode_atom("__hrr_role_entity__", dim)
        components = [bind(encode_text(content, dim), role_content)]
        for entity in entities:
            components.append(bind(encode_atom(entity.lower(), dim), role_entity))
        return bundle(*components)

    def phases_to_bytes(phases):
        return phases.tobytes()

    def bytes_to_phases(data):
        return np.frombuffer(data, dtype=np.float64).copy()

    def snr_estimate(dim: int, n_items: int) -> float:
        return float("inf") if n_items <= 0 else math.sqrt(dim / n_items)

    mod._HAS_NUMPY = True
    mod.encode_atom = encode_atom
    mod.bind = bind
    mod.unbind = unbind
    mod.bundle = bundle
    mod.similarity = similarity
    mod.encode_text = encode_text
    mod.encode_fact = encode_fact
    mod.phases_to_bytes = phases_to_bytes
    mod.bytes_to_phases = bytes_to_phases
    mod.snr_estimate = snr_estimate
    return mod


hrr = _build_hrr_module()

# Mirrors store.py's _RE_CAPITALIZED entity-extraction rule: multi-word
# capitalized phrases only (single capitalized words are too noisy to
# reliably identify an entity).
_RE_CAPITALIZED = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')


# ---------------------------------------------------------------------------
# MemoryStore (mirrors plugins/memory/holographic/store.py essentials)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class MemoryStore:
    def __init__(self, db_path=None, default_trust: float = 0.5, hrr_dim: int = 64):
        self.db_path = Path(db_path)
        self.default_trust = default_trust
        self.hrr_dim = hrr_dim
        self._hrr_available = True
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add_fact(self, content: str, category: str = "general", tags: str = "") -> int:
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")
            try:
                cur = self._conn.execute(
                    "INSERT INTO facts (content, category, tags, trust_score) VALUES (?, ?, ?, ?)",
                    (content, category, tags, self.default_trust),
                )
                self._conn.commit()
                fact_id = int(cur.lastrowid)
            except sqlite3.IntegrityError:
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                return int(row["fact_id"])
            for name in self._extract_entities(content):
                entity_id = self._resolve_entity(name)
                self._link_fact_entity(fact_id, entity_id)
            self._compute_hrr_vector(fact_id, content)
            self._rebuild_bank(category)
            return fact_id

    def _extract_entities(self, text: str) -> list:
        """Multi-word capitalized phrases only (mirrors store.py)."""
        seen: set = set()
        candidates: list = []
        for m in _RE_CAPITALIZED.finditer(text):
            name = m.group(1).strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                candidates.append(name)
        return candidates

    def _resolve_entity(self, name: str) -> int:
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name LIKE ?", (name,)
        ).fetchone()
        if row is not None:
            return int(row["entity_id"])
        cur = self._conn.execute("INSERT INTO entities (name) VALUES (?)", (name,))
        self._conn.commit()
        return int(cur.lastrowid)

    def _link_fact_entity(self, fact_id: int, entity_id: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
            (fact_id, entity_id),
        )
        self._conn.commit()

    def update_fact(self, fact_id, content=None, trust_delta=None, tags=None, category=None) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, category FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                return False
            assignments = ["updated_at = CURRENT_TIMESTAMP"]
            params = []
            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if trust_delta is not None:
                assignments.append("trust_score = ?")
                params.append(max(0.0, min(1.0, row["trust_score"] + trust_delta)))
            params.append(fact_id)
            self._conn.execute(
                f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?", params
            )
            self._conn.commit()
            if content is not None:
                self._compute_hrr_vector(fact_id, content)
            self._rebuild_bank(category or row["category"])
            return True

    def remove_fact(self, fact_id) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False
            self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._conn.commit()
            self._rebuild_bank(row["category"])
            return True

    def list_facts(self, category=None, min_trust: float = 0.0, limit: int = 50):
        with self._lock:
            params = [min_trust]
            clause = ""
            if category is not None:
                clause = "AND category = ?"
                params.append(category)
            params.append(limit)
            rows = self._conn.execute(
                f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts WHERE trust_score >= ? {clause}
                ORDER BY trust_score DESC LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        with self._lock:
            vector = hrr.encode_fact(content, [], self.hrr_dim)
            self._conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), fact_id),
            )
            self._conn.commit()

    def _rebuild_bank(self, category: str) -> None:
        with self._lock:
            bank_name = f"cat:{category}"
            rows = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL",
                (category,),
            ).fetchall()
            if not rows:
                self._conn.execute(
                    "DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,)
                )
                self._conn.commit()
                return
            vectors = [hrr.bytes_to_phases(r["hrr_vector"]) for r in rows]
            bank_vector = hrr.bundle(*vectors)
            self._conn.execute(
                """
                INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bank_name) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    fact_count = excluded.fact_count,
                    updated_at = excluded.updated_at
                """,
                (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, len(vectors)),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# FactRetriever (mirrors plugins/memory/holographic/retrieval.py search path,
# including the parent's per-candidate encode_text call)
# ---------------------------------------------------------------------------

class FactRetriever:
    def __init__(self, store, temporal_decay_half_life: int = 0,
                 fts_weight: float = 0.4, jaccard_weight: float = 0.3,
                 hrr_weight: float = 0.3, hrr_dim: int = 1024):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim
        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight

    def search(self, query, category=None, min_trust: float = 0.3, limit: int = 10):
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)
        if not candidates:
            return []
        query_tokens = self._tokenize(query)
        scored = []
        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens
            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
                query_vec = hrr.encode_text(query, self.hrr_dim)
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0
            else:
                hrr_sim = 0.5
            relevance = (self.fts_weight * fts_score
                         + self.jaccard_weight * jaccard
                         + self.hrr_weight * hrr_sim)
            fact["score"] = relevance * fact["trust_score"]
            scored.append(fact)
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        for fact in results:
            fact.pop("hrr_vector", None)
        return results

    def _fts_candidates(self, query, category, min_trust, limit):
        conn = self.store._conn
        params = [query]
        where_clauses = ["facts_fts MATCH ?"]
        if category:
            where_clauses.append("f.category = ?")
            params.append(category)
        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)
        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
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
            return []
        if not rows:
            return []
        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(max(raw_ranks) if raw_ranks else 1.0, 1e-6)
        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank
            results.append(fact)
        return results

    @staticmethod
    def _tokenize(text):
        if not text:
            return set()
        tokens = set()
        for word in text.lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    @staticmethod
    def _jaccard_similarity(set_a, set_b):
        if not set_a or not set_b:
            return 0.0
        union = len(set_a | set_b)
        return len(set_a & set_b) / union if union > 0 else 0.0

    def _temporal_decay(self, timestamp_str):
        return 1.0


# ---------------------------------------------------------------------------
# Parent provider (mirrors the HolographicMemoryProvider surface the subclass uses)
# ---------------------------------------------------------------------------

class HolographicMemoryProvider:
    def __init__(self, config=None):
        self._config = config or {}
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

    @property
    def name(self):
        return "holographic"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self._store = MemoryStore(
            db_path=self._config["db_path"],
            default_trust=float(self._config.get("default_trust", 0.5)),
            hrr_dim=int(self._config.get("hrr_dim", 64)),
        )
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=int(self._config.get("temporal_decay_half_life", 0)),
            hrr_dim=int(self._config.get("hrr_dim", 64)),
        )
        self._session_id = session_id

    def on_session_end(self, messages):
        pass

    def on_memory_write(self, action, target, content):
        if action == "add" and self._store and content:
            category = "user_pref" if target == "user" else "general"
            self._store.add_fact(content, category=category)

    def handle_tool_call(self, tool_name, args, **kwargs):
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _handle_fact_store(self, args):
        action = args.get("action")
        try:
            if action == "add":
                fact_id = self._store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                return json.dumps({"fact_id": fact_id, "status": "added"})
            if action == "update":
                updated = self._store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                return json.dumps({"updated": updated})
            if action == "remove":
                removed = self._store.remove_fact(int(args["fact_id"]))
                return json.dumps({"removed": removed})
            if action == "list":
                facts = self._store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})
            return json.dumps({"error": f"Unknown action: {action}"})
        except KeyError as exc:
            return json.dumps({"error": f"Missing required argument: {exc}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def shutdown(self):
        self._store = None
        self._retriever = None


# ---------------------------------------------------------------------------
# Fake embedder for the plugin's embedding layer
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Deterministic in-process embedder. No model, no network."""

    backend = "fake"

    def __init__(self, dim: int = 8, table=None):
        self.dim = dim
        self.table = table or {}
        self.embed_calls = []

    def _vec(self, text: str):
        if text in self.table:
            vec = np.asarray(self.table[text], dtype=np.float32)
        else:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed(self, text):
        if not text or not text.strip():
            return None
        self.embed_calls.append(text)
        return self._vec(text)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def is_available(self):
        return True


# ---------------------------------------------------------------------------
# sys.modules installation
# ---------------------------------------------------------------------------

def _module(name: str, is_pkg: bool = False) -> types.ModuleType:
    mod = types.ModuleType(name)
    if is_pkg:
        mod.__path__ = []
    return mod


def install_stubs() -> None:
    """Inject the fake hermes internals into sys.modules (idempotent)."""
    if "plugins.memory.holographic" in sys.modules:
        return

    plugins_pkg = _module("plugins", is_pkg=True)
    memory_pkg = _module("plugins.memory", is_pkg=True)
    holo_pkg = _module("plugins.memory.holographic", is_pkg=True)
    retrieval_mod = _module("plugins.memory.holographic.retrieval")
    store_mod = _module("plugins.memory.holographic.store")

    retrieval_mod.FactRetriever = FactRetriever
    store_mod.MemoryStore = MemoryStore
    holo_pkg.HolographicMemoryProvider = HolographicMemoryProvider
    holo_pkg.holographic = hrr
    holo_pkg.retrieval = retrieval_mod
    holo_pkg.store = store_mod
    memory_pkg.holographic = holo_pkg
    plugins_pkg.memory = memory_pkg

    agent_pkg = _module("agent", is_pkg=True)
    aux_mod = _module("agent.auxiliary_client")

    def call_llm(**kwargs):
        raise RuntimeError("stub call_llm not configured for this test")

    aux_mod.call_llm = call_llm
    agent_pkg.auxiliary_client = aux_mod

    constants_mod = _module("hermes_constants")
    _fake_home = Path(tempfile.mkdtemp(prefix="hp-test-home-"))

    def get_hermes_home():
        return _fake_home

    constants_mod.get_hermes_home = get_hermes_home
    constants_mod.display_hermes_home = lambda: str(_fake_home)

    sys.modules["plugins"] = plugins_pkg
    sys.modules["plugins.memory"] = memory_pkg
    sys.modules["plugins.memory.holographic"] = holo_pkg
    sys.modules["plugins.memory.holographic.retrieval"] = retrieval_mod
    sys.modules["plugins.memory.holographic.store"] = store_mod
    sys.modules["plugins.memory.holographic.holographic"] = hrr
    sys.modules["agent"] = agent_pkg
    sys.modules["agent.auxiliary_client"] = aux_mod
    sys.modules["hermes_constants"] = constants_mod
