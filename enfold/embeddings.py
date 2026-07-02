"""Embedding clients and serialization helpers for enfold.

Provides:
  - OllamaEmbedder: thin HTTP client for the /api/embed endpoint
  - FastEmbedder: local ONNX/FastEmbed client for CPU-optimized canaries
  - embedding_to_bytes / bytes_to_embedding: serialization helpers for SQLite BLOB storage
"""

from __future__ import annotations

import logging
import urllib.request
import urllib.error
import json
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default Ollama endpoint and model
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen3-embedding:8b"
_EMBED_DIM = 4096


class OllamaEmbedder:
    """Thin synchronous HTTP client for Ollama /api/embed.

    Handles connection errors gracefully: returns None on failure so callers
    can fall back to holographic-only scoring. Passes ``keep_alive`` on every
    request so the embedding model stays resident between calls.
    """

    backend = "ollama"

    def __init__(
        self,
        base_url: str = _DEFAULT_OLLAMA_URL,
        model: str = _DEFAULT_MODEL,
        timeout: float = 30.0,
        keep_alive: int | str = -1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Sent to Ollama on every embed call so the model stays resident.
        # -1 keeps it loaded indefinitely, avoiding a multi-second cold-load on
        # the first memory read/write after an idle gap. A duration string like
        # "30m" or a second count also works; 0 unloads immediately.
        self.keep_alive = keep_alive
        self._embed_url = f"{self.base_url}/api/embed"

    def embed(self, text: str) -> Optional[np.ndarray]:
        """Return a float32 embedding vector for *text*, or None on error."""
        if not text or not text.strip():
            return None
        payload = json.dumps(
            {"model": self.model, "input": text, "keep_alive": self.keep_alive}
        ).encode()
        req = urllib.request.Request(
            self._embed_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            logger.debug("Ollama embed request failed (URLError): %s", exc)
            return None
        except Exception as exc:
            logger.debug("Ollama embed request failed: %s", exc)
            return None

        embeddings = data.get("embeddings")
        if not embeddings or not isinstance(embeddings, list) or not embeddings[0]:
            logger.debug("Ollama embed: unexpected response shape: %s", list(data.keys()))
            return None

        vec = np.array(embeddings[0], dtype=np.float32)
        # Normalise in place so cosine similarity is just a dot product
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Embed a list of texts, returning one vector (or None) per text.

        Blank/whitespace inputs map to None without hitting the server, so the
        result is position-equivalent to calling embed() on each item. Modern
        Ollama accepts list input for /api/embed; falls back to sequential calls
        if a local build rejects it.
        """
        if not texts:
            return []
        send_idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if not send_idx:
            return [None for _ in texts]
        send_texts = [texts[i] for i in send_idx]
        payload = json.dumps(
            {"model": self.model, "input": send_texts, "keep_alive": self.keep_alive}
        ).encode()
        req = urllib.request.Request(
            self._embed_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        result: List[Optional[np.ndarray]] = [None] * len(texts)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and len(embeddings) == len(send_texts):
                for pos, raw in zip(send_idx, embeddings):
                    if not raw:
                        continue
                    vec = np.array(raw, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm
                    result[pos] = vec
                return result
            logger.debug("Ollama batch embed: length mismatch, falling back to sequential")
        except Exception as exc:
            logger.debug("Ollama batch embed request failed, falling back: %s", exc)
        return [self.embed(t) for t in texts]

    def is_available(self) -> bool:
        """Quick liveness check: returns True if Ollama is reachable."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5.0):
                return True
        except Exception:
            return False


class FastEmbedder:
    """Local FastEmbed/ONNX embedding client.

    This keeps memory embedding private/local while letting CPU-optimized ONNX
    models run outside Ollama. It intentionally mirrors OllamaEmbedder's small
    interface: embed(), embed_batch(), is_available().
    """

    backend = "fastembed"

    def __init__(
        self,
        model: str = "BAAI/bge-base-en-v1.5",
        cache_dir: Optional[str] = None,
        **_: object,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self._model = None
        self._load_error: Optional[Exception] = None

    def _client(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                kwargs = {"model_name": self.model}
                if self.cache_dir:
                    kwargs["cache_dir"] = self.cache_dir
                self._model = TextEmbedding(**kwargs)
            except Exception as exc:
                self._load_error = exc
                logger.debug("FastEmbed model load failed: %s", exc)
                return None
        return self._model

    @staticmethod
    def _normalise(raw) -> Optional[np.ndarray]:
        if raw is None:
            return None
        vec = np.array(raw, dtype=np.float32)
        if vec.size == 0:
            return None
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, text: str) -> Optional[np.ndarray]:
        if not text or not text.strip():
            return None
        client = self._client()
        if client is None:
            return None
        try:
            first = next(iter(client.embed([text])))
            return self._normalise(first)
        except Exception as exc:
            logger.debug("FastEmbed embed failed: %s", exc)
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        if not texts:
            return []
        client = self._client()
        if client is None:
            return [None for _ in texts]
        # Blank inputs map to None without hitting the model (matches embed()).
        send_idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if not send_idx:
            return [None for _ in texts]
        send_texts = [texts[i] for i in send_idx]
        result: List[Optional[np.ndarray]] = [None] * len(texts)
        try:
            for pos, raw in zip(send_idx, client.embed(send_texts)):
                result[pos] = self._normalise(raw)
            return result
        except Exception as exc:
            logger.debug("FastEmbed batch embed failed: %s", exc)
            return [self.embed(t) for t in texts]

    def is_available(self) -> bool:
        return self._client() is not None


# ---------------------------------------------------------------------------
# Vector serialization
# ---------------------------------------------------------------------------

def embedding_to_bytes(vec: np.ndarray) -> bytes:
    """Serialize a float32 numpy vector to raw bytes for SQLite BLOB storage.

    Uses an explicit little-endian dtype so a database written on one CPU
    architecture reads back correctly on another.
    """
    return np.asarray(vec, dtype="<f4").tobytes()


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialize a little-endian float32 BLOB from SQLite into a numpy vector."""
    return np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
