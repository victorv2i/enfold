import numpy as np
import pytest

from holographic_plus.embeddings import (
    FastEmbedder,
    OllamaEmbedder,
    bytes_to_embedding,
    cosine_similarity,
    embedding_to_bytes,
)


def test_serialization_roundtrip():
    v = np.array([0.1, 0.2, 0.3, -0.4], dtype=np.float32)
    restored = bytes_to_embedding(embedding_to_bytes(v))
    assert restored.dtype == np.float32
    assert np.array_equal(restored, v)


def test_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, np.array([0.0, 1.0, 0.0], dtype=np.float32)) == pytest.approx(0.0)
    # zero vector must not divide-by-zero
    assert cosine_similarity(np.zeros(3, dtype=np.float32), a) == 0.0


def test_ollama_unavailable_is_graceful():
    # Unreachable port — must return None / False, never raise.
    e = OllamaEmbedder(base_url="http://127.0.0.1:1", timeout=0.5)
    assert e.is_available() is False
    assert e.embed("anything") is None


def test_embed_batch_matches_single():
    e = FastEmbedder()
    if not e.is_available():
        pytest.skip("fastembed not installed in this environment")
    texts = ["alpha beta gamma", "the quick brown fox", "memory retrieval system"]
    single = [e.embed(t) for t in texts]
    batch = e.embed_batch(texts)
    assert len(batch) == len(single) == 3
    for s, b in zip(single, batch):
        assert s is not None and b is not None
        assert float(np.max(np.abs(s - b))) < 1e-5  # batch == one-at-a-time
