import numpy as np
import pytest

from holographic_plus.embeddings import (
    FastEmbedder,
    OllamaEmbedder,
    bytes_to_embedding,
    embedding_to_bytes,
)


def test_serialization_roundtrip():
    v = np.array([0.1, 0.2, 0.3, -0.4], dtype=np.float32)
    restored = bytes_to_embedding(embedding_to_bytes(v))
    assert restored.dtype == np.float32
    assert np.array_equal(restored, v)


def test_ollama_unavailable_is_graceful():
    # Unreachable port: must return None / False, never raise.
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


def test_embed_sends_keep_alive(monkeypatch):
    """Every embed request must carry keep_alive so Ollama keeps the model
    resident; otherwise each memory read/write after an idle gap pays a
    multi-second cold-load. Default is -1 (keep loaded indefinitely)."""
    import json as _json
    import urllib.request as _ur

    captured = {}

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self) -> bytes:
            return self._body

    def _fake_urlopen(req, timeout=None):
        payload = _json.loads(req.data.decode())
        captured["payload"] = payload
        n = len(payload["input"]) if isinstance(payload["input"], list) else 1
        body = _json.dumps({"embeddings": [[0.1, 0.2, 0.3]] * n}).encode()
        return _Resp(body)

    monkeypatch.setattr(_ur, "urlopen", _fake_urlopen)

    e = OllamaEmbedder()  # default keep_alive=-1
    assert e.embed("hello") is not None
    assert captured["payload"]["keep_alive"] == -1

    assert len(e.embed_batch(["alpha", "beta"])) == 2
    assert captured["payload"]["keep_alive"] == -1
    assert captured["payload"]["input"] == ["alpha", "beta"]
