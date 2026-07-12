"""Per-request read routing and bounded query-embedding reuse."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from contextlib import closing
import sqlite3
import threading
from typing import Any, Protocol

from .daemon import READ_ONLY_METHODS, RequestHandler
from .protocol import ClientContext, Request


class _QueryEmbedder(Protocol):
    def embed(self, text: str) -> Any: ...


class LRUQueryEmbedder:
    """Thread-safe LRU around a local model without serializing model calls."""

    def __init__(self, backend: _QueryEmbedder, *, model: str, maxsize: int = 256):
        if not model.strip():
            raise ValueError("model must not be empty")
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self._backend = backend
        self._model = model
        self._maxsize = maxsize
        self._cache: OrderedDict[tuple[str, str], tuple[float, ...]] = OrderedDict()
        self._lock = threading.Lock()

    def embed(self, text: str) -> tuple[float, ...] | None:
        key = (self._model, text)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        # Model I/O deliberately happens without the cache lock: one slow
        # query must not serialize unrelated query embeddings.
        vector = self._backend.embed(text)
        if vector is None:
            return None
        immutable = tuple(float(value) for value in vector)
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                return existing
            self._cache[key] = immutable
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        return immutable


class PerRequestReadHandler:
    """Route explicit reads through a fresh read-only SQLite connection."""

    def __init__(
        self,
        mutation_handler: RequestHandler,
        open_read_connection: Callable[[], sqlite3.Connection],
        build_read_handler: Callable[[sqlite3.Connection], RequestHandler],
    ):
        self._mutation_handler = mutation_handler
        self._open_read_connection = open_read_connection
        self._build_read_handler = build_read_handler

    def __call__(self, context: ClientContext, request: Request) -> Any:
        if request.method not in READ_ONLY_METHODS:
            return self._mutation_handler(context, request)
        with closing(self._open_read_connection()) as connection:
            return self._build_read_handler(connection)(context, request)
