"""Daemon-owned lifecycle for the bounded extraction processor."""

from __future__ import annotations

import threading
import time
from typing import Any

from .extraction_processor import ExtractionProcessor


class SupervisedExtractionWorker:
    """Run bounded queue drains and stop claiming as soon as shutdown begins."""

    def __init__(
        self,
        processor: ExtractionProcessor,
        *,
        poll_seconds: float = 1.0,
        drain_limit: int = 4,
    ) -> None:
        if poll_seconds <= 0 or drain_limit < 1:
            raise ValueError("worker polling configuration is invalid")
        self.processor = processor
        self.poll_seconds = float(poll_seconds)
        self.drain_limit = drain_limit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._heartbeat: float | None = None
        self._last_success: float | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("extraction worker already started")
        self._thread = threading.Thread(
            target=self._run, name="enfold-extraction-worker", daemon=False
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._pulse()
            try:
                for _ in range(self.drain_limit):
                    # This check is the claim boundary: shutdown never starts a
                    # new model call, but lets the currently leased call finish.
                    if self._stop.is_set():
                        break
                    result = self.processor.process_one()
                    self._pulse()
                    with self._lock:
                        if result.outcome == "idle":
                            self._last_error = None
                            break
                        if result.outcome == "completed":
                            self._last_success = time.monotonic()
                            self._last_error = None
                        elif result.outcome in {"retry", "dead"}:
                            # Processor errors are stable redacted error codes.
                            self._last_error = result.error or f"job_{result.outcome}"
            except Exception:
                # Never publish exception text or environment-derived details.
                with self._lock:
                    self._last_error = "worker_failure"
            self._stop.wait(self.poll_seconds)

    def _pulse(self) -> None:
        with self._lock:
            self._heartbeat = time.monotonic()

    def health(self, *, stale_after: float = 10.0) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            heartbeat = self._heartbeat
            age = None if heartbeat is None else now - heartbeat
            return {
                "running": bool(self._thread and self._thread.is_alive()),
                "stopping": self._stop.is_set(),
                "heartbeat_age_seconds": age,
                "heartbeat_stale": age is None or age > stale_after,
                "last_success_age_seconds": (
                    None if self._last_success is None else now - self._last_success
                ),
                "last_error": self._last_error,
            }

    def stop(self, timeout: float = 5.0) -> None:
        """Stop future claims and wait no longer than the caller's bound."""

        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("extraction worker did not stop cleanly")
