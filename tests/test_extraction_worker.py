from __future__ import annotations

import threading
import time

import pytest

from enfold.extraction_processor import ExtractionProcessResult
from enfold.extraction_worker import SupervisedExtractionWorker


class BlockingProcessor:
    def __init__(self):
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def process_one(self):
        self.calls += 1
        self.entered.set()
        self.release.wait(1.0)
        return ExtractionProcessResult("completed", self.calls, writes=1)


def test_shutdown_stops_new_claims_and_wait_is_bounded():
    processor = BlockingProcessor()
    worker = SupervisedExtractionWorker(
        processor, poll_seconds=0.01, drain_limit=8
    )
    worker.start()
    assert processor.entered.wait(0.5)

    with pytest.raises(RuntimeError, match="did not stop cleanly"):
        worker.stop(0.01)
    assert worker.health()["stopping"] is True

    processor.release.set()
    worker.stop(0.5)
    assert processor.calls == 1


class BrokenProcessor:
    def process_one(self):
        raise RuntimeError("secret raw exception details")


def test_health_redacts_unexpected_worker_errors():
    worker = SupervisedExtractionWorker(
        BrokenProcessor(), poll_seconds=0.01, drain_limit=1
    )
    worker.start()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if worker.health()["last_error"] is not None:
            break
        time.sleep(0.005)
    state = worker.health()
    worker.stop(0.5)
    assert state["last_error"] == "worker_failure"
    assert "secret" not in str(state)
