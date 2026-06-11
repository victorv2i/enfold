"""The _deferred_bank_rebuild seam: suppress per-add rebuilds, one per category."""

import threading

import fake_hermes


def test_rebuilds_suppressed_during_batch_then_one_per_category(make_provider, monkeypatch):
    provider = make_provider()
    store = provider._store

    rebuilds = []
    monkeypatch.setattr(
        fake_hermes.MemoryStore,
        "_rebuild_bank",
        lambda self, category: rebuilds.append(category),
    )

    with provider._deferred_bank_rebuild():
        store.add_fact("fact one about tools", category="tool")
        store.add_fact("fact two about tools", category="tool")
        store.add_fact("fact three about a project", category="project")
        assert rebuilds == [], "per-add rebuilds must be suppressed inside the batch"

    assert sorted(rebuilds) == ["project", "tool"]


def test_rebuilds_run_after_shadow_pop_and_without_store_lock(make_provider, monkeypatch):
    """The real rebuilds must run outside the batch's store lock.

    Each rebuild probes the store lock from ANOTHER thread: if the batching
    thread still held the outer lock during the rebuild loop, the probe could
    not acquire it and worker batches would block turn-thread prefetch and
    tool adds for the whole adds+rebuilds stretch.
    """
    provider = make_provider()
    store = provider._store

    observations = []

    def probing_rebuild(self, category):
        shadow_present = "_rebuild_bank" in self.__dict__
        acquired = []

        def probe():
            if self._lock.acquire(timeout=1.0):
                acquired.append(True)
                self._lock.release()

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
        observations.append((category, shadow_present, bool(acquired)))

    monkeypatch.setattr(fake_hermes.MemoryStore, "_rebuild_bank", probing_rebuild)

    with provider._deferred_bank_rebuild():
        store.add_fact("lock probe fact about tools", category="tool")
        store.add_fact("lock probe fact about a project", category="project")

    assert sorted(category for category, _, _ in observations) == ["project", "tool"]
    for category, shadow_present, lock_was_free in observations:
        assert not shadow_present, (
            f"instance shadow must be popped before the rebuild of {category!r}"
        )
        assert lock_was_free, (
            f"store lock must not be held during the rebuild of {category!r}"
        )


def test_instance_shadow_removed_after_batch(make_provider):
    provider = make_provider()
    store = provider._store
    with provider._deferred_bank_rebuild():
        assert "_rebuild_bank" in store.__dict__
    assert "_rebuild_bank" not in store.__dict__, "class method must be restored"
    # Normal adds rebuild immediately again (no exception path left behind)
    store.add_fact("a fact after the batch", category="general")
    row = store._conn.execute(
        "SELECT fact_count FROM memory_banks WHERE bank_name = 'cat:general'"
    ).fetchone()
    assert row is not None and row[0] == 1


def test_seam_restores_on_exception(make_provider):
    provider = make_provider()
    store = provider._store
    try:
        with provider._deferred_bank_rebuild():
            store.add_fact("fact before failure", category="tool")
            raise ValueError("boom")
    except ValueError:
        pass
    assert "_rebuild_bank" not in store.__dict__
    # The category touched before the exception still got its rebuild
    row = store._conn.execute(
        "SELECT fact_count FROM memory_banks WHERE bank_name = 'cat:tool'"
    ).fetchone()
    assert row is not None and row[0] == 1


def test_noop_when_store_missing(hp):
    provider = hp.HolographicPlusProvider(config={})
    with provider._deferred_bank_rebuild():
        pass  # must not raise without an initialized store
