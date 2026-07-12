import sqlite3

import pytest

from enfold.schema import migrate


def test_native_legacy_provider_refuses_v1_before_parent_startup(make_provider, tmp_path):
    db_path = tmp_path / "facts.db"
    conn = sqlite3.connect(str(db_path))
    try:
        migrate(conn)
    finally:
        conn.close()

    provider = make_provider(init=False)
    with pytest.raises(
        RuntimeError,
        match="schema-v0 writer.*standalone Enfold service.*No migration was attempted",
    ):
        provider.initialize("legacy-writer")

    assert provider._store is None


def test_native_legacy_provider_keeps_v0_behavior(make_provider):
    provider = make_provider()
    assert provider._store is not None
    assert provider._store.add_fact("v0 remains writable") > 0
