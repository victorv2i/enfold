from __future__ import annotations

import sqlite3

from memory_eval.schema_checks import inspect_memory_schema


def test_inspect_memory_schema_reports_embedding_coverage_and_empty_queue(tmp_path):
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT, hrr_vector BLOB)")
    conn.execute("CREATE TABLE fact_embeddings (fact_id INTEGER, embedding BLOB, dim INTEGER, embedding_identity TEXT)")
    conn.execute("CREATE TABLE extract_queue (id INTEGER PRIMARY KEY, status TEXT)")
    conn.executemany("INSERT INTO facts VALUES (?, ?, ?)", [(1, "one", b"hrr"), (2, "two", None)])
    conn.execute("INSERT INTO fact_embeddings VALUES (?, ?, ?, ?)", (1, b"vec", 3, "ollama:embeddinggemma:document:auto:v1"))
    conn.commit()
    conn.close()

    report = inspect_memory_schema(db, current_embedding_identity="ollama:embeddinggemma:document:auto:v1")

    assert report["counts"]["facts"] == 2
    assert report["counts"]["fact_embeddings"] == 1
    assert report["coverage"]["embedding"] == 0.5
    assert report["coverage"]["hrr"] == 0.5
    assert report["extract_queue"]["pending"] == 0


def test_inspect_memory_schema_flags_missing_sota_layers(tmp_path):
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT)")
    conn.commit()
    conn.close()

    report = inspect_memory_schema(db)

    assert report["sota_gates"]["provenance_tables"] is False
    assert report["sota_gates"]["temporal_supersession"] is False
    assert "fact_provenance" in report["missing"]["tables"]
    assert "valid_from" in report["missing"]["fact_columns"]


def test_inspect_memory_schema_counts_non_terminal_extract_queue_statuses(tmp_path):
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("CREATE TABLE extract_queue (id INTEGER PRIMARY KEY, status TEXT)")
    conn.executemany(
        "INSERT INTO extract_queue VALUES (?, ?)",
        [
            (1, "pending"),
            (2, "retrying"),
            (3, "done"),
            (4, "failed"),
        ],
    )
    conn.commit()
    conn.close()

    report = inspect_memory_schema(db)

    assert report["extract_queue"]["pending"] == 2
    assert report["extract_queue"]["empty"] is False
