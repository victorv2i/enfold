"""Smoke import, update-path async embedding, and 3.11 syntax compatibility."""

import ast
import json
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "enfold"
PLUGIN_SOURCES = sorted(PLUGIN_DIR.glob("*.py"))


def test_smoke_import_surface(hp):
    assert callable(hp.register)
    assert hasattr(hp, "EnfoldProvider")
    provider = hp.EnfoldProvider(config={"embedding_weight": 0.3})
    assert provider.name == "enfold"
    assert provider._embed_weight == 0.3
    # Constructing must not create stores, threads, or pools
    assert provider._store is None
    assert provider._queue_worker is None
    assert provider._embed_pool is None


def test_update_path_embeds_via_pool(make_provider):
    provider = make_provider()
    add_result = json.loads(
        provider.handle_tool_call(
            "fact_store", {"action": "add", "content": "original fact content"}
        )
    )
    fact_id = add_result["fact_id"]

    submitted = []
    provider._submit_embed = lambda fn, *args: submitted.append((fn, args))

    update_result = json.loads(
        provider.handle_tool_call(
            "fact_store",
            {"action": "update", "fact_id": fact_id, "content": "updated fact content"},
        )
    )
    assert update_result["updated"] is True
    assert len(submitted) == 1
    fn, args = submitted[0]
    assert fn == provider._embed_and_store
    assert args == (fact_id, "updated fact content")


def test_sources_use_no_python_312_plus_syntax():
    """The gateway runs Python 3.11; reject PEP 695 syntax and friends."""
    assert PLUGIN_SOURCES, "no plugin sources found"
    new_syntax_nodes = tuple(
        node for node in (
            getattr(ast, "TypeAlias", None),       # type X = ... (3.12)
            getattr(ast, "TypeVar", None),         # def f[T]() (3.12)
            getattr(ast, "TypeVarTuple", None),
            getattr(ast, "ParamSpec", None),
        )
        if node is not None
    )
    for source_file in PLUGIN_SOURCES:
        tree = ast.parse(source_file.read_text(), filename=str(source_file))
        for node in ast.walk(tree):
            assert not isinstance(node, new_syntax_nodes), (
                f"{source_file.name}:{getattr(node, 'lineno', '?')} uses 3.12+ syntax"
            )


def test_existing_summary_includes_recent_low_trust_facts(hp, make_provider):
    """The dedup context must see freshly inserted trust-0.5 facts.

    A pure top-by-trust window fills up with established high-trust facts,
    so repeated pre-compress events would re-extract paraphrases of facts
    that were just stored. The recent slice keeps them in the prompt.
    """
    provider = make_provider()
    store = provider._store
    for i in range(45):
        fact_id = store.add_fact(
            f"established high trust fact number {i}", category="general"
        )
        store.update_fact(fact_id, trust_delta=0.4)  # 0.5 -> 0.9
    store.add_fact("The user just adopted the uv package manager", category="tool")

    summary = hp.llm_extract._existing_summary(store)
    assert "The user just adopted the uv package manager" in summary
    assert len(summary.splitlines()) <= 40


def test_prefetch_renders_results(make_provider):
    provider = make_provider()
    provider._store.add_fact("The user prefers pnpm for node projects", category="tool")
    block = provider.prefetch("pnpm projects")
    assert block.startswith("## Enfold Memory")
    assert "pnpm" in block
