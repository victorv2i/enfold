"""Offline recall diagnostics CLI: explain_search() against a DB copy.

Standalone maintenance script mirroring ``cluster_merge.py``'s conventions:
meant to run against a *copy* of a fact store, never a live ``.hermes`` path,
and prints a JSON breakdown of how each candidate scored for a query (FTS,
Jaccard, HRR, entity boost, trust, raw embedding cosine, blended final score)
plus why any candidate was excluded (superseded, temporally invalid, or
ranked past the limit).

Usage::

    python -m holographic_plus.explain /path/to/facts-copy.db "query text"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

from . import HolographicPlusProvider


class GuardRailError(Exception):
    """Raised when explain refuses to run against a live path."""


def _refuse_if_live_path(db_path: str) -> None:
    literal_parts = str(db_path).replace("\\", "/").split("/")
    resolved_parts = os.path.realpath(db_path).replace("\\", "/").split("/")
    if ".hermes" in literal_parts or ".hermes" in resolved_parts:
        raise GuardRailError(
            f"refusing to run against a path under .hermes ({db_path}); "
            "this tool only ever runs against a copy"
        )


def explain(
    db_path: str,
    query: str,
    limit: int = 10,
    embedding_backend: str = "ollama",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "embeddinggemma:latest",
    embedding_prefix_policy: str = "auto",
    hrr_dim: int = 1024,
) -> list:
    """Run explain_search() against *db_path* and return the breakdown rows.

    Builds a real provider (same scoring core search() uses) pointed at the
    DB copy so the breakdown can never drift from actual ranking. If the
    embedding backend is unreachable this degrades to holographic-only
    diagnostics, same as a live provider would. *hrr_dim* must match the
    dimension the copied store's HRR vectors were encoded at (default 1024,
    the live default) or HRR similarity raises a shape mismatch.
    """
    _refuse_if_live_path(db_path)

    config = {
        "db_path": db_path,
        "embedding_backend": embedding_backend,
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
        "embedding_prefix_policy": embedding_prefix_policy,
        "hrr_dim": hrr_dim,
    }
    provider = HolographicPlusProvider(config=config)
    try:
        provider.initialize("explain-cli")
        return provider.explain_search(query, limit=limit)
    finally:
        provider.shutdown()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", help="path to the fact store copy to inspect")
    parser.add_argument("query", help="query text to explain")
    parser.add_argument("--limit", type=int, default=10,
                         help="number of kept results to rank (default 10)")
    parser.add_argument("--embedding-backend", default="ollama",
                         help="embedding backend: ollama or fastembed (default ollama)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                         help="ollama server URL (default http://localhost:11434)")
    parser.add_argument("--ollama-model", default="embeddinggemma:latest",
                         help="ollama model name (default embeddinggemma:latest)")
    parser.add_argument("--embedding-prefix-policy", default="auto",
                         help="none or auto (default auto)")
    parser.add_argument("--hrr-dim", type=int, default=1024,
                         help="HRR vector dimension the store was encoded at (default 1024)")
    args = parser.parse_args(argv)

    try:
        rows = explain(
            args.db_path,
            args.query,
            limit=args.limit,
            embedding_backend=args.embedding_backend,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            embedding_prefix_policy=args.embedding_prefix_policy,
            hrr_dim=args.hrr_dim,
        )
    except GuardRailError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
