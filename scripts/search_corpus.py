"""Query the local corpus from the command line and compare retrieval modes.

Usage: uv run python scripts/search_corpus.py "metformin AMPK glioblastoma" -k 5 \
           --modes bm25 dense hybrid_rerank
"""

from __future__ import annotations

import argparse

from bioagent.config import get_settings
from bioagent.retrieval.models import SEARCH_MODES
from bioagent.retrieval.retriever import retriever_from_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="+")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--modes", nargs="+", default=["bm25", "dense", "hybrid_rerank"], choices=SEARCH_MODES
    )
    parser.add_argument("--year-min", type=int)
    args = parser.parse_args()

    retriever = retriever_from_settings(get_settings())
    filters = {"year_min": args.year_min} if args.year_min else None
    for mode in args.modes:  # warm up models so timings below are steady-state
        retriever.search("warm up", k=1, mode=mode)
    for query in args.queries:
        print(f"\n## {query}")
        for mode in args.modes:
            results = retriever.search(query, k=args.k, mode=mode, filters=filters)
            print(f"\n### {mode}  ({results.timings_ms['total']:.0f} ms)")
            for i, hit in enumerate(results.hits, start=1):
                title = (hit.title or "")[:110]
                print(f"{i}. [{hit.pmid} {hit.year}] {title}  (score {hit.score:.3f})")


if __name__ == "__main__":
    main()
