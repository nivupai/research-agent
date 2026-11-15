"""Where do papers about a drug rank for the deterministic drug-discovery corpus searches?

Usage: uv run python scripts/discovery_recall.py --target AMPK --drug sorafenib [--depth 100]

Runs the same three corpus queries as discover_drugs (hybrid_rerank) to --depth, finds the
corpus papers whose title/abstract mention the drug together with the target, and prints each
one's rank per query (the agent keeps the top DISCOVERY_K). Local only: no API calls.
"""

from __future__ import annotations

import argparse
import re

from bioagent.agents.nodes import DISCOVERY_K, DISCOVERY_QUERIES
from bioagent.config import get_settings
from bioagent.retrieval.retriever import retriever_from_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="AMPK")
    parser.add_argument("--target-pattern", default=r"AMPK|AMP-activated protein kinase")
    parser.add_argument("--drug", default="sorafenib")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--mode", default="hybrid_rerank")
    parser.add_argument("--query", action="append", help="extra query templates ({target})")
    args = parser.parse_args()

    retriever = retriever_from_settings(get_settings())
    drug = re.compile(re.escape(args.drug), re.I)
    target = re.compile(args.target_pattern, re.I)
    papers = []
    for row in retriever.store.iter_records():
        rec = dict(row)
        text = f"{rec.get('title') or ''} {rec.get('abstract') or ''}"
        if drug.search(text) and target.search(text):
            papers.append(rec)
    print(f"{len(papers)} corpus papers mention {args.drug} and {args.target}:")
    for rec in papers:
        print(f"  {rec['pmid']} {rec.get('year')} {str(rec.get('title'))[:100]}")
    wanted = {str(rec["pmid"]) for rec in papers}
    for q in [*DISCOVERY_QUERIES, *(args.query or [])]:
        query = q.format(target=args.target)
        hits = retriever.search(query, k=args.depth, mode=args.mode).hits  # type: ignore[arg-type]
        ranked = list(dict.fromkeys(str(h.pmid) for h in hits))
        ranks = {pmid: ranked.index(pmid) + 1 for pmid in wanted if pmid in ranked}
        kept = sum(1 for r in ranks.values() if r <= DISCOVERY_K)
        print(f"\n{query!r}: {kept} in top {DISCOVERY_K}; ranks within {args.depth}: {ranks}")


if __name__ == "__main__":
    main()
