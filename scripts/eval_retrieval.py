"""Evaluate every retrieval mode against graded qrels and save the results.

Usage: uv run python scripts/eval_retrieval.py --qrels data/labels/pool.jsonl [-k 10]

The qrels file is JSONL with query_id, query, pmid and grade (0/1/2); lines with
grade null are ignored, so a graded label pool works directly. Writes a markdown table and
JSON to results/retrieval/.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from bioagent.config import get_settings
from bioagent.eval.retrieval import evaluate, load_qrels, markdown_table
from bioagent.retrieval.models import SEARCH_MODES
from bioagent.retrieval.retriever import retriever_from_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=Path("data/labels/pool.jsonl"))
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--modes", nargs="+", default=list(SEARCH_MODES), choices=SEARCH_MODES)
    parser.add_argument("--out", type=Path, default=Path("results/retrieval"))
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    if not qrels.queries:
        raise SystemExit(f"no graded judgements in {args.qrels}; fill in the 'grade' fields")

    settings = get_settings()
    retriever = retriever_from_settings(settings)
    for mode in args.modes:  # load models before timing
        retriever.search("warm up", k=1, mode=mode)

    def search(query: str, mode: str, depth: int) -> list[str]:
        hits = retriever.search(query, k=depth, mode=mode).hits  # type: ignore[arg-type]
        return [h.pmid for h in hits]

    results = evaluate(search, qrels, args.modes, k=args.k)
    table = markdown_table(results, k=args.k)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "qrels": str(args.qrels),
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "k": args.k,
        "timestamp": stamp,
    }
    (args.out / f"retrieval_{stamp}.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2)
    )
    (args.out / f"retrieval_{stamp}.md").write_text(
        f"# Retrieval eval {stamp}\n\n{json.dumps(meta)}\n\n{table}\n"
    )
    print(table)
    print(f"\nsaved to {args.out}/retrieval_{stamp}.{{json,md}}")


if __name__ == "__main__":
    main()
