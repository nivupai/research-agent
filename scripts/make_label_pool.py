"""Pool top results from every retrieval mode for manual relevance grading.

Usage: uv run python scripts/make_label_pool.py [--depth 20]

For each query in configs/retrieval_queries.yaml, takes the top --depth PMIDs from each mode,
merges them, and writes data/labels/pool.jsonl with title and abstract. Set "grade" to
0 (not relevant), 1 (partially relevant) or 2 (highly relevant); leave null to skip.
Existing grades in the output file are preserved when the pool is regenerated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from bioagent.config import get_settings
from bioagent.retrieval.models import SEARCH_MODES
from bioagent.retrieval.retriever import retriever_from_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=Path("configs/retrieval_queries.yaml"))
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("data/labels/pool.jsonl"))
    args = parser.parse_args()

    queries = yaml.safe_load(args.queries.read_text())["queries"]
    existing: dict[tuple[str, str], object] = {}
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("grade") is not None:
                    existing[(row["query_id"], row["pmid"])] = row["grade"]

    settings = get_settings()
    retriever = retriever_from_settings(settings)
    rows: list[dict[str, object]] = []
    for q in queries:
        found: dict[str, dict[str, int]] = {}
        for mode in SEARCH_MODES:
            pmids = list(
                dict.fromkeys(
                    h.pmid for h in retriever.search(q["query"], k=args.depth * 2, mode=mode).hits
                )
            )[: args.depth]
            for rank, pmid in enumerate(pmids, start=1):
                found.setdefault(pmid, {})[mode] = rank
        records = retriever.store.get_records(found)
        for pmid, ranks in sorted(found.items(), key=lambda kv: min(kv[1].values())):
            rec = records[pmid]
            rows.append(
                {
                    "query_id": q["id"],
                    "query": q["query"],
                    "type": q.get("type", "topical"),
                    "pmid": pmid,
                    "title": rec["title"],
                    "abstract": rec["abstract"],
                    "year": rec["year"],
                    "found_by": ranks,
                    "grade": existing.get((q["id"], pmid)),
                }
            )
        print(f"{q['id']}: {len(found)} unique PMIDs pooled")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    graded = sum(1 for r in rows if r["grade"] is not None)
    print(
        f"wrote {len(rows)} rows for {len(queries)} queries to {args.out} ({graded} already graded)"
    )


if __name__ == "__main__":
    main()
