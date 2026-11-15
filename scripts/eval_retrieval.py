"""Evaluate every retrieval mode against graded qrels and save the results.

Usage: uv run python scripts/eval_retrieval.py [--labels human|claude] [-k 10]

--labels human (default) reads the hand grades in data/labels/pool.jsonl; --labels claude reads
eval/labels/retrieval_claude.jsonl (relevance graded by Claude, labeler="claude"; results are
marked as such). --qrels overrides either. The qrels file is JSONL with query_id, query, pmid
and grade (0/1/2); lines with grade null are ignored. Reports each metric with a bootstrap 95%
CI over queries and paired comparisons (CI and sign-flip p) of every mode against --baseline.
Writes a markdown table and JSON to results/retrieval/.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from bioagent.config import get_settings
from bioagent.eval.retrieval import (
    compare_modes,
    evaluate,
    load_qrels,
    markdown_table,
    query_ci,
    recall_ceiling,
)
from bioagent.retrieval.models import SEARCH_MODES
from bioagent.retrieval.retriever import retriever_from_settings

LABEL_FILES = {
    "human": Path("data/labels/pool.jsonl"),
    "claude": Path("eval/labels/retrieval_claude.jsonl"),
}
LABELER = {"human": "human relevance grades", "claude": "Claude-graded relevance (not human)"}
METRICS = ("recall@{k}", "ndcg@{k}", "mrr", "precision@{k}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", choices=sorted(LABEL_FILES), default="human")
    parser.add_argument("--qrels", type=Path, help="override the --labels file")
    parser.add_argument("--baseline", default="dense", choices=SEARCH_MODES)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--modes", nargs="+", default=list(SEARCH_MODES), choices=SEARCH_MODES)
    parser.add_argument("--out", type=Path, default=Path("results/retrieval"))
    args = parser.parse_args()

    qrels_path = args.qrels or LABEL_FILES[args.labels]
    qrels = load_qrels(qrels_path)
    if not qrels.queries:
        raise SystemExit(f"no graded judgements in {qrels_path}; fill in the 'grade' fields")

    settings = get_settings()
    retriever = retriever_from_settings(settings)
    for mode in args.modes:  # load models before timing
        retriever.search("warm up", k=1, mode=mode)

    def search(query: str, mode: str, depth: int) -> list[str]:
        hits = retriever.search(query, k=depth, mode=mode).hits  # type: ignore[arg-type]
        return [h.pmid for h in hits]

    results = evaluate(search, qrels, args.modes, k=args.k)
    metrics = [m.format(k=args.k) for m in METRICS]
    cis = {mode: {m: query_ci(r["per_query"], m) for m in metrics} for mode, r in results.items()}
    paired = {
        mode: {m: compare_modes(results, args.baseline, mode, m) for m in metrics}
        for mode in results
        if mode != args.baseline and args.baseline in results
    }
    ceiling = round(recall_ceiling(qrels, args.k), 4)
    table = markdown_table(results, k=args.k)
    lines = [
        f"Labels: {LABELER[args.labels] if not args.qrels else qrels_path} "
        f"({len(qrels.queries)} queries). Best achievable mean recall@{args.k}: {ceiling}.",
        "",
        "| mode | " + " | ".join(metrics) + " |",
        "|---" * (len(metrics) + 1) + "|",
    ]
    for mode, row in cis.items():
        cells = [f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]" for c in row.values()]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines += ["", f"Paired vs {args.baseline} (delta [95% CI], sign-flip p):", ""]
    lines += ["| mode | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for mode, row in paired.items():
        cells = [
            f"{c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}], p={c['p']}" for c in row.values()
        ]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    table = table + "\n\n" + "\n".join(lines)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "qrels": str(qrels_path),
        "labels": LABELER[args.labels] if not args.qrels else str(qrels_path),
        "queries": len(qrels.queries),
        "recall_ceiling": ceiling,
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "k": args.k,
        "timestamp": stamp,
    }
    (args.out / f"retrieval_{stamp}.json").write_text(
        json.dumps({"meta": meta, "results": results, "ci": cis, "paired": paired}, indent=2)
    )
    (args.out / f"retrieval_{stamp}.md").write_text(
        f"# Retrieval eval {stamp}\n\n{json.dumps(meta)}\n\n{table}\n"
    )
    print(table)
    print(f"\nsaved to {args.out}/retrieval_{stamp}.{{json,md}}")


if __name__ == "__main__":
    main()
