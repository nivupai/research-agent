"""How many labels each pending metric needs, and the retrieval "minimum useful sample".

Recall@10 needs every relevant document of a query to be known, so a sample that grades a
few pool items for each of the 50 queries cannot estimate it (the denominator is unknown).
The minimum useful sample is therefore *query-complete*: grade every pooled item for a
type-stratified subset of queries. Metrics are means over queries, so the confidence
interval shrinks with the number of queries, not the number of items.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bioagent.labeling.sheets import Row

Z95 = 1.96
DEFAULT_SAMPLE_PATH = Path("data/labels/pool_sample.json")

# Assumed spreads until real labels exist (state them wherever the plan is printed).
ASSUMED_RECALL_SD = 0.25  # per-query recall@10 across queries
ASSUMED_PAIRED_SD = 0.15  # per-query difference between two retrieval modes
ASSUMED_VERIFIER_ACC = 0.85
ASSUMED_JUDGE_AGREEMENT = 0.85
ASSUMED_JUDGE_CHANCE = 0.50


def choose_focus_queries(rows: Sequence[Row], n_queries: int, seed: int = 0) -> list[str]:
    """Pick ``n_queries`` query ids, proportionally across query types, at least one each."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        qid, qtype = str(row["query_id"]), str(row.get("type", ""))
        if qid not in by_type[qtype]:
            by_type[qtype].append(qid)
    total = sum(len(v) for v in by_type.values())
    n_queries = max(1, min(n_queries, total))
    types = sorted(by_type, key=lambda t: (-len(by_type[t]), t))
    alloc = {t: 1 for t in types[:n_queries]}
    while sum(alloc.values()) < n_queries:
        # largest remaining shortfall against the proportional share gets the next slot
        t = max(
            (t for t in types if alloc.get(t, 0) < len(by_type[t])),
            key=lambda t: n_queries * len(by_type[t]) / total - alloc.get(t, 0),
        )
        alloc[t] = alloc.get(t, 0) + 1
    rng = random.Random(seed)
    chosen: list[str] = []
    for t in types:
        chosen += rng.sample(sorted(by_type[t]), alloc.get(t, 0))
    return sorted(chosen)


def save_sample(path: Path, query_ids: Sequence[str], *, seed: int, rows: Sequence[Row]) -> None:
    items = sum(1 for r in rows if str(r["query_id"]) in set(query_ids))
    path.write_text(
        json.dumps(
            {
                "method": "query-complete, stratified by query type",
                "seed": seed,
                "query_ids": list(query_ids),
                "items": items,
            },
            indent=2,
        )
        + "\n"
    )


def load_sample(path: Path) -> frozenset[str]:
    return frozenset(json.loads(path.read_text())["query_ids"])


def queries_for_half_width(sd: float, half_width: float) -> int:
    return math.ceil((Z95 * sd / half_width) ** 2)


def proportion_half_width(p: float, n: int) -> float:
    return Z95 * math.sqrt(p * (1 - p) / n)


def kappa_half_width(agreement: float, chance: float, n: int) -> float:
    """Large-sample approximation of a 95% half-width for Cohen's kappa."""
    return Z95 * math.sqrt(agreement * (1 - agreement) / n) / (1 - chance)


def plan(pool: Sequence[Row], verifier: Sequence[Row], judge: Sequence[Row]) -> dict[str, Any]:
    per_query: dict[str, int] = defaultdict(int)
    for row in pool:
        per_query[str(row["query_id"])] += 1
    mean_pool = sum(per_query.values()) / max(1, len(per_query))
    retrieval = []
    for h in (0.15, 0.10, 0.07):
        nq = min(len(per_query), queries_for_half_width(ASSUMED_RECALL_SD, h))
        retrieval.append({"half_width": h, "queries": nq, "items": round(nq * mean_pool)})
    paired = []
    for h in (0.10, 0.07):
        nq = min(len(per_query), queries_for_half_width(ASSUMED_PAIRED_SD, h))
        paired.append({"half_width": h, "queries": nq, "items": round(nq * mean_pool)})
    ver = [
        {"n": n, "half_width": round(proportion_half_width(ASSUMED_VERIFIER_ACC, n), 3)}
        for n in sorted({50, 100, len(verifier)})
        if n <= len(verifier)
    ]
    jud = [
        {
            "n": n,
            "half_width": round(
                kappa_half_width(ASSUMED_JUDGE_AGREEMENT, ASSUMED_JUDGE_CHANCE, n), 3
            ),
        }
        for n in sorted({len(judge), 40, 60})
        if n > 0
    ]
    return {
        "pool": {
            "items": len(pool),
            "queries": len(per_query),
            "mean_items_per_query": round(mean_pool, 1),
        },
        "retrieval_recall_at_10": retrieval,
        "retrieval_mode_difference": paired,
        "verifier_accuracy": ver,
        "judge_kappa": jud,
        "assumptions": {
            "recall_sd": ASSUMED_RECALL_SD,
            "paired_sd": ASSUMED_PAIRED_SD,
            "verifier_accuracy": ASSUMED_VERIFIER_ACC,
            "judge_agreement": ASSUMED_JUDGE_AGREEMENT,
            "judge_chance_agreement": ASSUMED_JUDGE_CHANCE,
        },
    }


def plan_markdown(p: dict[str, Any]) -> str:
    pool = p["pool"]
    lines = [
        f"Retrieval pool: {pool['items']} items over {pool['queries']} queries "
        f"({pool['mean_items_per_query']} per query). Recall needs whole queries graded.",
        "",
        "| metric | labels | 95% CI half-width |",
        "|---|---|---|",
    ]
    for r in p["retrieval_recall_at_10"]:
        lines.append(
            f"| recall@10 / nDCG@10 (one mode) | {r['queries']} queries, ~{r['items']} items "
            f"| ±{r['half_width']:.2f} |"
        )
    for r in p["retrieval_mode_difference"]:
        lines.append(
            f"| recall@10 difference, hybrid vs BM25 (paired) | {r['queries']} queries, "
            f"~{r['items']} items | ±{r['half_width']:.2f} |"
        )
    for r in p["verifier_accuracy"]:
        lines.append(f"| verifier accuracy vs human | {r['n']} pairs | ±{r['half_width']:.3f} |")
    for r in p["judge_kappa"]:
        lines.append(f"| judge vs human kappa | {r['n']} answers | ±{r['half_width']:.2f} |")
    a = p["assumptions"]
    lines += [
        "",
        f"Assumed: per-query recall SD {a['recall_sd']}, paired-difference SD "
        f"{a['paired_sd']}, verifier accuracy {a['verifier_accuracy']}, judge agreement "
        f"{a['judge_agreement']} with chance agreement {a['judge_chance_agreement']}. "
        "Recompute once real labels exist.",
    ]
    return "\n".join(lines)
