"""Retrieval evaluation: graded qrels, recall/precision@k, MRR, nDCG@k, latency.

Relevance is judged per PMID (passages are mapped to their PMID, keeping the first
occurrence), with grades 0 (not relevant), 1 (partially) and 2 (highly relevant).
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RELEVANT_MIN_GRADE = 1


@dataclass
class Qrels:
    queries: dict[str, str] = field(default_factory=dict)
    grades: dict[str, dict[str, int]] = field(default_factory=dict)

    def judged(self, query_id: str) -> dict[str, int]:
        return self.grades.get(query_id, {})


def load_qrels(path: str | Path) -> Qrels:
    """Read JSONL lines ``{query_id, query, pmid, grade}``; ungraded lines (grade null) skip."""
    qrels = Qrels()
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        grade = row.get("grade")
        if grade is None:
            continue
        grade = int(grade)
        if grade not in (0, 1, 2):
            raise ValueError(f"{path}:{lineno}: grade must be 0, 1 or 2")
        qid = str(row["query_id"])
        qrels.queries.setdefault(qid, str(row["query"]))
        qrels.grades.setdefault(qid, {})[str(row["pmid"])] = grade
    return qrels


def dedupe(ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def recall_at_k(ranked: Sequence[str], grades: dict[str, int], k: int) -> float:
    relevant = {d for d, g in grades.items() if g >= RELEVANT_MIN_GRADE}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(ranked: Sequence[str], grades: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if grades.get(d, 0) >= RELEVANT_MIN_GRADE)
    return hits / k


def reciprocal_rank(ranked: Sequence[str], grades: dict[str, int]) -> float:
    for rank, doc in enumerate(ranked, start=1):
        if grades.get(doc, 0) >= RELEVANT_MIN_GRADE:
            return 1.0 / rank
    return 0.0


def dcg(gains: Sequence[int]) -> float:
    return float(sum((2.0**g - 1.0) / math.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(ranked: Sequence[str], grades: dict[str, int], k: int) -> float:
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([grades.get(d, 0) for d in ranked[:k]]) / ideal


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


SearchFn = Callable[[str, str, int], list[str]]  # (query, mode, k) -> ranked PMIDs


def evaluate(
    search: SearchFn,
    qrels: Qrels,
    modes: Sequence[str],
    *,
    k: int = 10,
    depth: int = 100,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        per_query: dict[str, dict[str, float]] = {}
        latencies: list[float] = []
        for qid, query in qrels.queries.items():
            grades = qrels.judged(qid)
            t0 = time.perf_counter()
            ranked = dedupe(search(query, mode, depth))
            latencies.append((time.perf_counter() - t0) * 1000)
            per_query[qid] = {
                f"recall@{k}": recall_at_k(ranked, grades, k),
                f"precision@{k}": precision_at_k(ranked, grades, k),
                "mrr": reciprocal_rank(ranked, grades),
                f"ndcg@{k}": ndcg_at_k(ranked, grades, k),
            }
        metrics = {
            name: round(statistics.fmean(q[name] for q in per_query.values()), 4)
            if per_query
            else 0.0
            for name in (f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}")
        }
        metrics["latency_p50_ms"] = round(percentile(latencies, 50), 1)
        metrics["latency_p95_ms"] = round(percentile(latencies, 95), 1)
        metrics["queries"] = len(per_query)
        results[mode] = {"metrics": metrics, "per_query": per_query}
    return results


def markdown_table(results: dict[str, dict[str, Any]], k: int = 10) -> str:
    cols = [f"recall@{k}", f"ndcg@{k}", "mrr", f"precision@{k}", "latency_p50_ms", "latency_p95_ms"]
    lines = ["| mode | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for mode, data in results.items():
        m = data["metrics"]
        lines.append(f"| {mode} | " + " | ".join(str(m[c]) for c in cols) + " |")
    return "\n".join(lines)


def recall_ceiling(qrels: Qrels, k: int) -> float:
    """Best achievable mean recall@k: queries with more than k relevant PMIDs cap below 1."""
    caps = []
    for qid in qrels.queries:
        n_rel = sum(1 for g in qrels.judged(qid).values() if g >= RELEVANT_MIN_GRADE)
        if n_rel:
            caps.append(min(k, n_rel) / n_rel)
    return statistics.fmean(caps) if caps else 0.0


def query_ci(per_query: dict[str, dict[str, float]], metric: str) -> dict[str, float]:
    """Mean of a per-query metric with a bootstrap 95% CI over queries."""
    from bioagent.eval.stats import paired_bootstrap

    values = [q[metric] for q in per_query.values()]
    lo, hi = paired_bootstrap(values)
    return {"mean": round(statistics.fmean(values), 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def compare_modes(
    results: dict[str, dict[str, Any]], baseline: str, candidate: str, metric: str
) -> dict[str, float]:
    """Paired per-query difference (candidate - baseline): bootstrap CI and sign-flip p."""
    from bioagent.eval.stats import paired_bootstrap, permutation_test

    base, cand = results[baseline]["per_query"], results[candidate]["per_query"]
    diffs = [cand[q][metric] - base[q][metric] for q in base if q in cand]
    lo, hi = paired_bootstrap(diffs)
    return {
        "delta": round(statistics.fmean(diffs), 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "p": round(permutation_test(diffs), 4),
        "queries": len(diffs),
    }
