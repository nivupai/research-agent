"""Offline replay of GRAPH_CLAIM_CHECK on saved runs (no LLM calls; needs Neo4j).

Usage: uv run python scripts/replay_graph_check.py results/verify_policy/runs.tsv
       [--out results/graph_check/replay.json]

For each saved run (rows: target, condition, repeat, exit code, run_id) the saved claims of
its draft are relabelled with the graph check, and the remove policy is applied with and
without it (the NLI labels are the saved ones, so "off" reproduces the remove policy). The
simulated answers are scored against the approved gold lists like score_drug_answers.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from bioagent.config import get_settings
from bioagent.eval.stats import paired_bootstrap, permutation_test
from bioagent.verify.graph_check import GraphChecker, GraphDrug, lookup_from_settings
from bioagent.verify.text import answer_sentences
from bioagent.verify.verifier import (
    Claim,
    ClaimVerifier,
    VerifyConfig,
    apply_graph_check,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_drug_answers import drug_vocabulary, load_gold, named, score_run  # noqa: E402


def claims_from(rows: list[dict[str, Any]]) -> list[Claim]:
    out = []
    for r in rows:
        kind = "absence" if r["label"] == "absence" else "factual"
        c = Claim(r["n"], r["sentence"], r["claim"], list(r.get("citations") or []), kind)
        c.label, c.check, c.evidence = r["label"], r.get("check", ""), list(r.get("evidence") or [])
        out.append(c)
    return out


def simulate(record: dict[str, Any], checker: GraphChecker | None) -> dict[str, Any]:
    draft = record.get("draft_answer") or record["answer"]
    claims = claims_from(record["verification"].get("claims") or [])
    sentences = answer_sentences(draft)
    keep = apply_graph_check(claims, sentences, checker)
    verifier = ClaimVerifier(nli=None, config=VerifyConfig(policy="remove"))  # type: ignore[arg-type]
    answer, _, removed, flagged = verifier.apply_policy(
        draft, claims, record.get("citations", []), keep=keep
    )
    return {
        "answer": answer,
        "draft_answer": draft,
        "claims": claims,
        "graph_sentences": keep,
        "removed": removed,
        "flagged": flagged,
        "n_sentences": len(sentences),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results/graph_check/replay.json")
    args = parser.parse_args()
    gold = load_gold()
    names = {d["name"] for g in gold.values() for d in g["direct"] + g["indirect"]}
    syn, approved = await drug_vocabulary(names)
    lookup = lookup_from_settings(get_settings())
    graph_drugs: dict[str, list[GraphDrug]] = {}
    for t in gold:
        graph_drugs[t] = await lookup.drugs_for_target(t)
    rows = [line.split("\t") for line in args.runs.read_text().splitlines() if line.strip()]
    runs: dict[str, Any] = {}
    accepted = {"gold": 0, "non_gold": 0, "other": 0}
    accepted_names: dict[str, dict[str, int]] = {}
    for target, cond, rep, rc, run_id in rows:
        path = ROOT / "results" / "runs" / f"{run_id.strip()}.json"
        if rc != "0" or not path.exists():
            continue
        record = json.loads(path.read_text())
        checker = GraphChecker(graph_drugs[target], target)
        g = gold[target]
        gold_names = [d["name"] for d in g["direct"] + g["indirect"]]
        gold_terms = {t.lower() for n in gold_names for t in syn.get(n, [n])}
        out: dict[str, Any] = {}
        for label, ch in (("off", None), ("on", checker)):
            sim = simulate(record, ch)
            scored = score_run({**record, "answer": sim["answer"]}, g, syn, approved)
            scored["graph_frac"] = len(sim["graph_sentences"]) / max(1, sim["n_sentences"])
            scored["removed_frac"] = len(sim["removed"]) / max(1, sim["n_sentences"])
            out[label] = scored
            if label == "on":
                for c in sim["claims"]:
                    if c.label != "graph":
                        continue
                    for d in checker.drugs_named(c.text):
                        is_gold = any(t.lower() in gold_terms for t in d.terms) or any(
                            named(d.name, syn.get(n, [n])) for n in gold_names
                        )
                        is_approved = any(named(d.name, [a]) for a in approved)
                        kind = "gold" if is_gold else ("non_gold" if is_approved else "other")
                        accepted[kind] += 1
                        accepted_names.setdefault(kind, {})
                        accepted_names[kind][f"{target}:{d.name}"] = (
                            accepted_names[kind].get(f"{target}:{d.name}", 0) + 1
                        )
        runs[f"{target}|{cond}|{rep}"] = out
    # graph coverage of the gold lists
    coverage: dict[str, Any] = {}
    for t, g in gold.items():
        checker = GraphChecker(graph_drugs[t], t)
        missing = [
            d["name"]
            for d in g["direct"] + g["indirect"]
            if not any(checker.drugs_named(term) for term in syn.get(d["name"], [d["name"]]))
        ]
        coverage[t] = {
            "graph_drugs": len(graph_drugs[t]),
            "gold": len(g["direct"]) + len(g["indirect"]),
            "gold_missing_from_graph": missing,
        }
    summary: dict[str, Any] = {}
    for m in ["recall", "non_gold", "precision", "graph_frac", "removed_frac"]:
        for label in ("off", "on"):
            vals = [r[label][m] for r in runs.values() if r[label][m] is not None]
            summary[f"{label}: {m}"] = round(statistics.fmean(vals), 3)
        diffs = [
            r["on"][m] - r["off"][m]
            for r in runs.values()
            if r["on"][m] is not None and r["off"][m] is not None
        ]
        lo, hi = paired_bootstrap(diffs)
        summary[f"on - off: {m}"] = {
            "mean": round(statistics.fmean(diffs), 4),
            "lo": round(lo, 4),
            "hi": round(hi, 4),
            "p": round(permutation_test(diffs), 4),
            "n": len(diffs),
        }
    draft = [
        json.loads((ROOT / "results/runs" / f"{r[4].strip()}.json").read_text())
        for r in rows
        if r[3] == "0"
    ]
    summary["runs"] = len(runs)
    summary["draft_runs"] = len(draft)
    result = {
        "summary": summary,
        "accepted_mentions": accepted,
        "accepted_names": accepted_names,
        "coverage": coverage,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str))
    print(
        json.dumps(
            {k: result[k] for k in ("summary", "accepted_mentions", "accepted_names", "coverage")},
            indent=1,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
