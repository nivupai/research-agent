"""Score "which approved drugs act on X" answers against the approved gold lists.

Usage: uv run python scripts/score_drug_answers.py results/evidence_screen/runs.tsv
       [--condition-col 2] [--out results/evidence_screen/scores.json]

runs.tsv rows: target, condition, repeat, exit code, run_id (written by the comparison driver).
Per run: gold recall in the final answer and in the pre-verification draft (the verifier may
remove sentences) (direct + indirect drugs named; names and graph synonyms,
word-boundary, case-insensitive), non-gold approved drugs named (ChEMBL max_phase 4 drug
names from the graph, a precision proxy), precision (gold / all approved drugs named), drugs
named only in [unverified] sentences, share of sentences removed or flagged, groundedness,
references, latency. Then per-target
means and paired differences between the two conditions over (target, repeat) with a
bootstrap 95% CI and a sign-flip permutation p. Needs Neo4j (synonyms); no LLM calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from bioagent.eval.stats import paired_bootstrap, permutation_test

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "questions" / "drug_target_gold.jsonl"
MIN_NAME = 4  # ignore very short names/synonyms (e.g. "Met", "CA")
EXTRA_ALIASES = {
    "sirolimus": ["rapamycin"],
    "aspirin": ["acetylsalicylic acid"],
    "valproic acid": ["valproate"],
}


def load_gold() -> dict[str, dict[str, Any]]:
    gold = {}
    for line in GOLD.read_text().splitlines():
        row = json.loads(line)
        if "_meta" in row:
            if row["_meta"].get("status") != "approved_by_user":
                raise SystemExit("gold lists are not approved")
            continue
        gold[row["target"]] = row
    return gold


async def drug_vocabulary(names: set[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Synonyms for the given names, and all approved drug names in the graph."""
    from bioagent.config import get_settings
    from bioagent.kg.store import GraphStore

    store = GraphStore.from_settings(get_settings())
    syn: dict[str, list[str]] = {}
    for name in names:
        rows = await store.read(
            "MATCH (d:Drug) WHERE toLower(d.name) = toLower($n) OR toLower(d.chembl_name) = "
            "toLower($n) RETURN d.name AS n, d.chembl_name AS c, d.synonyms AS s LIMIT 3",
            {"n": name},
            max_records=3,
        )
        terms = {name}
        for r in rows:
            terms.update(x for x in [r["n"], r["c"]] if x)
            if isinstance(r["s"], list):
                terms.update(r["s"])
        terms.update(EXTRA_ALIASES.get(name.lower(), []))
        syn[name] = sorted({t for t in terms if len(t) >= MIN_NAME})
    approved = await store.read(
        "MATCH (d:Drug) WHERE coalesce(d.max_phase, 0) >= 4 RETURN d.name AS n",
        {},
        max_records=5000,
    )
    return syn, sorted({r["n"] for r in approved if r["n"] and len(r["n"]) >= MIN_NAME})


def named(text: str, terms: list[str]) -> bool:
    return any(re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", text, re.I) for t in terms)


def score_run(
    record: dict[str, Any], gold: dict[str, Any], syn: dict[str, list[str]], approved: list[str]
) -> dict[str, Any]:
    answer = record.get("answer", "")
    draft = record.get("draft_answer") or answer
    gold_names = [d["name"] for d in gold["direct"] + gold["indirect"]]
    hits = [g for g in gold_names if named(answer, syn.get(g, [g]))]
    draft_hits = [g for g in gold_names if named(draft, syn.get(g, [g]))]
    gold_terms = {t.lower() for g in gold_names for t in syn.get(g, [g])}
    others = [n for n in approved if n.lower() not in gold_terms and named(answer, [n])]
    draft_others = [n for n in approved if n.lower() not in gold_terms and named(draft, [n])]
    # drugs named only inside sentences the verifier flagged ([unverified]) in the final answer
    from bioagent.verify.text import split_sentences

    flagged_text = " ".join(x for x in split_sentences(answer) if "[unverified]" in x)
    clean_text = " ".join(x for x in split_sentences(answer) if "[unverified]" not in x)
    flagged_gold = [
        g
        for g in hits
        if named(flagged_text, syn.get(g, [g])) and not named(clean_text, syn.get(g, [g]))
    ]
    flagged_other = [n for n in others if named(flagged_text, [n]) and not named(clean_text, [n])]
    verification = record.get("verification") or {}
    checked = verification.get("sentences_checked") or 0
    stats = record.get("stats", {})
    out = {
        "gold_named": hits,
        "recall": len(hits) / len(gold_names) if gold_names else None,
        "non_gold_named": others,
        "non_gold": len(others),
        "draft_gold_named": draft_hits,
        "draft_recall": len(draft_hits) / len(gold_names) if gold_names else None,
        "draft_non_gold": len(draft_others),
        "precision": len(hits) / (len(hits) + len(others)) if hits or others else None,
        "flagged_gold": len(flagged_gold),
        "flagged_non_gold": len(flagged_other),
        "removed_frac": len(verification.get("removed_sentences") or []) / checked
        if checked
        else 0.0,
        "flagged_frac": len(verification.get("flagged_sentences") or []) / checked
        if checked
        else 0.0,
        "graph_frac": len(verification.get("graph_sentences") or []) / checked if checked else 0.0,
        "groundedness": (record.get("verification") or {}).get("groundedness"),
        "references": len(record.get("citations", [])),
        "seconds": stats.get("wall_seconds"),
        "llm_calls": stats.get("llm_calls"),
    }
    if gold["target"] == "AMPK":
        rest = [g for g in gold_names if g != "Sorafenib"]
        out["recall_without_sorafenib"] = len([g for g in hits if g != "Sorafenib"]) / len(rest)
        out["sorafenib_named"] = "Sorafenib" in hits
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results/evidence_screen/scores.json")
    args = parser.parse_args()
    gold = load_gold()
    rows = [line.split("\t") for line in args.runs.read_text().splitlines() if line.strip()]
    names = {d["name"] for g in gold.values() for d in g["direct"] + g["indirect"]}
    syn, approved = asyncio.run(drug_vocabulary(names))
    scores: dict[tuple[str, str, str], dict[str, Any]] = {}
    for target, cond, rep, rc, run_id in rows:
        path = ROOT / "results" / "runs" / f"{run_id.strip()}.json"
        if rc != "0" or not path.exists():
            continue
        scores[(target, cond, rep)] = score_run(
            json.loads(path.read_text()), gold[target], syn, approved
        )
    conds = sorted({c for _, c, _ in scores})
    metrics = [
        "recall",
        "draft_recall",
        "non_gold",
        "draft_non_gold",
        "precision",
        "flagged_gold",
        "flagged_non_gold",
        "removed_frac",
        "flagged_frac",
        "graph_frac",
        "groundedness",
        "references",
        "seconds",
    ]
    per_target: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for target in gold:
        for c in conds:
            rs = [v for (t, cc, _), v in scores.items() if t == target and cc == c]
            if rs:
                per_target[target][c] = {
                    m: round(statistics.fmean(vals), 3)
                    if (vals := [r[m] for r in rs if r[m] is not None])
                    else None
                    for m in metrics
                } | {"n": len(rs)}
    paired: dict[str, Any] = {}
    if len(conds) == 2:
        a, b = conds  # alphabetical: "off" < "sentence"
        keys = [(t, r) for (t, c, r) in scores if c == a and (t, b, r) in scores]
        for m in [*metrics, "recall_without_sorafenib"]:
            diffs = [
                scores[(t, b, r)][m] - scores[(t, a, r)][m]
                for t, r in keys
                if (m != "recall_without_sorafenib" or t == "AMPK")
                and scores[(t, b, r)].get(m) is not None
                and scores[(t, a, r)].get(m) is not None
            ]
            if diffs:
                lo, hi = paired_bootstrap(diffs)
                paired[f"{b} - {a}: {m}"] = {
                    "mean": round(statistics.fmean(diffs), 4),
                    "lo": round(lo, 4),
                    "hi": round(hi, 4),
                    "p": round(permutation_test(diffs), 4),
                    "n": len(diffs),
                }
    result = {
        "runs": {"|".join(k): v for k, v in scores.items()},
        "per_target": per_target,
        "paired": paired,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"per_target": per_target, "paired": paired}, indent=1))


if __name__ == "__main__":
    main()
