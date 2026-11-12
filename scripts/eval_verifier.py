"""Compare verifiers on the held-out test split; tune the router threshold on val.

Systems: the fine-tuned small model (models/verifier-lora), DeBERTa-v3-large NLI, and the
Gemini large model used as a verifier at inference (own cache namespace, so it does not reuse
the teacher's labels). References: the teacher labels, and the human labels in
eval/labels/verifier_test_human.jsonl once filled in. Reports accuracy and macro-F1 with
bootstrap CIs, confusion matrices, p50/p95 latency per claim and cost per 1k claims, plus the
router's escalation/cost/accuracy curve on val. Writes results/verifier/eval.{json,md}.

Usage: uv run python scripts/eval_verifier.py [--skip-llm] [--reference teacher|nli]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from bioagent.config import get_settings
from bioagent.finetune.data import Pair, read_pairs
from bioagent.finetune.metrics import accuracy, percentile, summary
from bioagent.verify.backends import make_llm_scorer, route

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "finetune"
OUT = ROOT / "results" / "verifier"
HUMAN = ROOT / "eval" / "labels" / "verifier_test_human.jsonl"
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
LABEL_ORDER = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")


def reference(p: Pair, source: str) -> str | None:
    if source == "teacher":
        return p.labels.get("teacher")
    return (p.labels.get("nli") or {}).get("label")


def small_predictions(pairs: list[Pair]) -> dict[str, Any]:
    from bioagent.finetune.model import SmallVerifier

    model = SmallVerifier(ROOT / "models" / "verifier-lora")
    model.predict_proba([(pairs[0].passage, pairs[0].claim)])  # load + warm up
    probs, secs = model.timed_predict([(p.passage, p.claim) for p in pairs])
    preds = [LABEL_ORDER[max(range(3), key=lambda i: pr[i])] for pr in probs]
    return {"pred": preds, "probs": probs, "seconds": secs}


def llm_predictions(pairs: list[Pair]) -> dict[str, Any]:
    scorer = make_llm_scorer(get_settings())
    t0 = time.monotonic()
    labels = scorer.labels([(p.passage, p.claim) for p in pairs])
    wall = time.monotonic() - t0
    calls = scorer.calls[-1] if scorer.calls else {}
    n_called = max(1, calls.get("requested", 0))
    per_pair = calls.get("latency_s", 0.0) / n_called if calls.get("calls") else None
    return {
        "pred": labels,
        "stopped": calls.get("stopped"),
        "new_calls": calls.get("calls", 0),
        "seconds_per_pair": per_pair,
        "cost_per_pair": (calls.get("cost_usd", 0.0) / n_called) if calls.get("calls") else None,
        "wall_seconds": wall,
        "model": str(scorer.spec),
    }


def system_row(
    name: str,
    gold: list[str],
    pred: list[str],
    secs: list[float] | None,
    cost_per_1k: float | None,
    note: str = "",
) -> dict[str, Any]:
    row = {
        "system": name,
        **summary(gold, pred),
        "note": note,
        "cost_per_1k_claims_usd": cost_per_1k,
    }
    if secs:
        row["latency_p50_s"] = percentile(secs, 0.5)
        row["latency_p95_s"] = percentile(secs, 0.95)
    return row


def router_curve(
    val: list[Pair], small: dict[str, Any], llm: dict[str, Any] | None, source: str
) -> list[dict[str, Any]]:
    gold = [reference(p, source) or "" for p in val]
    rows = []
    for t in THRESHOLDS:
        esc = route(small["probs"], t)
        if llm is None:
            break
        pred = [llm["pred"][i] if e else small["pred"][i] for i, e in enumerate(esc)]
        rate = sum(esc) / len(esc)
        rows.append(
            {
                "threshold": t,
                "escalation_rate": rate,
                "accuracy": accuracy(gold, pred),
                "cost_per_1k_claims_usd": rate * 1000 * (llm["cost_per_pair"] or 0.0),
            }
        )
    return rows


def pick_threshold(curve: list[dict[str, Any]], tolerance: float = 0.01) -> dict[str, Any] | None:
    """Cheapest threshold whose val accuracy is within ``tolerance`` of the best."""
    if not curve:
        return None
    best = max(r["accuracy"] for r in curve)
    ok = [r for r in curve if r["accuracy"] >= best - tolerance]
    return min(ok, key=lambda r: r["escalation_rate"])


def markdown(report: dict[str, Any]) -> str:
    def ci(m: dict[str, Any]) -> str:
        if m["mean"] is None:
            return "n/a"
        return f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"

    lines = [
        f"# Verifier comparison (test split, reference: {report['reference']})",
        "",
        "| system | n | accuracy [95% CI] | macro-F1 [95% CI] | p50 / p95 latency per claim | "
        "cost per 1k claims |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["systems"]:
        lat = (
            ""
            if "latency_p50_s" not in row
            else f"{row['latency_p50_s'] * 1000:.0f} / {row['latency_p95_s'] * 1000:.0f} ms"
        )
        cost = row["cost_per_1k_claims_usd"]
        cost_txt = "$0 (local)" if cost == 0 else ("n/a" if cost is None else f"${cost:.3f}")
        lines.append(
            f"| {row['system']} {row['note']} | {row['n']} | {ci(row['accuracy'])} | "
            f"{ci(row['macro_f1'])} | {lat} | {cost_txt} |"
        )
    if report.get("router"):
        lines += [
            "",
            "## Router (small model, escalate below threshold) on val",
            "",
            "| threshold | escalated | accuracy | cost per 1k claims |",
            "|---|---|---|---|",
        ]
        for r in report["router"]:
            lines.append(
                f"| {r['threshold']} | {r['escalation_rate']:.0%} | {r['accuracy']:.3f} "
                f"| ${r['cost_per_1k_claims_usd']:.3f} |"
            )
        if report.get("router_choice"):
            c = report["router_choice"]
            lines.append(
                f"\nChosen threshold: {c['threshold']} (cheapest within 1 point of the "
                f"best val accuracy)."
            )
    for name, row in report.get("human", {}).items():
        lines.append(f"\nvs human labels: {name} accuracy {ci(row['accuracy'])} (n={row['n']})")
    lines.append("\nConfusion matrices (rows = reference, columns = prediction) are in eval.json.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--reference", choices=["teacher", "nli"], default="teacher")
    args = parser.parse_args()
    test = [p for p in read_pairs(DATA / "test.jsonl") if reference(p, args.reference)]
    val = [p for p in read_pairs(DATA / "val.jsonl") if reference(p, args.reference)]
    gold = [reference(p, args.reference) or "" for p in test]
    small = small_predictions(test)
    nli = [(p.labels.get("nli") or {}) for p in test]
    systems = [
        system_row("fine-tuned small (LoRA)", gold, small["pred"], small["seconds"], 0.0),
        system_row(
            "DeBERTa-v3-large NLI",
            gold,
            [n.get("label", "NOT_ENOUGH_INFO") for n in nli],
            [n.get("seconds", 0.0) for n in nli],
            0.0,
        ),
    ]
    llm_test = None if args.skip_llm else llm_predictions(test)
    if llm_test is not None:
        per_pair = llm_test["seconds_per_pair"]
        cost = llm_test["cost_per_pair"]
        note = "(pending: quota)" if llm_test["stopped"] else ""
        systems.append(
            system_row(
                f"Gemini {llm_test['model']} (large model)",
                gold,
                llm_test["pred"],
                [per_pair] * len(test) if per_pair else None,
                cost * 1000 if cost is not None else None,
                note,
            )
        )
    small_val = small_predictions(val)
    llm_val = None if args.skip_llm else llm_predictions(val)
    curve = router_curve(val, small_val, llm_val, args.reference)
    report: dict[str, Any] = {
        "reference": args.reference,
        "systems": systems,
        "router": curve,
        "router_choice": pick_threshold(curve),
        "human": {},
    }
    if HUMAN.exists():
        human = {
            r["id"]: r["human_label"]
            for r in map(json.loads, HUMAN.read_text().splitlines())
            if r.get("human_label")
        }
        idx = [i for i, p in enumerate(test) if p.id in human]
        if idx:
            hg = [human[test[i].id] for i in idx]
            report["human"]["teacher"] = summary(hg, [gold[i] for i in idx])
            report["human"]["small"] = summary(hg, [small["pred"][i] for i in idx])
            report["human"]["nli"] = summary(hg, [nli[i].get("label", "") for i in idx])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval.json").write_text(json.dumps(report, indent=1) + "\n")
    (OUT / "eval.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
