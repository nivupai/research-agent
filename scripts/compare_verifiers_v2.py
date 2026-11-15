"""Compare verifiers, including the LoRA retrained on reference v2, on the held-out test split.

Usage: uv run python scripts/compare_verifiers_v2.py

Needs results/verifier/small_probs.json (v1 adapter), small_probs_v2.json (v2 adapter, timed)
and small_probs_v1_timed.json (v1 latency), from scripts/verifier_probs.py. References: v1
(teacher), v2 (rule-perturbed pairs CONTRADICTED by construction) and v2 without swap. Gemini
verifier labels come from the label cache only (never the API). Reports accuracy and macro-F1
with bootstrap CIs, per-class F1, p50/p95 latency, cost per 1k claims, paired tests, the 18
test edited contradictions the teacher called SUPPORTED, and the router curve on val.
Writes results/verifier/compare_v2.json and prints markdown.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bioagent.eval.stats import permutation_test
from bioagent.finetune.data import CONSTRUCTED_KINDS, Pair, read_pairs, relabel_by_construction
from bioagent.finetune.metrics import accuracy, bootstrap_ci, macro_f1, per_class_f1, percentile

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "verifier"
LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")
THRESHOLDS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


def argmax(p: Sequence[float]) -> str:
    return LABELS[max(range(3), key=lambda i: p[i])]


def load(split: str) -> list[Pair]:
    pairs = read_pairs(ROOT / "data" / "finetune" / f"{split}.jsonl")
    return [p for p in pairs if p.labels.get("teacher") in LABELS]


def references(pairs: list[Pair]) -> dict[str, list[str]]:
    no_swap = [k for k in CONSTRUCTED_KINDS if k != "swap"]
    return {
        "v2": [str(relabel_by_construction(p).labels["target"]) for p in pairs],
        "v2 without swap": [
            str(relabel_by_construction(p, no_swap).labels["target"]) for p in pairs
        ],
        "v1 teacher": [str(p.labels["teacher"]) for p in pairs],
    }


def paired(
    gold: Sequence[str],
    base: Sequence[str],
    cand: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    b: int = 2000,
    seed: int = 7,
) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(gold)
    deltas = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        deltas.append(metric(g, [cand[i] for i in idx]) - metric(g, [base[i] for i in idx]))
    deltas.sort()
    per_item = [float(c == x) - float(a == x) for x, a, c in zip(gold, base, cand, strict=True)]
    return {
        "delta": round(metric(gold, cand) - metric(gold, base), 4),
        "lo": round(deltas[int(0.025 * (b - 1))], 4),
        "hi": round(deltas[int(0.975 * (b - 1))], 4),
        "p": round(permutation_test(per_item), 4),
    }


def main() -> None:
    from preview_reference_v2 import cached_llm_labels  # same cache lookup, no API calls

    v1 = json.loads((RES / "small_probs.json").read_text())["splits"]
    v2_saved = json.loads((RES / "small_probs_v2.json").read_text())
    v2 = v2_saved["splits"]
    v1_secs = json.loads((RES / "small_probs_v1_timed.json").read_text())["seconds"]["test"]
    v2_secs = v2_saved["seconds"]["test"]
    stats = json.loads((RES / "llm_stats.json").read_text())
    llm_cost_1k = stats["cost_per_pair"] * 1000

    test = load("test")
    refs = references(test)
    llm = cached_llm_labels(test)
    if len(llm) != len(test):
        raise SystemExit(f"Gemini verifier labels cached for {len(llm)} of {len(test)} pairs")

    def router(probs: dict[str, list[float]], t: float, pairs: list[Pair], llm_: dict[str, str]):
        return [llm_[p.id] if max(probs[p.id]) < t else argmax(probs[p.id]) for p in pairs]

    def escalated(probs: dict[str, list[float]], t: float, pairs: list[Pair]) -> float:
        return sum(max(probs[p.id]) < t for p in pairs) / len(pairs)

    nli_secs = [float((p.labels.get("nli") or {}).get("seconds") or 0) for p in test]
    systems: dict[str, dict[str, Any]] = {
        "DeBERTa-v3-large NLI": {
            "pred": [str((p.labels.get("nli") or {}).get("label")) for p in test],
            "secs": nli_secs,
            "cost": 0.0,
        },
        "small LoRA v1 (teacher labels)": {
            "pred": [argmax(v1["test"][p.id]) for p in test],
            "secs": v1_secs,
            "cost": 0.0,
        },
        "small LoRA v2 (reference v2)": {
            "pred": [argmax(v2["test"][p.id]) for p in test],
            "secs": v2_secs,
            "cost": 0.0,
        },
        "Gemini verifier (flash-lite)": {
            "pred": [llm[p.id] for p in test],
            "secs": None,
            "cost": llm_cost_1k,
        },
        "router v1 @ 0.75": {
            "pred": router(v1["test"], 0.75, test, llm),
            "secs": None,
            "cost": escalated(v1["test"], 0.75, test) * llm_cost_1k,
        },
        "router v2 @ 0.75": {
            "pred": router(v2["test"], 0.75, test, llm),
            "secs": None,
            "cost": escalated(v2["test"], 0.75, test) * llm_cost_1k,
        },
    }
    out: dict[str, Any] = {"n_test": len(test), "systems": {}, "paired": {}}
    for name, s in systems.items():
        row: dict[str, Any] = {"cost_per_1k_usd": round(s["cost"], 4)}
        if s["secs"]:
            row["p50_ms"] = round(1000 * (percentile(s["secs"], 0.5) or 0))
            row["p95_ms"] = round(1000 * (percentile(s["secs"], 0.95) or 0))
        for rname, gold in refs.items():
            row[rname] = {
                "accuracy": bootstrap_ci(gold, s["pred"], accuracy),
                "macro_f1": bootstrap_ci(gold, s["pred"], macro_f1),
                "per_class_f1": {k: round(v, 3) for k, v in per_class_f1(gold, s["pred"]).items()},
            }
        out["systems"][name] = row
    lora1, lora2 = (
        systems["small LoRA v1 (teacher labels)"],
        systems["small LoRA v2 (reference v2)"],
    )
    nli = systems["DeBERTa-v3-large NLI"]
    for rname, gold in refs.items():
        out["paired"][rname] = {
            "LoRA v2 - LoRA v1": {
                "accuracy": paired(gold, lora1["pred"], lora2["pred"], accuracy),
                "macro_f1": paired(gold, lora1["pred"], lora2["pred"], macro_f1),
            },
            "LoRA v2 - NLI": {
                "accuracy": paired(gold, nli["pred"], lora2["pred"], accuracy),
                "macro_f1": paired(gold, nli["pred"], lora2["pred"], macro_f1),
            },
        }

    # the edited contradictions the teacher called SUPPORTED, and all perturbed test pairs
    flipped = [
        i
        for i, p in enumerate(test)
        if p.kind in CONSTRUCTED_KINDS and p.labels["teacher"] == "SUPPORTED"
    ]
    perturbed = [i for i, p in enumerate(test) if p.kind in CONSTRUCTED_KINDS]
    out["edited_contradictions"] = {
        name: {
            "teacher_supported_18": sum(s["pred"][i] == "CONTRADICTED" for i in flipped),
            "all_perturbed": f"{sum(s['pred'][i] == 'CONTRADICTED' for i in perturbed)}"
            f"/{len(perturbed)}",
        }
        for name, s in systems.items()
    }
    out["edited_contradictions"]["n_flipped"] = len(flipped)

    # router curve on val (v2 reference), both adapters
    val = load("val")
    val_gold = references(val)["v2"]
    val_llm = cached_llm_labels(val)
    curve: dict[str, list[dict[str, float]]] = {}
    for name, probs in (("v1", v1["val"]), ("v2", v2["val"])):
        rows = []
        for t in THRESHOLDS:
            esc = [p for p in val if max(probs[p.id]) < t]
            if any(p.id not in val_llm for p in esc):
                rows.append({"threshold": t, "missing_llm": True})
                continue
            pred = router(probs, t, val, val_llm)
            rows.append(
                {
                    "threshold": t,
                    "escalated": round(len(esc) / len(val), 3),
                    "val_accuracy_v2": round(accuracy(val_gold, pred), 3),
                    "cost_per_1k_usd": round(len(esc) / len(val) * llm_cost_1k, 4),
                }
            )
        curve[name] = rows
    small_only = {
        n: round(accuracy(val_gold, [argmax(pr[p.id]) for p in val]), 3)
        for n, pr in (("v1", v1["val"]), ("v2", v2["val"]))
    }
    out["router_curve_val_v2_reference"] = {
        "curve": curve,
        "small_only_val_accuracy": small_only,
        "gemini_only_val_accuracy": round(
            accuracy(val_gold, [val_llm.get(p.id, "") for p in val]), 3
        ),
    }
    (RES / "compare_v2.json").write_text(json.dumps(out, indent=1))

    def c(d: dict[str, Any]) -> str:
        return f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"

    for rname in refs:
        print(f"\n### reference: {rname}\n")
        print("| system | accuracy | macro-F1 | F1 S / C / N | p50 / p95 ms | $ per 1k |")
        print("|---|---|---|---|---|---|")
        for name, row in out["systems"].items():
            r = row[rname]
            f1 = " / ".join(f"{r['per_class_f1'][k]:.2f}" for k in LABELS)
            lat = f"{row['p50_ms']} / {row['p95_ms']}" if "p95_ms" in row else "-"
            print(
                f"| {name} | {c(r['accuracy'])} | {c(r['macro_f1'])} | {f1} | {lat} | "
                f"{row['cost_per_1k_usd']:.3f} |"
            )
        for comp, m in out["paired"][rname].items():
            a, f = m["accuracy"], m["macro_f1"]
            print(
                f"\n{comp}: accuracy {a['delta']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}] "
                f"p={a['p']}; macro-F1 {f['delta']:+.3f} [{f['lo']:+.3f}, {f['hi']:+.3f}]"
            )
    print("\nedited contradictions:", json.dumps(out["edited_contradictions"], indent=1))
    print(
        "\nrouter curve (val, v2 reference):",
        json.dumps(out["router_curve_val_v2_reference"], indent=1),
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
