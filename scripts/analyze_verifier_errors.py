"""Small-verifier error analysis and fixes that need no retraining (val-tuned, test-scored).

Usage: uv run python scripts/analyze_verifier_errors.py [--out results/verifier/errors.json]

Needs results/verifier/small_probs.json (scripts/verifier_probs.py). Reference labels are the
Gemini teacher's; the DeBERTa NLI scores already stored on every pair are reused (no model is
loaded and no API is called). Fixes considered, each tuned on val only and then scored once on
test against the unchanged model:

* class bias: add a log-prior offset to NOT_ENOUGH_INFO (and CONTRADICTED) before argmax;
* NLI gate: a SUPPORTED prediction whose NLI entailment is below t becomes NOT_ENOUGH_INFO;
* temperature scaling: changes no prediction, only calibration (ECE) and router escalation.

Test deltas come with a paired bootstrap 95% CI and a sign-flip permutation p-value.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bioagent.eval.stats import permutation_test
from bioagent.finetune.data import Pair, read_pairs
from bioagent.finetune.metrics import accuracy, bootstrap_ci, confusion, macro_f1, per_class_f1

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")
S, C, N = LABELS
PROBS = ROOT / "results" / "verifier" / "small_probs.json"
WORD = __import__("re").compile(r"[a-z0-9]+")


def words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if len(w) > 2}


def argmax(p: Sequence[float]) -> str:
    return LABELS[max(range(3), key=lambda i: p[i])]


def with_bias(p: Sequence[float], b_con: float, b_nei: float) -> str:
    logits = [math.log(max(p[0], 1e-9)), math.log(max(p[1], 1e-9)) + b_con]
    logits.append(math.log(max(p[2], 1e-9)) + b_nei)
    return LABELS[max(range(3), key=lambda i: logits[i])]


def nli_entail(pair: Pair) -> float:
    return float((pair.labels.get("nli") or {}).get("entailment", 1.0))


def gate(pred: str, pair: Pair, t: float) -> str:
    return N if pred == S and nli_entail(pair) < t else pred


def temperature(p: Sequence[float], t: float) -> list[float]:
    logs = [math.log(max(x, 1e-9)) / t for x in p]
    m = max(logs)
    e = [math.exp(x - m) for x in logs]
    z = sum(e)
    return [x / z for x in e]


def ece(gold: Sequence[str], probs: Sequence[Sequence[float]], bins: int = 10) -> float:
    total, out = len(gold), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo < max(p) <= hi or (b == 0 and max(p) == 0)]
        if not idx:
            continue
        conf = sum(max(probs[i]) for i in idx) / len(idx)
        acc = sum(argmax(probs[i]) == gold[i] for i in idx) / len(idx)
        out += len(idx) / total * abs(conf - acc)
    return out


def nll(gold: Sequence[str], probs: Sequence[Sequence[float]]) -> float:
    return -sum(math.log(max(p[LABELS.index(g)], 1e-9)) for g, p in zip(gold, probs, strict=True))


def paired_delta(
    gold: Sequence[str],
    base: Sequence[str],
    cand: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    b: int = 2000,
    seed: int = 7,
) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(gold)
    deltas = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        deltas.append(metric(g, [cand[i] for i in idx]) - metric(g, [base[i] for i in idx]))
    deltas.sort()
    per_item = [float(c == g) - float(x == g) for g, x, c in zip(gold, base, cand, strict=True)]
    return {
        "delta": round(metric(gold, cand) - metric(gold, base), 4),
        "lo": round(deltas[int(0.025 * (b - 1))], 4),
        "hi": round(deltas[int(0.975 * (b - 1))], 4),
        "p_accuracy_sign_flip": round(permutation_test(per_item), 4),
        "changed": sum(x != c for x, c in zip(base, cand, strict=True)),
    }


def describe(gold: Sequence[str], pred: Sequence[str]) -> dict[str, Any]:
    return {
        "accuracy": bootstrap_ci(gold, pred, accuracy),
        "macro_f1": bootstrap_ci(gold, pred, macro_f1),
        "per_class_f1": {k: round(v, 3) for k, v in per_class_f1(gold, pred).items()},
        "confusion": confusion(gold, pred),
    }


def load(split: str, probs: dict[str, Any]) -> tuple[list[Pair], list[str], list[list[float]]]:
    pairs = [p for p in read_pairs(ROOT / "data" / "finetune" / f"{split}.jsonl")]
    pairs = [p for p in pairs if p.labels.get("teacher") in LABELS]
    return pairs, [str(p.labels["teacher"]) for p in pairs], [probs[split][p.id] for p in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "verifier" / "errors.json")
    args = parser.parse_args()
    saved = json.loads(PROBS.read_text())
    probs = saved["splits"]
    vp, vg, vprob = load("val", probs)
    tp, tg, tprob = load("test", probs)
    v_base = [argmax(p) for p in vprob]
    t_base = [argmax(p) for p in tprob]
    report: dict[str, Any] = {"adapter": saved["adapter"], "baseline_test": describe(tg, t_base)}
    report["baseline_val"] = describe(vg, v_base)
    train = read_pairs(ROOT / "data" / "finetune" / "train.jsonl")
    report["label_balance"] = {
        split: dict(Counter(str(p.labels.get("teacher")) for p in pairs))
        for split, pairs in (("train", train), ("val", vp), ("test", tp))
    }

    # --- the NOT_ENOUGH_INFO -> SUPPORTED errors, with features for categorising them
    cases = []
    for pair, g, p in zip(tp, tg, tprob, strict=True):
        if g == N and argmax(p) == S:
            cw, pw = words(pair.claim), words(pair.passage)
            cases.append(
                {
                    "id": pair.id,
                    "kind": pair.kind,
                    "p": [round(x, 3) for x in p],
                    "nli": pair.labels.get("nli"),
                    "claim_words_in_passage": round(len(cw & pw) / max(1, len(cw)), 2),
                    "claim": pair.claim,
                    "passage": pair.passage,
                }
            )
    report["nei_as_supported"] = cases
    nei_test = [i for i, g in enumerate(tg) if g == N]
    report["nei_test_by_kind"] = {
        k: {
            "n": sum(1 for i in nei_test if tp[i].kind == k),
            "called_supported": sum(1 for i in nei_test if tp[i].kind == k and t_base[i] == S),
        }
        for k in sorted({tp[i].kind for i in nei_test})
    }

    # --- fix 1: class bias, tuned on val by macro-F1 (accuracy as tie-break)
    grid = [x / 4 for x in range(-4, 13)]
    best_bias = max(
        ((bc, bn) for bc in grid for bn in grid),
        key=lambda b: (
            round(macro_f1(vg, [with_bias(p, *b) for p in vprob]), 4),
            round(accuracy(vg, [with_bias(p, *b) for p in vprob]), 4),
            -abs(b[0]) - abs(b[1]),
        ),
    )
    nei_only = max(
        grid,
        key=lambda bn: (
            round(macro_f1(vg, [with_bias(p, 0.0, bn) for p in vprob]), 4),
            -abs(bn),
        ),
    )
    fixes: dict[str, Any] = {}
    for name, fn in (
        (f"bias NEI+{nei_only}", lambda p, _pair: with_bias(p, 0.0, nei_only)),
        (f"bias CON+{best_bias[0]} NEI+{best_bias[1]}", lambda p, _pair: with_bias(p, *best_bias)),
    ):
        v_pred = [fn(p, pair) for p, pair in zip(vprob, vp, strict=True)]
        t_pred = [fn(p, pair) for p, pair in zip(tprob, tp, strict=True)]
        fixes[name] = {
            "val_macro_f1": round(macro_f1(vg, v_pred), 4),
            "val_accuracy": round(accuracy(vg, v_pred), 4),
            "test": describe(tg, t_pred),
            "test_vs_base_accuracy": paired_delta(tg, t_base, t_pred, accuracy),
            "test_vs_base_macro_f1": paired_delta(tg, t_base, t_pred, macro_f1),
        }

    # --- fix 2: NLI gate on SUPPORTED, tuned on val
    ts = [x / 100 for x in range(0, 51, 1)]
    best_t = max(
        ts,
        key=lambda t: (
            round(macro_f1(vg, [gate(x, pr, t) for x, pr in zip(v_base, vp, strict=True)]), 4),
            -t,
        ),
    )
    v_pred = [gate(x, pr, best_t) for x, pr in zip(v_base, vp, strict=True)]
    t_pred = [gate(x, pr, best_t) for x, pr in zip(t_base, tp, strict=True)]
    fixes[f"NLI gate t={best_t}"] = {
        "val_macro_f1": round(macro_f1(vg, v_pred), 4),
        "val_accuracy": round(accuracy(vg, v_pred), 4),
        "test": describe(tg, t_pred),
        "test_vs_base_accuracy": paired_delta(tg, t_base, t_pred, accuracy),
        "test_vs_base_macro_f1": paired_delta(tg, t_base, t_pred, macro_f1),
    }
    report["fixes"] = fixes
    report["val_baseline"] = {"macro_f1": round(macro_f1(vg, v_base), 4),
                              "accuracy": round(accuracy(vg, v_base), 4)}  # fmt: skip

    # --- calibration: temperature on val (NLL), ECE before/after, router escalation at 0.75
    temps = [x / 20 for x in range(10, 61)]
    best_temp = min(temps, key=lambda t: nll(vg, [temperature(p, t) for p in vprob]))
    t_cal = [temperature(p, best_temp) for p in tprob]
    report["calibration"] = {
        "temperature": best_temp,
        "test_ece_before": round(ece(tg, tprob), 4),
        "test_ece_after": round(ece(tg, t_cal), 4),
        "escalated_at_0.75_before": round(sum(max(p) < 0.75 for p in tprob) / len(tprob), 3),
        "escalated_at_0.75_after": round(sum(max(p) < 0.75 for p in t_cal) / len(t_cal), 3),
        "nei_as_supported_mean_confidence": round(
            sum(max(c["p"]) for c in cases) / max(1, len(cases)), 3
        ),
        "nei_as_supported_escalated_at_0.75": sum(max(c["p"]) < 0.75 for c in cases),
    }
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("nei_as_supported",)},
                     indent=1, default=str)[:6000])  # fmt: skip
    print(f"\n{len(cases)} NOT_ENOUGH_INFO -> SUPPORTED cases written to {args.out}")


if __name__ == "__main__":
    main()
