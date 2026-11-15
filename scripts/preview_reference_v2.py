"""Score the existing verifiers against the v1 (teacher) and v2 (relabelled) test references.

Usage: uv run python scripts/preview_reference_v2.py [--threshold 0.75]

Reference v1 = Gemini teacher labels (data/finetune/test.jsonl). Reference v2 = the same pairs
with rule-perturbed pairs labelled CONTRADICTED by construction (scripts/make_verifier_data.py
relabel); "v2 without swap" leaves entity swaps on the teacher label, since a swapped entity
can read as NOT_ENOUGH_INFO rather than CONTRADICTED. Systems: DeBERTa NLI (stored on every
pair), the small LoRA (results/verifier/small_probs.json from scripts/verifier_probs.py), the
Gemini teacher itself, the Gemini verifier (read from the label cache only; never calls the
API) and the router (LoRA, escalating to the Gemini verifier label below the threshold).
Accuracy and macro-F1 with bootstrap 95% CIs. Nothing is retrained; this is a preview.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bioagent.config import get_settings
from bioagent.eval.cache import JSONCache
from bioagent.finetune.data import CONSTRUCTED_KINDS, Pair, read_pairs, relabel_by_construction
from bioagent.finetune.labels import LLM_VERIFIER_VERSION, teacher_key
from bioagent.finetune.metrics import accuracy, bootstrap_ci, macro_f1, per_class_f1
from bioagent.llm.providers import parse_spec

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")


def argmax(p: list[float]) -> str:
    return LABELS[max(range(3), key=lambda i: p[i])]


def cached_llm_labels(pairs: list[Pair]) -> dict[str, str]:
    settings = get_settings()
    spec = parse_spec(settings.verifier_llm_model)
    cache = JSONCache(ROOT / "data" / "finetune" / "labels_cache.sqlite")
    out: dict[str, str] = {}
    for p in pairs:
        item = Pair(claim=p.claim, passage=p.passage, pmid="0", kind="verify").finalise()
        hit = cache.get(teacher_key(item, spec, LLM_VERIFIER_VERSION))
        if hit:
            out[p.id] = str(hit["label"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "verifier" / "preview_v2.json"
    )
    args = parser.parse_args()

    pairs = [
        p
        for p in read_pairs(ROOT / "data" / "finetune" / "test.jsonl")
        if p.labels.get("teacher") in LABELS
    ]
    probs = json.loads((ROOT / "results" / "verifier" / "small_probs.json").read_text())["splits"]
    probs = probs["test"]
    llm = cached_llm_labels(pairs)
    refs = {
        "v1 teacher": [str(p.labels["teacher"]) for p in pairs],
        "v2 all constructed kinds": [
            str(relabel_by_construction(p).labels["target"]) for p in pairs
        ],
        "v2 without swap": [
            str(
                relabel_by_construction(p, [k for k in CONSTRUCTED_KINDS if k != "swap"]).labels[
                    "target"
                ]
            )
            for p in pairs
        ],
    }
    lora = [argmax(probs[p.id]) for p in pairs]
    systems: dict[str, list[str] | None] = {
        "DeBERTa-v3-large NLI": [str((p.labels.get("nli") or {}).get("label")) for p in pairs],
        "small LoRA (T4, trained on teacher labels)": lora,
        "Gemini teacher labels": [str(p.labels["teacher"]) for p in pairs],
        "Gemini verifier (cached)": [llm[p.id] for p in pairs] if len(llm) == len(pairs) else None,
        f"router @ {args.threshold}": [
            llm.get(p.id, lab) if max(probs[p.id]) < args.threshold else lab
            for p, lab in zip(pairs, lora, strict=True)
        ],
    }
    escalated = sum(max(probs[p.id]) < args.threshold for p in pairs)
    missing_escalated = sum(
        1 for p in pairs if max(probs[p.id]) < args.threshold and p.id not in llm
    )
    result: dict[str, Any] = {
        "n": len(pairs),
        "llm_cached": len(llm),
        "router_escalated": escalated,
        "router_escalated_missing_llm": missing_escalated,
        "rows": {},
    }
    for sname, pred in systems.items():
        if pred is None:
            continue
        for rname, gold in refs.items():
            result["rows"][f"{sname} | {rname}"] = {
                "accuracy": bootstrap_ci(gold, pred, accuracy),
                "macro_f1": bootstrap_ci(gold, pred, macro_f1),
                "per_class_f1": {k: round(v, 3) for k, v in per_class_f1(gold, pred).items()},
            }
    args.out.write_text(json.dumps(result, indent=1))

    def cell(d: dict[str, Any]) -> str:
        return f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"

    print(
        f"test pairs {len(pairs)}; Gemini verifier labels cached for {len(llm)}; router "
        f"escalates {escalated} (missing LLM label for {missing_escalated})"
    )
    print("\n| system | " + " | ".join(f"acc vs {r}" for r in refs) + " |")
    print("|---" * (len(refs) + 1) + "|")
    for sname in systems:
        cells = [result["rows"].get(f"{sname} | {r}") for r in refs]
        if cells[0] is None:
            print(f"| {sname} | not available |")
            continue
        print(f"| {sname} | " + " | ".join(cell(c["accuracy"]) for c in cells if c) + " |")
    print("\n| system | " + " | ".join(f"macro-F1 vs {r}" for r in refs) + " |")
    print("|---" * (len(refs) + 1) + "|")
    for sname in systems:
        cells = [result["rows"].get(f"{sname} | {r}") for r in refs]
        if cells[0] is not None:
            print(f"| {sname} | " + " | ".join(cell(c["macro_f1"]) for c in cells if c) + " |")


if __name__ == "__main__":
    main()
