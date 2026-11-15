"""Cache the small verifier's class probabilities for the val and test splits.

Usage: uv run python scripts/verifier_probs.py [--adapter models/verifier-lora] [--out FILE]
Writes results/verifier/small_probs.json: {"adapter": ..., "splits": {split: {pair_id: [p_sup,
p_con, p_nei]}}}.
Local model only (no API calls); about 0.6 s per pair on Apple MPS.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bioagent.finetune.data import read_pairs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "verifier" / "small_probs.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=ROOT / "models" / "verifier-lora")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--timed", action="store_true", help="batch size 1 with per-pair latency (slower)"
    )
    args = parser.parse_args()

    from bioagent.finetune.model import SmallVerifier

    model = SmallVerifier(args.adapter)
    saved = json.loads(args.out.read_text()) if args.out.exists() else {}
    if saved and saved.get("adapter") != str(args.adapter):
        raise SystemExit(f"{args.out} holds another adapter ({saved.get('adapter')}); use --out")
    out: dict[str, dict[str, list[float]]] = saved.get("splits", {})
    latency: dict[str, list[float]] = saved.get("seconds", {})
    for split in args.splits.split(","):
        pairs = read_pairs(ROOT / "data" / "finetune" / f"{split}.jsonl")
        t0 = time.perf_counter()
        inputs = [(p.passage, p.claim) for p in pairs]
        if args.timed:
            model.predict_proba(inputs[:1])  # warm up
            probs, latency[split] = model.timed_predict(inputs)
        else:
            probs = model.predict_proba(inputs)
        out[split] = {p.id: [round(x, 6) for x in pr] for p, pr in zip(pairs, probs, strict=True)}
        print(f"{split}: {len(pairs)} pairs in {time.perf_counter() - t0:.0f} s", flush=True)
    args.out.write_text(
        json.dumps({"adapter": str(args.adapter), "splits": out, "seconds": latency}, indent=1)
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
