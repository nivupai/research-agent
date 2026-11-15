"""Cache the small verifier's class probabilities for the val and test splits.

Usage: uv run python scripts/verifier_probs.py [--adapter models/verifier-lora]
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
    args = parser.parse_args()

    from bioagent.finetune.model import SmallVerifier

    model = SmallVerifier(args.adapter)
    saved = json.loads(OUT.read_text()) if OUT.exists() else {}
    out: dict[str, dict[str, list[float]]] = saved.get("splits", {})
    for split in args.splits.split(","):
        pairs = read_pairs(ROOT / "data" / "finetune" / f"{split}.jsonl")
        t0 = time.perf_counter()
        probs = model.predict_proba([(p.passage, p.claim) for p in pairs])
        out[split] = {p.id: [round(x, 6) for x in pr] for p, pr in zip(pairs, probs, strict=True)}
        print(f"{split}: {len(pairs)} pairs in {time.perf_counter() - t0:.0f} s", flush=True)
    OUT.write_text(json.dumps({"adapter": str(args.adapter), "splits": out}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
