"""Re-score one claim of a saved run with every local verifier backend (no API calls).

Usage: uv run python scripts/repro_claim.py RUN_ID CLAIM_N [--claim "edited claim text"]

Prints, for each cited paper: the DeBERTa NLI scores of the windows the pipeline would pick
(entailment / contradiction per window), the small LoRA verifier's class probabilities on the
whole passage, whether the router (threshold from settings) would escalate it to the LLM,
and the LLM label if one is already cached (the cache is read, never filled).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioagent.config import get_settings
from bioagent.finetune.data import Pair
from bioagent.finetune.labels import LLM_VERIFIER_VERSION, teacher_key
from bioagent.verify.nli import passage_windows
from bioagent.verify.verifier import VerifyConfig, top_windows

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("claim_n", type=int)
    parser.add_argument("--claim", help="score this text instead of the saved claim")
    args = parser.parse_args()

    record = json.loads((ROOT / "results" / "runs" / f"{args.run_id}.json").read_text())
    claim = next(c for c in record["verification"]["claims"] if c["n"] == args.claim_n)
    text = args.claim or claim["claim"]
    evidence = record["cited_evidence"]
    print(f"claim {claim['n']}: {text}")
    print(f"saved label: {claim['label']} ({claim['check']})\n")

    settings = get_settings()
    from bioagent.eval.cache import JSONCache
    from bioagent.finetune.model import SmallVerifier
    from bioagent.llm.providers import parse_spec
    from bioagent.verify.nli import TransformersNLI

    nli = TransformersNLI(settings.nli_model, device=settings.nli_device)
    small = SmallVerifier(Path(settings.verifier_adapter_dir))
    spec = parse_spec(settings.verifier_llm_model)
    cache = JSONCache(Path(settings.kg_cache_dir).parent / "finetune" / "labels_cache.sqlite")
    k = VerifyConfig().windows_per_citation

    for row in claim.get("evidence", []):
        pmid = row["ref"].removeprefix("PMID ").strip()
        texts = [e["text"] for e in evidence.values() if str(e.get("pmid")) == pmid]
        print(f"[{row['cite']}] {row['ref']} ({len(texts)} passage(s))")
        windows = top_windows([w for t in texts for w in passage_windows(t)], text, k)
        for w, s in zip(windows, nli.score([(w, text) for w in windows]), strict=True):
            print(f"  NLI  e={s.entailment:.3f} c={s.contradiction:.3f}  {w[:150]!r}")
        for t in texts:
            p = small.predict_proba([(t, text)])[0]
            top = LABELS[max(range(3), key=lambda i: p[i])]
            esc = max(p) < settings.verifier_router_threshold
            item = Pair(claim=text, passage=t, pmid="0", kind="verify").finalise()
            cached = cache.get(teacher_key(item, spec, LLM_VERIFIER_VERSION))
            llm = cached["label"] if cached else "not cached"
            print(
                f"  LoRA {top} p(S/C/N)={p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}; router "
                f"{'escalates (LLM: ' + llm + ')' if esc else 'keeps the LoRA label'}"
            )
        print()


if __name__ == "__main__":
    main()
