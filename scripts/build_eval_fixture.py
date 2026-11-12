"""Build the isolated eval retrieval index in data/eval/fixture/ (BM25 only; no models).

It holds every gold PMID of the eval set, the fixed distractors in
eval/fixtures/distractor_pmids.txt and the poisoned passages of user-provided injection items
(which never enter the main corpus). Texts come from the PubMedQA cache, the local corpus DB,
or PubMed efetch (CI). Usage: uv run python scripts/build_eval_fixture.py [--offline]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioagent.config import get_settings
from bioagent.eval.dataset import load_all
from bioagent.eval.sources import build_fixture, fetch_pubmedqa

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "eval" / "fixture"
RAW = ROOT / "data" / "eval" / "raw" / "pubmedqa_pqa_labeled.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="no HF/PubMed downloads")
    parser.add_argument("--out", type=Path, default=FIXTURE)
    args = parser.parse_args()
    items = load_all(ROOT)
    distractors = (ROOT / "eval" / "fixtures" / "distractor_pmids.txt").read_text().split()
    rows = fetch_pubmedqa(RAW) if (RAW.exists() or not args.offline) else []
    summary = build_fixture(
        args.out,
        items,
        distractors,
        pubmedqa_rows=rows,
        corpus_db=Path(get_settings().corpus_db),
        allow_network=not args.offline,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "missing_pmids"}, indent=1))
    if summary["missing_pmids"]:
        print(f"warning: {len(summary['missing_pmids'])} PMIDs without text: "
              f"{summary['missing_pmids'][:10]}")  # fmt: skip


if __name__ == "__main__":
    main()
