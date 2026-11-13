"""Claim/passage pairs for the small verifier (resumable; every label is cached).

Steps (default: all):
  build    pairs from saved runs/eval answers + synthetic positives, cross-paper and perturbed
           negatives; dedupe; split by PMID -> data/finetune/pairs.jsonl
  teacher  Gemini labels, 20 pairs per structured call (flash-lite unless --teacher-model);
           stops cleanly when the quota runs out; rerun to continue
  nli      DeBERTa-v3-large NLI prediction for every pair (second baseline)
  export   data/finetune/{train,val,test}.jsonl with labels, and the human spot-label sheet
           eval/labels/verifier_test_human.jsonl (blank human_label)
  bundle   data/finetune/verifier_bundle.zip for notebooks/train_verifier_lora.ipynb
           (teacher-labelled train/val/test JSONL + README with the notebook settings)

Usage: uv run python scripts/make_verifier_data.py [build teacher nli export] [--teacher-model M]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from pathlib import Path

from bioagent.config import get_settings
from bioagent.eval.cache import JSONCache
from bioagent.finetune.data import (
    Pair,
    Paper,
    check_no_leakage,
    claims_from_record,
    dedupe,
    read_pairs,
    run_records,
    synthetic_pairs,
    write_pairs,
)
from bioagent.finetune.labels import label_with_teacher, nli_predict, teacher_key
from bioagent.ingest.store import CorpusStore
from bioagent.llm.providers import make_chat_model, parse_spec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "finetune"
PAIRS = OUT / "pairs.jsonl"
NLI_FILE = OUT / "nli_predictions.json"
CACHE = ROOT / "data" / "finetune" / "labels_cache.sqlite"
HUMAN = ROOT / "eval" / "labels" / "verifier_test_human.jsonl"
N_PAPERS = 2600
TARGET = {"positive": 1000, "cross": 800, "perturbed": 900}
N_HUMAN = 150


def build() -> None:
    settings = get_settings()
    run_pairs = [p for rec in run_records(ROOT / "results") for p in claims_from_record(rec)]
    store = CorpusStore(settings.corpus_db)
    try:
        rows = [r for r in store.iter_records() if r["abstract"] and len(r["abstract"]) > 500]
    finally:
        store.close()
    rng = random.Random(21)
    sample = rng.sample(rows, min(N_PAPERS, len(rows)))
    papers = [Paper(r["pmid"], r["title"] or "", r["abstract"]) for r in sample]
    from bioagent.retrieval.retriever import retriever_from_settings

    retriever = retriever_from_settings(settings)
    in_sample = {p.pmid for p in papers}
    neighbours: dict[str, list[str]] = {}
    for p in papers:
        query = p.title or p.abstract[:200]
        hits = retriever.search(query, k=15, mode="bm25").hits
        neighbours[p.pmid] = [h.pmid for h in hits if h.pmid != p.pmid and h.pmid in in_sample]
    synth = synthetic_pairs(
        papers,
        neighbours,
        n_positive=TARGET["positive"],
        n_cross=TARGET["cross"],
        n_perturbed=TARGET["perturbed"],
    )
    pairs = dedupe([*run_pairs, *synth])
    counts = check_no_leakage(pairs)
    write_pairs(PAIRS, pairs)
    print(f"build: {len(pairs)} pairs {dict(Counter(p.kind for p in pairs))}; splits {counts}")


async def teacher(model_spec: str) -> None:
    spec = parse_spec(model_spec)
    pairs = read_pairs(PAIRS)
    cache = JSONCache(CACHE)
    model = make_chat_model(spec, get_settings())
    result = await label_with_teacher(pairs, model, spec, cache)
    done = sum(cache.get(teacher_key(p, spec)) is not None for p in pairs)
    cache.close()
    print(json.dumps({**result, "labelled_total": done, "pairs": len(pairs)}, indent=1))
    if result["stopped"]:
        print("teacher stopped (quota); rerun this step later to continue")


def nli() -> None:
    from bioagent.verify.nli import TransformersNLI

    pairs = read_pairs(PAIRS)
    done = json.loads(NLI_FILE.read_text()) if NLI_FILE.exists() else {}
    todo = [p for p in pairs if p.id not in done]
    scorer = TransformersNLI(get_settings().nli_model)
    t0 = time.monotonic()
    for start in range(0, len(todo), 100):
        done |= nli_predict(todo[start : start + 100], scorer)
        NLI_FILE.write_text(json.dumps(done))
        print(f"  nli: {len(done)}/{len(pairs)} ({time.monotonic() - t0:.0f}s)")


def export(model_spec: str) -> None:
    spec = parse_spec(model_spec)
    pairs = read_pairs(PAIRS)
    cache = JSONCache(CACHE)
    nli_preds = json.loads(NLI_FILE.read_text()) if NLI_FILE.exists() else {}
    by_split: dict[str, list[Pair]] = {"train": [], "val": [], "test": []}
    for p in pairs:
        t = cache.get(teacher_key(p, spec))
        if t is None and p.id not in nli_preds:
            continue
        p.labels = {
            "teacher": t["label"] if t else None,
            "teacher_model": str(spec),
            "nli": nli_preds.get(p.id),
        }
        by_split[p.split].append(p)
    cache.close()
    for split, items in by_split.items():
        write_pairs(OUT / f"{split}.jsonl", items)
    rows = human_sheet(by_split["test"])
    labels = {s: dict(Counter(str(p.labels["teacher"]) for p in v)) for s, v in by_split.items()}
    summary = {
        "splits": {s: len(v) for s, v in by_split.items()},
        "teacher": labels,
        "human_sheet": len(rows),
    }
    print(json.dumps(summary, indent=1))


def human_sheet(test: list[Pair]) -> list[dict[str, object]]:
    """150 test pairs, stratified by teacher label, written once; later exports keep the same
    ids (and any human labels) so labelling work is never lost."""
    existing: dict[str, dict[str, object]] = {}
    if HUMAN.exists():
        rows_in = [json.loads(line) for line in HUMAN.read_text().splitlines() if line.strip()]
        existing = {str(r["id"]): r for r in rows_in}
    by_id = {p.id: p for p in test}
    if existing:
        chosen = [by_id[i] for i in existing if i in by_id]
    else:
        labelled = [p for p in test if p.labels.get("teacher")]
        if not labelled:
            print("human sheet: not written yet (needs teacher labels)")
            return []
        rng = random.Random(4)
        by_label: dict[str, list[Pair]] = {}
        for p in labelled:
            by_label.setdefault(str(p.labels["teacher"]), []).append(p)
        chosen = []
        for _label, group in sorted(by_label.items()):
            k = max(1, round(N_HUMAN * len(group) / len(labelled)))
            chosen += rng.sample(group, min(k, len(group)))
        chosen = sorted(chosen, key=lambda x: x.id)[:N_HUMAN]
    rows: list[dict[str, object]] = [
        {
            "id": p.id,
            "claim": p.claim,
            "passage": p.passage,
            "pmid": p.pmid,
            "kind": p.kind,
            "options": ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"],
            "human_label": (existing.get(p.id) or {}).get("human_label", ""),
        }
        for p in chosen
    ]
    HUMAN.parent.mkdir(parents=True, exist_ok=True)
    HUMAN.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows


BUNDLE = OUT / "verifier_bundle.zip"
BUNDLE_README = """Verifier training bundle for notebooks/train_verifier_lora.ipynb

Files: train.jsonl, val.jsonl, test.jsonl (one claim/passage pair per line; the label to learn
is labels.teacher = SUPPORTED | CONTRADICTED | NOT_ENOUGH_INFO, from {teacher}).
Counts (teacher-labelled): {counts}

Notebook settings (defaults in the notebook; keep them for comparable results):
  base model      Qwen/Qwen3-1.7B (Apache-2.0), 3-way sequence-classification head
  quantization    4-bit nf4 (QLoRA), compute dtype float16
  LoRA            r=16, alpha=32, dropout=0.05, q/k/v/o/gate/up/down projections, head trained
  steps           600, batch 8, gradient accumulation 2 (effective batch 16)
  learning rate   2e-4, 20 warm-up steps, linear decay; class-weighted cross-entropy
  max length      512 tokens
  runtime         Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Accelerator GPU T4.

How to use: upload this zip to the notebook (file upload, Google Drive at
MyDrive/verifier_bundle.zip, or a Kaggle dataset), Run all, download verifier-lora.zip, unzip
it into models/verifier-lora/ in the repo, then run:  uv run python scripts/eval_verifier.py
test.jsonl is included for that local evaluation; the notebook does not train on it.
"""


def bundle() -> Path:
    """Zip the teacher-labelled splits with a README of the notebook settings."""
    import zipfile

    counts: dict[str, dict[str, int]] = {}
    texts: dict[str, str] = {}
    for split in ("train", "val", "test"):
        rows = [json.loads(line) for line in (OUT / f"{split}.jsonl").read_text().splitlines()]
        labelled = [r for r in rows if (r.get("labels") or {}).get("teacher")]
        counts[split] = dict(Counter(r["labels"]["teacher"] for r in labelled))
        texts[f"{split}.jsonl"] = "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in labelled
        )
    if not any(counts.values()):
        raise SystemExit("no teacher labels yet: run the teacher and export steps first")
    teacher = next(
        (
            json.loads(line)["labels"]["teacher_model"]
            for line in (OUT / "train.jsonl").read_text().splitlines()
        ),
        "the Gemini teacher",
    )
    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, text in texts.items():
            zf.writestr(name, text)
        zf.writestr("README.txt", BUNDLE_README.format(teacher=teacher, counts=counts))
    size = BUNDLE.stat().st_size
    print(f"bundle: {BUNDLE} ({size / 1e6:.1f} MB) {counts}")
    return BUNDLE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", nargs="*", default=["build", "teacher", "nli", "export", "bundle"])
    parser.add_argument("--teacher-model", default="gemini:gemini-3.5-flash-lite")
    args = parser.parse_args()
    for step in args.steps:
        if step == "build":
            build()
        elif step == "teacher":
            asyncio.run(teacher(args.teacher_model))
        elif step == "nli":
            nli()
        elif step == "export":
            export(args.teacher_model)
        elif step == "bundle":
            bundle()
        else:
            raise SystemExit(f"unknown step {step}")
