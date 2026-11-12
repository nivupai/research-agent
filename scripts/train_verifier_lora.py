"""LoRA fine-tuning of the small claim verifier (local MPS/CUDA/CPU).

Trains a sequence-classification head + LoRA adapters on data/finetune/train.jsonl and
evaluates on val.jsonl; saves to models/verifier-lora/ with training.json (params, steps,
time, peak memory, val metrics). For a free GPU use notebooks/train_verifier_lora.ipynb.

Usage: uv run python scripts/train_verifier_lora.py [--steps 200] [--batch 4] [--accum 4]
       [--labels teacher|nli]   (nli = provisional DeBERTa labels, for pipeline tests only)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from bioagent.finetune.data import LABELS, read_pairs
from bioagent.finetune.metrics import accuracy, macro_f1
from bioagent.finetune.model import (
    ADAPTER_DIR,
    BASE_MODEL,
    LABEL2ID,
    MAX_LENGTH,
    format_input,
    load_for_training,
    param_counts,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "finetune"


def label_of(pair: Any, source: str) -> str | None:
    if source == "teacher":
        return pair.labels.get("teacher")
    nli = pair.labels.get("nli") or {}
    return nli.get("label")


def memory_gb(device: str) -> float | None:
    import torch

    if device == "mps":
        return round(torch.mps.driver_allocated_memory() / 1e9, 2)
    if device == "cuda":
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return None


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--labels", choices=["teacher", "nli"], default="teacher")
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--out", type=Path, default=ROOT / ADAPTER_DIR)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--val-limit", type=int, default=120)
    parser.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="trade compute for memory (needed on 16-18 GB Macs)",
    )
    args = parser.parse_args()

    train = [p for p in read_pairs(DATA / "train.jsonl") if label_of(p, args.labels)]
    val = [p for p in read_pairs(DATA / "val.jsonl") if label_of(p, args.labels)][: args.val_limit]
    if not train:
        raise SystemExit("no labelled training pairs: run scripts/make_verifier_data.py first")
    counts = {lab: sum(label_of(p, args.labels) == lab for p in train) for lab in LABELS}
    weights = torch.tensor([len(train) / (len(LABELS) * max(1, counts[lab])) for lab in LABELS])
    print(f"train {len(train)} {counts}; val {len(val)}; labels from {args.labels}")

    t_load = time.monotonic()
    model, tok = load_for_training(args.base)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    device = str(next(model.parameters()).device)
    params = param_counts(model)
    print(f"loaded {args.base} on {device} in {time.monotonic() - t_load:.0f}s; params {params}")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / max(1, args.steps))
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    rng = random.Random(0)

    def batch_tensors(pairs: list[Any]) -> tuple[Any, Any]:
        enc = tok(
            [format_input(p.claim, p.passage) for p in pairs],
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        y = torch.tensor([LABEL2ID[label_of(p, args.labels)] for p in pairs], device=device)
        return enc, y

    def evaluate() -> dict[str, float]:
        model.eval()
        preds = []
        with torch.inference_mode():
            for i in range(0, len(val), 8):
                enc, _ = batch_tensors(val[i : i + 8])
                preds += model(**enc).logits.argmax(-1).cpu().tolist()
        model.train()
        gold = [label_of(p, args.labels) or "" for p in val]
        pred = [LABELS[i] for i in preds]
        return {"accuracy": accuracy(gold, pred), "macro_f1": macro_f1(gold, pred)}

    history: list[dict[str, Any]] = []
    model.train()
    t0 = time.monotonic()
    order: list[int] = []
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        total = 0.0
        for _ in range(args.accum):
            if len(order) < args.batch:
                order += rng.sample(range(len(train)), len(train))
            idx, order = order[: args.batch], order[args.batch :]
            enc, y = batch_tensors([train[i] for i in idx])
            loss = loss_fn(model(**enc).logits.float(), y) / args.accum
            loss.backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        if step % 10 == 0 or step == 1:
            rate = (time.monotonic() - t0) / step
            print(
                f"step {step}/{args.steps} loss {total:.3f} {rate:.1f}s/step "
                f"mem {memory_gb(device)} GB"
            )
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate()
            history.append({"step": step, "loss": total, **metrics})
            print(f"  val {metrics}")
    seconds = time.monotonic() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    meta = {
        "base_model": args.base,
        "label_source": args.labels,
        "params": params,
        "steps": args.steps,
        "batch": args.batch,
        "accum": args.accum,
        "effective_batch": args.batch * args.accum,
        "examples_seen": args.steps * args.batch * args.accum,
        "lr": args.lr,
        "max_length": args.max_length,
        "gradient_checkpointing": args.grad_checkpointing,
        "train_pairs": len(train),
        "device": device,
        "train_seconds": round(seconds, 1),
        "seconds_per_step": round(seconds / args.steps, 2),
        "peak_memory_gb": memory_gb(device),
        "history": history,
    }
    (args.out / "training.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
