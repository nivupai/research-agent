"""LoRA sequence-classification verifier on a small open model (default Qwen3-1.7B, Apache-2.0).

Why a classifier head instead of generating the label: one forward pass gives calibrated
probabilities for SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO, which the confidence router
needs, there is no output parsing, and inference is a single short pass per pair.
The adapter (LoRA on the attention and MLP projections) and the classification head are
saved together with ``save_pretrained`` under ``models/verifier-lora/``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bioagent.finetune.data import LABELS
from bioagent.verify.nli import NLIScores

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER_DIR = Path("models/verifier-lora")
MAX_LENGTH = 512
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def format_input(claim: str, passage: str) -> str:
    return f"Claim: {claim}\nPassage: {passage}\nDoes the passage support the claim?"


def pick_device(device: str | None = None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_for_training(
    base: str = BASE_MODEL,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    device: str | None = None,
    dtype: str = "bfloat16",
) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        num_labels=len(LABELS),
        torch_dtype=getattr(torch, dtype),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )
    model.config.pad_token_id = tok.pad_token_id
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, config)
    # LoRA weights in fp32 for stability (PEFT casts their inputs); the frozen base and the
    # classification head stay in the model dtype so the head matches the hidden states
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            param.data = param.data.float()
    return model.to(pick_device(device)), tok


def param_counts(model: Any) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


class SmallVerifier:
    """Inference wrapper; also an ``NLIScorer`` (entailment = P(SUPPORTED), contradiction =
    P(CONTRADICTED), neutral = P(NOT_ENOUGH_INFO)) so bioagent.verify can use it."""

    def __init__(
        self, adapter_dir: Path = ADAPTER_DIR, device: str | None = None, batch_size: int = 8
    ) -> None:
        self.adapter_dir = Path(adapter_dir)
        self.device = device
        self.batch_size = batch_size
        self.name = f"lora:{self.adapter_dir.name}"
        self._lock = threading.Lock()
        self._model: Any = None
        self._tok: Any = None

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from peft import AutoPeftModelForSequenceClassification
            from transformers import AutoTokenizer

            meta = json.loads((self.adapter_dir / "training.json").read_text())
            self.device = pick_device(self.device)
            self._tok = AutoTokenizer.from_pretrained(self.adapter_dir)
            model = AutoPeftModelForSequenceClassification.from_pretrained(
                self.adapter_dir, num_labels=len(LABELS), torch_dtype=torch.bfloat16
            )
            model.config.pad_token_id = self._tok.pad_token_id
            self._model = model.to(self.device).eval()
            self.base = meta.get("base_model")

    def predict_proba(self, pairs: Sequence[tuple[str, str]]) -> list[list[float]]:
        """``pairs`` are (passage, claim); returns probabilities in LABELS order."""
        if not pairs:
            return []
        self._load()
        import torch

        out: list[list[float]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            enc = self._tok(
                [format_input(c, p) for p, c in batch],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self._model(**enc).logits.float()
            out.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        return out

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        sup, con, nei = (LABEL2ID[k] for k in ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"))
        return [NLIScores(p[sup], p[nei], p[con]) for p in self.predict_proba(pairs)]

    def timed_predict(
        self, pairs: Sequence[tuple[str, str]]
    ) -> tuple[list[list[float]], list[float]]:
        """Per-pair probabilities and per-pair latency (batch size 1, as a claim check)."""
        probs, secs = [], []
        for pair in pairs:
            t0 = time.monotonic()
            probs.extend(self.predict_proba([pair]))
            secs.append(time.monotonic() - t0)
        return probs, secs
