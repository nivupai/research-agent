"""Natural-language inference scorers.

``NLIScorer.score`` takes (premise, hypothesis) pairs and returns entailment / neutral /
contradiction probabilities. ``TransformersNLI`` runs an MNLI-style cross-encoder locally
(needs the ``models`` extra); tests use a fake scorer with the same interface.
Long passages are split into sentence windows and the best window is kept.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
WINDOW_SENTENCES = 3
MAX_WINDOWS = 24
_PASSAGE_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n+")


@dataclass(frozen=True)
class NLIScores:
    entailment: float
    neutral: float
    contradiction: float


class NLIScorer(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]: ...


def passage_windows(
    text: str, size: int = WINDOW_SENTENCES, max_windows: int = MAX_WINDOWS
) -> list[str]:
    """Sliding windows of up to ``size`` sentences (title lines count as sentences)."""
    sentences = [s.strip() for s in _PASSAGE_SENT_RE.split(text) if s and s.strip()]
    if len(sentences) <= size:
        return [" ".join(sentences)] if sentences else []
    windows = [" ".join(sentences[i : i + size]) for i in range(len(sentences) - size + 1)]
    return windows[:max_windows]


class TransformersNLI:
    """Hugging Face sequence-classification NLI model, loaded lazily on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._lock = threading.Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._labels: dict[str, int] = {}

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = self.device
            if device is None:
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            self.device = device
            self._tokenizer = AutoTokenizer.from_pretrained(self.name)
            model = AutoModelForSequenceClassification.from_pretrained(self.name)
            self._model = model.to(device).eval()
            id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
            self._labels = {
                name: next(i for i, lab in id2label.items() if lab.startswith(name[:6]))
                for name in ("entailment", "neutral", "contradiction")
            }

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        if not pairs:
            return []
        self._load()
        import torch

        out: list[NLIScores] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            enc = self._tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self._model(**enc).logits.float(), dim=-1).cpu().tolist()
            for row in probs:
                out.append(
                    NLIScores(
                        entailment=row[self._labels["entailment"]],
                        neutral=row[self._labels["neutral"]],
                        contradiction=row[self._labels["contradiction"]],
                    )
                )
        return out
