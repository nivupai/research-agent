"""Verifier backends behind the ``NLIScorer`` interface (``VERIFIER_BACKEND``):

* ``nli``: DeBERTa-v3 NLI over sentence windows (default).
* ``small_lora``: the fine-tuned LoRA classifier (``bioagent.finetune.model``), whole passage.
* ``llm``: Gemini labels pairs in batches (structured output, cached) - the large model.
* ``router``: the small model, escalating pairs whose top probability is below
  ``VERIFIER_ROUTER_THRESHOLD`` to the LLM.

Scores are mapped to NLIScores (entailment = SUPPORTED, contradiction = CONTRADICTED,
neutral = NOT_ENOUGH_INFO); LLM labels become probabilities of ``LLM_CONFIDENCE``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bioagent.config import Settings
from bioagent.verify.nli import NLIScorer, NLIScores

LLM_CONFIDENCE = 0.9


class LLMVerifierUnavailableError(RuntimeError):
    """The LLM backend could not label every pair (quota exhausted or unparsable output)."""


class ProbaModel(Protocol):
    def predict_proba(self, pairs: Sequence[tuple[str, str]]) -> list[list[float]]: ...


def label_scores(label: str, confidence: float = LLM_CONFIDENCE) -> NLIScores:
    rest = (1 - confidence) / 2
    return NLIScores(
        entailment=confidence if label == "SUPPORTED" else rest,
        neutral=confidence if label == "NOT_ENOUGH_INFO" else rest,
        contradiction=confidence if label == "CONTRADICTED" else rest,
    )


def proba_scores(p: Sequence[float]) -> NLIScores:
    """LABELS order is SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO."""
    return NLIScores(entailment=p[0], neutral=p[2], contradiction=p[1])


@dataclass
class LLMPairScorer:
    """Batched Gemini labelling of (passage, claim) pairs; usable from a worker thread."""

    model: Any
    spec: Any
    cache: Any
    batch: int = 20
    name: str = "llm"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def labels(self, pairs: Sequence[tuple[str, str]]) -> list[str]:
        from bioagent.finetune.data import Pair
        from bioagent.finetune.labels import LLM_VERIFIER_VERSION, label_with_teacher, teacher_key

        items = [Pair(claim=c, passage=p, pmid="0", kind="verify").finalise() for p, c in pairs]
        result = asyncio.run(
            label_with_teacher(
                items,
                self.model,
                self.spec,
                self.cache,
                batch=self.batch,
                version=LLM_VERIFIER_VERSION,
                log=lambda _m: None,
            )
        )
        self.calls.append(result)
        out = []
        for item in items:
            cached = self.cache.get(teacher_key(item, self.spec, LLM_VERIFIER_VERSION))
            if cached is None:
                # never substitute a label silently (e.g. when the quota ran out)
                raise LLMVerifierUnavailableError(
                    f"no LLM verifier label for {len(items)} pair(s): "
                    f"{result.get('stopped') or 'the LLM output could not be parsed'}"
                )
            out.append(cached["label"])
        return out

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        return [label_scores(label) for label in self.labels(pairs)]


@dataclass
class RouterScorer:
    """Small model first; pairs with top probability < threshold go to the LLM."""

    small: ProbaModel
    llm: LLMPairScorer
    threshold: float
    name: str = "router"
    escalated: int = 0
    total: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        probs = self.small.predict_proba(pairs)
        low = [i for i, p in enumerate(probs) if max(p) < self.threshold]
        scores = [proba_scores(p) for p in probs]
        if low:
            for i, s in zip(low, self.llm.score([pairs[i] for i in low]), strict=True):
                scores[i] = s
        with self._lock:
            self.escalated += len(low)
            self.total += len(pairs)
        return scores


def route(probs: Sequence[Sequence[float]], threshold: float) -> list[bool]:
    """True where the small model's top probability is below the threshold (escalate)."""
    return [max(p) < threshold for p in probs]


def make_scorer(settings: Settings, backend: str | None = None) -> NLIScorer:
    backend = backend or settings.verifier_backend
    if backend == "nli":
        from bioagent.verify.nli import TransformersNLI

        return TransformersNLI(settings.nli_model, device=settings.nli_device)
    if backend in ("small_lora", "router"):
        from bioagent.finetune.model import SmallVerifier

        small = SmallVerifier(Path(settings.verifier_adapter_dir))
        if backend == "small_lora":
            return small
        return RouterScorer(small, make_llm_scorer(settings), settings.verifier_router_threshold)
    if backend == "llm":
        return make_llm_scorer(settings)
    raise ValueError(f"unknown VERIFIER_BACKEND {backend!r}")


def make_llm_scorer(settings: Settings) -> LLMPairScorer:
    from bioagent.eval.cache import JSONCache
    from bioagent.llm.providers import make_chat_model, parse_spec

    spec = parse_spec(settings.verifier_llm_model)
    cache = JSONCache(Path(settings.kg_cache_dir).parent / "finetune" / "labels_cache.sqlite")
    return LLMPairScorer(make_chat_model(spec, settings), spec, cache)


def windowed(backend: str) -> bool:
    """NLI scores 3-sentence windows; the classifier and the LLM read the whole passage."""
    return backend == "nli"
