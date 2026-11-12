"""Cross-encoder reranking (lazy sentence-transformers import)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from bioagent.retrieval.embedder import pick_device


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed; run `uv sync --extra models`"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = CrossEncoder(
            model_name, device=pick_device(device), max_length=max_length
        )

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict(
            [(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]
