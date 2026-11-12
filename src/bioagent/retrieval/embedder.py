"""Text embedders. The sentence-transformers one is imported lazily (optional ``models``
extra), so the rest of the package works without torch installed."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

# BGE v1.5 English models expect this prefix on queries (not on passages).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def query_instruction_for(model_name: str) -> str:
    name = model_name.lower()
    if "bge" in name and "-en" in name and "m3" not in name:
        return BGE_QUERY_INSTRUCTION
    return ""


def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    try:
        import torch
    except ImportError:  # pragma: no cover - torch comes with the models extra
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder:
    """Normalised sentence-transformers embeddings (cosine == dot product)."""

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed; run `uv sync --extra models`"
            ) from exc
        self.model_name = model_name
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.query_instruction = query_instruction_for(model_name)
        self._model: Any = SentenceTransformer(model_name, device=self.device)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self.query_instruction + text])[0]
