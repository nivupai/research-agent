"""BM25 index (rank-bm25 Okapi) persisted as gzipped JSON tokens.

Persisting tokens instead of pickling the model keeps the file format safe and portable;
rebuilding BM25Okapi from tokens takes well under a second for ~20k passages.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from bioagent.retrieval.tokenize import tokenize

INDEX_FILE = "bm25.json.gz"
FORMAT_VERSION = 1


class BM25Index:
    def __init__(
        self,
        ids: Sequence[str],
        tokens: Sequence[list[str]],
        fingerprint: str = "",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(ids) != len(tokens):
            raise ValueError("ids and tokens must have the same length")
        self.ids = list(ids)
        self.tokens = [list(t) for t in tokens]
        self.fingerprint = fingerprint
        self.k1, self.b = k1, b
        # BM25Okapi divides by corpus size; guard the empty corpus.
        self._bm25 = BM25Okapi(self.tokens, k1=k1, b=b) if self.ids else None

    @classmethod
    def build(cls, docs: Sequence[tuple[str, str]], fingerprint: str = "") -> BM25Index:
        return cls([d[0] for d in docs], [tokenize(d[1]) for d in docs], fingerprint)

    def search(
        self, query: str, n: int = 100, allowed: Collection[str] | None = None
    ) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = np.asarray(self._bm25.get_scores(q), dtype=float)
        if allowed is not None:
            mask = np.fromiter((i in allowed for i in self.ids), dtype=bool, count=len(self.ids))
            scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores, kind="stable")[:n]
        return [
            (self.ids[i], float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0
        ]

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / INDEX_FILE
        payload = {
            "version": FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "k1": self.k1,
            "b": self.b,
            "ids": self.ids,
            "tokens": self.tokens,
        }
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        with gzip.open(directory / INDEX_FILE, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") != FORMAT_VERSION:
            raise ValueError("unsupported BM25 index version; rebuild with `make index`")
        return cls(
            data["ids"],
            data["tokens"],
            data.get("fingerprint", ""),
            float(data["k1"]),
            float(data["b"]),
        )

    @staticmethod
    def stored_fingerprint(directory: Path) -> str | None:
        path = directory / INDEX_FILE
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            fingerprint = json.load(fh).get("fingerprint")
        return str(fingerprint) if fingerprint else None
