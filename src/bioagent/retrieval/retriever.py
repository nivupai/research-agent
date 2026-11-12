"""Single entry point for corpus search across BM25, dense, hybrid and hybrid+rerank."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from bioagent.config import Settings
from bioagent.ingest.store import CorpusStore
from bioagent.retrieval.bm25 import BM25Index
from bioagent.retrieval.dense import DenseIndex
from bioagent.retrieval.fusion import RRF_K, reciprocal_rank_fusion
from bioagent.retrieval.models import (
    SEARCH_MODES,
    PassageHit,
    SearchMode,
    SearchResults,
    StageScores,
)
from bioagent.retrieval.rerank import Reranker

N_CANDIDATES = 100
RERANK_TOP = 50
MAX_K = 100


class Retriever:
    """``search(query, k, mode, filters)`` over the passage store.

    Components are passed in (or built lazily via the ``*_factory`` callables) so that heavy
    models are only loaded when a mode needs them.
    """

    def __init__(
        self,
        store: CorpusStore,
        *,
        bm25: BM25Index | None = None,
        dense: DenseIndex | None = None,
        reranker: Reranker | None = None,
        bm25_factory: Callable[[], BM25Index] | None = None,
        dense_factory: Callable[[], DenseIndex] | None = None,
        reranker_factory: Callable[[], Reranker] | None = None,
        n_candidates: int = N_CANDIDATES,
        rerank_top: int = RERANK_TOP,
        rrf_k: int = RRF_K,
    ) -> None:
        self.store = store
        self._bm25, self._dense, self._reranker = bm25, dense, reranker
        self._bm25_factory, self._dense_factory = bm25_factory, dense_factory
        self._reranker_factory = reranker_factory
        self.n_candidates, self.rerank_top, self.rrf_k = n_candidates, rerank_top, rrf_k
        self._years: dict[str, int | None] | None = None
        self._lock = threading.Lock()  # SQLite connection + lazy model loading are shared

    # --- lazy components -----------------------------------------------------------------
    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            if self._bm25_factory is None:
                raise RuntimeError("no BM25 index configured")
            self._bm25 = self._bm25_factory()
        return self._bm25

    @property
    def dense(self) -> DenseIndex:
        if self._dense is None:
            if self._dense_factory is None:
                raise RuntimeError("no dense index configured")
            self._dense = self._dense_factory()
        return self._dense

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            if self._reranker_factory is None:
                raise RuntimeError("no reranker configured")
            self._reranker = self._reranker_factory()
        return self._reranker

    # --- search --------------------------------------------------------------------------
    def _allowed(self, year_min: int | None, year_max: int | None) -> set[str] | None:
        if year_min is None and year_max is None:
            return None
        if self._years is None:
            self._years = self.store.passage_years()
        lo = year_min if year_min is not None else -(10**9)
        hi = year_max if year_max is not None else 10**9
        return {pid for pid, y in self._years.items() if y is not None and lo <= y <= hi}

    def search(
        self,
        query: str,
        k: int = 10,
        mode: SearchMode = "hybrid_rerank",
        filters: Mapping[str, Any] | None = None,
    ) -> SearchResults:
        """Thread-safe search; see ``_search``."""
        with self._lock:
            return self._search(query, k, mode, filters)

    def _search(
        self,
        query: str,
        k: int,
        mode: SearchMode,
        filters: Mapping[str, Any] | None,
    ) -> SearchResults:
        if mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {SEARCH_MODES}")
        query = query.strip()
        if not query:
            raise ValueError("query is empty")
        k = max(1, min(int(k), MAX_K))
        filters = dict(filters or {})
        unknown = set(filters) - {"year_min", "year_max"}
        if unknown:
            raise ValueError(f"unsupported filters: {sorted(unknown)}")
        year_min = filters.get("year_min")
        year_max = filters.get("year_max")

        timings: dict[str, float] = {}
        stages: dict[str, StageScores] = {}

        def stage(pid: str) -> StageScores:
            return stages.setdefault(pid, StageScores())

        def timed(name: str, fn: Callable[[], Any]) -> Any:
            t0 = time.perf_counter()
            out = fn()
            timings[name] = round((time.perf_counter() - t0) * 1000, 2)
            return out

        n = max(k, self.n_candidates)
        bm25_hits: list[tuple[str, float]] = []
        dense_hits: list[tuple[str, float]] = []
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            allowed = self._allowed(year_min, year_max)
            bm25_hits = timed("bm25", lambda: self.bm25.search(query, n, allowed))
            for rank, (pid, score) in enumerate(bm25_hits, start=1):
                stage(pid).bm25, stage(pid).bm25_rank = round(score, 4), rank
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            dense_hits = timed("dense", lambda: self.dense.search(query, n, year_min, year_max))
            for rank, (pid, score) in enumerate(dense_hits, start=1):
                stage(pid).dense, stage(pid).dense_rank = round(score, 4), rank

        ranked: list[tuple[str, float]]
        if mode == "bm25":
            ranked = bm25_hits
        elif mode == "dense":
            ranked = dense_hits
        else:
            fused = timed(
                "rrf",
                lambda: reciprocal_rank_fusion(
                    [[p for p, _ in bm25_hits], [p for p, _ in dense_hits]], k=self.rrf_k
                ),
            )
            for rank, (pid, score) in enumerate(fused, start=1):
                stage(pid).rrf, stage(pid).rrf_rank = round(score, 6), rank
            ranked = fused
            if mode == "hybrid_rerank":
                pool = [pid for pid, _ in fused[: self.rerank_top]]
                texts = self.store.get_passages(pool)
                scores = timed(
                    "rerank", lambda: self.reranker.score(query, [texts[p].text for p in pool])
                )
                ranked = sorted(zip(pool, scores, strict=True), key=lambda x: -x[1])
                for rank, (pid, score) in enumerate(ranked, start=1):
                    stage(pid).rerank, stage(pid).rerank_rank = round(score, 4), rank

        top = ranked[:k]
        passages = self.store.get_passages([pid for pid, _ in top])
        hits = [
            PassageHit(
                passage_id=pid,
                pmid=passages[pid].pmid,
                chunk_idx=passages[pid].chunk_idx,
                title=passages[pid].title,
                text=passages[pid].text,
                year=passages[pid].year,
                journal=passages[pid].journal,
                doi=passages[pid].doi,
                score=round(float(score), 6),
                stages=stage(pid),
            )
            for pid, score in top
            if pid in passages
        ]
        timings["total"] = round(sum(timings.values()), 2)
        return SearchResults(query=query, mode=mode, hits=hits, timings_ms=timings)


def retriever_from_settings(settings: Settings, embed_model: str | None = None) -> Retriever:
    """Production retriever: SQLite store + persisted BM25 + Chroma + cross-encoder (lazy)."""
    from bioagent.retrieval.build import bm25_dir, chroma_dir
    from bioagent.retrieval.dense import make_client
    from bioagent.retrieval.embedder import SentenceTransformerEmbedder
    from bioagent.retrieval.rerank import CrossEncoderReranker

    store = CorpusStore(settings.corpus_db)
    model = embed_model or settings.embed_model

    def dense() -> DenseIndex:
        embedder = SentenceTransformerEmbedder(
            model, device=settings.embed_device, batch_size=settings.embed_batch_size
        )
        return DenseIndex(make_client(chroma_dir(settings)), embedder)

    return Retriever(
        store,
        bm25_factory=lambda: BM25Index.load(bm25_dir(settings)),
        dense_factory=dense,
        reranker_factory=lambda: CrossEncoderReranker(
            settings.rerank_model, device=settings.embed_device
        ),
    )
