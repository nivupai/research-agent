"""Result models for corpus retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from bioagent.mcp_servers.literature.models import UNTRUSTED_NOTICE

SearchMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
SEARCH_MODES: tuple[SearchMode, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")


class StageScores(BaseModel):
    """Score and 1-based rank from each stage that saw this passage (None = not retrieved)."""

    bm25: float | None = None
    bm25_rank: int | None = None
    dense: float | None = Field(default=None, description="cosine similarity")
    dense_rank: int | None = None
    rrf: float | None = None
    rrf_rank: int | None = None
    rerank: float | None = None
    rerank_rank: int | None = None


class PassageHit(BaseModel):
    passage_id: str
    pmid: str
    chunk_idx: int
    title: str | None
    text: str
    year: int | None
    journal: str | None
    doi: str | None
    score: float = Field(description="score from the final stage of the chosen mode")
    stages: StageScores
    untrusted_text: Literal[True] = True


class SearchResults(BaseModel):
    query: str
    mode: SearchMode
    hits: list[PassageHit]
    timings_ms: dict[str, float]
    untrusted_text: Literal[True] = True
    notice: str = UNTRUSTED_NOTICE
