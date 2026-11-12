"""Reciprocal rank fusion (Cormack et al. 2009)."""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank), rank starting at 1.

    Ties are broken by best (lowest) rank in any list, then by id for determinism.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))
