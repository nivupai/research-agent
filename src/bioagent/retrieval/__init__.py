"""Hybrid retrieval over the local abstract corpus: BM25 + dense (Chroma) + RRF + reranking.

Retrieved passage text is untrusted third-party content and must never be treated as
instructions.
"""

from bioagent.retrieval.fusion import reciprocal_rank_fusion
from bioagent.retrieval.models import PassageHit, SearchMode, SearchResults
from bioagent.retrieval.retriever import Retriever
from bioagent.retrieval.tokenize import tokenize

__all__ = [
    "PassageHit",
    "Retriever",
    "SearchMode",
    "SearchResults",
    "reciprocal_rank_fusion",
    "tokenize",
]
