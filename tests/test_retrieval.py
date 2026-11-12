from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Sequence
from pathlib import Path

import pytest

from bioagent.ingest.store import CorpusStore
from bioagent.mcp_servers.literature.models import PaperRecord
from bioagent.retrieval.bm25 import BM25Index
from bioagent.retrieval.build import build_indexes
from bioagent.retrieval.dense import DenseIndex, collection_name, make_client
from bioagent.retrieval.embedder import BGE_QUERY_INSTRUCTION, query_instruction_for
from bioagent.retrieval.fusion import reciprocal_rank_fusion
from bioagent.retrieval.retriever import Retriever
from bioagent.retrieval.tokenize import tokenize

DOCS = {
    "1": (
        "Metformin activates AMPK in glioblastoma cells",
        "Metformin reduced glioblastoma proliferation through AMPK activation.",
        2015,
    ),
    "2": (
        "Temozolomide resistance and MGMT",
        "MGMT promoter methylation predicts temozolomide benefit in glioma.",
        2018,
    ),
    "3": (
        "PI3K inhibitors in breast cancer",
        "Alpelisib targets PIK3CA-mutant breast tumours; hyperglycaemia is common.",
        2021,
    ),
    "4": (
        "Blood-brain barrier drug delivery",
        "Nanoparticles cross the blood-brain barrier to deliver drugs to brain tumours.",
        2010,
    ),
    "5": (
        "mTOR signalling in neurodegeneration",
        "Rapamycin inhibits mTORC1 and induces autophagy in neurodegeneration models.",
        2005,
    ),
}


class FakeEmbedder:
    """Deterministic hashed bag-of-words embeddings (no model download)."""

    def __init__(self, dim: int = 64) -> None:
        self.model_name = f"fake-{uuid.uuid4().hex[:8]}"
        self.dim = dim
        self.document_calls = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeReranker:
    model_name = "fake-reranker"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) / (1 + len(tokenize(t)) / 100) for t in texts]


def make_store() -> CorpusStore:
    store = CorpusStore(":memory:")
    store.add_records(
        [
            PaperRecord(source="pubmed", pmid=pmid, title=t, abstract=a, year=y, journal="J")
            for pmid, (t, a, y) in DOCS.items()
        ],
        "test",
    )
    store.sync_passages()
    return store


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    store = make_store()
    passages = store.all_passages()
    bm25 = BM25Index.build([(p.passage_id, p.text) for p in passages])
    dense = DenseIndex(make_client(None), FakeEmbedder())
    dense.sync(passages)
    return Retriever(
        store, bm25=bm25, dense=dense, reranker=FakeReranker(), n_candidates=10, rerank_top=5
    )


# --- tokenizer ------------------------------------------------------------------------
def test_tokenizer_keeps_biomedical_terms() -> None:
    toks = tokenize("AMPK and PI3K/mTOR signalling; the IL-6 blood-brain barrier, 5-FU vs. TMZ.")
    for t in [
        "ampk",
        "pi3k/mtor",
        "pi3k",
        "mtor",
        "il-6",
        "blood-brain",
        "blood",
        "brain",
        "barrier",
        "5-fu",
        "fu",
        "tmz",
        "signalling",
    ]:
        assert t in toks, t
    for stop in ["and", "the", "vs"]:
        assert stop not in toks
    assert "il" in toks and "6" in toks  # parts of hyphenated terms are kept too


def test_tokenizer_unicode_and_case() -> None:
    # single-letter parts such as "β" are dropped; the compound keeps it
    assert tokenize("β-Amyloid MTORC1") == ["β-amyloid", "amyloid", "mtorc1"]


# --- RRF ------------------------------------------------------------------------------
def test_rrf_maths() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)
    order = [d for d, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)]
    assert order == ["b", "a", "d", "c"]  # d (rank 2) beats c (rank 3) on score


def test_rrf_tie_break_deterministic() -> None:
    order = [d for d, _ in reciprocal_rank_fusion([["x", "y"], ["y", "x"]])]
    assert order == ["x", "y"]


# --- BM25 -----------------------------------------------------------------------------
def test_bm25_exact_term_and_persistence(tmp_path: Path) -> None:
    store = make_store()
    docs = [(p.passage_id, p.text) for p in store.all_passages()]
    index = BM25Index.build(docs, fingerprint="fp1")
    assert index.search("MGMT", n=3)[0][0] == "2:0"
    assert index.search("PIK3CA alpelisib", n=1)[0][0] == "3:0"
    assert index.search("zzzz") == []
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded.search("MGMT", n=3) == index.search("MGMT", n=3)
    assert BM25Index.stored_fingerprint(tmp_path) == "fp1"
    assert index.search("glioblastoma", allowed={"2:0"}) == []


# --- dense ----------------------------------------------------------------------------
def test_dense_sync_only_embeds_changes() -> None:
    store = make_store()
    embedder = FakeEmbedder()
    dense = DenseIndex(make_client(None), embedder)
    first = dense.sync(store.all_passages())
    assert first.embedded == 5 and embedder.document_calls == 5
    second = dense.sync(store.all_passages())
    assert second.embedded == 0 and second.unchanged == 5 and embedder.document_calls == 5
    keep = store.all_passages()[:3]
    third = dense.sync(keep)
    assert third.deleted == 2 and dense.count() == 3


def test_bge_query_instruction() -> None:
    assert query_instruction_for("BAAI/bge-small-en-v1.5") == BGE_QUERY_INSTRUCTION
    assert query_instruction_for("pritamdeka/S-PubMedBert-MS-MARCO") == ""
    assert collection_name("BAAI/bge-small-en-v1.5") == "passages__BAAI-bge-small-en-v1.5"


# --- retriever ------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_retriever_modes(retriever: Retriever, mode: str) -> None:
    results = retriever.search("temozolomide MGMT glioma", k=3, mode=mode)  # type: ignore[arg-type]
    assert results.hits[0].pmid == "2"
    assert results.mode == mode and results.untrusted_text
    assert all(h.untrusted_text for h in results.hits)
    assert "total" in results.timings_ms


def test_retriever_stage_scores(retriever: Retriever) -> None:
    hit = retriever.search("metformin AMPK glioblastoma", k=5, mode="hybrid_rerank").hits[0]
    assert hit.pmid == "1"
    s = hit.stages
    assert s.bm25_rank == 1 and s.dense_rank is not None
    assert s.rrf is not None and s.rerank_rank == 1
    assert hit.score == pytest.approx(s.rerank, abs=1e-3)


def test_retriever_year_filters(retriever: Retriever) -> None:
    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        hits = retriever.search(
            "glioblastoma temozolomide brain",
            k=5,
            mode=mode,  # type: ignore[arg-type]
            filters={"year_min": 2012, "year_max": 2019},
        ).hits
        assert hits and all(2012 <= (h.year or 0) <= 2019 for h in hits), mode


def test_retriever_validation(retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="mode"):
        retriever.search("x", mode="magic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        retriever.search("  ")
    with pytest.raises(ValueError, match="unsupported filters"):
        retriever.search("x", filters={"journal": "J"})


# --- build ----------------------------------------------------------------------------
def test_build_indexes_idempotent(tmp_path: Path) -> None:
    store = make_store()
    embedder = FakeEmbedder()
    client = make_client(None)
    first = build_indexes(
        store, bm25_path=tmp_path, embedder=embedder, chroma_client=client, log=lambda _: None
    )
    assert first.bm25_rebuilt and first.dense == {"embedded": 5, "deleted": 0, "unchanged": 0}
    second = build_indexes(
        store, bm25_path=tmp_path, embedder=embedder, chroma_client=client, log=lambda _: None
    )
    assert not second.bm25_rebuilt and second.dense == {"embedded": 0, "deleted": 0, "unchanged": 5}
    store.add_records(
        [PaperRecord(source="pubmed", pmid="6", title="New", abstract="AMPK.", year=2022)], "test"
    )
    third = build_indexes(
        store, bm25_path=tmp_path, embedder=embedder, chroma_client=client, log=lambda _: None
    )
    assert third.bm25_rebuilt and third.passages["added"] == 1
    assert third.dense == {"embedded": 1, "deleted": 0, "unchanged": 5}


# --- MCP tool -------------------------------------------------------------------------
async def test_search_corpus_tool(retriever: Retriever) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    from bioagent.mcp_servers.retrieval import server

    server.set_retriever(retriever)
    try:
        result = await server.search_corpus("rapamycin autophagy", k=2, mode="hybrid")
        assert result.hits[0].pmid == "5"
        with pytest.raises(ToolError, match="empty"):
            await server.search_corpus(" ")
        assert "search_corpus" in {t.name for t in await server.mcp.list_tools()}
    finally:
        server.set_retriever(None)
