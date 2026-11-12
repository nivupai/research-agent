"""Tests that load real sentence-transformers models (downloads ~200 MB on first run).

Run with ``make test-models``; excluded from CI.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def models() -> tuple[object, object]:
    pytest.importorskip("sentence_transformers")
    from bioagent.config import get_settings
    from bioagent.retrieval.embedder import SentenceTransformerEmbedder
    from bioagent.retrieval.rerank import CrossEncoderReranker

    settings = get_settings()
    return (
        SentenceTransformerEmbedder(settings.embed_model, device=settings.embed_device),
        CrossEncoderReranker(settings.rerank_model, device=settings.embed_device),
    )


def test_embedder_normalised_and_prefixed(models: tuple[object, object]) -> None:
    from bioagent.retrieval.embedder import SentenceTransformerEmbedder

    embedder = models[0]
    assert isinstance(embedder, SentenceTransformerEmbedder)
    [v] = embedder.embed_documents(["Metformin activates AMPK."])
    assert abs(sum(x * x for x in v) - 1.0) < 1e-3
    assert embedder.query_instruction.startswith("Represent this sentence")
    q = embedder.embed_query("does metformin activate AMPK?")
    related, unrelated = embedder.embed_documents(
        [
            "Metformin activates AMP-activated protein kinase in hepatocytes.",
            "The Treaty of Westphalia was signed in 1648.",
        ]
    )
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    assert dot(q, related) > dot(q, unrelated) + 0.1


def test_reranker_prefers_relevant(models: tuple[object, object]) -> None:
    from bioagent.retrieval.rerank import CrossEncoderReranker

    reranker = models[1]
    assert isinstance(reranker, CrossEncoderReranker)
    relevant, off_topic = reranker.score(
        "temozolomide resistance MGMT",
        [
            "MGMT expression confers resistance to temozolomide in glioblastoma.",
            "Statins lower LDL cholesterol in cardiovascular disease.",
        ],
    )
    assert relevant > off_topic


def test_real_retriever_on_built_index() -> None:
    from pathlib import Path

    from bioagent.config import get_settings
    from bioagent.retrieval.retriever import retriever_from_settings

    settings = get_settings()
    if not Path(settings.corpus_db).exists() or not (Path(settings.index_dir) / "bm25").exists():
        pytest.skip("corpus/indexes not built (make ingest && make index)")
    results = retriever_from_settings(settings).search(
        "temozolomide resistance MGMT", k=5, mode="hybrid_rerank"
    )
    assert len(results.hits) == 5
    assert any("MGMT" in h.text or "O6-methylguanine" in h.text for h in results.hits)
