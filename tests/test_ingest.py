from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bioagent.common.httpclient import UpstreamAPIError
from bioagent.ingest.passages import approx_tokens, make_passages, split_sentences
from bioagent.ingest.pipeline import (
    CorpusConfig,
    QuerySpec,
    build_term,
    ingest,
    load_corpus_config,
)
from bioagent.ingest.store import CorpusStore
from bioagent.mcp_servers.literature.models import PaperRecord


def rec(pmid: str, abstract: str | None = "Some abstract.", year: int | None = 2020) -> PaperRecord:
    return PaperRecord(
        source="pubmed",
        pmid=pmid,
        title=f"Title {pmid}",
        abstract=abstract,
        year=year,
        journal="J",
        mesh_terms=["Metformin"],
    )


# --- passages -------------------------------------------------------------------------
def test_split_sentences_keeps_abbreviations() -> None:
    text = "Metformin works, e.g. via AMPK. It reduces glucose! Does it help GBM? (Yes) 2 trials."
    assert split_sentences(text) == [
        "Metformin works, e.g. via AMPK.",
        "It reduces glucose!",
        "Does it help GBM?",
        "(Yes) 2 trials.",
    ]


def test_short_record_is_one_passage() -> None:
    [p] = make_passages("123", "A title", "Short abstract.", year=2021)
    assert p.passage_id == "123:0" and p.text == "A title\nShort abstract." and p.year == 2021


def test_long_record_chunked_with_overlap() -> None:
    sentences = [
        f"Sentence number {i} talks about AMPK and mTOR signalling in cells." for i in range(60)
    ]
    passages = make_passages(
        "9", "Title here", " ".join(sentences), max_tokens=120, overlap_tokens=30
    )
    assert len(passages) > 3
    assert [p.passage_id for p in passages] == [f"9:{i}" for i in range(len(passages))]
    for p in passages:
        assert p.text.startswith("Title here\n")
        assert approx_tokens(p.text) <= 120 + 15  # one sentence of slack
    # consecutive chunks share their boundary sentence(s)
    first_tail = passages[0].text.split(". ")[-1]
    assert first_tail.rstrip(".") in passages[1].text
    # every sentence is covered
    joined = " ".join(p.text for p in passages)
    assert all(s in joined for s in sentences)


def test_content_hash_changes_with_text() -> None:
    a = make_passages("1", "T", "Alpha.")[0]
    b = make_passages("1", "T", "Beta.")[0]
    assert a.content_hash != b.content_hash


# --- store ----------------------------------------------------------------------------
def test_store_idempotent_inserts() -> None:
    store = CorpusStore(":memory:")
    assert store.add_records([rec("1"), rec("2"), rec("3", abstract=None)], "q1") == 2
    assert store.add_records([rec("2"), rec("4")], "q2") == 1
    assert store.count_records() == 3
    row = store.get_records(["2"])["2"]
    assert row["source_query"] == "q1"  # first query wins; not overwritten
    assert row["mesh"] == '["Metformin"]'


def test_passage_sync_is_incremental() -> None:
    store = CorpusStore(":memory:")
    store.add_records([rec("1"), rec("2")], "q")
    first = store.sync_passages()
    assert (first.added, first.updated, first.removed) == (2, 0, 0)
    again = store.sync_passages()
    assert (again.added, again.updated, again.unchanged) == (0, 0, 2)
    fp = store.passage_fingerprint()
    store.conn.execute("UPDATE records SET abstract = 'Changed.' WHERE pmid = '1'")
    store.conn.execute("DELETE FROM passages WHERE pmid = '2'")
    changed = store.sync_passages()
    assert (changed.added, changed.updated) == (1, 1)
    assert store.passage_fingerprint() != fp


def test_store_on_disk_persists(tmp_path: Path) -> None:
    path = tmp_path / "c" / "corpus.sqlite"
    store = CorpusStore(path)
    store.add_records([rec("1")], "q")
    store.close()
    assert CorpusStore(path).existing_pmids() == {"1"}


# --- pipeline -------------------------------------------------------------------------
class FakePubMed:
    def __init__(self, ids: dict[str, list[str]], fail_batches: int = 0) -> None:
        self.ids = ids
        self.fail_batches = fail_batches
        self.fetched: list[list[str]] = []
        self.terms: list[str] = []

    async def esearch(self, term: str, **kw: Any) -> dict[str, Any]:
        self.terms.append(term)
        key = next(k for k in self.ids if k in term)
        ids = self.ids[key][: kw["retmax"]]
        return {"count": str(len(self.ids[key]) * 10), "idlist": ids}

    async def efetch_ids(self, pmids: list[str]) -> list[PaperRecord]:
        if self.fail_batches:
            self.fail_batches -= 1
            raise UpstreamAPIError("boom")
        self.fetched.append(list(pmids))
        return [
            rec(p, abstract=None if p == "x0" else "Abs.", year=1999 if p == "old" else 2020)
            for p in pmids
        ]


def config(cap: int = 300) -> CorpusConfig:
    return CorpusConfig(
        queries=[QuerySpec("a", "alpha", cap), QuerySpec("b", "beta", cap)], min_year=2000
    )


async def test_ingest_dedupes_and_filters() -> None:
    store = CorpusStore(":memory:")
    pubmed = FakePubMed({"alpha": ["1", "2", "x0", "old"], "beta": ["2", "3"]})
    reports = await ingest(config(), store, pubmed, log=lambda _: None)  # type: ignore[arg-type]
    assert store.existing_pmids() == {"1", "2", "3"}
    assert reports[0].stored == 2 and reports[0].skipped_no_abstract_or_year == 2
    assert reports[1].already_stored == 1 and reports[1].stored == 1
    assert pubmed.fetched[1] == ["3"]  # PMID 2 not refetched
    assert all("2000:3000[pdat]" in t and "hasabstract" in t for t in pubmed.terms)


async def test_ingest_resumes_after_failure() -> None:
    store = CorpusStore(":memory:")
    ids = {"alpha": [str(i) for i in range(450)], "beta": ["b1"]}
    failing = FakePubMed(ids, fail_batches=1)  # first batch of 200 fails
    reports = await ingest(config(cap=1000), store, failing, log=lambda _: None)  # type: ignore[arg-type]
    assert reports[0].status == "partial" and store.count_records() == 251
    cp = store.get_checkpoint("a")
    assert cp is not None and not cp.completed and cp.errors == 1

    healthy = FakePubMed(ids)
    reports = await ingest(config(cap=1000), store, healthy, log=lambda _: None)  # type: ignore[arg-type]
    assert reports[0].status == "completed" and reports[1].status == "skipped"
    assert healthy.fetched == [[str(i) for i in range(200)]]  # only the missing batch
    assert store.count_records() == 451

    again = await ingest(config(cap=1000), store, FakePubMed(ids), log=lambda _: None)  # type: ignore[arg-type]
    assert [r.status for r in again] == ["skipped", "skipped"]


async def test_changing_cap_reruns_query() -> None:
    store = CorpusStore(":memory:")
    ids = {"alpha": ["1", "2", "3"], "beta": ["4"]}
    await ingest(config(cap=2), store, FakePubMed(ids), log=lambda _: None)  # type: ignore[arg-type]
    reports = await ingest(config(cap=3), store, FakePubMed(ids), log=lambda _: None)  # type: ignore[arg-type]
    assert reports[0].status == "completed" and reports[0].stored == 1


def test_load_corpus_config(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "defaults: {min_year: 2005, max_results: 50}\n"
        "queries:\n  - {name: a, query: 'AMPK[tiab]'}\n"
        "  - {name: b, query: x, max_results: 20000}\n"
    )
    cfg = load_corpus_config(path)
    assert cfg.min_year == 2005
    assert [q.max_results for q in cfg.queries] == [50, 10_000]  # capped at PubMed's limit
    assert build_term(cfg.queries[0], cfg) == "(AMPK[tiab]) AND 2005:3000[pdat] AND hasabstract"
    path.write_text("queries:\n  - {name: a, query: x}\n  - {name: a, query: y}\n")
    with pytest.raises(ValueError, match="unique"):
        load_corpus_config(path)


def test_repo_corpus_config_is_valid() -> None:
    cfg = load_corpus_config(Path(__file__).parents[1] / "configs" / "corpus.yaml")
    assert len(cfg.queries) == 9 and cfg.min_year == 2000
