"""Source data for the eval set and the isolated eval index.

* PubMedQA ``pqa_labeled`` (HF ``qiaojin/PubMedQA``, MIT) via the HF datasets-server JSON
  API (no extra dependencies), cached under ``data/eval/raw/``. The context excludes the
  abstract's conclusion, as in the dataset.
* ``build_fixture`` builds the eval retrieval corpus: every gold PMID of the eval set, the
  poisoned passages of user-provided injection items (only ever in this index) and fixed
  distractors. Abstract text
  comes from the PubMedQA cache, the local corpus DB, or PubMed efetch (CI), in that order.
  It is indexed with BM25 only, so CI needs no models and no Neo4j.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import httpx

from bioagent.eval.dataset import EvalItem, PoisonedPassage, poisoned_passages
from bioagent.ingest.store import CorpusStore
from bioagent.mcp_servers.literature.models import PaperRecord

HF_ROWS = "https://datasets-server.huggingface.co/rows"
PUBMEDQA = {"dataset": "qiaojin/PubMedQA", "config": "pqa_labeled", "split": "train"}
PUBMEDQA_SOURCE = "pubmedqa"
POISONED_SOURCE = "eval-poisoned"
ROWS_PER_REQUEST = 100


def fetch_pubmedqa(cache: Path, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """All 1000 pqa_labeled rows (cached as JSON)."""
    if cache.exists():
        rows: list[dict[str, Any]] = json.loads(cache.read_text())
        return rows
    rows = []
    with httpx.Client(timeout=60.0) as client:
        offset = 0
        while True:
            params: dict[str, str | int] = {
                **PUBMEDQA,
                "offset": offset,
                "length": ROWS_PER_REQUEST,
            }
            response = client.get(HF_ROWS, params=params)
            response.raise_for_status()
            data = response.json()
            batch = [r["row"] for r in data["rows"]]
            rows.extend(batch)
            offset += len(batch)
            if not batch or offset >= int(data.get("num_rows_total", 0)):
                break
    log(f"pubmedqa: fetched {len(rows)} rows")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def pubmedqa_record(row: dict[str, Any]) -> PaperRecord:
    contexts = row["context"]["contexts"]
    labels = row["context"].get("labels") or []
    parts = [
        f"{label.title()}: {text}" if label else text
        for label, text in zip(labels or [None] * len(contexts), contexts, strict=False)
    ]
    return PaperRecord(
        source="pubmed",
        pmid=str(row["pubid"]),
        title=None,
        abstract=" ".join(parts),
        mesh_terms=list(row["context"].get("meshes") or []),
    )


def sample_pubmedqa(rows: Sequence[dict[str, Any]], n: int, seed: int = 7) -> list[dict[str, Any]]:
    """Stratified by label, proportional to the dataset (552 yes / 338 no / 110 maybe)."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: str(r["pubid"])):
        by_label.setdefault(row["final_decision"], []).append(row)
    total = len(rows)
    picked: list[dict[str, Any]] = []
    for _label, group in sorted(by_label.items()):
        k = round(n * len(group) / total)
        picked += rng.sample(group, min(k, len(group)))
    return sorted(picked, key=lambda r: str(r["pubid"]))[:n]


def ingest_records(store: CorpusStore, records: Iterable[PaperRecord], source: str) -> int:
    return store.add_records(list(records), source_query=source)


def poisoned_records(passages: Sequence[PoisonedPassage]) -> list[PaperRecord]:
    return [
        PaperRecord(source="pubmed", pmid=p.pmid, title=p.title, abstract=p.abstract, year=p.year)
        for p in passages
    ]


async def _efetch(pmids: Sequence[str]) -> list[PaperRecord]:
    from bioagent.common.httpclient import RateLimitedClient
    from bioagent.config import get_settings
    from bioagent.mcp_servers.literature.pubmed import EUTILS_HOST, PubMedClient, pubmed_rate

    settings = get_settings()
    key = settings.ncbi_api_key.get_secret_value() if settings.ncbi_api_key else None
    http = RateLimitedClient({EUTILS_HOST: pubmed_rate(key)}, timeout=60.0)
    try:
        client = PubMedClient(http, api_key=key, email=settings.ncbi_email)
        return await client.efetch_ids(list(pmids))
    finally:
        await http.aclose()


def fixture_pmids(items: Sequence[EvalItem], distractors: Sequence[str]) -> list[str]:
    gold = [p for item in items for p in item.gold_pmids]
    return sorted(dict.fromkeys([*gold, *distractors]))


def build_fixture(
    out_dir: Path,
    items: Sequence[EvalItem],
    distractors: Sequence[str],
    *,
    pubmedqa_rows: Sequence[dict[str, Any]] = (),
    corpus_db: Path | None = None,
    allow_network: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """(Re)build ``out_dir/corpus.sqlite`` and ``out_dir/bm25`` for the eval profile."""
    from bioagent.retrieval.build import build_indexes

    wanted = fixture_pmids(items, distractors)
    records: dict[str, PaperRecord] = {}
    qa = {str(r["pubid"]): r for r in pubmedqa_rows}
    for pmid in wanted:
        if pmid in qa:
            records[pmid] = pubmedqa_record(qa[pmid])
    if corpus_db is not None and corpus_db.exists():
        main = CorpusStore(corpus_db)
        try:
            for pmid, row in main.get_records([p for p in wanted if p not in records]).items():
                records[pmid] = PaperRecord(
                    source="pubmed",
                    pmid=pmid,
                    title=row["title"],
                    abstract=row["abstract"],
                    journal=row["journal"],
                    year=row["year"],
                    doi=row["doi"],
                )
        finally:
            main.close()
    missing = [p for p in wanted if p not in records]
    if missing and allow_network:
        log(f"fixture: fetching {len(missing)} abstracts from PubMed")
        for rec in asyncio.run(_efetch(missing)):
            if rec.pmid and rec.abstract:
                records[rec.pmid] = rec
        missing = [p for p in wanted if p not in records]
    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "corpus.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)
    store = CorpusStore(db)
    try:
        n_real = store.add_records(records.values(), "eval-fixture")
        n_poison = store.add_records(poisoned_records(poisoned_passages(items)), POISONED_SOURCE)
        report = build_indexes(
            store, bm25_path=out_dir / "bm25", embedder=None, chroma_client=None, log=log
        )
    finally:
        store.close()
    summary = {
        "records": n_real,
        "poisoned": n_poison,
        "missing_pmids": missing,
        "passages": report.passages,
    }
    (out_dir / "manifest.json").write_text(json.dumps({**summary, "pmids": wanted}, indent=1))
    return summary
