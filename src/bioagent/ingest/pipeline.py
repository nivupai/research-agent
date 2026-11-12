"""Corpus ingestion: run the configured PubMed queries and store new abstracts.

Uses the literature server's ``PubMedClient`` (same rate limiting and NCBI key). For each
query, esearch returns up to ``max_results`` PMIDs ranked by relevance; PMIDs already in the
store are skipped, the rest are fetched with efetch in batches of 200. A query is
checkpointed as completed only when every batch succeeded, so re-running resumes cleanly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from bioagent.common.httpclient import UpstreamAPIError
from bioagent.ingest.store import CorpusStore
from bioagent.mcp_servers.literature.pubmed import (
    MAX_FETCH_BATCH,
    PUBMED_RETRIEVAL_LIMIT,
    PubMedClient,
)


@dataclass(frozen=True)
class QuerySpec:
    name: str
    query: str
    max_results: int


@dataclass(frozen=True)
class CorpusConfig:
    queries: list[QuerySpec]
    min_year: int = 2000
    require_abstract: bool = True


def load_corpus_config(path: str | Path) -> CorpusConfig:
    data = yaml.safe_load(Path(path).read_text())
    defaults = data.get("defaults") or {}
    default_cap = int(defaults.get("max_results", 1000))
    queries = [
        QuerySpec(
            name=str(q["name"]),
            query=str(q["query"]).strip(),
            max_results=min(int(q.get("max_results", default_cap)), PUBMED_RETRIEVAL_LIMIT),
        )
        for q in data["queries"]
    ]
    names = [q.name for q in queries]
    if len(set(names)) != len(names):
        raise ValueError("query names in corpus config must be unique")
    return CorpusConfig(
        queries=queries,
        min_year=int(defaults.get("min_year", 2000)),
        require_abstract=bool(defaults.get("require_abstract", True)),
    )


def build_term(spec: QuerySpec, config: CorpusConfig) -> str:
    term = f"({spec.query}) AND {config.min_year}:3000[pdat]"
    if config.require_abstract:
        term += " AND hasabstract"
    return term


@dataclass
class QueryReport:
    name: str
    status: str  # "skipped" | "completed" | "partial"
    total_hits: int = 0
    ids_seen: int = 0
    already_stored: int = 0
    stored: int = 0
    skipped_no_abstract_or_year: int = 0
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0


async def ingest(
    config: CorpusConfig,
    store: CorpusStore,
    pubmed: PubMedClient,
    *,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[QueryReport]:
    reports: list[QueryReport] = []
    for spec in config.queries:
        term = build_term(spec, config)
        checkpoint = store.get_checkpoint(spec.name)
        if (
            not force
            and checkpoint is not None
            and checkpoint.completed
            and checkpoint.term == term
            and checkpoint.max_results == spec.max_results
        ):
            log(f"[{spec.name}] already completed; skipping")
            reports.append(QueryReport(spec.name, "skipped"))
            continue

        started = time.monotonic()
        report = QueryReport(spec.name, "completed")
        result = await pubmed.esearch(term, retmax=spec.max_results, sort="relevance")
        ids = list(result.get("idlist", []))
        report.total_hits = int(result.get("count", 0))
        report.ids_seen = len(ids)
        existing = store.existing_pmids()
        missing = [pmid for pmid in ids if pmid not in existing]
        report.already_stored = len(ids) - len(missing)
        log(f"[{spec.name}] {report.total_hits} hits, {len(ids)} ids, {len(missing)} new to fetch")
        for start in range(0, len(missing), MAX_FETCH_BATCH):
            batch = missing[start : start + MAX_FETCH_BATCH]
            try:
                records = await pubmed.efetch_ids(batch)
            except UpstreamAPIError as exc:
                report.errors.append(f"batch {start // MAX_FETCH_BATCH}: {exc}")
                log(f"[{spec.name}] batch at {start} failed: {exc}")
                continue
            keep = [
                r
                for r in records
                if r.abstract and r.year is not None and r.year >= config.min_year
            ]
            report.skipped_no_abstract_or_year += len(records) - len(keep)
            report.stored += store.add_records(keep, source_query=spec.name)
        if report.errors:
            report.status = "partial"
        report.seconds = round(time.monotonic() - started, 1)
        store.set_checkpoint(
            spec.name,
            term,
            spec.max_results,
            total_hits=report.total_hits,
            ids_seen=report.ids_seen,
            stored=report.stored,
            skipped=report.skipped_no_abstract_or_year,
            errors=len(report.errors),
            completed=not report.errors,
        )
        log(
            f"[{spec.name}] stored {report.stored} new records in {report.seconds}s "
            f"({len(report.errors)} failed batches)"
        )
        reports.append(report)
    return reports
