"""Build the knowledge graph from the local corpus (idempotent, resumable, cached).

Stages (default: all):
  fetch   PubTator3 annotations for every corpus PMID -> data/kg/pubtator/
  genes   NCBI Gene summaries for gene names -> data/kg/genes.json
  chembl  ChEMBL ids, max_phase and mechanisms for frequent drugs -> data/kg/chembl/
  load    aggregate and MERGE into Neo4j

Usage: uv run python scripts/build_kg.py [--stages fetch genes chembl load] [--fresh]
       [--chembl-cache-only]
(``make kg-build`` runs every stage with ``--fresh``: the graph is cleared, then reloaded.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from bioagent.common.httpclient import RateLimitedClient
from bioagent.config import get_settings
from bioagent.ingest.store import CorpusStore
from bioagent.kg.builder import GraphBuilder
from bioagent.kg.chembl import CHEMBL_HOST, CHEMBL_RATE, ChEMBLClient
from bioagent.kg.genes import fetch_gene_names
from bioagent.kg.pubtator import PUBTATOR_HOST, PUBTATOR_RATE, PubTatorFetcher
from bioagent.kg.store import GraphStore
from bioagent.mcp_servers.literature.pubmed import EUTILS_HOST, pubmed_rate

CHEMBL_MIN_PAPERS = 3

STAGES = ("fetch", "genes", "chembl", "load")


async def stage_fetch(pmids: list[str], cache: Path) -> dict[str, Any]:
    http = RateLimitedClient({PUBTATOR_HOST: PUBTATOR_RATE}, timeout=60.0)
    try:
        return await PubTatorFetcher(http, cache / "pubtator").fetch_all(pmids)
    finally:
        await http.aclose()


def aggregate(store: CorpusStore, cache: Path) -> tuple[GraphBuilder, dict[str, int]]:
    fetcher = PubTatorFetcher(RateLimitedClient({}), cache / "pubtator")
    builder = GraphBuilder()
    covered = 0
    for row in store.iter_records():
        doc = fetcher.load(row["pmid"])
        if doc is not None:
            covered += 1
            builder.add(doc, row["title"], row["year"])
    return builder, {"pubtator_covered": covered}


async def stage_genes(builder: GraphBuilder, cache: Path) -> dict[str, Any]:
    settings = get_settings()
    key = settings.ncbi_api_key.get_secret_value() if settings.ncbi_api_key else None
    http = RateLimitedClient({EUTILS_HOST: pubmed_rate(key)})
    ids = sorted({i for n in builder.nodes.values() if n.kind == "Gene" for i in n.ncbi_ids})
    try:
        names = await fetch_gene_names(
            http, ids, cache / "genes.json", api_key=key, email=settings.ncbi_email
        )
    finally:
        await http.aclose()
    return {"gene_ids": len(ids), "named": sum(1 for i in ids if names.get(i, {}).get("name"))}


async def stage_chembl(builder: GraphBuilder, cache: Path, cache_only: bool) -> dict[str, Any]:
    http = RateLimitedClient({CHEMBL_HOST: CHEMBL_RATE}, timeout=60.0)
    client = ChEMBLClient(http, cache / "chembl")
    drugs = builder.drug_ids_by_frequency(CHEMBL_MIN_PAPERS)
    matched = with_mech = approved = errors = uncached = 0
    mapping: dict[str, Any] = {}
    try:
        for n, (drug_id, name) in enumerate(drugs, start=1):
            info = await client.lookup(name, cache_only=cache_only)
            uncached += bool(info.get("uncached"))
            mapping[drug_id] = info
            matched += bool(info.get("chembl_id"))
            with_mech += bool(info.get("mechanisms"))
            approved += (info.get("max_phase") or 0) >= 4
            errors += "error" in info
            if n % 100 == 0:
                print(f"  chembl {n}/{len(drugs)} matched={matched} approved={approved}")
    finally:
        await http.aclose()
    (cache / "chembl_map.json").write_text(json.dumps(mapping))
    return {
        "drugs_queried": len(drugs),
        "matched": matched,
        "approved": approved,
        "with_mechanism": with_mech,
        "errors": errors,
        "uncached": uncached,
        "match_rate": round(matched / max(1, len(drugs)), 3),
    }


async def stage_load(builder: GraphBuilder, cache: Path, fresh: bool) -> dict[str, Any]:
    map_file = cache / "chembl_map.json"
    if not map_file.exists():
        raise SystemExit("data/kg/chembl_map.json is missing: run the chembl stage first")
    chembl: dict[str, dict[str, Any]] = json.loads(map_file.read_text())
    t0 = time.monotonic()
    rows = builder.build(chembl)
    report: dict[str, Any] = {
        "planned": rows.counts(),
        "aggregate_seconds": round(time.monotonic() - t0, 1),
    }
    async with GraphStore.from_settings(get_settings()) as store:
        await store.verify()
        if fresh:
            t0 = time.monotonic()
            await store.clear()
            report["clear_seconds"] = round(time.monotonic() - t0, 1)
        loaded = await store.load(rows)
        report["load_seconds"] = loaded["seconds"]
        report["in_graph"] = await store.counts()
    return report


async def main(
    stages: list[str], fresh: bool = False, chembl_cache_only: bool = False
) -> dict[str, Any]:
    settings = get_settings()
    cache = Path(settings.kg_cache_dir)
    store = CorpusStore(settings.corpus_db)
    pmids = sorted(store.existing_pmids())
    report: dict[str, Any] = {"corpus_pmids": len(pmids)}
    builder: GraphBuilder | None = None
    for stage in stages:
        t0 = time.monotonic()
        if stage == "fetch":
            report["fetch"] = await stage_fetch(pmids, cache)
        elif builder is None:
            builder, coverage = aggregate(store, cache)
            report["pubtator_coverage"] = round(coverage["pubtator_covered"] / len(pmids), 4)
        if stage != "fetch":
            assert builder is not None
        if stage == "genes":
            report["genes"] = await stage_genes(builder, cache)
        elif stage == "chembl":
            report["chembl"] = await stage_chembl(builder, cache, chembl_cache_only)
        elif stage == "load":
            report["load"] = await stage_load(builder, cache, fresh)
        report[f"{stage}_seconds"] = round(time.monotonic() - t0, 1)
        print(f"[{stage}] done in {report[f'{stage}_seconds']}s")
    store.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--fresh", action="store_true", help="clear the graph before loading")
    parser.add_argument(
        "--chembl-cache-only",
        action="store_true",
        help="use cached ChEMBL lookups only (uncached drugs get no ChEMBL data)",
    )
    args = parser.parse_args()
    report = asyncio.run(main(args.stages, args.fresh, args.chembl_cache_only))
    print(json.dumps(report, indent=2))
