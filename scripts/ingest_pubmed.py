"""Ingest the PubMed corpus defined in configs/corpus.yaml into data/corpus/corpus.sqlite.

Resumable and idempotent: completed queries are skipped and stored PMIDs are never refetched.
Usage: uv run python scripts/ingest_pubmed.py [--config configs/corpus.yaml] [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from bioagent.config import get_settings
from bioagent.ingest.pipeline import QueryReport, ingest, load_corpus_config
from bioagent.ingest.store import CorpusStore
from bioagent.mcp_servers.literature.server import LiteratureClients


async def run(config_path: Path, store: CorpusStore, force: bool) -> list[QueryReport]:
    clients = LiteratureClients.from_settings(get_settings())
    try:
        return await ingest(load_corpus_config(config_path), store, clients.pubmed, force=force)
    finally:
        await clients.http.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corpus.yaml"))
    parser.add_argument("--force", action="store_true", help="re-run completed queries")
    args = parser.parse_args()

    settings = get_settings()
    store = CorpusStore(settings.corpus_db)
    started = time.monotonic()
    try:
        reports = asyncio.run(run(args.config, store, args.force))
        stats = store.stats()
    finally:
        store.close()
    summary = {
        "seconds": round(time.monotonic() - started, 1),
        "records": stats["records"],
        "by_query": stats["by_query"],
        "year_range": [stats["year_min"], stats["year_max"]],
        "db_bytes": os.path.getsize(settings.corpus_db),
        "queries": [r.__dict__ for r in reports],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
