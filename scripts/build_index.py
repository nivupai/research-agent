"""Build or refresh the retrieval indexes from data/corpus/corpus.sqlite (idempotent).

Usage: uv run python scripts/build_index.py [--embed-model NAME] [--bm25-only]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

from bioagent.config import get_settings
from bioagent.ingest.store import CorpusStore
from bioagent.retrieval.build import bm25_dir, build_indexes, chroma_dir
from bioagent.retrieval.dense import make_client
from bioagent.retrieval.embedder import Embedder, SentenceTransformerEmbedder


def du(path: str) -> str:
    out = subprocess.run(["du", "-sh", path], capture_output=True, text=True, check=False)
    return out.stdout.split("\t")[0] if out.returncode == 0 else "?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-model", help="override EMBED_MODEL")
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    store = CorpusStore(settings.corpus_db)
    started = time.monotonic()
    embedder: Embedder | None = None
    client = None
    if not args.bm25_only:
        model = args.embed_model or settings.embed_model
        t0 = time.monotonic()
        st = SentenceTransformerEmbedder(
            model, device=settings.embed_device, batch_size=settings.embed_batch_size
        )
        print(f"loaded {model} on {st.device} in {time.monotonic() - t0:.1f}s")
        embedder = st
        client = make_client(chroma_dir(settings))
    report = build_indexes(
        store, bm25_path=bm25_dir(settings), embedder=embedder, chroma_client=client
    )
    stats = store.stats()
    store.close()
    summary = {
        **report.as_dict(),
        "records": stats["records"],
        "passages_total": stats["passages"],
        "multi_chunk_records": stats["multi_chunk_records"],
        "seconds": round(time.monotonic() - started, 1),
        "disk": {
            "corpus": du(str(settings.corpus_db)),
            "bm25": du(str(bm25_dir(settings))),
            "chroma": du(str(chroma_dir(settings))),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
