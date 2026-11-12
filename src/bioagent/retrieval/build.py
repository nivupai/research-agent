"""Build/refresh the passage table, the BM25 index and the dense index. Idempotent.

* passages: regenerated from records; only differences are written
* BM25: rebuilt only when the passage fingerprint changed (fast, < a few seconds)
* dense: only new/changed passages are embedded; removed ones are deleted
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bioagent.config import Settings
from bioagent.ingest.store import CorpusStore
from bioagent.retrieval.bm25 import BM25Index
from bioagent.retrieval.dense import DenseIndex
from bioagent.retrieval.embedder import Embedder


@dataclass
class BuildReport:
    passages: dict[str, int]
    bm25_rebuilt: bool
    bm25_seconds: float
    dense: dict[str, int] | None
    dense_seconds: float
    dense_model: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def bm25_dir(settings: Settings) -> Path:
    return Path(settings.index_dir) / "bm25"


def chroma_dir(settings: Settings) -> Path:
    return Path(settings.index_dir) / "chroma"


def build_indexes(
    store: CorpusStore,
    *,
    bm25_path: Path,
    embedder: Embedder | None,
    chroma_client: Any | None,
    log: Callable[[str], None] = print,
) -> BuildReport:
    sync = store.sync_passages()
    log(f"passages: {asdict(sync)}")

    fingerprint = store.passage_fingerprint()
    t0 = time.monotonic()
    rebuilt = BM25Index.stored_fingerprint(bm25_path) != fingerprint
    if rebuilt:
        passages = store.all_passages()
        BM25Index.build([(p.passage_id, p.text) for p in passages], fingerprint).save(bm25_path)
        log(f"bm25: rebuilt over {len(passages)} passages")
    else:
        log("bm25: up to date")
    bm25_seconds = round(time.monotonic() - t0, 2)

    dense_report: dict[str, int] | None = None
    dense_seconds = 0.0
    if embedder is not None and chroma_client is not None:
        t0 = time.monotonic()
        index = DenseIndex(chroma_client, embedder)
        result = index.sync(store.all_passages(), log=log)
        dense_report = asdict(result)
        dense_seconds = round(time.monotonic() - t0, 2)
        log(f"dense ({embedder.model_name}): {dense_report} in {dense_seconds}s")

    return BuildReport(
        passages=asdict(sync),
        bm25_rebuilt=rebuilt,
        bm25_seconds=bm25_seconds,
        dense=dense_report,
        dense_seconds=dense_seconds,
        dense_model=embedder.model_name if embedder is not None else None,
    )
