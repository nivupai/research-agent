"""Dense index: passage embeddings in a persistent Chroma collection.

One collection per embedding model (``passages__<model-slug>``), cosine space. Each vector
stores ``pmid``, ``year``, ``journal`` and the passage ``content_hash`` so ``sync`` only
embeds new or changed passages and deletes removed ones.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bioagent.ingest.passages import Passage
from bioagent.retrieval.embedder import Embedder

UPSERT_BATCH = 256


def collection_name(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-._")
    return f"passages__{slug}"[:500]


@dataclass(frozen=True)
class DenseSync:
    embedded: int
    deleted: int
    unchanged: int


def make_client(path: Path | None) -> Any:
    import chromadb
    from chromadb.config import Settings

    settings = Settings(anonymized_telemetry=False)
    if path is None:
        return chromadb.EphemeralClient(settings=settings)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=settings)


class DenseIndex:
    def __init__(self, client: Any, embedder: Embedder) -> None:
        self.embedder = embedder
        self.collection = client.get_or_create_collection(
            collection_name(embedder.model_name),
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embed_model": embedder.model_name},
            embedding_function=None,
        )

    def count(self) -> int:
        return int(self.collection.count())

    def stored_hashes(self) -> dict[str, str]:
        out: dict[str, str] = {}
        offset = 0
        while True:
            page = self.collection.get(include=["metadatas"], limit=5000, offset=offset)
            ids = page["ids"]
            if not ids:
                return out
            for pid, meta in zip(ids, page["metadatas"], strict=True):
                out[pid] = str((meta or {}).get("content_hash", ""))
            offset += len(ids)

    def sync(self, passages: Sequence[Passage], log: Any = None) -> DenseSync:
        wanted = {p.passage_id: p for p in passages}
        stored = self.stored_hashes()
        stale = [pid for pid in stored if pid not in wanted]
        todo = [p for pid, p in wanted.items() if stored.get(pid) != p.content_hash]
        for start in range(0, len(stale), 5000):
            self.collection.delete(ids=stale[start : start + 5000])
        for start in range(0, len(todo), UPSERT_BATCH):
            batch = todo[start : start + UPSERT_BATCH]
            vectors = self.embedder.embed_documents([p.text for p in batch])
            self.collection.upsert(
                ids=[p.passage_id for p in batch],
                embeddings=vectors,
                metadatas=[
                    {
                        "pmid": p.pmid,
                        "year": p.year if p.year is not None else -1,
                        "journal": p.journal or "",
                        "content_hash": p.content_hash,
                    }
                    for p in batch
                ],
            )
            if log is not None:
                log(f"  embedded {min(start + UPSERT_BATCH, len(todo))}/{len(todo)}")
        return DenseSync(embedded=len(todo), deleted=len(stale), unchanged=len(wanted) - len(todo))

    def search(
        self, query: str, n: int = 100, year_min: int | None = None, year_max: int | None = None
    ) -> list[tuple[str, float]]:
        total = self.count()
        if total == 0:
            return []
        clauses: list[dict[str, Any]] = []
        if year_min is not None:
            clauses.append({"year": {"$gte": year_min}})
        if year_max is not None:
            clauses.append({"year": {"$lte": year_max}})
            if year_min is None:
                clauses.append({"year": {"$gte": 0}})  # exclude unknown years (-1)
        where: dict[str, Any] | None = None
        if len(clauses) == 1:
            where = clauses[0]
        elif clauses:
            where = {"$and": clauses}
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(n, total),
            where=where,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [(pid, 1.0 - float(d)) for pid, d in zip(ids, distances, strict=True)]
