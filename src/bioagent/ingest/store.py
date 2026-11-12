"""SQLite storage for the abstract corpus, ingestion checkpoints and passages."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bioagent.ingest.passages import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    Passage,
    make_passages,
)
from bioagent.mcp_servers.literature.models import PaperRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    pmid TEXT PRIMARY KEY,
    title TEXT,
    abstract TEXT NOT NULL,
    journal TEXT,
    year INTEGER,
    doi TEXT,
    pmcid TEXT,
    mesh TEXT NOT NULL DEFAULT '[]',
    source_query TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS records_year ON records(year);
CREATE TABLE IF NOT EXISTS query_checkpoints (
    name TEXT PRIMARY KEY,
    term TEXT NOT NULL,
    max_results INTEGER NOT NULL,
    total_hits INTEGER,
    ids_seen INTEGER,
    stored INTEGER,
    skipped INTEGER,
    errors INTEGER,
    completed INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS passages (
    passage_id TEXT PRIMARY KEY,
    pmid TEXT NOT NULL REFERENCES records(pmid),
    chunk_idx INTEGER NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    year INTEGER,
    journal TEXT,
    doi TEXT,
    content_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS passages_pmid ON passages(pmid);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass(frozen=True)
class Checkpoint:
    name: str
    term: str
    max_results: int
    total_hits: int | None
    ids_seen: int | None
    stored: int | None
    skipped: int | None
    errors: int | None
    completed: bool
    updated_at: str


@dataclass(frozen=True)
class PassageSync:
    added: int
    updated: int
    removed: int
    unchanged: int


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class CorpusStore:
    """Thin wrapper over SQLite. Use ``":memory:"`` for tests."""

    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # The MCP server calls the retriever from worker threads; Retriever.search serialises
        # access with a lock, so sharing one connection across threads is safe.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- records -------------------------------------------------------------------------
    def existing_pmids(self) -> set[str]:
        return {row[0] for row in self.conn.execute("SELECT pmid FROM records")}

    def add_records(self, records: Iterable[PaperRecord], source_query: str) -> int:
        """Insert records that are new; existing PMIDs are left untouched. Returns # inserted."""
        rows = [
            (
                r.pmid,
                r.title,
                r.abstract,
                r.journal,
                r.year,
                r.doi,
                r.pmcid,
                json.dumps(r.mesh_terms),
                source_query,
                _now(),
            )
            for r in records
            if r.pmid and r.abstract
        ]
        with self.conn:
            before = self.conn.total_changes
            self.conn.executemany(
                "INSERT OR IGNORE INTO records (pmid, title, abstract, journal, year, doi, pmcid,"
                " mesh, source_query, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return self.conn.total_changes - before

    def count_records(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def iter_records(self) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute("SELECT * FROM records ORDER BY pmid")

    def get_records(self, pmids: Iterable[str]) -> dict[str, dict[str, Any]]:
        ids = list(pmids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(f"SELECT * FROM records WHERE pmid IN ({marks})", ids)
        return {row["pmid"]: dict(row) for row in rows}

    # --- checkpoints ---------------------------------------------------------------------
    def get_checkpoint(self, name: str) -> Checkpoint | None:
        row = self.conn.execute(
            "SELECT * FROM query_checkpoints WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["completed"] = bool(data["completed"])
        return Checkpoint(**data)

    def set_checkpoint(
        self,
        name: str,
        term: str,
        max_results: int,
        *,
        total_hits: int,
        ids_seen: int,
        stored: int,
        skipped: int,
        errors: int,
        completed: bool,
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO query_checkpoints (name, term, max_results, total_hits, ids_seen,"
                " stored, skipped, errors, completed, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(name) DO UPDATE SET term=excluded.term,"
                " max_results=excluded.max_results, total_hits=excluded.total_hits,"
                " ids_seen=excluded.ids_seen, stored=excluded.stored, skipped=excluded.skipped,"
                " errors=excluded.errors, completed=excluded.completed,"
                " updated_at=excluded.updated_at",
                (
                    name,
                    term,
                    max_results,
                    total_hits,
                    ids_seen,
                    stored,
                    skipped,
                    errors,
                    int(completed),
                    _now(),
                ),
            )

    # --- passages ------------------------------------------------------------------------
    def build_passages(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ) -> list[Passage]:
        passages: list[Passage] = []
        for row in self.iter_records():
            passages.extend(
                make_passages(
                    row["pmid"],
                    row["title"],
                    row["abstract"],
                    year=row["year"],
                    journal=row["journal"],
                    doi=row["doi"],
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
        return passages

    def sync_passages(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ) -> PassageSync:
        """Regenerate passages from records and apply only the differences."""
        wanted = {p.passage_id: p for p in self.build_passages(max_tokens, overlap_tokens)}
        current = {
            row[0]: row[1]
            for row in self.conn.execute("SELECT passage_id, content_hash FROM passages")
        }
        to_write = [p for pid, p in wanted.items() if current.get(pid) != p.content_hash]
        removed = [pid for pid in current if pid not in wanted]
        added = sum(1 for p in to_write if p.passage_id not in current)
        with self.conn:
            self.conn.executemany(
                "INSERT INTO passages (passage_id, pmid, chunk_idx, title, text, year, journal,"
                " doi, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(passage_id) DO UPDATE SET text=excluded.text,"
                " title=excluded.title, year=excluded.year, journal=excluded.journal,"
                " doi=excluded.doi, content_hash=excluded.content_hash",
                [
                    (
                        p.passage_id,
                        p.pmid,
                        p.chunk_idx,
                        p.title,
                        p.text,
                        p.year,
                        p.journal,
                        p.doi,
                        p.content_hash,
                    )
                    for p in to_write
                ],
            )
            self.conn.executemany(
                "DELETE FROM passages WHERE passage_id = ?", [(pid,) for pid in removed]
            )
        return PassageSync(
            added=added,
            updated=len(to_write) - added,
            removed=len(removed),
            unchanged=len(wanted) - len(to_write),
        )

    def all_passages(self) -> list[Passage]:
        rows = self.conn.execute(
            "SELECT passage_id, pmid, chunk_idx, title, text, year, journal, doi FROM passages"
            " ORDER BY pmid, chunk_idx"
        )
        return [Passage(*tuple(row)) for row in rows]

    def get_passages(self, passage_ids: Iterable[str]) -> dict[str, Passage]:
        ids = list(passage_ids)
        out: dict[str, Passage] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                "SELECT passage_id, pmid, chunk_idx, title, text, year, journal, doi"
                f" FROM passages WHERE passage_id IN ({marks})",
                chunk,
            )
            out.update({row[0]: Passage(*tuple(row)) for row in rows})
        return out

    def passage_years(self) -> dict[str, int | None]:
        return {
            row[0]: row[1] for row in self.conn.execute("SELECT passage_id, year FROM passages")
        }

    def passage_fingerprint(self) -> str:
        """Digest of all passage ids + content hashes (sorted, so order-independent)."""
        h = hashlib.sha256()
        for row in self.conn.execute(
            "SELECT passage_id, content_hash FROM passages ORDER BY passage_id"
        ):
            h.update(f"{row[0]}\t{row[1]}\n".encode())
        return h.hexdigest()

    def stats(self) -> dict[str, Any]:
        q = self.conn.execute
        return {
            "records": self.count_records(),
            "passages": int(q("SELECT COUNT(*) FROM passages").fetchone()[0]),
            "multi_chunk_records": int(
                q(
                    "SELECT COUNT(*) FROM (SELECT pmid FROM passages GROUP BY pmid"
                    " HAVING COUNT(*) > 1)"
                ).fetchone()[0]
            ),
            "year_min": q("SELECT MIN(year) FROM records").fetchone()[0],
            "year_max": q("SELECT MAX(year) FROM records").fetchone()[0],
            "by_query": {
                row[0]: row[1]
                for row in q("SELECT source_query, COUNT(*) FROM records GROUP BY source_query")
            },
        }
