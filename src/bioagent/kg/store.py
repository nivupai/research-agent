"""Async Neo4j access for the knowledge graph: schema, batch loading and capped reads.

Writes use ``UNWIND $rows`` templates from :mod:`bioagent.kg.queries`; every value travels
as a query parameter. Reads run in read-only transactions with a server-side timeout (plus
a client-side guard) and never return more than ``max_records`` rows.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
    AsyncManagedTransaction,
    Record,
    unit_of_work,
)
from neo4j.exceptions import Neo4jError
from neo4j.graph import Node, Path, Relationship

from bioagent.config import Settings
from bioagent.kg import queries as q
from bioagent.kg.builder import GraphRows

DEFAULT_BATCH = 2000
DEFAULT_MAX_RECORDS = 1000


class GraphUnavailableError(RuntimeError):
    """Neo4j is not configured or cannot be reached."""


def to_plain(value: Any) -> Any:
    """Convert driver values (nodes, relationships, paths) into JSON-friendly data."""
    if isinstance(value, Node):
        return {**dict(value), "labels": sorted(value.labels)}
    if isinstance(value, Relationship):
        return {**dict(value), "type": value.type}
    if isinstance(value, Path):
        return {
            "nodes": [to_plain(n) for n in value.nodes],
            "relationships": [to_plain(r) for r in value.relationships],
        }
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_plain(v) for v in value]
    return value


def to_plain_record(record: Record) -> dict[str, Any]:
    # Record.data() would turn nodes into bare property dicts and drop their labels.
    keys: list[str] = list(record.keys())
    return {k: to_plain(record[k]) for k in keys}


def edge_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Builder rows -> ``{id, source, target, props}`` for the edge upsert template."""
    return [
        {
            "id": r["id"],
            "source": r["source"],
            "target": r["target"],
            "props": {k: v for k, v in r.items() if k not in ("source", "target")},
        }
        for r in rows
    ]


def batches(rows: Sequence[Any], size: int) -> list[Sequence[Any]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [rows[i : i + size] for i in range(0, len(rows), size)]


class GraphStore:
    """Thin wrapper around an ``AsyncDriver``; use as an async context manager."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str | None = None,
        query_timeout_s: float = 10.0,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> None:
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.query_timeout_s = query_timeout_s
        self.max_records = max_records

    @classmethod
    def from_settings(cls, settings: Settings) -> GraphStore:
        if settings.neo4j_password is None:
            raise GraphUnavailableError("NEO4J_PASSWORD is not set (see .env.example)")
        return cls(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password.get_secret_value(),
            query_timeout_s=settings.neo4j_query_timeout_s,
        )

    async def __aenter__(self) -> GraphStore:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._driver.close()

    async def verify(self) -> None:
        try:
            await self._driver.verify_connectivity()
        except Exception as exc:  # ServiceUnavailable, AuthError, ...
            raise GraphUnavailableError(f"cannot reach Neo4j: {type(exc).__name__}") from exc

    # --- writes ------------------------------------------------------------------------
    async def run_write(self, query: str, params: Mapping[str, Any] | None = None) -> None:
        async def work(tx: AsyncManagedTransaction) -> None:
            result = await tx.run(query, dict(params or {}))
            await result.consume()

        async with self._driver.session(database=self.database) as session:
            await session.execute_write(work)

    async def ensure_schema(self) -> int:
        stmts = q.schema_statements()
        for stmt in stmts:
            await self.run_write(stmt)
        await self.run_write("CALL db.awaitIndexes(300)")
        return len(stmts)

    async def write_rows(
        self, query: str, rows: Sequence[Mapping[str, Any]], batch_size: int = DEFAULT_BATCH
    ) -> int:
        for chunk in batches(rows, batch_size):
            await self.run_write(query, {"rows": list(chunk)})
        return len(rows)

    async def clear(self) -> None:
        """Delete every node and relationship (batched, so large graphs fit in memory)."""
        while True:
            rows = await self.read(
                "MATCH (n) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS n",
                write=True,
            )
            if not rows or rows[0]["n"] == 0:
                return

    async def load(
        self,
        rows: GraphRows,
        *,
        batch_size: int = DEFAULT_BATCH,
        log: Callable[[str], None] = print,
    ) -> dict[str, Any]:
        """MERGE every node and edge; returns written row counts and per-step seconds."""
        timings: dict[str, float] = {}
        written: dict[str, int] = {}

        async def step(name: str, query: str, data: Sequence[Mapping[str, Any]]) -> None:
            t0 = time.monotonic()
            written[name] = await self.write_rows(query, data, batch_size)
            timings[name] = round(time.monotonic() - t0, 2)
            log(f"  loaded {name}: {written[name]} rows in {timings[name]}s")

        t0 = time.monotonic()
        await self.ensure_schema()
        timings["schema"] = round(time.monotonic() - t0, 2)
        for label, data in rows.nodes.items():
            await step(label, q.upsert_nodes(label), data)
        await step("Paper", q.UPSERT_PAPERS, rows.papers)
        await step("MENTIONS", q.UPSERT_MENTIONS, rows.mentions)
        for rel_type, data in sorted(rows.relations.items()):
            await step(rel_type, q.upsert_edges(rel_type), edge_rows(data))
        await step("CO_OCCURS", q.upsert_edges("CO_OCCURS"), edge_rows(rows.cooccurs))
        await step("TARGETS", q.upsert_edges("TARGETS"), edge_rows(rows.targets))
        return {"rows": written, "seconds": timings}

    # --- reads -------------------------------------------------------------------------
    async def read(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_records: int | None = None,
        timeout_s: float | None = None,
        write: bool = False,
    ) -> list[dict[str, Any]]:
        """Run one query and return at most ``max_records`` rows as plain dicts."""
        cap = max(1, min(max_records or self.max_records, self.max_records))
        timeout = timeout_s or self.query_timeout_s

        @unit_of_work(timeout=timeout)
        async def work(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(query, dict(params or {}))
            records = await result.fetch(cap)
            await result.consume()
            return [to_plain_record(rec) for rec in records]

        async with self._driver.session(database=self.database) as session:
            runner = session.execute_write if write else session.execute_read
            try:
                async with asyncio.timeout(timeout + 5.0):
                    return await runner(work)
            except TimeoutError as exc:
                raise TimeoutError(f"graph query exceeded {timeout:.0f}s") from exc
            except Neo4jError as exc:
                if "TransactionTimedOut" in (exc.code or ""):
                    raise TimeoutError(f"graph query exceeded {timeout:.0f}s") from exc
                raise

    async def counts(self) -> dict[str, dict[str, int]]:
        nodes = await self.read(q.COUNT_NODES, timeout_s=120.0)
        edges = await self.read(q.COUNT_EDGES, timeout_s=120.0)
        return {
            "nodes": {r["label"]: r["n"] for r in nodes},
            "edges": {r["type"]: r["n"] for r in edges},
        }
