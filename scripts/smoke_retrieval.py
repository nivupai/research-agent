"""End-to-end smoke test: launch the retrieval MCP server over stdio and call search_corpus.

Requires the built corpus and indexes (``make ingest && make index``).
"""

from __future__ import annotations

import asyncio
import sys

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "bioagent.mcp_servers.retrieval"]
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])
        for mode in ("bm25", "hybrid_rerank"):
            result = await session.call_tool(
                "search_corpus",
                {"query": "metformin AMPK glioblastoma", "k": 3, "mode": mode, "year_min": 2015},
            )
            data = result.structured_content or {}
            print(f"{mode}: {data['timings_ms']['total']:.0f} ms")
            for hit in data["hits"]:
                print(f"  [{hit['pmid']} {hit['year']}] {hit['title'][:90]}")
        bad = await session.call_tool("search_corpus", {"query": " "})
        print("empty query error:", bad.content[0].text)  # type: ignore[union-attr]


if __name__ == "__main__":
    asyncio.run(main())
