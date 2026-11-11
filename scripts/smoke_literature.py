"""End-to-end smoke test: launch the literature MCP server over stdio and call each tool."""

from __future__ import annotations

import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main(query: str) -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "bioagent.mcp_servers.literature"]
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])
        calls: list[tuple[str, dict[str, object]]] = [
            ("search_pubmed", {"query": query, "max_results": 3}),
            ("search_europepmc", {"query": query, "page_size": 3}),
        ]
        for name, args in calls:
            result = await session.call_tool(name, args)
            data = result.structured_content or json.loads(result.content[0].text)  # type: ignore[union-attr]
            pmids = [item.get("pmid") for item in data["items"]]
            print(f"{name}: total={data['total']} pmids={pmids} untrusted={data['untrusted_text']}")
            if name == "search_pubmed":
                fetched = await session.call_tool("fetch_pubmed", {"pmids": pmids[:2]})
                fdata = fetched.structured_content or {}
                print("fetch_pubmed:", [i.get("title", "")[:60] for i in fdata.get("items", [])])


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "metformin glioblastoma"))
