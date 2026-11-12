"""End-to-end smoke test: launch the graph MCP server over stdio and call every tool.

Requires Neo4j (``make graph-up``) and a loaded graph (``make kg-build``).
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def call(session: ClientSession, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    t0 = time.monotonic()
    result = await session.call_tool(tool, args)
    ms = (time.monotonic() - t0) * 1000
    if result.is_error:
        text = " ".join(getattr(c, "text", "") for c in result.content)
        print(f"{tool}({args}) -> ERROR in {ms:.0f} ms: {text[:200]}")
        return {}
    data = dict(result.structured_content or {})
    assert data.get("untrusted") is True, f"{tool} result not marked untrusted"
    print(f"{tool}({args}) -> {ms:.0f} ms")
    return data


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "bioagent.mcp_servers.graph"]
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])

        found = await call(session, "find_entity", {"name": "AMPK"})
        members = found.get("alias_group", {}).get("members", [])
        print("  AMPK subunits:", [m["symbol"] for m in members])

        drugs = await call(session, "drugs_targeting", {"target": "AMPK", "approved_only": True})
        for r in drugs.get("results", [])[:6]:
            print(f"  [{r['source']}] {r['drug']['name']} (max_phase {r['drug']['max_phase']})")

        paths = await call(
            session, "paths", {"source": "metformin", "target": "glioblastoma", "max_hops": 3}
        )
        for p in paths.get("paths", [])[:4]:
            chain = " -> ".join(n["name"] for n in p["nodes"])
            pmids = [e["pmids"][:2] for e in p["edges"]]
            print(f"  {chain}  pmids={pmids}")

        nb = await call(session, "neighbors", {"entity": "MGMT", "limit": 5})
        for e in nb.get("edges", []):
            print(f"  {e['type']} {e['neighbor']['name']} ({e['evidence_count']} papers)")

        if nb.get("edges"):
            ev = await call(session, "evidence", {"edge_id": nb["edges"][0]["edge_id"], "limit": 3})
            for p in ev.get("evidence", []):
                print(f"  PMID {p['pmid']} ({p.get('year')}): {str(p.get('title'))[:80]}")

        bad = await session.call_tool("paths", {"source": "a", "target": "b", "max_hops": 9})
        print("hop cap error:", bad.content[0].text)  # type: ignore[union-attr]


if __name__ == "__main__":
    asyncio.run(main())
