"""End-to-end smoke test: launch the chemistry MCP server over stdio and call each tool."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
METFORMIN = "CN(C)C(=N)N=C(N)N"


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "bioagent.mcp_servers.chemistry"]
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])

        async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
            result = await session.call_tool(name, args)
            if result.is_error:
                raise RuntimeError(result.content[0].text)  # type: ignore[union-attr]
            return result.structured_content or {}

        props = await call("molecular_properties", {"smiles_or_name": METFORMIN})
        print(
            f"molecular_properties(metformin): MW={props['molecular_weight']} "
            f"cLogP={props['clogp']} TPSA={props['tpsa']}"
        )
        dl = await call("drug_likeness", {"smiles": METFORMIN})
        print(
            f"drug_likeness(metformin): Ro5 violations={dl['lipinski']['violations']} "
            f"BBB likely={dl['bbb']['likely_permeable']} CNS MPO={dl['cns_mpo']['score']}"
        )
        sim = await call(
            "similarity", {"query": ASPIRIN, "candidates": ["OC(=O)c1ccccc1O", METFORMIN, ASPIRIN]}
        )
        print("similarity:", [(h["index"], h["score"]) for h in sim["hits"]])
        sub = await call(
            "substructure_search",
            {"smarts": "c[CX3](=O)[OX2H1]", "candidates": [ASPIRIN, METFORMIN]},
        )
        print("substructure_search hits:", [h["index"] for h in sub["hits"]])

        try:
            await call("drug_likeness", {"smiles": "C1CC"})
        except RuntimeError as exc:
            print("drug_likeness(invalid) error:", exc)

        # PubChem-dependent calls: report upstream failures instead of aborting the run.
        for name, args in [
            ("resolve_name", {"name": "caffeine"}),
            ("molecular_properties", {"smiles_or_name": "metformin"}),
        ]:
            try:
                out = await call(name, args)
                print(
                    f"{name}{tuple(args.values())}: CID={out.get('cid') or out.get('pubchem_cid')}"
                )
            except RuntimeError as exc:
                print(f"{name}{tuple(args.values())} FAILED (PubChem): {exc}")


if __name__ == "__main__":
    asyncio.run(main())
