"""``bioagent-graph`` MCP server with a fake graph backend (no Neo4j)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from bioagent.mcp_servers.graph import server
from tests.test_kg import GENE, METFORMIN, FakeReader


@pytest.fixture(autouse=True)
def fake_store() -> Iterator[FakeReader]:
    fake = FakeReader(
        {
            "WHERE g.symbol IN $symbols RETURN g": [{"n": GENE, "labels": ["Gene"], "score": 1.0}],
            "toLower(n.name) = $lower": [
                {"n": METFORMIN, "labels": ["Entity", "Drug"], "score": 10.0}
            ],
        }
    )
    server.set_store(fake)
    yield fake
    server.set_store(None)


async def test_tools_registered() -> None:
    names = {t.name for t in await server.mcp.list_tools()}
    assert names == {
        "find_entity",
        "neighbors",
        "drugs_targeting",
        "paths",
        "evidence",
        "drugs_mentioned",
    }


async def test_find_entity_marks_results_untrusted() -> None:
    out = await server.find_entity("metformin")
    assert out["untrusted"] is True and "untrusted" in out["notice"]
    assert out["entities"][0]["id"] == "MESH:D008687"
    assert out["entities"][0]["approved"] is True


async def test_hop_cap_and_bad_types_reach_client_as_errors(fake_store: FakeReader) -> None:
    with pytest.raises(ToolError, match="max_hops"):
        await server.mcp.call_tool(
            "paths", {"source": "metformin", "target": "glioblastoma", "max_hops": 4}
        )
    with pytest.raises(ToolError, match="unknown relationship types"):
        await server.mcp.call_tool("neighbors", {"entity": "MGMT", "rel_types": ["X]-()"]})
    with pytest.raises(ToolError, match="no edge"):
        await server.mcp.call_tool("evidence", {"edge_id": "TREATS|MESH:D1|MESH:D2"})


async def test_timeouts_become_tool_errors() -> None:
    class SlowReader(FakeReader):
        async def read(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
            raise TimeoutError("graph query exceeded 10s")

    server.set_store(SlowReader({}))
    with pytest.raises(ToolError, match="narrow the query"):
        await server.mcp.call_tool("find_entity", {"name": "metformin"})
