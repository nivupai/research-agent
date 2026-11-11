from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
import respx

from bioagent.mcp_servers.literature import server
from bioagent.mcp_servers.literature.europepmc import EuropePMCClient
from bioagent.mcp_servers.literature.httpclient import RateLimitedClient
from bioagent.mcp_servers.literature.pubmed import EUTILS_BASE, PubMedClient


@pytest.fixture(autouse=True)
def clients(http: RateLimitedClient) -> Iterator[None]:
    server.set_clients(
        server.LiteratureClients(
            http=http, pubmed=PubMedClient(http), europepmc=EuropePMCClient(http)
        )
    )
    yield
    server.set_clients(None)


async def test_tools_registered() -> None:
    names = {t.name for t in await server.mcp.list_tools()}
    assert {"search_pubmed", "fetch_pubmed", "search_europepmc"} <= names


@respx.mock
async def test_fetch_pubmed_tool(efetch_xml: bytes) -> None:
    route = respx.get(f"{EUTILS_BASE}/efetch.fcgi").mock(
        return_value=httpx.Response(200, content=efetch_xml)
    )
    result = await server.fetch_pubmed(["11111111", "22222222"])
    assert result.total == 2
    assert result.untrusted_text is True
    assert "untrusted" in result.notice
    assert route.calls.last.request.url.params["id"] == "11111111,22222222"


async def test_fetch_pubmed_rejects_too_many() -> None:
    with pytest.raises(ValueError, match="at most 200"):
        await server.fetch_pubmed([str(i) for i in range(1, 300)])
