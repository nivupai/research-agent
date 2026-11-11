"""Tests against the real PubMed and Europe PMC APIs. Run with ``make test-live``."""

from __future__ import annotations

import pytest

from bioagent.config import get_settings
from bioagent.mcp_servers.literature.server import LiteratureClients

pytestmark = pytest.mark.live

QUERY = "metformin glioblastoma"


async def test_live_pubmed_search_and_page() -> None:
    clients = LiteratureClients.from_settings(get_settings())
    try:
        page = await clients.pubmed.search(QUERY, max_results=3)
        assert page.total > 3
        assert 1 <= len(page.items) <= 3
        assert all(r.pmid and r.title for r in page.items)
        assert page.next_page_token
        page2 = await clients.pubmed.search(QUERY, max_results=3, page_token=page.next_page_token)
        assert {r.pmid for r in page2.items}.isdisjoint({r.pmid for r in page.items})
    finally:
        await clients.http.aclose()


async def test_live_europepmc_search() -> None:
    clients = LiteratureClients.from_settings(get_settings())
    try:
        page = await clients.europepmc.search(QUERY, page_size=3)
        assert page.total > 3
        assert len(page.items) == 3
        assert page.next_page_token
        assert any(r.abstract for r in page.items)
    finally:
        await clients.http.aclose()
