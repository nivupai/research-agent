from __future__ import annotations

import httpx
import pytest
import respx

from bioagent.mcp_servers.literature.pubmed import (
    EUTILS_BASE,
    PubMedClient,
    iter_pubmed,
    parse_efetch_xml,
    pubmed_rate,
    validate_pmids,
)

ESEARCH = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH = f"{EUTILS_BASE}/efetch.fcgi"


def esearch_json(ids: list[str], count: int) -> dict[str, object]:
    return {
        "esearchresult": {
            "count": str(count),
            "retmax": str(len(ids)),
            "retstart": "0",
            "querykey": "1",
            "webenv": "MCID_test",
            "idlist": ids,
        }
    }


def test_parse_structured_abstract(efetch_xml: bytes) -> None:
    records = parse_efetch_xml(efetch_xml)
    assert len(records) == 2
    rec = records[0]
    assert rec.source == "pubmed"
    assert rec.pmid == "11111111"
    assert rec.pmcid == "PMC1234567"
    assert rec.title == "Metformin and glioblastoma survival: a cohort study."
    assert [s.label for s in rec.abstract_sections] == ["BACKGROUND", "METHODS", "RESULTS"]
    assert rec.abstract is not None
    assert rec.abstract.startswith("BACKGROUND: Metformin may affect tumour growth.")
    assert "RESULTS: Median survival improved (HR 0.8)." in rec.abstract
    assert rec.authors == ["Smith Jane A", "Doe J", "Glioma Consortium"]
    assert rec.journal == "Journal of Neuro-Oncology"
    assert rec.year == 2021
    assert rec.doi == "10.1000/test.2021.1"  # ArticleIdList wins over ELocationID
    assert rec.mesh_terms == ["Glioblastoma", "Metformin"]
    assert rec.untrusted_text is True


def test_parse_missing_abstract(efetch_xml: bytes) -> None:
    rec = parse_efetch_xml(efetch_xml)[1]
    assert rec.pmid == "22222222"
    assert rec.abstract is None
    assert rec.abstract_sections == []
    assert rec.year == 1998  # from MedlineDate
    assert rec.doi is None
    assert rec.mesh_terms == []
    assert rec.authors == ["Brown Bob"]


def test_rate_depends_on_api_key() -> None:
    assert pubmed_rate("abc") == 10.0
    assert pubmed_rate(None) == 3.0


def test_validate_pmids() -> None:
    assert validate_pmids([" 123 ", "456", "123", ""]) == ["123", "456"]
    with pytest.raises(ValueError):
        validate_pmids(["123", "abc; DROP"])


@respx.mock
async def test_esearch_paging(pubmed: PubMedClient, efetch_xml: bytes) -> None:
    search_route = respx.get(ESEARCH).mock(
        return_value=httpx.Response(200, json=esearch_json(["11111111", "22222222"], 5))
    )
    fetch_route = respx.get(EFETCH).mock(return_value=httpx.Response(200, content=efetch_xml))

    page1 = await pubmed.search("metformin glioblastoma", max_results=2, min_year=2015)
    assert page1.total == 5
    assert [r.pmid for r in page1.items] == ["11111111", "22222222"]
    assert page1.next_page_token is not None

    params = search_route.calls.last.request.url.params
    assert params["usehistory"] == "y"
    assert params["retmax"] == "2"
    assert params["tool"] == "bioagent"
    assert params["email"] == "dev@example.org"
    assert params["api_key"] == "test-key"
    assert params["mindate"] == "2015"
    assert params["datetype"] == "pdat"
    assert fetch_route.calls.last.request.url.params["id"] == "11111111,22222222"

    page2 = await pubmed.search("ignored", max_results=2, page_token=page1.next_page_token)
    p2 = fetch_route.calls.last.request.url.params
    assert p2["WebEnv"] == "MCID_test"
    assert p2["query_key"] == "1"
    assert p2["retstart"] == "2"
    assert p2["retmax"] == "2"
    assert search_route.call_count == 1  # second page reuses the History server
    assert page2.next_page_token is not None

    page3 = await pubmed.search("ignored", max_results=2, page_token=page2.next_page_token)
    assert fetch_route.calls.last.request.url.params["retstart"] == "4"
    assert page3.next_page_token is None  # 4 + 2 >= total of 5


@respx.mock
async def test_search_without_api_key_omits_it(http: object, efetch_xml: bytes) -> None:
    from bioagent.common.httpclient import RateLimitedClient

    assert isinstance(http, RateLimitedClient)
    client = PubMedClient(http)
    route = respx.get(ESEARCH).mock(return_value=httpx.Response(200, json=esearch_json([], 0)))
    result = await client.search("nothing matches")
    assert result.items == [] and result.total == 0 and result.next_page_token is None
    params = route.calls.last.request.url.params
    assert "api_key" not in params and "email" not in params
    assert params["tool"] == "bioagent"


async def test_invalid_page_token(pubmed: PubMedClient) -> None:
    with pytest.raises(ValueError, match="invalid page_token"):
        await pubmed.search("q", page_token="not-a-token")


@respx.mock
async def test_iter_pubmed_batches(pubmed: PubMedClient, efetch_xml: bytes) -> None:
    respx.get(ESEARCH).mock(return_value=httpx.Response(200, json=esearch_json([], 10)))
    fetch_route = respx.get(EFETCH).mock(return_value=httpx.Response(200, content=efetch_xml))
    records = [r async for r in iter_pubmed(pubmed, "q", limit=3, batch_size=2)]
    assert len(records) == 3
    starts = [c.request.url.params["retstart"] for c in fetch_route.calls]
    assert starts == ["0", "2"]
    assert fetch_route.calls[1].request.url.params["retmax"] == "1"
