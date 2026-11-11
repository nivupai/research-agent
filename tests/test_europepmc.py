from __future__ import annotations

import httpx
import respx

from bioagent.mcp_servers.literature.europepmc import (
    EUROPEPMC_BASE,
    EuropePMCClient,
    parse_abstract,
)

SEARCH = f"{EUROPEPMC_BASE}/search"


def result(pmid: str) -> dict[str, object]:
    return {
        "id": pmid,
        "source": "MED",
        "pmid": pmid,
        "pmcid": f"PMC{pmid}",
        "doi": f"10.1000/{pmid}",
        "title": "Metformin &amp; <i>GBM</i>",
        "authorString": "Smith J, Doe J.",
        "authorList": {"author": [{"fullName": "Smith J"}, {"fullName": "Doe J"}]},
        "journalInfo": {"journal": {"title": "Neuro Oncol"}},
        "pubYear": "2022",
        "abstractText": "<h4>Background</h4>Metformin.<h4>Results</h4>It <b>works</b>.",
        "meshHeadingList": {"meshHeading": [{"descriptorName": "Metformin"}]},
    }


def test_parse_abstract_sections() -> None:
    sections = parse_abstract("Intro text.<h4>Methods</h4>Did it.")
    assert [(s.label, s.text) for s in sections] == [(None, "Intro text."), ("Methods", "Did it.")]
    assert parse_abstract(None) == []


@respx.mock
async def test_cursor_paging(europepmc: EuropePMCClient) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params["cursorMark"]
        if cursor == "*":
            body = {
                "hitCount": 3,
                "nextCursorMark": "AoE1",
                "resultList": {"result": [result("1"), result("2")]},
            }
        elif cursor == "AoE1":
            body = {
                "hitCount": 3,
                "nextCursorMark": "AoE2",
                "resultList": {"result": [result("3")]},
            }
        else:
            body = {"hitCount": 3, "nextCursorMark": cursor, "resultList": {"result": []}}
        return httpx.Response(200, json=body)

    route = respx.get(SEARCH).mock(side_effect=handler)

    page1 = await europepmc.search("metformin", page_size=2)
    params = route.calls.last.request.url.params
    assert params["resultType"] == "core"
    assert params["format"] == "json"
    assert params["pageSize"] == "2"
    assert page1.total == 3
    assert page1.next_page_token == "AoE1"
    rec = page1.items[0]
    assert rec.source == "europepmc"
    assert rec.pmid == "1" and rec.pmcid == "PMC1" and rec.doi == "10.1000/1"
    assert rec.title == "Metformin & GBM"
    assert rec.authors == ["Smith J", "Doe J"]
    assert rec.journal == "Neuro Oncol" and rec.year == 2022
    assert [s.label for s in rec.abstract_sections] == ["Background", "Results"]
    assert rec.abstract == "Background: Metformin.\nResults: It works ."
    assert rec.mesh_terms == ["Metformin"]
    assert rec.untrusted_text is True

    page2 = await europepmc.search("metformin", page_size=2, page_token=page1.next_page_token)
    assert [r.pmid for r in page2.items] == ["3"]
    assert page2.next_page_token == "AoE2"

    page3 = await europepmc.search("metformin", page_size=2, page_token=page2.next_page_token)
    assert page3.items == [] and page3.next_page_token is None
