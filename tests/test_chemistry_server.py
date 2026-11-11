from __future__ import annotations

import time
from collections.abc import Iterator

import httpx
import pytest
import respx
from mcp.server.mcpserver.exceptions import ToolError

from bioagent.common.httpclient import RateLimitedClient
from bioagent.mcp_servers.chemistry import server
from bioagent.mcp_servers.chemistry.core import ChemistryInputError
from bioagent.mcp_servers.chemistry.pubchem import PUBCHEM_BASE, NameNotFoundError, PubChemClient

METFORMIN_URL = (
    f"{PUBCHEM_BASE}/compound/name/metformin/property/"
    "SMILES,ConnectivitySMILES,MolecularFormula/JSON"
)
PUBCHEM_METFORMIN = {
    "PropertyTable": {
        "Properties": [
            {
                "CID": 4091,
                "MolecularFormula": "C4H11N5",
                "SMILES": "CN(C)C(=N)N=C(N)N",
                "ConnectivitySMILES": "CN(C)C(=N)N=C(N)N",
            }
        ]
    }
}


@pytest.fixture(autouse=True)
def clients(http: RateLimitedClient) -> Iterator[None]:
    server.set_clients(server.ChemistryClients(http=http, pubchem=PubChemClient(http)))
    yield
    server.set_clients(None)


async def test_tools_registered() -> None:
    names = {t.name for t in await server.mcp.list_tools()}
    assert names >= {
        "molecular_properties",
        "drug_likeness",
        "similarity",
        "substructure_search",
        "resolve_name",
    }


@respx.mock
async def test_resolve_name_mocked() -> None:
    route = respx.get(METFORMIN_URL).mock(return_value=httpx.Response(200, json=PUBCHEM_METFORMIN))
    resolved = await server.resolve_name("metformin")
    assert resolved.cid == 4091
    assert resolved.canonical_smiles == "CN(C)C(=N)N=C(N)N"
    assert resolved.formula == "C4H11N5"
    assert route.call_count == 1


@respx.mock
async def test_resolve_name_not_found() -> None:
    respx.get(url__startswith=f"{PUBCHEM_BASE}/compound/name/").mock(
        return_value=httpx.Response(404, json={"Fault": {"Code": "PUGREST.NotFound"}})
    )
    with pytest.raises(ToolError, match="no compound named"):
        await server.resolve_name("notarealdrugname")


async def test_resolve_name_rejects_bad_characters() -> None:
    with pytest.raises(ToolError, match="unsupported characters"):
        await server.resolve_name("metformin/../../x?y=1")


@respx.mock
async def test_molecular_properties_falls_back_to_name() -> None:
    respx.get(METFORMIN_URL).mock(return_value=httpx.Response(200, json=PUBCHEM_METFORMIN))
    props = await server.molecular_properties("metformin")
    assert props.resolved_from_name and props.pubchem_cid == 4091
    assert props.molecular_weight == pytest.approx(129.16, abs=0.02)


async def test_molecular_properties_smiles_needs_no_network() -> None:
    props = await server.molecular_properties("CC(=O)Oc1ccccc1C(=O)O")
    assert not props.resolved_from_name and props.formula == "C9H8O4"


@respx.mock
async def test_molecular_properties_unknown_input() -> None:
    respx.get(url__startswith=f"{PUBCHEM_BASE}/compound/name/").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(ToolError, match="invalid SMILES"):
        await server.molecular_properties("C1CCzz")


async def test_drug_likeness_tool() -> None:
    result = await server.drug_likeness("CN(C)C(=N)N=C(N)N")
    assert result.lipinski.violations == 0
    assert "not evidence" in result.bbb.label.lower()


def _slow_sync_work() -> None:
    time.sleep(0.5)


async def test_run_compute_deadline() -> None:
    with pytest.raises(ChemistryInputError, match="exceeded"):
        await server.run_compute(_slow_sync_work, deadline_s=0.05)


@respx.mock
async def test_pubchem_not_found_raises_name_error(http: RateLimitedClient) -> None:
    respx.get(url__startswith=f"{PUBCHEM_BASE}/compound/name/").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(NameNotFoundError):
        await PubChemClient(http).resolve_name("notarealdrugname")


@respx.mock
async def test_pubchem_busy_fails_fast() -> None:
    from tests.conftest import RecordingSleep

    sleeper = RecordingSleep()
    http = RateLimitedClient({}, default_rate=1000.0, max_retry_after=10.0, sleep=sleeper)
    server.set_clients(server.ChemistryClients(http=http, pubchem=PubChemClient(http)))
    route = respx.get(METFORMIN_URL).mock(
        return_value=httpx.Response(503, headers={"Retry-After": "30"})
    )
    with pytest.raises(ToolError, match="retry after 30s"):
        await server.resolve_name("metformin")
    assert route.call_count == 1
    assert sleeper.calls == []
    await http.aclose()


async def test_invalid_smiles_message_reaches_client() -> None:
    with pytest.raises(ToolError, match="invalid SMILES"):
        await server.mcp.call_tool("drug_likeness", {"smiles": "C1CC"})
