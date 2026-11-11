"""MCP server exposing RDKit-based chemistry tools and PubChem name resolution.

Every score is a computed estimate from 2D structure, never experimental evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import anyio
import anyio.to_thread
from mcp.server.mcpserver import MCPServer

from bioagent.common.httpclient import RateLimitedClient, UpstreamAPIError, quiet_http_logging
from bioagent.common.tools import tool_errors
from bioagent.mcp_servers.chemistry.core import (
    ChemistryInputError,
    Fingerprint,
    Metric,
    compute_drug_likeness,
    compute_properties,
    parse_smiles,
    rank_similarity,
    substructure_matches,
)
from bioagent.mcp_servers.chemistry.models import (
    ESTIMATE_NOTICE,
    DrugLikeness,
    MolecularProperties,
    ResolvedName,
    SimilarityResult,
    SubstructureResult,
)
from bioagent.mcp_servers.chemistry.pubchem import PUBCHEM_HOST, PUBCHEM_RATE, PubChemClient

COMPUTE_TIMEOUT_S = 20.0
T = TypeVar("T")

quiet_http_logging()


@dataclass
class ChemistryClients:
    http: RateLimitedClient
    pubchem: PubChemClient

    @classmethod
    def create(cls) -> ChemistryClients:
        http = RateLimitedClient(
            {PUBCHEM_HOST: PUBCHEM_RATE}, timeout=15.0, max_attempts=3, max_retry_after=10.0
        )
        return cls(http=http, pubchem=PubChemClient(http))


_clients: ChemistryClients | None = None


def get_clients() -> ChemistryClients:
    global _clients
    if _clients is None:
        _clients = ChemistryClients.create()
    return _clients


def set_clients(clients: ChemistryClients | None) -> None:
    """Override the shared clients (used by tests)."""
    global _clients
    _clients = clients


async def run_compute(func: Callable[[], T], deadline_s: float = COMPUTE_TIMEOUT_S) -> T:
    """Run CPU-bound RDKit work in a worker thread with a deadline."""
    try:
        with anyio.fail_after(deadline_s):
            return await anyio.to_thread.run_sync(func, abandon_on_cancel=True)
    except TimeoutError as exc:
        raise ChemistryInputError(
            f"computation exceeded {deadline_s:.0f}s; simplify the input"
        ) from exc


mcp = MCPServer(
    name="bioagent-chemistry",
    instructions="Cheminformatics tools (RDKit) and PubChem lookup. " + ESTIMATE_NOTICE,
)


@mcp.tool()
@tool_errors
async def molecular_properties(smiles_or_name: str) -> MolecularProperties:
    """Compute descriptors (MW, formula, cLogP, TPSA, HBD/HBA, QED...) for a SMILES or name.

    Input is parsed as SMILES first; if that fails it is resolved as a compound name via
    PubChem. Values are computed estimates.
    """
    text = smiles_or_name.strip()
    try:
        mol = parse_smiles(text)
    except ChemistryInputError as smiles_error:
        try:
            resolved = await get_clients().pubchem.resolve_name(text)
        except ValueError:
            raise ChemistryInputError(
                f"{smiles_error}; also not found as a compound name on PubChem"
            ) from None
        except UpstreamAPIError as exc:
            raise UpstreamAPIError(
                f"{smiles_error}; could not look it up as a name because PubChem failed: {exc}"
            ) from exc
        mol = parse_smiles(resolved.canonical_smiles)
        props = await run_compute(lambda: compute_properties(mol, text))
        return props.model_copy(update={"resolved_from_name": True, "pubchem_cid": resolved.cid})
    return await run_compute(lambda: compute_properties(mol, text))


@mcp.tool()
@tool_errors
async def drug_likeness(smiles: str) -> DrugLikeness:
    """Lipinski Ro5, Veber, Ghose, PAINS/Brenk alerts, CNS MPO and a BBB heuristic.

    The BBB result is a computed estimate from a simple rule set, not evidence.
    """
    mol = parse_smiles(smiles)
    return await run_compute(lambda: compute_drug_likeness(mol))


@mcp.tool()
@tool_errors
async def similarity(
    query: str,
    candidates: list[str],
    fingerprint: Fingerprint = "morgan2",
    metric: Metric = "tanimoto",
) -> SimilarityResult:
    """Rank up to 500 candidate SMILES by fingerprint similarity to the query SMILES."""
    return await run_compute(lambda: rank_similarity(query, candidates, fingerprint, metric))


@mcp.tool()
@tool_errors
async def substructure_search(smarts: str, candidates: list[str]) -> SubstructureResult:
    """Return which of up to 500 candidate SMILES contain the SMARTS pattern."""
    return await run_compute(lambda: substructure_matches(smarts, candidates))


@mcp.tool()
@tool_errors
async def resolve_name(name: str) -> ResolvedName:
    """Look up a compound name on PubChem and return its CID and canonical SMILES."""
    return await get_clients().pubchem.resolve_name(name)
