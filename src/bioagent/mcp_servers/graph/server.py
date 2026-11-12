"""MCP server exposing knowledge-graph lookups (find_entity, neighbors, drugs_targeting,
paths, evidence) over the local Neo4j graph built by ``make kg-build``.

The tools only run fixed, parameterised Cypher templates; raw Cypher is not accepted.
Every result is marked untrusted: entity names and paper titles are third-party data.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from bioagent.common.tools import tool_errors
from bioagent.config import get_settings
from bioagent.kg.service import GRAPH_NOTICE, Action, EntityKindFilter, GraphService, Reader
from bioagent.kg.store import GraphStore, GraphUnavailableError

_service: GraphService | None = None


def get_service() -> GraphService:
    global _service
    if _service is None:
        try:
            store = GraphStore.from_settings(get_settings())
        except GraphUnavailableError as exc:
            raise ValueError(str(exc)) from exc
        _service = GraphService(store)
    return _service


def set_store(store: Reader | None) -> None:
    """Override the graph backend (used by tests)."""
    global _service
    _service = GraphService(store) if store is not None else None


async def _call(coro: Any) -> dict[str, Any]:
    try:
        result: dict[str, Any] = await coro
    except TimeoutError as exc:
        raise ValueError(f"{exc}; narrow the query (fewer hops, higher min_evidence)") from exc
    except GraphUnavailableError as exc:
        raise ValueError(str(exc)) from exc
    except Exception as exc:
        # Driver errors (ServiceUnavailable, ClientError...) -> readable tool error.
        if type(exc).__module__.startswith("neo4j"):
            raise ValueError(f"graph query failed: {type(exc).__name__}") from exc
        raise
    return result


mcp = MCPServer(
    name="bioagent-graph",
    instructions=(
        "Query the biomedical knowledge graph (Gene/Drug/Disease nodes from PubTator3 "
        "annotations of ~11k PubMed abstracts, plus ChEMBL drug mechanisms). Use "
        "find_entity to resolve names, neighbors for relations, drugs_targeting for drugs "
        "acting on a gene or complex (AMPK, PI3K, mTORC1, AKT expand to their subunits), "
        "paths for multi-hop links (<= 3 hops) and evidence for an edge's PMIDs. " + GRAPH_NOTICE
    ),
)


@mcp.tool()
@tool_errors
async def find_entity(
    name: str, kind: EntityKindFilter | None = None, limit: int = 10
) -> dict[str, Any]:
    """Resolve a gene, drug or disease name to graph entities (exact, then fuzzy match).

    Returns ids, canonical names, paper counts and, for drugs, ChEMBL id and max_phase.
    Complex/family names such as AMPK also return their subunit genes (``alias_group``).
    """
    return await _call(get_service().find_entity(name, kind, limit))


@mcp.tool()
@tool_errors
async def neighbors(
    entity: str,
    rel_types: list[str] | None = None,
    neighbor_kind: EntityKindFilter | None = None,
    include_cooccurrence: bool = False,
    min_evidence: int = 1,
    limit: int = 20,
) -> dict[str, Any]:
    """Entities directly linked to ``entity`` (a name or id), strongest evidence first.

    ``rel_types`` restricts the relationship types (e.g. ["TREATS", "INHIBITS",
    "TARGETS"]); co-occurrence edges are excluded unless ``include_cooccurrence``.
    Every edge lists its source (pubtator or chembl), evidence count and PMIDs.
    """
    return await _call(
        get_service().neighbors(
            entity, rel_types, neighbor_kind, include_cooccurrence, min_evidence, limit
        )
    )


@mcp.tool()
@tool_errors
async def drugs_targeting(
    target: str,
    action: Action = "any",
    approved_only: bool = False,
    include_literature: bool = True,
    min_evidence: int = 2,
    limit: int = 20,
) -> dict[str, Any]:
    """Drugs acting on a gene or complex. ``action``: any, activator or inhibitor.

    Results labelled ``source: chembl`` are curated ChEMBL mechanisms of action; results
    labelled ``source: pubtator`` are text-mined relations with evidence PMIDs (at least
    ``min_evidence`` papers). ``approved_only`` keeps drugs with ChEMBL max_phase >= 4.
    """
    return await _call(
        get_service().drugs_targeting(
            target, action, approved_only, include_literature, min_evidence, limit
        )
    )


@mcp.tool()
@tool_errors
async def paths(
    source: str,
    target: str,
    max_hops: int = 3,
    rel_types: list[str] | None = None,
    min_evidence: int = 2,
    limit: int = 10,
) -> dict[str, Any]:
    """Shortest-first paths (1 to 3 hops) between two entities over typed relations.

    Each edge carries its type, source (pubtator/chembl), evidence count and PMIDs.
    Raise ``min_evidence`` or lower ``max_hops`` if the query times out.
    """
    return await _call(
        get_service().paths(source, target, max_hops, rel_types, min_evidence, limit)
    )


@mcp.tool()
@tool_errors
async def evidence(edge_id: str, limit: int = 10) -> dict[str, Any]:
    """The supporting papers (PMID, title, year, URL) for one edge id from another tool."""
    return await _call(get_service().evidence(edge_id, limit))


@mcp.tool()
@tool_errors
async def drugs_mentioned(
    pmids: list[str], approved_only: bool = False, limit: int = 30
) -> dict[str, Any]:
    """Drugs annotated (PubTator3) in the given papers, e.g. the PMIDs of a corpus search.

    Useful to turn search hits into a candidate drug list; ``approved_only`` keeps drugs
    with ChEMBL max_phase >= 4. Mentions are not mechanisms: check the abstracts.
    """
    return await _call(get_service().drugs_mentioned(pmids, approved_only, limit))
