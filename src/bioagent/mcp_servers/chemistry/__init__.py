"""Chemistry MCP server: RDKit descriptors, drug-likeness, similarity and substructure search.

All scores are computed estimates from 2D structure. They are not experimental evidence.
"""

from bioagent.mcp_servers.chemistry.core import (
    ChemistryInputError,
    compute_drug_likeness,
    compute_properties,
    parse_smiles,
    rank_similarity,
    substructure_matches,
)
from bioagent.mcp_servers.chemistry.pubchem import PubChemClient

__all__ = [
    "ChemistryInputError",
    "PubChemClient",
    "compute_drug_likeness",
    "compute_properties",
    "parse_smiles",
    "rank_similarity",
    "substructure_matches",
]
