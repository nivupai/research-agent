"""Gene alias groups: protein complexes/families that literature names as one entity.

"AMPK" is a heterotrimer (catalytic alpha1/2, beta1/2, gamma1-3); PubTator links "AMPK"
mentions to individual subunits (mostly PRKAA1), and ChEMBL models it as the protein
complex group CHEMBL2096907. Queries for an alias expand to every subunit gene.
"""

from __future__ import annotations

ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "AMPK": ("PRKAA1", "PRKAA2", "PRKAB1", "PRKAB2", "PRKAG1", "PRKAG2", "PRKAG3"),
    "PI3K": ("PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "PIK3R1", "PIK3R2"),
    "MTORC1": ("MTOR", "RPTOR", "MLST8"),
    "MTORC2": ("MTOR", "RICTOR", "MLST8", "MAPKAP1"),
    "AKT": ("AKT1", "AKT2", "AKT3"),
}
_SYNONYMS: dict[str, str] = {
    "AMP-ACTIVATED PROTEIN KINASE": "AMPK",
    "AMP ACTIVATED PROTEIN KINASE": "AMPK",
    "5'-AMP-ACTIVATED PROTEIN KINASE": "AMPK",
    "PHOSPHOINOSITIDE 3-KINASE": "PI3K",
    "PHOSPHATIDYLINOSITOL 3-KINASE": "PI3K",
    "MTORC1": "MTORC1",
    "MTOR COMPLEX 1": "MTORC1",
    "MTOR COMPLEX 2": "MTORC2",
    "PKB": "AKT",
    "PROTEIN KINASE B": "AKT",
}


def canonical_alias(name: str) -> str | None:
    key = " ".join(name.strip().upper().split())
    if key in ALIAS_GROUPS:
        return key
    return _SYNONYMS.get(key)


def expand_gene_query(name: str) -> tuple[str | None, list[str]]:
    """Return (alias group name or None, gene symbols to match)."""
    alias = canonical_alias(name)
    if alias is not None:
        return alias, list(ALIAS_GROUPS[alias])
    return None, [name.strip().upper()]


def alias_groups_for(symbol: str) -> list[str]:
    s = symbol.upper()
    return [alias for alias, members in ALIAS_GROUPS.items() if s in members]
