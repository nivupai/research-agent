"""Pydantic models returned by the chemistry tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ESTIMATE_NOTICE = (
    "Computed from 2D structure with RDKit heuristics. These are estimates, not experimental "
    "evidence, and must not be presented as measured properties."
)


class MolecularProperties(BaseModel):
    input: str
    canonical_smiles: str
    formula: str
    molecular_weight: float
    exact_mass: float
    clogp: float = Field(description="Crippen logP")
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atoms: int
    fraction_csp3: float
    formal_charge: int
    qed: float
    resolved_from_name: bool = False
    pubchem_cid: int | None = None
    notice: str = ESTIMATE_NOTICE


class RuleCheck(BaseModel):
    name: str
    value: float
    criterion: str
    passed: bool


class RuleSet(BaseModel):
    name: str
    passed: bool
    violations: int
    rules: list[RuleCheck]


class StructuralAlert(BaseModel):
    catalog: Literal["PAINS", "BRENK"]
    name: str
    atoms: list[int]


class CNSMPO(BaseModel):
    """Wager et al. 2010 CNS MPO desirability score.

    The full score uses six components (0-1 each, max 6). RDKit cannot compute ClogD or the
    most-basic pKa, so ClogD is approximated by Crippen cLogP and pKa is left out. ``score`` is
    the sum of the components that were computed, out of ``max_score``.
    """

    score: float
    max_score: float
    components: dict[str, float]
    approximations: list[str]


class BBBEstimate(BaseModel):
    likely_permeable: bool
    criteria: list[RuleCheck]
    label: str = (
        "Computed estimate from a simple heuristic (TPSA < 90, MW < 450, HBD <= 3, "
        "1 <= cLogP <= 4). Not evidence of brain penetration; transporters (e.g. P-gp, OCTs) "
        "are not modelled."
    )


class DrugLikeness(BaseModel):
    canonical_smiles: str
    lipinski: RuleSet
    veber: RuleSet
    ghose: RuleSet
    pains_alerts: list[StructuralAlert]
    brenk_alerts: list[StructuralAlert]
    cns_mpo: CNSMPO
    bbb: BBBEstimate
    notice: str = ESTIMATE_NOTICE


class InvalidCandidate(BaseModel):
    index: int
    smiles: str
    error: str


class SimilarityHit(BaseModel):
    index: int
    smiles: str
    canonical_smiles: str
    score: float


class SimilarityResult(BaseModel):
    query_canonical_smiles: str
    fingerprint: Literal["morgan2", "maccs"]
    metric: Literal["tanimoto", "dice"]
    hits: list[SimilarityHit]
    invalid: list[InvalidCandidate] = Field(default_factory=list)


class SubstructureHit(BaseModel):
    index: int
    smiles: str
    canonical_smiles: str
    matches: list[list[int]]


class SubstructureResult(BaseModel):
    smarts: str
    total_candidates: int
    hits: list[SubstructureHit]
    invalid: list[InvalidCandidate] = Field(default_factory=list)


class ResolvedName(BaseModel):
    name: str
    cid: int
    canonical_smiles: str
    pubchem_smiles: str
    formula: str | None = None
    source: Literal["pubchem"] = "pubchem"
