"""Pure, synchronous RDKit functions behind the chemistry tools."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from bioagent.mcp_servers.chemistry.models import (
    CNSMPO,
    BBBEstimate,
    DrugLikeness,
    InvalidCandidate,
    MolecularProperties,
    RuleCheck,
    RuleSet,
    SimilarityHit,
    SimilarityResult,
    StructuralAlert,
    SubstructureHit,
    SubstructureResult,
)

MAX_SMILES_LENGTH = 2_000
MAX_SMARTS_LENGTH = 1_000
MAX_CANDIDATES = 500
MAX_MATCHES_PER_MOLECULE = 50

Fingerprint = Literal["morgan2", "maccs"]
Metric = Literal["tanimoto", "dice"]

# RDKit prints parse errors to stderr; we report errors ourselves instead.
rdBase.DisableLog("rdApp.*")


class ChemistryInputError(ValueError):
    """Invalid or oversized SMILES/SMARTS input."""


def _mw(mol: Chem.Mol) -> float:
    return rdmd._CalcMolWt(mol)


def _crippen(mol: Chem.Mol) -> tuple[float, float]:
    """Return (Crippen logP, molar refractivity)."""
    logp, mr = rdmd.CalcCrippenDescriptors(mol)
    return float(logp), float(mr)


def _qed(mol: Chem.Mol) -> float:
    return float(QED.qed(mol))  # type: ignore[no-untyped-call]


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse and sanitise a SMILES string, raising ``ChemistryInputError`` on failure."""
    smiles = smiles.strip()
    if not smiles:
        raise ChemistryInputError("empty SMILES")
    if len(smiles) > MAX_SMILES_LENGTH:
        raise ChemistryInputError(f"SMILES longer than {MAX_SMILES_LENGTH} characters")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ChemistryInputError(f"invalid SMILES: {smiles[:80]!r}")
    return mol


def parse_smarts(smarts: str) -> Chem.Mol:
    smarts = smarts.strip()
    if not smarts:
        raise ChemistryInputError("empty SMARTS")
    if len(smarts) > MAX_SMARTS_LENGTH:
        raise ChemistryInputError(f"SMARTS longer than {MAX_SMARTS_LENGTH} characters")
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ChemistryInputError(f"invalid SMARTS: {smarts[:80]!r}")
    return patt


def _check_candidates(candidates: Sequence[str]) -> None:
    if len(candidates) > MAX_CANDIDATES:
        raise ChemistryInputError(
            f"at most {MAX_CANDIDATES} candidates per call (got {len(candidates)})"
        )


def compute_properties(mol: Chem.Mol, input_text: str) -> MolecularProperties:
    return MolecularProperties(
        input=input_text,
        canonical_smiles=Chem.MolToSmiles(mol),
        formula=rdmd.CalcMolFormula(mol),
        molecular_weight=round(_mw(mol), 3),
        exact_mass=round(rdmd.CalcExactMolWt(mol), 4),
        clogp=round(_crippen(mol)[0], 3),
        tpsa=round(rdmd.CalcTPSA(mol), 2),
        hbd=rdmd.CalcNumHBD(mol),
        hba=rdmd.CalcNumHBA(mol),
        rotatable_bonds=rdmd.CalcNumRotatableBonds(mol),
        aromatic_rings=rdmd.CalcNumAromaticRings(mol),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        fraction_csp3=round(rdmd.CalcFractionCSP3(mol), 3),
        formal_charge=Chem.GetFormalCharge(mol),
        qed=round(_qed(mol), 3),
    )


def _rule(name: str, value: float, criterion: str, passed: bool) -> RuleCheck:
    return RuleCheck(name=name, value=round(value, 3), criterion=criterion, passed=passed)


def _ruleset(name: str, rules: list[RuleCheck], max_violations: int = 0) -> RuleSet:
    violations = sum(not r.passed for r in rules)
    return RuleSet(
        name=name, passed=violations <= max_violations, violations=violations, rules=rules
    )


def _linear_desirability(value: float, best: float, worst: float) -> float:
    """1 at/below ``best``, 0 at/above ``worst`` (monotonic decreasing), linear between."""
    if value <= best:
        return 1.0
    if value >= worst:
        return 0.0
    return (worst - value) / (worst - best)


def _tpsa_desirability(tpsa: float) -> float:
    """Hump function from Wager 2010: 0 at <=20, 1 over 40-90, 0 at >=120."""
    if tpsa <= 20 or tpsa >= 120:
        return 0.0
    if tpsa < 40:
        return (tpsa - 20) / 20
    if tpsa <= 90:
        return 1.0
    return (120 - tpsa) / 30


def cns_mpo(mw: float, clogp: float, tpsa: float, hbd: int) -> CNSMPO:
    components = {
        "clogp": _linear_desirability(clogp, 3, 5),
        "clogd_approx": _linear_desirability(clogp, 2, 4),
        "mw": _linear_desirability(mw, 360, 500),
        "tpsa": _tpsa_desirability(tpsa),
        "hbd": _linear_desirability(hbd, 0.5, 3.5),
    }
    components = {k: round(v, 3) for k, v in components.items()}
    return CNSMPO(
        score=round(sum(components.values()), 3),
        max_score=5.0,
        components=components,
        approximations=[
            "ClogD approximated by Crippen cLogP (no ionisation model)",
            "most-basic pKa component omitted (not computable with RDKit); full scale is 0-6",
        ],
    )


@lru_cache(maxsize=2)
def _catalog(kind: Literal["PAINS", "BRENK"]) -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(
        FilterCatalogParams.FilterCatalogs.PAINS
        if kind == "PAINS"
        else FilterCatalogParams.FilterCatalogs.BRENK
    )
    return FilterCatalog(params)


def structural_alerts(mol: Chem.Mol, kind: Literal["PAINS", "BRENK"]) -> list[StructuralAlert]:
    alerts = []
    for entry in _catalog(kind).GetMatches(mol):
        atoms = sorted({pair.target for fm in entry.GetFilterMatches(mol) for pair in fm.atomPairs})
        alerts.append(StructuralAlert(catalog=kind, name=entry.GetDescription(), atoms=atoms))
    return alerts


def compute_drug_likeness(mol: Chem.Mol) -> DrugLikeness:
    mw = _mw(mol)
    clogp, molar_refractivity = _crippen(mol)
    tpsa = rdmd.CalcTPSA(mol)
    hbd = rdmd.CalcNumHBD(mol)
    lip_hbd = rdmd.CalcNumLipinskiHBD(mol)
    lip_hba = rdmd.CalcNumLipinskiHBA(mol)
    rotb = rdmd.CalcNumRotatableBonds(mol)
    total_atoms = Chem.AddHs(mol).GetNumAtoms()

    lipinski = _ruleset(
        "Lipinski Ro5",
        [
            _rule("molecular_weight", mw, "<= 500", mw <= 500),
            _rule("clogp", clogp, "<= 5", clogp <= 5),
            _rule("hbd (NH+OH)", lip_hbd, "<= 5", lip_hbd <= 5),
            _rule("hba (N+O)", lip_hba, "<= 10", lip_hba <= 10),
        ],
        max_violations=1,
    )
    veber = _ruleset(
        "Veber",
        [
            _rule("rotatable_bonds", rotb, "<= 10", rotb <= 10),
            _rule("tpsa", tpsa, "<= 140", tpsa <= 140),
        ],
    )
    ghose = _ruleset(
        "Ghose",
        [
            _rule("molecular_weight", mw, "160-480", 160 <= mw <= 480),
            _rule("clogp", clogp, "-0.4 to 5.6", -0.4 <= clogp <= 5.6),
            _rule(
                "molar_refractivity", molar_refractivity, "40-130", 40 <= molar_refractivity <= 130
            ),
            _rule("total_atoms", total_atoms, "20-70", 20 <= total_atoms <= 70),
        ],
    )
    bbb_rules = [
        _rule("tpsa", tpsa, "< 90", tpsa < 90),
        _rule("molecular_weight", mw, "< 450", mw < 450),
        _rule("hbd", hbd, "<= 3", hbd <= 3),
        _rule("clogp", clogp, "1-4", 1 <= clogp <= 4),
    ]
    return DrugLikeness(
        canonical_smiles=Chem.MolToSmiles(mol),
        lipinski=lipinski,
        veber=veber,
        ghose=ghose,
        pains_alerts=structural_alerts(mol, "PAINS"),
        brenk_alerts=structural_alerts(mol, "BRENK"),
        cns_mpo=cns_mpo(mw, clogp, tpsa, hbd),
        bbb=BBBEstimate(likely_permeable=all(r.passed for r in bbb_rules), criteria=bbb_rules),
    )


def _fingerprint(mol: Chem.Mol, kind: Fingerprint) -> DataStructs.ExplicitBitVect:
    fp: DataStructs.ExplicitBitVect
    if kind == "maccs":
        fp = rdmd.GetMACCSKeysFingerprint(mol)
    else:
        fp = _morgan_generator().GetFingerprint(mol)
    return fp


@lru_cache(maxsize=1)
def _morgan_generator() -> rdFingerprintGenerator.FingerprintGenerator64:
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def rank_similarity(
    query: str,
    candidates: Sequence[str],
    fingerprint: Fingerprint = "morgan2",
    metric: Metric = "tanimoto",
) -> SimilarityResult:
    """Score candidates against the query, highest first. Invalid candidates are reported."""
    if fingerprint not in ("morgan2", "maccs"):
        raise ChemistryInputError("fingerprint must be 'morgan2' or 'maccs'")
    if metric not in ("tanimoto", "dice"):
        raise ChemistryInputError("metric must be 'tanimoto' or 'dice'")
    _check_candidates(candidates)
    qmol = parse_smiles(query)
    qfp = _fingerprint(qmol, fingerprint)
    valid: list[tuple[int, str, Chem.Mol]] = []
    invalid: list[InvalidCandidate] = []
    for i, smi in enumerate(candidates):
        try:
            valid.append((i, smi, parse_smiles(smi)))
        except ChemistryInputError as exc:
            invalid.append(InvalidCandidate(index=i, smiles=smi[:200], error=str(exc)))
    fps = [_fingerprint(m, fingerprint) for _, _, m in valid]
    bulk = (
        DataStructs.BulkTanimotoSimilarity
        if metric == "tanimoto"
        else DataStructs.BulkDiceSimilarity
    )
    scores = bulk(qfp, fps) if fps else []
    hits = [
        SimilarityHit(
            index=i, smiles=smi, canonical_smiles=Chem.MolToSmiles(m), score=round(float(s), 4)
        )
        for (i, smi, m), s in zip(valid, scores, strict=True)
    ]
    hits.sort(key=lambda h: (-h.score, h.index))
    return SimilarityResult(
        query_canonical_smiles=Chem.MolToSmiles(qmol),
        fingerprint=fingerprint,
        metric=metric,
        hits=hits,
        invalid=invalid,
    )


def substructure_matches(smarts: str, candidates: Sequence[str]) -> SubstructureResult:
    """Return the candidates containing the SMARTS pattern, with matched atom indices."""
    _check_candidates(candidates)
    patt = parse_smarts(smarts)
    hits: list[SubstructureHit] = []
    invalid: list[InvalidCandidate] = []
    for i, smi in enumerate(candidates):
        try:
            mol = parse_smiles(smi)
        except ChemistryInputError as exc:
            invalid.append(InvalidCandidate(index=i, smiles=smi[:200], error=str(exc)))
            continue
        matches = mol.GetSubstructMatches(patt, maxMatches=MAX_MATCHES_PER_MOLECULE)
        if matches:
            hits.append(
                SubstructureHit(
                    index=i,
                    smiles=smi,
                    canonical_smiles=Chem.MolToSmiles(mol),
                    matches=[list(m) for m in matches],
                )
            )
    return SubstructureResult(
        smarts=smarts, total_candidates=len(candidates), hits=hits, invalid=invalid
    )
