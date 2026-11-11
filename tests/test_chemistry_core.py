from __future__ import annotations

import pytest

from bioagent.mcp_servers.chemistry.core import (
    MAX_CANDIDATES,
    ChemistryInputError,
    compute_drug_likeness,
    compute_properties,
    parse_smiles,
    rank_similarity,
    substructure_matches,
)

METFORMIN = "CN(C)C(=N)N=C(N)N"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
SALICYLIC_ACID = "OC(=O)c1ccccc1O"
BENZYLIDENE_RHODANINE = "O=C1NC(=S)SC1=Cc1ccccc1"


def test_metformin_properties_and_drug_likeness() -> None:
    mol = parse_smiles(METFORMIN)
    props = compute_properties(mol, METFORMIN)
    assert props.formula == "C4H11N5"
    assert props.molecular_weight == pytest.approx(129.16, abs=0.02)
    assert props.clogp < 0
    assert props.formal_charge == 0
    dl = compute_drug_likeness(mol)
    assert dl.lipinski.violations == 0
    assert dl.lipinski.passed
    assert dl.pains_alerts == []
    # TPSA ~91 and cLogP < 1: the simple BBB heuristic says "not likely" for metformin.
    assert dl.bbb.likely_permeable is False
    assert "estimate" in dl.bbb.label.lower()
    assert 0 <= dl.cns_mpo.score <= dl.cns_mpo.max_score


def test_aspirin() -> None:
    mol = parse_smiles(ASPIRIN)
    props = compute_properties(mol, ASPIRIN)
    assert props.formula == "C9H8O4"
    assert props.molecular_weight == pytest.approx(180.16, abs=0.02)
    assert props.hbd == 1
    assert props.aromatic_rings == 1
    assert props.canonical_smiles == "CC(=O)Oc1ccccc1C(=O)O"
    dl = compute_drug_likeness(mol)
    assert dl.lipinski.violations == 0 and dl.veber.passed
    assert {r.name for r in dl.lipinski.rules} == {
        "molecular_weight",
        "clogp",
        "hbd (NH+OH)",
        "hba (N+O)",
    }


def test_caffeine() -> None:
    mol = parse_smiles(CAFFEINE)
    props = compute_properties(mol, CAFFEINE)
    assert props.formula == "C8H10N4O2"
    assert props.molecular_weight == pytest.approx(194.19, abs=0.02)
    assert props.hbd == 0
    assert props.heavy_atoms == 14
    assert compute_drug_likeness(mol).lipinski.violations == 0


@pytest.mark.parametrize("bad", ["", "C1CC", "not a smiles", "C" * 5000])
def test_invalid_smiles_rejected(bad: str) -> None:
    with pytest.raises(ChemistryInputError):
        parse_smiles(bad)


def test_pains_hit_for_rhodanine() -> None:
    dl = compute_drug_likeness(parse_smiles(BENZYLIDENE_RHODANINE))
    names = [a.name for a in dl.pains_alerts]
    assert any("rhod" in n for n in names)
    assert all(a.atoms for a in dl.pains_alerts)


def test_similarity_ordering() -> None:
    result = rank_similarity(ASPIRIN, [SALICYLIC_ACID, METFORMIN, ASPIRIN, "garbage"])
    assert [h.index for h in result.hits] == [2, 0, 1]
    assert result.hits[0].score == 1.0
    assert result.hits[0].score > result.hits[1].score > result.hits[2].score
    assert [i.index for i in result.invalid] == [3]


def test_similarity_maccs_dice() -> None:
    result = rank_similarity(
        ASPIRIN, [SALICYLIC_ACID, CAFFEINE], fingerprint="maccs", metric="dice"
    )
    assert result.fingerprint == "maccs" and result.metric == "dice"
    assert result.hits[0].index == 0


def test_similarity_candidate_cap() -> None:
    with pytest.raises(ChemistryInputError, match="at most"):
        rank_similarity(ASPIRIN, ["C"] * (MAX_CANDIDATES + 1))


def test_substructure_match() -> None:
    # aromatic carboxylic acid
    result = substructure_matches("c[CX3](=O)[OX2H1]", [ASPIRIN, "CCO", SALICYLIC_ACID, CAFFEINE])
    assert [h.index for h in result.hits] == [0, 2]
    assert all(len(h.matches[0]) == 4 for h in result.hits)
    assert result.total_candidates == 4


def test_invalid_smarts_rejected() -> None:
    with pytest.raises(ChemistryInputError, match="invalid SMARTS"):
        substructure_matches("[C", [ASPIRIN])
