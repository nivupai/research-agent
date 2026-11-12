"""Integration tests against the local Neo4j (``make graph-up``); excluded from CI.

The fixture graph uses ``TESTKG`` ids and made-up names so it can live next to the real
graph; only those nodes are deleted afterwards.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from bioagent.config import get_settings
from bioagent.kg.builder import GraphBuilder, GraphRows
from bioagent.kg.model import Mention, ParsedDoc, Relation
from bioagent.kg.service import EntityNotFoundError, GraphService
from bioagent.kg.store import GraphStore, GraphUnavailableError

pytestmark = pytest.mark.neo4j

DRUG = Mention("Drug", "TESTKG:D1", "Zztestformin", "zztestformin")
DRUG2 = Mention("Drug", "TESTKG:D2", "Zzexperimab", "zzexperimab")
GENE = Mention("Gene", "TESTKG:G1", "ZZK1A", "ZZK1A")
GENE2 = Mention("Gene", "TESTKG:G2", "ZZK1B", "ZZK1B")
DISEASE = Mention("Disease", "TESTKG:X1", "Zzglioma", "zzglioma")

CLEANUP = (
    "MATCH (n) WHERE n.id STARTS WITH 'TESTKG' OR n.pmid STARTS WITH 'TESTKG' "
    "DETACH DELETE n RETURN count(*) AS n"
)
FIXTURE_COUNTS = (
    "MATCH (n) WHERE n.id STARTS WITH 'TESTKG' OR n.pmid STARTS WITH 'TESTKG' "
    "OPTIONAL MATCH (n)-[r]->() RETURN count(DISTINCT n) AS nodes, count(r) AS edges"
)


def fixture_rows() -> GraphRows:
    b = GraphBuilder(min_cooccurrence=2)
    stimulates = Relation("stimulate", DRUG.id, GENE.id)
    gene_disease = Relation("associate", GENE.id, DISEASE.id)
    for i in range(1, 4):
        doc = ParsedDoc(f"TESTKG{i}", [DRUG, GENE, DISEASE], [stimulates, gene_disease])
        b.add(doc, title=f"Zztestformin paper {i}", year=2020 + i)
    b.add(ParsedDoc("TESTKG4", [DRUG2, GENE2, GENE], [Relation("inhibit", DRUG2.id, GENE.id)]))
    chembl: dict[str, dict[str, Any]] = {
        DRUG.id: {
            "chembl_id": "CHEMBLTEST1",
            "pref_name": "ZZTESTFORMIN",
            "max_phase": 4.0,
            "mechanisms": [
                {"action_type": "ACTIVATOR", "genes": ["ZZK1A"], "target_chembl_id": "CHEMBLT1",
                 "target_name": "ZZ kinase", "mechanism_of_action": "ZZ kinase activator"}
            ],
        },
        DRUG2.id: {
            "chembl_id": "CHEMBLTEST2",
            "pref_name": "ZZEXPERIMAB",
            "max_phase": 2.0,
            "mechanisms": [
                {"action_type": "INHIBITOR", "genes": ["ZZK1A"], "target_chembl_id": "CHEMBLT1",
                 "target_name": "ZZ kinase", "mechanism_of_action": "ZZ kinase inhibitor"}
            ],
        },
    }  # fmt: skip
    return b.build(chembl)


@pytest.fixture
async def store() -> AsyncIterator[GraphStore]:
    try:
        st = GraphStore.from_settings(get_settings())
        await st.verify()
    except GraphUnavailableError as exc:
        pytest.skip(str(exc))
    await st.read(CLEANUP, write=True)
    await st.load(fixture_rows(), log=lambda _msg: None)
    try:
        yield st
    finally:
        await st.read(CLEANUP, write=True)
        await st.close()


async def test_load_is_idempotent(store: GraphStore) -> None:
    before = await store.read(FIXTURE_COUNTS)
    await store.load(fixture_rows(), log=lambda _msg: None)
    after = await store.read(FIXTURE_COUNTS)
    assert before == after
    # 5 entities + 4 papers; edges: 12 MENTIONS + STIMULATES + ASSOCIATED_WITH + INHIBITS
    # + 3 CO_OCCURS (pairs seen >= 2 times) + 2 TARGETS
    assert after[0] == {"nodes": 9, "edges": 12 + 3 + 3 + 2}


async def test_find_entity_and_alias_free_resolution(store: GraphStore) -> None:
    svc = GraphService(store)
    out = await svc.find_entity("zztestformin")
    assert out["untrusted"] is True
    top = out["entities"][0]
    assert top["id"] == "TESTKG:D1" and top["match"] == "exact" and top["approved"] is True
    fuzzy = await svc.find_entity("zztestformn", kind="Drug")  # typo -> fulltext
    assert fuzzy["entities"][0]["id"] == "TESTKG:D1"
    with pytest.raises(EntityNotFoundError):
        await svc.resolve("TESTKG:nope-no-such-entity-qqq", kind="Disease")


async def test_drugs_targeting_sources_and_approval(store: GraphStore) -> None:
    svc = GraphService(store)
    out = await svc.drugs_targeting("ZZK1A", min_evidence=1)
    pairs = {(r["drug"]["id"], r["source"]) for r in out["results"]}
    assert pairs == {
        ("TESTKG:D1", "chembl"),
        ("TESTKG:D2", "chembl"),
        ("TESTKG:D1", "pubtator"),
        ("TESTKG:D2", "pubtator"),
    }
    lit = next(
        r for r in out["results"] if r["source"] == "pubtator" and r["drug"]["id"] == DRUG.id
    )
    assert lit["pmids"] == ["TESTKG1", "TESTKG2", "TESTKG3"]
    approved = await svc.drugs_targeting("ZZK1A", approved_only=True, min_evidence=1)
    assert {r["drug"]["id"] for r in approved["results"]} == {"TESTKG:D1"}
    inhibitors = await svc.drugs_targeting("ZZK1A", action="inhibitor", min_evidence=1)
    assert {(r["drug"]["id"], r["source"]) for r in inhibitors["results"]} == {
        ("TESTKG:D2", "chembl"),
        ("TESTKG:D2", "pubtator"),
    }


async def test_paths_and_evidence(store: GraphStore) -> None:
    svc = GraphService(store)
    out = await svc.paths("TESTKG:D1", "TESTKG:X1", max_hops=3, min_evidence=1)
    first = out["paths"][0]
    assert first["hops"] == 2  # no direct drug-disease relation in the fixture
    assert [n["id"] for n in first["nodes"]] == ["TESTKG:D1", "TESTKG:G1", "TESTKG:X1"]
    assert all(e["pmids"] for e in first["edges"] if e["source"] == "pubtator")
    assert "TESTKG1" in out["papers"]
    edge_id = first["edges"][0]["edge_id"]
    ev = await svc.evidence(edge_id)
    assert [p["pmid"] for p in ev["evidence"]] == ["TESTKG1", "TESTKG2", "TESTKG3"]
    assert ev["evidence"][0]["title"].startswith("Zztestformin paper")
    strict = await svc.paths("TESTKG:D1", "TESTKG:X1", max_hops=1, min_evidence=1)
    assert strict["paths"] == []


async def test_neighbors_filters(store: GraphStore) -> None:
    svc = GraphService(store)
    out = await svc.neighbors("TESTKG:G1", neighbor_kind="Drug")
    types = {(e["neighbor"]["id"], e["type"]) for e in out["edges"]}
    assert types == {
        ("TESTKG:D1", "STIMULATES"),
        ("TESTKG:D1", "TARGETS"),
        ("TESTKG:D2", "INHIBITS"),
        ("TESTKG:D2", "TARGETS"),
    }
    co = await svc.neighbors("TESTKG:G1", rel_types=["CO_OCCURS"], min_evidence=3)
    assert {e["neighbor"]["id"] for e in co["edges"]} == {"TESTKG:D1", "TESTKG:X1"}


async def test_read_caps_and_timeout(store: GraphStore) -> None:
    rows = await store.read("UNWIND range(1, 5000) AS x RETURN x", max_records=10)
    assert len(rows) == 10
    with pytest.raises(TimeoutError):
        await store.read(
            "UNWIND range(1, 100000000) AS x WITH x WHERE x % 7 = 0 RETURN count(x) AS n",
            timeout_s=0.5,
        )
