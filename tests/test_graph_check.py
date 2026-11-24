"""Graph check for drug-target claims (GRAPH_CLAIM_CHECK), with a mocked graph."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest

from bioagent.agents.nodes import graph_checker
from bioagent.config import Settings
from bioagent.verify.graph_check import (
    GRAPH_REL_TYPES,
    GraphChecker,
    GraphDrug,
    GraphRelation,
    Neo4jGraphLookup,
    ReadStore,
    group_rows,
)
from bioagent.verify.verifier import ClaimVerifier, VerifyConfig
from tests.test_verify import FakeNLI, cite, ev

ROWS = [
    {"id": "CHEMBL1431", "name": "Metformin", "chembl_name": "METFORMIN",
     "synonyms": ["Glucophage", "DMBG"], "type": "STIMULATES", "gene": "PRKAA1", "count": 14,
     "edge_id": "STIMULATES|m|a1", "action_type": None, "source_db": "pubtator"},
    {"id": "CHEMBL1431", "name": "Metformin", "chembl_name": "METFORMIN",
     "synonyms": ["Glucophage", "DMBG"], "type": "TARGETS", "gene": "PRKAB1", "count": 1,
     "edge_id": "TARGETS|m|b1", "action_type": "ACTIVATOR", "source_db": "chembl"},
    {"id": "CHEMBL25", "name": "Aspirin", "chembl_name": "ASPIRIN",
     "synonyms": ["acetylsalicylic acid"], "type": "INHIBITS", "gene": "PRKAA2", "count": 3,
     "edge_id": "INHIBITS|a|a2", "action_type": None, "source_db": "pubtator"},
]  # fmt: skip


class FakeStore:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[Mapping[str, Any]] = []
        self.closed = 0

    async def read(
        self, query: str, params: Mapping[str, Any] | None = None, *, max_records: int | None = None
    ) -> list[dict[str, Any]]:
        self.calls.append(params or {})
        return self.rows


def lookup(store: FakeStore) -> Neo4jGraphLookup:
    @asynccontextmanager
    async def open_store() -> AsyncIterator[ReadStore]:
        yield store
        store.closed += 1

    return Neo4jGraphLookup(open_store, min_evidence=2)


def drugs() -> list[GraphDrug]:
    return group_rows(ROWS)


def test_group_rows_and_best_relation() -> None:
    by_name = {d.name: d for d in drugs()}
    metformin = by_name["Metformin"]
    assert "Glucophage" in metformin.terms and "DMBG" in metformin.terms
    assert len(metformin.relations) == 2
    best = metformin.best()  # curated ChEMBL mechanism wins over text-mined
    assert best.type == "TARGETS" and best.source == "chembl"
    assert best.describe() == "graph: Metformin activator PRKAB1 (chembl)"
    assert (
        by_name["Aspirin"].best().describe()
        == "graph: Aspirin inhibits PRKAA2 (pubtator, 3 papers)"
    )


async def test_lookup_queries_target_genes_once() -> None:
    store = FakeStore(ROWS)
    lk = lookup(store)
    first = await lk.drugs_for_target("AMPK")
    again = await lk.drugs_for_target("ampk")
    assert first is again and len(store.calls) == 1 and store.closed == 1
    params = store.calls[0]
    assert "PRKAA1" in params["symbols"] and params["min_evidence"] == 2
    assert set(params["types"]) == set(GRAPH_REL_TYPES)
    assert "CO_OCCURS" not in params["types"] and "ASSOCIATED_WITH" not in params["types"]


def test_match_needs_drug_and_target() -> None:
    checker = GraphChecker(drugs(), "AMPK")
    found = checker.match(
        [
            (1, "Metformin activates AMPK in hepatocytes."),
            (2, "Metformin lowers blood glucose."),  # no target
            (3, "Sorafenib activates AMP-activated protein kinase."),  # not in the graph
            (4, "Glucophage (metformin) and aspirin both act on AMPK."),
        ]
    )
    assert set(found) == {1, 4}
    assert {d.name for d in found[4].drugs} == {"Metformin", "Aspirin"}


def never_entails(premise: str, hypothesis: str) -> tuple[float, float]:
    return 0.05, 0.05


EVIDENCE = {
    "p1": ev("p1", "An unrelated abstract about diabetes outcomes.", "1"),
    "p2": ev("p2", "Another abstract about kinase signalling in general.", "2"),
}
CITATIONS = [cite(1, "p1", "1"), cite(2, "p2", "2")]
ANSWER = (
    "Metformin activates AMPK [1].\n"
    "Sorafenib activates AMPK in hepatoma cells [2].\n"
    "Aspirin activates AMPK and cures glioblastoma [2]."
)


async def run(graph: GraphChecker | None, policy: str = "remove") -> tuple[dict[str, Any], FakeNLI]:
    nli = FakeNLI(never_entails)
    verifier = ClaimVerifier(nli, VerifyConfig(policy=policy))  # type: ignore[arg-type]
    out = await verifier.verify(ANSWER, CITATIONS, EVIDENCE, graph=graph)
    return out, nli


async def test_off_is_unchanged() -> None:
    out, _ = await run(None)
    report = out["report"]
    assert report["graph_claim_check"] == "off" and report["graph_sentences"] == []
    assert report["removed_sentences"] == [1, 2, 3]
    assert set(report["sentence_paths"].values()) == {"nli"}


async def test_graph_verified_sentences_skip_nli_and_are_kept() -> None:
    out, nli = await run(GraphChecker(drugs(), "AMPK"))
    report = out["report"]
    claims = {c["sentence"]: c for c in report["claims"]}
    assert claims[1]["label"] == "graph"
    assert claims[1]["check"] == "graph: Metformin activator PRKAB1 (chembl)"
    assert claims[1]["evidence"][0]["ref"] == "graph:TARGETS|m|b1"
    # sentence 1 never reached NLI; sorafenib (no graph relation) did and was removed
    assert all("Metformin" not in h for _, h in nli.pairs)
    assert claims[2]["label"] == "not_supported"
    assert report["graph_sentences"] == [1, 3]
    assert report["sentence_paths"] == {"1": "graph", "2": "nli", "3": "graph"}
    assert report["removed_sentences"] == [2]
    # sentence 3: the one (fallback) claim names aspirin and AMPK -> graph; kept
    assert "Metformin activates AMPK" in out["answer"] and "Aspirin" in out["answer"]
    assert "Sorafenib" not in out["answer"]
    assert report["counts"]["graph"] == 2
    assert report["groundedness"] == round(2 / 3, 3)


async def test_graph_sentence_with_failed_claim_is_flagged_not_removed() -> None:
    verifier = ClaimVerifier(FakeNLI(never_entails), VerifyConfig(policy="remove"))
    answer = "Metformin activates AMPK and doubles survival in glioblastoma [1]."

    async def call(schema: Any, messages: Any) -> Any:
        from bioagent.verify.verifier import ClaimList, ExtractedClaim

        return ClaimList(
            claims=[
                ExtractedClaim(sentence=1, text="Metformin activates AMPK.", citations=[1]),
                ExtractedClaim(
                    sentence=1, text="Metformin doubles survival in glioblastoma.", citations=[1]
                ),
            ]
        )

    out = await verifier.verify(
        answer, CITATIONS, EVIDENCE, call=call, graph=GraphChecker(drugs(), "AMPK")
    )
    report = out["report"]
    assert [c["label"] for c in report["claims"]] == ["graph", "not_supported"]
    assert report["sentence_paths"] == {"1": "mixed"}
    assert report["removed_sentences"] == [] and report["flagged_sentences"] == [1]
    assert "[unverified]" in out["answer"]


async def test_graph_checker_from_plan() -> None:
    store = FakeStore(ROWS)
    checker, status = await graph_checker(
        lookup(store), [{"drug_target": None}, {"drug_target": "AMPK"}]
    )
    assert isinstance(checker, GraphChecker) and status is None and checker.target == "AMPK"
    assert await graph_checker(None, [{"drug_target": "AMPK"}]) == (None, None)
    none, status = await graph_checker(lookup(store), [{"drug_target": None}])
    assert none is None and status == "on (no drug-target task)"

    class Down:
        async def drugs_for_target(self, target: str) -> list[GraphDrug]:
            raise ConnectionError("neo4j down")

    down, status = await graph_checker(Down(), [{"drug_target": "AMPK"}])
    assert down is None and status is not None and status.startswith("error")


def test_setting_defaults_off() -> None:
    assert Settings(_env_file=None).graph_claim_check == "off"  # type: ignore[call-arg]
    from bioagent.agents.runner import build_graph_lookup

    assert build_graph_lookup(Settings(_env_file=None)) is None  # type: ignore[call-arg]
    on = Settings(_env_file=None, graph_claim_check="on")  # type: ignore[call-arg]
    assert isinstance(build_graph_lookup(on), Neo4jGraphLookup)


def test_relation_row_shape() -> None:
    r = GraphRelation("Aspirin", "PTGS2", "TARGETS", edge_id="e1", action_type="INHIBITOR")
    assert r.row()["source"] == "graph/chembl" and r.describe().startswith(
        "graph: Aspirin inhibitor"
    )


@pytest.mark.parametrize("target", ["mTOR", "EGFR"])
def test_unknown_drugs_never_match(target: str) -> None:
    assert GraphChecker([], target).match([(1, f"Metformin inhibits {target}.")]) == {}


@pytest.mark.parametrize(
    ("target", "gene", "text"),
    [
        ("HDAC", "HDAC1", "Vorinostat inhibits HDACs."),
        ("PARP", "PARP1", "Olaparib is a PARP inhibitor."),
        ("VEGFR", "KDR", "Axitinib blocks VEGFR2."),
        ("COX-2", "PTGS2", "Celecoxib inhibits cyclooxygenase-2."),
    ],
)
async def test_target_families(target: str, gene: str, text: str) -> None:
    rows = [{"id": "D1", "name": text.split()[0], "type": "TARGETS", "gene": gene}]
    store = FakeStore(rows)
    found = await lookup(store).drugs_for_target(target)
    assert gene in store.calls[0]["symbols"]
    assert set(GraphChecker(found, target).match([(1, text)])) == {1}
