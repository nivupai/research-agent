"""Unit tests for the knowledge-graph pipeline (no Neo4j needed)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from bioagent.kg import queries as q
from bioagent.kg.aliases import alias_groups_for, canonical_alias, expand_gene_query
from bioagent.kg.builder import GraphBuilder
from bioagent.kg.chembl import parse_phase, parse_target, pick_molecule
from bioagent.kg.model import Mention, ParsedDoc, Relation
from bioagent.kg.pubtator import normalise_entity, parse_document
from bioagent.kg.service import GraphService
from bioagent.kg.store import batches, edge_rows, to_plain

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def pubtator_docs() -> list[dict[str, Any]]:
    data = json.loads((FIXTURES / "pubtator_sample.json").read_text())
    docs: list[dict[str, Any]] = data["PubTator3"]
    return docs


# --- parsing / normalisation -------------------------------------------------------------
def test_parse_document_entities_and_relations(pubtator_docs: list[dict[str, Any]]) -> None:
    doc = parse_document(pubtator_docs[0])
    assert doc.pmid == "25867026"
    kinds = {m.id: m.kind for m in doc.mentions}
    assert kinds["MESH:D008687"] == "Drug"  # metformin (Chemical -> Drug)
    assert kinds["MESH:D005909"] == "Disease"  # glioblastoma
    rels = {(r.rel_type, r.source, r.target) for r in doc.relations}
    assert ("TREATS", "MESH:D008687", "MESH:D005909") in rels
    # canonical orientation Drug -> Gene -> Disease
    for r in doc.relations:
        assert r.source != r.target


def test_relation_gene_roles_reuse_homologene_ids(pubtator_docs: list[dict[str, Any]]) -> None:
    doc = parse_document(pubtator_docs[1])
    gene_ids = {m.id for m in doc.mentions if m.kind == "Gene"}
    assert all(i.startswith("HG:") for i in gene_ids)
    for r in doc.relations:
        for end in (r.source, r.target):
            if not end.startswith("MESH:"):
                assert end in gene_ids  # relation roles resolved to the mention's HG id


def test_normalise_gene_prefers_homologene() -> None:
    m = normalise_entity(
        {"type": "Gene", "identifier": "5562", "name": "Prkaa1", "ncbi_homologene": "49590"},
        "AMPKalpha1",
    )
    assert m == Mention("Gene", "HG:49590", "PRKAA1", "AMPKalpha1", "5562", "49590")
    no_hg = normalise_entity({"type": "Gene", "identifier": "5562;5563", "name": "PRKAA1"})
    assert no_hg is not None and no_hg.id == "NCBIGene:5562"


@pytest.mark.parametrize(
    ("infons", "expected"),
    [
        ({"type": "Chemical", "identifier": "MESH:D008687", "name": "Metformin"}, "MESH:D008687"),
        ({"type": "Chemical", "identifier": "D008687", "name": "Metformin"}, "MESH:D008687"),
        ({"type": "Disease", "identifier": "MESH:D005909|x", "name": "GBM"}, "MESH:D005909"),
        ({"type": "Chemical", "identifier": "-", "name": "unlinked"}, None),
        ({"type": "Species", "identifier": "9606", "name": "human"}, None),
        ({"type": "Gene", "identifier": "not-a-number", "name": "X"}, None),
    ],
)
def test_normalise_entity_ids(infons: dict[str, Any], expected: str | None) -> None:
    m = normalise_entity(infons)
    assert (m.id if m else None) == expected


def test_aliases() -> None:
    assert canonical_alias("amp-activated  protein kinase") == "AMPK"
    alias, genes = expand_gene_query("AMPK")
    assert alias == "AMPK" and "PRKAA1" in genes and len(genes) == 7
    assert expand_gene_query("mgmt") == (None, ["MGMT"])
    assert set(alias_groups_for("MTOR")) == {"MTORC1", "MTORC2"}


# --- ChEMBL parsing ----------------------------------------------------------------------
def test_pick_molecule_prefers_highest_phase_and_parent() -> None:
    molecules = [
        {"molecule_chembl_id": "CHEMBL_SALT", "pref_name": "X HCL", "max_phase": "4.0",
         "molecule_hierarchy": {"parent_chembl_id": "CHEMBL1431"}},
        {"molecule_chembl_id": "CHEMBL_OTHER", "pref_name": "X", "max_phase": None},
    ]  # fmt: skip
    picked = pick_molecule(molecules)
    assert picked == {"chembl_id": "CHEMBL1431", "pref_name": "X HCL", "max_phase": 4.0}
    assert pick_molecule([]) is None
    assert parse_phase("bad") is None and parse_phase(None) is None and parse_phase(2) == 2.0


def test_parse_target_gene_symbols() -> None:
    target = {
        "target_chembl_id": "CHEMBL2096907",
        "pref_name": "AMP-activated protein kinase",
        "target_type": "PROTEIN COMPLEX GROUP",
        "organism": "Homo sapiens",
        "target_components": [
            {"target_component_synonyms": [
                {"syn_type": "UNIPROT", "component_synonym": "AAPK1"},
                {"syn_type": "GENE_SYMBOL", "component_synonym": "prkaa1"},
            ]},
            {"target_component_synonyms": [
                {"syn_type": "GENE_SYMBOL", "component_synonym": "PRKAB1"},
            ]},
            {"target_component_synonyms": []},
        ],
    }  # fmt: skip
    info = parse_target(target)
    assert info["genes"] == ["PRKAA1", "PRKAB1"]
    assert info["target_type"] == "PROTEIN COMPLEX GROUP"


# --- builder -----------------------------------------------------------------------------
def _doc(pmid: str, ids: list[tuple[str, str]], rels: list[Relation] | None = None) -> ParsedDoc:
    mentions = [
        Mention(kind, i, i.split(":")[1], i.split(":")[1])  # type: ignore[arg-type]
        for kind, i in ids
    ]
    return ParsedDoc(pmid, mentions, rels or [])


def test_cooccurrence_threshold() -> None:
    b = GraphBuilder(min_cooccurrence=2)
    drug, gbm, other = ("Drug", "MESH:D1"), ("Disease", "MESH:D2"), ("Disease", "MESH:D3")
    b.add(_doc("1", [drug, gbm, other]))
    b.add(_doc("2", [drug, gbm]))
    rows = b.build()
    pairs = {(r["source"], r["target"]): r for r in rows.cooccurs}
    assert set(pairs) == {("MESH:D1", "MESH:D2")}  # D1-D3 and D2-D3 seen once only
    assert pairs[("MESH:D1", "MESH:D2")]["count"] == 2
    assert pairs[("MESH:D1", "MESH:D2")]["pmids"] == ["1", "2"]
    assert len(GraphBuilder(min_cooccurrence=1).build().cooccurs) == 0  # empty builder
    b1 = GraphBuilder(min_cooccurrence=1)
    b1.add(_doc("1", [drug, gbm, other]))
    assert len(b1.build().cooccurs) == 3


def test_builder_merges_mentions_of_folded_genes() -> None:
    b = GraphBuilder()
    hg = Mention("Gene", "HG:1", "PRKAA1", "AMPK", "5562", "1")
    ncbi_only = Mention("Gene", "NCBIGene:999", "PRKAA1", "AMPKa1", "999")
    b.add(ParsedDoc("1", [hg, ncbi_only]))
    rows = b.build()
    assert [g["id"] for g in rows.nodes["Gene"]] == ["HG:1"]
    assert rows.mentions == [{"pmid": "1", "id": "HG:1", "count": 2}]  # one MERGE row


def test_builder_relations_and_chembl_targets() -> None:
    b = GraphBuilder()
    rel = Relation("treat", "MESH:D1", "MESH:D2", 0.9)
    gene = Mention("Gene", "HG:1", "PRKAA1", "AMPK", "5562", "1")
    for pmid in ("10", "11"):
        doc = _doc(pmid, [("Drug", "MESH:D1"), ("Disease", "MESH:D2")], [rel])
        doc.mentions.append(gene)
        b.add(doc, title=f"paper {pmid}", year=2020)
    chembl = {
        "MESH:D1": {
            "chembl_id": "CHEMBL1431",
            "pref_name": "METFORMIN",
            "max_phase": 4.0,
            "mechanisms": [
                {"action_type": "ACTIVATOR", "genes": ["PRKAA1", "NOTINCORPUS"],
                 "target_chembl_id": "CHEMBL2096907", "target_name": "AMPK"}
            ],
        }
    }  # fmt: skip
    rows = b.build(chembl)
    (treat,) = rows.relations["TREATS"]
    assert treat["pmids"] == ["10", "11"] and treat["count"] == 2 and treat["max_score"] == 0.9
    assert treat["source_db"] == "pubtator"
    (target,) = rows.targets
    assert target["source"] == "MESH:D1" and target["target"] == "HG:1"
    assert target["source_db"] == "chembl" and target["action_type"] == "ACTIVATOR"
    drug = rows.nodes["Drug"][0]
    assert drug["approved"] is True and drug["chembl_id"] == "CHEMBL1431"
    gene_row = rows.nodes["Gene"][0]
    assert gene_row["alias_groups"] == ["AMPK"]
    assert rows.counts()["nodes"]["Paper"] == 2


# --- Cypher templates and parameters -------------------------------------------------------
def test_rel_pattern_whitelist() -> None:
    assert q.rel_pattern(["treats", "INHIBITS", "TREATS"]) == "TREATS|INHIBITS"
    with pytest.raises(q.QueryValidationError):
        q.rel_pattern(["TREATS]->() DETACH DELETE n //"])
    with pytest.raises(q.QueryValidationError):
        q.rel_pattern([])


@pytest.mark.parametrize("hops", [0, 4, -1, "2"])
def test_hop_cap(hops: Any) -> None:
    with pytest.raises(q.QueryValidationError):
        q.paths_query(hops, ["TREATS"])


def test_templates_use_parameters_only() -> None:
    assert "*1..3]" in q.paths_query(3, ["TREATS"])
    edge_q = q.upsert_edges("TREATS")
    assert edge_q.startswith("UNWIND $rows AS row") and "row.props" in edge_q
    with pytest.raises(q.QueryValidationError):
        q.upsert_edges("TREATS|INHIBITS")
    with pytest.raises(q.QueryValidationError):
        q.upsert_nodes("Gene) DETACH DELETE (x")
    with pytest.raises(q.QueryValidationError):
        q.evidence_query("BOGUS|a|b")
    assert "$edge_id" in q.evidence_query("TREATS|MESH:D1|MESH:D2")
    assert q.clamp_limit(10_000) == q.MAX_LIMIT and q.clamp_limit(0) == 1
    assert q.clamp_limit("x", 7) == 7  # type: ignore[arg-type]


def test_lucene_query_escapes() -> None:
    out = q.lucene_query("mTOR (complex) OR name:x")
    assert (
        out == r'"mTOR \(complex\) or name\:x"^3 OR (mTOR AND \(complex\)~1 AND or AND name\:x~1)'
    )
    with pytest.raises(q.QueryValidationError):
        q.lucene_query("   ")


def test_store_helpers() -> None:
    rows = edge_rows([{"id": "T|a|b", "source": "a", "target": "b", "count": 2, "pmids": ["1"]}])
    assert rows == [
        {"id": "T|a|b", "source": "a", "target": "b",
         "props": {"id": "T|a|b", "count": 2, "pmids": ["1"]}}
    ]  # fmt: skip
    assert [len(c) for c in batches(list(range(5)), 2)] == [2, 2, 1]
    with pytest.raises(ValueError):
        batches([1], 0)
    assert to_plain({"a": (1, [2])}) == {"a": [1, [2]]}


# --- service with a recording fake store ------------------------------------------------
class FakeReader:
    """Records (query, params) and answers from canned rows keyed by a query substring."""

    def __init__(self, answers: Mapping[str, list[dict[str, Any]]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def read(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_records: int | None = None,
        timeout_s: float | None = None,
        write: bool = False,
    ) -> list[dict[str, Any]]:
        self.calls.append((query, dict(params or {})))
        for key, rows in self.answers.items():
            if key in query:
                return rows
        return []


GENE = {"id": "HG:49590", "name": "PRKAA1", "symbol": "PRKAA1", "papers": 40}
METFORMIN = {"id": "MESH:D008687", "name": "Metformin", "papers": 900, "max_phase": 4.0,
             "chembl_id": "CHEMBL1431"}  # fmt: skip


async def test_drugs_targeting_passes_values_as_parameters() -> None:
    hostile = "AMPK'}) DETACH DELETE n //"
    fulltext_hit = [{"n": GENE, "labels": ["Gene"], "score": 2.0}]
    fake = FakeReader({"WHERE g.symbol IN $symbols RETURN g": [], "fulltext": fulltext_hit})
    await GraphService(fake).drugs_targeting(hostile, approved_only=True)
    for query, _ in fake.calls:
        assert "DETACH" not in query  # never interpolated
    curated = next(p for qy, p in fake.calls if "[t:TARGETS]" in qy)
    assert curated["symbols"] == ["PRKAA1"] and curated["approved_only"] is True


async def test_drugs_targeting_labels_sources() -> None:
    target_edge = {"id": "TARGETS|MESH:D008687|HG:49590|ACTIVATOR", "action_type": "ACTIVATOR",
                   "source_db": "chembl", "chembl_target": "CHEMBL2096907"}  # fmt: skip
    lit_edge = {"id": "STIMULATES|MESH:D008687|HG:49590", "count": 12, "pmids": ["1", "2"],
                "source_db": "pubtator"}  # fmt: skip
    fake = FakeReader(
        {
            "[t:TARGETS]": [{"d": METFORMIN, "gene": "PRKAA1", "t": target_edge}],
            "AS type ORDER BY r.count": [
                {"d": METFORMIN, "gene": "PRKAA1", "r": lit_edge, "type": "STIMULATES"}
            ],
            "p.pmid IN $pmids": [{"pmid": "1", "title": "T1", "year": 2020}],
        }
    )
    out = await GraphService(fake).drugs_targeting("AMPK", action="activator")
    assert out["untrusted"] is True and out["alias_group"] == "AMPK"
    assert [r["source"] for r in out["results"]] == ["chembl", "pubtator"]
    assert out["results"][0]["drug"]["approved"] is True
    assert out["results"][1]["pmids"] == ["1", "2"]
    assert out["papers"]["1"]["title"] == "T1"
    curated = next(p for qy, p in fake.calls if "[t:TARGETS]" in qy)
    assert "ACTIVATOR" in curated["actions"] and len(curated["symbols"]) == 7


async def test_paths_validates_before_querying() -> None:
    fake = FakeReader({})
    with pytest.raises(q.QueryValidationError):
        await GraphService(fake).paths("metformin", "glioblastoma", max_hops=5)
    with pytest.raises(q.QueryValidationError):
        await GraphService(fake).neighbors("MGMT", rel_types=["CO_OCCURS]-(x) DELETE x//"])
    assert fake.calls == []
