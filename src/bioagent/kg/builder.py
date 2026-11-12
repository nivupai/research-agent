"""Aggregate parsed PubTator documents (+ ChEMBL) into graph rows for the loader."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from bioagent.kg.aliases import alias_groups_for
from bioagent.kg.model import KIND_ORDER, ParsedDoc

MAX_EDGE_PMIDS = 50
MIN_COOCCURRENCE = 2


@dataclass
class _Node:
    kind: str
    names: Counter[str] = field(default_factory=Counter)
    texts: Counter[str] = field(default_factory=Counter)
    papers: set[str] = field(default_factory=set)
    ncbi_ids: set[str] = field(default_factory=set)
    homologene: str | None = None


@dataclass
class _Edge:
    pmids: list[str] = field(default_factory=list)
    papers: set[str] = field(default_factory=set)
    max_score: float | None = None


@dataclass
class GraphRows:
    nodes: dict[str, list[dict[str, Any]]]  # label -> rows
    papers: list[dict[str, Any]]
    mentions: list[dict[str, Any]]
    relations: dict[str, list[dict[str, Any]]]  # rel type -> rows
    cooccurs: list[dict[str, Any]]
    targets: list[dict[str, Any]]

    def counts(self) -> dict[str, Any]:
        return {
            "nodes": {k: len(v) for k, v in self.nodes.items()} | {"Paper": len(self.papers)},
            "edges": {
                "MENTIONS": len(self.mentions),
                "CO_OCCURS": len(self.cooccurs),
                "TARGETS": len(self.targets),
            }
            | {k: len(v) for k, v in sorted(self.relations.items())},
        }


def edge_id(rel_type: str, source: str, target: str, extra: str = "") -> str:
    return f"{rel_type}|{source}|{target}" + (f"|{extra}" if extra else "")


class GraphBuilder:
    def __init__(self, min_cooccurrence: int = MIN_COOCCURRENCE) -> None:
        self.min_cooccurrence = min_cooccurrence
        self.nodes: dict[str, _Node] = {}
        self.papers: dict[str, dict[str, Any]] = {}
        self.mentions: Counter[tuple[str, str]] = Counter()
        self.relations: dict[tuple[str, str, str], _Edge] = defaultdict(_Edge)
        self.cooccur: dict[tuple[str, str], _Edge] = defaultdict(_Edge)
        self.gene_alias: dict[str, str] = {}  # NCBIGene:x -> HG:y
        self.verbs: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

    def add(self, doc: ParsedDoc, title: str | None = None, year: int | None = None) -> None:
        self.papers[doc.pmid] = {"pmid": doc.pmid, "title": title, "year": year}
        for m in doc.mentions:
            node = self.nodes.setdefault(m.id, _Node(m.kind))
            node.names[m.name] += 1
            node.texts[m.text.strip()] += 1
            node.papers.add(doc.pmid)
            if m.ncbi_gene:
                node.ncbi_ids.add(m.ncbi_gene)
                if m.homologene:
                    node.homologene = m.homologene
                    self.gene_alias[f"NCBIGene:{m.ncbi_gene}"] = m.id
            self.mentions[(doc.pmid, m.id)] += 1
        for r in doc.relations:
            key = (r.rel_type, r.source, r.target)
            edge = self.relations[key]
            self.verbs[key][r.verb] += 1
            if doc.pmid not in edge.papers:
                edge.papers.add(doc.pmid)
                if len(edge.pmids) < MAX_EDGE_PMIDS:
                    edge.pmids.append(doc.pmid)
            if r.score is not None:
                edge.max_score = max(edge.max_score or 0.0, r.score)
        ids = sorted(doc.entity_ids)
        for a, b in itertools.combinations(ids, 2):
            edge = self.cooccur[(a, b)]
            edge.papers.add(doc.pmid)
            if len(edge.pmids) < MAX_EDGE_PMIDS:
                edge.pmids.append(doc.pmid)

    def _canon(self, node_id: str) -> str:
        return self.gene_alias.get(node_id, node_id)

    def _merge_gene_duplicates(self) -> None:
        """Fold NCBIGene-only nodes into the HomoloGene node with the same symbol."""
        by_symbol: dict[str, str] = {}
        for node_id, node in self.nodes.items():
            if node.kind == "Gene" and node_id.startswith("HG:"):
                symbol = node.names.most_common(1)[0][0].upper()
                best = by_symbol.get(symbol)
                if best is None or len(node.papers) > len(self.nodes[best].papers):
                    by_symbol[symbol] = node_id
        for node_id in [
            i for i, n in self.nodes.items() if n.kind == "Gene" and i.startswith("NCBIGene:")
        ]:
            node = self.nodes[node_id]
            target = by_symbol.get(node.names.most_common(1)[0][0].upper())
            if target is None:
                continue
            dest = self.nodes[target]
            dest.names.update(node.names)
            dest.texts.update(node.texts)
            dest.papers |= node.papers
            dest.ncbi_ids |= node.ncbi_ids
            self.gene_alias[node_id] = target
            del self.nodes[node_id]

    def build(self, chembl: dict[str, dict[str, Any]] | None = None) -> GraphRows:
        chembl = chembl or {}
        self._merge_gene_duplicates()
        nodes: dict[str, list[dict[str, Any]]] = {"Gene": [], "Drug": [], "Disease": []}
        symbol_to_gene: dict[str, str] = {}
        for node_id, node in self.nodes.items():
            name = node.names.most_common(1)[0][0]
            synonyms = [t for t, _ in node.texts.most_common(12) if t.lower() != name.lower()]
            row: dict[str, Any] = {
                "id": node_id,
                "name": name,
                "synonyms": synonyms,
                "synonyms_text": " | ".join([name, *synonyms]),
                "papers": len(node.papers),
            }
            if node.kind == "Gene":
                row.update(
                    symbol=name.upper(),
                    ncbi_ids=sorted(node.ncbi_ids),
                    homologene=node.homologene,
                    alias_groups=alias_groups_for(name),
                )
                symbol_to_gene.setdefault(name.upper(), node_id)
            if node.kind == "Drug":
                info = chembl.get(node_id) or {}
                row.update(
                    chembl_id=info.get("chembl_id"),
                    max_phase=info.get("max_phase"),
                    approved=(info.get("max_phase") or 0) >= 4,
                    chembl_name=info.get("pref_name"),
                )
            nodes[node.kind].append(row)

        relations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for (rel_type, src, tgt), edge in self.relations.items():
            src, tgt = self._canon(src), self._canon(tgt)
            if src == tgt or src not in self.nodes or tgt not in self.nodes:
                continue
            key = (rel_type, src, tgt)
            row = merged.setdefault(
                key,
                {
                    "id": edge_id(rel_type, src, tgt),
                    "source": src,
                    "target": tgt,
                    "pmids": [],
                    "max_score": None,
                    "source_db": "pubtator",
                },
            )
            row["pmids"] = list(dict.fromkeys([*row["pmids"], *edge.pmids]))[:MAX_EDGE_PMIDS]
            row["count"] = (
                len(set(row["pmids"]))
                if len(row["pmids"]) < MAX_EDGE_PMIDS
                else max(row.get("count", 0), len(edge.papers))
            )
            if edge.max_score is not None:
                row["max_score"] = max(row["max_score"] or 0.0, edge.max_score)
        for (rel_type, _s, _t), row in merged.items():
            relations[rel_type].append(row)

        pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
        for (a, b), e in self.cooccur.items():
            ca, cb = sorted((self._canon(a), self._canon(b)))
            if ca != cb:
                pairs[(ca, cb)] |= e.papers
        cooccurs = [
            {
                "id": edge_id("CO_OCCURS", a, b),
                "source": a,
                "target": b,
                "count": len(papers),
                "pmids": sorted(papers)[:MAX_EDGE_PMIDS],
            }
            for (a, b), papers in pairs.items()
            if len(papers) >= self.min_cooccurrence
        ]

        targets: list[dict[str, Any]] = []
        target_ids: set[str] = set()
        for drug_id, info in chembl.items():
            if drug_id not in self.nodes:
                continue
            for mech in info.get("mechanisms", []):
                for symbol in mech.get("genes", []):
                    gene_id = symbol_to_gene.get(symbol)
                    if gene_id is None:
                        continue  # gene not mentioned in the corpus
                    # one edge per (drug, gene, action, ChEMBL target): a complex and its
                    # single-protein target are separate curated mechanisms
                    tid = edge_id(
                        "TARGETS",
                        drug_id,
                        gene_id,
                        f"{mech['action_type']}|{mech.get('target_chembl_id')}",
                    )
                    if tid in target_ids:
                        continue
                    target_ids.add(tid)
                    targets.append(
                        {
                            "id": tid,
                            "source": drug_id,
                            "target": gene_id,
                            "action_type": mech["action_type"],
                            "mechanism": mech.get("mechanism_of_action"),
                            "chembl_target": mech.get("target_chembl_id"),
                            "chembl_target_name": mech.get("target_name"),
                            "chembl_id": info.get("chembl_id"),
                            "source_db": "chembl",
                            "pmids": [],
                            "count": 1,
                        }
                    )
        mention_counts: Counter[tuple[str, str]] = Counter()
        for (pmid, node_id), c in self.mentions.items():
            mention_counts[(pmid, self._canon(node_id))] += c  # merged genes share a node
        mentions = [{"pmid": p, "id": i, "count": c} for (p, i), c in mention_counts.items()]
        return GraphRows(
            nodes=nodes,
            papers=list(self.papers.values()),
            mentions=mentions,
            relations=dict(relations),
            cooccurs=cooccurs,
            targets=targets,
        )

    def drug_ids_by_frequency(self, min_papers: int) -> list[tuple[str, str]]:
        """(drug id, canonical name) for drugs mentioned in at least ``min_papers`` papers."""
        drugs = [
            (i, n)
            for i, n in self.nodes.items()
            if n.kind == "Drug" and len(n.papers) >= min_papers
        ]
        drugs.sort(key=lambda x: -len(x[1].papers))
        return [(i, n.names.most_common(1)[0][0]) for i, n in drugs]


def canonical_pair(a: str, kind_a: str, b: str, kind_b: str) -> tuple[str, str]:
    return (a, b) if (KIND_ORDER[kind_a], a) <= (KIND_ORDER[kind_b], b) else (b, a)
