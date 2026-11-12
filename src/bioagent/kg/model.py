"""Normalised entity and relation records shared by the KG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EntityKind = Literal["Gene", "Drug", "Disease"]
KIND_ORDER: dict[str, int] = {"Drug": 0, "Gene": 1, "Disease": 2}

# PubTator3 relation verbs (relations_display) -> Neo4j relationship types. Relationship
# types cannot be query parameters, so only these constants are ever placed in Cypher.
RELATION_TYPES: dict[str, str] = {
    "treat": "TREATS",
    "cause": "CAUSES",
    "associate": "ASSOCIATED_WITH",
    "positive_correlate": "POSITIVELY_CORRELATES",
    "negative_correlate": "NEGATIVELY_CORRELATES",
    "inhibit": "INHIBITS",
    "stimulate": "STIMULATES",
    "interact": "INTERACTS",
    "bind": "BINDS",
    "cotreat": "COTREATED_WITH",
    "drug_interact": "DRUG_INTERACTS",
    "compare": "COMPARED_WITH",
    "prevent": "PREVENTS",
    "convert": "CONVERTS",
}
FALLBACK_RELATION = "RELATED_TO"
LITERATURE_REL_TYPES: tuple[str, ...] = (*dict.fromkeys(RELATION_TYPES.values()), FALLBACK_RELATION)
CURATED_REL_TYPES: tuple[str, ...] = ("TARGETS",)
ALL_ENTITY_REL_TYPES: tuple[str, ...] = (*LITERATURE_REL_TYPES, *CURATED_REL_TYPES, "CO_OCCURS")


@dataclass(frozen=True)
class Mention:
    kind: EntityKind
    id: str  # normalised node id, e.g. MESH:D008687, HG:49590, NCBIGene:5562
    name: str  # canonical name (PubTator 'name'; gene symbol for genes)
    text: str  # surface form in the abstract
    ncbi_gene: str | None = None
    homologene: str | None = None


@dataclass(frozen=True)
class Relation:
    verb: str  # PubTator verb, e.g. "treat"
    source: str  # node id
    target: str  # node id
    score: float | None = None

    @property
    def rel_type(self) -> str:
        return RELATION_TYPES.get(self.verb, FALLBACK_RELATION)


@dataclass
class ParsedDoc:
    pmid: str
    mentions: list[Mention] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)

    @property
    def entity_ids(self) -> set[str]:
        return {m.id for m in self.mentions}
