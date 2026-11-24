"""Knowledge-graph check for drug-target claims (``GRAPH_CLAIM_CHECK=on``; default off).

For a "which drugs act on target X" question, answer sentences such as "Metformin activates
AMPK [3]" often fail NLI because the cited passage does not say it word for word, although
the graph holds the relation (failure analysis sections 17-18). This check runs before NLI:

* the drugs with a drug->target relation in Neo4j are looked up once per answer: curated
  ChEMBL mechanisms (``TARGETS``) and text-mined PubTator relations with an action
  (``STIMULATES``, ``INHIBITS``, ``POSITIVELY_/NEGATIVELY_CORRELATES``, ``INTERACTS``,
  ``BINDS``) with at least ``min_evidence`` supporting papers. Co-occurrence and generic
  association edges do not count.
* a sentence that names one of those drugs (name or graph synonym) and the target (or an
  alias) is graph-verified: the claims in it that name the drug are labelled ``graph``,
  cite the relation and skip NLI, and the sentence is never removed (other claims in it
  still go through NLI and can get it flagged).

Everything else is verified exactly as before. No LLM is involved.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

from bioagent.kg.aliases import expand_gene_query

GRAPH_REL_TYPES: tuple[str, ...] = (
    "TARGETS",
    "STIMULATES",
    "INHIBITS",
    "POSITIVELY_CORRELATES",
    "NEGATIVELY_CORRELATES",
    "INTERACTS",
    "BINDS",
)
CURATED_TYPES = frozenset({"TARGETS"})
MIN_DRUG_TERM = 4  # skip very short names/synonyms ("Met", "CA")
MAX_ROWS = 5000

RELATIONS_QUERY = (
    "MATCH (d:Drug)-[r]->(g:Gene) WHERE g.symbol IN $symbols AND type(r) IN $types "
    "AND (type(r) = 'TARGETS' OR coalesce(r.count, 1) >= $min_evidence) "
    "RETURN d.id AS id, d.name AS name, d.chembl_name AS chembl_name, d.synonyms AS synonyms, "
    "type(r) AS type, g.symbol AS gene, coalesce(r.count, 1) AS count, r.id AS edge_id, "
    "r.action_type AS action_type, r.source_db AS source_db "
    "ORDER BY id LIMIT $limit"
)


@dataclass(frozen=True)
class GraphRelation:
    drug: str
    gene: str
    type: str
    count: int = 1
    edge_id: str | None = None
    action_type: str | None = None

    @property
    def source(self) -> str:
        return "chembl" if self.type in CURATED_TYPES else "pubtator"

    def describe(self) -> str:
        what = self.action_type.lower() if self.action_type else self.type.lower()
        papers = "" if self.source == "chembl" else f", {self.count} papers"
        return f"graph: {self.drug} {what} {self.gene} ({self.source}{papers})"

    def row(self) -> dict[str, Any]:
        return {
            "ref": f"graph:{self.edge_id or self.type}",
            "source": f"graph/{self.source}",
            "relation": self.type,
            "gene": self.gene,
            "action_type": self.action_type,
            "count": self.count,
        }


@dataclass
class GraphDrug:
    id: str
    name: str
    terms: list[str]
    relations: list[GraphRelation] = field(default_factory=list)

    def best(self) -> GraphRelation:
        """Curated relation first, then the text-mined one with most papers."""
        return min(self.relations, key=lambda r: (r.source != "chembl", -r.count))


class GraphLookup(Protocol):
    async def drugs_for_target(self, target: str) -> list[GraphDrug]: ...


class ReadStore(Protocol):
    async def read(
        self, query: str, params: Mapping[str, Any] | None = None, *, max_records: int | None = None
    ) -> list[dict[str, Any]]: ...


StoreFactory = Callable[[], AbstractAsyncContextManager[ReadStore]]


# Target families the alias groups in kg.aliases do not cover; without them "PARP" or
# "COX-2" resolve to no gene (or the wrong one) and the check could never fire. Local to
# this check: discovery's drugs_targeting is unchanged.
FAMILIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # key: (gene symbols, extra text terms)
    "HDAC": (
        tuple(f"HDAC{i}" for i in range(1, 12)),
        ("histone deacetylase", "histone deacetylases"),
    ),
    "PARP": (("PARP1", "PARP2", "PARP3"), ("poly(ADP-ribose) polymerase", "PARP-1", "PARP-2")),
    "VEGFR": (
        ("KDR", "FLT1", "FLT4"),
        ("VEGF receptor", "VEGF receptors", "VEGFR-1", "VEGFR-2", "VEGFR-3", "VEGFR1",
         "VEGFR2", "VEGFR3", "VEGFRs"),
    ),
    "COX-2": (("PTGS2",), ("COX2", "cyclooxygenase-2", "cyclooxygenase 2", "COX 2")),
    "EGFR": (("EGFR",), ("epidermal growth factor receptor", "ErbB1", "HER1")),
}  # fmt: skip


def _family(target: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    key = " ".join(target.strip().upper().split())
    return next((v for k, v in FAMILIES.items() if k.upper() == key), None)


def target_symbols(target: str) -> list[str]:
    family = _family(target)
    if family is not None:
        return list(family[0])
    return expand_gene_query(target)[1]


def group_rows(rows: Iterable[Mapping[str, Any]]) -> list[GraphDrug]:
    drugs: dict[str, GraphDrug] = {}
    for row in rows:
        name = str(row.get("name") or row.get("chembl_name") or "")
        if not name:
            continue
        drug_id = str(row.get("id") or name)
        drug = drugs.get(drug_id)
        if drug is None:
            names = [name, row.get("chembl_name"), *(row.get("synonyms") or [])]
            terms = {str(t).strip() for t in names if t and len(str(t).strip()) >= MIN_DRUG_TERM}
            drug = drugs[drug_id] = GraphDrug(drug_id, name, sorted(terms, key=len, reverse=True))
        drug.relations.append(
            GraphRelation(
                drug=name,
                gene=str(row.get("gene") or ""),
                type=str(row.get("type") or ""),
                count=int(row.get("count") or 1),
                edge_id=str(row["edge_id"]) if row.get("edge_id") else None,
                action_type=str(row["action_type"]) if row.get("action_type") else None,
            )
        )
    return [d for d in drugs.values() if d.terms]


class Neo4jGraphLookup:
    """Reads drug->target relations from the knowledge graph (one query per target).

    ``open_store`` opens a store for the lookup and closes it afterwards (a fresh Neo4j
    driver per run; the result is cached per target)."""

    def __init__(self, open_store: StoreFactory, min_evidence: int = 2) -> None:
        self.open_store = open_store
        self.min_evidence = max(1, int(min_evidence))
        self._cache: dict[str, list[GraphDrug]] = {}

    async def drugs_for_target(self, target: str) -> list[GraphDrug]:
        key = target.strip().lower()
        if key not in self._cache:
            async with self.open_store() as store:
                rows = await store.read(
                    RELATIONS_QUERY,
                    {
                        "symbols": target_symbols(target),
                        "types": list(GRAPH_REL_TYPES),
                        "min_evidence": self.min_evidence,
                        "limit": MAX_ROWS,
                    },
                    max_records=MAX_ROWS,
                )
            self._cache[key] = group_rows(rows)
        return self._cache[key]


def _pattern(terms: Iterable[str]) -> re.Pattern[str] | None:
    parts = [re.escape(t) for t in sorted(set(terms), key=len, reverse=True) if t]
    if not parts:
        return None
    return re.compile(r"(?<![\w-])(?:" + "|".join(parts) + r")(?![\w-])", re.I)


def target_pattern(target: str) -> re.Pattern[str] | None:
    from bioagent.agents.evidence_screen import target_terms

    terms = set(target_terms(target))
    family = _family(target)
    if family is not None:
        terms.update(family[0])
        terms.update(family[1])
    parts = [re.escape(t) for t in sorted(terms, key=len, reverse=True) if t]
    if not parts:
        return None
    # plural forms ("HDACs", "PARPs") count as the target
    return re.compile(r"(?<![\w-])(?:" + "|".join(parts) + r")s?(?![\w-])", re.I)


@dataclass(frozen=True)
class GraphMatch:
    sentence: int
    drugs: tuple[GraphDrug, ...]

    def relations(self) -> list[GraphRelation]:
        return [d.best() for d in self.drugs]


class GraphChecker:
    """Finds graph-verified sentences; holds the compiled drug and target patterns."""

    def __init__(self, drugs: Sequence[GraphDrug], target: str) -> None:
        self.target = target
        self.target_re = target_pattern(target)
        self.drugs = [d for d in drugs if d.relations]
        self._patterns = {d.id: _pattern(d.terms) for d in self.drugs}

    def drugs_named(self, text: str) -> list[GraphDrug]:
        return [d for d in self.drugs if (p := self._patterns[d.id]) is not None and p.search(text)]

    def names_target(self, text: str) -> bool:
        return self.target_re is not None and self.target_re.search(text) is not None

    def match(self, sentences: Sequence[tuple[int, str]]) -> dict[int, GraphMatch]:
        """Sentence id -> the graph drugs it names, for sentences that also name the target."""
        out: dict[int, GraphMatch] = {}
        for sid, text in sentences:
            if not self.names_target(text):
                continue
            named = self.drugs_named(text)
            if named:
                out[sid] = GraphMatch(sid, tuple(named))
        return out


def lookup_from_settings(settings: Any) -> Neo4jGraphLookup:
    """Graph lookup on the configured Neo4j (``settings``: ``bioagent.config.Settings``)."""
    from bioagent.kg.store import GraphStore

    def open_store() -> GraphStore:
        store = GraphStore.from_settings(settings)
        store.max_records = max(store.max_records, MAX_ROWS)
        return store

    return Neo4jGraphLookup(open_store, min_evidence=settings.discovery_min_evidence)
