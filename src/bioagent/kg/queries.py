"""Cypher templates. All user-supplied values are passed as query parameters.

Relationship types, labels and hop counts cannot be parameters in Cypher, so they are only
ever taken from the whitelists below (``rel_pattern`` / ``hop_range`` validate them);
nothing a caller or a model sends is ever interpolated into query text. Raw Cypher from
callers is not supported.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from bioagent.kg.model import ALL_ENTITY_REL_TYPES, CURATED_REL_TYPES, LITERATURE_REL_TYPES

ENTITY_LABELS: tuple[str, ...] = ("Gene", "Drug", "Disease")
MAX_HOPS = 3
MAX_LIMIT = 50
FULLTEXT_INDEX = "entity_names"

ACTIVATING_ACTIONS = (
    "ACTIVATOR",
    "AGONIST",
    "PARTIAL AGONIST",
    "POSITIVE ALLOSTERIC MODULATOR",
    "OPENER",
    "STABILISER",
)
INHIBITING_ACTIONS = (
    "INHIBITOR",
    "ANTAGONIST",
    "NEGATIVE ALLOSTERIC MODULATOR",
    "BLOCKER",
    "INVERSE AGONIST",
    "DEGRADER",
    "ANTISENSE INHIBITOR",
    "RNAI INHIBITOR",
    "DISRUPTING AGENT",
)
LITERATURE_ACTIONS: dict[str, tuple[str, ...]] = {
    "activator": ("STIMULATES", "POSITIVELY_CORRELATES"),
    "inhibitor": ("INHIBITS", "NEGATIVELY_CORRELATES"),
    "any": (
        "STIMULATES",
        "POSITIVELY_CORRELATES",
        "INHIBITS",
        "NEGATIVELY_CORRELATES",
        "INTERACTS",
        "BINDS",
    ),
}
CURATED_ACTIONS: dict[str, tuple[str, ...] | None] = {
    "activator": ACTIVATING_ACTIONS,
    "inhibitor": INHIBITING_ACTIONS,
    "any": None,
}
DEFAULT_TRAVERSAL_TYPES: tuple[str, ...] = (*LITERATURE_REL_TYPES, *CURATED_REL_TYPES)


class QueryValidationError(ValueError):
    pass


def rel_pattern(types: Iterable[str]) -> str:
    """``TREATS|INHIBITS`` from whitelisted relationship types only."""
    chosen = list(dict.fromkeys(t.upper() for t in types))
    if not chosen:
        raise QueryValidationError("at least one relationship type is required")
    unknown = [t for t in chosen if t not in ALL_ENTITY_REL_TYPES]
    if unknown:
        raise QueryValidationError(
            f"unknown relationship types {unknown}; allowed: {list(ALL_ENTITY_REL_TYPES)}"
        )
    return "|".join(chosen)


def hop_range(max_hops: int) -> int:
    if not isinstance(max_hops, int) or not 1 <= max_hops <= MAX_HOPS:
        raise QueryValidationError(f"max_hops must be an integer from 1 to {MAX_HOPS}")
    return max_hops


def clamp_limit(limit: int, default: int = 20) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = default
    return max(1, min(MAX_LIMIT, value))


_LUCENE_OPERATORS = frozenset({"AND", "OR", "NOT", "TO"})
_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def lucene_query(text: str) -> str:
    """Escape user text for the fulltext index and add light fuzziness per term."""
    escaped = _LUCENE_SPECIAL.sub(r"\\\1", text.strip())
    # Upper-case AND/OR/NOT/TO are Lucene operators; the analyser lower-cases terms anyway.
    terms = [t.lower() if t in _LUCENE_OPERATORS else t for t in escaped.split()][:8]
    if not terms:
        raise QueryValidationError("empty entity name")
    phrase = " ".join(terms)
    fuzzy = " AND ".join(f"{t}~1" if len(t) >= 5 else t for t in terms)
    return f'"{phrase}"^3 OR ({fuzzy})'


# --- schema ----------------------------------------------------------------------------
def schema_statements() -> list[str]:
    stmts = [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT paper_pmid IF NOT EXISTS FOR (p:Paper) REQUIRE p.pmid IS UNIQUE",
        "CREATE INDEX gene_symbol IF NOT EXISTS FOR (g:Gene) ON (g.symbol)",
        "CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)",
        "CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name)",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
        f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS FOR (n:Gene|Drug|Disease) "
        "ON EACH [n.name, n.symbol, n.synonyms_text]",
    ]
    for rel in (*ALL_ENTITY_REL_TYPES,):
        stmts.append(
            f"CREATE INDEX rel_{rel.lower()}_id IF NOT EXISTS FOR ()-[r:{rel}]-() ON (r.id)"
        )
    return stmts


# --- loading ---------------------------------------------------------------------------
def upsert_nodes(label: str) -> str:
    if label not in ENTITY_LABELS:
        raise QueryValidationError(f"unknown label {label}")
    return f"UNWIND $rows AS row MERGE (n:Entity {{id: row.id}}) SET n:{label} SET n += row"


UPSERT_PAPERS = "UNWIND $rows AS row MERGE (p:Paper {pmid: row.pmid}) SET p += row"
UPSERT_MENTIONS = (
    "UNWIND $rows AS row MATCH (p:Paper {pmid: row.pmid}) MATCH (e:Entity {id: row.id}) "
    "MERGE (p)-[r:MENTIONS]->(e) SET r.count = row.count"
)


def upsert_edges(rel_type: str) -> str:
    rel = rel_pattern([rel_type])
    if "|" in rel:
        raise QueryValidationError("one relationship type per upsert")
    return (
        "UNWIND $rows AS row MATCH (a:Entity {id: row.source}) MATCH (b:Entity {id: row.target}) "
        f"MERGE (a)-[r:{rel} {{id: row.id}}]->(b) SET r += row.props"
    )


COUNT_NODES = "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n"
COUNT_EDGES = "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n"

# --- read queries ----------------------------------------------------------------------
FIND_EXACT = (
    "MATCH (n:Entity) WHERE n.id = $raw OR toLower(n.name) = $lower OR n.symbol = $upper "
    "RETURN n, labels(n) AS labels, 10.0 AS score ORDER BY n.papers DESC LIMIT $limit"
)
FIND_FULLTEXT = (
    f"CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX}', $lucene) YIELD node, score "
    "RETURN node AS n, labels(node) AS labels, score ORDER BY score DESC, n.papers DESC "
    "LIMIT $limit"
)
GENES_BY_SYMBOL = (
    "MATCH (g:Gene) WHERE g.symbol IN $symbols RETURN g AS n, labels(g) AS labels, "
    "1.0 AS score ORDER BY g.papers DESC"
)
GET_ENTITY = "MATCH (n:Entity {id: $id}) RETURN n, labels(n) AS labels"


def neighbors_query(rel_types: Sequence[str]) -> str:
    return (
        f"MATCH (n:Entity {{id: $id}})-[r:{rel_pattern(rel_types)}]-(m:Entity) "
        "WHERE coalesce(r.count, 1) >= $min_evidence "
        "AND (size($labels) = 0 OR any(l IN labels(m) WHERE l IN $labels)) "
        "RETURN r, type(r) AS type, startNode(r).id = n.id AS outgoing, m, labels(m) AS labels "
        "ORDER BY coalesce(r.count, 1) DESC, m.papers DESC LIMIT $limit"
    )


def drugs_targeting_curated(filter_actions: bool) -> str:
    action = "AND t.action_type IN $actions " if filter_actions else ""
    return (
        "MATCH (d:Drug)-[t:TARGETS]->(g:Gene) WHERE g.symbol IN $symbols "
        f"{action}AND (NOT $approved_only OR coalesce(d.max_phase, 0) >= 4) "
        "RETURN d, g.symbol AS gene, t LIMIT 500"
    )


def drugs_targeting_literature(rel_types: Sequence[str]) -> str:
    return (
        f"MATCH (d:Drug)-[r:{rel_pattern(rel_types)}]->(g:Gene) WHERE g.symbol IN $symbols "
        "AND coalesce(r.count, 1) >= $min_evidence "
        "AND (NOT $approved_only OR coalesce(d.max_phase, 0) >= 4) "
        "RETURN d, g.symbol AS gene, r, type(r) AS type "
        "ORDER BY r.count DESC LIMIT 1000"
    )


def paths_query(hops: int, rel_types: Sequence[str]) -> str:
    h = hop_range(hops)
    return (
        "MATCH (s:Entity {id: $source}) MATCH (t:Entity {id: $target}) "
        f"MATCH p = (s)-[rels:{rel_pattern(rel_types)}*1..{h}]-(t) "
        "WHERE all(r IN rels WHERE coalesce(r.count, 1) >= $min_evidence) "
        "AND all(x IN nodes(p) WHERE single(y IN nodes(p) WHERE y = x)) "
        "WITH p, reduce(m = 1000000, r IN relationships(p) | "
        "CASE WHEN coalesce(r.count, 1) < m THEN coalesce(r.count, 1) ELSE m END) AS strength "
        "RETURN [n IN nodes(p) | {id: n.id, name: n.name, label: [l IN labels(n) "
        "WHERE l <> 'Entity'][0]}] AS nodes, "
        "[r IN relationships(p) | {id: r.id, type: type(r), source: startNode(r).id, "
        "target: endNode(r).id, count: coalesce(r.count, 1), pmids: r.pmids[0..5], "
        "source_db: r.source_db, action_type: r.action_type, mechanism: r.mechanism}] AS edges, "
        "strength "
        "ORDER BY length(p) ASC, strength DESC LIMIT $limit"
    )


def evidence_query(edge_id: str) -> str:
    rel_type = edge_id.split("|", 1)[0]
    return (
        f"MATCH (a:Entity)-[r:{rel_pattern([rel_type])} {{id: $edge_id}}]->(b:Entity) "
        "RETURN r, type(r) AS type, a, labels(a) AS a_labels, b, labels(b) AS b_labels"
    )


PAPERS_BY_PMID = (
    "MATCH (p:Paper) WHERE p.pmid IN $pmids RETURN p.pmid AS pmid, p.title AS title, p.year AS year"
)

DRUGS_MENTIONED = (
    "MATCH (p:Paper)-[:MENTIONS]->(d:Drug) WHERE p.pmid IN $pmids "
    "AND (NOT $approved_only OR coalesce(d.max_phase, 0) >= 4) "
    "WITH d, collect(DISTINCT p.pmid) AS pmids "
    "RETURN d, pmids, size(pmids) AS n ORDER BY n DESC, d.papers DESC LIMIT $limit"
)
