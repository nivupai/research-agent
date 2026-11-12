"""PubTator3 BioC-JSON export: fetch (rate-limited, cached per PMID) and parse.

API: https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson
Each cached file ``<cache>/<pmid>.json`` holds one BioC document; PMIDs PubTator did not
return are recorded in ``_missing.json`` so reruns stay offline.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from bioagent.common.httpclient import RateLimitedClient, UpstreamAPIError
from bioagent.kg.model import KIND_ORDER, EntityKind, Mention, ParsedDoc, Relation

PUBTATOR_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
PUBTATOR_HOST = "www.ncbi.nlm.nih.gov"
PUBTATOR_RATE = 3.0  # PubTator asks for at most 3 requests per second
BATCH = 100
MISSING_FILE = "_missing.json"

_KINDS: dict[str, EntityKind] = {"Gene": "Gene", "Chemical": "Drug", "Disease": "Disease"}


def _first_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("-", "None"):
        return None
    return text.split(";")[0].split("|")[0].strip() or None


def normalise_entity(infons: dict[str, Any], text: str = "") -> Mention | None:
    """Map a PubTator annotation/role to a node id; None for other types or unlinked ones."""
    kind = _KINDS.get(str(infons.get("type")))
    if kind is None:
        return None
    ident = _first_id(infons.get("identifier"))
    if ident is None:
        return None
    name = str(infons.get("name") or text or ident).strip()
    if kind == "Gene":
        ncbi = ident.split(":")[-1]
        if not ncbi.isdigit():
            return None
        homologene = _first_id(infons.get("ncbi_homologene"))
        node_id = f"HG:{homologene}" if homologene else f"NCBIGene:{ncbi}"
        return Mention("Gene", node_id, name.upper(), text or name, ncbi, homologene)
    if not ident.upper().startswith("MESH:"):
        ident = f"MESH:{ident}"
    return Mention(kind, "MESH:" + ident.split(":", 1)[1], name, text or name)


def _resolve_gene(m: Mention | None, by_ncbi: dict[str, Mention]) -> Mention | None:
    if m is None or m.kind != "Gene" or m.homologene or m.ncbi_gene not in by_ncbi:
        return m
    return by_ncbi[m.ncbi_gene]


def parse_document(doc: dict[str, Any]) -> ParsedDoc:
    pmid = str(doc.get("pmid") or doc.get("id"))
    parsed = ParsedDoc(pmid)
    for passage in doc.get("passages", []):
        for ann in passage.get("annotations", []):
            m = normalise_entity(ann.get("infons", {}), str(ann.get("text", "")))
            if m is not None:
                parsed.mentions.append(m)
    # Relation roles lack the HomoloGene id; reuse the id of the same gene's mentions.
    gene_ids = {m.ncbi_gene: m for m in parsed.mentions if m.ncbi_gene}
    relations = doc.get("relations", []) or []
    display = doc.get("relations_display", []) or []
    for i, rel in enumerate(relations):
        infons = rel.get("infons", {})
        a = _resolve_gene(normalise_entity(infons.get("role1") or {}), gene_ids)
        b = _resolve_gene(normalise_entity(infons.get("role2") or {}), gene_ids)
        if a is None or b is None or a.id == b.id:
            continue
        verb = None
        if len(display) == len(relations):
            verb = str(display[i].get("name", "")).split("|")[0] or None
        verb = (verb or str(infons.get("type", "related"))).lower()
        # canonical orientation: Drug -> Gene -> Disease, then by id
        if (KIND_ORDER[a.kind], a.id) > (KIND_ORDER[b.kind], b.id):
            a, b = b, a
        try:
            score = float(infons["score"]) if "score" in infons else None
        except (TypeError, ValueError):
            score = None
        parsed.relations.append(Relation(verb, a.id, b.id, score))
    return parsed


class PubTatorFetcher:
    def __init__(self, http: RateLimitedClient, cache_dir: Path) -> None:
        self.http = http
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(self, pmid: str) -> Path:
        return self.cache_dir / f"{pmid}.json"

    def missing_recorded(self) -> set[str]:
        path = self.cache_dir / MISSING_FILE
        return set(json.loads(path.read_text())) if path.exists() else set()

    def _record_missing(self, pmids: Iterable[str]) -> None:
        path = self.cache_dir / MISSING_FILE
        current = self.missing_recorded() | set(pmids)
        path.write_text(json.dumps(sorted(current)))

    def todo(self, pmids: Sequence[str]) -> list[str]:
        known_missing = self.missing_recorded()
        return [p for p in pmids if p not in known_missing and not self.cached(p).exists()]

    async def fetch_batch(self, pmids: Sequence[str]) -> list[dict[str, Any]]:
        response = await self.http.get(PUBTATOR_URL, params={"pmids": ",".join(pmids)})
        payload = response.json()
        docs = payload.get("PubTator3", payload) if isinstance(payload, dict) else payload
        return [d for d in docs if isinstance(d, dict)]

    async def fetch_all(
        self, pmids: Sequence[str], log: Callable[[str], None] = print
    ) -> dict[str, int]:
        todo = self.todo(pmids)
        fetched = missing = errors = 0
        for start in range(0, len(todo), BATCH):
            batch = todo[start : start + BATCH]
            try:
                docs = await self.fetch_batch(batch)
            except UpstreamAPIError as exc:
                errors += 1
                log(f"  pubtator batch {start // BATCH} failed: {exc}")
                continue
            got = set()
            for doc in docs:
                pmid = str(doc.get("pmid") or doc.get("id"))
                self.cached(pmid).write_text(json.dumps(doc))
                got.add(pmid)
            lost = [p for p in batch if p not in got]
            if lost:
                self._record_missing(lost)
            fetched += len(got)
            missing += len(lost)
            if (start // BATCH) % 10 == 0:
                log(f"  pubtator {start + len(batch)}/{len(todo)} (missing so far {missing})")
        return {
            "requested": len(todo),
            "fetched": fetched,
            "missing": missing,
            "failed_batches": errors,
            "already_cached": len(pmids) - len(todo),
        }

    def load(self, pmid: str) -> ParsedDoc | None:
        path = self.cached(pmid)
        if not path.exists():
            return None
        return parse_document(json.loads(path.read_text()))
