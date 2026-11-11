"""PubMed client built on NCBI E-utilities (esearch + efetch with the History server)."""

from __future__ import annotations

import base64
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator, Sequence
from typing import Any

from bioagent.common.httpclient import LiteratureAPIError, RateLimitedClient
from bioagent.mcp_servers.literature.models import AbstractSection, PagedResults, PaperRecord

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUTILS_HOST = "eutils.ncbi.nlm.nih.gov"
TOOL_NAME = "bioagent"
MAX_FETCH_BATCH = 200
# PubMed's esearch/history paging cannot go past the first 10,000 records of a query.
PUBMED_RETRIEVAL_LIMIT = 10_000

_PMID_RE = re.compile(r"^\d{1,9}$")
_YEAR_RE = re.compile(r"(\d{4})")


def pubmed_rate(api_key: str | None) -> float:
    """NCBI allows 10 req/s with an API key, 3 req/s without."""
    return 10.0 if api_key else 3.0


def _text(elem: ET.Element | None) -> str | None:
    if elem is None:
        return None
    text = " ".join("".join(elem.itertext()).split())
    return text or None


def _year(article: ET.Element) -> int | None:
    candidates = [
        article.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year"),
        article.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate"),
        article.find("./MedlineCitation/Article/ArticleDate/Year"),
    ]
    for elem in candidates:
        value = _text(elem)
        if value and (match := _YEAR_RE.search(value)):
            return int(match.group(1))
    return None


def _author_name(author: ET.Element) -> str | None:
    collective = _text(author.find("CollectiveName"))
    if collective:
        return collective
    last = _text(author.find("LastName"))
    first = _text(author.find("ForeName")) or _text(author.find("Initials"))
    if last and first:
        return f"{last} {first}"
    return last


def parse_pubmed_article(article: ET.Element) -> PaperRecord:
    """Convert one ``<PubmedArticle>`` element into a ``PaperRecord``."""
    citation = article.find("MedlineCitation")
    art = citation.find("Article") if citation is not None else None

    sections: list[AbstractSection] = []
    if art is not None:
        for node in art.findall("./Abstract/AbstractText"):
            text = _text(node)
            if text:
                sections.append(AbstractSection(label=node.get("Label") or None, text=text))
    abstract = "\n".join(f"{s.label}: {s.text}" if s.label else s.text for s in sections) or None

    ids = {
        (aid.get("IdType") or ""): (aid.text or "").strip()
        for aid in article.findall("./PubmedData/ArticleIdList/ArticleId")
    }
    doi = ids.get("doi") or None
    if not doi and art is not None:
        for loc in art.findall("ELocationID"):
            if loc.get("EIdType") == "doi" and loc.text:
                doi = loc.text.strip()
                break

    authors = []
    if art is not None:
        for author in art.findall("./AuthorList/Author"):
            name = _author_name(author)
            if name:
                authors.append(name)

    mesh = [
        t
        for d in article.findall("./MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName")
        if (t := _text(d))
    ]

    return PaperRecord(
        source="pubmed",
        pmid=_text(citation.find("PMID")) if citation is not None else None,
        pmcid=ids.get("pmc") or None,
        title=_text(art.find("ArticleTitle")) if art is not None else None,
        abstract=abstract,
        abstract_sections=sections,
        authors=authors,
        journal=_text(art.find("./Journal/Title")) if art is not None else None,
        year=_year(article),
        doi=doi,
        mesh_terms=mesh,
    )


def parse_efetch_xml(xml_text: str | bytes) -> list[PaperRecord]:
    """Parse an efetch ``PubmedArticleSet`` document."""
    root = ET.fromstring(xml_text)
    return [parse_pubmed_article(a) for a in root.iter("PubmedArticle")]


def _encode_token(data: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).decode()


def _decode_token(token: str) -> dict[str, Any]:
    try:
        data = json.loads(base64.urlsafe_b64decode(token.encode()))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid page_token") from exc
    required = {"webenv", "query_key", "retstart", "total"}
    if not isinstance(data, dict) or not required <= data.keys():
        raise ValueError("invalid page_token")
    return data


class PubMedClient:
    """Async PubMed search/fetch. Sends ``tool``, ``email`` and ``api_key`` to NCBI."""

    def __init__(
        self,
        http: RateLimitedClient,
        *,
        api_key: str | None = None,
        email: str | None = None,
        base_url: str = EUTILS_BASE,
    ) -> None:
        self._http = http
        self._api_key = api_key
        self._email = email
        self._base = base_url.rstrip("/")

    def _params(self, **extra: Any) -> dict[str, Any]:
        params: dict[str, Any] = {"db": "pubmed", "tool": TOOL_NAME}
        if self._email:
            params["email"] = self._email
        if self._api_key:
            params["api_key"] = self._api_key
        params.update({k: v for k, v in extra.items() if v is not None})
        return params

    async def esearch(
        self,
        query: str,
        *,
        retmax: int = 20,
        retstart: int = 0,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> dict[str, Any]:
        """Run esearch with ``usehistory=y``; returns the ``esearchresult`` JSON object."""
        date: dict[str, Any] = {}
        if min_year is not None or max_year is not None:
            date = {
                "datetype": "pdat",
                "mindate": str(min_year or 1800),
                "maxdate": str(max_year or 3000),
            }
        response = await self._http.get(
            f"{self._base}/esearch.fcgi",
            params=self._params(
                term=query,
                retmode="json",
                usehistory="y",
                retmax=retmax,
                retstart=retstart,
                **date,
            ),
        )
        payload = response.json()
        result = payload.get("esearchresult")
        if not isinstance(result, dict):
            raise LiteratureAPIError(f"unexpected esearch response: {payload.get('error')}")
        if "ERROR" in result:
            raise LiteratureAPIError(f"esearch error: {result['ERROR']}")
        return result

    async def efetch_ids(self, pmids: Sequence[str]) -> list[PaperRecord]:
        """Fetch full records for explicit PMIDs (batched by ``MAX_FETCH_BATCH``)."""
        records: list[PaperRecord] = []
        for i in range(0, len(pmids), MAX_FETCH_BATCH):
            chunk = pmids[i : i + MAX_FETCH_BATCH]
            response = await self._http.get(
                f"{self._base}/efetch.fcgi",
                params=self._params(id=",".join(chunk), retmode="xml", rettype="abstract"),
            )
            records.extend(parse_efetch_xml(response.content))
        return records

    async def efetch_history(
        self, webenv: str, query_key: str, *, retstart: int, retmax: int
    ) -> list[PaperRecord]:
        """Fetch a slice of a stored History-server result set."""
        response = await self._http.get(
            f"{self._base}/efetch.fcgi",
            params=self._params(
                WebEnv=webenv,
                query_key=query_key,
                retstart=retstart,
                retmax=retmax,
                retmode="xml",
                rettype="abstract",
            ),
        )
        return parse_efetch_xml(response.content)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        page_token: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> PagedResults:
        """Search PubMed and return one page of full records.

        The first call runs esearch (storing results on the History server); subsequent pages
        page through that stored set with efetch ``retstart``/``retmax`` via ``page_token``.
        """
        if page_token:
            state = _decode_token(page_token)
            webenv, query_key = state["webenv"], state["query_key"]
            retstart, total = int(state["retstart"]), int(state["total"])
            items = await self.efetch_history(
                webenv, query_key, retstart=retstart, retmax=max_results
            )
        else:
            result = await self.esearch(
                query, retmax=max_results, min_year=min_year, max_year=max_year
            )
            total = int(result.get("count", 0))
            webenv, query_key, retstart = result.get("webenv"), result.get("querykey"), 0
            ids = list(result.get("idlist", []))
            items = await self.efetch_ids(ids) if ids else []

        next_start = retstart + max_results
        reachable = min(total, PUBMED_RETRIEVAL_LIMIT)
        next_token = None
        if items and webenv and query_key and next_start < reachable:
            next_token = _encode_token(
                {"webenv": webenv, "query_key": query_key, "retstart": next_start, "total": total}
            )
        return PagedResults(items=items, total=total, next_page_token=next_token)


async def iter_pubmed(
    client: PubMedClient, query: str, limit: int, *, batch_size: int = MAX_FETCH_BATCH
) -> AsyncIterator[PaperRecord]:
    """Yield up to ``limit`` records for ``query`` using the History server.

    Intended for bulk ingestion (e.g. ~10k abstracts). PubMed only exposes the first 10,000
    records of a query; split larger jobs by date range.
    """
    result = await client.esearch(query, retmax=0)
    total = min(int(result.get("count", 0)), limit, PUBMED_RETRIEVAL_LIMIT)
    webenv, query_key = result.get("webenv"), result.get("querykey")
    if not webenv or not query_key:
        return
    yielded = 0
    for retstart in range(0, total, batch_size):
        batch = await client.efetch_history(
            webenv, query_key, retstart=retstart, retmax=min(batch_size, total - retstart)
        )
        if not batch:
            return
        for record in batch:
            yield record
            yielded += 1
            if yielded >= total:
                return


def validate_pmids(pmids: Sequence[str]) -> list[str]:
    """Strip and validate PMIDs; raises ``ValueError`` on anything non-numeric."""
    cleaned = [p.strip() for p in pmids if p and p.strip()]
    bad = [p for p in cleaned if not _PMID_RE.match(p)]
    if bad:
        raise ValueError(f"invalid PMIDs: {bad[:5]}")
    return list(dict.fromkeys(cleaned))
