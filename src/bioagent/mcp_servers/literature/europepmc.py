"""Europe PMC REST search client (``resultType=core`` with cursorMark paging)."""

from __future__ import annotations

import html
import re
from typing import Any

from bioagent.common.httpclient import RateLimitedClient
from bioagent.mcp_servers.literature.models import AbstractSection, PagedResults, PaperRecord

EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUROPEPMC_HOST = "www.ebi.ac.uk"
EUROPEPMC_RATE = 10.0
MAX_PAGE_SIZE = 100

_TAG_RE = re.compile(r"<[^>]+>")
_H4_RE = re.compile(r"<h4>(.*?)</h4>", re.IGNORECASE | re.DOTALL)


def _clean(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = " ".join(html.unescape(_TAG_RE.sub(" ", text)).split())
    return cleaned or None


def parse_abstract(raw: str | None) -> list[AbstractSection]:
    """Split a Europe PMC abstract into sections using its ``<h4>`` headings, if present."""
    if not raw:
        return []
    parts = _H4_RE.split(raw)
    sections: list[AbstractSection] = []
    lead = _clean(parts[0])
    if lead:
        sections.append(AbstractSection(text=lead))
    for label, body in zip(parts[1::2], parts[2::2], strict=False):
        text = _clean(body)
        if text:
            sections.append(AbstractSection(label=_clean(label), text=text))
    return sections


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_result(item: dict[str, Any]) -> PaperRecord:
    """Convert one Europe PMC ``core`` result into a ``PaperRecord``."""
    sections = parse_abstract(item.get("abstractText"))
    abstract = "\n".join(f"{s.label}: {s.text}" if s.label else s.text for s in sections) or None
    author_list = (item.get("authorList") or {}).get("author") or []
    authors = [a["fullName"] for a in author_list if isinstance(a, dict) and a.get("fullName")]
    if not authors and item.get("authorString"):
        authors = [a.strip() for a in str(item["authorString"]).rstrip(".").split(",") if a.strip()]
    journal = ((item.get("journalInfo") or {}).get("journal") or {}).get("title")
    mesh_headings = (item.get("meshHeadingList") or {}).get("meshHeading") or []
    mesh = [m["descriptorName"] for m in mesh_headings if m.get("descriptorName")]
    return PaperRecord(
        source="europepmc",
        pmid=item.get("pmid"),
        pmcid=item.get("pmcid"),
        title=_clean(item.get("title")),
        abstract=abstract,
        abstract_sections=sections,
        authors=authors,
        journal=journal,
        year=_int(item.get("pubYear")),
        doi=item.get("doi"),
        mesh_terms=mesh,
    )


class EuropePMCClient:
    """Async Europe PMC search returning the shared ``PaperRecord`` schema."""

    def __init__(self, http: RateLimitedClient, *, base_url: str = EUROPEPMC_BASE) -> None:
        self._http = http
        self._base = base_url.rstrip("/")

    async def search(
        self, query: str, *, page_size: int = 25, page_token: str | None = None
    ) -> PagedResults:
        """Search Europe PMC. ``page_token`` is the opaque ``cursorMark`` from a prior page."""
        cursor = page_token or "*"
        response = await self._http.get(
            f"{self._base}/search",
            params={
                "query": query,
                "resultType": "core",
                "format": "json",
                "pageSize": page_size,
                "cursorMark": cursor,
            },
        )
        payload = response.json()
        results = (payload.get("resultList") or {}).get("result") or []
        items = [parse_result(r) for r in results]
        next_cursor = payload.get("nextCursorMark")
        next_token = next_cursor if items and next_cursor and next_cursor != cursor else None
        return PagedResults(
            items=items, total=int(payload.get("hitCount", 0)), next_page_token=next_token
        )
