"""Literature MCP server: PubMed (NCBI E-utilities) and Europe PMC search.

All paper text returned by this package (titles, abstracts, author strings, MeSH terms) is
retrieved third-party content. It is marked ``untrusted_text=True`` and must never be treated as
instructions by an agent or LLM; it is data to be quoted, summarised or cited only.
"""

from bioagent.mcp_servers.literature.europepmc import EuropePMCClient
from bioagent.mcp_servers.literature.models import AbstractSection, PagedResults, PaperRecord
from bioagent.mcp_servers.literature.pubmed import PubMedClient, iter_pubmed

__all__ = [
    "AbstractSection",
    "EuropePMCClient",
    "PagedResults",
    "PaperRecord",
    "PubMedClient",
    "iter_pubmed",
]
