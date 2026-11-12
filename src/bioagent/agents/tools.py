"""Tool access for the specialists.

``MCPToolHub`` starts our four MCP servers over stdio and exposes their tools as LangChain
``StructuredTool``s (using each server's JSON schema). ``TOOL_SCOPES`` decides which tools
each specialist may see *and* call: a specialist is only bound to its own tools, and calls
to anything else are refused before execution.

The knowledge-graph tools (``bioagent-graph``) belong to the literature specialist: every
graph edge is text-mined from (or, for ChEMBL mechanisms, curated alongside) the same
PubMed literature that specialist already searches and cites, so graph results become
PMID-backed evidence and the specialist can check an edge against the abstracts.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool

from bioagent.agents.state import Evidence

GRAPH_TOOLS: tuple[str, ...] = (
    "find_entity",
    "neighbors",
    "drugs_targeting",
    "paths",
    "evidence",
    "drugs_mentioned",
)

TOOL_SCOPES: dict[str, tuple[str, ...]] = {
    "literature": (
        "search_corpus",
        "search_pubmed",
        "fetch_pubmed",
        "search_europepmc",
        *GRAPH_TOOLS,
    ),
    "chemistry": (
        "molecular_properties",
        "drug_likeness",
        "similarity",
        "substructure_search",
        "resolve_name",
    ),
    "analysis": ("run_python",),
    # The graph tools on their own (also part of the literature scope above).
    "graph": GRAPH_TOOLS,
}

MCP_SERVERS: dict[str, str] = {
    "literature": "bioagent.mcp_servers.literature",
    "chemistry": "bioagent.mcp_servers.chemistry",
    "sandbox": "bioagent.mcp_servers.sandbox",
    "retrieval": "bioagent.mcp_servers.retrieval",
    "graph": "bioagent.mcp_servers.graph",
}


class ToolProvider(Protocol):
    def tools_for(self, kind: str) -> list[BaseTool]: ...


class ToolScopeError(PermissionError):
    pass


def scoped(tools: Sequence[BaseTool], kind: str) -> dict[str, BaseTool]:
    allowed = set(TOOL_SCOPES.get(kind, ()))
    return {t.name: t for t in tools if t.name in allowed}


class StaticToolProvider:
    """Provider over an in-memory list of tools (tests, or a pre-built hub)."""

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        self._tools = list(tools)

    def tools_for(self, kind: str) -> list[BaseTool]:
        return list(scoped(self._tools, kind).values())


class MCPToolHub:
    """Async context manager owning one stdio session per MCP server."""

    def __init__(
        self,
        servers: Mapping[str, str] | None = None,
        log_dir: Path | None = None,
        server_env: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.servers = dict(servers or MCP_SERVERS)
        self.log_dir = log_dir
        # Extra environment per server (e.g. point the retrieval server at the eval index).
        self.server_env = {k: dict(v) for k, v in (server_env or {}).items()}
        self._stack = AsyncExitStack()
        self._tools: list[BaseTool] = []

    async def __aenter__(self) -> MCPToolHub:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        try:
            for name, module in self.servers.items():
                errlog: Any = sys.stderr
                if self.log_dir is not None:
                    self.log_dir.mkdir(parents=True, exist_ok=True)
                    errlog = self._stack.enter_context(
                        (self.log_dir / f"mcp-{name}.log").open("a", encoding="utf-8")
                    )
                env = None
                if name in self.server_env:
                    from mcp.client.stdio import get_default_environment

                    env = {**get_default_environment(), **self.server_env[name]}
                params = StdioServerParameters(command=sys.executable, args=["-m", module], env=env)
                read, write = await self._stack.enter_async_context(
                    stdio_client(params, errlog=errlog)
                )
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
                self._tools.extend(_mcp_tool(session, t) for t in listed.tools)
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    def tools_for(self, kind: str) -> list[BaseTool]:
        return list(scoped(self._tools, kind).values())

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]


def _mcp_tool(session: Any, tool: Any) -> BaseTool:
    async def call(**kwargs: Any) -> dict[str, Any]:
        args = {k: v for k, v in kwargs.items() if v is not None}
        result = await session.call_tool(tool.name, args)
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            return {"error": text[:1000] or "tool failed"}
        if result.structured_content is not None:
            return dict(result.structured_content)
        text = " ".join(getattr(c, "text", "") for c in result.content)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError:
            return {"result": text}

    return StructuredTool(
        name=tool.name,
        description=(tool.description or tool.name).strip()[:1000],
        args_schema=tool.input_schema,
        coroutine=call,
    )


# --- evidence extraction / rendering --------------------------------------------------
ABSTRACT_CHARS_FOR_MODEL = 900


def extract_evidence(
    tool: str,
    result: Mapping[str, Any],
    task_id: str,
    start: int = 0,
    args: Mapping[str, Any] | None = None,
) -> list[Evidence]:
    """Turn a tool result into evidence items (papers or computed results)."""
    if "error" in result:
        return []
    out: list[Evidence] = []
    if tool == "search_corpus":
        for hit in result.get("hits", []):
            out.append(
                Evidence(
                    key=str(hit["passage_id"]),
                    pmid=str(hit["pmid"]),
                    passage_id=str(hit["passage_id"]),
                    title=hit.get("title"),
                    year=hit.get("year"),
                    journal=hit.get("journal"),
                    doi=hit.get("doi"),
                    text=str(hit.get("text", "")),
                    source="corpus",
                    task_id=task_id,
                    computed=False,
                    untrusted=True,
                )
            )
    elif tool in ("search_pubmed", "fetch_pubmed", "search_europepmc"):
        source = "europepmc" if tool == "search_europepmc" else "pubmed"
        for item in result.get("items", []):
            ident = item.get("pmid") or item.get("pmcid") or item.get("doi")
            if not ident or not item.get("abstract"):
                continue
            title = item.get("title") or ""
            out.append(
                Evidence(
                    key=f"{ident}:abstract",
                    pmid=item.get("pmid"),
                    passage_id=f"{ident}:abstract",
                    title=title,
                    year=item.get("year"),
                    journal=item.get("journal"),
                    doi=item.get("doi"),
                    text=f"{title}\n{item['abstract']}".strip(),
                    source=source,
                    task_id=task_id,
                    computed=False,
                    untrusted=True,
                )
            )
    elif tool in GRAPH_TOOLS:
        out.extend(graph_evidence(tool, result, task_id))
    elif tool == "resolve_name":
        pass  # a PubChem name lookup identifies a compound; it is not evidence for a claim
    elif tool in TOOL_SCOPES["chemistry"] or tool == "run_python":
        source = "sandbox" if tool == "run_python" else "rdkit"
        out.append(
            Evidence(
                key=f"computed:{task_id}:{start + 1}",
                pmid=None,
                passage_id=None,
                title=_computed_title(tool, args),
                year=None,
                journal=None,
                doi=None,
                text=render_computed(tool, result),
                source=source,
                task_id=task_id,
                computed=True,
                untrusted=True,
            )
        )
    return out


def _computed_title(tool: str, args: Mapping[str, Any] | None) -> str:
    """E.g. ``molecular_properties(metformin)`` so references say which input was used."""
    if not args:
        return f"{tool} output"
    if tool == "run_python":
        return "run_python output"
    first = next((str(v) for v in args.values() if v not in (None, "")), "")
    return f"{tool}({first[:60]})" if first else f"{tool} output"


def render_computed(tool: str, result: Mapping[str, Any]) -> str:
    data = {k: v for k, v in result.items() if k not in ("notice", "untrusted_text")}
    if tool == "run_python":
        data = {k: data.get(k) for k in ("exit_code", "stdout", "stderr", "timed_out")}
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text[:4000]


def render_for_model(tool: str, result: Mapping[str, Any], evidence: Sequence[Evidence]) -> str:
    """Compact, size-capped view of a tool result for the specialist's context."""
    if "error" in result:
        return f"Tool error: {result['error']}"
    if tool in GRAPH_TOOLS:
        return render_graph(result, evidence)
    if evidence and not evidence[0]["computed"]:
        lines = [f"{len(evidence)} papers (cite by PMID):"]
        for e in evidence:
            body = e["text"][:ABSTRACT_CHARS_FOR_MODEL]
            lines.append(f"- PMID {e['pmid'] or e['key']} ({e['year']}): {body}")
        return "\n".join(lines)
    if evidence:
        return evidence[0]["text"]
    return json.dumps(dict(result), default=str)[:3000] if result else "No results."


# --- knowledge-graph evidence -----------------------------------------------------------
MAX_GRAPH_EVIDENCE = 20
GRAPH_RESULT_CHARS = 6000
CHEMBL_COMPOUND_URL = "https://www.ebi.ac.uk/chembl/explore/compound/{chembl_id}"


def _claim(a: str | None, rel_type: str, b: str | None, count: Any) -> str:
    if rel_type == "CO_OCCURS":
        return f"{a} and {b} are co-mentioned (graph: {count} papers)"
    verb = rel_type.lower().replace("_", " ")
    return f"{a} {verb} {b} (text-mined relation; graph: {count} papers)"


def _graph_claims(tool: str, result: Mapping[str, Any]) -> tuple[dict[str, list[str]], list[Any]]:
    """(pmid -> claims supported by that paper, ChEMBL mechanism records)."""
    claims: dict[str, list[str]] = {}
    curated: list[Any] = []

    def add(pmids: Sequence[Any], claim: str) -> None:
        for pmid in pmids:
            bucket = claims.setdefault(str(pmid), [])
            if claim not in bucket:
                bucket.append(claim)

    if tool == "neighbors":
        centre = (result.get("entity") or {}).get("name")
        for e in result.get("edges", []):
            other = e["neighbor"].get("name")
            a, b = (centre, other) if e.get("direction") == "out" else (other, centre)
            if e.get("source") == "chembl":
                curated.append({"drug": a, "gene": b, **e})
            else:
                add(e.get("pmids", []), _claim(a, e["type"], b, e.get("evidence_count")))
    elif tool == "drugs_targeting":
        for item in result.get("results", []):
            drug = item["drug"]
            if item.get("source") == "chembl":
                for t in item.get("targets", []):
                    curated.append({"drug": drug.get("name"), "drug_info": drug, **t})
            for r in item.get("relations", []):
                add(
                    r.get("pmids", []),
                    _claim(drug.get("name"), r["type"], r.get("gene"), r.get("evidence_count")),
                )
    elif tool == "paths":
        for path in result.get("paths", []):
            names = {n["id"]: n.get("name") for n in path.get("nodes", [])}
            chain = " - ".join(str(n.get("name")) for n in path.get("nodes", []))
            for e in path.get("edges", []):
                a, b = names.get(e["from"]), names.get(e["to"])
                if e.get("source") == "chembl":
                    curated.append({"drug": a, "gene": b, **e})
                    continue
                claim = _claim(a, e["type"], b, e.get("evidence_count"))
                add(e.get("pmids", []), f"{claim}; part of graph path {chain}")
    elif tool == "evidence":
        edge = result.get("edge") or {}
        a = (result.get("source_entity") or {}).get("name")
        b = (result.get("target_entity") or {}).get("name")
        if edge.get("source") == "chembl":
            curated.append({"drug": a, "gene": b, **edge})
        else:
            add(
                edge.get("pmids", []),
                _claim(a, str(edge.get("type")), b, edge.get("evidence_count")),
            )
    return claims, curated


def graph_evidence(tool: str, result: Mapping[str, Any], task_id: str) -> list[Evidence]:
    """Graph claims become evidence on the PMIDs that support them (plus ChEMBL records)."""
    claims, curated = _graph_claims(tool, result)
    papers: Mapping[str, Any] = result.get("papers") or {}
    out: list[Evidence] = []
    for pmid, items in list(claims.items())[:MAX_GRAPH_EVIDENCE]:
        info = papers.get(pmid) or {}
        title = info.get("title") or ""
        text = (
            f"{title}\nKnowledge-graph relations text-mined by PubTator3 from this paper's "
            "abstract: " + "; ".join(items) + "."
        )
        out.append(
            Evidence(
                key=f"graph:{pmid}:{tool}:{task_id}",
                pmid=pmid,
                passage_id=None,
                title=title or None,
                year=info.get("year"),
                journal=None,
                doi=None,
                text=text.strip(),
                source="graph",
                task_id=task_id,
                computed=False,
                untrusted=True,
            )
        )
    seen: set[str] = set()
    for c in curated:
        drug_info = c.get("drug_info") or {}
        chembl_id = drug_info.get("chembl_id")
        target = c.get("chembl_target") or c.get("chembl_target_name") or c.get("gene")
        key = f"chembl:{chembl_id or c.get('drug')}:{target}:{c.get('action_type')}"
        if key in seen:
            continue
        seen.add(key)
        phase = drug_info.get("max_phase")
        text = (
            f"ChEMBL curated mechanism of action: {c.get('drug')} -> "
            f"{c.get('chembl_target_name') or c.get('gene')} ({c.get('chembl_target') or ''}); "
            f"action {c.get('action_type')}; mechanism: {c.get('mechanism')}; gene {c.get('gene')}"
        )
        if phase is not None:
            status = "approved" if phase >= 4 else "not approved"
            text += f"; ChEMBL max_phase {phase} ({status})"
        out.append(
            Evidence(
                key=key,
                pmid=None,
                passage_id=chembl_id,  # used for the ChEMBL compound URL in citations
                title=f"ChEMBL mechanism: {c.get('drug')} / {c.get('gene')}",
                year=None,
                journal="ChEMBL",
                doi=None,
                text=text,
                source="chembl",
                task_id=task_id,
                computed=False,
                untrusted=True,
            )
        )
        if len(out) >= MAX_GRAPH_EVIDENCE * 2:
            break
    return out


def _strip_graph(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            k: _strip_graph(v)
            for k, v in value.items()
            if k not in ("notice", "untrusted", "papers", "synonyms") and v not in (None, [], {})
        }
    if isinstance(value, list):
        return [_strip_graph(v) for v in value]
    return value


def render_graph(result: Mapping[str, Any], evidence: Sequence[Evidence]) -> str:
    """Compact JSON of a graph result (ids kept for follow-up calls) plus its evidence."""
    body = json.dumps(_strip_graph(dict(result)), ensure_ascii=False, default=str)
    if len(body) > GRAPH_RESULT_CHARS:
        body = body[:GRAPH_RESULT_CHARS] + " ...(truncated)"
    lines = [f"Graph result: {body}"]
    if evidence:
        lines.append(f"{len(evidence)} evidence items (cite graph claims by their PMID):")
        for e in evidence:
            ref = f"PMID {e['pmid']}" if e["pmid"] else f"ChEMBL {e['passage_id'] or ''}".strip()
            lines.append(f"- {ref}: {e['text'][:400]}")
    return "\n".join(lines)
