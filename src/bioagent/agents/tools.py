"""Tool access for the specialists.

``MCPToolHub`` starts our four MCP servers over stdio and exposes their tools as LangChain
``StructuredTool``s (using each server's JSON schema). ``TOOL_SCOPES`` decides which tools
each specialist may see *and* call: a specialist is only bound to its own tools, and calls
to anything else are refused before execution.

``GraphQueryTool`` is the hook for the knowledge-graph tool coming in a later PR.
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

TOOL_SCOPES: dict[str, tuple[str, ...]] = {
    "literature": ("search_corpus", "search_pubmed", "fetch_pubmed", "search_europepmc"),
    "chemistry": (
        "molecular_properties",
        "drug_likeness",
        "similarity",
        "substructure_search",
        "resolve_name",
    ),
    "analysis": ("run_python",),
    # Reserved for the knowledge-graph PR: the literature specialist will gain graph_query.
    "graph": ("graph_query",),
}

MCP_SERVERS: dict[str, str] = {
    "literature": "bioagent.mcp_servers.literature",
    "chemistry": "bioagent.mcp_servers.chemistry",
    "sandbox": "bioagent.mcp_servers.sandbox",
    "retrieval": "bioagent.mcp_servers.retrieval",
}


class ToolProvider(Protocol):
    def tools_for(self, kind: str) -> list[BaseTool]: ...


class GraphQueryTool(Protocol):
    """Interface for the future knowledge-graph tool (entity/relation lookups).

    Implementations will be registered under the name ``graph_query`` and added to the
    ``graph`` scope (and to the literature specialist) once the KG lands.
    """

    async def graph_query(self, query: str, limit: int = 20) -> dict[str, Any]: ...


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
        self, servers: Mapping[str, str] | None = None, log_dir: Path | None = None
    ) -> None:
        self.servers = dict(servers or MCP_SERVERS)
        self.log_dir = log_dir
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
                params = StdioServerParameters(command=sys.executable, args=["-m", module])
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
    elif tool in TOOL_SCOPES["chemistry"] or tool == "run_python":
        source = {"resolve_name": "pubchem", "run_python": "sandbox"}.get(tool, "rdkit")
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
    if evidence and not evidence[0]["computed"]:
        lines = [f"{len(evidence)} papers (cite by PMID):"]
        for e in evidence:
            body = e["text"][:ABSTRACT_CHARS_FOR_MODEL]
            lines.append(f"- PMID {e['pmid'] or e['key']} ({e['year']}): {body}")
        return "\n".join(lines)
    if evidence:
        return evidence[0]["text"]
    return json.dumps(dict(result), default=str)[:3000] if result else "No results."
