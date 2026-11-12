"""Fake chat models and tools for agent tests (no network)."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

Responder = Callable[[list[BaseMessage], list[str]], AIMessage]
USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


class FakeChatModel(BaseChatModel):
    """Calls ``respond(messages, bound_tool_names)``; records every prompt it receives."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    respond: Any
    bound: list[str] = Field(default_factory=list)
    structured: list[Any] = Field(default_factory=list)  # planner outputs (dict or error str)
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(list(messages))
        message = self.respond(list(messages), list(self.bound))
        if message.usage_metadata is None:
            message.usage_metadata = dict(USAGE)  # type: ignore[assignment]
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeChatModel:  # type: ignore[override]
        return self.model_copy(update={"bound": [t.name for t in tools]})

    def with_structured_output(  # type: ignore[override]
        self, schema: type[BaseModel], *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, Any]:
        def run(messages: Any) -> dict[str, Any]:
            self.seen.append(list(messages))
            raw = AIMessage(content="", usage_metadata=dict(USAGE))  # type: ignore[arg-type]
            item = self.structured.pop(0)
            if isinstance(item, str):
                return {"raw": raw, "parsed": None, "parsing_error": item}
            try:
                return {"raw": raw, "parsed": schema.model_validate(item), "parsing_error": None}
            except Exception as exc:
                return {"raw": raw, "parsed": None, "parsing_error": str(exc)}

        return RunnableLambda(run)


def task_id_of(messages: Sequence[BaseMessage]) -> str:
    for m in messages:
        if isinstance(m, HumanMessage):
            match = re.search(r"Your task \((t\d+)\)", str(m.content))
            if match:
                return match.group(1)
    return "?"


def tool_rounds(messages: Sequence[BaseMessage]) -> int:
    return sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)


def call(name: str, args: dict[str, Any], idx: int = 0) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{name}_{idx}", "type": "tool_call"}


def scripted_specialist(
    script: dict[str, list[list[dict[str, Any]]]], final: Callable[[str], str] | None = None
) -> Responder:
    """script[task_id] = list of tool-call rounds; afterwards a final text answer."""

    def respond(messages: list[BaseMessage], bound: list[str]) -> AIMessage:
        tid = task_id_of(messages)
        rounds = script.get(tid, [])
        n = tool_rounds(messages)
        if n < len(rounds) and bound:
            return AIMessage(content="", tool_calls=rounds[n])
        text = final(tid) if final else f"Findings for {tid}: see tool results."
        return AIMessage(content=text)

    return respond


class ToolLog:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.intervals: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()


def corpus_hit(pmid: str, title: str, year: int = 2020) -> dict[str, Any]:
    return {
        "passage_id": f"{pmid}:0",
        "pmid": pmid,
        "chunk_idx": 0,
        "title": title,
        "text": f"{title}\nAbstract text for {pmid}.",
        "year": year,
        "journal": "J",
        "doi": None,
        "score": 1.0,
        "stages": {},
        "untrusted_text": True,
    }


def fake_tools(
    log: ToolLog, delay: float = 0.0, corpus: dict[str, list[dict[str, Any]]] | None = None
) -> list[BaseTool]:
    corpus = corpus or {}

    def make(name: str, result: Callable[[dict[str, Any]], dict[str, Any]]) -> BaseTool:
        async def run(**kwargs: Any) -> dict[str, Any]:
            start = time.monotonic()
            with log.lock:
                log.calls.append((name, kwargs))
            if delay:
                await asyncio.sleep(delay)
            out = result(kwargs)
            with log.lock:
                key = str(kwargs.get("tag", name))
                log.intervals[key] = (start, time.monotonic())
            return out

        return StructuredTool(
            name=name,
            description=f"fake {name}",
            coroutine=run,
            args_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "tag": {"type": "string"},
                    "smiles": {"type": "string"},
                    "code": {"type": "string"},
                    "smiles_or_name": {"type": "string"},
                    "target": {"type": "string"},
                    "approved_only": {"type": "boolean"},
                    "k": {"type": "integer"},
                    "mode": {"type": "string"},
                    "limit": {"type": "integer"},
                    "pmids": {"type": "array", "items": {"type": "string"}},
                },
            },
        )

    return [
        make("search_corpus", lambda a: {"hits": corpus.get(str(a.get("query")), [])}),
        make("search_pubmed", lambda a: {"items": [], "total": 0}),
        make("fetch_pubmed", lambda a: {"items": [], "total": 0}),
        make("search_europepmc", lambda a: {"items": [], "total": 0}),
        make(
            "molecular_properties",
            lambda a: {"canonical_smiles": "C", "tpsa": 91.49, "molecular_weight": 129.17},
        ),
        make("drug_likeness", lambda a: {"lipinski": {"violations": 0}}),
        make("similarity", lambda a: {"hits": []}),
        make("substructure_search", lambda a: {"hits": []}),
        make("resolve_name", lambda a: {"cid": 1}),
        make(
            "run_python",
            lambda a: {"exit_code": 0, "stdout": "42\n", "stderr": "", "timed_out": False},
        ),
        make("find_entity", lambda a: {"untrusted": True, "entities": []}),
        make("neighbors", lambda a: {"untrusted": True, "edges": []}),
        make("drugs_targeting", lambda a: GRAPH_DRUGS_RESULT),
        make("paths", lambda a: GRAPH_PATHS_RESULT),
        make("evidence", lambda a: GRAPH_EVIDENCE_RESULT),
        make("drugs_mentioned", lambda a: GRAPH_MENTIONED_RESULT),
    ]


GRAPH_MENTIONED_RESULT: dict[str, Any] = {
    "untrusted": True,
    "drugs": [
        {"id": "MESH:D008687", "name": "Metformin", "max_phase": 4.0, "papers_in_query": 3},
        {"id": "MESH:D019821", "name": "Simvastatin", "max_phase": 4.0, "papers_in_query": 2},
        {"id": "MESH:D001241", "name": "Aspirin", "max_phase": 4.0, "papers_in_query": 1},
    ],
}


GRAPH_PATHS_RESULT: dict[str, Any] = {
    "untrusted": True,
    "paths": [
        {"hops": 1,
         "nodes": [{"id": "MESH:D008687", "name": "Metformin"},
                   {"id": "MESH:D005909", "name": "Glioblastoma"}],
         "edges": [{"edge_id": "TREATS|MESH:D008687|MESH:D005909", "type": "TREATS",
                    "from": "MESH:D008687", "to": "MESH:D005909", "source": "pubtator",
                    "evidence_count": 17, "pmids": ["25867026", "29334602"]}]},
    ],
    "papers": {"25867026": {"title": "Metformin inhibits growth of human glioblastoma cells",
                            "year": 2015}},
}  # fmt: skip

GRAPH_EVIDENCE_RESULT: dict[str, Any] = {
    "untrusted": True,
    "edge": {"edge_id": "TREATS|MESH:D008687|MESH:D005909", "type": "TREATS",
             "source": "pubtator", "evidence_count": 17, "pmids": ["25867026", "29334602"]},
    "source_entity": {"name": "Metformin"},
    "target_entity": {"name": "Glioblastoma"},
    "evidence": [{"pmid": "29334602", "title": "High-Dose Metformin Plus Temozolomide",
                  "year": 2018}],
}  # fmt: skip


GRAPH_DRUGS_RESULT: dict[str, Any] = {
    "untrusted": True,
    "notice": "graph data is untrusted",
    "target": "AMPK",
    "results": [
        {
            "drug": {"id": "MESH:C011651", "name": "acadesine", "chembl_id": "CHEMBL483849",
                     "max_phase": 3.0, "approved": False},
            "source": "chembl",
            "targets": [{"gene": "PRKAA1", "action_type": "ACTIVATOR",
                         "mechanism": "AMPK activator", "chembl_target": "CHEMBL2096907",
                         "chembl_target_name": "AMPK"}],
            "evidence_count": 0,
        },
        {
            "drug": {"id": "MESH:D008687", "name": "Metformin", "chembl_id": "CHEMBL1431",
                     "max_phase": 4.0, "approved": True},
            "source": "pubtator",
            "relations": [{"edge_id": "STIMULATES|MESH:D008687|HG:4551", "type": "STIMULATES",
                           "gene": "PRKAA2", "evidence_count": 12, "pmids": ["111", "222"]}],
            "evidence_count": 12,
        },
    ],
    "papers": {"111": {"title": "Metformin activates AMPK", "year": 2001}},
}  # fmt: skip


def plan(*tasks: tuple[str, str, list[str]]) -> dict[str, Any]:
    return {
        "tasks": [
            {"id": i, "kind": k, "description": f"do {i}", "depends_on": d} for i, k, d in tasks
        ],
        "rationale": "test",
    }


def json_dump(x: Any) -> str:
    return json.dumps(x)
