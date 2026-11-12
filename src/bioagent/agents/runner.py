"""Run (or resume) the agent graph with a SQLite checkpointer and save a JSON run record."""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from bioagent.agents.graph import build_graph
from bioagent.agents.nodes import AgentDeps, RoleModel
from bioagent.agents.state import AgentState
from bioagent.agents.tools import MCPToolHub, ToolProvider
from bioagent.config import Settings
from bioagent.llm.providers import RoleModels, make_chat_model, role_specs
from bioagent.llm.stats import summarize

RECURSION_LIMIT = 60


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


@dataclass
class RunMeta:
    run_id: str
    question: str
    provider: str | None
    models: dict[str, str]
    started_at: str
    wall_seconds: float = 0.0
    attempts: int = 0
    status: str = "running"

    def path(self, runs_dir: Path) -> Path:
        return runs_dir / f"{self.run_id}.meta.json"

    def save(self, runs_dir: Path) -> None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.path(runs_dir).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, runs_dir: Path, run_id: str) -> RunMeta:
        path = runs_dir / f"{run_id}.meta.json"
        if not path.exists():
            raise FileNotFoundError(f"no run {run_id!r} in {runs_dir}")
        return cls(**json.loads(path.read_text()))


def build_deps(settings: Settings, specs: RoleModels, tools: ToolProvider) -> AgentDeps:
    return AgentDeps(
        planner=RoleModel(specs.planner, make_chat_model(specs.planner, settings)),
        specialist=RoleModel(specs.specialist, make_chat_model(specs.specialist, settings)),
        synth=RoleModel(specs.synth, make_chat_model(specs.synth, settings)),
        tools=tools,
        specialists={
            kind: RoleModel(spec, make_chat_model(spec, settings))
            for kind, spec in specs.specialists.items()
        },
        max_retries=settings.llm_max_retries,
        # Small local models lose track of the question with long prompts.
        synth_max_evidence=16 if specs.synth.provider == "ollama" else 40,
        synth_evidence_chars=700 if specs.synth.provider == "ollama" else 1100,
    )


async def execute(
    deps: AgentDeps,
    checkpoint_db: Path,
    run_id: str,
    question: str | None,
) -> AgentState:
    """Run a new thread (question given) or resume an existing one (question None)."""
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db)) as saver:
        graph = build_graph(deps).compile(checkpointer=saver)
        config: dict[str, Any] = {
            "configurable": {"thread_id": run_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        if question is None:
            snapshot = await graph.aget_state(config)
            if not snapshot.values:
                raise ValueError(f"no checkpoint for run {run_id!r}")
            if not snapshot.next:
                return dict(snapshot.values)  # type: ignore[return-value]
            result = await graph.ainvoke(None, config)
        else:
            result = await graph.ainvoke({"run_id": run_id, "question": question}, config)
        return result  # type: ignore[no-any-return]


def run_record(state: AgentState, meta: RunMeta) -> dict[str, Any]:
    evidence = state.get("evidence", {})
    papers = {e["pmid"] for e in evidence.values() if e["pmid"]}
    passages = [e for e in evidence.values() if not e["computed"]]
    citations = state.get("citations", [])
    stats = summarize(state.get("llm_calls", []), state.get("tool_calls", []))
    stats.update(
        {
            "passages": len(passages),
            "papers": len(papers),
            "computed_results": sum(1 for e in evidence.values() if e["computed"]),
            "cited": len(citations),
            "cited_papers": len({c["pmid"] for c in citations if c["pmid"]}),
            "wall_seconds": round(meta.wall_seconds, 1),
            "attempts": meta.attempts,
        }
    )
    cited_keys = {c["key"] for c in citations}
    return {
        "run_id": meta.run_id,
        "status": meta.status,
        "question": meta.question,
        "provider": meta.provider,
        "models": meta.models,
        "started_at": meta.started_at,
        "plan": state.get("plan", []),
        "plan_rationale": state.get("plan_rationale", ""),
        "plan_notes": state.get("plan_notes", []),
        "task_results": state.get("results", {}),
        "answer": state.get("answer", ""),
        "citations": citations,
        "cited_evidence": {k: v for k, v in evidence.items() if k in cited_keys},
        "evidence_keys": sorted(evidence),
        "verification": state.get("verification", {}),
        "errors": state.get("errors", []),
        "stats": stats,
        "llm_calls": state.get("llm_calls", []),
        "tool_calls": state.get("tool_calls", []),
    }


ToolsFactory = Callable[[Settings], AbstractAsyncContextManager[ToolProvider]]


def default_tools(settings: Settings) -> AbstractAsyncContextManager[ToolProvider]:
    return MCPToolHub(log_dir=Path(settings.mcp_log_dir))


async def ask(
    settings: Settings,
    question: str | None = None,
    *,
    provider: str | None = None,
    resume: str | None = None,
    tools_factory: ToolsFactory = default_tools,
    deps_factory: Callable[[Settings, RoleModels, ToolProvider], AgentDeps] = build_deps,
) -> dict[str, Any]:
    runs_dir = Path(settings.runs_dir)
    if resume:
        meta = RunMeta.load(runs_dir, resume)
        provider = meta.provider
    elif question:
        meta = RunMeta(
            run_id=new_run_id(),
            question=question,
            provider=provider,
            models={},
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    else:
        raise ValueError("give a question or --resume RUN_ID")
    specs = role_specs(settings, provider)
    meta.models = specs.describe()
    meta.attempts += 1
    meta.status = "running"
    meta.save(runs_dir)

    started = time.monotonic()
    try:
        async with tools_factory(settings) as tools:
            deps = deps_factory(settings, specs, tools)
            state = await execute(
                deps, Path(settings.checkpoint_db), meta.run_id, None if resume else meta.question
            )
    except BaseException as exc:
        meta.wall_seconds += time.monotonic() - started
        meta.status = "failed"
        meta.save(runs_dir)
        exc.add_note(f"run_id={meta.run_id}")
        raise
    meta.wall_seconds += time.monotonic() - started
    meta.status = "completed"
    meta.save(runs_dir)
    record = run_record(state, meta)
    (runs_dir / f"{meta.run_id}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str)
    )
    return record
