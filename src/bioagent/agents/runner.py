"""Run (or resume) the agent graph with a SQLite checkpointer and save a JSON run record."""

from __future__ import annotations

import importlib.util
import json
import secrets
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from bioagent.agents.graph import build_graph
from bioagent.agents.nodes import AgentDeps, RoleModel
from bioagent.agents.state import AgentState
from bioagent.agents.tools import TOOL_SCOPES, MCPToolHub, ToolProvider
from bioagent.agents.untrusted import new_run_mark, set_run_mark
from bioagent.config import Settings
from bioagent.llm.providers import RoleModels, make_chat_model, role_specs
from bioagent.llm.stats import summarize
from bioagent.security.audit import AuditLog, audit, sha256, use_audit
from bioagent.security.policy import ToolPolicy
from bioagent.verify.backends import make_scorer, windowed
from bioagent.verify.verifier import ClaimVerifier, VerifyConfig

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


class VerifierUnavailableError(RuntimeError):
    pass


def build_claim_verifier(settings: Settings) -> ClaimVerifier | None:
    if not settings.verify_claims:
        return None
    if importlib.util.find_spec("transformers") is None:
        raise VerifierUnavailableError(
            "claim verification needs the `models` extra (uv sync --extra models); "
            "or run with --no-verify"
        )
    return ClaimVerifier(
        make_scorer(settings),
        VerifyConfig(
            entail_threshold=settings.nli_entail_threshold,
            contradiction_threshold=settings.nli_contradiction_threshold,
            policy=settings.verify_policy,
            windowed=windowed(settings.verifier_backend),
        ),
    )


def build_deps(settings: Settings, specs: RoleModels, tools: ToolProvider) -> AgentDeps:
    return AgentDeps(
        claim_verifier=build_claim_verifier(settings),
        policy=ToolPolicy(TOOL_SCOPES) if settings.security_defenses else None,
        defenses=settings.security_defenses,
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


EventSink = Callable[[dict[str, Any]], Awaitable[None]]


def progress_event(node: str, update: Any) -> dict[str, Any]:
    """A compact, JSON-friendly summary of one graph node's state update."""
    event: dict[str, Any] = {"type": "progress", "node": node}
    if not isinstance(update, dict):
        return event
    if node == "planner":
        event["plan"] = [
            {"id": t["id"], "kind": t["kind"], "description": t["description"][:200]}
            for t in update.get("plan", [])
        ]
    elif "results" in update:
        for task_id, r in update["results"].items():
            event |= {
                "task_id": task_id,
                "status": r["status"],
                "steps": r["steps"],
                "evidence": len(r["evidence_keys"]),
            }
    elif node == "synthesizer":
        event["citations"] = len(update.get("citations", []))
    elif node == "verifier":
        v = update.get("verification", {})
        event |= {
            "groundedness": v.get("groundedness"),
            "claims": len(v.get("claims", [])),
            "removed_claims": len(v.get("removed_claims", [])),
        }
    calls = update.get("llm_calls", [])
    if calls:
        event["llm_calls"] = len(calls)
        event["tokens"] = sum(c["input_tokens"] + c["output_tokens"] for c in calls)
    return event


async def execute(
    deps: AgentDeps,
    checkpoint_db: Path,
    run_id: str,
    question: str | None,
    on_event: EventSink | None = None,
) -> AgentState:
    """Run a new thread (question given) or resume an existing one (question None).
    With ``on_event`` the graph is streamed and each node's update is reported."""
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db)) as saver:
        graph = build_graph(deps).compile(checkpointer=saver)
        config: dict[str, Any] = {
            "configurable": {"thread_id": run_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        inputs: dict[str, Any] | None = {"run_id": run_id, "question": question}
        if question is None:
            snapshot = await graph.aget_state(config)
            if not snapshot.values:
                raise ValueError(f"no checkpoint for run {run_id!r}")
            if not snapshot.next:
                return dict(snapshot.values)  # type: ignore[return-value]
            inputs = None
        if on_event is None:
            result = await graph.ainvoke(inputs, config)
            return result  # type: ignore[no-any-return]
        async for chunk in graph.astream(inputs, config, stream_mode="updates"):
            for node, update in chunk.items():
                await on_event(progress_event(node, update))
        final = await graph.aget_state(config)
        return dict(final.values)  # type: ignore[return-value]


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
    verification = state.get("verification", {})
    if verification.get("claim_verification") == "on":
        stats.update(
            {
                "claims": len(verification.get("claims", [])),
                "groundedness": verification.get("groundedness"),
                "removed_claims": len(verification.get("removed_claims", [])),
                "flagged_sentences": len(verification.get("flagged_sentences", [])),
                "verify_seconds": verification.get("seconds"),
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
        "draft_answer": state.get("draft_answer", state.get("answer", "")),
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
    on_event: EventSink | None = None,
    run_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    runs_dir = Path(settings.runs_dir)
    if resume:
        meta = RunMeta.load(runs_dir, resume)
        provider = meta.provider
    elif question:
        meta = RunMeta(
            run_id=run_id or new_run_id(),
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
    log = AuditLog(Path(settings.audit_dir), meta.run_id, job_id, settings.audit_sqlite)
    set_run_mark(new_run_mark() if settings.security_defenses else None)
    try:
        with use_audit(log):
            audit(
                "run_start" if not resume else "run_resume",
                question_sha256=sha256(meta.question),
                question_chars=len(meta.question),
                models=meta.models,
                defenses=settings.security_defenses,
                attempt=meta.attempts,
            )
            async with tools_factory(settings) as tools:
                deps = deps_factory(settings, specs, tools)
                state = await execute(
                    deps,
                    Path(settings.checkpoint_db),
                    meta.run_id,
                    None if resume else meta.question,
                    on_event,
                )
    except BaseException as exc:
        meta.wall_seconds += time.monotonic() - started
        meta.status = "failed"
        meta.save(runs_dir)
        log.append("run_failed", {"error_type": type(exc).__name__})
        exc.add_note(f"run_id={meta.run_id}")
        raise
    finally:
        set_run_mark(None)
    meta.wall_seconds += time.monotonic() - started
    meta.status = "completed"
    meta.save(runs_dir)
    record = run_record(state, meta)
    stats = record["stats"]
    log.append(
        "run_end",
        {
            "answer_sha256": sha256(record["answer"]),
            "cited_pmids": sorted({str(c["pmid"]) for c in record["citations"] if c["pmid"]}),
            "llm_calls": stats.get("llm_calls"),
            "tool_calls": stats.get("tool_calls"),
            "input_tokens": stats.get("input_tokens"),
            "output_tokens": stats.get("output_tokens"),
            "cost_usd": stats.get("cost_usd"),
            "wall_seconds": stats.get("wall_seconds"),
        },
    )
    record["audit_log"] = str(log.path)
    (runs_dir / f"{meta.run_id}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str)
    )
    return record
