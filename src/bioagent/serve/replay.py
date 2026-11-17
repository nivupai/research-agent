"""A stand-in pipeline for cluster smoke tests and demos (``WORKER_PIPELINE=replay``).

No LLM calls, no MCP servers, no models, so it runs in the slim image and never spends API
quota. If ``runs_dir`` holds a saved run whose question matches the job's, that run is
replayed: the planner's plan, each recorded tool call, the synthesizer and the verifier are
emitted as the same progress events the agent sends, paced over ``seconds``, and the saved
record is returned (marked ``replayed_from``). Otherwise a fixed, clearly labelled answer is
returned after ``seconds``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bioagent.serve.worker import EventSink, Pipeline

NODES = ("planner", "literature", "synthesizer", "verifier")


def fixed_answer(job: Mapping[str, Any], seconds: float) -> dict[str, Any]:
    return {
        "run_id": job.get("run_id"),
        "answer": "[replay pipeline] No model was called; this is a fixed test answer.",
        "citations": [],
        "verification": {"claims": [], "groundedness": None},
        "plan": [{"id": "t1", "kind": "literature", "description": "replay"}],
        "models": {"pipeline": "replay"},
        "stats": {
            "wall_seconds": seconds,
            "llm_calls": 0,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        },
    }


def replay_events(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The progress events a saved run would have produced, in order."""
    plan = record.get("plan", []) or []
    events: list[dict[str, Any]] = [
        {
            "type": "progress",
            "node": "planner",
            "replay": True,
            "plan": [
                {
                    "id": t.get("id"),
                    "kind": t.get("kind"),
                    "description": str(t.get("description", ""))[:200],
                }
                for t in plan
            ],
        }
    ]
    kinds = {t.get("id"): t.get("kind") for t in plan}
    for call in record.get("tool_calls", []) or []:
        events.append(
            {
                "type": "progress",
                "node": kinds.get(call.get("task_id")) or "specialist",
                "replay": True,
                "task_id": call.get("task_id"),
                "tool": call.get("tool"),
                "ok": call.get("ok"),
                "latency_ms": call.get("latency_ms"),
            }
        )
    events.append(
        {
            "type": "progress",
            "node": "synthesizer",
            "replay": True,
            "citations": len(record.get("citations", []) or []),
        }
    )
    verification = record.get("verification", {}) or {}
    if verification.get("claims"):
        events.append(
            {
                "type": "progress",
                "node": "verifier",
                "replay": True,
                "groundedness": verification.get("groundedness"),
                "claims": len(verification.get("claims", [])),
                "removed_claims": len(verification.get("removed_claims", []) or []),
            }
        )
    return events


def replay_pipeline(seconds: float, runs_dir: Path | None = None) -> Pipeline:
    async def run(job: Mapping[str, Any], on_event: EventSink) -> dict[str, Any]:
        record = None
        if runs_dir is not None:
            from bioagent.serve.runs import find_by_question

            record = find_by_question(runs_dir, str(job.get("question", "")))
        if record is None:
            for node in NODES:
                await on_event({"type": "progress", "node": node, "replay": True})
                await asyncio.sleep(seconds / len(NODES))
            return fixed_answer(job, seconds)
        events = replay_events(record)
        for event in events:
            await on_event(event)
            await asyncio.sleep(seconds / len(events))
        return {**record, "replayed_from": record.get("run_id")}

    return run
