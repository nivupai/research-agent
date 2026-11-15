"""A stand-in pipeline for cluster smoke tests and load/scaling demos (``WORKER_PIPELINE=replay``).

It emits the same progress events as the agent and returns a fixed, clearly labelled answer
after ``replay_seconds``: no LLM calls, no MCP servers, no models, so it runs in the slim image
and never spends API quota.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from bioagent.serve.worker import EventSink, Pipeline

NODES = ("planner", "literature", "synthesizer", "verifier")


def replay_pipeline(seconds: float) -> Pipeline:
    async def run(job: Mapping[str, Any], on_event: EventSink) -> dict[str, Any]:
        for node in NODES:
            await on_event({"type": "progress", "node": node, "replay": True})
            await asyncio.sleep(seconds / len(NODES))
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

    return run
