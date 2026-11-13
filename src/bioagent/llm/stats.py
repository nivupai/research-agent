"""Per-call LLM and tool statistics. Records are plain dicts so they can live in graph state
(and therefore survive checkpoint/resume)."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, TypedDict

from bioagent.telemetry import record_llm_call, record_tool_call


class LLMCallRecord(TypedDict):
    role: str
    task_id: str | None
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float | None
    retries: int


class ToolCallRecord(TypedDict):
    task_id: str
    tool: str
    ok: bool
    latency_ms: int
    error: str | None


class RunStatsCollector:
    """Accumulates records inside one node; the node returns them into graph state."""

    def __init__(self) -> None:
        self.llm_calls: list[LLMCallRecord] = []
        self.tool_calls: list[ToolCallRecord] = []

    def add_llm(self, record: LLMCallRecord) -> None:
        self.llm_calls.append(record)
        record_llm_call(record)  # span + Prometheus counters

    def add_tool(
        self, task_id: str, tool: str, ok: bool, started: float, error: str | None = None
    ) -> None:
        record = ToolCallRecord(
            task_id=task_id,
            tool=tool,
            ok=ok,
            latency_ms=int((time.monotonic() - started) * 1000),
            error=error[:300] if error else None,
        )
        self.tool_calls.append(record)
        record_tool_call(record)

    def update(self) -> dict[str, Any]:
        return {"llm_calls": self.llm_calls, "tool_calls": self.tool_calls}


def summarize(
    llm_calls: Iterable[LLMCallRecord], tool_calls: Iterable[ToolCallRecord]
) -> dict[str, Any]:
    llm = list(llm_calls)
    tools = list(tool_calls)
    costs = [c["cost_usd"] for c in llm]
    known = [c for c in costs if c is not None]
    by_model: dict[str, dict[str, int]] = {}
    for c in llm:
        m = by_model.setdefault(
            f"{c['provider']}:{c['model']}", {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        )
        m["calls"] += 1
        m["input_tokens"] += c["input_tokens"]
        m["output_tokens"] += c["output_tokens"]
    return {
        "llm_calls": len(llm),
        "input_tokens": sum(c["input_tokens"] for c in llm),
        "output_tokens": sum(c["output_tokens"] for c in llm),
        "llm_latency_ms": sum(c["latency_ms"] for c in llm),
        "llm_retries": sum(c["retries"] for c in llm),
        "cost_usd": round(sum(known), 6) if known else (0.0 if not costs else None),
        "cost_complete": len(known) == len(costs),
        "by_model": by_model,
        "tool_calls": len(tools),
        "tool_errors": sum(1 for t in tools if not t["ok"]),
        "tools_used": sorted({t["tool"] for t in tools}),
    }
