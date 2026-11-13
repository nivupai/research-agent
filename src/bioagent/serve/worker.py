"""Worker process: takes jobs from the Redis queue and runs the agent pipeline.

* Progress events from the graph (planner, specialists, synthesizer, verifier) go to the
  job's event stream, followed by a ``final`` event with the answer and claim table.
* Transient failures (timeouts, connection errors, retryable LLM errors) are retried with
  exponential backoff up to ``job_max_attempts``. A retry resumes the LangGraph checkpoint
  of the same run id, so finished graph nodes are not redone.
* Exhausted LLM quota fails the job with a clear message (retrying soon would not help).
* Completed answers are stored in the semantic cache.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from bioagent.config import Settings
from bioagent.llm.providers import LLMError, QuotaExhaustedError
from bioagent.serve.cache import SemanticCache
from bioagent.serve.models import TERMINAL
from bioagent.serve.queue import JobQueue
from bioagent.telemetry import JOB_SECONDS, JOBS, attached_context, span

EventSink = Callable[[dict[str, Any]], Awaitable[None]]
Pipeline = Callable[[Mapping[str, Any], EventSink], Awaitable[dict[str, Any]]]
TRANSIENT = (TimeoutError, ConnectionError, asyncio.TimeoutError)


def slim_result(record: Mapping[str, Any]) -> dict[str, Any]:
    """What a client needs: answer, references, claim table, plan and run stats."""
    verification = record.get("verification", {}) or {}
    claims = [
        {k: c.get(k) for k in ("n", "claim", "label", "check", "citations")}
        for c in verification.get("claims", []) or []
    ]
    return {
        "run_id": record.get("run_id"),
        "answer": record.get("answer", ""),
        "citations": record.get("citations", []),
        "claims": claims,
        "groundedness": verification.get("groundedness"),
        "plan": record.get("plan", []),
        "models": record.get("models", {}),
        "stats": record.get("stats", {}),
    }


def summarise(result: Mapping[str, Any], queue_seconds: float | None) -> dict[str, Any]:
    stats = result.get("stats", {}) or {}
    return {
        "wall_seconds": stats.get("wall_seconds"),
        "queue_seconds": round(queue_seconds, 3) if queue_seconds is not None else None,
        "llm_calls": stats.get("llm_calls"),
        "tool_calls": stats.get("tool_calls"),
        "input_tokens": stats.get("input_tokens"),
        "output_tokens": stats.get("output_tokens"),
        "cost_usd": stats.get("cost_usd"),
        "groundedness": result.get("groundedness"),
    }


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, QuotaExhaustedError):
        return False
    if isinstance(exc, TRANSIENT):
        return True
    return isinstance(exc, LLMError) and "fatal" not in str(exc)


class Worker:
    def __init__(
        self,
        queue: JobQueue,
        pipeline: Pipeline,
        cache: SemanticCache | None = None,
        *,
        max_attempts: int = 3,
        retry_base_s: float = 5.0,
    ) -> None:
        self.queue = queue
        self.pipeline = pipeline
        self.cache = cache
        self.max_attempts = max_attempts
        self.retry_base_s = retry_base_s

    async def run_one(self, job_id: str) -> str | None:
        job = await self.queue.get(job_id)
        if job is None or job["status"] in TERMINAL:
            return None
        attempt = int(job["attempts"]) + 1
        started = time.time()
        job = await self.queue.update(
            job_id, status="running", attempts=attempt, started_at=started
        )
        await self.queue.add_event(job_id, {"type": "running", "attempt": attempt})

        async def on_event(event: dict[str, Any]) -> None:
            await self.queue.add_event(job_id, event)

        with (
            attached_context(job.get("trace_context")),
            span("worker.job", **{"bioagent.job_id": job_id, "bioagent.attempt": attempt}) as s,
        ):
            try:
                record = await self.pipeline(job, on_event)
            except Exception as exc:
                s.record_exception(exc)
                error = f"{type(exc).__name__}: {exc}"[:1000]
                if is_transient(exc) and attempt < self.max_attempts:
                    delay = self.retry_base_s * 2 ** (attempt - 1)
                    await self.queue.schedule_retry(job_id, delay, error)
                    JOBS.labels("retrying").inc()
                    return "retrying"
                await self.queue.update(job_id, status="failed", error=error)
                await self.queue.add_event(job_id, {"type": "failed", "error": error})
                JOBS.labels("failed").inc()
                return "failed"
            result = slim_result(record)
            summary = summarise(result, started - float(job["created_at"]))
            s.set_attribute("bioagent.cost_usd", summary["cost_usd"] or 0.0)
            await self.queue.update(
                job_id, status="completed", result=result, summary=summary, error=None
            )
            await self.queue.add_event(
                job_id, {"type": "final", "result": result, "summary": summary}
            )
            JOBS.labels("completed").inc()
            JOB_SECONDS.labels("miss").observe(time.time() - float(job["created_at"]))
            if self.cache is not None and result.get("answer"):
                await self.cache.store(str(job["question"]), {"result": result, "summary": summary})
            return "completed"

    async def run_forever(self, stop: asyncio.Event, concurrency: int = 1) -> None:
        async def loop() -> None:
            while not stop.is_set():
                job_id = await self.queue.dequeue(timeout_s=1.0)
                if job_id is not None:
                    await self.run_one(job_id)

        await asyncio.gather(*(loop() for _ in range(max(1, concurrency))))


def agent_pipeline(settings: Settings, tools: Any) -> Pipeline:
    """The real LangGraph pipeline; ``tools`` is a shared, already-started MCPToolHub."""
    from bioagent.agents.runner import RunMeta, ask

    @asynccontextmanager
    async def shared(_settings: Settings) -> AsyncIterator[Any]:
        yield tools

    async def run(job: Mapping[str, Any], on_event: EventSink) -> dict[str, Any]:
        run_id = str(job["run_id"])
        resumable = RunMeta(run_id, "", None, {}, "").path(Path(settings.runs_dir)).exists()
        provider = (job.get("options") or {}).get("provider")
        return await ask(
            settings,
            None if resumable else str(job["question"]),
            provider=provider,
            resume=run_id if resumable else None,
            tools_factory=shared,
            on_event=on_event,
            run_id=run_id,
        )

    return run
