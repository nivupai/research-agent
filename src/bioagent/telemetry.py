"""Tracing (OpenTelemetry) and metrics (Prometheus) for the agent and the serving stack.

The OpenTelemetry API calls here are no-ops until ``setup_tracing`` installs an SDK tracer
provider (the API and worker do; the CLI and tests do not), so instrumented code costs
nothing when tracing is off.

* Spans: ``api.ask`` -> ``queue.enqueue`` -> ``worker.job`` -> ``graph.node.<name>`` ->
  ``llm.call`` / ``mcp.tool``. LLM and tool spans are recorded from the measured latency when
  the call finishes (see ``record_llm_call`` / ``record_tool_call``), with model, role,
  tokens, cost (list price) and errors as attributes. Trace context crosses the Redis queue
  in the job payload (W3C ``traceparent``).
* Metrics: request, job, LLM, tool and cache counters/histograms on the default Prometheus
  registry (``/metrics`` on the API, a separate port on the worker).
"""

from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from opentelemetry.trace import Span, Status, StatusCode
from prometheus_client import Counter, Histogram

TRACER_NAME = "bioagent"
P = ParamSpec("P")
R = TypeVar("R")

# --- metrics ----------------------------------------------------------------------------------
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160, 320)
HTTP_REQUESTS = Counter("bioagent_http_requests_total", "API requests", ["route", "status"])
JOBS = Counter("bioagent_jobs_total", "Jobs by final status", ["status"])
JOB_SECONDS = Histogram(
    "bioagent_job_seconds", "Job wall time (queue to done)", ["cache"], buckets=LATENCY_BUCKETS
)
NODE_SECONDS = Histogram(
    "bioagent_graph_node_seconds", "Graph node latency", ["node"], buckets=LATENCY_BUCKETS
)
LLM_CALLS = Counter("bioagent_llm_calls_total", "LLM calls", ["model", "role"])
LLM_TOKENS = Counter("bioagent_llm_tokens_total", "LLM tokens", ["model", "direction"])
LLM_COST = Counter("bioagent_llm_cost_usd_total", "LLM cost at list price (USD)", ["model"])
LLM_SECONDS = Histogram(
    "bioagent_llm_call_seconds", "LLM call latency", ["model"], buckets=LATENCY_BUCKETS
)
TOOL_CALLS = Counter("bioagent_tool_calls_total", "MCP tool calls", ["tool", "ok"])
CACHE = Counter("bioagent_semantic_cache_total", "Semantic cache lookups", ["result"])
RATE_LIMITED = Counter("bioagent_rate_limited_total", "Requests refused by the rate limiter")


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def setup_tracing(service_name: str, otlp_endpoint: str | None) -> bool:
    """Install the SDK tracer provider with an OTLP/HTTP exporter (e.g. Jaeger on :4318).
    Returns False (tracing stays a no-op) when no endpoint is configured."""
    if not otlp_endpoint:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    endpoint = otlp_endpoint.rstrip("/")
    if not endpoint.endswith("/v1/traces"):
        endpoint += "/v1/traces"
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return True


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    with tracer().start_as_current_span(name, attributes=_clean(attributes)) as s:
        yield s


def _clean(attributes: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attributes.items() if isinstance(v, str | bool | int | float)}


# --- context propagation across the queue ---------------------------------------------------
def inject_context() -> dict[str, str]:
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


@contextmanager
def attached_context(carrier: Mapping[str, str] | None) -> Iterator[None]:
    token = otel_context.attach(propagate.extract(dict(carrier or {})))
    try:
        yield
    finally:
        otel_context.detach(token)


# --- records from the stats collector -------------------------------------------------------
def _span_from_latency(
    name: str, latency_ms: int, attributes: Mapping[str, Any], error: str | None = None
) -> None:
    end = time.time_ns()
    s = tracer().start_span(
        name, start_time=end - latency_ms * 1_000_000, attributes=_clean(attributes)
    )
    if error:
        s.set_status(Status(StatusCode.ERROR, error[:200]))
    s.end(end_time=end)


def record_llm_call(record: Mapping[str, Any]) -> None:
    model = f"{record['provider']}:{record['model']}"
    LLM_CALLS.labels(model, record["role"]).inc()
    LLM_TOKENS.labels(model, "input").inc(record["input_tokens"])
    LLM_TOKENS.labels(model, "output").inc(record["output_tokens"])
    LLM_SECONDS.labels(model).observe(record["latency_ms"] / 1000)
    if record.get("cost_usd") is not None:
        LLM_COST.labels(model).inc(record["cost_usd"])
    _span_from_latency(
        "llm.call",
        record["latency_ms"],
        {
            "gen_ai.system": record["provider"],
            "gen_ai.request.model": record["model"],
            "gen_ai.usage.input_tokens": record["input_tokens"],
            "gen_ai.usage.output_tokens": record["output_tokens"],
            "bioagent.role": record["role"],
            "bioagent.task_id": record.get("task_id") or "",
            "bioagent.cost_usd": record.get("cost_usd") or 0.0,
            "bioagent.retries": record.get("retries", 0),
            "bioagent.latency_ms": record["latency_ms"],
        },
    )


def record_tool_call(record: Mapping[str, Any]) -> None:
    TOOL_CALLS.labels(record["tool"], str(record["ok"]).lower()).inc()
    _span_from_latency(
        "mcp.tool",
        record["latency_ms"],
        {
            "bioagent.tool": record["tool"],
            "bioagent.task_id": record["task_id"],
            "bioagent.ok": record["ok"],
            "bioagent.latency_ms": record["latency_ms"],
        },
        error=record.get("error"),
    )


# --- graph nodes ------------------------------------------------------------------------------
def traced_node(name: str, fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Wrap a LangGraph node in a ``graph.node.<name>`` span and a latency histogram."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = time.monotonic()
        with span(f"graph.node.{name}", **{"bioagent.node": name}) as s:
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                s.record_exception(exc)
                s.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
                raise
            finally:
                NODE_SECONDS.labels(name).observe(time.monotonic() - started)

    return wrapper
