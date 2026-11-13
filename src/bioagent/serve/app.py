"""FastAPI app: enqueue questions, poll jobs, stream progress (SSE), health and metrics.

POST /v1/ask            -> 202 + job id (or an immediately completed job on a cache hit)
GET  /v1/jobs/{id}      -> status, result, per-query cost/latency summary
GET  /v1/jobs/{id}/events -> text/event-stream: queued, running, progress, retrying, final
GET  /healthz /readyz /metrics
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis

from bioagent.config import Settings
from bioagent.serve.cache import SemanticCache
from bioagent.serve.models import AskRequest, AskResponse, JobSummary, JobView
from bioagent.serve.queue import JobQueue
from bioagent.serve.ratelimit import TokenBucket
from bioagent.telemetry import (
    CACHE,
    HTTP_REQUESTS,
    JOB_SECONDS,
    JOBS,
    RATE_LIMITED,
    inject_context,
    span,
)


def create_app(
    settings: Settings,
    redis: Redis | None = None,
    cache: SemanticCache | None = None,
) -> FastAPI:
    """``redis``/``cache`` are injected in tests; otherwise built from the settings."""
    state: dict[str, Any] = {"redis": redis, "cache": cache}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if state["redis"] is None:
            state["redis"] = Redis.from_url(settings.redis_url)
        if state["cache"] is None and settings.semantic_cache_enabled:
            state["cache"] = build_cache(settings, state["redis"])
        yield
        if redis is None:
            await state["redis"].aclose()

    app = FastAPI(title="BioResearch Agent", version="0.1.0", lifespan=lifespan)

    def queue() -> JobQueue:
        return JobQueue(state["redis"], settings.job_ttl_s)

    async def authorise(x_api_key: str | None = Header(default=None)) -> str | None:
        expected = settings.api_key.get_secret_value() if settings.api_key else None
        if expected is not None and x_api_key != expected:
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
        return x_api_key

    def client_id(request: Request, api_key: str | None) -> str:
        if api_key:
            return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
        return "ip:" + (request.client.host if request.client else "unknown")

    @app.post("/v1/ask", status_code=202, response_model=AskResponse)
    async def ask(
        body: AskRequest, request: Request, api_key: str | None = Depends(authorise)
    ) -> AskResponse | JSONResponse:
        client = client_id(request, api_key)
        limiter = TokenBucket(
            state["redis"], settings.rate_limit_per_minute, settings.rate_limit_burst
        )
        allowed, wait = await limiter.allow(client)
        if not allowed:
            RATE_LIMITED.inc()
            HTTP_REQUESTS.labels("/v1/ask", "429").inc()
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(int(wait) + 1)},
                content={"detail": "rate limit exceeded", "retry_after_s": round(wait, 1)},
            )
        with span("api.ask", **{"bioagent.client": client}) as s:
            started = time.time()
            q = queue()
            options = {"provider": body.provider, "no_cache": body.no_cache}
            hit, similarity = None, None
            cache_ = state["cache"]
            if cache_ is not None and not body.no_cache:
                with span("cache.lookup"):
                    hit, similarity = await cache_.lookup(body.question)
                CACHE.labels("hit" if hit else "miss").inc()
            s.set_attribute("bioagent.cache_hit", hit is not None)
            if hit is not None:
                summary = {
                    **hit.get("summary", {}),
                    "queue_seconds": 0.0,
                    "served_from_cache": True,
                }
                job_id = await q.create(
                    body.question,
                    options=options,
                    trace_context=inject_context(),
                    client=client,
                    status="completed",
                    result=hit["result"],
                    cached=True,
                )
                await q.update(job_id, summary=summary)
                await q.add_event(
                    job_id,
                    {
                        "type": "final",
                        "cached": True,
                        "similarity": similarity,
                        "result": hit["result"],
                        "summary": summary,
                    },
                )
                JOBS.labels("cached").inc()
                JOB_SECONDS.labels("hit").observe(time.time() - started)
            else:
                with span("queue.enqueue"):
                    job_id = await q.create(
                        body.question,
                        options=options,
                        trace_context=inject_context(),
                        client=client,
                    )
            s.set_attribute("bioagent.job_id", job_id)
        HTTP_REQUESTS.labels("/v1/ask", "202").inc()
        return AskResponse(
            job_id=job_id,
            status="completed" if hit is not None else "queued",
            cached=hit is not None,
            job_url=f"/v1/jobs/{job_id}",
            events_url=f"/v1/jobs/{job_id}/events",
            cache_similarity=round(similarity, 4) if similarity else None,
        )

    @app.get("/v1/jobs/{job_id}", response_model=JobView)
    async def get_job(job_id: str, _key: str | None = Depends(authorise)) -> JobView:
        record = await queue().get(job_id)
        if record is None:
            HTTP_REQUESTS.labels("/v1/jobs", "404").inc()
            raise HTTPException(status_code=404, detail="unknown job")
        HTTP_REQUESTS.labels("/v1/jobs", "200").inc()
        summary = record.get("summary")
        return JobView(
            job_id=record["job_id"],
            status=record["status"],
            question=record["question"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            attempts=record["attempts"],
            cached=record["cached"],
            error=record.get("error"),
            summary=JobSummary.model_validate(summary) if summary else None,
            result=record.get("result"),
        )

    @app.get("/v1/jobs/{job_id}/events")
    async def job_events(job_id: str, _key: str | None = Depends(authorise)) -> StreamingResponse:
        q = queue()
        if await q.get(job_id) is None:
            raise HTTPException(status_code=404, detail="unknown job")

        async def stream() -> AsyncIterator[str]:
            async for event in q.events(job_id):
                if event is None:
                    yield ": keep-alive\n\n"
                    continue
                data = json.dumps(event, default=str)
                yield f"event: {event.get('type', 'message')}\ndata: {data}\n\n"

        HTTP_REQUESTS.labels("/v1/jobs/events", "200").inc()
        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        try:
            await state["redis"].ping()
            depth = await queue().depth()
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "redis unavailable", "error": type(exc).__name__},
            )
        return JSONResponse(
            {"status": "ready", "queue": depth, "semantic_cache": state["cache"] is not None}
        )

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def build_cache(settings: Settings, redis: Redis) -> SemanticCache | None:
    """Semantic cache with the corpus embedder; None (cache off) if the models extra or the
    corpus is missing."""
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is None:
        return None
    from bioagent.ingest.store import CorpusStore
    from bioagent.serve.cache import config_version, default_embedder

    store = CorpusStore(settings.corpus_db)
    try:
        fingerprint = store.passage_fingerprint()
    finally:
        store.close()
    return SemanticCache(
        redis,
        default_embedder(settings),
        config_version(settings, fingerprint),
        settings.semantic_cache_threshold,
        settings.semantic_cache_ttl_s,
    )
