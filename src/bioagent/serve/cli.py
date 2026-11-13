"""``bioagent serve`` (API) and ``bioagent worker`` (queue consumer) entry points."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

from bioagent.config import Settings, get_settings


def add_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    serve = sub.add_parser("serve", help="run the HTTP API (FastAPI + uvicorn)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    worker = sub.add_parser("worker", help="run a queue worker (the agent pipeline)")
    worker.add_argument("--concurrency", type=int, help="override WORKER_CONCURRENCY")


def run_api(args: argparse.Namespace) -> int:
    import uvicorn

    from bioagent.serve.app import create_app
    from bioagent.telemetry import setup_tracing

    settings = get_settings()
    setup_tracing("bioagent-api", settings.otlp_endpoint)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


async def worker_main(settings: Settings, concurrency: int) -> None:
    from prometheus_client import start_http_server
    from redis.asyncio import Redis

    from bioagent.agents.tools import MCPToolHub
    from bioagent.serve.app import build_cache
    from bioagent.serve.queue import JobQueue
    from bioagent.serve.worker import Worker, agent_pipeline

    start_http_server(settings.worker_metrics_port)
    redis = Redis.from_url(settings.redis_url)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    cache = build_cache(settings, redis) if settings.semantic_cache_enabled else None
    try:
        async with MCPToolHub(log_dir=Path(settings.mcp_log_dir)) as hub:
            worker = Worker(
                JobQueue(redis, settings.job_ttl_s),
                agent_pipeline(settings, hub),
                cache,
                max_attempts=settings.job_max_attempts,
                retry_base_s=settings.job_retry_base_s,
            )
            print(
                f"worker ready (concurrency {concurrency}, metrics :{settings.worker_metrics_port})"
            )
            await worker.run_forever(stop, concurrency)
    finally:
        await redis.aclose()


def run_worker(args: argparse.Namespace) -> int:
    from bioagent.telemetry import setup_tracing

    settings = get_settings()
    setup_tracing("bioagent-worker", settings.otlp_endpoint)
    asyncio.run(worker_main(settings, args.concurrency or settings.worker_concurrency))
    return 0
