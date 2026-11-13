"""p50/p95 end-to-end latency with the semantic cache on vs off (no LLM calls).

Runs the real API (uvicorn), real Redis, the real semantic cache (bge-small embeddings) and
the real worker loop. The worker's pipeline *replays* recorded smoke-baseline answers: it
emits progress events and sleeps for the recorded wall time x ``--scale`` (default 0.1), so
the benchmark costs no Gemini quota. Each of 10 questions is asked in 5 forms (exact,
lowercase, extra spaces, a polite prefix, a typo), sequentially, once with the cache off
(no_cache) and once with it on (fresh cache version). Spans go to Jaeger if OTLP_ENDPOINT is
set. Writes results/serve/bench.json.

Usage: uv run python scripts/bench_serve.py [--scale 0.1] [--port 8765]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from redis.asyncio import Redis

from bioagent.config import get_settings
from bioagent.finetune.metrics import percentile
from bioagent.serve.app import create_app
from bioagent.serve.cache import SemanticCache, default_embedder
from bioagent.serve.queue import JobQueue
from bioagent.serve.worker import Worker
from bioagent.telemetry import setup_tracing

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "results" / "eval" / "smoke" / "smoke-baseline" / "items"
OUT = ROOT / "results" / "serve" / "bench.json"


def variants(q: str) -> list[str]:
    return [
        q,
        q.lower(),
        "  " + q.replace(" ", "  ") + " ",
        "Please answer: " + q,
        q.replace("a", "e", 1),
    ]


def load_records(n: int = 10) -> dict[str, dict[str, Any]]:
    """question -> recorded run record from the smoke baseline (first repeat)."""
    from bioagent.eval.dataset import load_all
    from bioagent.eval.scoring import agent_question

    items = {i.id: i for i in load_all(ROOT, lambda _m: None)}
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(SMOKE.glob("*__r0.json")):
        data = json.loads(path.read_text())
        item = items.get(data["item_id"])
        if item and data.get("status") == "ok" and data["record"].get("answer"):
            out[agent_question(item)] = data["record"]
        if len(out) >= n:
            break
    return out


def replay_pipeline(records: Mapping[str, dict[str, Any]], scale: float) -> Any:
    """Replay the recorded answer for any form of a question, taking its recorded time."""
    by_form = {form: rec for q, rec in records.items() for form in variants(q)}

    async def run(job: Mapping[str, Any], on_event: Any) -> dict[str, Any]:
        record = by_form[str(job["question"])]
        wall = float(record.get("stats", {}).get("wall_seconds") or 10.0)
        for node in ("planner", "literature", "synthesizer", "verifier"):
            await on_event({"type": "progress", "node": node})
            await asyncio.sleep(wall * scale / 4)
        return record

    return run


async def ask_and_wait(client: httpx.AsyncClient, question: str, no_cache: bool) -> dict[str, Any]:
    t0 = time.monotonic()
    r = (await client.post("/v1/ask", json={"question": question, "no_cache": no_cache})).json()
    job_id = r["job_id"]
    while True:
        job = (await client.get(f"/v1/jobs/{job_id}")).json()
        if job["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)
    return {
        "seconds": time.monotonic() - t0,
        "cached": r["cached"],
        "status": job["status"],
        "similarity": r.get("cache_similarity"),
    }


async def main(scale: float, port: int) -> dict[str, Any]:
    settings = get_settings().model_copy(
        update={"rate_limit_burst": 10_000, "rate_limit_per_minute": 1e6}
    )
    traced = setup_tracing("bioagent-bench", settings.otlp_endpoint)
    records = load_records()
    redis = Redis.from_url(settings.redis_url)
    embed = default_embedder(settings)
    embed("warm up")
    results: dict[str, Any] = {
        "scale": scale,
        "questions": len(records),
        "forms": 5,
        "tracing": traced,
    }
    for mode in ("off", "on"):
        cache = SemanticCache(
            redis,
            embed,
            version=f"bench-{mode}-{time.time_ns()}",
            threshold=settings.semantic_cache_threshold,
        )
        app = create_app(settings, redis=redis, cache=cache)
        server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
        serve_task = asyncio.create_task(server.serve())
        for _ in range(200):  # wait up to ~10 s for uvicorn to bind
            if server.started:
                break
            await asyncio.sleep(0.05)
        stop = asyncio.Event()
        worker = Worker(JobQueue(redis), replay_pipeline(records, scale), cache)
        worker_task = asyncio.create_task(worker.run_forever(stop))
        rows = []
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=120) as client:
            for question in records:
                for form in variants(question):
                    rows.append(await ask_and_wait(client, form, no_cache=(mode == "off")))
        stop.set()
        await worker_task
        server.should_exit = True
        await serve_task
        secs = [r["seconds"] for r in rows]
        hits = [r for r in rows if r["cached"]]
        results[mode] = {
            "requests": len(rows),
            "cache_hits": len(hits),
            "hit_rate": len(hits) / len(rows),
            "p50_s": percentile(secs, 0.5),
            "p95_s": percentile(secs, 0.95),
            "mean_s": sum(secs) / len(secs),
            "hit_p50_ms": (percentile([r["seconds"] for r in hits], 0.5) or 0) * 1000,
            "hit_p95_ms": (percentile([r["seconds"] for r in hits], 0.95) or 0) * 1000,
            "missed_forms": [round(r["similarity"] or 0, 3) for r in rows if not r["cached"]],
        }
        print(mode, {k: v for k, v in results[mode].items() if k != "missed_forms"})
    await redis.aclose()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    asyncio.run(main(args.scale, args.port))
    time.sleep(2)  # let the batch span processor flush to Jaeger
