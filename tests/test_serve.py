"""Serving stack with fakeredis and a fake pipeline (no Gemini, no Redis server, no models)."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from fakeredis import FakeAsyncRedis
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from bioagent.config import Settings
from bioagent.llm.providers import LLMError, QuotaExhaustedError
from bioagent.serve.app import create_app
from bioagent.serve.cache import SemanticCache, cosine, normalise
from bioagent.serve.queue import DELAYED, JobQueue
from bioagent.serve.ratelimit import TokenBucket
from bioagent.serve.worker import Worker, is_transient, slim_result

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)


@pytest.fixture
async def redis() -> AsyncIterator[FakeAsyncRedis]:
    r = FakeAsyncRedis()
    yield r
    await r.flushall()
    await r.aclose()


def bag_embed(text: str) -> list[float]:
    """Deterministic fake embedder: hashed bag of words (similar questions -> high cosine)."""
    vec = [0.0] * 64
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[sum(map(ord, word)) % 64] += 1.0
    return vec


def record(answer: str = "Metformin activates AMPK [1].") -> dict[str, Any]:
    return {
        "run_id": "r1",
        "answer": answer,
        "citations": [{"n": 1, "pmid": "1"}],
        "verification": {
            "groundedness": 1.0,
            "claims": [
                {
                    "n": 1,
                    "claim": "Metformin activates AMPK.",
                    "label": "supported",
                    "check": "NLI 0.99",
                    "citations": [1],
                    "evidence": [],
                }
            ],
        },
        "plan": [{"id": "t1", "kind": "literature"}],
        "models": {"planner": "fake"},
        "stats": {
            "wall_seconds": 1.5,
            "llm_calls": 3,
            "tool_calls": 2,
            "input_tokens": 100,
            "output_tokens": 20,
            "cost_usd": 0.0001,
        },
    }


class FakePipeline:
    def __init__(self, failures: list[BaseException] | None = None) -> None:
        self.failures = list(failures or [])
        self.jobs: list[Mapping[str, Any]] = []

    async def __call__(self, job: Mapping[str, Any], on_event: Any) -> dict[str, Any]:
        self.jobs.append(job)
        await on_event({"type": "progress", "node": "planner", "plan": [{"id": "t1"}]})
        if self.failures:
            raise self.failures.pop(0)
        await on_event({"type": "progress", "node": "verifier", "groundedness": 1.0})
        return record()


# --- queue -------------------------------------------------------------------------------------
async def test_queue_lifecycle_and_events(redis: FakeAsyncRedis) -> None:
    q = JobQueue(redis)
    job_id = await q.create("What does metformin do?", options={}, trace_context={}, client="c")
    assert (await q.get(job_id) or {})["status"] == "queued"
    assert await q.depth() == {"ready": 1, "delayed": 0}
    assert await q.dequeue(timeout_s=0.1) == job_id
    assert await q.dequeue(timeout_s=0.1) is None
    await q.schedule_retry(job_id, 0.0, "TimeoutError")
    assert (await q.get(job_id) or {})["status"] == "retrying"
    assert await q.promote_due() == 1 and await redis.zcard(DELAYED) == 0
    await q.update(job_id, status="completed")
    await q.add_event(job_id, {"type": "final"})
    events = [e async for e in q.events(job_id, block_ms=10)]
    assert [e["type"] for e in events if e] == ["queued", "retrying", "final"]


# --- rate limiting ------------------------------------------------------------------------------
async def test_token_bucket(redis: FakeAsyncRedis) -> None:
    bucket = TokenBucket(redis, rate_per_minute=60, burst=3)  # 1 token per second
    results = [await bucket.allow("a", now=100.0) for _ in range(4)]
    assert [ok for ok, _ in results] == [True, True, True, False]
    assert results[-1][1] == pytest.approx(1.0)
    assert (await bucket.allow("b", now=100.0))[0]  # clients are independent
    assert (await bucket.allow("a", now=101.5))[0]  # refilled after 1.5 s
    assert not (await bucket.allow("a", now=101.6))[0]


# --- semantic cache ---------------------------------------------------------------------
def test_normalise_and_cosine() -> None:
    assert normalise("  What does Metformin  do?? ") == "what does metformin do"
    assert cosine([1, 0], [1, 0]) == 1.0 and cosine([1, 0], [0, 1]) == 0.0
    assert math.isclose(cosine([1, 1], [1, 0]), 1 / math.sqrt(2))


async def test_semantic_cache_threshold_and_versions(redis: FakeAsyncRedis) -> None:
    cache = SemanticCache(redis, bag_embed, version="v1", threshold=0.95)
    assert await cache.lookup("Which drugs act on AMPK?") == (None, 0.0)
    await cache.store("Which drugs act on AMPK?", {"result": {"answer": "x"}})
    hit, score = await cache.lookup("which drugs act on ampk")
    assert hit == {"result": {"answer": "x"}} and score == pytest.approx(1.0)
    miss, score = await cache.lookup("How does temozolomide damage DNA?")
    assert miss is None and score < 0.95
    other = SemanticCache(redis, bag_embed, version="v2", threshold=0.95)
    assert (await other.lookup("Which drugs act on AMPK?"))[0] is None  # new config version
    await redis.delete(cache.entry_key("Which drugs act on AMPK?"))  # expired entry
    assert (await cache.lookup("Which drugs act on AMPK?"))[0] is None
    assert await redis.scard(cache.index_key) == 0  # stale key pruned


# --- worker --------------------------------------------------------------------------------------
async def test_worker_completes_and_caches(redis: FakeAsyncRedis) -> None:
    q = JobQueue(redis)
    cache = SemanticCache(redis, bag_embed, version="v1")
    pipeline = FakePipeline()
    job_id = await q.create("Which drugs act on AMPK?", options={}, trace_context={}, client="c")
    assert await Worker(q, pipeline, cache).run_one(job_id) == "completed"
    job = await q.get(job_id) or {}
    assert job["status"] == "completed" and job["attempts"] == 1
    assert job["result"]["claims"][0]["label"] == "supported"
    assert job["summary"]["cost_usd"] == 0.0001 and job["summary"]["queue_seconds"] >= 0
    events = [e["type"] for e in [ev async for ev in q.events(job_id, block_ms=10)] if e]
    assert events == ["queued", "running", "progress", "progress", "final"]
    assert (await cache.lookup("Which drugs act on AMPK?"))[0] is not None


async def test_worker_retries_transient_then_resumes_same_run(redis: FakeAsyncRedis) -> None:
    q = JobQueue(redis)
    pipeline = FakePipeline(failures=[TimeoutError("slow upstream")])
    worker = Worker(q, pipeline, max_attempts=3, retry_base_s=0.0)
    job_id = await q.create("Q about AMPK?", options={}, trace_context={}, client="c")
    assert await worker.run_one(job_id) == "retrying"
    assert await q.promote_due() == 1
    assert await q.dequeue(timeout_s=0.1) == job_id
    assert await worker.run_one(job_id) == "completed"
    assert [j["run_id"] for j in pipeline.jobs] == [f"job-{job_id}"] * 2  # same checkpoint
    assert (await q.get(job_id) or {})["attempts"] == 2


async def test_worker_gives_up(redis: FakeAsyncRedis) -> None:
    q = JobQueue(redis)
    quota = Worker(q, FakePipeline(failures=[QuotaExhaustedError("quota")]))
    job_id = await q.create("Q about AMPK?", options={}, trace_context={}, client="c")
    assert await quota.run_one(job_id) == "failed"  # quota is not retried
    assert "quota" in ((await q.get(job_id)) or {})["error"]
    worn = Worker(q, FakePipeline(failures=[TimeoutError()] * 2), max_attempts=2, retry_base_s=0.0)
    job2 = await q.create("Another AMPK question?", options={}, trace_context={}, client="c")
    assert await worn.run_one(job2) == "retrying"
    await q.promote_due()
    assert await worn.run_one(job2) == "failed"


def test_transient_classification() -> None:
    assert is_transient(TimeoutError()) and is_transient(ConnectionError())
    assert is_transient(LLMError("call failed (transient) after 5 retries"))
    assert not is_transient(LLMError("call failed (fatal) after 0 retries"))
    assert not is_transient(QuotaExhaustedError("quota")) and not is_transient(ValueError())
    assert slim_result(record())["groundedness"] == 1.0


# --- API -------------------------------------------------------------------------------------
def settings(**kw: Any) -> Settings:
    return Settings(_env_file=None, rate_limit_burst=5, rate_limit_per_minute=60, **kw)


async def client_for(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_api_ask_job_events_and_cache(redis: FakeAsyncRedis) -> None:
    cache = SemanticCache(redis, bag_embed, version="v1")
    app = create_app(settings(), redis=redis, cache=cache)
    async with await client_for(app) as client:
        r = await client.post("/v1/ask", json={"question": "Which drugs act on AMPK?"})
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "queued" and not body["cached"]
        job_id = body["job_id"]
        q = JobQueue(redis)
        assert await q.dequeue(timeout_s=0.1) == job_id
        await Worker(q, FakePipeline(), cache).run_one(job_id)
        job = (await client.get(f"/v1/jobs/{job_id}")).json()
        assert job["status"] == "completed" and job["summary"]["llm_calls"] == 3
        async with client.stream("GET", f"/v1/jobs/{job_id}/events") as stream:
            text = "".join([chunk async for chunk in stream.aiter_text()])
        assert "event: final" in text and "event: progress" in text
        # same question again (different wording/case): served from the cache, no queueing
        again = (await client.post("/v1/ask", json={"question": "which drugs act on AMPK"})).json()
        assert again["cached"] and again["status"] == "completed" and again["cache_similarity"]
        cached_job = (await client.get(f"/v1/jobs/{again['job_id']}")).json()
        assert cached_job["result"]["answer"].startswith("Metformin")
        assert await q.depth() == {"ready": 0, "delayed": 0}
        fresh = (
            await client.post(
                "/v1/ask", json={"question": "Which drugs act on AMPK?", "no_cache": True}
            )
        ).json()
        assert not fresh["cached"] and fresh["status"] == "queued"
        assert (await client.get("/v1/jobs/nope")).status_code == 404
        metrics = (await client.get("/metrics")).text
        assert "bioagent_semantic_cache_total" in metrics
        assert (await client.get("/healthz")).json() == {"status": "ok"}
        assert (await client.get("/readyz")).json()["status"] == "ready"


async def test_api_key_rate_limit_and_validation(redis: FakeAsyncRedis) -> None:
    app = create_app(settings(api_key=SecretStr("k1")), redis=redis, cache=None)
    async with await client_for(app) as client:
        q = {"question": "Which drugs act on AMPK?"}
        assert (await client.post("/v1/ask", json=q)).status_code == 401
        headers = {"X-API-Key": "k1"}
        codes = [
            (await client.post("/v1/ask", json=q, headers=headers)).status_code for _ in range(6)
        ]
        assert codes == [202] * 5 + [429]
        assert (
            await client.post("/v1/ask", json={"question": "x"}, headers=headers)
        ).status_code == 422


# --- telemetry -----------------------------------------------------------------------------
async def test_spans_cover_queue_worker_node_and_llm(redis: FakeAsyncRedis) -> None:
    from bioagent.llm.stats import RunStatsCollector
    from bioagent.telemetry import inject_context, span, traced_node

    EXPORTER.clear()

    async def node(_state: Any) -> dict[str, Any]:
        RunStatsCollector().add_llm(
            {
                "role": "planner",
                "task_id": None,
                "provider": "gemini",
                "model": "m",
                "input_tokens": 10,
                "output_tokens": 5,
                "latency_ms": 20,
                "cost_usd": 0.001,
                "retries": 0,
            }
        )
        return {}

    async def pipeline(job: Mapping[str, Any], on_event: Any) -> dict[str, Any]:
        await traced_node("planner", node)({})
        return record()

    q = JobQueue(redis)
    with span("api.ask"):
        job_id = await q.create(
            "Q about AMPK?", options={}, trace_context=inject_context(), client="c"
        )
    await Worker(q, pipeline).run_one(job_id)
    spans = {s.name: s for s in EXPORTER.get_finished_spans()}
    assert {"api.ask", "worker.job", "graph.node.planner", "llm.call"} <= set(spans)
    trace_id = spans["api.ask"].context.trace_id
    assert all(
        spans[n].context.trace_id == trace_id
        for n in ("worker.job", "graph.node.planner", "llm.call")
    )
    llm = spans["llm.call"]
    assert (
        llm.parent is not None and llm.parent.span_id == spans["graph.node.planner"].context.span_id
    )
    assert llm.attributes is not None and llm.attributes["gen_ai.usage.input_tokens"] == 10
    assert llm.attributes["bioagent.cost_usd"] == 0.001


# --- progress events from the real graph (fake models) ----------------------------------------
async def test_execute_streams_progress_events(tmp_path: Path) -> None:
    from bioagent.agents.runner import execute
    from tests.agent_fakes import ToolLog, call, corpus_hit, fake_tools, plan, scripted_specialist
    from tests.test_agents import make_deps

    deps = make_deps(
        [plan(("t1", "literature", []))],
        scripted_specialist({"t1": [[call("search_corpus", {"query": "q"})]]}),
        "Metformin activates AMPK [1].",
        fake_tools(ToolLog(), corpus={"q": [corpus_hit("111", "Paper")]}),
    )
    events: list[dict[str, Any]] = []

    async def sink(event: dict[str, Any]) -> None:
        events.append(event)

    state = await execute(deps, tmp_path / "ck.sqlite", "r1", "Q? about AMPK", on_event=sink)
    nodes = [e["node"] for e in events]
    assert nodes[0] == "planner" and "literature" in nodes and nodes[-1] == "verifier"
    assert events[0]["plan"][0]["kind"] == "literature"
    lit = next(e for e in events if e["node"] == "literature")
    assert lit["task_id"] == "t1" and lit["evidence"] == 1
    assert state["answer"].startswith("Metformin")
    assert json.dumps(events)  # JSON-friendly
    await asyncio.sleep(0)


async def test_replay_pipeline_runs_through_the_worker(redis: FakeAsyncRedis) -> None:
    from bioagent.serve.replay import replay_pipeline

    q = JobQueue(redis)
    job_id = await q.create("Which drugs act on AMPK?", options={}, trace_context={}, client="c")
    assert await Worker(q, replay_pipeline(0.0)).run_one(job_id) == "completed"
    job = await q.get(job_id) or {}
    assert job["result"]["answer"].startswith("[replay pipeline]")
    assert job["summary"]["llm_calls"] == 0 and job["summary"]["cost_usd"] == 0.0


# --- UI support: result payload, saved runs, replay of saved runs, /ui --------------------------


def ui_record(run_id: str = "20261112-083229-fd26dc", question: str = "Does X work?") -> Any:
    rec = record("X works [1]. X is safe [2].")
    url = "https://pubmed.ncbi.nlm.nih.gov/11/"
    rec |= {
        "run_id": run_id,
        "question": question,
        "started_at": "2026-11-12T08:32:29+00:00",
        "citations": [
            {"n": 1, "key": "11:0", "pmid": "11", "title": "T1", "url": url},
            {"n": 2, "key": "22:0", "pmid": "22", "title": "T2", "url": None},
        ],
        "cited_evidence": {
            "11:0": {"text": "X works well. More text."},
            "22:0": {"text": "X is toxic."},
        },
        "tool_calls": [{"task_id": "t1", "tool": "search_corpus", "ok": True, "latency_ms": 5}],
    }
    supported = {
        "n": 1,
        "claim": "X works.",
        "label": "supported",
        "check": "NLI 0.9",
        "citations": [1],
        "sentence": 1,
        "evidence": [{"cite": 1, "entailment": 0.9, "contradiction": 0.0, "window": "X works."}],
    }
    contradicted = {
        "n": 2,
        "claim": "X is safe.",
        "label": "contradicted",
        "check": "NLI contradiction 0.99",
        "citations": [2],
        "sentence": 2,
        "evidence": [
            {
                "cite": 2,
                "entailment": 0.0,
                "contradiction": 0.99,
                "window": "X is toxic.",
                "contradiction_window": "X is toxic.",
            }
        ],
    }
    rec["verification"] = {
        "groundedness": 0.5,
        "removed_sentences": [2],
        "flagged_sentences": [],
        "claims": [supported, contradicted],
    }
    return rec


def test_slim_result_carries_status_evidence_and_passages() -> None:
    from bioagent.serve.worker import slim_result

    out = slim_result(ui_record())
    assert [c["status"] for c in out["claims"]] == ["supported", "removed"]
    assert out["claims"][1]["evidence"][0]["contradiction_window"] == "X is toxic."
    assert out["passages"]["1"]["text"] == "X works well. More text."
    assert out["passages"]["1"]["url"].endswith("/11/")
    assert out["removed_sentences"] == [2] and out["question"] == "Does X work?"


def test_saved_runs_listing_loading_and_path_safety(tmp_path: Path) -> None:
    from bioagent.serve.runs import find_by_question, list_runs, load_run

    (tmp_path / "20261112-083229-fd26dc.json").write_text(json.dumps(ui_record()))
    unverified = ui_record("20261113-000000-aaaaaa") | {"verification": {}}
    (tmp_path / "20261113-000000-aaaaaa.json").write_text(json.dumps(unverified))
    (tmp_path / "notes.json").write_text("{}")
    runs = list_runs(tmp_path)
    assert [r["run_id"] for r in runs] == ["20261113-000000-aaaaaa", "20261112-083229-fd26dc"]
    assert runs[1]["claims"] == 2 and runs[1]["groundedness"] == 0.5
    run = load_run(tmp_path, "20261112-083229-fd26dc") or {}
    assert run["summary"]["groundedness"] == 0.5 and len(run["result"]["claims"]) == 2
    assert load_run(tmp_path, "../secrets") is None and load_run(tmp_path, "notes") is None
    # the newer run has no claim table, so the verified one is preferred
    found = find_by_question(tmp_path, "  does x WORK ")
    assert found is not None and found["run_id"] == "20261112-083229-fd26dc"
    assert find_by_question(tmp_path, "something else") is None


async def test_api_saved_runs_and_ui(redis: FakeAsyncRedis, tmp_path: Path) -> None:
    (tmp_path / "20261112-083229-fd26dc.json").write_text(json.dumps(ui_record()))
    app = create_app(settings(runs_dir=tmp_path, api_key=SecretStr("k1")), redis=redis, cache=None)
    async with await client_for(app) as c:
        page = await c.get("/ui")
        assert page.status_code == 200 and "BioResearch Agent" in page.text
        assert "X-API-Key" in page.text and "k1" not in page.text  # no secret in the page
        assert (await c.get("/v1/runs")).status_code == 401
        h = {"X-API-Key": "k1"}
        listing = (await c.get("/v1/runs", headers=h)).json()["runs"]
        assert listing[0]["run_id"] == "20261112-083229-fd26dc"
        run = (await c.get("/v1/runs/20261112-083229-fd26dc", headers=h)).json()
        assert run["result"]["passages"]["2"]["text"] == "X is toxic."
        assert (await c.get("/v1/runs/..%2F..%2Fetc", headers=h)).status_code == 404
        assert (await c.get("/", follow_redirects=False)).headers["location"] == "/ui"


async def test_replay_pipeline_replays_a_saved_run(redis: FakeAsyncRedis, tmp_path: Path) -> None:
    from bioagent.serve.replay import replay_events, replay_pipeline

    rec = ui_record()
    (tmp_path / "20261112-083229-fd26dc.json").write_text(json.dumps(rec))
    nodes = [e["node"] for e in replay_events(rec)]
    assert nodes == ["planner", "literature", "synthesizer", "verifier"]
    q = JobQueue(redis)
    job_id = await q.create("does x work", options={}, trace_context={}, client="c")
    assert await Worker(q, replay_pipeline(0.0, tmp_path)).run_one(job_id) == "completed"
    job = await q.get(job_id) or {}
    assert job["result"]["replayed_from"] == "20261112-083229-fd26dc"
    assert job["result"]["claims"][1]["status"] == "removed"
    events = [e async for e in q.events(job_id, block_ms=10) if e is not None]
    assert [e.get("tool") for e in events if e.get("tool")] == ["search_corpus"]
