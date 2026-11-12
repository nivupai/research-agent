"""Tests that call real LLMs (Gemini free tier / local Ollama). Run with ``make test-llm``."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from bioagent.agents import prompts
from bioagent.agents.state import Plan
from bioagent.config import get_settings
from bioagent.llm.providers import ModelSpec, ainvoke_with_stats, make_chat_model
from bioagent.llm.stats import RunStatsCollector

pytestmark = pytest.mark.llm


async def _plan_with(spec: ModelSpec) -> tuple[Plan, RunStatsCollector]:
    settings = get_settings()
    model = make_chat_model(spec, settings).with_structured_output(Plan, include_raw=True)
    stats = RunStatsCollector()
    out = await ainvoke_with_stats(
        model,
        [
            SystemMessage(prompts.PLANNER),
            HumanMessage("Question: Compare the TPSA of metformin and temozolomide."),
        ],
        spec=spec,
        role="planner",
        stats=stats,
    )
    assert out["parsed"] is not None, out.get("parsing_error")
    return out["parsed"], stats


async def test_gemini_planner_structured_output() -> None:
    settings = get_settings()
    if settings.gemini_api_key is None:
        pytest.skip("GEMINI_API_KEY not set")
    plan, stats = await _plan_with(ModelSpec("gemini", settings.gemini_model))
    assert any(t.kind == "chemistry" for t in plan.tasks)
    assert stats.llm_calls[0]["input_tokens"] > 0


async def test_ollama_planner_structured_output() -> None:
    import httpx

    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            tags = (await client.get(f"{settings.ollama_base_url}/api/tags")).json()
    except httpx.HTTPError:
        pytest.skip("Ollama not running")
    if not any(m["name"] == settings.ollama_model for m in tags.get("models", [])):
        pytest.skip(f"{settings.ollama_model} not pulled")
    plan, stats = await _plan_with(ModelSpec("ollama", settings.ollama_model))
    assert plan.tasks and stats.llm_calls[0]["cost_usd"] == 0.0
