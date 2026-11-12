from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from bioagent.config import Settings
from bioagent.llm.prices import estimate_cost
from bioagent.llm.providers import (
    LLMError,
    ModelSpec,
    QuotaExhaustedError,
    ainvoke_with_stats,
    classify_error,
    parse_spec,
    role_specs,
)
from bioagent.llm.stats import RunStatsCollector, summarize

SPEC = ModelSpec("gemini", "gemini-3.5-flash-lite")


def settings(**kw: Any) -> Settings:
    return Settings(_env_file=None, gemini_api_key=None, **kw)  # type: ignore[call-arg]


def test_parse_spec() -> None:
    assert parse_spec("ollama:qwen3:4b") == ModelSpec("ollama", "qwen3:4b")
    assert str(parse_spec("gemini:gemini-3.5-flash")) == "gemini:gemini-3.5-flash"
    for bad in ["qwen3", "openai:gpt", "gemini:"]:
        with pytest.raises(ValueError):
            parse_spec(bad)


def test_role_specs_defaults_and_overrides() -> None:
    s = settings()
    specs = role_specs(s)
    assert specs.planner.model == s.gemini_model and specs.specialist.model == s.gemini_fast_model
    o = role_specs(s, "ollama")
    assert o.planner == o.specialist == o.synth == ModelSpec("ollama", s.ollama_model)
    custom = role_specs(settings(specialist_model="ollama:qwen3:4b"))
    assert custom.specialist == ModelSpec("ollama", "qwen3:4b")
    assert custom.planner.provider == "gemini"
    # --provider wins over per-role settings
    assert (
        role_specs(settings(specialist_model="gemini:x"), "ollama").specialist.provider == "ollama"
    )


def test_cost_estimates() -> None:
    assert estimate_cost("ollama", "qwen3:4b", 10_000, 10_000) == 0.0
    assert estimate_cost("gemini", "gemini-3.5-flash-lite", 1_000_000, 100_000) == pytest.approx(
        0.30 + 0.25
    )
    assert estimate_cost("gemini", "unknown-model", 1, 1) is None


class Err(Exception):
    def __init__(self, msg: str, code: int | None = None) -> None:
        super().__init__(msg)
        self.code = code


def test_classify_error() -> None:
    assert classify_error(Err("429 RESOURCE_EXHAUSTED retryDelay: '7s'", 429)) == (
        "rate_limit",
        7.0,
    )
    assert (
        classify_error(Err("429 ... GenerateRequestsPerDayPerProjectPerModel-FreeTier"))[0]
        == "quota"
    )
    assert classify_error(Err("503 UNAVAILABLE high demand", 503))[0] == "transient"
    assert classify_error(Err("400 INVALID_ARGUMENT", 400)) == ("fatal", None)


class Flaky:
    def __init__(self, errors: list[Exception]) -> None:
        self.errors = errors
        self.calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100},
        )


async def test_retries_rate_limits_then_records_stats() -> None:
    waits: list[float] = []

    async def sleep(d: float) -> None:
        waits.append(d)

    model = Flaky([Err("429 RESOURCE_EXHAUSTED retryDelay: '3s'", 429), Err("503", 503)])
    stats = RunStatsCollector()
    out = await ainvoke_with_stats(model, [], spec=SPEC, role="planner", stats=stats, sleep=sleep)
    assert out.content == "ok" and model.calls == 3
    assert 3.0 <= waits[0] <= 4.0 and 4.0 <= waits[1] <= 5.0
    [rec] = stats.llm_calls
    assert rec["retries"] == 2 and rec["input_tokens"] == 1000
    assert rec["cost_usd"] == pytest.approx((1000 * 0.30 + 100 * 2.50) / 1e6)


async def test_quota_exhausted_is_clear_and_not_retried() -> None:
    model = Flaky([Err("429 Quota exceeded for metric: GenerateRequestsPerDay", 429)])
    with pytest.raises(QuotaExhaustedError, match="quota exhausted"):
        await ainvoke_with_stats(model, [], spec=SPEC, role="x", stats=RunStatsCollector())
    assert model.calls == 1


async def test_gives_up_after_max_retries() -> None:
    async def sleep(d: float) -> None:
        pass

    model = Flaky([Err("503", 503)] * 5)
    with pytest.raises(LLMError, match="after 2 retries"):
        await ainvoke_with_stats(
            model, [], spec=SPEC, role="x", stats=RunStatsCollector(), max_retries=2, sleep=sleep
        )


def test_summarize_partial_cost() -> None:
    calls: Any = [
        {
            "provider": "gemini",
            "model": "a",
            "input_tokens": 10,
            "output_tokens": 5,
            "latency_ms": 100,
            "cost_usd": 0.01,
            "retries": 1,
            "role": "p",
            "task_id": None,
        },
        {
            "provider": "gemini",
            "model": "b",
            "input_tokens": 20,
            "output_tokens": 5,
            "latency_ms": 100,
            "cost_usd": None,
            "retries": 0,
            "role": "s",
            "task_id": "t1",
        },
    ]
    s = summarize(
        calls, [{"task_id": "t1", "tool": "x", "ok": False, "latency_ms": 1, "error": "e"}]
    )
    assert s["cost_usd"] == 0.01 and s["cost_complete"] is False
    assert s["input_tokens"] == 30 and s["tool_errors"] == 1 and s["llm_retries"] == 1
