"""Chat-model construction and a resilient invoke helper.

Role models are configured as ``"provider:model"`` (e.g. ``gemini:gemini-3.5-flash`` or
``ollama:qwen3:4b``). ``ainvoke_with_stats`` adds retries with backoff for free-tier 429s and
transient 5xx errors, raises ``QuotaExhaustedError`` with a clear message when the daily
quota is used up, and records tokens, latency and estimated cost.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from bioagent.config import Settings
from bioagent.llm.prices import estimate_cost
from bioagent.llm.stats import LLMCallRecord, RunStatsCollector

SleepFn = Callable[[float], Awaitable[None]]


class LLMError(RuntimeError):
    """An LLM call failed after retries."""


class QuotaExhaustedError(LLMError):
    """The provider's (free-tier) quota is used up; retrying now will not help."""


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class RoleModels:
    planner: ModelSpec
    specialist: ModelSpec
    synth: ModelSpec
    # Per-specialist overrides keyed by task kind (literature, chemistry, analysis).
    specialists: Mapping[str, ModelSpec] = field(default_factory=dict)

    def specialist_for(self, kind: str) -> ModelSpec:
        return self.specialists.get(kind, self.specialist)

    def describe(self) -> dict[str, str]:
        out = {"planner": str(self.planner), "specialist": str(self.specialist)}
        out |= {f"specialist:{k}": str(v) for k, v in sorted(self.specialists.items())}
        return out | {"synth": str(self.synth)}


def parse_spec(spec: str) -> ModelSpec:
    provider, sep, model = spec.partition(":")
    if not sep or provider not in ("gemini", "ollama") or not model:
        raise ValueError(f"model spec must be 'gemini:<model>' or 'ollama:<model>', got {spec!r}")
    return ModelSpec(provider, model)


def role_specs(settings: Settings, provider: str | None = None) -> RoleModels:
    """Resolve role models. ``provider`` (CLI ``--provider``) overrides LLM_PROVIDER and the
    per-role settings."""
    chosen = provider or settings.llm_provider
    if chosen == "ollama":
        default = ModelSpec("ollama", settings.ollama_model)
        defaults = RoleModels(default, default, default)
    elif chosen == "gemini":
        defaults = RoleModels(
            ModelSpec("gemini", settings.gemini_model),
            ModelSpec("gemini", settings.gemini_fast_model),
            ModelSpec("gemini", settings.gemini_model),
        )
    else:
        raise ValueError(f"unknown provider {chosen!r}")
    if provider is not None:
        return defaults
    per_kind = {
        "literature": settings.literature_model,
        "chemistry": settings.chemistry_model,
        "analysis": settings.analysis_model,
    }
    return RoleModels(
        parse_spec(settings.planner_model) if settings.planner_model else defaults.planner,
        parse_spec(settings.specialist_model) if settings.specialist_model else defaults.specialist,
        parse_spec(settings.synth_model) if settings.synth_model else defaults.synth,
        {k: parse_spec(v) for k, v in per_kind.items() if v},
    )


def make_chat_model(spec: ModelSpec, settings: Settings) -> BaseChatModel:
    if spec.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if settings.gemini_api_key is None:
            raise LLMError("GEMINI_API_KEY is not set (see .env.example)")
        return ChatGoogleGenerativeAI(
            model=spec.model,
            google_api_key=settings.gemini_api_key,
            # Gemini 3.x models use fixed sampling defaults; temperature would be ignored.
            max_retries=0,  # retries are handled by ainvoke_with_stats
            timeout=settings.llm_timeout_s,
        )
    if spec.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=spec.model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            reasoning=False,  # qwen3: skip the thinking phase for tool loops
            num_ctx=16384,
        )
    raise ValueError(f"unknown provider {spec.provider!r}")


# --- error classification --------------------------------------------------------------
_RETRY_DELAY_RE = re.compile(r"retry(?:Delay|_delay|\s+in)[\"':\s]*([\d.]+)\s*s", re.I)


def classify_error(exc: BaseException) -> tuple[str, float | None]:
    """Return (kind, suggested_delay_s): kind is 'quota', 'rate_limit', 'transient' or
    'fatal'."""
    text = f"{type(exc).__name__}: {exc}"
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    lowered = text.lower()
    delay_match = _RETRY_DELAY_RE.search(text)
    delay = float(delay_match.group(1)) if delay_match else None
    if code == 429 or "429" in text or "resource_exhausted" in lowered or "rate limit" in lowered:
        if "perday" in lowered.replace(" ", "") or "per day" in lowered or "daily" in lowered:
            return "quota", delay
        return "rate_limit", delay
    if code in (500, 502, 503, 504) or any(
        s in text for s in ("503", "UNAVAILABLE", "500 INTERNAL", "Deadline", "timed out")
    ):
        return "transient", delay
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient", delay
    return "fatal", None


async def ainvoke_with_stats(
    model: Any,
    messages: Sequence[BaseMessage],
    *,
    spec: ModelSpec,
    role: str,
    stats: RunStatsCollector,
    task_id: str | None = None,
    max_retries: int = 5,
    sleep: SleepFn = asyncio.sleep,
) -> Any:
    """Invoke a chat model (or a structured-output runnable with ``include_raw=True``)."""
    retries = 0
    while True:
        started = time.monotonic()
        try:
            result = await model.ainvoke(list(messages))
            break
        except Exception as exc:
            kind, delay = classify_error(exc)
            if kind == "quota":
                raise QuotaExhaustedError(
                    f"{spec} quota exhausted (free-tier daily limit?). Wait for the quota to "
                    "reset, use a different GEMINI_MODEL, or run with --provider ollama. "
                    f"Provider said: {str(exc)[:300]}"
                ) from exc
            if kind == "fatal" or retries >= max_retries:
                raise LLMError(
                    f"{spec} call failed ({kind}) after {retries} retries: {str(exc)[:500]}"
                ) from exc
            wait = delay if delay is not None else min(60.0, 2.0 * 2**retries)
            wait = wait + random.uniform(0, 1.0)
            retries += 1
            await sleep(wait)
    raw = result.get("raw") if isinstance(result, dict) else result
    usage: dict[str, Any] = dict(raw.usage_metadata or {}) if isinstance(raw, AIMessage) else {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    stats.add_llm(
        LLMCallRecord(
            role=role,
            task_id=task_id,
            provider=spec.provider,
            model=spec.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=int((time.monotonic() - started) * 1000),
            cost_usd=estimate_cost(spec.provider, spec.model, input_tokens, output_tokens),
            retries=retries,
        )
    )
    return result
