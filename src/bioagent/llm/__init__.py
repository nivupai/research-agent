"""LLM provider layer: Gemini (free tier) and Ollama (local) chat models with per-call stats."""

from bioagent.llm.providers import (
    LLMError,
    ModelSpec,
    QuotaExhaustedError,
    RoleModels,
    make_chat_model,
    parse_spec,
    role_specs,
)
from bioagent.llm.stats import LLMCallRecord, RunStatsCollector, ToolCallRecord

__all__ = [
    "LLMCallRecord",
    "LLMError",
    "ModelSpec",
    "QuotaExhaustedError",
    "RoleModels",
    "RunStatsCollector",
    "ToolCallRecord",
    "make_chat_model",
    "parse_spec",
    "role_specs",
]
