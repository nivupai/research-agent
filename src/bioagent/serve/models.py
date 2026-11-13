"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "retrying", "completed", "failed"]
TERMINAL: frozenset[str] = frozenset({"completed", "failed"})


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    provider: Literal["gemini", "ollama"] | None = Field(
        default=None, description="override LLM_PROVIDER for this job"
    )
    no_cache: bool = Field(default=False, description="skip the semantic cache lookup")


class AskResponse(BaseModel):
    job_id: str
    status: JobStatus
    cached: bool
    job_url: str
    events_url: str
    cache_similarity: float | None = None


class JobSummary(BaseModel):
    """Per-query cost and latency, stored with the result."""

    wall_seconds: float | None = None
    queue_seconds: float | None = None
    llm_calls: int | None = None
    tool_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = Field(default=None, description="list-price equivalent")
    groundedness: float | None = None


class JobView(BaseModel):
    job_id: str
    status: JobStatus
    question: str
    created_at: float
    updated_at: float
    attempts: int
    cached: bool
    error: str | None = None
    summary: JobSummary | None = None
    result: dict[str, Any] | None = None
