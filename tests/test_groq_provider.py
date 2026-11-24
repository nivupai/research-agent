"""Groq provider (OpenAI-compatible endpoint), fully mocked: no network, no key."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, SecretStr

from bioagent.config import Settings
from bioagent.llm import openai_compat
from bioagent.llm.openai_compat import ChatOpenAICompat, ProviderHTTPError
from bioagent.llm.providers import (
    LLMError,
    ModelSpec,
    QuotaExhaustedError,
    ainvoke_with_stats,
    classify_error,
    make_chat_model,
    parse_spec,
    role_specs,
)
from bioagent.llm.stats import RunStatsCollector


def settings(**kw: Any) -> Settings:
    return Settings(_env_file=None, **({"groq_api_key": None} | kw))  # type: ignore[call-arg]


@tool
def search_pubmed(query: str, k: int = 5) -> str:
    """Search PubMed."""
    return query


class Answer(BaseModel):
    text: str
    confidence: float


def completion(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "llama-3.3-70b-versatile",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Route httpx.AsyncClient to a MockTransport; responses are queued in ``replies``."""
    requests: list[dict[str, Any]] = []
    replies: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(
            {"url": str(request.url), "headers": request.headers, **json.loads(request.content)}
        )
        return replies.pop(0)

    real = httpx.AsyncClient

    def client(**kw: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(openai_compat.httpx, "AsyncClient", client)
    requests.append({"_replies": replies})
    return requests


def model() -> ChatOpenAICompat:
    return ChatOpenAICompat(model="llama-3.3-70b-versatile", api_key=SecretStr("test-key"))


def test_spec_and_roles() -> None:
    assert parse_spec("groq:llama-3.3-70b-versatile") == ModelSpec(
        "groq", "llama-3.3-70b-versatile"
    )
    s = settings(groq_model="big", groq_fast_model="small")
    r = role_specs(s, "groq")
    assert (r.planner, r.specialist, r.synth) == (
        ModelSpec("groq", "big"),
        ModelSpec("groq", "small"),
        ModelSpec("groq", "big"),
    )
    assert role_specs(settings()).planner.provider == "gemini"  # Gemini stays the default
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        make_chat_model(ModelSpec("groq", "x"), settings())
    m = make_chat_model(ModelSpec("groq", "x"), settings(groq_api_key="gsk-secret-123"))
    assert isinstance(m, ChatOpenAICompat) and "gsk-secret-123" not in repr(m)


async def test_tool_calling_round_trip(mock_http: list[dict[str, Any]]) -> None:
    replies = mock_http.pop(0)["_replies"]
    replies.append(
        httpx.Response(
            200,
            json=completion(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_pubmed",
                                "arguments": '{"query": "AMPK", "k": 3}',
                            },
                        }
                    ],
                }
            ),
        )
    )
    replies.append(httpx.Response(200, json=completion({"content": "Metformin activates AMPK."})))
    bound = model().bind_tools([search_pubmed])
    first = await bound.ainvoke([HumanMessage("drugs acting on AMPK?")])
    assert first.tool_calls == [
        {
            "name": "search_pubmed",
            "args": {"query": "AMPK", "k": 3},
            "id": "call_1",
            "type": "tool_call",
        }
    ]
    assert first.usage_metadata["input_tokens"] == 12
    second = await bound.ainvoke(
        [HumanMessage("drugs acting on AMPK?"), first, ToolMessage("PMID 1", tool_call_id="call_1")]
    )
    assert second.content == "Metformin activates AMPK."
    req1, req2 = mock_http
    assert req1["url"].endswith("/openai/v1/chat/completions")
    assert req1["headers"]["authorization"] == "Bearer test-key"
    assert req1["tools"][0]["function"]["name"] == "search_pubmed"
    assert req1["model"] == "llama-3.3-70b-versatile"
    # the assistant tool call and the tool result are sent back in OpenAI format
    assert req2["messages"][1]["tool_calls"][0]["id"] == "call_1"
    assert req2["messages"][2] == {"role": "tool", "content": "PMID 1", "tool_call_id": "call_1"}


async def test_structured_output_include_raw(mock_http: list[dict[str, Any]]) -> None:
    replies = mock_http.pop(0)["_replies"]
    args = json.dumps({"text": "yes", "confidence": 0.9})
    replies.append(
        httpx.Response(
            200,
            json=completion(
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "Answer", "arguments": args},
                        }
                    ],
                }
            ),
        )
    )
    out = await model().with_structured_output(Answer, include_raw=True).ainvoke("q")
    assert out["parsed"] == Answer(text="yes", confidence=0.9)
    assert mock_http[0]["tool_choice"] in (
        "required",
        {"type": "function", "function": {"name": "Answer"}},
    )


async def test_errors_are_classified(mock_http: list[dict[str, Any]]) -> None:
    replies = mock_http.pop(0)["_replies"]
    daily = (
        '{"error":{"message":"Rate limit reached for model on tokens per day (TPD): Limit 100000"}}'
    )
    replies.append(httpx.Response(429, text=daily))
    with pytest.raises(QuotaExhaustedError):
        await ainvoke_with_stats(
            model(),
            [HumanMessage("q")],
            spec=ModelSpec("groq", "m"),
            role="planner",
            stats=RunStatsCollector(),
        )
    assert classify_error(ProviderHTTPError(429, "requests per minute"))[0] == "rate_limit"
    assert classify_error(ProviderHTTPError(503, "over capacity"))[0] == "transient"
    assert classify_error(ProviderHTTPError(400, "bad request"))[0] == "fatal"
