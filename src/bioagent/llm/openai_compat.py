"""A small chat model for OpenAI-compatible chat-completions endpoints (used for Groq).

Supports what the agent needs: plain chat, ``bind_tools`` (specialist tool loops) and
``with_structured_output`` (LangChain's default tool-calling implementation, built on
``bind_tools``). Talks HTTP through httpx, so no provider SDK is required. Errors carry
``status_code`` so ``providers.classify_error`` can tell rate limits, daily quota and 5xx apart.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import SecretStr

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class ProviderHTTPError(RuntimeError):
    """Non-2xx response from the endpoint; ``status_code`` is the HTTP status."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body[:800]}")
        self.status_code = status_code


class ChatOpenAICompat(BaseChatModel):
    model: str
    api_key: SecretStr
    base_url: str = GROQ_BASE_URL
    temperature: float = 0.0
    timeout: float = 120.0
    provider_name: str = "groq"

    @property
    def _llm_type(self) -> str:
        return f"openai-compat:{self.provider_name}"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | dict[str, Any] | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        if tool_choice is True or tool_choice == "any":
            tool_choice = "required"
        if isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
            tool_choice = {"type": "function", "function": {"name": tool_choice}}
        if tool_choice is not None and tool_choice is not False:
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted, **kwargs)

    # --- request / response -------------------------------------------------------------
    def _payload(
        self, messages: list[BaseMessage], stop: list[str] | None, **kwargs: Any
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": convert_to_openai_messages(messages),
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop
        for key in ("tools", "tool_choice", "parallel_tool_calls"):
            if kwargs.get(key) is not None:
                payload[key] = kwargs[key]
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _result(self, response: httpx.Response) -> ChatResult:
        if response.status_code >= 400:
            raise ProviderHTTPError(response.status_code, response.text)
        return parse_completion(response.json())

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, stop, **kwargs),
            )
        return self._result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, stop, **kwargs),
            )
        return self._result(response)


def parse_completion(body: dict[str, Any]) -> ChatResult:
    """Turn a chat-completions response into a ChatResult (tool calls, token usage)."""
    message = body["choices"][0]["message"]
    calls = []
    invalid = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            calls.append(tool_call(name=fn.get("name", ""), args=args, id=call.get("id")))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            invalid.append(
                {
                    "name": fn.get("name"),
                    "args": raw_args,
                    "id": call.get("id"),
                    "error": str(exc),
                    "type": "invalid_tool_call",
                }
            )
    usage = body.get("usage") or {}
    ai = AIMessage(
        content=message.get("content") or "",
        tool_calls=calls,
        invalid_tool_calls=invalid,
        usage_metadata={
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
        response_metadata={"model_name": body.get("model"), "finish_reason": None},
    )
    finish = body["choices"][0].get("finish_reason")
    ai.response_metadata["finish_reason"] = finish
    return ChatResult(generations=[ChatGeneration(message=ai)])
