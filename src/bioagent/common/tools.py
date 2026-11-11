"""Helpers shared by the MCP servers."""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from mcp.server.mcpserver.exceptions import ToolError

from bioagent.common.httpclient import UpstreamAPIError

P = ParamSpec("P")
R = TypeVar("R")


def tool_errors(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Report expected failures (bad input, upstream API errors) to the MCP client.

    MCP 2.x hides the message of any exception other than ``ToolError`` and returns a generic
    "Error executing tool" instead. Expected errors are re-raised as ``ToolError`` so the
    caller sees a useful message; anything else stays a crash whose text is kept server-side.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await func(*args, **kwargs)
        except (ValueError, UpstreamAPIError) as exc:
            raise ToolError(str(exc)) from exc

    return wrapper
