"""HTTP client with per-host rate limiting and retries (exponential backoff + jitter)."""

from __future__ import annotations

import asyncio
import email.utils
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from bioagent.mcp_servers.literature.ratelimit import TokenBucket

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
USER_AGENT = "bioagent/0.1 (+https://github.com/nivupai/research-agent)"

SleepFn = Callable[[float], Awaitable[None]]


def _redact(url: str) -> str:
    """Drop query string and fragment so secrets in parameters never reach logs or errors."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


class LiteratureAPIError(RuntimeError):
    """Raised when an upstream API fails after all retries or returns an error payload."""


def parse_retry_after(value: str | None, max_wait: float = 60.0) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP date) into seconds."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        seconds = when.timestamp() - time.time()
    return min(max(seconds, 0.0), max_wait)


class RateLimitedClient:
    """Wraps ``httpx.AsyncClient``; each host gets its own token bucket."""

    def __init__(
        self,
        rates: Mapping[str, float],
        *,
        default_rate: float = 3.0,
        max_attempts: int = 5,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
        timeout: float = 30.0,
        sleep: SleepFn = asyncio.sleep,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._buckets = {host: TokenBucket(rate) for host, rate in rates.items()}
        self._default_rate = default_rate
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._sleep = sleep
        self._client = client or httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        )

    def bucket_for(self, url: str) -> TokenBucket:
        host = urlsplit(url).hostname or ""
        if host not in self._buckets:
            self._buckets[host] = TokenBucket(self._default_rate)
        return self._buckets[host]

    def _backoff(self, attempt: int) -> float:
        ceiling = min(self._backoff_max, self._backoff_base * 2**attempt)
        return random.uniform(ceiling / 2, ceiling)

    async def request(
        self, method: str, url: str, *, params: Mapping[str, Any] | None = None
    ) -> httpx.Response:
        """Send a request, retrying on 429/5xx and transport errors/timeouts."""
        bucket = self.bucket_for(url)
        last_error: str = "unknown error"
        for attempt in range(self._max_attempts):
            await bucket.acquire()
            try:
                response = await self._client.request(method, url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                delay = self._backoff(attempt)
            else:
                if response.status_code not in RETRY_STATUSES:
                    if response.is_error:
                        # Never include the query string: it may carry an API key.
                        raise LiteratureAPIError(
                            f"{method} {_redact(url)} returned HTTP {response.status_code}"
                        )
                    return response
                last_error = f"HTTP {response.status_code}"
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                delay = retry_after if retry_after is not None else self._backoff(attempt)
            if attempt + 1 < self._max_attempts:
                await self._sleep(delay)
        raise LiteratureAPIError(
            f"{method} {url} failed after {self._max_attempts} attempts: {last_error}"
        )

    async def get(self, url: str, *, params: Mapping[str, Any] | None = None) -> httpx.Response:
        return await self.request("GET", url, params=params)

    async def aclose(self) -> None:
        await self._client.aclose()
