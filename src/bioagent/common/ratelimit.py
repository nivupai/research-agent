"""Async token-bucket rate limiter."""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Token bucket allowing ``rate`` acquisitions per second with bursts up to ``capacity``.

    The default capacity of 1 spaces requests evenly, which is the safest way to stay under a
    per-second quota such as NCBI's.
    """

    def __init__(self, rate: float, capacity: float = 1.0) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                await asyncio.sleep((1 - self._tokens) / self.rate)
