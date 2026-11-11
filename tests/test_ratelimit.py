from __future__ import annotations

import asyncio
import time

import pytest

from bioagent.mcp_servers.literature.ratelimit import TokenBucket


async def test_bucket_spaces_requests() -> None:
    bucket = TokenBucket(rate=20.0)  # one token every 50 ms, no burst
    start = time.monotonic()
    await asyncio.gather(*(bucket.acquire() for _ in range(6)))
    elapsed = time.monotonic() - start
    # first is immediate, the remaining 5 wait ~50 ms each
    assert 0.22 <= elapsed < 0.6


async def test_bucket_allows_burst_up_to_capacity() -> None:
    bucket = TokenBucket(rate=5.0, capacity=3)
    start = time.monotonic()
    for _ in range(3):
        await bucket.acquire()
    assert time.monotonic() - start < 0.05
    await bucket.acquire()  # fourth must wait ~200 ms
    assert time.monotonic() - start >= 0.18


def test_bucket_rejects_bad_rate() -> None:
    with pytest.raises(ValueError):
        TokenBucket(rate=0)
