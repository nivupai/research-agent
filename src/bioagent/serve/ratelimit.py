"""Per-client token bucket in Redis (optimistic WATCH/MULTI transaction; no Lua needed)."""

from __future__ import annotations

import time

from redis.asyncio import Redis
from redis.exceptions import WatchError

from bioagent.serve.queue import PREFIX


class TokenBucket:
    """``rate_per_minute`` tokens are added continuously up to ``burst``; a request costs 1."""

    def __init__(self, redis: Redis, rate_per_minute: float, burst: int) -> None:
        self.redis = redis
        self.rate = rate_per_minute / 60.0
        self.burst = burst

    def key(self, client: str) -> str:
        return f"{PREFIX}ratelimit:{client}"

    async def allow(self, client: str, now: float | None = None) -> tuple[bool, float]:
        """(allowed, seconds until the next token if refused)."""
        key = self.key(client)
        for _ in range(10):
            async with self.redis.pipeline() as pipe:
                try:
                    await pipe.watch(key)
                    data = await pipe.hgetall(key)
                    t = now if now is not None else time.time()
                    tokens = float(data.get(b"tokens", data.get("tokens", self.burst)))
                    last = float(data.get(b"ts", data.get("ts", t)))
                    tokens = min(self.burst, tokens + (t - last) * self.rate)
                    allowed = tokens >= 1.0
                    if allowed:
                        tokens -= 1.0
                    pipe.multi()  # type: ignore[no-untyped-call]
                    pipe.hset(key, mapping={"tokens": tokens, "ts": t})
                    pipe.expire(key, int(self.burst / max(self.rate, 1e-9)) + 60)
                    await pipe.execute()
                except WatchError:
                    continue
            wait = 0.0 if allowed else (1.0 - tokens) / max(self.rate, 1e-9)
            return allowed, wait
        return False, 1.0
