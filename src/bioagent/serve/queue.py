"""A small Redis job queue (redis-py asyncio).

Keys (prefix ``bioagent:``):

* ``job:<id>``            hash: the job record (JSON fields), expires after JOB_TTL
* ``queue``               list of ready job ids (LPUSH / BRPOP)
* ``delayed``             sorted set of job ids by retry time (exponential backoff)
* ``events:<id>``         stream of progress events (XADD; read by the SSE endpoint)

The worker takes a job with BRPOP, marks it running, and either completes it, schedules a
retry (``delayed``) or fails it. Delayed jobs whose time has come are promoted to the ready
list by ``promote_due`` on every dequeue.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

from redis.asyncio import Redis

from bioagent.serve.models import TERMINAL

PREFIX = "bioagent:"
QUEUE = f"{PREFIX}queue"
DELAYED = f"{PREFIX}delayed"
MAX_EVENTS = 500


def job_key(job_id: str) -> str:
    return f"{PREFIX}job:{job_id}"


def events_key(job_id: str) -> str:
    return f"{PREFIX}events:{job_id}"


def new_job_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(4)


class JobQueue:
    def __init__(self, redis: Redis, job_ttl_s: int = 7 * 24 * 3600) -> None:
        self.redis = redis
        self.job_ttl_s = job_ttl_s

    # --- jobs ---------------------------------------------------------------------------
    async def create(
        self,
        question: str,
        *,
        options: Mapping[str, Any],
        trace_context: Mapping[str, str],
        client: str,
        status: str = "queued",
        result: Mapping[str, Any] | None = None,
        cached: bool = False,
    ) -> str:
        job_id = new_job_id()
        now = time.time()
        record = {
            "job_id": job_id,
            "question": question,
            "options": dict(options),
            "trace_context": dict(trace_context),
            "client": client,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "attempts": 0,
            "cached": cached,
            "run_id": f"job-{job_id}",
            "error": None,
            "result": dict(result) if result is not None else None,
        }
        await self._save(record)
        if status == "queued":
            await self.redis.lpush(QUEUE, job_id)
            await self.add_event(job_id, {"type": "queued"})
        return job_id

    async def get(self, job_id: str) -> dict[str, Any] | None:
        raw = await self.redis.hget(job_key(job_id), "data")
        return json.loads(raw) if raw else None

    async def _save(self, record: Mapping[str, Any]) -> None:
        key = job_key(str(record["job_id"]))
        await self.redis.hset(key, mapping={"data": json.dumps(record, default=str)})
        await self.redis.expire(key, self.job_ttl_s)

    async def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        record = await self.get(job_id)
        if record is None:
            raise KeyError(job_id)
        record.update(fields, updated_at=time.time())
        await self._save(record)
        return record

    # --- queue --------------------------------------------------------------------------
    async def promote_due(self, now: float | None = None) -> int:
        due = await self.redis.zrangebyscore(DELAYED, 0, now or time.time())
        moved = 0
        for raw in due:
            job_id = raw.decode() if isinstance(raw, bytes) else str(raw)
            if await self.redis.zrem(DELAYED, job_id):
                await self.redis.lpush(QUEUE, job_id)
                moved += 1
        return moved

    async def dequeue(self, timeout_s: float = 1.0) -> str | None:
        await self.promote_due()
        item = await self.redis.brpop([QUEUE], timeout=timeout_s)
        if not item:
            return None
        raw = item[1]
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    async def schedule_retry(self, job_id: str, delay_s: float, error: str) -> None:
        await self.update(job_id, status="retrying", error=error)
        await self.redis.zadd(DELAYED, {job_id: time.time() + delay_s})
        await self.add_event(
            job_id, {"type": "retrying", "in_seconds": round(delay_s, 1), "error": error[:300]}
        )

    async def depth(self) -> dict[str, int]:
        return {
            "ready": int(await self.redis.llen(QUEUE)),
            "delayed": int(await self.redis.zcard(DELAYED)),
        }

    # --- events ---------------------------------------------------------------------------
    async def add_event(self, job_id: str, event: Mapping[str, Any]) -> None:
        payload = {"ts": time.time(), **event}
        key = events_key(job_id)
        await self.redis.xadd(
            key, {"data": json.dumps(payload, default=str)}, maxlen=MAX_EVENTS, approximate=True
        )
        await self.redis.expire(key, self.job_ttl_s)

    async def events(
        self, job_id: str, *, block_ms: int = 15_000, idle_limit: int = 40
    ) -> AsyncIterator[dict[str, Any] | None]:
        """Replay stored events, then follow new ones until the job ends. Yields None as a
        heartbeat when nothing arrived within ``block_ms`` (at most ``idle_limit`` times)."""
        last = "0-0"
        idle = 0
        while True:
            items: Any = await self.redis.xread(
                {events_key(job_id): last}, block=block_ms, count=100
            )
            if not items:
                record = await self.get(job_id)
                if record is None or record["status"] in TERMINAL or idle >= idle_limit:
                    return
                idle += 1
                yield None
                continue
            idle = 0
            for _stream, entries in items:
                for entry_id, fields in entries:
                    last = entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
                    data = fields.get(b"data") or fields.get("data")
                    event = json.loads(data)
                    yield event
                    if event.get("type") in ("final", "failed"):
                        return
