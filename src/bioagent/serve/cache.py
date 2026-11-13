"""Semantic answer cache in Redis.

The normalised question is embedded (the corpus embedder, bge-small by default) and compared
by cosine similarity with cached questions of the same *version*. A hit needs similarity >=
``threshold`` (default 0.95). The version hashes everything that changes answers: the corpus
passage fingerprint, the role models, the verifier setup and the prompts, so a corpus or
config change never serves stale answers. Entries expire after ``ttl_s``.

Storage: ``cache:<version>:<question hash>`` hashes (question, vector, result) and a
``cache:<version>:index`` set of their keys. Lookup is a linear scan over the version's
entries, which is fine for thousands of entries; swap in a vector index beyond that.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from redis.asyncio import Redis

from bioagent.config import Settings
from bioagent.eval.cache import stable_hash
from bioagent.serve.queue import PREFIX

Embed = Callable[[str], Sequence[float]]


def normalise(question: str) -> str:
    text = question.strip().lower()
    text = re.sub(r"[\s]+", " ", text)
    return re.sub(r"[?!.\s]+$", "", text)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def config_version(settings: Settings, corpus_fingerprint: str) -> str:
    """Hash of what determines an answer: corpus, models, verifier and prompts."""
    from bioagent.agents import prompts
    from bioagent.llm.providers import role_specs

    return stable_hash(
        {
            "corpus": corpus_fingerprint,
            "models": role_specs(settings).describe(),
            "verify": [settings.verify_claims, settings.verifier_backend, settings.verify_policy],
            "prompts": stable_hash(inspect.getsource(prompts)),
        }
    )


class SemanticCache:
    def __init__(
        self,
        redis: Redis,
        embed: Embed,
        version: str,
        threshold: float = 0.95,
        ttl_s: int = 7 * 24 * 3600,
    ) -> None:
        self.redis = redis
        self.embed = embed
        self.version = version
        self.threshold = threshold
        self.ttl_s = ttl_s

    @property
    def index_key(self) -> str:
        return f"{PREFIX}cache:{self.version}:index"

    def entry_key(self, question: str) -> str:
        return f"{PREFIX}cache:{self.version}:{stable_hash(normalise(question))}"

    async def lookup(self, question: str) -> tuple[dict[str, Any] | None, float]:
        """(cached result or None, best similarity)."""
        keys = [
            k.decode() if isinstance(k, bytes) else str(k)
            for k in await self.redis.smembers(self.index_key)
        ]
        if not keys:
            return None, 0.0
        vector = list(self.embed(normalise(question)))
        best_key, best = None, 0.0
        stale = []
        for key in keys:
            raw = await self.redis.hget(key, "vector")
            if raw is None:
                stale.append(key)
                continue
            score = cosine(vector, json.loads(raw))
            if score > best:
                best_key, best = key, score
        if stale:
            await self.redis.srem(self.index_key, *stale)
        if best_key is None or best < self.threshold:
            return None, best
        raw_result = await self.redis.hget(best_key, "result")
        if raw_result is None:
            return None, best
        return json.loads(raw_result), best

    async def store(self, question: str, result: Mapping[str, Any]) -> None:
        key = self.entry_key(question)
        vector = list(self.embed(normalise(question)))
        await self.redis.hset(
            key,
            mapping={
                "question": question,
                "vector": json.dumps(vector),
                "result": json.dumps(result, default=str),
                "stored_at": time.time(),
            },
        )
        await self.redis.expire(key, self.ttl_s)
        await self.redis.sadd(self.index_key, key)
        await self.redis.expire(self.index_key, self.ttl_s)


def default_embedder(settings: Settings) -> Embed:
    """The corpus query embedder (bge-small by default; needs the ``models`` extra)."""
    from bioagent.retrieval.embedder import SentenceTransformerEmbedder

    model = SentenceTransformerEmbedder(settings.embed_model, device=settings.embed_device)
    return model.embed_query
