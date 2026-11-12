"""Content-addressed caches for evaluation (JSON values in SQLite).

Keys are SHA-256 hashes of a canonical JSON payload, so they are stable across processes
and machines: ``judge_key(question, answer, prompt_version, kind)`` and
``item_key(item_id, question, repeat, config_hash)``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def stable_hash(payload: Any, length: int = 16) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def answer_hash(answer: str) -> str:
    return stable_hash(" ".join(answer.split()), 24)


def judge_key(question: str, answer: str, prompt_version: str, kind: str) -> str:
    return stable_hash(
        {"q": question.strip(), "a": answer_hash(answer), "v": prompt_version, "k": kind}, 32
    )


def item_key(item_id: str, question: str, repeat: int, config_hash: str) -> str:
    return stable_hash({"id": item_id, "q": question.strip(), "r": repeat, "c": config_hash}, 24)


class JSONCache:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def close(self) -> None:
        self._conn.close()
