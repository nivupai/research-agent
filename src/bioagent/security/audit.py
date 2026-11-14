"""Append-only, hash-chained audit log per run (``data/audit/<run id>.jsonl``).

Each line is one entry ``{seq, ts, run_id, job_id, trace_id, type, data, prev, hash}`` where
``hash = sha256(prev + canonical JSON of the entry without hash)``; the first entry chains
from ``"0" * 64``. Editing, deleting or reordering any line breaks the chain, which
``verify_log`` (``bioagent audit verify``) detects. Entries are optionally mirrored into
``data/audit/audit.sqlite``.

Recorded: question (hash and length only), plan, every tool call (role, tool, args hash,
redacted args, allowed/denied, reason, latency), every LLM call (model, tokens, list-price
cost, latency), verifier decisions, output-check flags and the final answer hash. Arguments
are redacted: long strings are truncated and anything that looks like a credential is masked.
Questions and answers themselves are not stored, only their hashes.

The log for the current run is held in a context variable (``use_audit``), so instrumented
code (stats collector, tool policy, graph nodes) writes to it without extra plumbing.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
MAX_ARG_CHARS = 200
_SECRET_RE = re.compile(
    r"(AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|"
    r"(?:api[_-]?key|token|password|secret|bearer)\s*[:=]\s*\S+|[A-Za-z0-9+/_\-]{40,}={0,2})",
    re.I,
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def redact(value: Any) -> Any:
    """Truncate long strings and mask credential-like substrings, recursively."""
    if isinstance(value, str):
        masked = _SECRET_RE.sub("[redacted]", value)
        return masked if len(masked) <= MAX_ARG_CHARS else masked[:MAX_ARG_CHARS] + "...[truncated]"
    if isinstance(value, Mapping):
        return {str(k): redact(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value[:50]]
    return value


def current_trace_id() -> str | None:
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    return f"{ctx.trace_id:032x}" if ctx.is_valid else None


class AuditLog:
    def __init__(
        self, directory: Path, run_id: str, job_id: str | None = None, sqlite: bool = False
    ) -> None:
        self.directory = Path(directory)
        self.run_id = run_id
        self.job_id = job_id
        self.path = self.directory / f"{run_id}.jsonl"
        self.sqlite_path = self.directory / "audit.sqlite" if sqlite else None
        self._lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._seq, self._prev = self._tail()

    def _tail(self) -> tuple[int, str]:
        """Continue an existing log (e.g. a resumed run) from its last entry."""
        if not self.path.exists():
            return 0, GENESIS
        last: dict[str, Any] | None = None
        for line in self.path.read_text().splitlines():
            if line.strip():
                last = json.loads(line)
        return (int(last["seq"]), str(last["hash"])) if last else (0, GENESIS)

    def append(self, type_: str, data: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            entry: dict[str, Any] = {
                "seq": self._seq + 1,
                "ts": round(time.time(), 3),
                "run_id": self.run_id,
                "job_id": self.job_id,
                "trace_id": current_trace_id(),
                "type": type_,
                "data": dict(data),
                "prev": self._prev,
            }
            entry["hash"] = sha256(self._prev + canonical(entry))
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(canonical(entry) + "\n")
            if self.sqlite_path is not None:
                self._mirror(entry)
            self._seq, self._prev = int(entry["seq"]), str(entry["hash"])
            return entry

    def _mirror(self, entry: Mapping[str, Any]) -> None:
        assert self.sqlite_path is not None
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS audit (run_id TEXT, seq INTEGER, ts REAL, job_id TEXT,"
                " trace_id TEXT, type TEXT, data TEXT, prev TEXT, hash TEXT,"
                " PRIMARY KEY (run_id, seq))"
            )
            conn.execute(
                "INSERT INTO audit VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry["run_id"],
                    entry["seq"],
                    entry["ts"],
                    entry["job_id"],
                    entry["trace_id"],
                    entry["type"],
                    canonical(entry["data"]),
                    entry["prev"],
                    entry["hash"],
                ),
            )
            conn.commit()
        finally:
            conn.close()


@dataclass(frozen=True)
class Verification:
    ok: bool
    entries: int
    error: str | None = None
    bad_seq: int | None = None


def verify_log(path: Path) -> Verification:
    prev = GENESIS
    n = 0
    for n, line in enumerate((ln for ln in path.read_text().splitlines() if ln.strip()), 1):
        entry = json.loads(line)
        claimed = entry.pop("hash", None)
        if entry.get("seq") != n:
            return Verification(False, n, f"entry {n} has seq {entry.get('seq')}", n)
        if entry.get("prev") != prev:
            return Verification(False, n, f"entry {n} does not chain from entry {n - 1}", n)
        if sha256(prev + canonical(entry)) != claimed:
            return Verification(False, n, f"entry {n} was modified (hash mismatch)", n)
        prev = str(claimed)
    return Verification(True, n)


def find_log(directory: Path, ident: str) -> Path:
    """Accept a run id, a job id (runs are named ``job-<id>``) or a path."""
    for candidate in (Path(ident), directory / f"{ident}.jsonl", directory / f"job-{ident}.jsonl"):
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no audit log for {ident!r} in {directory}")


# --- current-run context ---------------------------------------------------------------------
_CURRENT: ContextVar[AuditLog | None] = ContextVar("audit_log", default=None)


@contextmanager
def use_audit(log: AuditLog | None) -> Iterator[AuditLog | None]:
    token = _CURRENT.set(log)
    try:
        yield log
    finally:
        _CURRENT.reset(token)


def audit(type_: str, **data: Any) -> None:
    """Append to the current run's audit log (no-op outside a run)."""
    log = _CURRENT.get()
    if log is not None:
        log.append(type_, data)


def args_digest(args: Mapping[str, Any]) -> dict[str, Any]:
    return {"args_hash": sha256(canonical(args))[:16], "args": redact(dict(args))}
