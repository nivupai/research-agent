"""Saved agent runs (``RUNS_DIR``, default results/runs) for the UI's gallery and replay.

Run ids are validated against the pattern the runner writes, so a request can never name a
path outside the runs directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bioagent.serve.worker import slim_result, summarise

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def _records(runs_dir: Path) -> list[Path]:
    if not runs_dir.is_dir():
        return []
    return sorted(
        (p for p in runs_dir.glob("*.json") if RUN_ID_RE.match(p.stem)),
        key=lambda p: p.stem,
        reverse=True,
    )


def list_runs(runs_dir: Path, limit: int = 100) -> list[dict[str, Any]]:
    """Newest first: id, question and headline numbers of each completed run."""
    out = []
    for path in _records(runs_dir):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not record.get("answer"):
            continue
        stats = record.get("stats", {}) or {}
        verification = record.get("verification", {}) or {}
        out.append(
            {
                "run_id": path.stem,
                "question": record.get("question", ""),
                "started_at": record.get("started_at"),
                "groundedness": verification.get("groundedness"),
                "claims": len(verification.get("claims", []) or []),
                "cost_usd": stats.get("cost_usd"),
                "wall_seconds": stats.get("wall_seconds"),
                "models": record.get("models", {}),
            }
        )
        if len(out) >= limit:
            break
    return out


def load_record(runs_dir: Path, run_id: str) -> dict[str, Any] | None:
    if not RUN_ID_RE.match(run_id):
        return None
    path = runs_dir / f"{run_id}.json"
    if not path.is_file():
        return None
    record: dict[str, Any] = json.loads(path.read_text())
    return record


def load_run(runs_dir: Path, run_id: str) -> dict[str, Any] | None:
    """The same shape as a completed job: ``result`` and ``summary``."""
    record = load_record(runs_dir, run_id)
    if record is None:
        return None
    result = slim_result(record)
    return {"run_id": run_id, "result": result, "summary": summarise(result, None)}


def find_by_question(runs_dir: Path, question: str) -> dict[str, Any] | None:
    """The newest saved run whose question matches (case and trailing punctuation ignored),
    preferring runs that were verified (have a claim table)."""

    def norm(q: str) -> str:
        return " ".join(q.casefold().split()).rstrip("?.! ")

    target = norm(question)
    fallback: dict[str, Any] | None = None
    for path in _records(runs_dir):
        try:
            record: dict[str, Any] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not record.get("answer") or norm(str(record.get("question", ""))) != target:
            continue
        if (record.get("verification") or {}).get("claims"):
            return record  # prefer a run with a claim table
        fallback = fallback or record
    return fallback
