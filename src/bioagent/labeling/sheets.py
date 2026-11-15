"""Label sheets: the three JSONL files people grade by hand, behind one small interface.

* ``retrieval``: ``data/labels/pool.jsonl``, field ``grade`` (0, 1, 2), one row per
  (query, PMID) pooled from every retrieval mode.
* ``verifier``: ``eval/labels/verifier_test_human.jsonl``, field ``human_label``
  (SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO) for a claim and its passage.
* ``judge``: ``eval/labels/judge_calibration.jsonl``, field ``human_label`` on the judge's
  scale. The judge's own label and rationale are never shown (blind labelling).

Every write re-reads the file, changes one row (found by its key, not its position) and
replaces the file atomically, so labels survive a crash and a regenerated sheet (``make
label-pool`` keeps existing grades) is picked up without restarting the server.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Row = dict[str, Any]
Label = str | int


@dataclass(frozen=True)
class Option:
    value: Label
    text: str
    key: str  # keyboard shortcut


@dataclass(frozen=True)
class Section:
    heading: str
    body: str
    kind: str = "text"  # "text" or "list"
    items: tuple[str, ...] = ()


@dataclass(frozen=True)
class SheetSpec:
    name: str
    title: str
    default_path: Path
    label_field: str
    key: Callable[[Row], str]
    options: Callable[[Row], list[Option]]
    render: Callable[[Row], list[Section]]
    group: Callable[[Row], str] = lambda _row: ""
    instructions: str = ""


def _numbered(values: Sequence[str]) -> list[Option]:
    return [Option(v, v.replace("_", " ").lower(), str(i)) for i, v in enumerate(values, 1)]


RETRIEVAL_OPTIONS = [
    Option(0, "not relevant", "0"),
    Option(1, "partially relevant", "1"),
    Option(2, "highly relevant", "2"),
]


def _retrieval_render(row: Row) -> list[Section]:
    meta = f"PMID {row.get('pmid')} · {row.get('year', '?')} · query type {row.get('type')}"
    return [
        Section("Query", str(row.get("query", ""))),
        Section(str(row.get("title", "")), str(row.get("abstract", "")) or "(no abstract)"),
        Section("", meta, kind="meta"),
    ]


def _verifier_render(row: Row) -> list[Section]:
    return [
        Section("Claim", str(row.get("claim", ""))),
        Section(f"Passage (PMID {row.get('pmid')})", str(row.get("passage", ""))),
    ]


def _judge_render(row: Row) -> list[Section]:
    sections = [
        Section("Question", str(row.get("question", ""))),
        Section("Gold answer", str(row.get("gold", ""))),
    ]
    facts = [str(f) for f in row.get("key_facts") or []]
    if facts:
        sections.append(Section("Key facts", "", kind="list", items=tuple(facts)))
    if row.get("notes"):
        sections.append(Section("Notes", str(row["notes"])))
    sections.append(Section("Agent answer (grade this)", str(row.get("answer", ""))))
    return sections


SPECS: dict[str, SheetSpec] = {
    "retrieval": SheetSpec(
        name="retrieval",
        title="Retrieval relevance",
        default_path=Path("data/labels/pool.jsonl"),
        label_field="grade",
        key=lambda r: f"{r['query_id']}:{r['pmid']}",
        options=lambda _r: list(RETRIEVAL_OPTIONS),
        render=_retrieval_render,
        group=lambda r: str(r["query_id"]),
        instructions=(
            "Would this abstract help answer the query? 2 = directly answers or is central "
            "evidence; 1 = related and partly useful; 0 = off-topic or only shares terms."
        ),
    ),
    "verifier": SheetSpec(
        name="verifier",
        title="Claim verification",
        default_path=Path("eval/labels/verifier_test_human.jsonl"),
        label_field="human_label",
        key=lambda r: str(r["id"]),
        options=lambda r: _numbered(
            r.get("options") or ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
        ),
        render=_verifier_render,
        instructions=(
            "Judge the claim against this passage only, not background knowledge. Supported: "
            "the passage states or directly implies it. Contradicted: the passage says the "
            "opposite. Not enough info: anything else."
        ),
    ),
    "judge": SheetSpec(
        name="judge",
        title="Judge calibration",
        default_path=Path("eval/labels/judge_calibration.jsonl"),
        label_field="human_label",
        key=lambda r: str(r["id"]),
        options=lambda r: _numbered(r.get("scale") or ["incorrect", "partially", "correct"]),
        render=_judge_render,
        instructions=(
            "Grade the agent answer against the gold answer and key facts, as the LLM judge "
            "would. The judge's own label is hidden so it does not anchor you."
        ),
    ),
}


def read_jsonl(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: Sequence[Row]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def is_labelled(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


@dataclass
class Sheet:
    """One label file plus the order items are presented in.

    ``focus`` (optional) is a set of group ids (retrieval: query ids) to label first; the
    progress target is then those groups, and the rest follow after them.
    """

    spec: SheetSpec
    path: Path
    focus: frozenset[str] = frozenset()
    seed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def rows(self) -> list[Row]:
        return read_jsonl(self.path)

    def order(self, rows: Sequence[Row]) -> list[str]:
        """Keys in presentation order: focus groups first, whole groups together.

        Within a group the order is shuffled (seeded), so pool order (which follows the
        retrieval modes' ranks) does not prime the grader.
        """
        groups: dict[str, list[str]] = {}
        for row in rows:
            groups.setdefault(self.spec.group(row), []).append(self.spec.key(row))
        rng = random.Random(self.seed)
        for keys in groups.values():
            rng.shuffle(keys)
        names = [g for g in groups if g in self.focus] + [g for g in groups if g not in self.focus]
        return [k for g in names for k in groups[g]]

    def progress(self, rows: Sequence[Row] | None = None) -> dict[str, Any]:
        rows = self.rows() if rows is None else rows
        done = sum(is_labelled(r.get(self.spec.label_field)) for r in rows)
        out: dict[str, Any] = {"done": done, "total": len(rows)}
        if self.focus:
            in_focus = [r for r in rows if self.spec.group(r) in self.focus]
            out["focus_done"] = sum(is_labelled(r.get(self.spec.label_field)) for r in in_focus)
            out["focus_total"] = len(in_focus)
            out["focus_groups"] = len(self.focus)
        return out

    def next_key(self, after: str | None = None, skipped: Sequence[str] = ()) -> str | None:
        """The next unlabelled key after ``after`` (wrapping); skipped keys come last."""
        rows = self.rows()
        by_key = {self.spec.key(r): r for r in rows}
        order = self.order(rows)
        start = order.index(after) + 1 if after in by_key else 0
        rotated = order[start:] + order[:start]
        todo = [k for k in rotated if not is_labelled(by_key[k].get(self.spec.label_field))]
        skip = set(skipped)
        fresh = [k for k in todo if k not in skip]
        if fresh:
            return fresh[0]
        return todo[0] if todo else None

    def previous_key(self, before: str) -> str | None:
        order = self.order(self.rows())
        if before not in order:
            return None
        i = order.index(before)
        return order[i - 1] if i > 0 else None

    def item(self, key: str) -> dict[str, Any]:
        rows = self.rows()
        row = next((r for r in rows if self.spec.key(r) == key), None)
        if row is None:
            raise KeyError(key)
        order = self.order(rows)
        return {
            "key": key,
            "position": order.index(key) + 1,
            "group": self.spec.group(row),
            "in_focus": self.spec.group(row) in self.focus if self.focus else None,
            "label": row.get(self.spec.label_field),
            "options": [o.__dict__ for o in self.spec.options(row)],
            "sections": [s.__dict__ | {"items": list(s.items)} for s in self.spec.render(row)],
            "progress": self.progress(rows),
        }

    def set_label(self, key: str, value: Label | None) -> dict[str, Any]:
        """Set (or clear, with None) one row's label and save the file atomically."""
        with self._lock:
            rows = self.rows()
            row = next((r for r in rows if self.spec.key(r) == key), None)
            if row is None:
                raise KeyError(key)
            if value is not None:
                allowed = [o.value for o in self.spec.options(row)]
                matched = [a for a in allowed if str(a) == str(value)]
                if not matched:
                    raise ValueError(f"{value!r} is not one of {allowed}")
                value = matched[0]
            row[self.spec.label_field] = value if value is not None else _blank(self.spec)
            write_jsonl_atomic(self.path, rows)
            return self.progress(rows)


def _blank(spec: SheetSpec) -> Any:
    return None if spec.name == "retrieval" else ""
