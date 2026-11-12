"""Evaluation question set: schema, loading and validation.

Items live in ``eval/questions/<version>/*.jsonl`` (one JSON object per line). The schema is
:class:`EvalItem`; ``eval/questions/schema.json`` is generated from it. ``validate_items``
checks cross-item rules (unique ids, gold consistent with the scoring type, poisoned passages
referenced by injection items exist).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

DATASET_VERSION = "v1"
Category = Literal["pubmedqa", "corpus", "injection", "unanswerable", "false_premise", "computed"]
ScoringType = Literal["label", "free_text", "numeric", "injection", "abstain", "premise"]
EXPECTED_SCORING: dict[str, str] = {
    "pubmedqa": "label",
    "corpus": "free_text",
    "injection": "injection",
    "unanswerable": "abstain",
    "false_premise": "premise",
    "computed": "numeric",
}
LABELS = ("yes", "no", "maybe")


class NumericGold(BaseModel):
    name: str = Field(description="property, e.g. tpsa")
    value: float
    tolerance: float = Field(default=0.05, description="absolute tolerance")


class PoisonedPassage(BaseModel):
    pmid: str = Field(pattern=r"^99\d{6}$", description="reserved fake PMID range 99xxxxxx")
    title: str
    abstract: str
    year: int = 2023


class Injection(BaseModel):
    """A poisoned passage (only ever added to the isolated eval index) and what it must not
    achieve. Injection items are user-provided: see ``INJECTION_FILE``."""

    passage: PoisonedPassage
    canaries: list[str] = Field(min_length=1, description="strings that must not be output")
    forbidden_tools: list[str] = Field(default_factory=list)
    forbidden_pmids: list[str] = Field(default_factory=list)

    @property
    def passage_pmid(self) -> str:
        return self.passage.pmid


class EvalItem(BaseModel):
    id: str = Field(pattern=r"^[a-z_]+-[0-9a-z_-]+$")
    category: Category
    question: str = Field(min_length=10)
    scoring_type: ScoringType
    gold: str | list[NumericGold] | None = Field(
        default=None, description="label (yes/no/maybe), short answer, numeric values or null"
    )
    gold_pmids: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    injection: Injection | None = None
    needs_review: bool = False
    source: str = Field(description="where the item comes from (dataset, generator, author)")
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> EvalItem:
        expected = EXPECTED_SCORING[self.category]
        if self.scoring_type != expected:
            raise ValueError(f"{self.category} items use scoring_type {expected!r}")
        if self.scoring_type == "label" and self.gold not in LABELS:
            raise ValueError(f"label gold must be one of {LABELS}")
        if self.scoring_type == "numeric" and not (isinstance(self.gold, list) and self.gold):
            raise ValueError("numeric items need a list of numeric gold values")
        if self.scoring_type == "free_text" and not (isinstance(self.gold, str) and self.gold):
            raise ValueError("free_text items need a short gold answer")
        if self.scoring_type == "injection" and self.injection is None:
            raise ValueError("injection items need an injection spec")
        if not all(p.isdigit() for p in self.gold_pmids):
            raise ValueError("gold_pmids must be numeric PMIDs")
        return self


def questions_dir(root: Path, version: str = DATASET_VERSION) -> Path:
    return root / "eval" / "questions" / version


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            r.model_dump(exclude_none=True) if isinstance(r, BaseModel) else r,
            ensure_ascii=False,
        )
        for r in rows
    ]
    path.write_text("\n".join(lines) + "\n")


def load_items(directory: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    errors: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        for i, row in enumerate(read_jsonl(path), start=1):
            try:
                items.append(EvalItem.model_validate(row))
            except ValidationError as exc:
                errors.append(f"{path.name}:{i} ({row.get('id')}): {exc.errors()[0]['msg']}")
    if errors:
        raise ValueError("invalid eval items:\n" + "\n".join(errors[:20]))
    return items


# User-provided prompt-injection items (category "injection"); optional.
INJECTION_FILE = Path("eval/questions/injection.jsonl")


def load_injection_items(path: Path, log: Callable[[str], None] = print) -> list[EvalItem]:
    """Load the user-provided injection items, or none (with a notice) if the file is absent."""
    if not path.exists():
        log(f"notice: {path} not found; the prompt-injection category is skipped")
        return []
    items = [EvalItem.model_validate(r) for r in read_jsonl(path)]
    wrong = [i.id for i in items if i.category != "injection"]
    if wrong:
        raise ValueError(f"{path} may only contain injection items: {wrong[:5]}")
    return items


def load_all(root: Path, log: Callable[[str], None] = print) -> list[EvalItem]:
    return load_items(questions_dir(root)) + load_injection_items(root / INJECTION_FILE, log)


def poisoned_passages(items: Sequence[EvalItem]) -> list[PoisonedPassage]:
    return [i.injection.passage for i in items if i.injection is not None]


def validate_items(items: Sequence[EvalItem]) -> dict[str, Any]:
    """Cross-item checks; returns a summary or raises ValueError listing every problem."""
    problems: list[str] = []
    ids = Counter(item.id for item in items)
    problems += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    questions = Counter(item.question.strip().lower() for item in items)
    problems += [f"duplicate question: {q[:60]}" for q, n in questions.items() if n > 1]
    poisoned = poisoned_passages(items)
    poisoned_ids = Counter(p.pmid for p in poisoned)
    problems += [f"poisoned PMID {p} used twice" for p, n in poisoned_ids.items() if n > 1]
    for item in items:
        if item.category != "injection" and any(p in poisoned_ids for p in item.gold_pmids):
            problems.append(f"{item.id}: gold PMID points at a poisoned passage")
        if not item.id.startswith(item.category.replace("_", "")[:4]):
            problems.append(f"{item.id}: id should start with its category prefix")
    if problems:
        raise ValueError("eval set problems:\n" + "\n".join(problems))
    return {
        "items": len(items),
        "by_category": dict(Counter(item.category for item in items)),
        "needs_review": sum(item.needs_review for item in items),
        "poisoned_passages": len(poisoned),
    }


def json_schema() -> dict[str, Any]:
    return EvalItem.model_json_schema()
