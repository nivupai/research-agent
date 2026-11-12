"""Graph state, plan models and reducers.

State values are plain JSON-compatible dicts/lists (pydantic models are dumped) so the
SQLite checkpointer can store them without custom serialisers.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

from bioagent.llm.stats import LLMCallRecord, ToolCallRecord

TaskKind = Literal["literature", "chemistry", "analysis"]
TASK_KINDS: tuple[TaskKind, ...] = ("literature", "chemistry", "analysis")
MAX_TASKS = 6


class Task(BaseModel):
    id: str = Field(description="short id such as t1")
    kind: TaskKind = Field(
        description="literature = find papers/evidence; chemistry = RDKit properties, "
        "drug-likeness, similarity; analysis = run Python on numbers from other tasks"
    )
    description: str = Field(description="what this task must find or compute")
    depends_on: list[str] = Field(
        default_factory=list, description="ids of tasks whose results this task needs"
    )
    drug_target: str | None = Field(
        default=None,
        description="literature tasks that ask which drugs act on a gene, complex or "
        "pathway: that target's name (e.g. AMPK, MGMT); otherwise null",
    )
    approved_only: bool = Field(
        default=False, description="with drug_target: only approved drugs are wanted"
    )


class Plan(BaseModel):
    tasks: list[Task] = Field(description=f"1 to {MAX_TASKS} tasks forming a DAG")
    rationale: str = Field(default="", description="one or two sentences")

    @model_validator(mode="after")
    def _check_dag(self) -> Plan:
        if not 1 <= len(self.tasks) <= MAX_TASKS:
            raise ValueError(f"plan must have 1-{MAX_TASKS} tasks")
        ids = [t.id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be unique")
        known = set(ids)
        for t in self.tasks:
            missing = [d for d in t.depends_on if d not in known]
            if missing:
                raise ValueError(f"task {t.id} depends on unknown tasks {missing}")
            if t.id in t.depends_on:
                raise ValueError(f"task {t.id} depends on itself")
        # cycle check (Kahn)
        indeg = {t.id: len(set(t.depends_on)) for t in self.tasks}
        children: dict[str, list[str]] = {i: [] for i in ids}
        for t in self.tasks:
            for d in set(t.depends_on):
                children[d].append(t.id)
        ready = [i for i, n in indeg.items() if n == 0]
        seen = 0
        while ready:
            node = ready.pop()
            seen += 1
            for c in children[node]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
        if seen != len(ids):
            raise ValueError("task dependencies contain a cycle")
        return self


class Evidence(TypedDict):
    key: str  # passage_id for papers ("pmid:chunk"), or "computed:<task>:<n>"
    pmid: str | None
    passage_id: str | None
    title: str | None
    year: int | None
    journal: str | None
    doi: str | None
    text: str
    source: str  # corpus | pubmed | europepmc | rdkit | pubchem | sandbox
    task_id: str
    computed: bool
    untrusted: bool


class TaskResult(TypedDict):
    task_id: str
    kind: str
    status: Literal["done", "failed"]
    summary: str
    evidence_keys: list[str]
    steps: int
    error: str | None


class Citation(TypedDict):
    n: int
    key: str
    pmid: str | None
    title: str | None
    year: int | None
    url: str | None
    computed: bool
    source: str


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {**left, **right}


class AgentState(TypedDict, total=False):
    run_id: str
    question: str
    plan: list[dict[str, Any]]
    plan_rationale: str
    plan_notes: list[str]
    results: Annotated[dict[str, TaskResult], merge_dicts]
    evidence: Annotated[dict[str, Evidence], merge_dicts]
    answer: str
    citations: list[Citation]
    verification: dict[str, Any]
    llm_calls: Annotated[list[LLMCallRecord], operator.add]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    errors: Annotated[list[str], operator.add]


class SpecialistInput(TypedDict):
    """Payload sent to a specialist via the Send API."""

    question: str
    task: dict[str, Any]
    dependency_notes: list[dict[str, str]]
