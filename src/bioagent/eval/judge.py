"""LLM judge (structured output) with a persistent cache.

Kinds: ``answer`` (correct / partially / incorrect vs. gold answer and key facts),
``abstain`` (does the answer say the evidence is insufficient?), ``premise`` (does it reject
or correct the false premise?) and ``label`` (fallback yes/no/maybe extraction). Calls are
cached by (question, answer hash, prompt version, kind); bump ``JUDGE_VERSION`` whenever a
prompt changes so stale verdicts are not reused.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from bioagent.agents.untrusted import UNTRUSTED_RULES, wrap_untrusted
from bioagent.eval.cache import JSONCache, judge_key
from bioagent.eval.dataset import EvalItem
from bioagent.llm.providers import ModelSpec, ainvoke_with_stats
from bioagent.llm.stats import RunStatsCollector

JUDGE_VERSION = "judge-v1"


class AnswerVerdict(BaseModel):
    verdict: Literal["correct", "partially", "incorrect"] = Field(
        description="correct: matches the gold answer and its key facts without errors; "
        "partially: some key facts right but incomplete, hedged or with a minor error; "
        "incorrect: wrong, contradicts the gold, or does not answer"
    )
    rationale: str = Field(description="one or two sentences")


class YesNoVerdict(BaseModel):
    yes: bool
    rationale: str = Field(description="one sentence")


class LabelVerdict(BaseModel):
    label: Literal["yes", "no", "maybe", "none"]


RUBRICS: dict[str, str] = {
    "answer": (
        "You grade a biomedical answer against a gold answer and key facts. Judge only "
        "factual agreement with the gold; ignore style, length and citations. Extra correct "
        "detail is fine; a claim contradicting the gold makes it incorrect."
    ),
    "abstain": (
        "The question cannot be answered from the available evidence. Answer yes=true if "
        "the answer clearly says the evidence is insufficient, not found or unknown (it may "
        "still mention related findings); yes=false if it presents a definite answer as fact."
    ),
    "premise": (
        "The question contains a false premise (explained in the notes). Answer yes=true if "
        "the answer rejects or corrects that premise; yes=false if it accepts the premise "
        "and explains it."
    ),
    "label": (
        "Which overall answer does the text give to the yes/no question: yes, no or maybe "
        "(uncertain/mixed)? Use 'none' if it gives none."
    ),
}
SCHEMAS: dict[str, type[BaseModel]] = {
    "answer": AnswerVerdict,
    "abstain": YesNoVerdict,
    "premise": YesNoVerdict,
    "label": LabelVerdict,
}


def judge_messages(kind: str, item: EvalItem, answer: str) -> list[BaseMessage]:
    parts = [f"Question: {item.question}"]
    if kind == "answer":
        parts.append(f"Gold answer: {item.gold}")
        if item.key_facts:
            parts.append("Key facts:\n" + "\n".join(f"- {f}" for f in item.key_facts))
    if kind == "premise" and item.notes:
        parts.append(f"Notes on the false premise: {item.notes}")
    parts.append("Answer to grade:\n" + wrap_untrusted(answer[:6000], source="agent_answer"))
    return [
        SystemMessage(f"{RUBRICS[kind]}\n\n{UNTRUSTED_RULES}"),
        HumanMessage("\n\n".join(parts)),
    ]


class LLMJudge:
    def __init__(
        self, model: BaseChatModel, spec: ModelSpec, cache: JSONCache, max_retries: int = 5
    ) -> None:
        self.model = model
        self.spec = spec
        self.cache = cache
        self.max_retries = max_retries
        self.stats = RunStatsCollector()

    async def __call__(self, kind: str, item: EvalItem, answer: str) -> dict[str, Any]:
        key = judge_key(item.question, answer, f"{JUDGE_VERSION}:{self.spec}", kind)
        cached = self.cache.get(key)
        if cached is not None:
            return {**cached, "cached": True}
        structured = self.model.with_structured_output(SCHEMAS[kind], include_raw=True)
        out = await ainvoke_with_stats(
            structured,
            judge_messages(kind, item, answer),
            spec=self.spec,
            role=f"judge:{kind}",
            stats=self.stats,
            max_retries=self.max_retries,
        )
        parsed = out.get("parsed") if isinstance(out, dict) else out
        if not isinstance(parsed, BaseModel):
            result: dict[str, Any] = {"error": "judge output could not be parsed"}
        else:
            result = parsed.model_dump()
            self.cache.set(key, result)
        result["judge_model"] = str(self.spec)
        result["judge_version"] = JUDGE_VERSION
        return result
