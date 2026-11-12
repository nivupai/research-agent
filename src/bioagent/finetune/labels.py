"""Labels for claim/passage pairs: the Gemini teacher (batched, cached) and the DeBERTa NLI
baseline. Both map to SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from bioagent.agents.untrusted import UNTRUSTED_RULES, wrap_untrusted
from bioagent.eval.cache import JSONCache, stable_hash
from bioagent.finetune.data import LABELS, Pair
from bioagent.llm.providers import ModelSpec, QuotaExhaustedError, ainvoke_with_stats
from bioagent.llm.stats import RunStatsCollector
from bioagent.verify.nli import NLIScorer, passage_windows

TEACHER_VERSION = "teacher-v1"
BATCH = 20

TEACHER_PROMPT = f"""You label claim/passage pairs for training a biomedical claim verifier.
For each numbered pair decide whether the PASSAGE alone:
- SUPPORTED: states or directly entails the claim (paraphrase is fine);
- CONTRADICTED: states something incompatible with the claim (opposite direction, a
  different number, the claim says an effect exists and the passage says it does not, ...);
- NOT_ENOUGH_INFO: neither (the passage is about something else, or is silent on the claim).
Use only the passage, not background knowledge. Return one label per pair id.

{UNTRUSTED_RULES}"""


class PairLabel(BaseModel):
    id: str = Field(description="the pair id")
    label: Literal["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]


class PairLabels(BaseModel):
    labels: list[PairLabel]


# Inference-time LLM verification uses its own cache namespace, so it never reuses the
# teacher's training labels for the same pair.
LLM_VERIFIER_VERSION = "llm-verifier-v1"


def teacher_key(pair: Pair, spec: ModelSpec, version: str = TEACHER_VERSION) -> str:
    return stable_hash({"v": version, "m": str(spec), "id": pair.id}, 32)


def teacher_messages(pairs: Sequence[Pair]) -> list[BaseMessage]:
    blocks = [
        f"### Pair {p.id}\nCLAIM: {p.claim}\n"
        + wrap_untrusted(p.passage, source="passage", ref=p.id)
        for p in pairs
    ]
    return [SystemMessage(TEACHER_PROMPT), HumanMessage("\n\n".join(blocks))]


def parse_labels(parsed: Any, ids: Sequence[str]) -> dict[str, str]:
    """Labels for the requested ids only; unknown ids, duplicates and bad labels are dropped."""
    if not isinstance(parsed, PairLabels):
        return {}
    wanted = set(ids)
    out: dict[str, str] = {}
    for item in parsed.labels:
        if item.id in wanted and item.id not in out and item.label in LABELS:
            out[item.id] = item.label
    return out


async def label_with_teacher(
    pairs: Sequence[Pair],
    model: BaseChatModel,
    spec: ModelSpec,
    cache: JSONCache,
    *,
    batch: int = BATCH,
    version: str = TEACHER_VERSION,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Label every uncached pair; stops cleanly (returns) when the quota runs out."""
    todo = [p for p in pairs if cache.get(teacher_key(p, spec, version)) is None]
    structured = model.with_structured_output(PairLabels, include_raw=True)
    stats = RunStatsCollector()
    labelled = calls = 0
    stopped = None
    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        t0 = time.monotonic()
        try:
            out = await ainvoke_with_stats(
                structured, teacher_messages(chunk), spec=spec, role="teacher", stats=stats
            )
        except QuotaExhaustedError as exc:
            stopped = str(exc)[:200]
            break
        calls += 1
        parsed = out.get("parsed") if isinstance(out, dict) else out
        got = parse_labels(parsed, [p.id for p in chunk])
        seconds = time.monotonic() - t0
        for p in chunk:
            if p.id in got:
                cache.set(
                    teacher_key(p, spec, version),
                    {"label": got[p.id], "batch_seconds": seconds, "batch_size": len(chunk)},
                )
                labelled += 1
        if calls % 10 == 0:
            log(f"  teacher: {labelled}/{len(todo)} labelled in {calls} calls")
    records = stats.update().get("llm_calls", [])
    return {
        "requested": len(todo),
        "labelled": labelled,
        "calls": calls,
        "stopped": stopped,
        "input_tokens": sum(r["input_tokens"] for r in records),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "cost_usd": sum(r["cost_usd"] or 0 for r in records),
        "latency_s": sum(r["latency_ms"] for r in records) / 1000,
    }


def nli_label(entail: float, contra: float, threshold: float = 0.5) -> str:
    if entail >= threshold and entail >= contra:
        return "SUPPORTED"
    if contra >= threshold:
        return "CONTRADICTED"
    return "NOT_ENOUGH_INFO"


def nli_predict(
    pairs: Sequence[Pair], scorer: NLIScorer, *, windows: int = 6, threshold: float = 0.5
) -> dict[str, dict[str, Any]]:
    """Max-over-windows NLI per pair (as in bioagent.verify), with timing per pair."""
    from bioagent.verify.verifier import top_windows

    out: dict[str, dict[str, Any]] = {}
    for p in pairs:
        t0 = time.monotonic()
        ws = top_windows(passage_windows(p.passage), p.claim, windows) or [p.passage]
        scores = scorer.score([(w, p.claim) for w in ws])
        entail = max(s.entailment for s in scores)
        contra = max(s.contradiction for s in scores)
        out[p.id] = {
            "label": nli_label(entail, contra, threshold),
            "entailment": round(entail, 4),
            "contradiction": round(contra, 4),
            "seconds": time.monotonic() - t0,
        }
    return out
