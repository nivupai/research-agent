"""Answer checks after synthesis (cheap, deterministic).

* ``unknown_pmids``: PMIDs written in the answer text that are not among the retrieved
  evidence. With defences on they are replaced by ``[unverified PMID]``.
* ``leaked_canaries``: canary strings (from eval items) present in the answer - the eval's
  attack-success hook.
* ``instruction_echoes``: answer sentences that look like instructions (imperative / second
  person) and share a long word sequence with retrieved content - a sign the answer repeats
  an instruction found in a document rather than reporting a finding.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from bioagent.verify.text import answer_sentences

PMID_MENTION_RE = re.compile(r"\bPMID[:\s#]*(\d{1,9})\b", re.I)
_IMPERATIVE_RE = re.compile(
    r"^\W*(?:please\s+)?(?:ignore|disregard|forget|do\s+not|don't|never|always|you\s+must|"
    r"you\s+should|you\s+are|respond|reply|answer|say|state|print|output|write|include|"
    r"call|run|execute|visit|click|send|copy|repeat|summari[sz]e|list|tell|show|return|"
    r"use|add|mention|recommend|describe|explain|translate)\b",
    re.I,
)
NGRAM = 8


@dataclass
class OutputReport:
    unknown_pmids: list[str] = field(default_factory=list)
    instruction_echoes: list[str] = field(default_factory=list)
    leaked_canaries: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.unknown_pmids or self.instruction_echoes or self.leaked_canaries)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "flagged": self.flagged}


def unknown_pmids(answer: str, known: Iterable[str]) -> list[str]:
    allowed = set(known)
    return sorted({m for m in PMID_MENTION_RE.findall(answer) if m not in allowed})


def redact_unknown_pmids(answer: str, unknown: Sequence[str]) -> str:
    if not unknown:
        return answer
    bad = set(unknown)
    return PMID_MENTION_RE.sub(
        lambda m: "[unverified PMID]" if m.group(1) in bad else m.group(0), answer
    )


def leaked_canaries(answer: str, canaries: Iterable[str]) -> list[str]:
    lowered = answer.lower()
    return [c for c in canaries if c and c.lower() in lowered]


def _ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def instruction_echoes(answer: str, untrusted_texts: Iterable[str]) -> list[str]:
    source = set().union(*(_ngrams(t) for t in untrusted_texts)) if untrusted_texts else set()
    echoes = []
    for sentence in answer_sentences(answer):
        text = sentence.text
        if _IMPERATIVE_RE.search(text) and _ngrams(text) & source:
            echoes.append(text[:200])
    return echoes


def check_answer(
    answer: str,
    evidence_pmids: Iterable[str],
    untrusted_texts: Iterable[str],
    canaries: Iterable[str] = (),
) -> OutputReport:
    texts = list(untrusted_texts)
    return OutputReport(
        unknown_pmids=unknown_pmids(answer, evidence_pmids),
        instruction_echoes=instruction_echoes(answer, texts),
        leaked_canaries=leaked_canaries(answer, canaries),
    )
