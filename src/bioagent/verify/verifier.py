"""Claim verification: atomic claims x cited passages.

1. The answer's sentences are numbered; an LLM (structured output) splits them into
   atomic claims that keep their ``[n]`` citation numbers and marks absence claims.
2. Claims citing computed results (RDKit, sandbox) are checked numerically against the
   tool output. Other claims are scored with NLI against sentence windows of every cited
   passage; the best window counts.
3. Labels: supported (entailment >= threshold for some cited passage), contradicted
   (contradiction >= threshold and not supported), not_supported, computed (numbers match
   the cited tool output) and absence (an explicit "nothing was found" claim that an LLM
   judged consistent with the retrieved evidence; weaker, reported separately).
4. Policy: sentences whose claims all failed are removed ("remove") or marked ("flag");
   sentences with some failed claims are marked. Citations are renumbered afterwards.

Groundedness = (supported + computed) / (supported + computed + contradicted +
not_supported); absence claims are not counted.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import anyio.to_thread
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from bioagent.agents.state import Citation, Evidence
from bioagent.agents.untrusted import UNTRUSTED_RULES, wrap_untrusted
from bioagent.verify.computed import check_numbers
from bioagent.verify.nli import NLIScorer, NLIScores, passage_windows
from bioagent.verify.text import (
    CITE_RE,
    AnswerSentence,
    answer_sentences,
    cited_in,
    renumber_citations,
    rewrite_answer,
)

Label = Literal["supported", "contradicted", "not_supported", "computed", "absence"]
FAILED: frozenset[str] = frozenset({"contradicted", "not_supported"})
GROUNDED: frozenset[str] = frozenset({"supported", "computed"})
Policy = Literal["remove", "flag"]
MAX_CLAIMS = 60
MAX_ABSENCE_TITLES = 60


class ExtractedClaim(BaseModel):
    sentence: int = Field(description="number of the sentence the claim comes from")
    text: str = Field(description="one self-contained factual claim, without [n] markers")
    citations: list[int] = Field(
        default_factory=list, description="the [n] citation numbers that support this claim"
    )
    kind: Literal["factual", "absence"] = Field(
        default="factual",
        description="absence = the claim says something was NOT found / does not exist",
    )


class ClaimList(BaseModel):
    claims: list[ExtractedClaim]


class AbsenceCheck(BaseModel):
    consistent: bool = Field(
        description="true if none of the listed evidence contradicts the absence claim"
    )
    reason: str = Field(default="", description="one sentence")


CLAIM_PROMPT = f"""You split an answer into atomic, checkable claims.
For each numbered sentence, write one claim per distinct fact (a drug's effect, a number, a
study finding). Make each claim self-contained: replace pronouns with names. Copy the [n]
citation numbers of the sentence that support each claim into its citations list. Mark
claims that state something was NOT found or does not exist as kind="absence". Skip
sentences with no factual content (transitions, summaries of what follows). Do not add
facts that are not in the sentence.

{UNTRUSTED_RULES}"""

ABSENCE_PROMPT = f"""You check an absence claim (a statement that something was not found
or does not exist) against the titles of all evidence retrieved for the answer. Answer
consistent=false only if some listed item clearly reports the thing claimed to be absent.

{UNTRUSTED_RULES}"""

# (schema, messages) -> parsed model or None; provided by the agent graph (uses the
# synthesizer's model with retries and records the call in the run stats).
StructuredCall = Callable[[type[BaseModel], list[BaseMessage]], Awaitable[BaseModel | None]]


@dataclass
class VerifyConfig:
    entail_threshold: float = 0.5
    contradiction_threshold: float = 0.5
    policy: Policy = "remove"
    paper_level: bool = True  # a citation covers every retrieved passage of that PMID
    windows_per_citation: int = 8  # NLI only on the windows with most word overlap
    windowed: bool = True  # False: score whole passages (classifier / LLM backends)


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2}


def top_windows(windows: Sequence[str], claim: str, k: int) -> list[str]:
    """The ``k`` windows sharing most words with the claim (original order kept)."""
    unique = list(dict.fromkeys(windows))
    if len(unique) <= k:
        return unique
    target = _words(claim)
    ranked = sorted(range(len(unique)), key=lambda i: -len(target & _words(unique[i])))
    keep = sorted(ranked[:k])
    return [unique[i] for i in keep]


@dataclass
class Claim:
    n: int
    sentence: int
    text: str
    citations: list[int]
    kind: str = "factual"
    label: str = "not_supported"
    check: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "claim": self.text,
            "label": self.label,
            "check": self.check,
            "citations": self.citations,
            "sentence": self.sentence,
            "evidence": self.evidence,
        }


def _strip_markers(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", " ".join(CITE_RE.sub("", text).split()))


def fallback_claims(sentences: Sequence[AnswerSentence]) -> list[ExtractedClaim]:
    """One claim per sentence (used when the LLM split fails)."""
    return [
        ExtractedClaim(sentence=s.id, text=_strip_markers(s.text), citations=cited_in(s.text))
        for s in sentences
        if len(_strip_markers(s.text)) > 20
    ]


def normalise_claims(
    extracted: Sequence[ExtractedClaim], sentences: Sequence[AnswerSentence], valid: set[int]
) -> list[Claim]:
    by_id = {s.id: s for s in sentences}
    claims: list[Claim] = []
    for c in extracted:
        sentence = by_id.get(c.sentence)
        if sentence is None or not c.text.strip():
            continue
        in_sentence = cited_in(sentence.text)
        cites = [n for n in c.citations if n in in_sentence] or in_sentence
        claims.append(
            Claim(
                n=len(claims) + 1,
                sentence=c.sentence,
                text=_strip_markers(c.text),
                citations=[n for n in cites if n in valid],
                kind=c.kind,
            )
        )
        if len(claims) >= MAX_CLAIMS:
            break
    return claims


def _ref(e: Evidence) -> str:
    return f"PMID {e['pmid']}" if e["pmid"] else f"{e['source']}:{e['title'] or e['key']}"


class ClaimVerifier:
    def __init__(self, nli: NLIScorer, config: VerifyConfig | None = None) -> None:
        self.nli = nli
        self.config = config or VerifyConfig()

    async def extract(
        self, sentences: Sequence[AnswerSentence], call: StructuredCall | None
    ) -> tuple[list[ExtractedClaim], str]:
        if not sentences:
            return [], "none"
        if call is not None:
            numbered = "\n".join(f"{s.id}. {s.text}" for s in sentences)
            messages: list[BaseMessage] = [
                SystemMessage(CLAIM_PROMPT),
                HumanMessage("Answer sentences:\n" + wrap_untrusted(numbered, source="answer")),
            ]
            parsed = await call(ClaimList, messages)
            if isinstance(parsed, ClaimList) and parsed.claims:
                return parsed.claims, "llm"
        return fallback_claims(sentences), "sentences"

    async def verify(
        self,
        answer: str,
        citations: Sequence[Citation],
        evidence: Mapping[str, Evidence],
        *,
        call: StructuredCall | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        sentences = answer_sentences(answer)
        by_n = {c["n"]: c for c in citations}
        extracted, method = await self.extract(sentences, call)
        claims = normalise_claims(extracted, sentences, set(by_n))

        cited: dict[int, Evidence] = {
            n: evidence[c["key"]] for n, c in by_n.items() if c["key"] in evidence
        }
        # A citation names a paper: check every retrieved passage of that paper (abstract
        # chunks, graph relation items), not only the one chunk shown to the synthesizer.
        by_pmid: dict[str, list[str]] = {}
        for e in evidence.values():
            if e["pmid"] and not e["computed"] and e["text"] not in by_pmid.get(e["pmid"], []):
                by_pmid.setdefault(e["pmid"], []).append(e["text"])

        def premises(e: Evidence) -> list[str]:
            if self.config.paper_level and e["pmid"]:
                return by_pmid.get(e["pmid"], [e["text"]])
            return [e["text"]]

        pairs: list[tuple[str, str]] = []
        slots: list[tuple[Claim, int, str]] = []  # (claim, citation n, window)
        pending_nli: list[Claim] = []
        for claim in claims:
            if claim.kind == "absence":
                continue
            items = {n: cited[n] for n in claim.citations if n in cited}
            if not items:
                claim.label, claim.check = "not_supported", "no resolvable citation"
                continue
            computed = {n: e for n, e in items.items() if e["computed"]}
            papers = {n: e for n, e in items.items() if not e["computed"]}
            if computed:
                result = check_numbers(claim.text, [e["text"] for e in computed.values()])
                claim.evidence = [
                    {"cite": n, "ref": _ref(e), "source": e["source"]} for n, e in computed.items()
                ]
                if result.ok:
                    claim.label, claim.check = "computed", result.detail
                    continue
                if not papers:  # never run NLI on tool JSON
                    claim.label, claim.check = "not_supported", result.detail
                    continue
            targets = papers
            pending_nli.append(claim)
            for n, e in targets.items():
                texts = premises(e)
                windows = (
                    [w for text in texts for w in passage_windows(text)]
                    if self.config.windowed
                    else texts
                )
                for window in top_windows(windows, claim.text, self.config.windows_per_citation):
                    pairs.append((window, claim.text))
                    slots.append((claim, n, window))

        unique = list(dict.fromkeys(pairs))
        unique_scores: list[NLIScores] = (
            await anyio.to_thread.run_sync(lambda: self.nli.score(unique)) if unique else []
        )
        by_pair = dict(zip(unique, unique_scores, strict=True))
        scores = [by_pair[pair] for pair in pairs]
        # per (claim, citation): best entailment window and highest contradiction
        best: dict[tuple[int, int], tuple[float, str]] = {}
        contra: dict[tuple[int, int], float] = {}
        for (claim, n, window), sc in zip(slots, scores, strict=True):
            key = (claim.n, n)
            if key not in best or sc.entailment > best[key][0]:
                best[key] = (sc.entailment, window)
            contra[key] = max(contra.get(key, 0.0), sc.contradiction)
        cfg = self.config
        for claim in pending_nli:
            rows: list[dict[str, Any]] = []
            for n in claim.citations:
                if (claim.n, n) not in best:
                    continue
                entail, window = best[(claim.n, n)]
                e = cited[n]
                rows.append(
                    {
                        "cite": n,
                        "ref": _ref(e),
                        "source": e["source"],
                        "entailment": round(entail, 3),
                        "contradiction": round(contra[(claim.n, n)], 3),
                        "window": window[:240],
                    }
                )
            claim.evidence = [*claim.evidence, *rows]
            top_e = max(rows, key=lambda r: r["entailment"], default=None)
            top_c = max(rows, key=lambda r: r["contradiction"], default=None)
            if top_e and top_e["entailment"] >= cfg.entail_threshold:
                claim.label = "supported"
                claim.check = f"NLI entailment {top_e['entailment']:.2f} vs [{top_e['cite']}]"
            elif top_c and top_c["contradiction"] >= cfg.contradiction_threshold:
                claim.label = "contradicted"
                claim.check = f"NLI contradiction {top_c['contradiction']:.2f} vs [{top_c['cite']}]"
            else:
                best_e = top_e["entailment"] if top_e else 0.0
                claim.label = "not_supported"
                claim.check = f"NLI max entailment {best_e:.2f} < {cfg.entail_threshold:.2f}"
        await self._check_absence(claims, evidence, call)
        nli_seconds = time.monotonic() - started

        new_answer, kept, removed, flagged = self.apply_policy(answer, claims, citations)
        counts = {label: sum(1 for c in claims if c.label == label) for label in (
            "supported", "computed", "contradicted", "not_supported", "absence")}  # fmt: skip
        checkable = sum(counts[k] for k in ("supported", "computed", "contradicted",
                                            "not_supported"))  # fmt: skip
        grounded = counts["supported"] + counts["computed"]
        removed_claims = [c.row() for c in claims if c.sentence in removed and c.label in FAILED]
        return {
            "answer": new_answer,
            "citations": kept,
            "report": {
                "claims": [c.row() for c in claims],
                "counts": counts,
                "groundedness": round(grounded / checkable, 3) if checkable else None,
                "checkable_claims": checkable,
                "removed_claims": removed_claims,
                "removed_sentences": sorted(removed),
                "flagged_sentences": sorted(flagged),
                "policy": self.config.policy,
                "claim_split": method,
                "nli_model": self.nli.name,
                "nli_pairs": len(unique),
                "entail_threshold": cfg.entail_threshold,
                "contradiction_threshold": cfg.contradiction_threshold,
                "seconds": round(nli_seconds, 2),
            },
        }

    async def _check_absence(
        self, claims: list[Claim], evidence: Mapping[str, Evidence], call: StructuredCall | None
    ) -> None:
        absence = [c for c in claims if c.kind == "absence"]
        if not absence:
            return
        titles = "\n".join(
            f"- {e['title'] or e['text'][:120]}"
            for e in list(evidence.values())[:MAX_ABSENCE_TITLES]
        )
        for claim in absence:
            if call is None:
                claim.label, claim.check = "absence", "absence claim (not checked: no LLM)"
                continue
            parsed = await call(
                AbsenceCheck,
                [
                    SystemMessage(ABSENCE_PROMPT),
                    HumanMessage(
                        f"Absence claim: {claim.text}\n\nRetrieved evidence:\n"
                        + wrap_untrusted(titles, source="evidence_titles")
                    ),
                ],
            )
            if isinstance(parsed, AbsenceCheck) and not parsed.consistent:
                claim.label = "contradicted"
                claim.check = f"absence contradicted by retrieved evidence: {parsed.reason[:160]}"
            else:
                reason = parsed.reason[:160] if isinstance(parsed, AbsenceCheck) else "unchecked"
                claim.label, claim.check = "absence", f"supported by absence (LLM): {reason}"

    def apply_policy(
        self, answer: str, claims: Sequence[Claim], citations: Sequence[Citation]
    ) -> tuple[str, list[Citation], set[int], set[int]]:
        by_sentence: dict[int, list[Claim]] = {}
        for c in claims:
            by_sentence.setdefault(c.sentence, []).append(c)
        removed: set[int] = set()
        flagged: set[int] = set()
        for sid, items in by_sentence.items():
            failed = [c for c in items if c.label in FAILED]
            if not failed:
                continue
            if len(failed) == len(items) and self.config.policy == "remove":
                removed.add(sid)
            else:
                flagged.add(sid)
        if not removed and not flagged:
            return answer, list(citations), removed, flagged
        rewritten = rewrite_answer(answer, removed, flagged)
        text, kept = renumber_citations(rewritten, [dict(c) for c in citations])
        return text, [Citation(**c) for c in kept], removed, flagged  # type: ignore[typeddict-item]
