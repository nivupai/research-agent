"""Claim/passage pairs for the small verifier.

Sources (all local):

* ``run``: claims (answer sentences with citations) from saved agent runs and eval results,
  paired with each cited passage. The pre-verification ``draft_answer`` is used so claims the
  verifier removed are included.
* ``positive``: a results-like sentence from an abstract paired with its own passage.
* ``cross``: a sentence from paper A paired with a same-topic passage of paper B (BM25), a
  hard "not enough info" candidate.
* ``negation`` / ``direction`` / ``swap`` / ``number``: rule-based perturbations of a
  sentence paired with its original passage (likely contradicted or unsupported).

Labels are not assumed from the construction: a teacher LLM labels every pair (SUPPORTED,
CONTRADICTED, NOT_ENOUGH_INFO). Splits are assigned per PMID (hash of the PMID), and cross
pairs only combine papers from the same split, so no paper appears in two splits.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from bioagent.ingest.passages import split_sentences
from bioagent.verify.text import CITE_RE, answer_sentences, cited_in

LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")
Label = Literal["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
Split = Literal["train", "val", "test"]
MAX_PASSAGE_CHARS = 1800


def split_for(pmid: str, test: int = 10, val: int = 10) -> Split:
    """Deterministic split by PMID (percent buckets of a SHA-1 hash)."""
    bucket = int(hashlib.sha1(f"verifier-split:{pmid}".encode()).hexdigest(), 16) % 100
    if bucket < test:
        return "test"
    if bucket < test + val:
        return "val"
    return "train"


@dataclass
class Pair:
    claim: str
    passage: str
    pmid: str  # passage paper
    kind: str
    claim_pmid: str | None = None  # paper the claim text came from (synthetic pairs)
    detail: str | None = None
    split: str = ""
    id: str = ""
    labels: dict[str, Any] = field(default_factory=dict)

    def finalise(self) -> Pair:
        self.claim = " ".join(self.claim.split())
        self.passage = self.passage.strip()[:MAX_PASSAGE_CHARS]
        self.split = split_for(self.pmid)
        self.id = pair_id(self.claim, self.passage)
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def pair_id(claim: str, passage: str) -> str:
    norm = " ".join(claim.lower().split()) + "\x00" + " ".join(passage.lower().split())[:600]
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


# --- claims from runs -----------------------------------------------------------------------
def _strip(text: str) -> str:
    return re.sub(r"\s+([.,;:])", r"\1", " ".join(CITE_RE.sub("", text).split()))


def claims_from_record(record: Mapping[str, Any]) -> Iterator[Pair]:
    answer = str(record.get("draft_answer") or record.get("answer") or "")
    by_n = {int(c["n"]): c for c in record.get("citations", [])}
    evidence: Mapping[str, Any] = record.get("cited_evidence", {})
    for sentence in answer_sentences(answer):
        claim = _strip(sentence.text)
        if len(claim) < 40:
            continue
        for n in cited_in(sentence.text):
            cite = by_n.get(n)
            ev = evidence.get(cite["key"]) if cite else None
            if not ev or ev.get("computed") or not ev.get("pmid"):
                continue
            yield Pair(claim, str(ev["text"]), str(ev["pmid"]), "run").finalise()


def run_records(results: Path) -> Iterator[Mapping[str, Any]]:
    """Agent run records (results/runs/*.json) and eval item records (results/eval/**)."""
    for path in sorted((results / "runs").glob("*.json")):
        if not path.name.endswith(".meta.json"):
            yield json.loads(path.read_text())
    for path in sorted((results / "eval").glob("*/*/items/*.json")):
        data = json.loads(path.read_text())
        if data.get("record"):
            yield data["record"]


# --- synthetic pairs ---------------------------------------------------------------------------
_RESULT_HINT = re.compile(
    r"\b(increas|decreas|reduc|inhibit|activat|enhanc|improv|suppress|induc|associat|"
    r"higher|lower|significant|correlat|prolong|block|promot|attenuat)\w*",
    re.I,
)

DIRECTION_SWAPS: list[tuple[str, str]] = [
    ("increased", "decreased"),
    ("increases", "decreases"),
    ("increase", "decrease"),
    ("reduced", "increased"),
    ("reduces", "increases"),
    ("inhibited", "activated"),
    ("inhibits", "activates"),
    ("inhibit", "activate"),
    ("activated", "inhibited"),
    ("activates", "inhibits"),
    ("enhanced", "impaired"),
    ("enhances", "impairs"),
    ("improved", "worsened"),
    ("higher", "lower"),
    ("lower", "higher"),
    ("suppressed", "promoted"),
    ("promoted", "suppressed"),
    ("prolonged", "shortened"),
    ("upregulated", "downregulated"),
    ("downregulated", "upregulated"),
]
NEGATIONS: list[tuple[str, str]] = [
    (" was associated with ", " was not associated with "),
    (" were associated with ", " were not associated with "),
    (" significantly ", " did not significantly "),
    (" is ", " is not "),
    (" are ", " are not "),
    (" was ", " was not "),
    (" were ", " were not "),
    (" can ", " cannot "),
]
ENTITIES: list[str] = [
    "metformin",
    "temozolomide",
    "rapamycin",
    "everolimus",
    "aspirin",
    "phenformin",
    "berberine",
    "resveratrol",
    "simvastatin",
    "bevacizumab",
    "AMPK",
    "mTOR",
    "AKT",
    "PI3K",
    "MGMT",
    "EGFR",
    "p53",
    "glioblastoma",
    "glioma",
    "breast cancer",
    "prostate cancer",
    "diabetes",
    "insulin",
    "glucose",
]
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s?%| ?mg| ?months| ?patients| ?fold)?")


def _swap_word(text: str, pairs: Sequence[tuple[str, str]]) -> tuple[str, str] | None:
    for old, new in pairs:
        pattern = re.compile(
            rf"\b{re.escape(old.strip())}\b" if old.strip() == old else re.escape(old), re.I
        )
        if pattern.search(text):
            return pattern.sub(new.strip() if old.strip() == old else new, text, count=1), (
                f"{old.strip()} -> {new.strip()}"
            )
    return None


def perturb(claim: str, kind: str, rng: random.Random) -> tuple[str, str] | None:
    """(perturbed claim, description) or None if the rule does not apply."""
    if kind == "direction":
        return _swap_word(claim, DIRECTION_SWAPS)
    if kind == "negation":
        return _swap_word(claim, NEGATIONS)
    if kind == "swap":
        present = [e for e in ENTITIES if re.search(rf"\b{re.escape(e)}\b", claim, re.I)]
        if not present:
            return None
        old = present[0]
        new = rng.choice([e for e in ENTITIES if e.lower() != old.lower() and e not in present])
        return re.sub(rf"\b{re.escape(old)}\b", new, claim, count=1, flags=re.I), f"{old} -> {new}"
    if kind == "number":
        match = _NUMBER_RE.search(claim)
        if not match:
            return None
        value = float(match.group(1))
        changed = value * 2 + 3 if value < 10 else round(value * rng.choice((0.4, 1.8)), 1)
        text = f"{changed:g}" if "." not in match.group(1) else f"{changed:.1f}"
        return claim[: match.start(1)] + text + claim[match.end(1) :], f"{value:g} -> {text}"
    raise ValueError(kind)


def candidate_sentences(abstract: str) -> list[str]:
    sentences = split_sentences(abstract)
    return [
        s
        for s in sentences[1:]
        if 60 <= len(s) <= 260
        and _RESULT_HINT.search(s)
        and not s.lower().startswith(("in this", "we "))
    ]


@dataclass(frozen=True)
class Paper:
    pmid: str
    title: str
    abstract: str

    @property
    def passage(self) -> str:
        return f"{self.title}\n{self.abstract}".strip()


def synthetic_pairs(
    papers: Sequence[Paper],
    neighbours: Mapping[str, Sequence[str]],
    *,
    n_positive: int,
    n_cross: int,
    n_perturbed: int,
    seed: int = 13,
) -> list[Pair]:
    """``neighbours[pmid]``: same-topic PMIDs (e.g. BM25 on the title), best first."""
    rng = random.Random(seed)
    by_pmid = {p.pmid: p for p in papers}
    pool = [(p, s) for p in papers for s in candidate_sentences(p.abstract)[:2]]
    rng.shuffle(pool)
    pairs: list[Pair] = []
    for paper, sentence in pool[:n_positive]:
        pairs.append(Pair(sentence, paper.passage, paper.pmid, "positive", paper.pmid).finalise())
    made = 0
    for paper, sentence in pool[n_positive:]:
        if made >= n_cross:
            break
        split = split_for(paper.pmid)
        others = [
            o
            for o in neighbours.get(paper.pmid, [])
            if o != paper.pmid and o in by_pmid and split_for(o) == split
        ]
        if not others:
            continue
        other = by_pmid[others[0]]
        pairs.append(
            Pair(
                sentence,
                other.passage,
                other.pmid,
                "cross",
                paper.pmid,
                f"claim from PMID {paper.pmid}",
            ).finalise()
        )
        made += 1
    kinds = ["direction", "negation", "swap", "number"]
    made = 0
    for i, (paper, sentence) in enumerate(pool[n_positive + n_cross :]):
        if made >= n_perturbed:
            break
        kind = kinds[i % len(kinds)]
        result = perturb(sentence, kind, rng)
        if result is None or result[0] == sentence:
            continue
        claim, detail = result
        pairs.append(Pair(claim, paper.passage, paper.pmid, kind, paper.pmid, detail).finalise())
        made += 1
    return pairs


def dedupe(pairs: Iterable[Pair]) -> list[Pair]:
    seen: dict[str, Pair] = {}
    for p in pairs:
        seen.setdefault(p.id, p)
    return list(seen.values())


def check_no_leakage(pairs: Sequence[Pair]) -> dict[str, int]:
    """Every PMID (passage or claim source) belongs to exactly one split."""
    owner: dict[str, str] = {}
    for p in pairs:
        for pmid in filter(None, (p.pmid, p.claim_pmid)):
            prev = owner.setdefault(pmid, p.split)
            if prev != p.split:
                raise ValueError(f"PMID {pmid} appears in {prev} and {p.split}")
    counts: dict[str, int] = {}
    for p in pairs:
        counts[p.split] = counts.get(p.split, 0) + 1
    return counts


def write_pairs(path: Path, pairs: Iterable[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(p.as_dict(), ensure_ascii=False) + "\n" for p in pairs))


def read_pairs(path: Path) -> list[Pair]:
    return [Pair(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
