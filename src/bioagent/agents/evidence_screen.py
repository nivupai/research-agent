"""Target-specific evidence for drug candidates (``TARGET_EVIDENCE_SCREEN=sentence``).

For each drug the corpus searches surfaced, find a sentence in the retrieved passages that
names the drug (or a synonym), the target (or an alias) and an action word. Candidates with
such a sentence are what a "which drugs act on X" answer needs; mere co-mention in a paper
(ATP, glucose, drugs listed in a review) is not. No LLM is involved. See
docs/design_candidate_evidence.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bioagent.ingest.passages import split_sentences
from bioagent.kg.aliases import ALIAS_GROUPS, canonical_alias

ACTION_RE = re.compile(
    r"\b(activat\w*|inhibit\w*|stimulat\w*|induc\w*|suppress\w*|increas\w*|decreas\w*|"
    r"phosphorylat\w*|block\w*|target\w*|agonist\w*|antagonist\w*)\b",
    re.I,
)
MAX_SENTENCE_CHARS = 300
MIN_TERM_CHARS = 3
# A sentence naming several candidate drugs is usually a list (reviews); only sentences about
# at most this many candidates count as evidence for a drug.
MAX_DRUGS_PER_SENTENCE = 1


@dataclass(frozen=True)
class Evidence:
    pmid: str
    rank: int  # best search rank (1-based) of the passage the sentence came from
    sentence: str


def target_terms(target: str) -> list[str]:
    """The target and its aliases: the group name, member gene symbols and full names."""
    from bioagent.kg import aliases

    terms = {target.strip()}
    alias = canonical_alias(target)
    if alias is not None:
        terms.add(alias)
        terms.update(ALIAS_GROUPS.get(alias, ()))
        terms.update(k.title() for k, v in aliases._SYNONYMS.items() if v == alias)
    return sorted((t for t in terms if len(t) >= MIN_TERM_CHARS - 1), key=len, reverse=True)


def drug_terms(drug: Mapping[str, Any]) -> list[str]:
    names = [str(drug.get("name") or "")] + [str(s) for s in drug.get("synonyms") or []]
    keep = {n.strip() for n in names if len(n.strip()) > MIN_TERM_CHARS}
    return sorted(keep, key=len, reverse=True)


def _pattern(terms: Iterable[str]) -> re.Pattern[str] | None:
    parts = [re.escape(t) for t in terms if t]
    if not parts:
        return None
    return re.compile(r"(?<![\w-])(?:" + "|".join(parts) + r")(?![\w-])", re.I)


def find_evidence(
    drugs: Sequence[Mapping[str, Any]],
    searches: Sequence[Mapping[str, Any]],
    target: str,
    max_drugs_per_sentence: int = MAX_DRUGS_PER_SENTENCE,
) -> dict[str, Evidence]:
    """Best drug+target+action sentence per drug id (lowest search rank, then shortest).

    Only sentences that name at most ``max_drugs_per_sentence`` of ``drugs`` count.
    """
    target_re = _pattern(target_terms(target))
    if target_re is None:
        return {}
    passages: list[tuple[int, str, str]] = []  # (rank, pmid, text)
    for result in searches:
        seen: dict[str, int] = {}
        for hit in result.get("hits", []):
            pmid = str(hit.get("pmid", ""))
            rank = seen.setdefault(pmid, len(seen) + 1)
            text = str(hit.get("text") or "")
            if text:
                passages.append((rank, pmid, text))
    sentences = [
        (rank, pmid, s)
        for rank, pmid, text in passages
        for s in split_sentences(text)
        if target_re.search(s) and ACTION_RE.search(s)
    ]
    patterns = {str(d["id"]): _pattern(drug_terms(d)) for d in drugs}
    found: dict[str, Evidence] = {}
    for rank, pmid, sentence in sentences:
        named = [i for i, p in patterns.items() if p is not None and p.search(sentence)]
        if not named or len(named) > max_drugs_per_sentence:
            continue
        cand = Evidence(pmid, rank, " ".join(sentence.split())[:MAX_SENTENCE_CHARS])
        for drug_id in named:
            best = found.get(drug_id)
            if best is None or (cand.rank, len(cand.sentence)) < (best.rank, len(best.sentence)):
                found[drug_id] = cand
    return found
