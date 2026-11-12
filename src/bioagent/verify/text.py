"""Answer text utilities: citation markers, sentence splitting and answer rewriting."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

CITE_RE = re.compile(r"\[(\d+(?:\s*[,;\u2013-]\s*\d+)*)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
ESTIMATE_RE = re.compile(r"\b(computed|estimate[sd]?|predicted|calculated)\b", re.I)
LIST_MARKER_RE = re.compile(r"^(?:[*\-+]|\d+[.)])\s+")
BOLD_LINE_RE = re.compile(r"^\*\*[^*]+\*\*:?$")
ABBREV_END_RE = re.compile(r"\b(?:vs|e\.g|i\.e|cf|al|approx|Fig|ca|no)\.$", re.I)
FLAG = " [unverified]"


def cite_numbers(group: str) -> list[int]:
    """Numbers inside one marker body, e.g. ``"2, 5-7"`` -> [2, 5, 6, 7]."""
    out: list[int] = []
    for part in re.split(r"\s*[,;]\s*", group):
        if re.fullmatch(r"\d+\s*[\u2013-]\s*\d+", part):
            lo, hi = (int(x) for x in re.split(r"\s*[\u2013-]\s*", part))
            out.extend(range(lo, hi + 1) if hi >= lo and hi - lo < 20 else [lo, hi])
        elif part.strip().isdigit():
            out.append(int(part))
    return out


def cited_in(text: str) -> list[int]:
    return list(dict.fromkeys(n for m in CITE_RE.finditer(text) for n in cite_numbers(m.group(1))))


def split_sentences(text: str) -> list[str]:
    """Sentence split that does not break after abbreviations such as "vs." or "e.g."."""
    out: list[str] = []
    for piece in SENTENCE_RE.split(text):
        if out and ABBREV_END_RE.search(out[-1]):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def is_structural(line: str) -> bool:
    """Headings, bold-only lines and lead-in lines ending in ':' carry no claim to cite."""
    stripped = LIST_MARKER_RE.sub("", line.strip()).strip()
    bare = stripped.strip("*_ ")
    return (
        not bare
        or stripped.startswith("#")
        or bool(BOLD_LINE_RE.match(stripped))
        or bare.endswith(":")
        or set(bare) <= set("-=_ ")
    )


@dataclass(frozen=True)
class AnswerSentence:
    id: int
    line: int
    text: str


def answer_sentences(answer: str) -> list[AnswerSentence]:
    """Number every claim-bearing sentence of the answer (structural lines are skipped)."""
    out: list[AnswerSentence] = []
    for i, line in enumerate(answer.splitlines()):
        if is_structural(line):
            continue
        body = LIST_MARKER_RE.sub("", line.strip())
        for sentence in split_sentences(body):
            if sentence.strip():
                out.append(AnswerSentence(len(out) + 1, i, sentence.strip()))
    return out


def rewrite_answer(answer: str, remove: Collection[int], flag: Collection[int]) -> str:
    """Drop the sentences in ``remove`` and mark those in ``flag`` (ids from answer_sentences).

    A list item or paragraph left without sentences is removed; headings left without any
    content below them are kept (they are harmless and keep the structure readable).
    """
    sentences = answer_sentences(answer)
    by_line: dict[int, list[AnswerSentence]] = {}
    for s in sentences:
        by_line.setdefault(s.line, []).append(s)
    lines = answer.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        parts = by_line.get(i)
        if parts is None:
            out.append(line)
            continue
        kept = []
        for s in parts:
            if s.id in remove:
                continue
            text = s.text
            if s.id in flag:
                text = _flag(text)
            kept.append(text)
        if not kept:
            continue
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        marker = LIST_MARKER_RE.match(stripped)
        out.append(indent + (marker.group(0) if marker else "") + " ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _flag(sentence: str) -> str:
    match = re.search(r"([.!?])(\s*)$", sentence)
    if match:
        return sentence[: match.start()] + FLAG + sentence[match.start() :]
    return sentence + FLAG


def renumber_citations(
    answer: str, citations: Sequence[Mapping[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Keep only citations still used in ``answer`` and renumber them 1..n by appearance."""
    by_n = {int(c["n"]): c for c in citations}
    order: dict[int, int] = {}

    def replace(match: re.Match[str]) -> str:
        parts = []
        for n in cite_numbers(match.group(1)):
            if n in by_n:
                order.setdefault(n, len(order) + 1)
                parts.append(str(order[n]))
        return "[" + ", ".join(dict.fromkeys(parts)) + "]" if parts else ""

    text = CITE_RE.sub(replace, answer)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    kept = [{**by_n[old], "n": new} for old, new in sorted(order.items(), key=lambda kv: kv[1])]
    return text, kept
