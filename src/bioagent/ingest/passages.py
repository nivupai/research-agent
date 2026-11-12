"""Turn a record (title + abstract) into retrieval passages.

Short records become one passage. Records longer than ``max_tokens`` are split on sentence
boundaries into overlapping chunks, each prefixed with the title so it stands alone.
Passage ids are ``"{pmid}:{chunk_idx}"``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

DEFAULT_MAX_TOKENS = 350
DEFAULT_OVERLAP_TOKENS = 60

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
# Split after ., ! or ? followed by whitespace and an uppercase letter, digit or bracket.
# Avoids splitting "e.g. the" and "vs. control" (lowercase follows).
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass(frozen=True)
class Passage:
    passage_id: str
    pmid: str
    chunk_idx: int
    title: str
    text: str
    year: int | None
    journal: str | None
    doi: str | None

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(self.text.encode()).hexdigest()


def approx_tokens(text: str) -> int:
    """Cheap token estimate (words + punctuation); close to WordPiece counts for English."""
    return len(_TOKEN_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


def make_passages(
    pmid: str,
    title: str | None,
    abstract: str | None,
    *,
    year: int | None = None,
    journal: str | None = None,
    doi: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Passage]:
    title = (title or "").strip()
    abstract = (abstract or "").strip()

    def passage(idx: int, body: str) -> Passage:
        text = f"{title}\n{body}".strip() if title else body
        return Passage(f"{pmid}:{idx}", pmid, idx, title, text, year, journal, doi)

    if approx_tokens(f"{title} {abstract}") <= max_tokens:
        return [passage(0, abstract)]

    budget = max(50, max_tokens - approx_tokens(title))
    sentences = split_sentences(abstract)
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        n = approx_tokens(sentence)
        if current and current_tokens + n > budget:
            chunks.append(current)
            # carry trailing sentences (up to overlap_tokens) into the next chunk
            carry: list[str] = []
            carry_tokens = 0
            for prev in reversed(current):
                t = approx_tokens(prev)
                if carry_tokens + t > overlap_tokens:
                    break
                carry.insert(0, prev)
                carry_tokens += t
            current, current_tokens = carry, carry_tokens
        current.append(sentence)
        current_tokens += n
    if current and (not chunks or current != chunks[-1][-len(current) :]):
        chunks.append(current)
    return [passage(i, " ".join(chunk)) for i, chunk in enumerate(chunks)]
