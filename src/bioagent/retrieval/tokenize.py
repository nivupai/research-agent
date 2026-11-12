"""Biomedical-friendly tokenizer for BM25.

Lowercases, keeps alphanumeric terms such as ``ampk``, ``pi3k``, ``mtorc1``, ``il-6`` and
``5-fu``, and keeps hyphenated compounds as one token while also emitting their parts
(``blood-brain`` -> ``blood-brain``, ``blood``, ``brain``) so "blood brain barrier" and
"blood-brain barrier" match. Drops a small English stopword list and 1-letter tokens (but
keeps single digits, which matter in names like "4-ebp1" parts).
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[^\W_]+(?:[-/][^\W_]+)*")

_STOPWORDS_TEXT = """
a an and are as at be been being but by can could did do does for from had has have
how in into is it its may might more most no nor not of on or our such than that the
their them then there these they this those through to under up was we were what when
where which while who whom why will with within without would you your also however
thus therefore both each other some any all between during after before about above
below over among via vs versus et al
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        term = match.group(0)
        if "-" in term or "/" in term:
            parts = [p for p in re.split(r"[-/]", term) if p]
            tokens.append(term)
            tokens.extend(p for p in parts if _keep(p))
        elif _keep(term):
            tokens.append(term)
    return tokens


def _keep(term: str) -> bool:
    if term in STOPWORDS:
        return False
    return len(term) > 1 or term.isdigit()
