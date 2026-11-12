"""Checks for claims that cite computed results (RDKit, sandbox): numbers, not NLI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from bioagent.verify.text import CITE_RE

# Standalone numbers: not part of identifiers such as U87, PRKAA1, CHEMBL1431 or 5-HT2A.
_NUMBER_RE = re.compile(r"(?<![\w.\-])[-\u2212]?\d+(?:\.\d+)?(?![\w]|\.\d)")
# Phrases whose digits are not measured values.
_IGNORE_RE = re.compile(
    r"\b(?:rule of (?:five|5)|ro5|type [12]|phase [0-4]|lipinski'?s? rule of 5|"
    r"step \d|\d+(?:st|nd|rd|th))\b",
    re.I,
)


def numbers_in(text: str) -> list[float]:
    cleaned = _IGNORE_RE.sub(" ", CITE_RE.sub(" ", text))
    out: list[float] = []
    for raw in _NUMBER_RE.findall(cleaned):
        value = raw.replace("\u2212", "-")
        try:
            out.append(float(value))
        except ValueError:
            continue
    return out


def _decimals(value: float, raw_text: str | None = None) -> int:
    text = raw_text if raw_text is not None else repr(value)
    return len(text.split(".")[1]) if "." in text else 0


def matches(claimed: float, actual: float, claimed_decimals: int) -> bool:
    """True if ``claimed`` is ``actual`` rounded to the claim's precision (or within 0.5%)."""
    tolerance = 0.5 * 10 ** (-claimed_decimals) + 1e-9
    if abs(claimed - actual) <= tolerance:
        return True
    return actual != 0 and abs(claimed - actual) / abs(actual) <= 0.005


_COUNT_WORDS = {"zero": "0", "no": "0", "one": "1", "two": "2", "three": "3", "four": "4"}
_COUNT_RE = re.compile(
    r"\b(zero|no|one|two|three|four)\s+(?=(?:\w+\s+)?(?:violations?|alerts?|donors?|"
    r"acceptors?|rotatable|rings?)\b)",
    re.I,
)
# Rule-set names as they appear in claims -> key of that rule in drug_likeness output.
_RULES: dict[str, str] = {
    r"lipinski|rule of (?:five|5)|\bro5\b": "lipinski",
    r"\bveber\b": "veber",
    r"\bghose\b": "ghose",
    r"blood[- ]brain|\bbbb\b": "bbb",
}
_NEGATIVE_RE = re.compile(
    r"\b(?:fail(?:s|ed|ing)?|violat\w*|not\s+(?:pass|meet|compl|satisf|permeable|likely)\w*|"
    r"non[- ]?(?:compliant|permeable|penetrant)|impermeable|unlikely|low)\b",
    re.I,
)
_POSITIVE_RE = re.compile(
    r"\b(?:pass(?:es|ed|ing)?|compl(?:y|ies|iant)|meets?|satisf\w*|permeable|penetrant|"
    r"cross(?:es)?)\b",
    re.I,
)
_NO_VIOLATIONS_RE = re.compile(r"\b(?:zero|no|0)\s+violations?\b", re.I)
_ALL_RE = re.compile(r"\b(?:both|all|each)\b", re.I)


def _expand_counts(text: str) -> str:
    return _COUNT_RE.sub(lambda m: _COUNT_WORDS[m.group(1).lower()] + " ", text)


def _rule_value(output: dict[str, Any], key: str) -> bool | None:
    rule = output.get(key)
    if not isinstance(rule, dict):
        return None
    value = rule.get("likely_permeable" if key == "bbb" else "passed")
    return value if isinstance(value, bool) else None


def check_rules(claim: str, outputs: list[str]) -> tuple[list[str], list[str]]:
    """(checked, mismatched) pass/fail assertions about named rule sets."""
    parsed = []
    for out in outputs:
        try:
            data = json.loads(out)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            parsed.append(data)
    if not parsed:
        return [], []
    text = _NO_VIOLATIONS_RE.sub(" passes ", CITE_RE.sub(" ", claim))
    negative = bool(_NEGATIVE_RE.search(text))
    positive = bool(_POSITIVE_RE.search(_NEGATIVE_RE.sub(" ", text)))  # "non-permeable"
    if negative == positive:
        return [], []  # no (or mixed) polarity: nothing to assert
    expected = not negative
    checked: list[str] = []
    mismatched: list[str] = []
    for pattern, key in _RULES.items():
        if not re.search(pattern, text, re.I):
            continue
        values = [v for v in (_rule_value(d, key) for d in parsed) if v is not None]
        if not values:
            continue
        ok = all(v == expected for v in values) if _ALL_RE.search(text) else expected in values
        checked.append(key)
        if not ok:
            mismatched.append(f"{key} {'passed' if expected else 'failed'}")
    return checked, mismatched


@dataclass(frozen=True)
class ComputedCheck:
    ok: bool
    checked: list[float]
    unmatched: list[float]
    rules: list[str] = field(default_factory=list)
    rule_mismatches: list[str] = field(default_factory=list)

    @property
    def anything_checked(self) -> bool:
        return bool(self.checked or self.rules)

    @property
    def detail(self) -> str:
        if not self.anything_checked:
            return "no numbers or rule outcomes to check"
        parts = []
        if self.checked:
            if self.unmatched:
                shown = ", ".join(f"{v:g}" for v in self.unmatched[:4])
                parts.append(f"numbers not in the tool output: {shown}")
            else:
                parts.append(f"all {len(self.checked)} numbers match the tool output")
        if self.rules:
            if self.rule_mismatches:
                parts.append(f"tool output disagrees: {', '.join(self.rule_mismatches)}")
            else:
                parts.append(f"rule outcomes match ({', '.join(self.rules)})")
        return "; ".join(parts)


def check_numbers(claim: str, outputs: list[str]) -> ComputedCheck:
    """Numbers and rule pass/fail statements in a claim against the cited tool outputs."""
    rules, rule_mismatches = check_rules(claim, outputs)
    cleaned = _IGNORE_RE.sub(" ", CITE_RE.sub(" ", _expand_counts(claim)))
    raw_numbers = [r.replace("\u2212", "-") for r in _NUMBER_RE.findall(cleaned)]
    actual = [n for out in outputs for n in numbers_in(out)]
    checked: list[float] = []
    unmatched: list[float] = []
    for raw in raw_numbers:
        try:
            value = float(raw)
        except ValueError:
            continue
        checked.append(value)
        if not any(matches(value, a, _decimals(value, raw)) for a in actual):
            unmatched.append(value)
    ok = bool(checked or rules) and not unmatched and not rule_mismatches
    return ComputedCheck(ok, checked, unmatched, rules, rule_mismatches)
