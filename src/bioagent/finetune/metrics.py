"""Classification metrics for the verifier comparison (pure Python, seeded bootstrap)."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

from bioagent.finetune.data import LABELS


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold) if gold else math.nan


def per_class_f1(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = LABELS
) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, pred, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return out


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    """Macro-F1 over the classes present in the gold labels."""
    present = [label for label in labels if label in set(gold)]
    if not present:
        return math.nan
    f1 = per_class_f1(gold, pred, present)
    return sum(f1.values()) / len(present)


def confusion(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = LABELS
) -> dict[str, dict[str, int]]:
    """rows = gold, columns = predicted."""
    m = {g: dict.fromkeys(labels, 0) for g in labels}
    for g, p in zip(gold, pred, strict=True):
        if g in m and p in m[g]:
            m[g][p] += 1
    return m


def bootstrap_ci(
    gold: Sequence[str],
    pred: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    *,
    b: int = 2000,
    alpha: float = 0.05,
    seed: int = 1234,
) -> dict[str, Any]:
    n = len(gold)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    stats = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(metric([gold[i] for i in idx], [pred[i] for i in idx]))
    stats = sorted(s for s in stats if not math.isnan(s))
    lo = stats[int(alpha / 2 * (len(stats) - 1))]
    hi = stats[int((1 - alpha / 2) * (len(stats) - 1))]
    return {"mean": metric(gold, pred), "lo": lo, "hi": hi, "n": n}


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summary(gold: Sequence[str], pred: Sequence[str], b: int = 2000) -> dict[str, Any]:
    return {
        "n": len(gold),
        "accuracy": bootstrap_ci(gold, pred, accuracy, b=b),
        "macro_f1": bootstrap_ci(gold, pred, macro_f1, b=b),
        "per_class_f1": per_class_f1(gold, pred),
        "confusion": confusion(gold, pred),
    }
