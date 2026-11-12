"""Statistics for repeated-run evaluation (pure Python, seeded, no numpy needed).

* ``cluster_bootstrap``: mean over questions with a 95% CI from a two-stage bootstrap
  (resample questions, then each chosen question's repeats), so repeat noise is included.
* ``flip_rate``: share of questions whose outcome is not identical across repeats.
* ``paired_bootstrap`` / ``permutation_test``: paired comparison of two runs on the same
  questions (per-question means); the permutation test flips the sign of each paired
  difference (exact for n <= 16, Monte Carlo otherwise).
* ``cohen_kappa`` / ``weighted_kappa`` (+ ``bootstrap_kappa``) for judge calibration.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_B = 2000
SEED = 1234


@dataclass(frozen=True)
class Estimate:
    mean: float | None
    lo: float | None
    hi: float | None
    n: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1]) of already sorted values."""
    if not sorted_values:
        raise ValueError("no values")
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _clean(values: Mapping[str, Sequence[float | None]]) -> dict[str, list[float]]:
    out = {}
    for key, vs in values.items():
        kept = [float(v) for v in vs if v is not None and not math.isnan(float(v))]
        if kept:
            out[key] = kept
    return out


def cluster_bootstrap(
    values: Mapping[str, Sequence[float | None]],
    *,
    b: int = DEFAULT_B,
    alpha: float = 0.05,
    seed: int = SEED,
) -> Estimate:
    """Mean of per-question means, with a two-stage percentile bootstrap CI."""
    data = _clean(values)
    if not data:
        return Estimate(None, None, None, 0)
    groups = list(data.values())
    point = statistics.fmean(statistics.fmean(g) for g in groups)
    if len(groups) == 1 and len(groups[0]) == 1:
        return Estimate(point, None, None, 1)
    rng = random.Random(seed)
    n = len(groups)
    means = []
    for _ in range(b):
        total = 0.0
        for _ in range(n):
            g = groups[rng.randrange(n)]
            total += sum(g[rng.randrange(len(g))] for _ in range(len(g))) / len(g)
        means.append(total / n)
    means.sort()
    return Estimate(point, _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2), n)


def flip_rate(values: Mapping[str, Sequence[float | None]]) -> float | None:
    """Share of questions (with >= 2 repeats) whose scores differ between repeats."""
    data = {k: v for k, v in _clean(values).items() if len(v) >= 2}
    if not data:
        return None
    return sum(1 for v in data.values() if max(v) != min(v)) / len(data)


def _paired(
    base: Mapping[str, Sequence[float | None]], cand: Mapping[str, Sequence[float | None]]
) -> list[float]:
    b, c = _clean(base), _clean(cand)
    return [statistics.fmean(c[k]) - statistics.fmean(b[k]) for k in sorted(set(b) & set(c))]


@dataclass(frozen=True)
class Comparison:
    n: int
    base_mean: float | None
    cand_mean: float | None
    delta: float | None
    lo: float | None
    hi: float | None
    p_value: float | None
    regression: bool
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_bootstrap(
    diffs: Sequence[float], *, b: int = DEFAULT_B, alpha: float = 0.05, seed: int = SEED
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2)


def permutation_test(
    diffs: Sequence[float], *, b: int = 10_000, seed: int = SEED, exact_max: int = 16
) -> float:
    """Two-sided paired sign-flip test of mean difference == 0."""
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    eps = 1e-12
    if len(nonzero) <= exact_max:
        hits = total = 0
        for signs in itertools.product((1, -1), repeat=len(nonzero)):
            total += 1
            hits += abs(sum(s * d for s, d in zip(signs, nonzero, strict=True))) >= observed - eps
        return hits / total
    rng = random.Random(seed)
    hits = sum(
        abs(sum(d if rng.random() < 0.5 else -d for d in nonzero)) >= observed - eps
        for _ in range(b)
    )
    return (hits + 1) / (b + 1)


def compare(
    base: Mapping[str, Sequence[float | None]],
    cand: Mapping[str, Sequence[float | None]],
    *,
    margin: float = 0.0,
    alpha: float = 0.05,
    higher_is_better: bool = True,
    seed: int = SEED,
) -> Comparison:
    """Paired comparison on shared questions. Regression: the CI of (cand - base) lies
    entirely below ``-margin`` (or above ``+margin`` when lower is better)."""
    diffs = _paired(base, cand)
    if not diffs:
        return Comparison(0, None, None, None, None, None, None, False, "no shared questions")
    b, c = _clean(base), _clean(cand)
    keys = sorted(set(b) & set(c))
    base_mean = statistics.fmean(statistics.fmean(b[k]) for k in keys)
    cand_mean = statistics.fmean(statistics.fmean(c[k]) for k in keys)
    delta = statistics.fmean(diffs)
    if len(diffs) == 1:
        return Comparison(1, base_mean, cand_mean, delta, None, None, None, False, "n=1")
    lo, hi = paired_bootstrap(diffs, alpha=alpha, seed=seed)
    p = permutation_test(diffs, seed=seed)
    regression = hi < -margin if higher_is_better else lo > margin
    if regression:
        verdict = "regression"
    elif (lo > margin if higher_is_better else hi < -margin) and p < alpha:
        verdict = "improvement"
    else:
        verdict = "no significant change"
    return Comparison(len(diffs), base_mean, cand_mean, delta, lo, hi, p, regression, verdict)


# --- agreement ---------------------------------------------------------------------------
def _confusion(a: Sequence[str], b: Sequence[str], cats: Sequence[str]) -> list[list[int]]:
    index = {c: i for i, c in enumerate(cats)}
    m = [[0] * len(cats) for _ in cats]
    for x, y in zip(a, b, strict=True):
        m[index[x]][index[y]] += 1
    return m


def weighted_kappa(
    a: Sequence[str],
    b: Sequence[str],
    categories: Sequence[str],
    weights: str = "quadratic",
) -> float:
    """Cohen's kappa with disagreement weights: 'none', 'linear' or 'quadratic'.
    ``categories`` is the ordinal order (e.g. incorrect < partially < correct)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label lists")
    k = len(categories)
    m = _confusion(a, b, categories)
    n = len(a)
    rows = [sum(r) for r in m]
    cols = [sum(m[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        if weights == "none":
            return 0.0 if i == j else 1.0
        d = abs(i - j) / max(1, k - 1)
        return d if weights == "linear" else d * d

    observed = sum(w(i, j) * m[i][j] for i in range(k) for j in range(k)) / n
    expected = sum(w(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k)) / (n * n)
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return 1.0 - observed / expected


def cohen_kappa(a: Sequence[str], b: Sequence[str], categories: Sequence[str]) -> float:
    return weighted_kappa(a, b, categories, weights="none")


def bootstrap_kappa(
    a: Sequence[str],
    b: Sequence[str],
    categories: Sequence[str],
    *,
    weights: str = "none",
    b_samples: int = DEFAULT_B,
    alpha: float = 0.05,
    seed: int = SEED,
) -> Estimate:
    point = weighted_kappa(a, b, categories, weights)
    rng = random.Random(seed)
    n = len(a)
    stats: list[float] = []
    for _ in range(b_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(weighted_kappa([a[i] for i in idx], [b[i] for i in idx], categories, weights))
    stats.sort()
    return Estimate(point, _percentile(stats, alpha / 2), _percentile(stats, 1 - alpha / 2), n)


def summarize_metric(
    values: Mapping[str, Sequence[float | None]], fn: Callable[..., Estimate] = cluster_bootstrap
) -> dict[str, Any]:
    est = fn(values)
    return {**est.as_dict(), "flip_rate": flip_rate(values)}
