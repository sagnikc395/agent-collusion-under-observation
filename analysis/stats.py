"""Paired tests for within-seed comparisons, Holm correction, binomial test.

Seeds are paired across settings (a seed fixes the instance, the agent order and
the coalition), so every comparison is a test on per-seed differences, never on
pooled means.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Alternative = str  # "greater" | "less" | "two-sided"


@dataclass(frozen=True)
class PairedResult:
    n: int
    mean_diff: float
    statistic: float
    p_value: float
    alternative: Alternative
    exact: bool
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "mean_diff": self.mean_diff,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "alternative": self.alternative,
            "exact": self.exact,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def paired_values(
    treatment: Dict[int, float], control: Dict[int, float]
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Keep only seeds present and finite in both settings, in seed order."""
    seeds = sorted(
        s
        for s in set(treatment) & set(control)
        if treatment[s] is not None
        and control[s] is not None
        and math.isfinite(float(treatment[s]))
        and math.isfinite(float(control[s]))
    )
    a = np.array([float(treatment[s]) for s in seeds], dtype=float)
    b = np.array([float(control[s]) for s in seeds], dtype=float)
    return seeds, a, b


def sign_flip_permutation_test(
    diffs: Sequence[float],
    *,
    alternative: Alternative = "two-sided",
    n_permutations: int = 10000,
    seed: int = 0,
    exact_limit: int = 18,
) -> PairedResult:
    """Paired sign-flip permutation test on the mean difference.

    Exact (all 2^n sign assignments) when n is small, sampled otherwise.
    """
    d = np.asarray([float(x) for x in diffs], dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n == 0:
        return PairedResult(0, float("nan"), float("nan"), float("nan"), alternative, False)

    observed = float(d.mean())

    if n <= exact_limit:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        # `signs @ d` emits spurious FP warnings on some numpy builds; dot does not.
        stats = signs.dot(d) / n
        exact = True
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([1.0, -1.0], size=(int(n_permutations), n))
        stats = signs.dot(d) / n
        exact = False

    total = stats.size
    if alternative == "greater":
        count = int(np.sum(stats >= observed - 1e-12))
    elif alternative == "less":
        count = int(np.sum(stats <= observed + 1e-12))
    else:
        count = int(np.sum(np.abs(stats) >= abs(observed) - 1e-12))

    # Add-one correction: a sampled p-value is never reported as exactly 0.
    p = (count + (0 if exact else 1)) / (total + (0 if exact else 1))

    low, high = bootstrap_ci(d, seed=seed)
    return PairedResult(n, observed, observed, float(p), alternative, exact, low, high)


def paired_test(
    treatment: Dict[int, float],
    control: Dict[int, float],
    *,
    alternative: Alternative = "two-sided",
    n_permutations: int = 10000,
    seed: int = 0,
) -> PairedResult:
    """Sign-flip test on treatment minus control, seed by seed."""
    _, a, b = paired_values(treatment, control)
    return sign_flip_permutation_test(
        a - b, alternative=alternative, n_permutations=n_permutations, seed=seed
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap interval for the mean, resampling units (seeds)."""
    v = np.asarray([float(x) for x in values], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(int(n_boot), v.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, object]]:
    """Holm-Bonferroni step-down correction over a family of comparisons."""
    items = [(k, v) for k, v in p_values.items() if v is not None and math.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict[str, object]] = {}
    running_max = 0.0
    for index, (key, p) in enumerate(items):
        adjusted = min(1.0, (m - index) * p)
        running_max = max(running_max, adjusted)  # enforce monotonicity
        out[key] = {
            "p_raw": float(p),
            "p_holm": float(running_max),
            "reject": bool(running_max <= alpha),
            "rank": index + 1,
            "family_size": m,
        }
    for key, p in p_values.items():
        if key not in out:
            out[key] = {
                "p_raw": p,
                "p_holm": None,
                "reject": False,
                "rank": None,
                "family_size": m,
            }
    return out


def exact_binomial_test(
    successes: int, trials: int, p_null: float, *, alternative: Alternative = "greater"
) -> Dict[str, object]:
    """Exact binomial test; RQ5 uses it against chance partner identification."""
    from scipy.stats import binomtest

    if trials <= 0:
        return {"successes": 0, "trials": 0, "p_value": float("nan"), "estimate": None}
    result = binomtest(int(successes), int(trials), float(p_null), alternative=alternative)
    ci = result.proportion_ci(confidence_level=0.95)
    return {
        "successes": int(successes),
        "trials": int(trials),
        "p_null": float(p_null),
        "estimate": float(successes) / float(trials),
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
        "p_value": float(result.pvalue),
        "alternative": alternative,
    }


def summarise(values: Iterable[float]) -> Dict[str, Optional[float]]:
    v = np.asarray([float(x) for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": None, "sd": None, "ci_low": None, "ci_high": None}
    low, high = bootstrap_ci(v)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
    }
