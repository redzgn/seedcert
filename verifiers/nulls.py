"""Rank p-value, bootstrap intervals, effect sizes for the reproduction test
(DESIGN Sec 4).

The FIX 2 analog is structural: the published value (statistic) and the seed
distribution (reference) enter :func:`permutation_p_value` directly; no summary
statistic of the seed set stands in for the set.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from seedcert.certificate import TestDirection


def _gen(rng: int | np.random.Generator) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def p_floor(n_seeds: int, direction: TestDirection = TestDirection.TWO_SIDED) -> float:
    """Smallest reportable rank p-value (DESIGN D4).

    ``2 / (n_seeds + 1)`` for the two-sided doubled-smaller-tail test,
    ``1 / (n_seeds + 1)`` for a one-sided (``LOWER`` / ``UPPER``) test.
    """
    k = 2.0 if direction is TestDirection.TWO_SIDED else 1.0
    return k / (n_seeds + 1)


def permutation_p_value(
    statistic: float,
    reference: ArrayLike,
    *,
    direction: TestDirection,
) -> float:
    """Rank-based p-value of ``statistic`` (the published value) against
    ``reference`` (the ``n`` per-seed metric values). All forms are *location*
    tests.

    * ``UPPER``: ``p_up = (1 + #{r_i >= statistic}) / (n + 1)`` - small p means
      the claim sits above the seed distribution (the re-implementation
      underperforms the claim).
    * ``LOWER``: ``p_lo = (1 + #{r_i <= statistic}) / (n + 1)`` - small p means
      the claim sits below the seeds (the re-implementation overperforms).
    * ``TWO_SIDED``: ``min(1, 2 * min(p_lo, p_up))`` - the doubled-smaller-tail
      form. Floors at ``2 / (n + 1)``, hence ``minimum_seeds`` needs
      ``2/alpha`` seeds (DESIGN D4).
    """
    arr = np.asarray(reference, dtype=float)
    n = arr.size
    p_up = (1 + int(np.sum(arr >= statistic))) / (n + 1)
    p_lo = (1 + int(np.sum(arr <= statistic))) / (n + 1)
    if direction is TestDirection.UPPER:
        return p_up
    if direction is TestDirection.LOWER:
        return p_lo
    return min(1.0, 2.0 * min(p_lo, p_up))


def m_run_mean_reference(
    seed_values: ArrayLike,
    *,
    m: int,
    n_bootstrap: int,
    rng: int | np.random.Generator,
) -> np.ndarray:
    """The sampling distribution of an ``m``-run re-implementation mean, implied
    by the ``n`` cached single-run values (DESIGN D10).

    ``n_bootstrap`` bootstrap samples, each the mean of ``m`` values drawn with
    replacement from ``seed_values``. Used as the reference an
    ``aggregation="mean"`` published value is ranked against.
    """
    arr = np.asarray(seed_values, dtype=float)
    gen = _gen(rng)
    idx = gen.integers(0, arr.size, size=(n_bootstrap, m))
    out: np.ndarray = arr[idx].mean(axis=1)
    return out


def cliffs_delta(sample: ArrayLike, reference: ArrayLike) -> float:
    """Cliff's delta of ``sample`` vs ``reference`` in ``[-1, 1]``.

    Here ``sample`` is the singleton ``{claim_value}``; a positive value means
    the published value tends to exceed the seed metrics (the re-implementation
    underperforms the claim).
    """
    s = np.asarray(sample, dtype=float)
    r = np.asarray(reference, dtype=float)
    gt = int(np.sum(s[:, None] > r[None, :]))
    lt = int(np.sum(s[:, None] < r[None, :]))
    return (gt - lt) / float(s.size * r.size)


def standardized_gap(statistic: float, reference: ArrayLike) -> float:
    """``(mean(reference) - statistic) / sd(reference)`` - signed gap of the
    re-implementation mean from the published value, in seed SDs. Positive means
    the re-implementation scores *above* the published value. ``nan`` if the seed
    distribution has no spread."""
    arr = np.asarray(reference, dtype=float)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return (float(arr.mean()) - statistic) / sd if sd > 1e-12 else float("nan")


def bootstrap_mean_ci(
    reference: ArrayLike,
    *,
    ci_level: float,
    n_bootstrap: int,
    rng: int | np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``mean(reference)``, resampling the per-seed
    metric values with replacement."""
    arr = np.asarray(reference, dtype=float)
    gen = _gen(rng)
    n = arr.size
    means = arr[gen.integers(0, n, size=(n_bootstrap, n))].mean(axis=1)
    lo = float(np.quantile(means, (1 - ci_level) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci_level) / 2))
    return lo, hi


def bootstrap_effect_ci(
    statistic: float,
    reference: ArrayLike,
    *,
    ci_level: float,
    n_bootstrap: int,
    rng: int | np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the Cliff's-delta effect size, resampling the
    per-seed metric values with replacement and recomputing
    :func:`cliffs_delta` against the fixed ``statistic``."""
    arr = np.asarray(reference, dtype=float)
    gen = _gen(rng)
    n = arr.size
    resamples = arr[gen.integers(0, n, size=(n_bootstrap, n))]  # [B, n]
    gt = np.sum(resamples < statistic, axis=1)  # statistic > r_i
    lt = np.sum(resamples > statistic, axis=1)  # statistic < r_i
    deltas = (gt - lt) / float(n)
    lo = float(np.quantile(deltas, (1 - ci_level) / 2))
    hi = float(np.quantile(deltas, 1 - (1 - ci_level) / 2))
    return lo, hi


def tost_equivalence_abs(
    statistic: float,
    reference: ArrayLike,
    *,
    margin_points: float,
    alpha: float,
    n_bootstrap: int,
    rng: int | np.random.Generator,
) -> dict[str, float | bool]:
    """Bootstrap two one-sided tests that ``|mean(reference) - statistic|`` is
    within ``margin_points`` (absolute, in metric units).

    On the bootstrap distribution of ``mean(reference)``:
    ``p_low  = P(mean <= statistic - margin)`` and
    ``p_high = P(mean >= statistic + margin)``. ``tost_p = max(p_low, p_high)``;
    ``equivalent`` when ``tost_p < alpha`` (both one-sided nulls rejected) -
    the affirmative "reproduced within +/- margin" statement (DESIGN D6).
    """
    arr = np.asarray(reference, dtype=float)
    gen = _gen(rng)
    n = arr.size
    means = arr[gen.integers(0, n, size=(n_bootstrap, n))].mean(axis=1)
    p_low = float(np.mean(means <= statistic - margin_points))
    p_high = float(np.mean(means >= statistic + margin_points))
    tost_p = max(p_low, p_high)
    return {
        "margin_points": float(margin_points),
        "tost_p": tost_p,
        "equivalent": bool(tost_p < alpha),
    }
