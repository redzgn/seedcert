"""Rank p-value + bootstrap helpers (DESIGN Sec 4, WP2)."""

from __future__ import annotations

import numpy as np
import pytest

from seedcert.certificate import TestDirection
from seedcert.verifiers import nulls

# n = 44 so the two-sided test can resolve (floor 2/45 < 0.05)
_REF = list(np.round(0.805 + 0.004 * np.random.default_rng(3).normal(size=44), 5))


def test_p_floor_direction_aware() -> None:
    assert nulls.p_floor(44, TestDirection.TWO_SIDED) == pytest.approx(2 / 45)
    assert nulls.p_floor(44, TestDirection.UPPER) == pytest.approx(1 / 45)
    assert nulls.p_floor(44) == pytest.approx(2 / 45)  # two-sided default


def test_two_sided_is_doubled_smaller_tail() -> None:
    # claim above every seed -> p_up = 1/(n+1), p_lo = 1 -> p = 2/(n+1)
    p = nulls.permutation_p_value(0.95, _REF, direction=TestDirection.TWO_SIDED)
    assert p == pytest.approx(2 / 45)
    # claim at the median -> both tails ~0.5 -> capped at 1.0
    p_mid = nulls.permutation_p_value(
        float(np.median(_REF)), _REF, direction=TestDirection.TWO_SIDED
    )
    assert p_mid == pytest.approx(1.0)


def test_one_sided_location_semantics() -> None:
    # claim above all seeds: UPPER small (re-impl underperforms), LOWER large
    assert nulls.permutation_p_value(0.95, _REF, direction=TestDirection.UPPER) == pytest.approx(
        1 / 45
    )
    assert nulls.permutation_p_value(0.95, _REF, direction=TestDirection.LOWER) == pytest.approx(
        1.0
    )
    # claim below all seeds: mirror image
    assert nulls.permutation_p_value(0.1, _REF, direction=TestDirection.LOWER) == pytest.approx(
        1 / 45
    )


def test_m_run_mean_reference_shape_and_variance() -> None:
    rng = np.random.default_rng(0)
    s = 0.80 + 0.02 * rng.normal(size=60)
    ref1 = nulls.m_run_mean_reference(s, m=1, n_bootstrap=5000, rng=0)
    ref25 = nulls.m_run_mean_reference(s, m=25, n_bootstrap=5000, rng=0)
    assert ref1.shape == (5000,) and ref25.shape == (5000,)
    # an m-run mean has ~1/sqrt(m) the spread of a single run
    assert ref25.std() == pytest.approx(ref1.std() / np.sqrt(25), rel=0.15)
    # both centre on the sample mean
    assert ref25.mean() == pytest.approx(float(s.mean()), abs=3e-3)


def test_cliffs_delta_sign() -> None:
    assert nulls.cliffs_delta([0.95], _REF) == 1.0
    assert nulls.cliffs_delta([0.1], _REF) == -1.0


def test_standardized_gap_sign_and_scale() -> None:
    arr = np.array(_REF, dtype=float)
    g = nulls.standardized_gap(arr.mean() - arr.std(ddof=1), _REF)
    assert g == pytest.approx(1.0, abs=1e-6)  # re-impl one SD above the claim


def test_bootstrap_mean_ci_brackets_mean() -> None:
    lo, hi = nulls.bootstrap_mean_ci(_REF, ci_level=0.95, n_bootstrap=2000, rng=0)
    assert lo <= np.mean(_REF) <= hi and lo < hi


def test_bootstrap_effect_ci_in_range() -> None:
    lo, hi = nulls.bootstrap_effect_ci(0.81, _REF, ci_level=0.95, n_bootstrap=2000, rng=0)
    assert -1.0 <= lo <= hi <= 1.0


def test_tost_equivalent_when_claim_near_mean() -> None:
    mean = float(np.mean(_REF))
    near = nulls.tost_equivalence_abs(
        mean, _REF, margin_points=0.02, alpha=0.05, n_bootstrap=4000, rng=0
    )
    assert set(near) == {"margin_points", "tost_p", "equivalent"}
    assert near["equivalent"] is True
    far = nulls.tost_equivalence_abs(
        mean + 0.1, _REF, margin_points=0.02, alpha=0.05, n_bootstrap=4000, rng=0
    )
    assert far["equivalent"] is False
