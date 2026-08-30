"""FIX 1 analog: the rank floor and minimum seeds (DESIGN D4).

Two-sided is the default and floors at 2/(n+1); one-sided floors at 1/(n+1)."""

from __future__ import annotations

import math

import pytest

from seedcert.certificate import minimum_seeds, minimum_seeds_aggregate


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(0.05, 40), (0.10, 20), (0.02, 100), (0.04, 50)],
)
def test_minimum_seeds_two_sided_default(alpha: float, expected: int) -> None:
    assert minimum_seeds(alpha) == expected


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(0.05, 20), (0.03, 33), (0.10, 10)],
)
def test_minimum_seeds_one_sided(alpha: float, expected: int) -> None:
    assert minimum_seeds(alpha, two_sided=False) == expected


def test_two_sided_floor_crosses_alpha_at_the_minimum() -> None:
    alpha = 0.05
    n = minimum_seeds(alpha)  # 40
    assert 2.0 / (n + 1) < alpha
    assert 2.0 / n >= alpha  # one fewer seed cannot reject


def test_matches_closed_form() -> None:
    for alpha in (0.05, 0.033, 0.02):
        assert minimum_seeds(alpha) == math.floor(2.0 / alpha - 1.0) + 1
        assert minimum_seeds(alpha, two_sided=False) == math.floor(1.0 / alpha - 1.0) + 1


def test_minimum_seeds_aggregate() -> None:
    # bounded below by the one-sided single-run floor (20 at alpha=0.05)
    assert minimum_seeds_aggregate(5) == 20
    assert minimum_seeds_aggregate(20) == 20
    # for a large m-run mean, m dominates
    assert minimum_seeds_aggregate(100) == 100
    assert minimum_seeds_aggregate(30) == 30
