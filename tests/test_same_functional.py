"""FIX 2 analog (DESIGN Sec 4, Sec 1a): the published value and the seed set
enter one rank function; no summary statistic of the seed set substitutes for
the set. Stage 1: the check is that the module exposes exactly that surface."""

from __future__ import annotations

import inspect

from seedcert.verifiers import nulls


def test_rank_test_takes_the_reference_set_not_a_summary() -> None:
    sig = inspect.signature(nulls.permutation_p_value)
    params = list(sig.parameters)
    assert params[:2] == ["statistic", "reference"]
    # the reference parameter is a sequence, not a precomputed mean/median
    assert "mean" not in params and "median" not in params


def test_no_distance_to_pooled_mean_helper_exists() -> None:
    # certiforget carried a warning against a "distance to ensemble mean"
    # shortcut; the seedcert analog is that no such summary-substitute helper is
    # exported here.
    banned = {"mean_diff_under_assignment", "distance_to_pooled_mean", "seed_mean_gap"}
    assert not (banned & set(dir(nulls)))
