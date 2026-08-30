"""timing_pilot: the 2-point linear fit and the sweep projection (WP5)."""

from __future__ import annotations

import pytest

from seedcert.experiment import timing_pilot


def test_run_timing_pilot_fits_two_points(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {"Cora": (0.20, 13_264), "PubMed": (1.50, 108_365)}
    monkeypatch.setattr(
        timing_pilot,
        "_time_recipe",
        lambda ds, bb, *, n_seeds, device: fake[ds],  # noqa: ARG005
    )
    monkeypatch.setattr(
        timing_pilot, "project_sweep_hours", lambda a, b, *, device="cpu": (300, 0.05)
    )

    r = timing_pilot.run_timing_pilot(small="Cora", large="PubMed", n_seeds=2, device="cpu")

    # slope solves the two points; intercept = s_small - slope * size_small
    slope = (1.50 - 0.20) / (108_365 - 13_264)
    assert r.fit_slope_per_elem == pytest.approx(slope)
    assert r.fit_intercept == pytest.approx(0.20 - slope * 13_264)
    assert r.seconds_per_run == {"Cora": 0.20, "PubMed": 1.50}
    assert r.projected_sweep_runs == 300


def test_fit_is_flat_when_sizes_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        timing_pilot,
        "_time_recipe",
        lambda ds, bb, *, n_seeds, device: (0.3, 1000),  # noqa: ARG005
    )
    monkeypatch.setattr(
        timing_pilot, "project_sweep_hours", lambda a, b, *, device="cpu": (12, 0.001)
    )
    r = timing_pilot.run_timing_pilot(small="A", large="B", n_seeds=1)
    assert r.fit_slope_per_elem == 0.0
    assert r.fit_intercept == pytest.approx(0.3)
