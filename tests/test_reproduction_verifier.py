"""ReproductionVerifier behaviour (DESIGN Sec 4, Sec 8, WP2).

Uses a hand-built RecipeRuns stand-in with canned per-seed metrics - no training
or download."""

from __future__ import annotations

import numpy as np
import pytest

from seedcert.cache.spec import RecipeKey
from seedcert.certificate import Certificate, Decision, TestDirection
from seedcert.claim import PublishedClaim
from seedcert.recipe import Recipe
from seedcert.verifiers.reproduction import ReproductionVerifier

_RECIPE = Recipe("gcn", overrides={"hidden_dim": 16}, label="kipf")


class _Run:
    env = {"gpu_model": "TestGPU"}


class _FakeRuns:
    """Implements the read surface ReproductionVerifier / assumptions need."""

    def __init__(self, acc: np.ndarray, split_protocol: str = "planetoid-public") -> None:
        self._acc = np.asarray(acc, dtype=float)
        self.dataset = "Cora"
        self.split_protocol = split_protocol
        self.recipe = RecipeKey("Cora", split_protocol, "gcn", _RECIPE.recipe_hash())

    @property
    def n_seeds(self) -> int:
        return int(self._acc.size)

    @property
    def seed_list(self) -> tuple[int, ...]:
        return tuple(range(self.n_seeds))

    def metric_values(self, name: str) -> np.ndarray:
        if name == "test_accuracy":
            return self._acc
        return self._acc - {"precision": 0.01, "recall": 0.015, "f1": 0.012}[name]

    def recipe_descriptor(self) -> dict[str, object]:
        return _RECIPE.descriptor()

    def __iter__(self):
        return iter([_Run() for _ in range(self.n_seeds)])


@pytest.fixture
def consistent() -> _FakeRuns:
    rng = np.random.default_rng(0)
    return _FakeRuns(np.clip(0.815 + 0.004 * rng.normal(size=50), 0, 1))


def test_consistent_claim_reproduced(consistent: _FakeRuns) -> None:
    claim = PublishedClaim("test_accuracy", 0.815, "Kipf 2017", "planetoid-public")
    cert = ReproductionVerifier().certify(consistent, claim)  # type: ignore[arg-type]
    assert isinstance(cert, Certificate)
    assert cert.decision is Decision.REPRODUCED
    assert cert.n_seeds == 50
    assert cert.p_floor == pytest.approx(2 / 51)
    assert cert.reimpl_ci[0] <= cert.reimpl_mean <= cert.reimpl_ci[1]
    assert set(cert.secondary) == {"precision", "recall", "f1"}


def test_shifted_claim_discrepant(consistent: _FakeRuns) -> None:
    claim = PublishedClaim("test_accuracy", 0.870, "shifted", "planetoid-public")
    cert = ReproductionVerifier().certify(consistent, claim)  # type: ignore[arg-type]
    assert cert.decision is Decision.DISCREPANT
    assert cert.p_value < 0.05
    assert cert.effect_size == pytest.approx(1.0)  # claim above every seed


def test_systematically_offset_claim_is_caught_two_sided() -> None:
    # 50 seeds spread over [0.792, 0.808] (mean 0.800); claim 0.815 sits above
    # every one -> doubled-smaller-tail p = 2/51 ~ 0.039 -> DISCREPANT. The old
    # dispersion-style two-sided form returned REPRODUCED for this shape.
    runs = _FakeRuns(np.linspace(0.792, 0.808, 50))
    claim = PublishedClaim("test_accuracy", 0.815, "Kipf 2017", "planetoid-public")
    cert = ReproductionVerifier().certify(runs, claim)  # type: ignore[arg-type]
    assert cert.decision is Decision.DISCREPANT
    assert cert.p_value == pytest.approx(2 / 51)
    assert cert.standardized_gap < -1.0  # re-impl mean well below the claim


def test_split_mismatch_raises(consistent: _FakeRuns) -> None:
    claim = PublishedClaim("test_accuracy", 0.815, "s", "geom-gcn-split0")
    with pytest.raises(ValueError, match="split protocol mismatch"):
        ReproductionVerifier().certify(consistent, claim)  # type: ignore[arg-type]


@pytest.mark.parametrize("n", [8, 30, 39])
def test_below_minimum_seeds_is_inconclusive(n: int) -> None:
    runs = _FakeRuns(np.full(n, 0.80))
    claim = PublishedClaim("test_accuracy", 0.815, "s", "planetoid-public")
    cert = ReproductionVerifier().certify(runs, claim)  # type: ignore[arg-type]
    assert cert.decision is Decision.INCONCLUSIVE
    assert "too small" in cert.summary()


def test_equivalence_margin_produces_tost(consistent: _FakeRuns) -> None:
    claim = PublishedClaim("test_accuracy", 0.815, "s", "planetoid-public")
    cert = ReproductionVerifier().certify(
        consistent, claim, equivalence_margin_points=0.02
    )  # type: ignore[arg-type]
    assert cert.equivalence is not None
    assert cert.equivalence["equivalent"] is True
    assert "equivalence" in cert.summary()


def test_upper_direction_flags_underperformance() -> None:
    rng = np.random.default_rng(2)
    runs = _FakeRuns(np.clip(0.800 + 0.006 * rng.normal(size=25), 0, 1))  # n=25
    claim = PublishedClaim("test_accuracy", 0.815, "s", "planetoid-public")
    # one-sided UPPER floor is 1/26 < 0.05, so 25 seeds already resolve
    cert = ReproductionVerifier().certify(
        runs, claim, direction=TestDirection.UPPER
    )  # type: ignore[arg-type]
    assert cert.test_direction is TestDirection.UPPER
    assert cert.decision is Decision.DISCREPANT
    assert cert.p_floor == pytest.approx(1 / 26)


@pytest.fixture
def runs120() -> _FakeRuns:
    rng = np.random.default_rng(4)
    return _FakeRuns(np.clip(0.800 + 0.008 * rng.normal(size=120), 0, 1))


def test_aggregation_mean_on_target_reproduced(runs120: _FakeRuns) -> None:
    mean = float(runs120.metric_values("test_accuracy").mean())
    claim = PublishedClaim(
        "test_accuracy", round(mean, 4), "100-run mean", "planetoid-public",
        aggregation="mean", claimed_n_seeds=100,
    )
    cert = ReproductionVerifier().certify(runs120, claim, n_bootstrap=4000)  # type: ignore[arg-type]
    assert cert.decision is Decision.REPRODUCED
    ref = cert.aggregate_reference
    assert ref is not None
    assert ref["m"] == 100 and ref["n_bootstrap"] == 4000
    assert ref["ref_sd"] < 0.008 / 5  # an m-run mean is much tighter than a single run
    assert cert.p_floor == pytest.approx(2 / 4001)


def test_aggregation_mean_small_offset_is_discrepant(runs120: _FakeRuns) -> None:
    mean = float(runs120.metric_values("test_accuracy").mean())
    # +0.005 is < 1 single-run SD but many m-run-mean SDs away
    claim = PublishedClaim(
        "test_accuracy", round(mean + 0.005, 4), "100-run mean", "planetoid-public",
        aggregation="mean", claimed_n_seeds=100,
    )
    cert = ReproductionVerifier().certify(runs120, claim, n_bootstrap=4000)  # type: ignore[arg-type]
    assert cert.decision is Decision.DISCREPANT
    assert abs(cert.standardized_gap) < 1.0  # still inside the single-run spread


def test_aggregation_mean_too_few_seeds_inconclusive() -> None:
    runs = _FakeRuns(np.full(50, 0.80))
    claim = PublishedClaim(
        "test_accuracy", 0.80, "100-run mean", "planetoid-public",
        aggregation="mean", claimed_n_seeds=100,
    )
    cert = ReproductionVerifier().certify(runs, claim)  # type: ignore[arg-type]
    assert cert.decision is Decision.INCONCLUSIVE
    assert "100-run-mean reference" in cert.summary()


def test_aggregation_mean_requires_claimed_n_seeds(runs120: _FakeRuns) -> None:
    claim = PublishedClaim(
        "test_accuracy", 0.80, "mean", "planetoid-public", aggregation="mean",
    )
    with pytest.raises(ValueError, match="claimed_n_seeds"):
        ReproductionVerifier().certify(runs120, claim)  # type: ignore[arg-type]


def test_hold_one_seed_out_calibration() -> None:
    """Each seed value, treated as a 'claim' against the other 44, should almost
    always reproduce (the held-out draw is exchangeable with the rest)."""
    rng = np.random.default_rng(7)
    acc = np.clip(0.805 + 0.004 * rng.normal(size=45), 0, 1)
    reproduced = 0
    for i in range(45):
        rest = _FakeRuns(np.delete(acc, i))  # n = 44 -> two-sided floor 2/45 < 0.05
        claim = PublishedClaim("test_accuracy", float(acc[i]), "loo", "planetoid-public")
        cert = ReproductionVerifier().certify(rest, claim)  # type: ignore[arg-type]
        reproduced += cert.decision is Decision.REPRODUCED
    assert reproduced >= 43  # ~5% false-discrepant at most
