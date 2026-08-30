"""Certificate construction contract (DESIGN Sec 2.2, WP2)."""

from __future__ import annotations

import dataclasses

import pytest

from seedcert.certificate import Certificate, Decision, TestDirection
from seedcert.recipe import Recipe

_RECIPE = Recipe("gcn", overrides={"hidden_dim": 16}, label="kipf")
_N = 50  # two-sided floor 2/51 ~ 0.039 < alpha 0.05


def _kw(**over: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        schema_version="1",
        verifier_name="reproduction",
        verifier_version="0.1.0",
        created_at="2026-01-01T00:00:00Z",
        env={},
        wall_clock_s=1.0,
        dataset="Cora",
        split_protocol="planetoid-public",
        recipe=_RECIPE.descriptor(),
        recipe_hash=_RECIPE.recipe_hash(),
        n_seeds=_N,
        seed_list=tuple(range(_N)),
        metric_name="test_accuracy",
        claim={
            "metric": "test_accuracy",
            "value": 0.815,
            "source": "Kipf & Welling 2017",
            "split_protocol": "planetoid-public",
        },
        statistic=0.815,
        null_distribution=tuple([0.805] * _N),
        p_value=0.30,
        p_floor=2.0 / (_N + 1),
        test_direction=TestDirection.TWO_SIDED,
        reimpl_mean=0.805,
        reimpl_ci=(0.800, 0.810),
        effect_size=0.4,
        effect_size_ci=(-0.1, 0.8),
        standardized_gap=-2.0,
        ci_level=0.95,
        alpha=0.05,
        n_bootstrap=1000,
        decision=Decision.REPRODUCED,
        assumptions=("a",),
        assumptions_checked={"a": None},
    )
    base.update(over)
    return base


def test_enums_and_helpers_are_final() -> None:
    assert {d.value for d in Decision} == {"reproduced", "discrepant", "inconclusive"}
    assert TestDirection.TWO_SIDED.value == "two_sided"
    assert "NOT proof of equality" in str(Decision.REPRODUCED)


def test_valid_certificate_constructs_and_is_frozen() -> None:
    cert = Certificate(**_kw())  # type: ignore[arg-type]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cert.p_value = 0.9  # type: ignore[misc]


def test_float_raises() -> None:
    with pytest.raises(TypeError, match="not a number"):
        float(Certificate(**_kw()))  # type: ignore[arg-type]


def test_summary_is_multiline_string() -> None:
    text = Certificate(**_kw()).summary()  # type: ignore[arg-type]
    assert "decision: reproduced" in text
    assert "planetoid-public" in text
    assert "assumptions (1)" in text


def test_json_roundtrip() -> None:
    cert = Certificate(**_kw())  # type: ignore[arg-type]
    assert Certificate.from_json(cert.to_json()) == cert


def test_two_sided_p_floor_is_2_over_n_plus_1() -> None:
    with pytest.raises(ValueError, match=r"2/\(n_seeds\+1\)"):
        Certificate(**_kw(p_floor=1.0 / (_N + 1)))  # type: ignore[arg-type]
    # one-sided certificate uses the 1/(n+1) floor
    Certificate(
        **_kw(test_direction=TestDirection.UPPER, p_floor=1.0 / (_N + 1))  # type: ignore[arg-type]
    )


def test_p_below_floor_rejected() -> None:
    with pytest.raises(ValueError, match="below p_floor"):
        Certificate(**_kw(p_value=0.001))  # type: ignore[arg-type]


def test_fix1_floor_forces_inconclusive() -> None:
    # n_seeds=20, two-sided -> p_floor = 2/21 ~ 0.095 >= alpha 0.05
    kw = dict(
        n_seeds=20,
        seed_list=tuple(range(20)),
        null_distribution=tuple([0.8] * 20),
        p_floor=2.0 / 21,
        p_value=0.5,
    )
    with pytest.raises(ValueError, match="INCONCLUSIVE"):
        Certificate(**_kw(**kw, decision=Decision.REPRODUCED))  # type: ignore[arg-type]
    Certificate(**_kw(**kw, decision=Decision.INCONCLUSIVE))  # type: ignore[arg-type]


def test_decision_must_agree_with_p() -> None:
    with pytest.raises(ValueError, match="DISCREPANT"):
        Certificate(**_kw(decision=Decision.DISCREPANT, p_value=0.5))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="REPRODUCED"):
        Certificate(**_kw(decision=Decision.REPRODUCED, p_value=0.045))  # type: ignore[arg-type]


def test_recipe_hash_must_match_recipe() -> None:
    with pytest.raises(ValueError, match="recipe_hash"):
        Certificate(**_kw(recipe_hash="f" * 32))  # type: ignore[arg-type]


def test_claim_split_and_metric_must_match() -> None:
    with pytest.raises(ValueError, match="split_protocol"):
        Certificate(
            **_kw(
                claim={
                    "metric": "test_accuracy",
                    "value": 0.8,
                    "source": "s",
                    "split_protocol": "geom-gcn-split0",
                }
            )  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="metric"):
        Certificate(
            **_kw(
                claim={
                    "metric": "f1",
                    "value": 0.8,
                    "source": "s",
                    "split_protocol": "planetoid-public",
                }
            )  # type: ignore[arg-type]
        )


def test_seed_list_and_null_length_must_be_n_seeds() -> None:
    with pytest.raises(ValueError, match="seed_list"):
        Certificate(**_kw(seed_list=(0, 1, 2)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="null_distribution"):
        Certificate(**_kw(null_distribution=(0.8, 0.8)))  # type: ignore[arg-type]


def test_effect_ci_must_bracket_effect() -> None:
    with pytest.raises(ValueError, match="effect_size .* outside"):
        Certificate(**_kw(effect_size=0.9, effect_size_ci=(-0.1, 0.5)))  # type: ignore[arg-type]


_AGG = {"m": 100, "n_bootstrap": 10000, "ref_mean": 0.805, "ref_sd": 0.0008,
        "ref_ci": [0.8034, 0.8066], "min_seeds": 100}


def test_aggregate_reference_certificate_constructs_and_roundtrips() -> None:
    kw = _kw(
        claim={
            "metric": "test_accuracy", "value": 0.815, "source": "Kipf 2017",
            "split_protocol": "planetoid-public", "aggregation": "mean",
            "claimed_n_seeds": 100,
        },
        aggregate_reference=_AGG,
        p_floor=2.0 / 10001,      # k/(n_bootstrap+1), not k/(n_seeds+1)
        p_value=0.30,
        decision=Decision.REPRODUCED,
    )
    cert = Certificate(**kw)  # type: ignore[arg-type]
    assert Certificate.from_json(cert.to_json()) == cert
    assert "100-run-mean reference" in cert.summary()


def test_aggregate_reference_uses_bootstrap_floor_not_seed_floor() -> None:
    with pytest.raises(ValueError, match=r"n_bootstrap"):
        Certificate(
            **_kw(
                claim={
                    "metric": "test_accuracy", "value": 0.815, "source": "s",
                    "split_protocol": "planetoid-public", "aggregation": "mean",
                    "claimed_n_seeds": 100,
                },
                aggregate_reference=_AGG,
                p_floor=2.0 / (_N + 1),   # wrong: this is the seed floor
            )  # type: ignore[arg-type]
        )


def test_aggregate_reference_ref_ci_must_be_ordered() -> None:
    bad = dict(_AGG, ref_ci=[0.81, 0.80])
    with pytest.raises(ValueError, match="ref_ci not ordered"):
        Certificate(
            **_kw(
                claim={
                    "metric": "test_accuracy", "value": 0.815, "source": "s",
                    "split_protocol": "planetoid-public", "aggregation": "mean",
                    "claimed_n_seeds": 100,
                },
                aggregate_reference=bad,
                p_floor=2.0 / 10001,
            )  # type: ignore[arg-type]
        )
