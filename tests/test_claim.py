"""``PublishedClaim`` is data-only and final in Stage 1 - these pass."""

from __future__ import annotations

from seedcert.claim import PublishedClaim


def test_roundtrip_json() -> None:
    c = PublishedClaim(
        metric="test_accuracy",
        value=0.815,
        source="Kipf & Welling 2017, Table 2",
        split_protocol="planetoid-public",
        claimed_sd=0.005,
    )
    back = PublishedClaim.from_json(c.to_json())
    assert back == c


def test_optional_fields_default_none() -> None:
    c = PublishedClaim("test_accuracy", 0.7, "src", "planetoid-public")
    assert c.aggregation == "single_run"
    assert c.claimed_sd is None
    assert c.claimed_n_seeds is None
    assert c.doi is None


def test_aggregation_field_roundtrips() -> None:
    c = PublishedClaim(
        "test_accuracy", 0.815, "Kipf 2017", "planetoid-public",
        aggregation="mean", claimed_n_seeds=100,
    )
    back = PublishedClaim.from_json(c.to_json())
    assert back == c
    assert back.aggregation == "mean" and back.claimed_n_seeds == 100


def test_from_dict_ignores_unknown_keys() -> None:
    payload = {
        "metric": "test_accuracy",
        "value": 0.8,
        "source": "s",
        "split_protocol": "planetoid-public",
        "surprise": 123,
    }
    c = PublishedClaim.from_dict(payload)
    assert c.value == 0.8
    assert not hasattr(c, "surprise")


def test_carries_split_protocol_and_metric() -> None:
    c = PublishedClaim("f1", 0.66, "s", "heterophilous-split0")
    d = c.to_dict()
    assert d["metric"] == "f1"
    assert d["split_protocol"] == "heterophilous-split0"
