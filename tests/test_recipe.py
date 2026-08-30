"""Recipe: descriptor, override whitelist, canonical hash, resolution (WP1)."""

from __future__ import annotations

import pytest

from seedcert.recipe import OVERRIDABLE_FIELDS, Recipe, resolve_hyperparameters


def test_descriptor_shape() -> None:
    r = Recipe("gcn", overrides={"hidden_dim": 16}, label="kipf")
    d = r.descriptor()
    assert d == {"backbone": "gcn", "overrides": {"hidden_dim": 16}, "label": "kipf"}


def test_whitelist_contains_expected_axes() -> None:
    for field in ("hidden_dim", "num_layers", "dropout", "gat_heads", "sage_aggregator"):
        assert field in OVERRIDABLE_FIELDS


def test_hash_is_order_independent_and_32_hex() -> None:
    a = Recipe("gcn", overrides={"hidden_dim": 16, "dropout": 0.5})
    b = Recipe("gcn", overrides={"dropout": 0.5, "hidden_dim": 16})
    assert a.recipe_hash() == b.recipe_hash()
    assert len(a.recipe_hash()) == 32 and all(c in "0123456789abcdef" for c in a.recipe_hash())


def test_hash_ignores_label_but_tracks_backbone_and_values() -> None:
    assert Recipe("gcn", label="x").recipe_hash() == Recipe("gcn", label="y").recipe_hash()
    assert Recipe("gcn").recipe_hash() != Recipe("gat").recipe_hash()
    assert (
        Recipe("gcn", overrides={"hidden_dim": 16}).recipe_hash()
        != Recipe("gcn", overrides={"hidden_dim": 32}).recipe_hash()
    )


def test_non_whitelisted_override_rejected() -> None:
    with pytest.raises(ValueError, match="not permitted"):
        Recipe("gcn", overrides={"optimizer": "sgd"}).validate()


def test_unknown_backbone_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backbone"):
        Recipe("mlp").validate()


def test_resolve_applies_overrides_and_leaves_rest_fixed() -> None:
    hp = resolve_hyperparameters(Recipe("gcn", overrides={"hidden_dim": 16}))
    assert hp.hidden_dim == 16
    assert hp.lr == 0.01 and hp.dropout == 0.5  # untouched defaults


def test_resolve_no_overrides_returns_base() -> None:
    from seedcert.models.config import HYPERPARAMETERS

    assert resolve_hyperparameters(Recipe("gcn")) is HYPERPARAMETERS
