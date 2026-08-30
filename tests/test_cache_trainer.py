"""train_run caches [N, C] logits + the four metrics (DESIGN Sec 5, WP1).

Trains on the tiny synthetic graph on CPU - fast, no download."""

from __future__ import annotations

import numpy as np

from seedcert.cache.spec import RunKey
from seedcert.cache.trainer import train_run
from seedcert.models.config import HYPERPARAMETERS
from seedcert.recipe import Recipe
from seedcert.verifiers.metrics import accuracy


def _key(recipe: Recipe, seed: int = 0) -> RunKey:
    return RunKey("Toy", "planetoid-public", recipe.backbone, recipe.recipe_hash(), seed)


def test_train_run_caches_logits_and_metrics(tiny_graph) -> None:
    recipe = Recipe("gcn")
    art = train_run(
        key=_key(recipe),
        data=tiny_graph,
        recipe=recipe,
        hp=HYPERPARAMETERS,
        device="cpu",
        dataset_sha256="deadbeef",
    )
    assert art.logits.shape == (tiny_graph.num_nodes, tiny_graph.num_classes)
    assert art.logits.dtype == np.float32
    for m in ("test_accuracy", "precision", "recall", "f1", "val_accuracy"):
        assert m in art.metrics and 0.0 <= float(art.metrics[m]) <= 1.0
    assert art.metrics["split_protocol"] == "planetoid-public"
    assert art.metrics["recipe_hash"] == recipe.recipe_hash()


def test_cached_metrics_recomputed_from_logits_match(tiny_graph) -> None:
    recipe = Recipe("gcn")
    art = train_run(
        key=_key(recipe),
        data=tiny_graph,
        recipe=recipe,
        hp=HYPERPARAMETERS,
        device="cpu",
        dataset_sha256="x",
    )
    y = tiny_graph.y.numpy()
    test_m = tiny_graph.test_mask.numpy()
    assert accuracy(art.logits, y, test_m) == float(art.metrics["test_accuracy"])


def test_content_hash_in_metrics_matches_key(tiny_graph) -> None:
    recipe = Recipe("gcn", overrides={"hidden_dim": 16})
    key = _key(recipe, seed=3)
    art = train_run(
        key=key,
        data=tiny_graph,
        recipe=recipe,
        hp=HYPERPARAMETERS,
        device="cpu",
        dataset_sha256="abc123",
    )
    assert art.metrics["content_hash"] == key.content_hash(dataset_sha256="abc123")


def test_overrides_reach_training(tiny_graph) -> None:
    # hidden_dim override changes the model; the run should still complete and
    # the recipe hash should differ from the default.
    r_default = Recipe("gcn")
    r_wide = Recipe("gcn", overrides={"hidden_dim": 8})
    assert r_default.recipe_hash() != r_wide.recipe_hash()
    art = train_run(
        key=_key(r_wide),
        data=tiny_graph,
        recipe=r_wide,
        hp=HYPERPARAMETERS,
        device="cpu",
        dataset_sha256="x",
    )
    # first GCN layer weight has hidden_dim rows
    sd = art.state_dict
    lin_key = next(k for k in sd if k.endswith("lin.weight"))
    assert sd[lin_key].shape[0] == 8
