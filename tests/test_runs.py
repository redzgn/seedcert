"""RecipeRuns reads one recipe's cached seed runs (DESIGN Sec 6, WP1).

The cache tree is written by hand so the test needs no training/download."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seedcert.cache.registry import RunRegistry
from seedcert.cache.runs import RecipeRuns
from seedcert.cache.spec import RecipeKey

_H = "c" * 32


def _write_run(root: Path, rk: RecipeKey, seed: int, *, test_acc: float, f1: float) -> None:
    rd = rk.with_seed(seed).run_dir(root)
    rd.mkdir(parents=True, exist_ok=True)
    np.save(rd / "logits.npy", np.zeros((5, 3), dtype=np.float32))
    (rd / "state_dict.pt").write_bytes(b"")
    (rd / "metrics.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "test_accuracy": test_acc,
                "precision": 0.7,
                "recall": 0.7,
                "f1": f1,
                "val_accuracy": 0.6,
            }
        )
    )
    (rd / "env.json").write_text(json.dumps({"gpu_model": "TestGPU"}))


@pytest.fixture
def runs(tmp_path: Path) -> RecipeRuns:
    reg = RunRegistry(tmp_path / "run_cache")
    rk = RecipeKey("Cora", "planetoid-public", "gcn", _H)
    (reg.root / rk.rel_path()).mkdir(parents=True, exist_ok=True)
    (reg.root / rk.rel_path() / "recipe.json").write_text(
        json.dumps({"backbone": "gcn", "overrides": {"hidden_dim": 16}, "label": "kipf"})
    )
    for s, (acc, f1) in enumerate([(0.80, 0.79), (0.81, 0.80), (0.79, 0.78)]):
        _write_run(reg.root, rk, s, test_acc=acc, f1=f1)
    return RecipeRuns(reg, rk)


def test_len_and_seed_list(runs: RecipeRuns) -> None:
    assert runs.n_seeds == len(runs) == 3
    assert runs.seed_list == (0, 1, 2)
    assert runs.dataset == "Cora"
    assert runs.split_protocol == "planetoid-public"


def test_metric_values_are_seed_ordered(runs: RecipeRuns) -> None:
    np.testing.assert_allclose(runs.metric_values("test_accuracy"), [0.80, 0.81, 0.79])
    np.testing.assert_allclose(runs.metric_values("f1"), [0.79, 0.80, 0.78])


def test_metric_values_unknown_key_raises(runs: RecipeRuns) -> None:
    with pytest.raises(KeyError):
        runs.metric_values("auc")


def test_logits_stack_shape(runs: RecipeRuns) -> None:
    assert runs.logits_stack().shape == (3, 5, 3)


def test_recipe_descriptor_from_file(runs: RecipeRuns) -> None:
    d = runs.recipe_descriptor()
    assert d["overrides"] == {"hidden_dim": 16}
    assert d["label"] == "kipf"


def test_empty_recipe_dir_metric_values_raises(tmp_path: Path) -> None:
    reg = RunRegistry(tmp_path / "run_cache")
    rr = RecipeRuns(reg, RecipeKey("Cora", "planetoid-public", "gcn", _H))
    assert rr.n_seeds == 0
    with pytest.raises(ValueError, match="no cached runs"):
        rr.metric_values("test_accuracy")
