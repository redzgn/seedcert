"""RunKey path helpers + content hash + index rebuild (WP1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seedcert.cache.registry import RunRecord
from seedcert.cache.spec import RecipeKey, RunKey

_H = "a" * 32


def test_key_string_and_path_roundtrip() -> None:
    key = RunKey("Cora", "planetoid-public", "gcn", _H, 7)
    assert key.key_string() == f"Cora/planetoid-public/gcn/{_H}/seed7"
    assert key.without_seed() == RecipeKey("Cora", "planetoid-public", "gcn", _H)
    assert key.run_dir(Path("root")).as_posix() == f"root/Cora/planetoid-public/gcn/{_H}/seed7"
    assert key.recipe_dir(Path("root")).as_posix() == f"root/Cora/planetoid-public/gcn/{_H}"


def test_recipe_key_with_seed_is_inverse_of_without_seed() -> None:
    rk = RecipeKey("PubMed", "planetoid-public", "sage", "b" * 32)
    assert rk.with_seed(3).without_seed() == rk


def test_validate_rejects_bad_fields() -> None:
    with pytest.raises(ValueError, match="backbone"):
        RunKey("Cora", "planetoid-public", "mlp", _H, 0).validate()
    with pytest.raises(ValueError, match="recipe_hash"):
        RunKey("Cora", "planetoid-public", "gcn", "xyz", 0).validate()
    with pytest.raises(ValueError, match="seed"):
        RunKey("Cora", "planetoid-public", "gcn", _H, -1).validate()


def test_content_hash_depends_on_dataset_sha_and_key() -> None:
    key = RunKey("Cora", "planetoid-public", "gcn", _H, 0)
    assert key.content_hash(dataset_sha256="x") != key.content_hash(dataset_sha256="y")
    other = RunKey("Cora", "planetoid-public", "gcn", _H, 1)
    assert key.content_hash(dataset_sha256="x") != other.content_hash(dataset_sha256="x")
    assert len(key.content_hash(dataset_sha256="x")) == 32


def _fake_run(root: Path, key: RunKey, gpu: str = "TestGPU", test_acc: float = 0.8) -> None:
    rd = key.run_dir(root)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "state_dict.pt").write_bytes(b"")
    (rd / "logits.npy").write_bytes(b"")
    (rd / "metrics.json").write_text(
        json.dumps(
            {
                "key_string": key.key_string(),
                "content_hash": key.content_hash(dataset_sha256="s"),
                "dataset": key.dataset,
                "split_protocol": key.split_protocol,
                "backbone": key.backbone,
                "recipe_hash": key.recipe_hash,
                "seed": key.seed,
                "test_accuracy": test_acc,
                "precision": 0.79,
                "recall": 0.78,
                "f1": 0.785,
                "val_accuracy": 0.77,
                "early_stop_epoch": 42,
                "wall_clock_s": 1.5,
            }
        )
    )
    (rd / "env.json").write_text(json.dumps({"gpu_model": gpu}))


def test_rebuild_index_on_empty_tree(tmp_registry) -> None:
    tmp_registry.rebuild_index()
    assert tmp_registry.read_index().empty


def test_rebuild_index_collects_runs(tmp_registry) -> None:
    rk = RecipeKey("Cora", "planetoid-public", "gcn", _H)
    for s in range(3):
        _fake_run(tmp_registry.root, rk.with_seed(s))
    tmp_registry.rebuild_index()
    df = tmp_registry.read_index()
    assert len(df) == 3
    assert set(df["seed"]) == {0, 1, 2}
    assert list(df.columns) == list(RunRecord.__dataclass_fields__)


def test_rebuild_index_rejects_mixed_gpu_within_recipe(tmp_registry) -> None:
    rk = RecipeKey("Cora", "planetoid-public", "gcn", _H)
    _fake_run(tmp_registry.root, rk.with_seed(0), gpu="GPU-A")
    _fake_run(tmp_registry.root, rk.with_seed(1), gpu="GPU-B")
    with pytest.raises(RuntimeError, match="mixed GPU"):
        tmp_registry.rebuild_index()
