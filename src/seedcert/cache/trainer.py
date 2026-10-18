"""One re-implementation run: train on the full graph, cache logits + metrics
(DESIGN Sec 5, Sec 6).

Adapted from ``certiforget``'s oracle trainer, minus the retain graph: every run
trains on the full training split of the full canonical graph, so the
``TrainResult`` logits are already the full-graph ``[N, C]`` matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from seedcert.cache.spec import RunKey
    from seedcert.data.bundle import GraphData
    from seedcert.models.config import Hyperparameters
    from seedcert.recipe import Recipe


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """The full output of one run, ready for the registry."""

    key: RunKey
    state_dict: dict[str, torch.Tensor]
    logits: np.ndarray  # [N, C] float32, full-graph eval with best weights
    metrics: dict[str, float | int | str]
    env: dict[str, object]
    seed_provenance: dict[str, object]


def train_run(
    *,
    key: RunKey,
    data: GraphData,
    recipe: Recipe,
    hp: Hyperparameters,
    device: str,
    dataset_sha256: str,
) -> RunArtifacts:
    """Train one run and score it.

    Assumptions:
        * ``resolve_hyperparameters(recipe, base=hp)`` gives the effective
          hyperparameters; no tuning. Early stopping on validation loss,
          best-epoch weights restored (``models.train.train_node_classifier``).
        * ``seed_everything(key.seed)`` is called once at entry; cuDNN stays
          non-deterministic so the seed distribution carries real training noise.
        * The cached ``logits`` are the ``TrainResult`` full-graph forward in
          eval mode with the restored weights; all four metrics
          (``test_accuracy`` / ``precision`` / ``recall`` / ``f1``) are computed
          from that cache on the canonical test mask, plus ``val_accuracy`` /
          ``train_accuracy`` for auditing.
        * Single GPU; ``env`` records ``gpu_model``.
    """
    from seedcert.env import capture_environment
    from seedcert.models.train import train_node_classifier
    from seedcert.recipe import resolve_hyperparameters
    from seedcert.rng import seed_everything
    from seedcert.verifiers.metrics import accuracy, macro_prf

    key.validate()
    recipe.validate()
    effective_hp = resolve_hyperparameters(recipe, base=hp)
    provenance = seed_everything(key.seed)

    result = train_node_classifier(
        data, backbone=key.backbone, hp=effective_hp, device=device
    )
    logits = np.asarray(result.logits, dtype=np.float32)

    y = data.y.detach().cpu().numpy()
    test_m = data.test_mask.detach().cpu().numpy().astype(bool)
    val_m = data.val_mask.detach().cpu().numpy().astype(bool)
    train_m = data.train_mask.detach().cpu().numpy().astype(bool)
    precision, recall, f1 = macro_prf(logits, y, test_m)

    metrics: dict[str, float | int | str] = {
        "key_string": key.key_string(),
        "content_hash": key.content_hash(dataset_sha256=dataset_sha256),
        "dataset": key.dataset,
        "split_protocol": key.split_protocol,
        "backbone": key.backbone,
        "recipe_hash": key.recipe_hash,
        "seed": key.seed,
        "test_accuracy": accuracy(logits, y, test_m),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "val_accuracy": accuracy(logits, y, val_m),
        "train_accuracy": accuracy(logits, y, train_m),
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "early_stop_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "wall_clock_s": result.wall_clock_s,
        "n_nodes": int(data.num_nodes),
        "n_classes": int(data.num_classes),
    }
    return RunArtifacts(
        key=key,
        state_dict=result.state_dict,
        logits=logits,
        metrics=metrics,
        env=capture_environment(),
        seed_provenance=asdict(provenance),
    )
