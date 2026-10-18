"""Fixed-recipe node-classification training loop (DESIGN D8).

Trains whatever graph and masks it is handed. Used by the reproduction sanity
checks and by :mod:`seedcert.cache.trainer`.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import torch
import torch.nn.functional as F

from seedcert.models import build_backbone
from seedcert.models.config import HYPERPARAMETERS, Hyperparameters

if TYPE_CHECKING:
    import numpy as np

    from seedcert.data.bundle import GraphData
    from seedcert.data.graph_ops import RetainGraph

_HistoryRow = tuple[int, float, float, float]  # epoch, train_loss, val_loss, val_acc

#: Accepted by :func:`train_node_classifier` - the full graph or a post-deletion
#: retain graph; both expose x / edge_index / y / masks / num_features /
#: num_classes.
GraphLike = Union["GraphData", "RetainGraph"]


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Outcome of one training run, with best-epoch weights restored."""

    backbone: str
    train_acc: float
    val_acc: float
    test_acc: float
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    wall_clock_s: float
    history: tuple[_HistoryRow, ...]
    logits: np.ndarray  # [N, C] forward on the graph passed in, eval mode, best weights
    state_dict: dict[str, torch.Tensor]  # best-epoch weights, on CPU


def _evaluate(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        out: dict[str, float] = {}
        for split, mask in masks.items():
            out[f"{split}_loss"] = float(F.cross_entropy(logits[mask], y[mask]))
            out[f"{split}_acc"] = float((logits[mask].argmax(dim=1) == y[mask]).float().mean())
    return logits, out


def train_node_classifier(
    data: GraphLike,
    *,
    backbone: str = "gcn",
    hp: Hyperparameters = HYPERPARAMETERS,
    device: str | None = None,
    seed: int | None = None,
) -> TrainResult:
    """Train a 2-layer backbone with the fixed recipe: Adam, ``max_epochs``,
    early stopping on validation loss (patience ``hp.patience``), best-epoch
    weights restored. ``logits`` in the result is a full-graph forward pass in
    eval mode with the restored weights."""
    if seed is not None:
        from seedcert.rng import seed_everything

        seed_everything(seed)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x = data.x.to(dev)
    edge_index = data.edge_index.to(dev)
    y = data.y.to(dev)
    masks = {
        "train": data.train_mask.to(dev),
        "val": data.val_mask.to(dev),
        "test": data.test_mask.to(dev),
    }

    model = build_backbone(
        backbone, in_dim=data.num_features, out_dim=data.num_classes, hp=hp
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[_HistoryRow] = []
    epochs_run = 0
    t0 = time.perf_counter()

    for epoch in range(1, hp.max_epochs + 1):
        epochs_run = epoch
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]])
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()

        _, ev = _evaluate(model, x, edge_index, y, masks)
        history.append((epoch, float(loss), ev["val_loss"], ev["val_acc"]))

        if ev["val_loss"] < best_val_loss - 1e-6:
            best_val_loss = ev["val_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif hp.restore_best_weights and (epoch - best_epoch) >= hp.patience:
            break

    model.load_state_dict(best_state)
    final_logits, ev = _evaluate(model, x, edge_index, y, masks)
    return TrainResult(
        backbone=backbone,
        train_acc=ev["train_acc"],
        val_acc=ev["val_acc"],
        test_acc=ev["test_acc"],
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        wall_clock_s=time.perf_counter() - t0,
        history=tuple(history),
        logits=final_logits.detach().cpu().numpy(),
        state_dict={k: v.detach().cpu().clone() for k, v in best_state.items()},
    )
