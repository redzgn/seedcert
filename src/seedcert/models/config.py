"""The single frozen hyperparameter set shared by every backbone and every oracle.

These values are fixed by the experiment grid (DESIGN Sec 8.4) and must not be
tuned. Backbone-specific fields (``gat_*``, ``sage_aggregator``) are the only
per-architecture knobs and are also fixed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Hyperparameters:
    """Fixed training and architecture hyperparameters."""

    num_layers: int = 2
    hidden_dim: int = 64
    lr: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.5
    max_epochs: int = 200
    optimizer: str = "adam"
    early_stopping_metric: str = "val_loss"
    patience: int = 30
    restore_best_weights: bool = True

    # GAT: 8 heads x 8 dims, concatenated -> 64 (DESIGN Sec 9.10)
    gat_heads: int = 8
    gat_head_dim: int = 8
    gat_concat: bool = True

    # GraphSAGE
    sage_aggregator: str = "mean"


HYPERPARAMETERS = Hyperparameters()


def resolve_layer_shapes(
    backbone: str,
    *,
    in_dim: int,
    out_dim: int,
    hp: Hyperparameters = HYPERPARAMETERS,
) -> dict[str, int]:
    """Return the concrete per-layer channel counts for ``backbone``.

    For GAT, layer-1 output is ``gat_heads * gat_head_dim`` (== ``hidden_dim``);
    for GCN and SAGE it is ``hidden_dim``.

    Raises:
        KeyError: unknown ``backbone``.
    """
    key = backbone.lower()
    if key == "gat":
        hidden = hp.gat_heads * hp.gat_head_dim
    elif key in ("gcn", "sage"):
        hidden = hp.hidden_dim
    else:
        raise KeyError(f"unknown backbone {backbone!r}")
    return {"in_dim": in_dim, "hidden_dim": hidden, "out_dim": out_dim}
