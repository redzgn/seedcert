"""GNN backbones. Fixed architecture and hyperparameters; no tuning (DESIGN Sec 8.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seedcert.models.config import HYPERPARAMETERS, Hyperparameters
from seedcert.models.gat import GAT
from seedcert.models.gcn import GCN
from seedcert.models.sage import GraphSAGE

if TYPE_CHECKING:
    import torch

BACKBONE_REGISTRY: dict[str, type[torch.nn.Module]] = {"gcn": GCN, "gat": GAT, "sage": GraphSAGE}

__all__ = [
    "GCN",
    "GAT",
    "GraphSAGE",
    "BACKBONE_REGISTRY",
    "Hyperparameters",
    "HYPERPARAMETERS",
    "build_backbone",
]


def build_backbone(
    name: str,
    *,
    in_dim: int,
    out_dim: int,
    hp: Hyperparameters = HYPERPARAMETERS,
) -> torch.nn.Module:
    """Instantiate a 2-layer backbone by registry name with the fixed
    hyperparameters. GAT uses ``heads * head_dim == hidden_dim``; GCN and SAGE
    use ``hidden_dim`` directly.

    Raises:
        KeyError: if ``name`` is unknown.
    """
    key = name.lower()
    if key == "gcn":
        return GCN(in_dim, hp.hidden_dim, out_dim, dropout=hp.dropout)
    if key == "gat":
        return GAT(
            in_dim,
            out_dim,
            heads=hp.gat_heads,
            head_dim=hp.gat_head_dim,
            concat=hp.gat_concat,
            dropout=hp.dropout,
        )
    if key == "sage":
        return GraphSAGE(
            in_dim, hp.hidden_dim, out_dim, aggregator=hp.sage_aggregator, dropout=hp.dropout
        )
    raise KeyError(f"unknown backbone {name!r}")
