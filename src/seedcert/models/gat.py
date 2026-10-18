"""2-layer GAT backbone (reduced grid; DESIGN Sec 8.5, Sec 9.10)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """Two ``GATConv`` layers. Layer 1: ``heads`` x ``head_dim`` (concat -> 64 by
    default). Layer 2: ``out_dim`` with a single averaged head. ELU + dropout
    between; dropout also on the input features and inside attention. Returns raw
    ``[N, out_dim]`` logits.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int,
        head_dim: int,
        concat: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        hidden = head_dim * heads if concat else head_dim
        self.conv1 = GATConv(in_dim, head_dim, heads=heads, concat=concat, dropout=dropout)
        self.conv2 = GATConv(hidden, out_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Raw logits for every node in ``x``."""
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits: torch.Tensor = self.conv2(x, edge_index)
        return logits
