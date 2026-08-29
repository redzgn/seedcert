"""2-layer GCN backbone (primary; DESIGN Sec 8.5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """Two ``GCNConv`` layers, ReLU + dropout between them, raw logits out.

    Self-loops and symmetric normalization are done inside ``GCNConv`` (its
    defaults), not in the stored graph. Returns ``[N, out_dim]`` logits.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Raw logits for every node in ``x``."""
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits: torch.Tensor = self.conv2(x, edge_index)
        return logits
