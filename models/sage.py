"""2-layer GraphSAGE backbone (reduced grid; DESIGN Sec 8.5, Sec 9.10)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    """Two ``SAGEConv`` layers with the mean aggregator, ReLU + dropout between
    them. Returns raw ``[N, out_dim]`` logits.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        aggregator: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr=aggregator)
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr=aggregator)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Raw logits for every node in ``x``."""
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits: torch.Tensor = self.conv2(x, edge_index)
        return logits
