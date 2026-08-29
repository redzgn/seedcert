"""Graph surgery for full node deletion (DESIGN D1).

Forgotten nodes and every incident edge are removed; the oracle trains on the
induced subgraph over the survivors. Disconnection is measured and reported here,
not discovered downstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from seedcert.data.bundle import GraphData


@dataclass(frozen=True, slots=True)
class RetainGraph:
    """The induced subgraph after full node deletion, plus its topology report.

    Structurally a drop-in for :func:`seedcert.models.train.train_node_classifier`:
    ``x`` / ``edge_index`` / ``y`` / masks are over survivors ``0..n_retain-1``.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_features: int
    num_classes: int
    old_to_new: np.ndarray  # [N_full], -1 for deleted nodes
    new_to_old: np.ndarray  # [N_retain]
    n_retain: int
    e_before: int
    e_after: int
    n_components: int
    largest_cc_frac: float
    n_isolated_nodes: int


def degree(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Per-node degree, ``int64[num_nodes]``.

    For the canonical undirected form (every edge stored both ways) counting the
    ``row`` entries gives the true undirected degree; ``degree().sum()`` equals
    ``edge_index.size(1)`` (i.e. ``2 * |undirected edges|``).
    """
    row = edge_index[0].to(torch.int64)
    counts = torch.bincount(row, minlength=num_nodes)
    return counts.detach().cpu().numpy().astype(np.int64)


def component_stats(edge_index: torch.Tensor, num_nodes: int) -> dict[str, float | int]:
    """``n_components``, ``largest_cc_frac`` (of ``num_nodes``), ``n_isolated_nodes``
    for an undirected graph."""
    if num_nodes == 0:
        return {"n_components": 0, "largest_cc_frac": 0.0, "n_isolated_nodes": 0}

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ei = edge_index.detach().cpu().numpy()
    if ei.shape[1] == 0:
        return {
            "n_components": num_nodes,
            "largest_cc_frac": 1.0 / num_nodes,
            "n_isolated_nodes": num_nodes,
        }

    adj = coo_matrix(
        (np.ones(ei.shape[1], dtype=np.int8), (ei[0], ei[1])),
        shape=(num_nodes, num_nodes),
    )
    n_comp, labels = connected_components(adj, directed=False)
    largest = int(np.bincount(labels).max())
    deg = np.bincount(ei[0], minlength=num_nodes)
    return {
        "n_components": int(n_comp),
        "largest_cc_frac": float(largest / num_nodes),
        "n_isolated_nodes": int((deg == 0).sum()),
    }


def induced_retain_graph(data: GraphData, forget_indices: Sequence[int]) -> RetainGraph:
    """Delete ``forget_indices`` (canonical ids) and every incident edge; remap
    survivors to ``0..n_retain-1``.

    Raises:
        ValueError: a forget index is out of range, or is not a training node
            (violates D3).
    """
    n_full = data.num_nodes
    forget = np.asarray(sorted({int(i) for i in forget_indices}), dtype=np.int64)
    if forget.size and (forget.min() < 0 or forget.max() >= n_full):
        raise ValueError("forget index out of range")

    train_np = data.train_mask.detach().cpu().numpy().astype(bool)
    if forget.size and not train_np[forget].all():
        raise ValueError("forget_indices contains non-training nodes (violates D3)")

    keep = np.ones(n_full, dtype=bool)
    keep[forget] = False
    new_to_old = np.nonzero(keep)[0].astype(np.int64)
    n_retain = int(new_to_old.size)
    old_to_new = np.full(n_full, -1, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(n_retain, dtype=np.int64)

    ei = data.edge_index.detach().cpu().numpy()
    edge_keep = keep[ei[0]] & keep[ei[1]]
    new_ei = old_to_new[ei[:, edge_keep]]
    edge_index = torch.from_numpy(np.ascontiguousarray(new_ei)).long()

    keep_t = torch.from_numpy(keep)
    stats = component_stats(edge_index, n_retain)
    return RetainGraph(
        x=data.x[keep_t],
        edge_index=edge_index,
        y=data.y[keep_t],
        train_mask=data.train_mask[keep_t],
        val_mask=data.val_mask[keep_t],
        test_mask=data.test_mask[keep_t],
        num_features=data.num_features,
        num_classes=data.num_classes,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        n_retain=n_retain,
        e_before=int(data.edge_index.size(1)),
        e_after=int(edge_index.size(1)),
        n_components=int(stats["n_components"]),
        largest_cc_frac=float(stats["largest_cc_frac"]),
        n_isolated_nodes=int(stats["n_isolated_nodes"]),
    )
