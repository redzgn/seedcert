"""Load datasets read-only from the frozen bundle and canonicalize their edges.

The bundle at :data:`BUNDLE_ROOT` is never written to. Its own loader
(``scripts.load_dataset.load_dataset``) is used - never a local ``torch.load`` -
and we only add edge canonicalization on top (DESIGN D7).
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

BUNDLE_ROOT = Path("/work/graph-data-protocol/datasets")
_PROTOCOL_ROOT = Path("/work/graph-data-protocol")

#: Grid datasets that are genuinely directed in the bundle - ``to_undirected``
#: changes their edge count (verified by WP1, DESIGN Sec 1 / paper-notes). NOTE:
#: an earlier recon wrongly also listed Questions and Roman-Empire; both are in
#: fact stored symmetric with no self-loops.
KNOWN_ASYMMETRIC: tuple[str, ...] = (
    "Actor",
    "Chameleon",
    "Squirrel",
)

#: Grid datasets that are already undirected but carry self-loops the bundle
#: stores and we strip - edge count changes, but ``was_symmetric`` is True.
KNOWN_SELF_LOOPED: tuple[str, ...] = ("WikiCS",)


@dataclass(frozen=True, slots=True)
class GraphData:
    """A canonicalized dataset: undirected edges (no self-loops - those are added
    at the model layer), the frozen 80/10/10 split masks, and the recorded effect
    of symmetrization."""

    name: str
    x: torch.Tensor  # [N, F] float32
    edge_index: torch.Tensor  # [2, E_canonical] long, undirected, coalesced, loop-free
    y: torch.Tensor  # [N] long
    train_mask: torch.Tensor  # [N] bool
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_nodes: int
    num_features: int
    num_classes: int
    num_edges_raw: int
    num_edges_canonical: int
    edge_count_changed: bool
    was_symmetric: bool


@functools.lru_cache(maxsize=1)
def _shared_loader() -> Callable[..., Any]:
    """Return the frozen bundle's own ``load_dataset`` (CLAUDE.md: never a local
    ``torch.load`` on a bundle ``.pt``)."""
    root = str(_PROTOCOL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts.load_dataset import load_dataset

    fn: Callable[..., Any] = load_dataset
    return fn


def canonicalize_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, dict[str, int | bool]]:
    """Return ``(canonical_edge_index, info)``.

    Canonical form: self-loops removed, made undirected, coalesced (dedup +
    sorted). Self-loops are *not* added here - ``GCNConv`` etc. add them at the
    layer. ``info`` carries ``e_raw``, ``e_canonical``, ``n_self_loops_removed``,
    ``changed`` (``e_canonical != e_raw``), and ``was_symmetric`` (was the *raw*
    graph already undirected).
    """
    from torch_geometric.utils import is_undirected, remove_self_loops, to_undirected

    e_raw = int(edge_index.size(1))
    was_symmetric = bool(is_undirected(edge_index, num_nodes=num_nodes))
    no_loops, _ = remove_self_loops(edge_index)
    n_self_loops = e_raw - int(no_loops.size(1))
    canonical = to_undirected(no_loops.contiguous(), num_nodes=num_nodes)
    e_canonical = int(canonical.size(1))
    info: dict[str, int | bool] = {
        "e_raw": e_raw,
        "e_canonical": e_canonical,
        "n_self_loops_removed": n_self_loops,
        "changed": e_canonical != e_raw,
        "was_symmetric": was_symmetric,
    }
    return canonical.long(), info


def _split_mask(mask: torch.Tensor, name: str, which: str) -> torch.Tensor:
    if mask.dim() == 2:
        raise ValueError(
            f"{name}: expected a 1-D {which}; got shape {tuple(mask.shape)}. The frozen "
            "bundle should ship a single 80/10/10 split (instruction.md Sec 3)."
        )
    return mask.bool()


def load_bundle_dataset(name: str, *, root: Path = BUNDLE_ROOT) -> GraphData:
    """Load one bundle dataset via its shared loader and canonicalize edges.

    Assumptions:
        * ``root`` is read-only; nothing is written under it.
        * ``num_classes`` is ``int(y.max()) + 1`` (:func:`resolve_num_classes`),
          not inherited from the bundle ``MANIFEST.json``.
        * Split masks are the bundle's frozen 80/10/10 seed-42 masks, 1-D bool.
    """
    from seedcert.data.manifest import resolve_num_classes

    data = _shared_loader()(str(root), name)
    x = data.x.float()
    y = data.y.view(-1).long()
    num_nodes = int(x.size(0))
    edge_index, info = canonicalize_edges(data.edge_index.long(), num_nodes)
    return GraphData(
        name=name,
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=_split_mask(data.train_mask, name, "train_mask"),
        val_mask=_split_mask(data.val_mask, name, "val_mask"),
        test_mask=_split_mask(data.test_mask, name, "test_mask"),
        num_nodes=num_nodes,
        num_features=int(x.size(1)),
        num_classes=resolve_num_classes(y),
        num_edges_raw=int(info["e_raw"]),
        num_edges_canonical=int(info["e_canonical"]),
        edge_count_changed=bool(info["changed"]),
        was_symmetric=bool(info["was_symmetric"]),
    )
