"""Canonical published-split dataset loader - the PRIMARY data path (DESIGN D2).

A reproduction is run under the split protocol the published claim names, so this
module loads each dataset with its canonical (published) split. It downloads to
:data:`CANONICAL_SPLITS_ROOT`, outside this repository. ``seedcert`` never writes
into a dataset directory.

``split_protocol_for(name)`` returns the stable string recorded in every
``RunKey`` and asserted against ``PublishedClaim.split_protocol``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from seedcert.data.bundle import GraphData

CANONICAL_SPLITS_ROOT = Path("/work/graph-experiments/canonical-splits")

#: dataset -> (torch_geometric loader family, note on the split used)
CANONICAL_SOURCES: dict[str, tuple[str, str]] = {
    "Cora": ("Planetoid", "split='public' (20 per class / 500 val / 1000 test)"),
    "CiteSeer": ("Planetoid", "split='public'"),
    "PubMed": ("Planetoid", "split='public'"),
    "Actor": ("Actor", "Geom-GCN 10 splits, index selects one"),
    "Chameleon": ("WikipediaNetwork", "Geom-GCN 10 splits (geom_gcn_preprocess=True)"),
    "Squirrel": ("WikipediaNetwork", "Geom-GCN 10 splits (geom_gcn_preprocess=True)"),
    "Roman-Empire": ("HeterophilousGraphDataset", "10 fixed splits, index selects one"),
    "Amazon-Ratings": ("HeterophilousGraphDataset", "10 fixed splits"),
    "Minesweeper": ("HeterophilousGraphDataset", "10 fixed splits"),
    "Tolokers": ("HeterophilousGraphDataset", "10 fixed splits"),
    "Questions": ("HeterophilousGraphDataset", "10 fixed splits"),
    "WikiCS": ("WikiCS", "20 standard training splits, index selects one"),
}

#: dataset -> the stable ``split_protocol`` string carried in ``RunKey`` and
#: matched against ``PublishedClaim.split_protocol`` (DESIGN D2, Sec 6).
SPLIT_PROTOCOLS: dict[str, str] = {
    "Cora": "planetoid-public",
    "CiteSeer": "planetoid-public",
    "PubMed": "planetoid-public",
    "Actor": "geom-gcn-split0",
    "Chameleon": "geom-gcn-split0",
    "Squirrel": "geom-gcn-split0",
    "Roman-Empire": "heterophilous-split0",
    "Amazon-Ratings": "heterophilous-split0",
    "Minesweeper": "heterophilous-split0",
    "Tolokers": "heterophilous-split0",
    "Questions": "heterophilous-split0",
    "WikiCS": "wikics-split0",
}


def split_protocol_for(name: str, *, split_index: int = 0) -> str:
    """Return the ``split_protocol`` string for ``name`` (DESIGN D2).

    For the multi-split datasets the trailing ``0`` reflects ``split_index``;
    passing a non-zero index substitutes it (e.g. ``geom-gcn-split3``).

    Raises:
        KeyError: ``name`` has no canonical split source.
    """
    base = SPLIT_PROTOCOLS[name]
    if split_index and base.endswith("split0"):
        return base[:-1] + str(split_index)
    return base


def _pick_split(mask: torch.Tensor, split_index: int) -> torch.Tensor:
    """Select column ``split_index`` from a multi-split mask, or pass a 1-D mask
    through unchanged."""
    return mask.bool() if mask.dim() == 1 else mask[:, split_index].bool()


def load_canonical(
    name: str,
    *,
    split_index: int = 0,
    root: Path = CANONICAL_SPLITS_ROOT,
) -> GraphData:
    """Load ``name`` with its canonical split ``split_index`` as a
    :class:`~seedcert.data.bundle.GraphData` (edges canonicalized the same
    way as the bundle path). Downloads on first use.

    Raises:
        KeyError: if ``name`` has no canonical source (e.g. MixHop).
    """
    from torch_geometric.datasets import (
        Actor,
        HeterophilousGraphDataset,
        Planetoid,
        WikiCS,
        WikipediaNetwork,
    )

    from seedcert.data.bundle import GraphData, canonicalize_edges
    from seedcert.data.manifest import resolve_num_classes

    if name not in CANONICAL_SOURCES:
        raise KeyError(f"no canonical split source for {name!r}")
    family = CANONICAL_SOURCES[name][0]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if family == "Planetoid":
        data = Planetoid(root=str(root / "Planetoid"), name=name, split="public")[0]
    elif family == "Actor":
        data = Actor(root=str(root / "Actor"))[0]
    elif family == "WikipediaNetwork":
        data = WikipediaNetwork(
            root=str(root / "WikipediaNetwork"), name=name.lower(), geom_gcn_preprocess=True
        )[0]
    elif family == "WikiCS":
        data = WikiCS(root=str(root / "WikiCS"), is_undirected=True)[0]
    elif family == "HeterophilousGraphDataset":
        data = HeterophilousGraphDataset(root=str(root / "Heterophilous"), name=name)[0]
    else:  # pragma: no cover - guarded by CANONICAL_SOURCES
        raise KeyError(family)

    x = data.x.float()
    y = data.y.view(-1).long()
    num_nodes = int(x.size(0))
    edge_index, info = canonicalize_edges(data.edge_index.long(), num_nodes)
    return GraphData(
        name=name,
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=_pick_split(data.train_mask, split_index),
        val_mask=_pick_split(data.val_mask, split_index),
        test_mask=_pick_split(data.test_mask, split_index),
        num_nodes=num_nodes,
        num_features=int(x.size(1)),
        num_classes=resolve_num_classes(y),
        num_edges_raw=int(info["e_raw"]),
        num_edges_canonical=int(info["e_canonical"]),
        edge_count_changed=bool(info["changed"]),
        was_symmetric=bool(info["was_symmetric"]),
    )
