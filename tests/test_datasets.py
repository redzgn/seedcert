"""Canonical-loader metadata is final in Stage 1; the network-touching
``load_canonical`` path is exercised in the reproduction anchor test."""

from __future__ import annotations

from seedcert.data.datasets import (
    CANONICAL_SOURCES,
    SPLIT_PROTOCOLS,
    split_protocol_for,
)
from seedcert.experiment.grid import GRID_DATASETS


def test_every_grid_dataset_has_a_source_and_protocol() -> None:
    for name in GRID_DATASETS:
        assert name in CANONICAL_SOURCES, name
        assert name in SPLIT_PROTOCOLS, name


def test_split_protocol_for_index_substitution() -> None:
    assert split_protocol_for("Cora") == "planetoid-public"
    assert split_protocol_for("Actor") == "geom-gcn-split0"
    assert split_protocol_for("Actor", split_index=3) == "geom-gcn-split3"
    # planetoid public has no per-index variant
    assert split_protocol_for("Cora", split_index=2) == "planetoid-public"


def test_mixhop_absent() -> None:
    assert "MixHop" not in GRID_DATASETS
    assert "MixHop" not in CANONICAL_SOURCES
