"""Shared fixtures for the Stage 1 skeleton tests.

Data fixtures are real (test scaffolding); anything that touches package
algorithm logic goes through the ``NotImplementedError`` stubs, so the tests that
use them are marked ``xfail(strict=True)`` until Stage 2.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from seedcert.cache.registry import RunRegistry
from seedcert.claim import PublishedClaim
from seedcert.data.bundle import GraphData


@pytest.fixture
def tiny_graph() -> GraphData:
    """A 12-node, 3-class undirected toy graph with an 8/2/2 split."""
    n = 12
    ring = [(i, (i + 1) % n) for i in range(n)]
    chords = [(0, 6), (2, 9), (3, 8), (1, 7)]
    und = ring + chords + [(b, a) for a, b in ring + chords]
    edge_index = torch.tensor(und, dtype=torch.long).t().contiguous()
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(n, 5)), dtype=torch.float)
    y = torch.tensor([0, 1, 2] * 4, dtype=torch.long)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:8] = True
    val_mask[8:10] = True
    test_mask[10:] = True
    return GraphData(
        name="Toy",
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=n,
        num_features=5,
        num_classes=3,
        num_edges_raw=edge_index.size(1),
        num_edges_canonical=edge_index.size(1),
        edge_count_changed=False,
        was_symmetric=True,
    )


@pytest.fixture
def seed_metrics() -> np.ndarray:
    """A plausible vector of 20 per-seed test-accuracy values (mean ~0.805)."""
    rng = np.random.default_rng(1)
    return np.clip(0.805 + 0.004 * rng.normal(size=20), 0.0, 1.0)


@pytest.fixture
def sample_claim() -> PublishedClaim:
    return PublishedClaim(
        metric="test_accuracy",
        value=0.815,
        source="Kipf & Welling 2017, Table 2",
        split_protocol="planetoid-public",
        claimed_sd=0.005,
        claimed_n_seeds=100,
    )


@pytest.fixture
def tmp_registry(tmp_path) -> RunRegistry:
    """An empty :class:`RunRegistry` rooted at a temp dir."""
    return RunRegistry(tmp_path / "run_cache")
