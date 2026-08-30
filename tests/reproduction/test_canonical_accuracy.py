"""Sanity anchor: GCN on a canonical split reproduces literature accuracy
(distinct from the certified sweep). Needs a dataset download + training, so it
is opt-in: set ``SEEDCERT_RUN_REPRO_ANCHOR=1`` to run.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SEEDCERT_RUN_REPRO_ANCHOR") != "1",
    reason="set SEEDCERT_RUN_REPRO_ANCHOR=1 to run the download+train anchor",
)

# Expected GCN test accuracy on the Planetoid public split, hidden 16
# (Kipf & Welling 2017): Cora ~0.815, CiteSeer ~0.703, PubMed ~0.790.
EXPECTED = {"Cora": (0.79, 0.83), "CiteSeer": (0.66, 0.72), "PubMed": (0.76, 0.81)}


@pytest.mark.parametrize("dataset", list(EXPECTED))
def test_gcn_reproduces_literature(dataset: str) -> None:
    from seedcert.data.datasets import load_canonical
    from seedcert.models.config import HYPERPARAMETERS
    from seedcert.models.train import train_node_classifier

    data = load_canonical(dataset)
    hp = HYPERPARAMETERS
    accs = [
        train_node_classifier(data, backbone="gcn", hp=hp, device="cpu", seed=s).test_acc
        for s in range(3)
    ]
    lo, hi = EXPECTED[dataset]
    assert lo <= sum(accs) / len(accs) <= hi
