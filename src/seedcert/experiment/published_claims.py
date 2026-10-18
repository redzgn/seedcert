"""``PUBLISHED_CLAIMS`` - the table of reported numbers the reproduction sweep
tests against (DESIGN Sec 7).

v1 scope: only numbers measured under a **single fixed split** ``seedcert`` can
reproduce exactly - the Planetoid ``public`` split. Numbers reported as a mean
over many random / rotated splits (Geom-GCN, Heterophilous, WikiCS) need the
multi-split aggregation deferred to a later version (DESIGN Sec 5), and are
excluded here rather than certified across a split-protocol mismatch.

``RECIPE_OVERRIDES`` sets each target's recipe to the paper's *architecture and
optimiser* settings. ``seedcert`` runs one fixed training loop; anything not
exposed as a hyperparameter (per-layer weight decay, initialisation, the exact
early-stopping window) is not matched, and that residual is part of what the
certificate measures - reproduction *under a stated recipe*, not a bit-for-bit
replay.
"""

from __future__ import annotations

from typing import Any

from seedcert.claim import PublishedClaim

_KIPF = "Kipf & Welling, ICLR 2017, Table 2"
_KIPF_DOI = "10.48550/arXiv.1609.02907"
_GAT = "Velickovic et al., ICLR 2018, Table 2"
_GAT_DOI = "10.48550/arXiv.1710.10903"

#: Both sources report the mean accuracy over 100 random-initialisation runs, so
#: every claim is ``aggregation="mean"`` with ``claimed_n_seeds=100`` (DESIGN D10).
_M = 100


def _kipf(value: float) -> PublishedClaim:
    return PublishedClaim(
        "test_accuracy", value, _KIPF, "planetoid-public",
        aggregation="mean", claimed_n_seeds=_M, doi=_KIPF_DOI,
    )


def _gat(value: float, sd: float) -> PublishedClaim:
    return PublishedClaim(
        "test_accuracy", value, _GAT, "planetoid-public",
        aggregation="mean", claimed_sd=sd, claimed_n_seeds=_M, doi=_GAT_DOI,
    )


#: ``(backbone, dataset) -> PublishedClaim``. All Planetoid ``public`` split.
PUBLISHED_CLAIMS: dict[tuple[str, str], PublishedClaim] = {
    ("gcn", "Cora"): _kipf(0.815),
    ("gcn", "CiteSeer"): _kipf(0.703),
    ("gcn", "PubMed"): _kipf(0.790),
    ("gat", "Cora"): _gat(0.830, 0.007),
    ("gat", "CiteSeer"): _gat(0.725, 0.007),
    ("gat", "PubMed"): _gat(0.790, 0.003),
}

#: ``(backbone, dataset) -> Hyperparameters overrides`` that put the fixed recipe
#: on the paper's architecture / optimiser (all keys are in
#: ``recipe.OVERRIDABLE_FIELDS``).
RECIPE_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    # Kipf GCN: 16 hidden units (vs seedcert's default 64).
    ("gcn", "Cora"): {"hidden_dim": 16},
    ("gcn", "CiteSeer"): {"hidden_dim": 16},
    ("gcn", "PubMed"): {"hidden_dim": 16},
    # Velickovic GAT: 8 heads x 8 (== seedcert default); Cora/CiteSeer use
    # dropout 0.6 and lr 0.005, PubMed dropout 0.6 and lr 0.01.
    ("gat", "Cora"): {"dropout": 0.6, "lr": 0.005},
    ("gat", "CiteSeer"): {"dropout": 0.6, "lr": 0.005},
    ("gat", "PubMed"): {"dropout": 0.6},
}


def claim_for(backbone: str, dataset: str) -> PublishedClaim | None:
    """Return the published claim for ``(backbone, dataset)`` or ``None``."""
    return PUBLISHED_CLAIMS.get((backbone, dataset))


def overrides_for(backbone: str, dataset: str) -> dict[str, Any]:
    """Return the recipe overrides for ``(backbone, dataset)`` (possibly empty)."""
    return dict(RECIPE_OVERRIDES.get((backbone, dataset), {}))
