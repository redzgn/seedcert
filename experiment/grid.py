"""The reproduction grid, enumerated (DESIGN Sec 7).

This module only *describes* the grid; it does not run anything. A
``ReproTarget`` binds a recipe (backbone + paper-matching overrides) to the
dataset and the published claim it is tested against.

Stage 1: the dataset list and ``ReproTarget`` shape are final;
:func:`repro_targets` returns the enumerated list from :data:`PUBLISHED_CLAIMS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from seedcert.claim import PublishedClaim
from seedcert.data.datasets import split_protocol_for
from seedcert.experiment.published_claims import PUBLISHED_CLAIMS, overrides_for
from seedcert.recipe import Recipe

#: Datasets with a usable canonical split and at least one published GCN/GAT/SAGE
#: number (DESIGN Sec 7). The synthetic MixHop is deliberately absent.
GRID_DATASETS: tuple[str, ...] = (
    "Cora",
    "CiteSeer",
    "PubMed",
    "Actor",
    "Chameleon",
    "Squirrel",
    "Roman-Empire",
    "Amazon-Ratings",
    "Minesweeper",
    "Tolokers",
    "Questions",
    "WikiCS",
)

#: Seed counts. The single-run two-sided rank test floors at 2/(n+1), so it needs
#: minimum_seeds(0.05) == 40. The demonstrated claims are 100-run means
#: (aggregation="mean"), which needs n >= 100 for the matched m-run-mean
#: reference (DESIGN D4, D10); the default matches that.
DEFAULT_N_SEEDS: int = 100
MIN_N_SEEDS: int = 40

@dataclass(frozen=True, slots=True)
class ReproTarget:
    """One cell of the reproduction sweep."""

    dataset: str
    backbone: str
    claim: PublishedClaim
    overrides: dict[str, Any] = field(default_factory=dict)
    n_seeds: int = DEFAULT_N_SEEDS  # DESIGN D4: >= MIN_N_SEEDS to resolve two-sided

    def recipe(self) -> Recipe:
        """The :class:`~seedcert.recipe.Recipe` for this target."""
        return Recipe(
            backbone=self.backbone,
            overrides=dict(self.overrides),
            label=f"{self.backbone}:{self.dataset}",
        )

    def split_protocol(self) -> str:
        return split_protocol_for(self.dataset)


def repro_targets(*, n_seeds: int = DEFAULT_N_SEEDS) -> list[ReproTarget]:
    """Enumerate one :class:`ReproTarget` per ``(backbone, dataset)`` in
    :data:`PUBLISHED_CLAIMS` whose dataset is in :data:`GRID_DATASETS`, sorted by
    ``(backbone, dataset)``.

    Raises:
        ValueError: a claim's ``split_protocol`` does not match the dataset's
            canonical protocol (a table error - ``seedcert`` will not certify
            across a split-protocol mismatch, DESIGN D2).
    """
    targets: list[ReproTarget] = []
    for (backbone, dataset), claim in sorted(PUBLISHED_CLAIMS.items()):
        if dataset not in GRID_DATASETS:
            continue
        canonical = split_protocol_for(dataset)
        if claim.split_protocol != canonical:
            raise ValueError(
                f"PUBLISHED_CLAIMS[({backbone!r}, {dataset!r})].split_protocol "
                f"{claim.split_protocol!r} != canonical {canonical!r}"
            )
        targets.append(
            ReproTarget(
                dataset=dataset,
                backbone=backbone,
                claim=claim,
                overrides=overrides_for(backbone, dataset),
                n_seeds=n_seeds,
            )
        )
    return targets


def run_count(targets: list[ReproTarget]) -> int:
    """Total model trainings implied by ``targets`` (``sum(t.n_seeds)``)."""
    return sum(t.n_seeds for t in targets)
