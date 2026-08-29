"""Canonical assumption strings and the checks that populate
``Certificate.assumptions_checked`` (DESIGN Sec 2.1).

The strings are the deliverable text that appears in every certificate and in the
paper's methodology section; keep them stable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seedcert.cache.runs import RecipeRuns
    from seedcert.claim import PublishedClaim

RECIPE_MATCHES_CLAIM = (
    "The recipe (backbone, layer count, width, and every fixed hyperparameter) "
    "is the one the published value was measured under; any deviation is listed "
    "in the certificate's recipe.overrides."
)
SPLIT_PROTOCOL_MATCHES = (
    "The re-implementation was run under the split protocol the claim names; "
    "seedcert refuses to certify across a split-protocol mismatch."
)
METRIC_MATCHES = (
    "The certified metric and the claimed metric are the same quantity on the "
    "same scale (accuracy / macro precision / recall / F1 as a fraction)."
)
FROZEN_TEST_SPLIT = (
    "The metric is computed on the dataset's frozen canonical test nodes, read "
    "from each run's cached full-graph logits."
)
NO_POST_HOC_CALIBRATION = (
    "No temperature scaling or other post-hoc calibration is applied before the "
    "metric is computed."
)
SEEDS_ARE_IID = (
    "The n runs differ only in the random seed (initialisation and training "
    "stochasticity); their distribution - single runs, or the m-run-mean "
    "sampling distribution derived from them for an aggregate claim - is the "
    "reference the claim is tested against."
)
SAME_RANK_FUNCTIONAL = (
    "The published value and the reference distribution enter one rank "
    "function; no summary statistic substitutes for the distribution."
)
SINGLE_GPU_ACROSS_SEEDS = (
    "All n runs were trained on a single GPU model, recorded in each run's "
    "env.json; mixed-GPU recipes are rejected at index build."
)


def reproduction_assumptions() -> tuple[str, ...]:
    """The assumption strings attached to every reproduction certificate."""
    return (
        RECIPE_MATCHES_CLAIM,
        SPLIT_PROTOCOL_MATCHES,
        METRIC_MATCHES,
        FROZEN_TEST_SPLIT,
        NO_POST_HOC_CALIBRATION,
        SEEDS_ARE_IID,
        SAME_RANK_FUNCTIONAL,
        SINGLE_GPU_ACROSS_SEEDS,
    )


def check_reproduction_assumptions(
    runs: RecipeRuns,
    claim: PublishedClaim,
    *,
    metric_name: str,
) -> dict[str, bool | None]:
    """Verify the programmatically checkable assumptions; leave the rest ``None``.

    Checked: split-protocol match (``runs`` vs ``claim``), metric match, single
    GPU model across the seed runs, and that the rank functional is shared
    (always ``True`` by construction). ``RECIPE_MATCHES_CLAIM``,
    ``FROZEN_TEST_SPLIT``, ``NO_POST_HOC_CALIBRATION`` and ``SEEDS_ARE_IID``
    cannot be verified here and stay ``None``.
    """
    checked: dict[str, bool | None] = dict.fromkeys(reproduction_assumptions())
    checked[SPLIT_PROTOCOL_MATCHES] = runs.split_protocol == claim.split_protocol
    checked[METRIC_MATCHES] = claim.metric == metric_name
    checked[SAME_RANK_FUNCTIONAL] = True

    try:
        gpus = {str(r.env.get("gpu_model")) for r in runs}
        checked[SINGLE_GPU_ACROSS_SEEDS] = len(gpus) == 1
    except (AttributeError, TypeError):  # pragma: no cover - non-iterable stand-in
        checked[SINGLE_GPU_ACROSS_SEEDS] = None
    return checked
