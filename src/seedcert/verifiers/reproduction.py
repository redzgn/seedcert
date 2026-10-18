"""``ReproductionVerifier`` - the one concrete verifier (DESIGN Sec 3, Sec 4).

Tests whether a published metric value is consistent with a re-implementation's
run distribution, and returns a :class:`~seedcert.certificate.Certificate`. The
reference is the single-run distribution (``aggregation="single_run"``) or the
sampling distribution of an ``m``-run mean (``aggregation="mean"``, DESIGN D10).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from seedcert.certificate import (
    Certificate,
    Decision,
    TestDirection,
    minimum_seeds_aggregate,
)
from seedcert.claim import AGGREGATIONS
from seedcert.verifiers import nulls
from seedcert.verifiers.assumptions import check_reproduction_assumptions, reproduction_assumptions
from seedcert.verifiers.base import BaseVerifier
from seedcert.verifiers.metrics import METRIC_NAMES

if TYPE_CHECKING:
    from seedcert.cache.runs import RecipeRuns
    from seedcert.claim import PublishedClaim

SCHEMA_VERSION = "1"


class ReproductionVerifier(BaseVerifier):
    """Certify a re-implementation against one published number."""

    name = "reproduction"
    version = "0.1.0"

    def certify(
        self,
        runs: RecipeRuns,
        claim: PublishedClaim,
        *,
        alpha: float = 0.05,
        ci_level: float = 0.95,
        n_bootstrap: int = 10_000,
        equivalence_margin_points: float | None = None,
        direction: TestDirection = TestDirection.TWO_SIDED,
        rng: int | np.random.Generator = 0,
    ) -> Certificate:
        """See DESIGN Sec 4 for the pipeline.

        Raises:
            ValueError: ``runs`` and ``claim`` disagree on split protocol, the
                claim metric is not one of the four reported, or no runs are
                cached.
        """
        t0 = time.perf_counter()
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        if runs.split_protocol != claim.split_protocol:
            raise ValueError(
                f"split protocol mismatch: runs={runs.split_protocol!r} "
                f"claim={claim.split_protocol!r} - seedcert will not certify across a "
                "split-protocol mismatch (DESIGN D2)"
            )
        if claim.metric not in METRIC_NAMES:
            raise ValueError(f"claim.metric {claim.metric!r} not in {METRIC_NAMES}")
        if claim.aggregation not in AGGREGATIONS:
            raise ValueError(f"claim.aggregation {claim.aggregation!r} not in {AGGREGATIONS}")
        if runs.n_seeds < 1:
            raise ValueError(f"no cached runs for {runs.recipe.rel_path()}")

        metric_name = claim.metric
        s = np.asarray(runs.metric_values(metric_name), dtype=float)
        n = int(s.size)
        statistic = float(claim.value)

        aggregate_reference: dict[str, object] | None = None
        if claim.aggregation == "mean":
            m_runs = claim.claimed_n_seeds
            if not m_runs or m_runs < 2:
                raise ValueError(
                    "aggregation='mean' requires claim.claimed_n_seeds >= 2 "
                    "(the number of runs the published value averages)"
                )
            min_n = minimum_seeds_aggregate(m_runs, alpha=alpha)
            ref = nulls.m_run_mean_reference(
                s, m=m_runs, n_bootstrap=n_bootstrap, rng=gen
            )
            pf = nulls.p_floor(n_bootstrap, direction)
            p_value = max(
                nulls.permutation_p_value(statistic, ref, direction=direction), pf
            )
            aggregate_reference = {
                "m": int(m_runs),
                "n_bootstrap": int(n_bootstrap),
                "ref_mean": float(ref.mean()),
                "ref_sd": float(ref.std(ddof=1)),
                "ref_ci": [
                    float(np.quantile(ref, (1 - ci_level) / 2)),
                    float(np.quantile(ref, 1 - (1 - ci_level) / 2)),
                ],
                "min_seeds": int(min_n),
            }
            inconclusive = n < min_n
        else:
            pf = nulls.p_floor(n, direction)
            p_value = max(
                nulls.permutation_p_value(statistic, s, direction=direction), pf
            )
            inconclusive = pf >= alpha

        reimpl_mean = float(s.mean())
        _rlo, _rhi = nulls.bootstrap_mean_ci(
            s, ci_level=ci_level, n_bootstrap=n_bootstrap, rng=gen
        )
        reimpl_ci = (min(_rlo, reimpl_mean), max(_rhi, reimpl_mean))
        effect_size = nulls.cliffs_delta([statistic], s)
        _elo, _ehi = nulls.bootstrap_effect_ci(
            statistic, s, ci_level=ci_level, n_bootstrap=n_bootstrap, rng=gen
        )
        # the reported CI always contains the point estimate (percentile bootstrap
        # can otherwise miss it on a skewed / boundary statistic)
        effect_size_ci = (min(_elo, effect_size), max(_ehi, effect_size))
        std_gap = nulls.standardized_gap(statistic, s)

        if inconclusive:
            decision = Decision.INCONCLUSIVE
        elif p_value < alpha:
            decision = Decision.DISCREPANT
        else:
            decision = Decision.REPRODUCED

        equivalence = None
        if equivalence_margin_points is not None:
            equivalence = nulls.tost_equivalence_abs(
                statistic,
                s,
                margin_points=equivalence_margin_points,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                rng=gen,
            )

        secondary: dict[str, dict[str, object]] = {}
        for m in METRIC_NAMES:
            if m == metric_name:
                continue
            try:
                vals = np.asarray(runs.metric_values(m), dtype=float)
            except KeyError:  # pragma: no cover - all four are always cached
                continue
            secondary[m] = {
                "reimpl_mean": float(vals.mean()),
                "reimpl_ci": list(
                    nulls.bootstrap_mean_ci(
                        vals, ci_level=ci_level, n_bootstrap=n_bootstrap, rng=gen
                    )
                ),
                "claim_value": None,
                "p_value": None,
            }

        from seedcert.env import capture_environment

        return Certificate(
            schema_version=SCHEMA_VERSION,
            verifier_name=self.name,
            verifier_version=self.version,
            created_at=datetime.now(timezone.utc).isoformat(),
            env=capture_environment(),
            wall_clock_s=time.perf_counter() - t0,
            dataset=runs.dataset,
            split_protocol=runs.split_protocol,
            recipe=runs.recipe_descriptor(),
            recipe_hash=runs.recipe.recipe_hash,
            n_seeds=n,
            seed_list=tuple(runs.seed_list),
            metric_name=metric_name,
            claim=claim.to_dict(),
            statistic=statistic,
            null_distribution=tuple(float(v) for v in s),
            p_value=p_value,
            p_floor=pf,
            test_direction=direction,
            reimpl_mean=reimpl_mean,
            reimpl_ci=reimpl_ci,
            effect_size=effect_size,
            effect_size_ci=effect_size_ci,
            standardized_gap=std_gap,
            aggregate_reference=aggregate_reference,
            ci_level=ci_level,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            decision=decision,
            assumptions=reproduction_assumptions(),
            assumptions_checked=check_reproduction_assumptions(
                runs, claim, metric_name=metric_name
            ),
            equivalence=equivalence,
            secondary=secondary,
        )
