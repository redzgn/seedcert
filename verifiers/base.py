"""``BaseVerifier`` - the abstract contract every verifier implements
(DESIGN Sec 3).

The signature is fixed: a :class:`~seedcert.cache.runs.RecipeRuns` (the cached
seed ensemble for one recipe) and a :class:`~seedcert.claim.PublishedClaim`, and
the return type is always :class:`~seedcert.certificate.Certificate`.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from seedcert.certificate import TestDirection

if TYPE_CHECKING:
    import numpy as np

    from seedcert.cache.runs import RecipeRuns
    from seedcert.certificate import Certificate
    from seedcert.claim import PublishedClaim


class BaseVerifier(abc.ABC):
    """Base class for the reproduction verifier (and any future variant).

    Subclasses set :attr:`name` and :attr:`version` and implement
    :meth:`certify`.
    """

    name: str
    version: str

    @abc.abstractmethod
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
        """Produce a :class:`~seedcert.certificate.Certificate`.

        Args:
            runs: The cached seed runs for one recipe on one dataset + split
                protocol. Carries the recipe, the seed list, and the per-seed
                metric values.
            claim: The published value under test. Its ``metric`` and
                ``split_protocol`` must match ``runs``; a mismatch is a
                ``ValueError`` (DESIGN D2).
            alpha: Significance level. If ``1 / (runs.n_seeds + 1) >= alpha`` the
                certificate is forced to ``INCONCLUSIVE`` (FIX 1).
            ci_level: Confidence level for ``reimpl_ci`` and ``effect_size_ci``.
            n_bootstrap: Bootstrap resamples for the intervals.
            equivalence_margin_points: Absolute TOST margin in metric points
                (e.g. ``0.01`` for one accuracy point). ``None`` disables the
                equivalence result.
            direction: ``TWO_SIDED`` (default) or ``LOWER`` ("not worse than the
                paper").
            rng: Seed or generator for bootstrap resampling only.

        Returns:
            A validated ``Certificate``. Never a float.

        Raises:
            NotImplementedError: Stage 1 skeleton.
            ValueError: ``runs`` and ``claim`` disagree on metric or split
                protocol.
        """
        raise NotImplementedError
