"""The ``Certificate`` dataclass and its enums (DESIGN Sec 2).

A certificate is the only thing :class:`~seedcert.verifiers.base.BaseVerifier`
may return. It carries the re-implementation's estimate, its uncertainty, the
published claim under test, and the premises together, and it validates every
invariant at construction, so a bare score cannot masquerade as a result.
"""

from __future__ import annotations

import enum
import json
import math
from dataclasses import dataclass, field, fields
from typing import Any

_TOL = 1e-9


class Decision(enum.Enum):
    """Outcome of the reproduction test (DESIGN Sec 1b).

    Note the valence is the opposite of ``certiforget``: ``REPRODUCED`` is a
    *failure to reject* the published value and is the desired outcome, but it is
    **not** proof of equality - :meth:`Certificate.summary` says so and points at
    the equivalence margin. ``INCONCLUSIVE`` means ``n`` seeds is too small for
    the chosen ``alpha`` (FIX 1) or a required assumption check failed.
    """

    REPRODUCED = "reproduced"
    DISCREPANT = "discrepant"
    INCONCLUSIVE = "inconclusive"

    def __str__(self) -> str:  # pragma: no cover - trivial
        caveat = {
            "reproduced": "seed distribution is consistent with the published value "
            "(NOT proof of equality)",
            "discrepant": "re-implementation differs from the published value at alpha",
            "inconclusive": "test cannot decide at this alpha / seed count",
        }
        return f"{self.value} ({caveat[self.value]})"


class TestDirection(enum.Enum):
    """Tail(s) tested. ``LOWER`` flags a re-implementation that sits *below* the
    published value (a regression); ``TWO_SIDED`` is the default."""

    TWO_SIDED = "two_sided"
    LOWER = "lower"
    UPPER = "upper"


def minimum_seeds(alpha: float, *, two_sided: bool = True) -> int:
    """Smallest ``n_seeds`` for which the rank test can reject at ``alpha``.

    A one-sample rank p-value cannot fall below ``1 / (n + 1)`` one-sided, or
    ``2 / (n + 1)`` for the two-sided doubled-smaller-tail form (DESIGN D4).
    Requiring the floor ``< alpha`` gives:

        two-sided:  ``minimum_seeds(alpha) = floor(2/alpha - 1) + 1``
        one-sided:  ``minimum_seeds(alpha, two_sided=False) = floor(1/alpha - 1) + 1``

    so ``minimum_seeds(0.05) == 40`` (two-sided default) and
    ``minimum_seeds(0.05, two_sided=False) == 20``. Planner helper only; the
    binding check lives in :meth:`Certificate._validate`.
    """
    k = 2.0 if two_sided else 1.0
    return math.floor(k / alpha - 1.0) + 1


def minimum_seeds_aggregate(claimed_n_seeds: int, *, alpha: float = 0.05) -> int:
    """Smallest ``n_seeds`` for the ``aggregation="mean"`` path (DESIGN D10).

    The published value is a mean over ``claimed_n_seeds`` runs, so the
    re-implementation must provide at least that many seeds for the matched
    ``m``-run-mean reference to be non-degenerate, and never fewer than the
    one-sided single-run floor.
    """
    return max(minimum_seeds(alpha, two_sided=False), int(claimed_n_seeds))


@dataclass(frozen=True, slots=True)
class Certificate:
    """An immutable reproduction certificate (DESIGN Sec 2.1).

    Assumptions carried (populated by the verifier, see
    ``verifiers/assumptions.py``): the recipe matches the one the claim was
    measured under; the run used the claim's split protocol; the metric matches;
    a single GPU model across the seed runs; no post-hoc calibration; the
    published value and the seed distribution enter one rank function.

    Construction validates every invariant in :meth:`_validate`; an invalid
    certificate cannot exist. ``float(cert)`` is deliberately a ``TypeError``.
    """

    # --- provenance / identity ---
    schema_version: str
    verifier_name: str
    verifier_version: str
    created_at: str
    env: dict[str, Any]
    wall_clock_s: float

    # --- target ---
    dataset: str
    split_protocol: str
    recipe: dict[str, Any]
    recipe_hash: str
    n_seeds: int
    seed_list: tuple[int, ...]
    metric_name: str
    claim: dict[str, Any]

    # --- statistical result ---
    statistic: float  # = claim value under test
    null_distribution: tuple[float, ...]  # the n per-seed metric values
    p_value: float
    p_floor: float
    test_direction: TestDirection
    reimpl_mean: float
    reimpl_ci: tuple[float, float]
    effect_size: float
    effect_size_ci: tuple[float, float]
    standardized_gap: float
    ci_level: float
    alpha: float
    n_bootstrap: int
    decision: Decision

    # --- assumptions ---
    assumptions: tuple[str, ...]
    assumptions_checked: dict[str, bool | None]

    # --- optional ---
    equivalence: dict[str, Any] | None = None
    aggregate_reference: dict[str, Any] | None = None
    secondary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Enforce every certificate invariant (DESIGN Sec 2.2).

        Checks, in intent:
            * ``0 <= p_value <= 1`` and ``p_value >= p_floor``;
            * ``p_floor == k / (n_seeds + 1)`` (single-run mode) or
              ``k / (n_bootstrap + 1)`` (``aggregate_reference`` present), with
              ``k = 2`` two-sided else ``1`` (DESIGN D4, D10);
            * ``aggregate_reference.ref_ci`` ordered and ``m >= 2`` when present;
            * **FIX 1** - if ``p_floor >= alpha`` then ``decision`` must be
              ``INCONCLUSIVE``;
            * ``reimpl_ci`` ordered; ``effect_size_ci`` ordered and bracketing
              ``effect_size``;
            * ``0 < alpha < 1`` and ``0 < ci_level < 1``; ``n_seeds >= 1``;
            * ``len(seed_list) == n_seeds`` and
              ``len(null_distribution) == n_seeds``;
            * ``assumptions`` non-empty; ``test_direction`` / ``decision`` enums;
            * ``recipe_hash`` matches ``recipe``;
            * ``claim['split_protocol'] == split_protocol`` and
              ``claim['metric'] == metric_name``;
            * ``decision`` consistent with ``p_value`` vs ``alpha`` unless
              ``INCONCLUSIVE``;
            * every ``secondary[m]`` has an ordered ``reimpl_ci``.

        Raises:
            ValueError: any invariant is violated.
        """
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value {self.p_value} not in [0, 1]")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha {self.alpha} not in (0, 1)")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError(f"ci_level {self.ci_level} not in (0, 1)")
        if self.n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")

        two_sided = self.test_direction is TestDirection.TWO_SIDED
        k = 2.0 if two_sided else 1.0
        agg = self.aggregate_reference
        if agg is not None:
            # aggregation="mean": the reference is a bootstrap of m-run means,
            # so the rank floor is k / (n_bootstrap + 1) (DESIGN D10).
            expected_floor = k / (int(agg["n_bootstrap"]) + 1)
            floor_desc = f"{k:g}/(n_bootstrap+1)"
        else:
            expected_floor = k / (self.n_seeds + 1)
            floor_desc = f"{k:g}/(n_seeds+1)"
        if abs(self.p_floor - expected_floor) > _TOL:
            raise ValueError(
                f"p_floor {self.p_floor} != {floor_desc} = {expected_floor}"
            )
        if self.p_value < self.p_floor - _TOL:
            raise ValueError(f"p_value {self.p_value} below p_floor {self.p_floor}")

        if self.p_floor >= self.alpha and self.decision is not Decision.INCONCLUSIVE:
            raise ValueError(
                f"p_floor {self.p_floor:.4f} >= alpha {self.alpha}: "
                "decision must be INCONCLUSIVE (FIX 1)"
            )

        if agg is not None:
            lo, hi = agg["ref_ci"]
            if lo > hi + _TOL:
                raise ValueError(f"aggregate_reference.ref_ci not ordered: {(lo, hi)}")
            if int(agg["m"]) < 2:
                raise ValueError("aggregate_reference.m must be >= 2")

        for label, (lo, hi), point in (
            ("reimpl_ci", self.reimpl_ci, self.reimpl_mean),
            ("effect_size_ci", self.effect_size_ci, self.effect_size),
        ):
            if lo > hi + _TOL:
                raise ValueError(f"{label} not ordered: {(lo, hi)}")
            if label == "effect_size_ci" and not lo - _TOL <= point <= hi + _TOL:
                raise ValueError(f"effect_size {point} outside {label} {(lo, hi)}")

        if len(self.seed_list) != self.n_seeds:
            raise ValueError(
                f"seed_list has {len(self.seed_list)} entries; expected n_seeds={self.n_seeds}"
            )
        if len(self.null_distribution) != self.n_seeds:
            raise ValueError(
                f"null_distribution has {len(self.null_distribution)} entries; "
                f"expected n_seeds={self.n_seeds}"
            )
        if not self.assumptions:
            raise ValueError("assumptions must be non-empty")
        if not isinstance(self.test_direction, TestDirection):
            raise ValueError("test_direction must be a TestDirection")
        if not isinstance(self.decision, Decision):
            raise ValueError("decision must be a Decision")

        from seedcert.recipe import Recipe

        recomputed = Recipe(
            self.recipe["backbone"],
            dict(self.recipe.get("overrides", {})),
            str(self.recipe.get("label", "")),
        ).recipe_hash()
        if recomputed != self.recipe_hash:
            raise ValueError(
                f"recipe_hash {self.recipe_hash} does not match recipe {self.recipe}"
            )

        if self.claim.get("split_protocol") != self.split_protocol:
            raise ValueError(
                f"claim split_protocol {self.claim.get('split_protocol')!r} != "
                f"certificate split_protocol {self.split_protocol!r}"
            )
        if self.claim.get("metric") != self.metric_name:
            raise ValueError(
                f"claim metric {self.claim.get('metric')!r} != metric_name {self.metric_name!r}"
            )

        if self.decision is Decision.DISCREPANT and self.p_value >= self.alpha:
            raise ValueError(
                f"decision DISCREPANT but p_value {self.p_value} >= alpha {self.alpha}"
            )
        if self.decision is Decision.REPRODUCED and self.p_value < self.alpha:
            raise ValueError(
                f"decision REPRODUCED but p_value {self.p_value} < alpha {self.alpha}"
            )

        for name, entry in self.secondary.items():
            ci = entry.get("reimpl_ci") if isinstance(entry, dict) else None
            if ci is not None and ci[0] > ci[1] + _TOL:
                raise ValueError(f"secondary[{name!r}].reimpl_ci not ordered: {tuple(ci)}")

    def __float__(self) -> float:
        raise TypeError(
            "A Certificate is not a number. Read .decision, .p_value, "
            ".reimpl_mean / .reimpl_ci, .equivalence, and .assumptions - together."
        )

    def summary(self) -> str:
        """One-screen human summary (DESIGN Sec 1b, Sec 2.3).

        Always prints, together: the decision with its caveat, the published
        value and the re-implementation mean + CI, the p-value alongside
        ``p_floor``, the effect size with CI, ``alpha``, ``n_seeds``, and - when
        ``decision is Decision.INCONCLUSIVE`` because of FIX 1 - the string
        ``"n_seeds=<n> too small: minimum achievable p = <p_floor> >= alpha =
        <alpha>"``. When ``equivalence`` is ``None`` it states that
        ``REPRODUCED`` is not proof of equality.
        """
        cv = self.claim.get("value")
        lines = [
            f"Certificate [{self.verifier_name} v{self.verifier_version}] "
            f"schema {self.schema_version}",
            f"  dataset: {self.dataset}   split_protocol: {self.split_protocol}",
            f"  recipe: {self.recipe.get('backbone')} {self.recipe.get('overrides', {})}"
            f"  ({self.recipe.get('label', '')})",
            f"  metric: {self.metric_name}   claim: {cv}  [{self.claim.get('source', '')}]",
            f"  decision: {self.decision}",
        ]
        agg = self.aggregate_reference
        if self.decision is Decision.INCONCLUSIVE:
            if agg is not None and self.n_seeds < int(agg.get("min_seeds", 0)):
                lines.append(
                    f"    n_seeds={self.n_seeds} < {agg['min_seeds']} needed for a "
                    f"{agg['m']}-run-mean reference"
                )
            elif self.p_floor >= self.alpha:
                lines.append(
                    f"    n_seeds={self.n_seeds} too small: minimum achievable p = "
                    f"{self.p_floor:.4f} >= alpha = {self.alpha}"
                )
        lines += [
            f"  re-implementation: mean {self.reimpl_mean:.4f}  "
            f"{round(self.ci_level * 100)}% CI [{self.reimpl_ci[0]:.4f}, {self.reimpl_ci[1]:.4f}]"
            f"  (n_seeds={self.n_seeds})",
        ]
        if agg is not None:
            lines.append(
                f"  reference: distribution of {agg['m']}-run means "
                f"(bootstrap B={agg['n_bootstrap']}), mean {agg['ref_mean']:.4f}  "
                f"sd {agg['ref_sd']:.4f}"
            )
        lines += [
            f"  published value: {cv}   gap: {self.standardized_gap:+.2f} single-run SD "
            f"(re-impl minus claim)",
            f"  p_value: {self.p_value:.4f}   (p_floor = {self.p_floor:.4f})   "
            f"test_direction: {self.test_direction.value}   alpha: {self.alpha}"
            + (f"   (vs {agg['m']}-run-mean reference)" if agg is not None else ""),
            f"  effect_size (Cliff's delta, claim vs single runs): {self.effect_size:+.3f}   "
            f"{round(self.ci_level * 100)}% CI "
            f"[{self.effect_size_ci[0]:+.3f}, {self.effect_size_ci[1]:+.3f}]",
        ]
        if self.equivalence is None:
            if self.decision is Decision.REPRODUCED:
                lines.append(
                    "  note: REPRODUCED is a failure to reject, NOT proof of equality "
                    "(pass equivalence_margin_points for a TOST result)."
                )
        else:
            eq = self.equivalence
            lines.append(
                f"  equivalence (TOST, margin +/-{eq['margin_points']} points): "
                f"tost_p={eq['tost_p']:.4f}  equivalent={eq['equivalent']}"
            )
        if self.secondary:
            lines.append("  secondary metrics (no verdict):")
            for name, entry in self.secondary.items():
                m = entry.get("reimpl_mean")
                ci = entry.get("reimpl_ci")
                if m is not None and ci is not None:
                    lines.append(f"    {name}: mean {m:.4f}  CI [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append(f"  assumptions ({len(self.assumptions)}):")
        lines += [f"    - {a}" for a in self.assumptions]
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize every field, including ``null_distribution`` and ``env``."""
        payload: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (Decision, TestDirection)):
                value = value.value
            elif isinstance(value, tuple):
                value = list(value)
            payload[f.name] = value
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> Certificate:
        """Inverse of :meth:`to_json`; round-trips all fields and re-validates."""
        raw = json.loads(payload)
        raw["decision"] = Decision(raw["decision"])
        raw["test_direction"] = TestDirection(raw["test_direction"])
        for key in ("seed_list", "null_distribution", "assumptions"):
            raw[key] = tuple(raw[key])
        for key in ("reimpl_ci", "effect_size_ci"):
            raw[key] = tuple(raw[key])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})
