"""``PublishedClaim`` - the reported number a re-implementation is tested against
(DESIGN Sec 3.2).

A claim names the metric, its value, the source it is quoted from, the split
protocol it was measured under, and how it was aggregated. The split protocol is
what makes the comparison honest (``ReproductionVerifier.certify`` refuses to run
across a mismatch, DESIGN D2); the aggregation identifies the estimand
(``single_run`` vs ``mean`` over ``claimed_n_seeds`` runs, DESIGN D10).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

#: Accepted values for ``PublishedClaim.aggregation``.
AGGREGATIONS: tuple[str, ...] = ("single_run", "mean")


@dataclass(frozen=True, slots=True)
class PublishedClaim:
    """One published metric value.

    Args:
        metric: Metric name, matching ``Certificate.metric_name`` (e.g.
            ``"test_accuracy"``).
        value: The reported value, on the same scale the verifier computes
            (accuracy as a fraction in ``[0, 1]``).
        source: Human-readable citation, e.g. ``"Kipf & Welling 2017, Table 2"``.
        split_protocol: The split the number was measured under, matching
            :func:`seedcert.data.datasets.split_protocol_for`.
        aggregation: The estimand the value targets. ``"single_run"`` (default):
            the value is one run's metric, tested by its rank among single
            re-implementation runs. ``"mean"``: the value is a mean over
            ``claimed_n_seeds`` runs, tested against the sampling distribution of
            an equally-sized re-implementation mean (DESIGN D10).
        claimed_sd: Reported standard deviation across the paper's own runs, if
            given. Recorded only; not used in the test.
        claimed_n_seeds: How many runs the paper averaged. Required when
            ``aggregation == "mean"``.
        doi: DOI or URL of the source, if available.
    """

    metric: str
    value: float
    source: str
    split_protocol: str
    aggregation: str = "single_run"
    claimed_sd: float | None = None
    claimed_n_seeds: int | None = None
    doi: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for ``Certificate.claim``."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublishedClaim:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    @classmethod
    def from_json(cls, payload: str) -> PublishedClaim:
        return cls.from_dict(json.loads(payload))
