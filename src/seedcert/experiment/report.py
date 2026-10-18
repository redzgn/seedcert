"""Aggregate reproduction certificates into the paper's table (DESIGN Sec 8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

_FLAT_FIELDS = (
    "backbone",
    "dataset",
    "split_protocol",
    "decision",
    "p_value",
    "reimpl_mean",
    "ci_lo",
    "ci_hi",
    "claim",
    "standardized_gap",
    "cliffs_delta",
    "equivalent",
    "n_seeds",
    "source",
)


def _flatten(cert: dict[str, Any]) -> dict[str, Any]:
    claim = cert["claim"]
    eq = cert.get("equivalence") or {}
    ci = cert["reimpl_ci"]
    return {
        "backbone": cert["recipe"]["backbone"],
        "dataset": cert["dataset"],
        "split_protocol": cert["split_protocol"],
        "decision": cert["decision"],
        "p_value": cert["p_value"],
        "reimpl_mean": cert["reimpl_mean"],
        "ci_lo": ci[0],
        "ci_hi": ci[1],
        "claim": claim["value"],
        "standardized_gap": cert["standardized_gap"],
        "cliffs_delta": cert["effect_size"],
        "equivalent": bool(eq.get("equivalent", False)),
        "n_seeds": cert["n_seeds"],
        "source": claim.get("source", ""),
    }


def aggregate_certificates(cert_dir: Path) -> pd.DataFrame:
    """Load every ``*.json`` certificate under ``cert_dir`` into one
    row-per-certificate frame."""
    import pandas as pd

    rows = [
        _flatten(json.loads(p.read_text()))
        for p in sorted(Path(cert_dir).glob("*.json"))
        if not p.name.startswith("_")
    ]
    return pd.DataFrame(rows, columns=list(_FLAT_FIELDS))


def reproduction_table(df: pd.DataFrame) -> pd.DataFrame:
    """The paper's Table 2: one row per ``(backbone, dataset)`` with the decision,
    ``reimpl_mean`` +/- CI, published value, standardized gap, and equivalence."""
    if df.empty:
        return df
    out = df.copy()
    out["reimpl"] = out.apply(
        lambda r: f"{r['reimpl_mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", axis=1
    )
    out["gap (SD)"] = out["standardized_gap"].map(lambda g: f"{g:+.2f}")
    cols = [
        "backbone",
        "dataset",
        "claim",
        "reimpl",
        "gap (SD)",
        "p_value",
        "decision",
        "equivalent",
    ]
    return out[cols].sort_values(["backbone", "dataset"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict[str, float | int]:
    """Headline counts: number of certificates and the fraction ``REPRODUCED`` /
    ``DISCREPANT`` / ``INCONCLUSIVE`` / ``equivalent``."""
    n = len(df)
    if n == 0:
        return {"n": 0}
    dec = df["decision"].value_counts().to_dict()
    return {
        "n": n,
        "reproduced": dec.get("reproduced", 0) / n,
        "discrepant": dec.get("discrepant", 0) / n,
        "inconclusive": dec.get("inconclusive", 0) / n,
        "equivalent": float(df["equivalent"].mean()),
    }
