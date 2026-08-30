"""Certificate aggregation into the paper table (DESIGN Sec 8, WP4)."""

from __future__ import annotations

import json
from pathlib import Path

from seedcert.experiment import report


def _write_cert(
    d: Path,
    name: str,
    *,
    backbone: str,
    dataset: str,
    decision: str,
    claim: float,
    reimpl_mean: float,
    equivalent: bool,
) -> None:
    (d / name).write_text(
        json.dumps(
            {
                "recipe": {"backbone": backbone, "overrides": {}, "label": ""},
                "dataset": dataset,
                "split_protocol": "planetoid-public",
                "decision": decision,
                "p_value": 0.2 if decision == "reproduced" else 0.03,
                "reimpl_mean": reimpl_mean,
                "reimpl_ci": [reimpl_mean - 0.003, reimpl_mean + 0.003],
                "claim": {"value": claim, "source": "Paper X"},
                "standardized_gap": (reimpl_mean - claim) / 0.008,
                "effect_size": 0.5,
                "n_seeds": 50,
                "equivalence": {"margin_points": 0.01, "tost_p": 0.0, "equivalent": equivalent},
            }
        )
    )


def test_aggregate_empty_dir(tmp_path: Path) -> None:
    df = report.aggregate_certificates(tmp_path)
    assert df.empty
    assert list(df.columns) == list(report._FLAT_FIELDS)
    assert report.summarize(df) == {"n": 0}


def test_aggregate_and_table(tmp_path: Path) -> None:
    _write_cert(
        tmp_path, "gcn__Cora.json", backbone="gcn", dataset="Cora",
        decision="discrepant", claim=0.815, reimpl_mean=0.799, equivalent=False,
    )
    _write_cert(
        tmp_path, "gcn__PubMed.json", backbone="gcn", dataset="PubMed",
        decision="reproduced", claim=0.790, reimpl_mean=0.789, equivalent=True,
    )
    (tmp_path / "_certificates.parquet").write_bytes(b"")  # ignored (underscore prefix)

    df = report.aggregate_certificates(tmp_path)
    assert len(df) == 2
    assert set(df["decision"]) == {"discrepant", "reproduced"}

    table = report.reproduction_table(df)
    assert list(table["dataset"]) == ["Cora", "PubMed"]
    assert "reimpl" in table.columns and "gap (SD)" in table.columns

    s = report.summarize(df)
    assert s["n"] == 2
    assert s["discrepant"] == 0.5 and s["reproduced"] == 0.5
    assert s["equivalent"] == 0.5
