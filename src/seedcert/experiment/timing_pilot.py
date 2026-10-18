"""Measure per-run wall-clock on a small and a large dataset and project the
full reproduction-sweep cost (DESIGN Sec 7).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPORT_PATH = Path("timing_pilot_report.json")


@dataclass(frozen=True, slots=True)
class PilotReport:
    """Timings from the pilot and the projected sweep cost."""

    seconds_per_run: dict[str, float]  # dataset -> s/run
    graph_size: dict[str, int]  # dataset -> nodes + canonical edges
    fit_intercept: float
    fit_slope_per_elem: float  # s per (nodes + edges)
    projected_sweep_runs: int
    projected_sweep_hours: float


def _time_recipe(dataset: str, backbone: str, *, n_seeds: int, device: str) -> tuple[float, int]:
    """Return (mean seconds/run, nodes+edges) for a fresh recipe on ``dataset``."""
    from seedcert.data.datasets import load_canonical
    from seedcert.experiment.published_claims import overrides_for
    from seedcert.models.train import train_node_classifier
    from seedcert.recipe import Recipe, resolve_hyperparameters

    data = load_canonical(dataset)
    hp = resolve_hyperparameters(Recipe(backbone, overrides_for(backbone, dataset)))
    times: list[float] = []
    for seed in range(n_seeds):
        t0 = time.perf_counter()
        train_node_classifier(data, backbone=backbone, hp=hp, device=device, seed=seed)
        times.append(time.perf_counter() - t0)
    return float(sum(times) / len(times)), int(data.num_nodes + data.num_edges_canonical)


def project_sweep_hours(
    intercept: float,
    slope_per_elem: float,
    *,
    device: str = "cpu",
) -> tuple[int, float]:
    """(total runs, GPU/CPU-hours) for :func:`repro_targets` from the linear fit."""
    from seedcert.data.datasets import load_canonical
    from seedcert.experiment.grid import repro_targets

    size_cache: dict[str, int] = {}
    runs = 0
    seconds = 0.0
    for t in repro_targets():
        if t.dataset not in size_cache:
            d = load_canonical(t.dataset)
            size_cache[t.dataset] = int(d.num_nodes + d.num_edges_canonical)
        runs += t.n_seeds
        seconds += t.n_seeds * (intercept + slope_per_elem * size_cache[t.dataset])
    return runs, seconds / 3600.0


def run_timing_pilot(
    *,
    small: str = "Cora",
    large: str = "PubMed",
    backbone: str = "gcn",
    n_seeds: int = 3,
    device: str = "cpu",
) -> PilotReport:
    """Time ``n_seeds`` runs each for ``small`` and ``large``, fit
    ``s/run ~ a + b*(nodes + edges)``, and project the sweep."""
    s_small, size_small = _time_recipe(small, backbone, n_seeds=n_seeds, device=device)
    s_large, size_large = _time_recipe(large, backbone, n_seeds=n_seeds, device=device)

    slope = (s_large - s_small) / (size_large - size_small) if size_large != size_small else 0.0
    intercept = s_small - slope * size_small
    total_runs, hours = project_sweep_hours(intercept, slope, device=device)

    return PilotReport(
        seconds_per_run={small: s_small, large: s_large},
        graph_size={small: size_small, large: size_large},
        fit_intercept=intercept,
        fit_slope_per_elem=slope,
        projected_sweep_runs=total_runs,
        projected_sweep_hours=hours,
    )


def main() -> None:
    """CLI: ``seedcert-timing [--small Cora] [--large PubMed] [--seeds 3]``."""
    parser = argparse.ArgumentParser(prog="seedcert-timing")
    parser.add_argument("--small", default="Cora")
    parser.add_argument("--large", default="PubMed")
    parser.add_argument("--backbone", default="gcn", choices=("gcn", "gat", "sage"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args()

    report = run_timing_pilot(
        small=args.small,
        large=args.large,
        backbone=args.backbone,
        n_seeds=args.seeds,
        device=args.device,
    )
    Path(args.out).write_text(json.dumps(asdict(report), indent=2))
    for ds, s in report.seconds_per_run.items():
        print(f"  {ds:10} {s:.3f} s/run   (nodes+edges = {report.graph_size[ds]:,})")
    print(f"  fit: s/run ~ {report.fit_intercept:.3f} + {report.fit_slope_per_elem:.2e} * size")
    print(
        f"  projected sweep: {report.projected_sweep_runs} runs, "
        f"{report.projected_sweep_hours:.2f} {args.device}-hours"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
