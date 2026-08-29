"""Driver: build the runs for each reproduction target, certify, collect
certificates to disk (DESIGN Sec 7, Sec 8).
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from seedcert.cache.build import build_target
from seedcert.cache.registry import RunRegistry
from seedcert.cache.runs import RecipeRuns
from seedcert.experiment.grid import repro_targets
from seedcert.verifiers.reproduction import ReproductionVerifier

if TYPE_CHECKING:
    from seedcert.certificate import Certificate
    from seedcert.experiment.grid import ReproTarget

DEFAULT_OUT = Path("repro_certs")
DEFAULT_MARGIN_POINTS = 0.01


def _cert_filename(target: ReproTarget) -> str:
    return f"{target.backbone}__{target.dataset}.json"


def certify_target(
    target: ReproTarget,
    *,
    registry_root: str = "run_cache",
    lock_path: str = "datasets.lock.json",
    device: str = "cuda",
    build_missing: bool = True,
    equivalence_margin_points: float | None = DEFAULT_MARGIN_POINTS,
) -> Certificate:
    """Build ``target``'s runs if missing, then run
    :class:`~seedcert.verifiers.reproduction.ReproductionVerifier`."""
    registry = RunRegistry(registry_root, lock_path=lock_path)
    if build_missing:
        recipe_key = build_target(
            target, registry=registry, device=device, resume=True, rebuild_index=True
        )
    else:
        recipe = target.recipe()
        from seedcert.cache.spec import RecipeKey

        recipe_key = RecipeKey(
            target.dataset, target.split_protocol(), target.backbone, recipe.recipe_hash()
        )
    runs = RecipeRuns(registry, recipe_key)
    return ReproductionVerifier().certify(
        runs, target.claim, equivalence_margin_points=equivalence_margin_points
    )


def run_reproduction_grid(
    *,
    out_dir: Path = DEFAULT_OUT,
    registry_root: str = "run_cache",
    lock_path: str = "datasets.lock.json",
    device: str = "cuda",
    build_missing: bool = True,
    n_seeds: int | None = None,
) -> Path:
    """Certify every :func:`~seedcert.experiment.grid.repro_targets` target,
    writing one ``Certificate`` JSON per target plus a
    ``_certificates.parquet`` summary. Per-target failures are logged to
    ``_failures.log``, not fatal. Returns the parquet path."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    targets = repro_targets() if n_seeds is None else repro_targets(n_seeds=n_seeds)

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for target in targets:
        try:
            cert = certify_target(
                target,
                registry_root=registry_root,
                lock_path=lock_path,
                device=device,
                build_missing=build_missing,
            )
            (out_dir / _cert_filename(target)).write_text(cert.to_json())
            eq = cert.equivalence or {}
            rows.append(
                {
                    "backbone": target.backbone,
                    "dataset": target.dataset,
                    "split_protocol": cert.split_protocol,
                    "overrides": json.dumps(target.overrides, sort_keys=True),
                    "decision": cert.decision.value,
                    "p_value": cert.p_value,
                    "reimpl_mean": cert.reimpl_mean,
                    "ci_lo": cert.reimpl_ci[0],
                    "ci_hi": cert.reimpl_ci[1],
                    "claim": cert.claim["value"],
                    "standardized_gap": cert.standardized_gap,
                    "cliffs_delta": cert.effect_size,
                    "equivalent": bool(eq.get("equivalent", False)),
                    "n_seeds": cert.n_seeds,
                    "source": cert.claim.get("source", ""),
                }
            )
        except Exception as exc:  # noqa: BLE001 - logged, not fatal
            failures.append(
                f"{target.backbone}/{target.dataset}: {exc!r}\n{traceback.format_exc()}"
            )

    pd.DataFrame(rows).to_parquet(out_dir / "_certificates.parquet", index=False)
    if failures:
        (out_dir / "_failures.log").write_text("\n\n".join(failures))
    print(f"{len(rows)} certificates -> {out_dir}  ({len(failures)} failures)")
    return out_dir / "_certificates.parquet"


def main() -> None:
    """CLI: ``seedcert-certify [--out repro_certs] [--device cuda]``."""
    parser = argparse.ArgumentParser(prog="seedcert-certify")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--registry", default="run_cache")
    parser.add_argument("--lock", default="datasets.lock.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, default=None, help="override per-target n_seeds")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    path = run_reproduction_grid(
        out_dir=Path(args.out),
        registry_root=args.registry,
        lock_path=args.lock,
        device=args.device,
        build_missing=not args.no_build,
        n_seeds=args.seeds,
    )
    import pandas as pd

    from seedcert.experiment.report import reproduction_table, summarize

    df = pd.read_parquet(path)
    print(reproduction_table(df).to_string(index=False))
    print()
    for k, v in summarize(df).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":  # pragma: no cover
    main()
