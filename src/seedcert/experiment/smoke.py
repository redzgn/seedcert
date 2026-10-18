"""Small end-to-end checks, one per subcommand (DESIGN Sec 8).

    seedcert-smoke datasets        # load each canonical dataset, print shape + protocol
    seedcert-smoke train           # one GCN run on Cora, print the four metrics
    seedcert-smoke calibrate       # hold-one-seed-out p-value uniformity (KS) -> Fig 2
    seedcert-smoke discrepancy     # shifted claim -> DISCREPANT; on-target -> REPRODUCED
    seedcert-smoke split-mismatch  # a mismatched claim.split_protocol raises
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from seedcert.cache.build import build_recipe
from seedcert.cache.registry import RunRegistry
from seedcert.cache.runs import RecipeRuns
from seedcert.cache.spec import RecipeKey
from seedcert.certificate import Decision, TestDirection
from seedcert.claim import PublishedClaim
from seedcert.data.datasets import load_canonical, split_protocol_for
from seedcert.experiment.grid import GRID_DATASETS
from seedcert.experiment.published_claims import overrides_for
from seedcert.recipe import Recipe
from seedcert.verifiers import nulls
from seedcert.verifiers.reproduction import ReproductionVerifier


def _runs_for(
    dataset: str, backbone: str, *, n_seeds: int, registry: RunRegistry, device: str
) -> RecipeRuns:
    recipe = Recipe(
        backbone, overrides=overrides_for(backbone, dataset), label=f"{backbone}:{dataset}"
    )
    key = RecipeKey(dataset, split_protocol_for(dataset), backbone, recipe.recipe_hash())
    build_recipe(key, recipe=recipe, registry=registry, n_seeds=n_seeds, device=device)
    return RecipeRuns(registry, key)


def cmd_datasets(args: argparse.Namespace) -> None:
    print(f"{'dataset':16}{'nodes':>9}{'feat':>7}{'cls':>5}  split tr/va/te   protocol")
    for name in GRID_DATASETS:
        try:
            d = load_canonical(name)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:16}  ERROR: {type(exc).__name__}: {exc}")
            continue
        tr, va, te = int(d.train_mask.sum()), int(d.val_mask.sum()), int(d.test_mask.sum())
        split = f"{tr}/{va}/{te}"
        print(
            f"{name:16}{d.num_nodes:>9}{d.num_features:>7}{d.num_classes:>5}  "
            f"{split:<18}{split_protocol_for(name)}"
        )


def cmd_train(args: argparse.Namespace) -> None:
    from seedcert.models.train import train_node_classifier
    from seedcert.recipe import resolve_hyperparameters
    from seedcert.verifiers.metrics import all_metrics

    data = load_canonical(args.dataset)
    hp = resolve_hyperparameters(Recipe(args.backbone, overrides_for(args.backbone, args.dataset)))
    res = train_node_classifier(data, backbone=args.backbone, hp=hp, device=args.device, seed=0)
    y = data.y.numpy()
    m = all_metrics(res.logits, y, data.test_mask.numpy())
    print(
        f"{args.backbone} {args.dataset}  epochs {res.epochs_run} (best {res.best_epoch})  "
        f"{res.wall_clock_s:.2f}s"
    )
    for k, v in m.items():
        print(f"  {k:14} {v:.4f}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    registry = RunRegistry(args.registry)
    runs = _runs_for(
        args.dataset, args.backbone, n_seeds=args.seeds, registry=registry, device=args.device
    )
    s = runs.metric_values("test_accuracy")
    n = s.size
    m = n - 1  # reference size after holding one out
    floor = nulls.p_floor(m, TestDirection.TWO_SIDED)
    alpha = args.alpha

    pvals: list[float] = []
    reproduced = 0
    for i in range(n):
        ref = np.delete(s, i)
        raw = nulls.permutation_p_value(float(s[i]), ref, direction=TestDirection.TWO_SIDED)
        p = max(raw, floor)
        pvals.append(p)
        if floor < alpha and p >= alpha:
            reproduced += 1

    from scipy import stats

    ks = stats.kstest(pvals, "uniform")
    repro_rate = reproduced / n

    # text histogram, 10 bins on [0, 1]
    hist, _ = np.histogram(pvals, bins=10, range=(0.0, 1.0))
    print(f"hold-one-seed-out calibration: {args.backbone} {args.dataset}  n={n} (ref m={m})")
    for b in range(10):
        print(f"  [{b/10:.1f},{(b+1)/10:.1f})  {'#' * int(hist[b])} {hist[b]}")
    print(f"KS vs uniform: D={ks.statistic:.3f}  p={ks.pvalue:.3f}")
    print(f"REPRODUCED rate: {repro_rate:.3f}  (expected ~{1 - alpha:.2f})")

    out = Path(args.out)
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "backbone": args.backbone,
                "n_seeds": int(n),
                "reference_m": int(m),
                "alpha": alpha,
                "p_floor": floor,
                "pvalues": [float(p) for p in pvals],
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "reproduced_rate": repro_rate,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


def cmd_discrepancy(args: argparse.Namespace) -> None:
    registry = RunRegistry(args.registry)
    runs = _runs_for(
        args.dataset, args.backbone, n_seeds=args.seeds, registry=registry, device=args.device
    )
    mean = float(runs.metric_values("test_accuracy").mean())
    proto = split_protocol_for(args.dataset)
    v = ReproductionVerifier()

    on_target = v.certify(
        runs, PublishedClaim("test_accuracy", round(mean, 4), "on-target", proto),
        equivalence_margin_points=0.01,
    )
    shifted = v.certify(
        runs, PublishedClaim("test_accuracy", round(mean + args.shift, 4), "shifted", proto),
        equivalence_margin_points=0.01,
    )
    print(
        f"ON-TARGET claim (= re-impl mean): {on_target.decision.value}  p={on_target.p_value:.3f}"
    )
    print(
        f"SHIFTED claim (+{args.shift}): {shifted.decision.value}  p={shifted.p_value:.3f}  "
        f"gap={shifted.standardized_gap:+.2f} SD"
    )
    assert on_target.decision is Decision.REPRODUCED, on_target.decision
    assert shifted.decision is Decision.DISCREPANT, shifted.decision
    print("OK: on-target REPRODUCED, shifted DISCREPANT")


def cmd_split_mismatch(args: argparse.Namespace) -> None:
    registry = RunRegistry(args.registry)
    runs = _runs_for("Cora", "gcn", n_seeds=args.seeds, registry=registry, device=args.device)
    bad = PublishedClaim("test_accuracy", 0.815, "wrong-split", "geom-gcn-split0")
    try:
        ReproductionVerifier().certify(runs, bad)
    except ValueError as exc:
        print(f"OK: certify refused the mismatch -> ValueError: {exc}")
        return
    raise SystemExit("FAIL: split-protocol mismatch was not rejected")


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registry", default="run_cache")
    common.add_argument("--device", default="cpu")
    common.add_argument("--dataset", default="Cora")
    common.add_argument("--backbone", default="gcn", choices=("gcn", "gat", "sage"))
    common.add_argument("--seeds", type=int, default=45)
    common.add_argument("--alpha", type=float, default=0.05)
    common.add_argument("--shift", type=float, default=0.05)
    common.add_argument("--out", default="calibration.json")

    parser = argparse.ArgumentParser(prog="seedcert-smoke")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("datasets", "train", "calibrate", "discrepancy", "split-mismatch"):
        sub.add_parser(name, parents=[common])
    args = parser.parse_args()
    {
        "datasets": cmd_datasets,
        "train": cmd_train,
        "calibrate": cmd_calibrate,
        "discrepancy": cmd_discrepancy,
        "split-mismatch": cmd_split_mismatch,
    }[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover
    main()
