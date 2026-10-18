"""Driver that populates the run cache for a recipe (DESIGN Sec 6, Sec 7).

Resumable: a recipe's ``recipe_manifest.json`` records completed seeds, so a
re-invocation only trains what is missing.

Stage 1/WP1: :func:`build_recipe` and :func:`main` are implemented;
:func:`build_target` (needs the grid) stays a stub until WP4.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

from seedcert.cache.registry import RunRegistry
from seedcert.cache.spec import RecipeKey

if TYPE_CHECKING:
    from seedcert.experiment.grid import ReproTarget
    from seedcert.recipe import Recipe


def _parse_overrides(text: str) -> dict[str, Any]:
    """``"hidden_dim=16,dropout=0.5"`` -> ``{"hidden_dim": 16, "dropout": 0.5}``.
    Values are parsed as int, then float, then left as str."""
    out: dict[str, Any] = {}
    for piece in filter(None, (p.strip() for p in text.split(","))):
        k, _, v = piece.partition("=")
        k = k.strip()
        v = v.strip()
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = v
    return out


def build_recipe(
    recipe_key: RecipeKey,
    *,
    recipe: Recipe,
    registry: RunRegistry,
    n_seeds: int,
    device: str = "cuda",
    resume: bool = True,
    rebuild_index: bool = True,
) -> None:
    """Train every missing seed ``0..n_seeds-1`` for one recipe.

    Persists ``recipe.json`` once, then trains, updating
    ``recipe_manifest.json`` after each seed so the run is resumable. Rebuilds
    ``index.parquet`` at the end unless ``rebuild_index`` is False.

    Raises:
        ValueError: ``recipe.recipe_hash() != recipe_key.recipe_hash``.
    """
    import numpy as np
    import torch

    from seedcert.cache.trainer import train_run
    from seedcert.data.datasets import load_canonical, split_protocol_for
    from seedcert.models.config import HYPERPARAMETERS

    recipe.validate()
    if recipe.recipe_hash() != recipe_key.recipe_hash:
        raise ValueError(
            f"recipe hash {recipe.recipe_hash()} != recipe_key.recipe_hash {recipe_key.recipe_hash}"
        )
    expected_protocol = split_protocol_for(recipe_key.dataset)
    if recipe_key.split_protocol != expected_protocol:
        raise ValueError(
            f"split_protocol {recipe_key.split_protocol!r} != canonical "
            f"{expected_protocol!r} for {recipe_key.dataset}"
        )

    data = load_canonical(recipe_key.dataset)
    sha = registry.dataset_sha256(recipe_key.dataset)

    recipe_dir = registry.root / recipe_key.rel_path()
    recipe_dir.mkdir(parents=True, exist_ok=True)
    (recipe_dir / "recipe.json").write_text(json.dumps(recipe.descriptor(), indent=2))

    manifest_path = recipe_dir / "recipe_manifest.json"
    done: set[int] = set()
    hashes: dict[str, str] = {}
    if resume and manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        done = {int(s) for s in stored.get("completed_seeds", [])}
        hashes = dict(stored.get("hashes", {}))

    for seed in range(n_seeds):
        if seed in done:
            continue
        key = recipe_key.with_seed(seed)
        art = train_run(
            key=key,
            data=data,
            recipe=recipe,
            hp=HYPERPARAMETERS,
            device=device,
            dataset_sha256=sha,
        )
        rd = key.run_dir(registry.root)
        rd.mkdir(parents=True, exist_ok=True)
        torch.save(art.state_dict, rd / "state_dict.pt")
        np.save(rd / "logits.npy", art.logits)
        (rd / "metrics.json").write_text(json.dumps(art.metrics, indent=2))
        (rd / "env.json").write_text(
            json.dumps({**art.env, "seed_provenance": art.seed_provenance}, indent=2)
        )
        done.add(seed)
        hashes[str(seed)] = str(art.metrics["content_hash"])
        manifest_path.write_text(
            json.dumps({"completed_seeds": sorted(done), "hashes": hashes}, indent=2)
        )

    if rebuild_index:
        registry.rebuild_index()


def build_target(
    target: ReproTarget,
    *,
    registry: RunRegistry,
    n_seeds: int | None = None,
    device: str = "cuda",
    resume: bool = True,
    rebuild_index: bool = True,
) -> RecipeKey:
    """Resolve a :class:`~seedcert.experiment.grid.ReproTarget` to its
    ``RecipeKey`` and call :func:`build_recipe`. Returns the key so the caller
    can open a :class:`~seedcert.cache.runs.RecipeRuns`."""
    recipe = target.recipe()
    recipe_key = RecipeKey(
        dataset=target.dataset,
        split_protocol=target.split_protocol(),
        backbone=target.backbone,
        recipe_hash=recipe.recipe_hash(),
    )
    build_recipe(
        recipe_key,
        recipe=recipe,
        registry=registry,
        n_seeds=n_seeds if n_seeds is not None else target.n_seeds,
        device=device,
        resume=resume,
        rebuild_index=rebuild_index,
    )
    return recipe_key


def main() -> None:
    """CLI: build one recipe's runs.

    ``seedcert-build-runs --dataset Cora --backbone gcn --overrides hidden_dim=16
    --seeds 0-49``
    """
    parser = argparse.ArgumentParser(prog="seedcert-build-runs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True, choices=("gcn", "gat", "sage"))
    parser.add_argument("--overrides", default="", help="k=v,k=v against OVERRIDABLE_FIELDS")
    parser.add_argument("--label", default="")
    parser.add_argument("--seeds", default="0-49", help="'0-N' (inclusive) or 'N'")
    parser.add_argument("--registry", default="run_cache")
    parser.add_argument("--lock", default="datasets.lock.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    from seedcert.data.datasets import split_protocol_for
    from seedcert.recipe import Recipe

    n_seeds = (int(args.seeds.split("-")[1]) + 1) if "-" in args.seeds else int(args.seeds)
    recipe = Recipe(args.backbone, overrides=_parse_overrides(args.overrides), label=args.label)
    recipe_key = RecipeKey(
        dataset=args.dataset,
        split_protocol=split_protocol_for(args.dataset),
        backbone=args.backbone,
        recipe_hash=recipe.recipe_hash(),
    )
    registry = RunRegistry(args.registry, lock_path=args.lock)
    build_recipe(
        recipe_key,
        recipe=recipe,
        registry=registry,
        n_seeds=n_seeds,
        device=args.device,
        resume=not args.no_resume,
    )
    df = registry.read_index()
    print(f"index.parquet: {len(df)} rows")
    cols = ["key_string", "test_accuracy", "f1", "wall_clock_s", "gpu_model"]
    print(df[df["recipe_hash"] == recipe_key.recipe_hash][cols].to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
