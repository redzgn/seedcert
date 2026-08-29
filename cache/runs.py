"""``RecipeRuns`` - read access to one recipe's cached seed runs (DESIGN Sec 6).

A ``RecipeRuns`` is the cached seed ensemble for one
``(dataset, split_protocol, backbone, recipe_hash)``. It is what
:meth:`~seedcert.verifiers.reproduction.ReproductionVerifier.certify` reads.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seedcert.cache.spec import RecipeKey, RunKey

if TYPE_CHECKING:
    import numpy as np
    import torch

    from seedcert.cache.registry import RunRegistry


@dataclass(frozen=True, slots=True)
class Run:
    """One cached run, loaded lazily from disk."""

    key: RunKey
    state_dict_path: Path
    logits_path: Path
    metrics: dict[str, object]
    env: dict[str, object]

    def load_state_dict(self) -> dict[str, torch.Tensor]:
        import torch

        state: dict[str, torch.Tensor] = torch.load(
            self.state_dict_path, map_location="cpu", weights_only=True
        )
        return state

    def load_logits(self) -> np.ndarray:
        import numpy as np

        arr: np.ndarray = np.load(self.logits_path)
        return arr


class RecipeRuns:
    """Every cached run for one recipe on one dataset + split protocol.
    Iterable and sized."""

    def __init__(self, registry: RunRegistry, recipe: RecipeKey) -> None:
        self.registry = registry
        self.recipe = recipe

    @classmethod
    def load(
        cls,
        registry: RunRegistry,
        *,
        dataset: str,
        split_protocol: str,
        backbone: str,
        recipe_hash: str,
    ) -> RecipeRuns:
        return cls(registry, RecipeKey(dataset, split_protocol, backbone, recipe_hash))

    @property
    def dataset(self) -> str:
        return self.recipe.dataset

    @property
    def split_protocol(self) -> str:
        return self.recipe.split_protocol

    @property
    def recipe_dir(self) -> Path:
        return self.registry.root / self.recipe.rel_path()

    @cached_property
    def _runs(self) -> tuple[Run, ...]:
        found: list[Run] = []
        for metrics_path in sorted(self.recipe_dir.glob("seed*/metrics.json")):
            seed_dir = metrics_path.parent
            metrics = json.loads(metrics_path.read_text())
            env = json.loads((seed_dir / "env.json").read_text())
            found.append(
                Run(
                    key=self.recipe.with_seed(int(metrics["seed"])),
                    state_dict_path=seed_dir / "state_dict.pt",
                    logits_path=seed_dir / "logits.npy",
                    metrics=metrics,
                    env=env,
                )
            )
        found.sort(key=lambda r: r.key.seed)
        return tuple(found)

    def __len__(self) -> int:
        return len(self._runs)

    def __iter__(self) -> Iterator[Run]:
        return iter(self._runs)

    @property
    def n_seeds(self) -> int:
        return len(self._runs)

    @property
    def seed_list(self) -> tuple[int, ...]:
        return tuple(r.key.seed for r in self._runs)

    def metric_values(self, name: str) -> np.ndarray:
        """The per-seed values of metric ``name`` (one entry per seed, ordered by
        :attr:`seed_list`), read from each run's ``metrics.json``.

        Raises:
            KeyError: ``name`` is not one of the cached metric keys.
            ValueError: no runs are cached for this recipe.
        """
        import numpy as np

        if not self._runs:
            raise ValueError(f"no cached runs under {self.recipe_dir}")
        missing = [r.key.seed for r in self._runs if name not in r.metrics]
        if missing:
            raise KeyError(
                f"metric {name!r} not in cached metrics (seeds {missing}); have "
                f"{sorted(self._runs[0].metrics)}"
            )
        return np.array([r.metrics[name] for r in self._runs], dtype=float)

    def logits_stack(self) -> np.ndarray:
        """``[n_seeds, N, C]`` stack of the cached logits."""
        import numpy as np

        return np.stack([r.load_logits() for r in self._runs])

    def recipe_descriptor(self) -> dict[str, Any]:
        """``{"backbone", "overrides", "label"}`` from ``recipe.json``; falls
        back to a backbone-only descriptor if the file is absent."""
        path = self.recipe_dir / "recipe.json"
        if path.exists():
            payload: dict[str, Any] = json.loads(path.read_text())
            return payload
        return {"backbone": self.recipe.backbone, "overrides": {}, "label": ""}

    def registry_key_strings(self) -> tuple[str, ...]:
        return tuple(r.key.key_string() for r in self._runs)
