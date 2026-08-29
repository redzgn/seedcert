"""On-disk run registry: layout, index, staleness (DESIGN Sec 6).

Structure adapted from ``certiforget``'s ``OracleRegistry``. One
``index.parquet`` row per completed run; :meth:`RunRegistry.rebuild_index`
enforces a single GPU model per recipe.

On-disk layout::

    run_cache/
      index.parquet
      <dataset>/<split_protocol>/<backbone>/<recipe_hash>/
        recipe.json              # Recipe.descriptor(), written by build
        recipe_manifest.json     # {"completed_seeds": [...], "hashes": {...}}
        seed<k>/
          state_dict.pt
          logits.npy             # [N, C] float32, full-graph eval, best weights
          metrics.json
          env.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seedcert.cache.spec import RunKey

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_ROOT = Path("run_cache")
DEFAULT_LOCK = Path("datasets.lock.json")


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One row of ``index.parquet``: key fields + pointers + the metrics needed
    to select and audit runs without walking the tree."""

    key_string: str
    dataset: str
    split_protocol: str
    backbone: str
    recipe_hash: str
    seed: int
    content_hash: str
    state_dict_path: str
    logits_path: str
    test_accuracy: float
    precision: float
    recall: float
    f1: float
    val_accuracy: float
    early_stop_epoch: int
    wall_clock_s: float
    gpu_model: str


class RunRegistry:
    """Read/write access to a ``run_cache/`` tree (DESIGN Sec 6)."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        *,
        lock_path: str | Path = DEFAULT_LOCK,
    ) -> None:
        self.root = Path(root)
        self.lock_path = Path(lock_path)

    # --- paths ---
    @property
    def index_path(self) -> Path:
        return self.root / "index.parquet"

    def run_dir(self, key: RunKey) -> Path:
        return key.run_dir(self.root)

    def recipe_dir(self, key: RunKey) -> Path:
        return key.recipe_dir(self.root)

    # --- integrity ---
    def load_lock(self) -> dict[str, Any]:
        """Parse ``datasets.lock.json`` (sha256 per dataset + manifest hash)."""
        payload = json.loads(self.lock_path.read_text())
        assert isinstance(payload, dict)
        return payload

    def dataset_sha256(self, dataset: str) -> str:
        """The pinned sha256 for ``dataset`` from the lock file.

        Raises:
            KeyError: no (non-null) sha256 recorded for ``dataset``.
        """
        entry = self.load_lock().get("datasets", {}).get(dataset, {})
        sha = entry.get("sha256")
        if not sha:
            raise KeyError(
                f"no sha256 for {dataset!r} in {self.lock_path} - run `seedcert-manifest` first"
            )
        return str(sha)

    def is_stale(self, key: RunKey) -> bool:
        """``True`` if the run's stored ``content_hash`` no longer matches the
        current dataset sha256 (or the run is absent)."""
        metrics = self.run_dir(key) / "metrics.json"
        if not metrics.exists():
            return True
        stored = json.loads(metrics.read_text()).get("content_hash")
        expected = key.content_hash(dataset_sha256=self.dataset_sha256(key.dataset))
        return bool(stored != expected)

    def completed_seeds(self, key: RunKey) -> list[int]:
        """Seeds with a complete artifact set for ``key``'s recipe."""
        manifest = self.recipe_dir(key) / "recipe_manifest.json"
        if not manifest.exists():
            return []
        raw = json.loads(manifest.read_text()).get("completed_seeds", [])
        return sorted(int(s) for s in raw)

    # --- records / index ---
    def write_record(self, record: RunRecord) -> None:
        """Append or replace the row for ``record.key_string`` in the index."""
        import pandas as pd

        new = pd.DataFrame([asdict(record)])
        if self.index_path.exists():
            existing = pd.read_parquet(self.index_path)
            existing = existing[existing["key_string"] != record.key_string]
            new = pd.concat([existing, new], ignore_index=True)
        self.root.mkdir(parents=True, exist_ok=True)
        new.sort_values("key_string").to_parquet(self.index_path, index=False)

    def read_index(self) -> pd.DataFrame:
        """Load ``index.parquet`` (one row per completed run)."""
        import pandas as pd

        if not self.index_path.exists():
            return pd.DataFrame(columns=list(RunRecord.__dataclass_fields__))
        return pd.read_parquet(self.index_path)

    def rebuild_index(self) -> None:
        """Walk the tree and regenerate ``index.parquet`` from every
        ``seed*/metrics.json`` + ``env.json``.

        Raises:
            RuntimeError: a recipe whose ``env.json`` files disagree on
                ``gpu_model`` (DESIGN Sec 6).
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        by_recipe: dict[Path, set[str]] = {}
        for metrics_path in sorted(self.root.rglob("seed*/metrics.json")):
            seed_dir = metrics_path.parent
            metrics = json.loads(metrics_path.read_text())
            env = json.loads((seed_dir / "env.json").read_text())
            by_recipe.setdefault(seed_dir.parent, set()).add(str(env.get("gpu_model")))
            rows.append(
                {
                    "key_string": metrics["key_string"],
                    "dataset": metrics["dataset"],
                    "split_protocol": metrics["split_protocol"],
                    "backbone": metrics["backbone"],
                    "recipe_hash": metrics["recipe_hash"],
                    "seed": metrics["seed"],
                    "content_hash": metrics["content_hash"],
                    "state_dict_path": str((seed_dir / "state_dict.pt").relative_to(self.root)),
                    "logits_path": str((seed_dir / "logits.npy").relative_to(self.root)),
                    "test_accuracy": metrics["test_accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "val_accuracy": metrics["val_accuracy"],
                    "early_stop_epoch": metrics["early_stop_epoch"],
                    "wall_clock_s": metrics["wall_clock_s"],
                    "gpu_model": env.get("gpu_model"),
                }
            )
        mixed = {d: g for d, g in by_recipe.items() if len(g) > 1}
        if mixed:
            detail = "; ".join(
                f"{d.relative_to(self.root)}: {sorted(g)}" for d, g in mixed.items()
            )
            raise RuntimeError(f"mixed GPU models within a recipe (DESIGN Sec 6): {detail}")

        self.root.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows, columns=list(RunRecord.__dataclass_fields__))
        frame.sort_values("key_string").to_parquet(self.index_path, index=False)
