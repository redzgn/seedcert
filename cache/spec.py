"""``RunKey`` / ``RecipeKey`` - the coordinates of one cached run (DESIGN Sec 6).

A run is identified by ``(dataset, split_protocol, backbone, recipe_hash,
seed)``. ``RecipeKey`` is a ``RunKey`` without ``seed`` and identifies one
:class:`~seedcert.cache.runs.RecipeRuns`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

BACKBONES: tuple[str, ...] = ("gcn", "gat", "sage")
_RECIPE_HASH_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True, slots=True)
class RecipeKey:
    """One recipe on one dataset + split protocol (a ``RunKey`` without seed)."""

    dataset: str
    split_protocol: str
    backbone: str
    recipe_hash: str

    def rel_path(self) -> Path:
        return Path(self.dataset, self.split_protocol, self.backbone, self.recipe_hash)

    def with_seed(self, seed: int) -> RunKey:
        return RunKey(
            dataset=self.dataset,
            split_protocol=self.split_protocol,
            backbone=self.backbone,
            recipe_hash=self.recipe_hash,
            seed=seed,
        )


@dataclass(frozen=True, slots=True)
class RunKey:
    """One cached run's identity."""

    dataset: str
    split_protocol: str
    backbone: str
    recipe_hash: str
    seed: int

    def validate(self) -> None:
        """Check ``backbone in BACKBONES``, ``recipe_hash`` is a 32-char lower
        hex digest, ``seed >= 0``.

        Raises:
            ValueError: on an out-of-range field.
        """
        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone {self.backbone!r} not in {BACKBONES}")
        if not _RECIPE_HASH_RE.match(self.recipe_hash):
            raise ValueError(f"recipe_hash {self.recipe_hash!r} is not 32 lowercase hex chars")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

    def key_string(self) -> str:
        """``{dataset}/{split_protocol}/{backbone}/{recipe_hash}/seed{seed}``."""
        return f"{self.without_seed().rel_path().as_posix()}/seed{self.seed}"

    def content_hash(self, *, dataset_sha256: str) -> str:
        """Short stable hash of the key fields bound to ``dataset_sha256`` - the
        run is stale if the underlying dataset graph changes. ``blake2b``,
        16-byte digest."""
        payload = f"{self.key_string()}|{dataset_sha256}".encode()
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def run_dir(self, root: Path) -> Path:
        """Directory holding this run's ``state_dict.pt`` / ``logits.npy`` /
        ``metrics.json`` / ``env.json``."""
        return root / self.without_seed().rel_path() / f"seed{self.seed}"

    def recipe_dir(self, root: Path) -> Path:
        """Directory holding the recipe-level artifacts shared by every seed
        (``recipe_manifest.json``, ``recipe.json``)."""
        return root / self.without_seed().rel_path()

    def without_seed(self) -> RecipeKey:
        return RecipeKey(
            dataset=self.dataset,
            split_protocol=self.split_protocol,
            backbone=self.backbone,
            recipe_hash=self.recipe_hash,
        )
