"""Centralized seeding and seed-provenance records.

The seed distribution must reflect real training stochasticity, so cuDNN is
**not** put into deterministic mode. Only the entropy sources are seeded, per
``seed``, and what was done is recorded (DESIGN D7).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SeedProvenance:
    """What :func:`seed_everything` actually did, for the oracle ``env.json``."""

    train_seed: int
    seeded: tuple[str, ...]
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    torch_num_threads: int | None


def seed_everything(train_seed: int) -> SeedProvenance:
    """Seed ``random``, ``numpy``, and ``torch`` (CPU + CUDA) from ``train_seed``.

    cuDNN determinism is left OFF and benchmark left OFF: the intent is
    reproducible *seeding*, not bit-reproducible kernels. Call once at the start
    of each training run.
    """
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(train_seed)
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    seeded = ["random", "numpy", "torch"]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
        seeded.append("torch.cuda")

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    return SeedProvenance(
        train_seed=train_seed,
        seeded=tuple(seeded),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        torch_num_threads=int(torch.get_num_threads()),
    )


def resampling_generator(rng: int | Any) -> Any:
    """Coerce ``rng`` (an int seed or an existing ``numpy.random.Generator``) to a
    generator for bootstrap / permutation resampling."""
    import numpy as np

    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
