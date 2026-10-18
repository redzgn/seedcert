"""Environment capture for reproducibility records.

Every oracle's ``env.json`` and every ``Certificate.env`` is produced here so
provenance is uniform across the package (DESIGN Sec 2.1, Sec 7.2).
"""

from __future__ import annotations

import functools
import platform as _platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ENV_KEYS: tuple[str, ...] = (
    "python",
    "torch",
    "torch_geometric",
    "cuda",
    "cudnn",
    "numpy",
    "scipy",
    "networkx",
    "pandas",
    "package_version",
    "git_sha",
    "gpu_model",
    "gpu_driver",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "hostname",
    "platform",
)


@functools.lru_cache(maxsize=4)
def git_sha(short: bool = True) -> str:
    """Repo commit SHA, or ``"unknown"`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out[:12] if short else out


@functools.lru_cache(maxsize=1)
def _gpu_driver() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.splitlines()[0] if out else None


def _package_version() -> str:
    try:
        from seedcert import __version__

        return __version__
    except Exception:
        return "unknown"


@functools.lru_cache(maxsize=1)
def _static_env() -> tuple[tuple[str, Any], ...]:
    """The parts of the environment that do not change within a process run
    (versions, git SHA, GPU model/driver, host). Cached so a full-grid run does
    not shell out to nvidia-smi / git thousands of times."""
    import networkx
    import numpy
    import pandas
    import scipy
    import torch
    import torch_geometric

    cuda_ok = torch.cuda.is_available()
    return tuple(
        {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "cuda": torch.version.cuda if cuda_ok else None,
            "cudnn": torch.backends.cudnn.version() if cuda_ok else None,  # type: ignore[no-untyped-call]
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "networkx": networkx.__version__,
            "pandas": pandas.__version__,
            "package_version": _package_version(),
            "git_sha": git_sha(),
            "gpu_model": torch.cuda.get_device_name(0) if cuda_ok else None,
            "gpu_driver": _gpu_driver(),
            "hostname": socket.gethostname(),
            "platform": _platform.platform(),
        }.items()
    )


def capture_environment() -> dict[str, Any]:
    """Return a dict keyed by :data:`ENV_KEYS`.

    On GPU, ``gpu_model`` is device 0's name; oracle draws are single-GPU by
    policy and the registry rejects draws that disagree on it (DESIGN Sec 9.12).
    ``cudnn_deterministic`` is expected to be ``False`` for oracle training. The
    static parts are cached (:func:`_static_env`); only the cuDNN flags are
    re-read, since they can be toggled per run.
    """
    import torch

    env: dict[str, Any] = dict(_static_env())
    env["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    env["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    return env
