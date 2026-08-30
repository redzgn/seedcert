"""Stage 1 gate: the package and every submodule import cleanly."""

from __future__ import annotations

import importlib
import pkgutil

import seedcert


def test_public_surface() -> None:
    for name in seedcert.__all__:
        assert hasattr(seedcert, name), name
    assert seedcert.__version__ == "0.1.0"


def test_every_submodule_imports() -> None:
    pkg = seedcert
    failed: list[str] = []
    for mod in pkgutil.walk_packages(pkg.__path__, prefix="seedcert."):
        try:
            importlib.import_module(mod.name)
        except NotImplementedError:  # pragma: no cover - defensive
            failed.append(f"{mod.name}: NotImplementedError at import time")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{mod.name}: {exc!r}")
    assert not failed, "\n".join(failed)
