"""Generate ``seedcert``'s dataset manifest and the ``datasets.lock.json`` pins
(DESIGN Sec 6).

Class counts are not inherited: each dataset is loaded once and
``num_classes = int(y.max()) + 1`` is computed directly, with a contiguity
check. The manifest is over the *canonical* graphs (the primary data path).
Datasets that fail to load (e.g. not yet downloaded) are recorded with an
``error`` and left unpinned.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

GENERATED_MANIFEST_PATH = Path("generated_manifest.json")
LOCK_PATH = Path("datasets.lock.json")


def resolve_num_classes(y: torch.Tensor) -> int:
    """``int(y.max()) + 1`` after checking labels are ``0..max`` contiguous.

    Raises:
        ValueError: labels do not start at 0, or the label set has gaps.
    """
    import torch

    flat = y.view(-1).long()
    y_min = int(flat.min())
    if y_min != 0:
        raise ValueError(f"labels do not start at 0 (min={y_min})")
    y_max = int(flat.max())
    n_unique = int(torch.unique(flat).numel())
    if n_unique != y_max + 1:
        raise ValueError(f"labels are not contiguous 0..{y_max} (found {n_unique} distinct)")
    return y_max + 1


def sha256_of(path: Path) -> str:
    """Streaming SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_graph(x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor) -> str:
    """Stable SHA-256 over a canonical graph's tensors (features, canonicalized
    edges, labels) - the pin recorded in ``datasets.lock.json``. Shapes are
    folded in so a reshape cannot collide."""
    h = hashlib.sha256()
    for name, t in (("x", x), ("edge_index", edge_index), ("y", y)):
        arr = t.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def generate_manifest(
    *,
    datasets: tuple[str, ...],
    out_path: Path = GENERATED_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load each canonical dataset once and write a manifest: node/edge/feature
    counts, ``num_classes`` and contiguity, split sizes, the split-protocol
    string, and a graph sha256. Load failures are captured, not raised."""
    import torch
    import torch_geometric

    from seedcert.data.datasets import load_canonical, split_protocol_for

    entries: dict[str, Any] = {}
    for name in datasets:
        try:
            d = load_canonical(name)
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            entries[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        y = d.y.view(-1).long()
        try:
            n_classes = resolve_num_classes(y)
            contiguous = True
        except ValueError:
            n_classes = int(y.max()) + 1
            contiguous = False
        entries[name] = {
            "split_protocol": split_protocol_for(name),
            "num_nodes": int(d.num_nodes),
            "num_features": int(d.num_features),
            "num_classes": int(n_classes),
            "labels_contiguous": contiguous,
            "num_edges_raw": int(d.num_edges_raw),
            "num_edges_canonical": int(d.num_edges_canonical),
            "edge_count_changed": bool(d.edge_count_changed),
            "was_symmetric": bool(d.was_symmetric),
            "split": {
                "train": int(d.train_mask.sum()),
                "val": int(d.val_mask.sum()),
                "test": int(d.test_mask.sum()),
            },
            "sha256": sha256_of_graph(d.x, d.edge_index, d.y),
        }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "datasets": entries,
    }
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_lock(
    generated_manifest: dict[str, Any],
    *,
    out_path: Path = LOCK_PATH,
) -> None:
    """Fill each dataset's ``sha256`` / ``num_classes`` in the lock file from the
    manifest (skipping entries that carry an ``error``) and stamp
    ``generated_manifest_sha256``."""
    lock = json.loads(out_path.read_text())
    gen: dict[str, Any] = generated_manifest["datasets"]
    for name, entry in lock.get("datasets", {}).items():
        g = gen.get(name)
        if not g or "error" in g:
            continue
        entry["sha256"] = g["sha256"]
        entry["num_classes"] = g["num_classes"]
    lock["generated_manifest_path"] = str(GENERATED_MANIFEST_PATH)
    lock["generated_manifest_sha256"] = hashlib.sha256(
        (json.dumps(generated_manifest, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    out_path.write_text(json.dumps(lock, indent=2) + "\n")


def main() -> None:
    """Regenerate ``generated_manifest.json`` + ``datasets.lock.json`` and print
    a review table."""
    from seedcert.experiment.grid import GRID_DATASETS

    manifest = generate_manifest(datasets=tuple(GRID_DATASETS))
    write_lock(manifest)

    header = f"{'dataset':16}{'nodes':>9}{'e_canon':>10}{'feat':>7}{'cls':>5}  split tr/va/te"
    print(header)
    print("-" * len(header))
    for name, e in manifest["datasets"].items():
        if "error" in e:
            print(f"{name:16}  ERROR: {e['error']}")
            continue
        s = e["split"]
        print(
            f"{name:16}{e['num_nodes']:>9}{e['num_edges_canonical']:>10}"
            f"{e['num_features']:>7}{e['num_classes']:>5}  {s['train']}/{s['val']}/{s['test']}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
