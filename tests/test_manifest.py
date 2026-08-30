"""Manifest helpers (DESIGN Sec 6, WP1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from seedcert.data.manifest import (
    generate_manifest,
    resolve_num_classes,
    sha256_of_graph,
    write_lock,
)


def test_resolve_num_classes_ok() -> None:
    assert resolve_num_classes(torch.tensor([0, 1, 2, 1, 0])) == 3


def test_resolve_num_classes_rejects_gap_and_nonzero_start() -> None:
    with pytest.raises(ValueError, match="start at 0"):
        resolve_num_classes(torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError, match="contiguous"):
        resolve_num_classes(torch.tensor([0, 1, 3]))


def test_sha256_of_graph_is_deterministic_and_shape_sensitive() -> None:
    x = torch.arange(12, dtype=torch.float).reshape(4, 3)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    y = torch.tensor([0, 1, 0, 1])
    h1 = sha256_of_graph(x, ei, y)
    h2 = sha256_of_graph(x.clone(), ei.clone(), y.clone())
    assert h1 == h2
    assert sha256_of_graph(x.reshape(3, 4), ei, y) != h1
    assert sha256_of_graph(x, ei, y + 0) == h1
    assert sha256_of_graph(x, ei, torch.tensor([1, 0, 1, 0])) != h1


def test_generate_manifest_records_load_errors(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    m = generate_manifest(datasets=("NotADataset",), out_path=out)
    assert "error" in m["datasets"]["NotADataset"]
    assert json.loads(out.read_text())["datasets"]["NotADataset"]["error"]


def test_write_lock_fills_only_successful_entries(tmp_path: Path) -> None:
    lock = tmp_path / "datasets.lock.json"
    lock.write_text(
        json.dumps(
            {"datasets": {"Cora": {"split_protocol": "planetoid-public", "sha256": None,
                                   "num_classes": None}}}
        )
    )
    manifest = {
        "datasets": {"Cora": {"sha256": "abc", "num_classes": 7}},
    }
    write_lock(manifest, out_path=lock)
    filled = json.loads(lock.read_text())
    assert filled["datasets"]["Cora"]["sha256"] == "abc"
    assert filled["datasets"]["Cora"]["num_classes"] == 7
    assert filled["generated_manifest_sha256"]
