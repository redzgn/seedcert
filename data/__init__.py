"""Data surface: the canonical published-split loader and the manifest helpers."""

from __future__ import annotations

from seedcert.data.datasets import (
    CANONICAL_SOURCES,
    SPLIT_PROTOCOLS,
    load_canonical,
    split_protocol_for,
)
from seedcert.data.manifest import resolve_num_classes

__all__ = [
    "load_canonical",
    "split_protocol_for",
    "CANONICAL_SOURCES",
    "SPLIT_PROTOCOLS",
    "resolve_num_classes",
]
