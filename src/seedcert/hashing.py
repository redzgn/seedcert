"""Content hashing shared across the package.

A forget set, a control pool, and a certificate all identify a node set by the
SHA-256 of its sorted indices - keep that in one place so a certificate's
``forget_set_hash`` matches its cell's ``forget_set.json``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable


def hash_node_indices(indices: Iterable[int]) -> str:
    """SHA-256 hex digest of ``sorted(int(i) for i in indices)``."""
    payload = json.dumps(sorted(int(i) for i in indices)).encode()
    return hashlib.sha256(payload).hexdigest()
