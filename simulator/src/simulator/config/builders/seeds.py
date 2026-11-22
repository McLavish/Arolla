"""Reproducible seeds for the simulator random generators."""

from __future__ import annotations

import hashlib


def get_stable_seed(base_seed: int, name: str) -> int:
    """Derive a deterministic seed from `base_seed`.

    md5 is used to mix the label into the base seed.
    Note that python built-in `hash()` is notoriously
    unstable across runs and versions, so it should not be used for this purpose.
    """
    label = f"{base_seed}:{name}".encode("utf-8")
    digest = hashlib.md5(label, usedforsecurity=False).digest()
    return int.from_bytes(digest[-4:], "big")  # last 4 bytes for a 32-bit seed
