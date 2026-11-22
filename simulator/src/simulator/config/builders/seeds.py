"""Reproducible seeds for the simulator's random generators."""

from __future__ import annotations

import hashlib


def get_stable_seed(base_seed: int, name: str) -> int:
    """Derive a 32-bit seed from `base_seed` and a label.

    This is how each service and client gets its own random stream from the
    single `seed` of a scenario: the same (base_seed, name) always gives the
    same seed, on every run and every machine, and different labels give
    unrelated seeds.

    md5 is only used to mix the label into a number here, not for security.
    Python's built-in `hash()` would not do: it changes from one process to
    the next, so runs would not be reproducible.
    """
    label = f"{base_seed}:{name}".encode("utf-8")
    digest = hashlib.md5(label, usedforsecurity=False).digest()
    return int.from_bytes(digest[-4:], "big")  # last 4 bytes: a 32-bit seed
