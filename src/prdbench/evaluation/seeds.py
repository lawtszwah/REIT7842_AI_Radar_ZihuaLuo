"""Shared, ordered seed protocol for evaluation and model selection."""

from __future__ import annotations

import numpy as np


def make_streams(root_seed: int, names, required=()) -> dict[str, np.random.SeedSequence]:
    """Append names only: a child's position, not its name, determines its seed."""
    if not names or not isinstance(names, (list, tuple)):
        raise ValueError("protocol.seed_streams is required as an ordered list of names")
    if any(not isinstance(name, str) for name in names):
        raise ValueError("protocol.seed_streams must contain string names")
    if len(set(names)) != len(names):
        raise ValueError("protocol.seed_streams must contain unique names")
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f"protocol.seed_streams missing required names: {sorted(missing)}")
    return dict(zip(names, np.random.SeedSequence(root_seed).spawn(len(names)), strict=True))


def stream_record(stream: np.random.SeedSequence, *, model: bool = False) -> dict:
    # Model initialization uses a native uint32 draw: sklearn's integer random_state
    # is limited to 32 bits, and this same integer also works with numpy and torch.
    dtype = np.uint32 if model else np.uint64
    return {
        "spawn_key": list(stream.spawn_key),
        "materialised_seed": int(stream.generate_state(1, dtype=dtype)[0]),
    }
