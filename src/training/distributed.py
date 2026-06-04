"""Distributed training helpers for accelerate/DDP launches."""

from __future__ import annotations

import os


def process_rank() -> int:
    """Return the current distributed process rank, defaulting to rank 0."""
    for name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def world_size() -> int:
    """Return distributed world size, defaulting to 1."""
    value = os.environ.get("WORLD_SIZE")
    if value is None:
        return 1
    try:
        return int(value)
    except ValueError:
        return 1


def is_main_process() -> bool:
    """Whether this process should perform side effects such as logging."""
    return process_rank() == 0


def is_distributed() -> bool:
    """Whether the current process appears to be part of a distributed job."""
    return world_size() > 1
