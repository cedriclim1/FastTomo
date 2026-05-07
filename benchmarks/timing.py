"""Timing harness ported from fasttomo-wiki/raw/code/TorchRadon/benchmark.py."""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch


def benchmark_function(f, x, samples: int, warmup: int, sync: bool = False) -> float:
    """Mean wall-clock time per call of ``f(x)`` over ``samples`` runs.

    Matches the convention from the TorchRadon benchmark: drop ``warmup`` runs,
    then time ``samples`` runs. ``sync=True`` brackets the timed block with
    ``torch.cuda.synchronize`` so async GPU launches are included in the time.
    """
    for _ in range(warmup):
        f(x)
    if sync:
        torch.cuda.synchronize()
    s = time.time()
    for _ in range(samples):
        f(x)
    if sync:
        torch.cuda.synchronize()
    e = time.time()
    return (e - s) / samples


@contextmanager
def cuda_timer(device: torch.device | str | None = None):
    """Context manager that yields a ``[elapsed_s]`` list, populated on exit.

    Use for one-shot timing of long blocks (e.g. a full reconstruction loop)
    where ``benchmark_function``'s repeat-and-average model isn't appropriate.
    Synchronises before and after so async GPU work is captured.
    """
    if device is not None and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
    out: list[float] = [float("nan")]
    s = time.time()
    try:
        yield out
    finally:
        if device is not None and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        out[0] = time.time() - s
