"""
SIRT CUDA-Graph throughput benchmark.

Compares eager vs graph-captured run_sirt wall-clock on a 200^3 volume
with 1000 iterations.  Uses torch.cuda.Event for device-accurate timing.

Skip automatically when no CUDA GPU is available.
"""

from __future__ import annotations

import sys

import torch

try:
    import fasttomo as ft
except ImportError:
    print("fasttomo not installed; skipping benchmark.")
    sys.exit(0)

if not torch.cuda.is_available():
    print("No CUDA GPU available; skipping benchmark.")
    sys.exit(0)


DEVICE   = torch.device("cuda")
N        = 200
N_ANGLES = 60
N_ITER   = 1000


def _time_sirt(sino, angles, proj, use_graph: bool, n_iter: int) -> float:
    """Return wall-clock time in milliseconds for one run_sirt call."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    # Pass return_loss=True to force eager; default (False) uses graph when available.
    return_loss = not use_graph

    torch.cuda.synchronize()
    start.record()
    result = ft.run_sirt(
        sino, angles,
        n_iter=n_iter,
        Nz=N,
        projector=proj,
        return_loss=return_loss,
    )
    end.record()
    torch.cuda.synchronize()

    # Discard return_loss tuple
    if isinstance(result, tuple):
        result = result[0]

    return start.elapsed_time(end)  # milliseconds


def main() -> None:
    print(f"SIRT CUDA-Graph benchmark  |  volume {N}^3  |  {N_ANGLES} angles  |  {N_ITER} iters")
    print()

    torch.manual_seed(0)
    vol     = torch.randn(1, N, N, device=DEVICE)
    angles  = torch.linspace(-70.0, 70.0, N_ANGLES, device=DEVICE)
    sino    = ft.forward_project(vol, angles)

    # Build a persistent projector so both paths share the same texture cache.
    proj = ft.ParallelProjector((1, N, N), angles, device=DEVICE)

    # --- warmup (not timed) ---
    print("Warming up projector textures …")
    _ = ft.run_sirt(sino, angles, n_iter=2, Nz=N, projector=proj)
    torch.cuda.synchronize()

    # --- eager ---
    print("Timing eager path (return_loss=True) …")
    t_eager = _time_sirt(sino, angles, proj, use_graph=False, n_iter=N_ITER)

    # --- graph ---
    # First call builds the graph; second call reuses the cached graph.
    # We time the second call so we measure pure replay overhead.
    print("Building CUDA Graph (first call) …")
    _ = ft.run_sirt(sino, angles, n_iter=N_ITER, Nz=N, projector=proj)
    torch.cuda.synchronize()

    print("Timing graph path (cached graph replay) …")
    t_graph = _time_sirt(sino, angles, proj, use_graph=True, n_iter=N_ITER)

    speedup = t_eager / t_graph if t_graph > 0 else float("inf")

    print()
    print(f"  Eager  : {t_eager:8.1f} ms")
    print(f"  Graph  : {t_graph:8.1f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print()

    if speedup >= 1.0:
        print("CUDA Graph path is faster.")
    else:
        print("CUDA Graph path is NOT faster on this configuration (overhead > saving).")


if __name__ == "__main__":
    main()
