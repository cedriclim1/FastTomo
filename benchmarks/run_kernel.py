"""Kernel-level microbenchmark: FastTomo FP/BP vs TorchRadon FP/BP.

Isolates pure projection-kernel cost from AD-loop overhead (Adam, TV, autograd
graph). Used to validate the per-call cudaArray allocation hypothesis before
implementing the texture-cache refactor.

Reports mean wall-clock per call (warmup + sync) on the same shape as the
production benchmark (200³ volume, 29 angles by default; override with
``--shape`` and ``--n-angles``). Sample count is configurable via ``--samples``.

Usage:
    conda run -n FastTomo-bench python -m benchmarks.run_kernel
    conda run -n FastTomo-bench python -m benchmarks.run_kernel --shape 256 --n-angles 60
"""

from __future__ import annotations

import argparse
import importlib
import math

import numpy as np
import torch

import fasttomo as ft
from benchmarks.timing import benchmark_function


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        print(f"  [skipped: {name} unavailable: {type(e).__name__}: {e}]")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shape", type=int, default=200, help="cubic volume edge length")
    parser.add_argument("--n-angles", type=int, default=29)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    N = args.shape
    n_angles = args.n_angles

    vol = torch.randn(N, N, N, device=device, dtype=torch.float32)
    sino = torch.randn(n_angles, N, N, device=device, dtype=torch.float32)
    angles_deg = torch.linspace(-70.0, 70.0, n_angles, device=device, dtype=torch.float32)
    angles_rad = (angles_deg * math.pi / 180.0).cpu().numpy().astype(np.float32)

    print(
        f"shape={N}^3, n_angles={n_angles}, samples={args.samples} "
        f"(warmup={args.warmup}), device={device}"
    )

    rows: list[dict] = []

    def record(name: str, t: float) -> None:
        rows.append({"op": name, "ms": t * 1e3, "calls_per_s": 1.0 / t})
        print(f"  {name:<30} {t*1e3:8.3f} ms   {1.0/t:8.1f} calls/s")

    # --- FastTomo (current, uncached path) ---
    print("\n[FastTomo] uncached (current)")

    def ft_fp(_):
        return ft.forward_project(vol, angles_deg, N_det=N)

    def ft_bp(_):
        return ft.back_project(sino, angles_deg, Nz=N)

    record("ft-forward_project", benchmark_function(ft_fp, None, args.samples, args.warmup, sync=True))
    record("ft-back_project",    benchmark_function(ft_bp, None, args.samples, args.warmup, sync=True))

    # --- FastTomo (ParallelProjector, cached) ---
    print("\n[FastTomo] cached (ParallelProjector)")
    proj = ft.ParallelProjector((N, N, N), angles_deg, device=device)

    def ft_fp_cached(_):
        return proj.forward(vol)

    def ft_bp_cached(_):
        return proj.back(sino)

    record("ft-projector.forward", benchmark_function(ft_fp_cached, None, args.samples, args.warmup, sync=True))
    record("ft-projector.back",    benchmark_function(ft_bp_cached, None, args.samples, args.warmup, sync=True))

    # --- TorchRadon ---
    print("\n[TorchRadon]")
    if _try_import("torch_radon") is not None:
        from torch_radon import Radon

        radon = Radon(N, angles_rad, clip_to_circle=False)
        # TorchRadon is 2D-batched: [batch=Ny, H=N, W=N] in, [Ny, n_angles, N_det] out.
        vol_b = vol.contiguous()                                # [Ny=N, H=N, W=N]
        sino_b = sino.permute(1, 0, 2).contiguous()             # [Ny, n_angles, N_det]

        def tr_fp(_):
            return radon.forward(vol_b)

        def tr_bp(_):
            return radon.backprojection(sino_b)

        record("tr-forward",          benchmark_function(tr_fp, None, args.samples, args.warmup, sync=True))
        record("tr-backprojection",   benchmark_function(tr_bp, None, args.samples, args.warmup, sync=True))

    # --- summary table ---
    print("\nSummary:")
    print(f"  {'op':<30} {'ms/call':>10} {'calls/s':>12}")
    for r in rows:
        print(f"  {r['op']:<30} {r['ms']:>10.3f} {r['calls_per_s']:>12.1f}")

    # cross-backend ratios for FP and BP if both ran
    by_op = {r["op"]: r["ms"] for r in rows}
    if "ft-forward_project" in by_op and "ft-projector.forward" in by_op:
        fp_speedup = by_op["ft-forward_project"] / by_op["ft-projector.forward"]
        bp_speedup = by_op["ft-back_project"]    / by_op["ft-projector.back"]
        print(f"\n  FP cache speedup: uncached / cached = {fp_speedup:.2f}x")
        print(f"  BP cache speedup: uncached / cached = {bp_speedup:.2f}x")
    if "ft-forward_project" in by_op and "tr-forward" in by_op:
        fp_ratio = by_op["ft-forward_project"]   / by_op["tr-forward"]
        bp_ratio = by_op["ft-back_project"]      / by_op["tr-backprojection"]
        print(f"\n  FP gap (uncached): FastTomo / TorchRadon = {fp_ratio:.2f}x")
        print(f"  BP gap (uncached): FastTomo / TorchRadon = {bp_ratio:.2f}x")
    if "ft-projector.forward" in by_op and "tr-forward" in by_op:
        fp_ratio_c = by_op["ft-projector.forward"] / by_op["tr-forward"]
        bp_ratio_c = by_op["ft-projector.back"]    / by_op["tr-backprojection"]
        print(f"  FP gap (cached):   FastTomo / TorchRadon = {fp_ratio_c:.2f}x")
        print(f"  BP gap (cached):   FastTomo / TorchRadon = {bp_ratio_c:.2f}x")


if __name__ == "__main__":
    main()
