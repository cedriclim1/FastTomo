"""Volume-size scaling benchmark for FastTomo.

Sweeps cubic volume size N and times two cached reconstruction methods at a
fixed iteration count:

* ``ft-ad-with-pose-cached`` — Adam volume + Euler + shifts via
  :func:`benchmarks.ad_reconstruct.ad_reconstruct_with_pose_cached`.
* ``ft-sirt-cached`` — :func:`fasttomo.run_sirt` with a persistent
  ``ParallelProjector`` and positivity.

For each N the script generates a synthetic 3D Shepp-Logan-style phantom of
shape ``(N, N, N)``, forward-projects it through ``ft.forward_project`` at the
fixed tilt schedule, and reconstructs from that sinogram. The phantom is only
used to produce a realistic sinogram; reconstruction quality is not scored —
this is a pure timing/memory sweep.

Results are printed and written to ``benchmarks/scaling_results.json``.

Usage:
    conda run -n FastTomo python -m benchmarks.run_scaling
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import fasttomo as ft
from benchmarks.scripts.utils.ad_reconstruct import ad_reconstruct_with_pose_cached
from benchmarks.scripts.utils.timing import cuda_timer


def shepp_logan_3d(N: int, device: torch.device) -> torch.Tensor:
    """Synthetic 3D Shepp-Logan-style phantom of shape ``(N, N, N)``.

    Sum of axis-aligned ellipsoids in normalised coordinates ``[-1, 1]``.
    Values are clamped to ``[0, 1]`` so it behaves like a transmission phantom.
    Layout: one large background ellipsoid + a handful of smaller positive and
    negative inclusions. The exact values don't matter for timing; what matters
    is that the sinogram has the same structure as a real reconstruction
    target so the optimisers do real work each iteration.
    """
    coords = torch.linspace(-1.0, 1.0, N, device=device, dtype=torch.float32)
    y, z, x = torch.meshgrid(coords, coords, coords, indexing="ij")

    ellipsoids = [
        # (cy, cz, cx, ay, az, ax, value)
        (0.0,  0.0,  0.0,  0.92, 0.69, 0.90,  1.00),
        (0.0, -0.0184, 0.0, 0.874, 0.6624, 0.88, -0.80),
        (0.0,  0.0, -0.22, 0.31, 0.11, 0.41, -0.20),
        (0.0,  0.0,  0.22, 0.41, 0.16, 0.31, -0.20),
        (0.0,  0.35, 0.0,  0.21, 0.25, 0.25,  0.10),
        (0.0,  0.10, 0.0,  0.046, 0.046, 0.046, 0.10),
        (0.0, -0.10, 0.0,  0.046, 0.046, 0.046, 0.10),
        (-0.08, -0.605, 0.0, 0.046, 0.023, 0.046, 0.10),
        (0.0, -0.605, 0.0,  0.023, 0.023, 0.023, 0.10),
        (0.06, -0.605, 0.0, 0.023, 0.046, 0.023, 0.10),
    ]

    vol = torch.zeros((N, N, N), device=device, dtype=torch.float32)
    for cy, cz, cx, ay, az, ax, val in ellipsoids:
        mask = (
            ((y - cy) / ay) ** 2
            + ((z - cz) / az) ** 2
            + ((x - cx) / ax) ** 2
        ) <= 1.0
        vol = vol + val * mask.to(torch.float32)
    return vol.clamp_(min=0.0, max=1.0)


def make_problem(
    N: int, angles_deg: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Forward-project a synthetic phantom of size ``N`` to a sinogram.

    Returns sinogram of shape ``[n_angles, N, N]``. The phantom itself is
    discarded after projection — we don't need it past sinogram generation.
    """
    phantom = shepp_logan_3d(N, device)
    sino = ft.forward_project(phantom, angles_deg, N_det=N)
    del phantom
    torch.cuda.empty_cache()
    return sino


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400, 500],
        help="Cubic volume sizes N to sweep (one reconstruction per size per method).",
    )
    parser.add_argument("--ad-iter", type=int, default=100)
    parser.add_argument("--sirt-iter", type=int, default=100)
    parser.add_argument("--tv-weight", type=float, default=1.0)
    parser.add_argument(
        "--angle-start", type=float, default=-70.0,
        help="First tilt angle in degrees (inclusive).",
    )
    parser.add_argument(
        "--angle-stop", type=float, default=70.0,
        help="Last tilt angle in degrees (inclusive).",
    )
    parser.add_argument(
        "--angle-step", type=float, default=5.0,
        help="Tilt-angle step in degrees.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "scaling_results.json",
    )
    parser.add_argument(
        "--skip", choices=["ad", "sirt", "none"], default="none",
        help="Skip one method (useful when one OOMs at the largest size).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    ang_deg = torch.arange(
        args.angle_start, args.angle_stop + 0.5 * args.angle_step, args.angle_step,
        device=device, dtype=torch.float32,
    )
    n_angles = ang_deg.numel()
    print(
        f"angles: {n_angles} from {args.angle_start}° to {args.angle_stop}° "
        f"step {args.angle_step}°"
    )
    print(f"sizes : {args.sizes}")
    print(
        f"iters : ad-with-pose-cached={args.ad_iter}, sirt-cached={args.sirt_iter}\n"
    )

    rows: list[dict] = []

    def run_method(name: str, fn, N: int, n_iter: int) -> None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            with cuda_timer(device) as t:
                fn()
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            rows.append({
                "N": N,
                "method": name,
                "n_iter": n_iter,
                "time_s": t[0],
                "it_per_s": n_iter / t[0],
                "peak_mem_gb": peak_gb,
            })
            print(
                f"  {name:<24} N={N:<4} {t[0]:>8.2f}s "
                f"{n_iter / t[0]:>7.2f} it/s  peak {peak_gb:>5.2f} GB"
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"  {name:<24} N={N:<4} OOM ({e})")
            rows.append({
                "N": N, "method": name, "n_iter": n_iter,
                "time_s": None, "it_per_s": None, "peak_mem_gb": None,
                "oom": True,
            })

    for N in args.sizes:
        print(f"--- N = {N} ---")
        # Build sinogram once for this size; both methods reconstruct from it.
        try:
            sino = make_problem(N, ang_deg, device)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  sinogram generation OOM at N={N}: {e}")
            continue
        print(f"  sinogram {tuple(sino.shape)}")

        if args.skip != "ad":
            # Warm-up: 2-iter run to compile/cache before timing.
            _ = ad_reconstruct_with_pose_cached(
                sino, ang_deg, n_iter=2, tv_weight=args.tv_weight
            )
            run_method(
                "ft-ad-with-pose-cached",
                lambda: ad_reconstruct_with_pose_cached(
                    sino, ang_deg, n_iter=args.ad_iter, tv_weight=args.tv_weight
                ),
                N, args.ad_iter,
            )

        if args.skip != "sirt":
            proj = ft.ParallelProjector(
                (N, N, N), ang_deg, device=device, N_det=N
            )
            _ = ft.run_sirt(
                sino, ang_deg, n_iter=2, positivity=True, projector=proj
            )
            run_method(
                "ft-sirt-cached",
                lambda: ft.run_sirt(
                    sino, ang_deg, n_iter=args.sirt_iter,
                    positivity=True, projector=proj,
                ),
                N, args.sirt_iter,
            )
            del proj

        del sino
        torch.cuda.empty_cache()
        print()

    print("Results:")
    print(
        f"  {'N':>4} {'method':<24} {'n_iter':>7} {'time_s':>9} "
        f"{'it/s':>7} {'peak_GB':>8}"
    )
    for r in rows:
        if r.get("oom"):
            print(f"  {r['N']:>4} {r['method']:<24} {r['n_iter']:>7}      OOM")
            continue
        print(
            f"  {r['N']:>4} {r['method']:<24} {r['n_iter']:>7} "
            f"{r['time_s']:>9.2f} {r['it_per_s']:>7.2f} {r['peak_mem_gb']:>8.2f}"
        )

    payload = {
        "device": str(device),
        "angles_deg": ang_deg.cpu().tolist(),
        "ad_iter": args.ad_iter,
        "sirt_iter": args.sirt_iter,
        "tv_weight": args.tv_weight,
        "sizes": list(args.sizes),
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
