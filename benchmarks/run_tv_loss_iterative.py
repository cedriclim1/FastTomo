"""End-to-end speedup of the fused TV kernel on AD reconstruction loops.

Runs the four AD reconstruction variants from ``benchmarks/ad_reconstruct``
twice each — once with the original pure-PyTorch ``_tv_charbonnier``
expression, once with the fused ``ft.tv_loss_3d`` kernel. Same dataset,
same iterations, same hyperparameters, same seed. Reports wall-time and
SSIM/NMSE side-by-side so the regularizer swap shows up cleanly.

The four variants:

* ``ad_no_pose`` — ``forward_project`` (volume-only Adam)
* ``ad_no_pose_cached`` — ``ParallelProjector`` (cached volume-only Adam)
* ``ad_with_pose`` — ``forward_project_3d`` (volume + Euler + shifts)
* ``ad_with_pose_cached`` — ``RotmatProjector`` (cached volume + Euler + shifts)

Pairs with ``benchmarks/run_tv_loss.py`` (per-call kernel microbench) and
the ``ft-ad-*`` rows of ``benchmarks/run_iterative.py``.

Usage:
    conda run -n FastTomo python -m benchmarks.run_tv_loss_iterative
    conda run -n FastTomo python -m benchmarks.run_tv_loss_iterative --ad-iter 500
"""

from __future__ import annotations

import argparse

import torch

from benchmarks.ad_reconstruct import (
    _tv_charbonnier,
    _tv_fused,
    ad_reconstruct,
    ad_reconstruct_cached,
    ad_reconstruct_with_pose,
    ad_reconstruct_with_pose_cached,
)
from benchmarks.datasets import load_phantom_tilt_series
from benchmarks.metrics import nmse, ssim
from benchmarks.timing import cuda_timer


VARIANTS = [
    ("ad-no-pose",          ad_reconstruct),
    ("ad-no-pose-cached",   ad_reconstruct_cached),
    ("ad-with-pose",        ad_reconstruct_with_pose),
    ("ad-with-pose-cached", ad_reconstruct_with_pose_cached),
]

TV_IMPLS = [
    ("pytorch", _tv_charbonnier),
    ("fused",   _tv_fused),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ad-iter", type=int, default=1000)
    parser.add_argument("--tv-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = torch.device(args.device)
    phantom, sino, ang_deg = load_phantom_tilt_series(device=device)
    n_angles, Ny, N_det = sino.shape
    print(f"sinogram {(n_angles, Ny, N_det)}, phantom {tuple(phantom.shape)}")
    print(f"ad_iter={args.ad_iter}  tv_weight={args.tv_weight}  device={device}\n")

    # warm-up: a couple of iters of each loop on a single Y-slice so kernel
    # JIT and texture-cache initialization don't end up in the timed window
    print("warming up …")
    for variant_name, fn in VARIANTS:
        for _, tv_fn in TV_IMPLS:
            torch.manual_seed(args.seed)
            _ = fn(sino[:, :1, :], ang_deg, n_iter=2, tv_weight=args.tv_weight,
                   tv_fn=tv_fn)
    torch.cuda.synchronize(device)

    rows: list[dict] = []
    for variant_name, fn in VARIANTS:
        print(f"\n── {variant_name} ──")
        per_variant: list[dict] = []
        for tv_label, tv_fn in TV_IMPLS:
            torch.manual_seed(args.seed)
            with cuda_timer(device) as t:
                res = fn(sino, ang_deg, n_iter=args.ad_iter,
                         tv_weight=args.tv_weight, tv_fn=tv_fn)
            ips = args.ad_iter / t[0]
            ssim_v = ssim(res.volume.cpu(), phantom.cpu())
            nmse_v = nmse(res.volume.cpu(), phantom.cpu())
            row = {
                "variant": variant_name, "tv": tv_label,
                "time_s": t[0], "it/s": ips,
                "ssim": ssim_v, "nmse": nmse_v,
            }
            per_variant.append(row)
            rows.append(row)
            print(f"  TV={tv_label:<7}  {t[0]:7.2f}s   {ips:6.1f} it/s   "
                  f"SSIM={ssim_v:.4f}  NMSE={nmse_v:.3e}")
        # speedup line
        pt, fu = per_variant[0], per_variant[1]
        speedup_t   = pt["time_s"] / fu["time_s"]
        ms_per_iter = (pt["time_s"] - fu["time_s"]) / args.ad_iter * 1e3
        print(f"  → fused speedup: {speedup_t:.2f}×   "
              f"saving {ms_per_iter:.3f} ms/iter   "
              f"ΔSSIM = {fu['ssim'] - pt['ssim']:+.4f}")

    # summary table
    print("\nSummary (paired per variant):")
    print(f"  {'variant':<22} {'TV impl':<8} {'time_s':>8} {'it/s':>8} "
          f"{'SSIM':>8} {'NMSE':>10}")
    for r in rows:
        print(f"  {r['variant']:<22} {r['tv']:<8} {r['time_s']:>8.2f} "
              f"{r['it/s']:>8.1f} {r['ssim']:>8.4f} {r['nmse']:>10.3e}")

    print("\nPer-variant fused/PyTorch ratio:")
    print(f"  {'variant':<22} {'PyTorch':>10} {'Fused':>10} "
          f"{'speedup':>10} {'Δms/iter':>10} {'ΔSSIM':>9}")
    for variant_name, _ in VARIANTS:
        pt = next(r for r in rows if r["variant"] == variant_name and r["tv"] == "pytorch")
        fu = next(r for r in rows if r["variant"] == variant_name and r["tv"] == "fused")
        speedup     = pt["time_s"] / fu["time_s"]
        ms_per_iter = (pt["time_s"] - fu["time_s"]) / args.ad_iter * 1e3
        d_ssim      = fu["ssim"] - pt["ssim"]
        print(f"  {variant_name:<22} {pt['time_s']:>9.2f}s {fu['time_s']:>9.2f}s "
              f"{speedup:>9.2f}x {ms_per_iter:>9.3f}  {d_ssim:>+9.4f}")


if __name__ == "__main__":
    main()
