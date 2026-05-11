"""Stage 2 FBP speed comparison: FastTomo (functional + cached), ASTRA, TorchRadon.

All four backends reconstruct the same sinogram (the 1°-tilt-axis-error tilt
series). SSIM/NMSE are computed against the ground-truth phantom; mean wall-clock
per call is measured via ``benchmark_function`` (3 warmup + 30 samples) so that
sub-ms-scale differences between cached and uncached FBP are not lost in
single-call jitter. Backends that fail to import are skipped.

Usage:
    conda run -n FastTomo-bench python -m benchmarks.run_ssim
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch

import fasttomo as ft
from benchmarks.scripts.utils.datasets import load_phantom_tilt_series
from benchmarks.scripts.utils.metrics import nmse, ssim
from benchmarks.scripts.utils.timing import benchmark_function


_SAMPLES = 30
_WARMUP = 3


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        print(f"  [skipped: {name} unavailable: {type(e).__name__}: {e}]")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=_SAMPLES,
                        help="timed reps per backend")
    parser.add_argument("--warmup", type=int, default=_WARMUP,
                        help="warmup reps per backend (dropped from timing)")
    args = parser.parse_args()

    device = torch.device(args.device)
    phantom, sino, ang_deg = load_phantom_tilt_series(device=device)
    n_angles, Ny, N_det = sino.shape
    print(f"sinogram {(n_angles, Ny, N_det)}, phantom {tuple(phantom.shape)}")
    print(f"timing  : warmup={args.warmup}, samples={args.samples}\n")

    ang_rad = (ang_deg.cpu().numpy() * np.pi / 180.0).astype(np.float32)
    sino_np = sino.cpu().numpy()

    rows: list[dict] = []

    # --- FastTomo run_fbp (functional path) ---
    print("[FastTomo] run_fbp")
    fn_ft = lambda x: ft.run_fbp(x, ang_deg, positivity=True)
    t_ft = benchmark_function(fn_ft, sino, samples=args.samples,
                              warmup=args.warmup, sync=True)
    rec_ft = fn_ft(sino)
    rows.append({
        "backend": "fasttomo",
        "time_s": t_ft,
        "ssim": ssim(rec_ft, phantom),
        "nmse": nmse(rec_ft, phantom),
        "min": float(rec_ft.min()),
        "max": float(rec_ft.max()),
    })
    print(f"  mean {t_ft*1000:.3f} ms/call, recon range "
          f"[{rec_ft.min():.3f}, {rec_ft.max():.3f}]")

    # --- FastTomo run_fbp (ParallelProjector cached) ---
    print("\n[FastTomo] run_fbp + ParallelProjector cache")
    proj = ft.ParallelProjector((Ny, N_det, N_det), ang_deg,
                                device=device, N_det=N_det)
    fn_ft_c = lambda x: ft.run_fbp(x, ang_deg, positivity=True, projector=proj)
    t_ft_c = benchmark_function(fn_ft_c, sino, samples=args.samples,
                                warmup=args.warmup, sync=True)
    rec_ft_c = fn_ft_c(sino)
    rows.append({
        "backend": "fasttomo-cached",
        "time_s": t_ft_c,
        "ssim": ssim(rec_ft_c, phantom),
        "nmse": nmse(rec_ft_c, phantom),
        "min": float(rec_ft_c.min()),
        "max": float(rec_ft_c.max()),
    })
    print(f"  mean {t_ft_c*1000:.3f} ms/call, recon range "
          f"[{rec_ft_c.min():.3f}, {rec_ft_c.max():.3f}]")
    # Verify the cached path is numerically equivalent (bit-identical expected).
    diff = (rec_ft_c - rec_ft).abs().max().item()
    print(f"  max |cached − functional| = {diff:.2e}")

    # --- ASTRA FBP_CUDA per slice ---
    print("\n[ASTRA] FBP_CUDA per Y-slice")
    if _try_import("astra") is not None:
        from benchmarks.scripts.utils.astra_wrapper import AstraParallelBeam
        astra_w = AstraParallelBeam(N_det, ang_rad)
        fn_a = lambda x: astra_w.fbp(x, positivity=True)
        t_a = benchmark_function(fn_a, sino_np, samples=args.samples,
                                 warmup=args.warmup, sync=True)
        rec_a_np = fn_a(sino_np)
        rec_a = torch.from_numpy(rec_a_np)
        rows.append({
            "backend": "astra",
            "time_s": t_a,
            "ssim": ssim(rec_a, phantom),
            "nmse": nmse(rec_a, phantom),
            "min": float(rec_a.min()),
            "max": float(rec_a.max()),
        })
        print(f"  mean {t_a*1000:.3f} ms/call, recon range "
              f"[{rec_a.min():.3f}, {rec_a.max():.3f}]")

    # --- TorchRadon FBP ---
    print("\n[TorchRadon] filter_sinogram + backprojection")
    if _try_import("torch_radon") is not None:
        from benchmarks.scripts.utils.torch_radon_wrapper import TorchRadonParallelBeam
        tr_w = TorchRadonParallelBeam(N_det, ang_rad)
        fn_tr = lambda x: tr_w.fbp(x, positivity=True)
        t_tr = benchmark_function(fn_tr, sino, samples=args.samples,
                                  warmup=args.warmup, sync=True)
        rec_tr = fn_tr(sino)
        rows.append({
            "backend": "torch_radon",
            "time_s": t_tr,
            "ssim": ssim(rec_tr, phantom),
            "nmse": nmse(rec_tr, phantom),
            "min": float(rec_tr.min()),
            "max": float(rec_tr.max()),
        })
        print(f"  mean {t_tr*1000:.3f} ms/call, recon range "
              f"[{rec_tr.min():.3f}, {rec_tr.max():.3f}]")

    # --- table ---
    print("\nResults:")
    print(f"  {'backend':<18} {'ms/call':>9} {'ssim':>8} {'nmse':>10} "
          f"{'min':>8} {'max':>8}")
    for r in rows:
        print(
            f"  {r['backend']:<18} {r['time_s']*1000:>9.3f} {r['ssim']:>8.4f} "
            f"{r['nmse']:>10.4e} {r['min']:>8.3f} {r['max']:>8.3f}"
        )


if __name__ == "__main__":
    main()
