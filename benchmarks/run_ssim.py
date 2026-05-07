"""Stage 2 SSIM comparison: FBP across FastTomo, ASTRA, and TorchRadon.

All three backends reconstruct the same sinogram (the 1°-tilt-axis-error tilt
series). SSIM/NMSE are computed against the ground-truth phantom. Backends
that fail to import are skipped.

Usage:
    conda run -n FastTomo-bench python -m benchmarks.run_ssim
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch

import fasttomo as ft
from benchmarks.datasets import load_phantom_tilt_series
from benchmarks.metrics import nmse, ssim
from benchmarks.timing import cuda_timer


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        print(f"  [skipped: {name} unavailable: {type(e).__name__}: {e}]")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    phantom, sino, ang_deg = load_phantom_tilt_series(device=device)
    n_angles, Ny, N_det = sino.shape
    print(f"sinogram {(n_angles, Ny, N_det)}, phantom {tuple(phantom.shape)}")

    ang_rad = (ang_deg.cpu().numpy() * np.pi / 180.0).astype(np.float32)
    sino_np = sino.cpu().numpy()

    rows: list[dict] = []

    # --- FastTomo run_fbp ---
    print("\n[FastTomo] run_fbp")
    with cuda_timer(device) as t:
        rec_ft = ft.run_fbp(sino, ang_deg, positivity=True)
    rows.append({
        "backend": "fasttomo",
        "time_s": t[0],
        "ssim": ssim(rec_ft, phantom),
        "nmse": nmse(rec_ft, phantom),
        "min": float(rec_ft.min()),
        "max": float(rec_ft.max()),
    })
    print(f"  done in {t[0]:.3f}s, recon range [{rec_ft.min():.3f}, {rec_ft.max():.3f}]")

    # --- ASTRA FBP_CUDA per slice ---
    print("\n[ASTRA] FBP_CUDA per Y-slice")
    if _try_import("astra") is not None:
        from benchmarks.astra_wrapper import AstraParallelBeam
        astra_w = AstraParallelBeam(N_det, ang_rad)
        # warm up to avoid context-init time skewing the measurement
        _ = astra_w.fbp(sino_np[:, :1, :], positivity=True)
        with cuda_timer(device) as t:
            rec_a_np = astra_w.fbp(sino_np, positivity=True)
        rec_a = torch.from_numpy(rec_a_np)
        rows.append({
            "backend": "astra",
            "time_s": t[0],
            "ssim": ssim(rec_a, phantom),
            "nmse": nmse(rec_a, phantom),
            "min": float(rec_a.min()),
            "max": float(rec_a.max()),
        })
        print(f"  done in {t[0]:.3f}s, recon range [{rec_a.min():.3f}, {rec_a.max():.3f}]")

    # --- TorchRadon FBP ---
    print("\n[TorchRadon] filter_sinogram + backprojection")
    if _try_import("torch_radon") is not None:
        from benchmarks.torch_radon_wrapper import TorchRadonParallelBeam
        tr_w = TorchRadonParallelBeam(N_det, ang_rad)
        _ = tr_w.fbp(sino[:, :1, :], positivity=True)  # warm-up
        with cuda_timer(device) as t:
            rec_tr = tr_w.fbp(sino, positivity=True)
        rows.append({
            "backend": "torch_radon",
            "time_s": t[0],
            "ssim": ssim(rec_tr, phantom),
            "nmse": nmse(rec_tr, phantom),
            "min": float(rec_tr.min()),
            "max": float(rec_tr.max()),
        })
        print(f"  done in {t[0]:.3f}s, recon range [{rec_tr.min():.3f}, {rec_tr.max():.3f}]")

    # --- table ---
    print("\nResults:")
    print(f"  {'backend':<14} {'time_s':>9} {'ssim':>8} {'nmse':>10} {'min':>8} {'max':>8}")
    for r in rows:
        print(
            f"  {r['backend']:<14} {r['time_s']:>9.3f} {r['ssim']:>8.4f} "
            f"{r['nmse']:>10.4e} {r['min']:>8.3f} {r['max']:>8.3f}"
        )


if __name__ == "__main__":
    main()
