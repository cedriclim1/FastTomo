"""FP16 FastTomo SIRT vs TorchRadon iterative-FBP (fp16 attempt + fp32 baseline).

FastTomo SIRT runs natively in fp16 via the packed-Y texture path.
TorchRadon is probed for fp16 support; if its CUDA kernels reject half-precision
tensors the wrapper falls back to fp32 and annotates the table accordingly.

Both compared at 1000 iterations, positivity enforced throughout.

Usage:
    conda run -n FastTomo-bench python -m benchmarks.run_fp16_comparison
"""

from __future__ import annotations

import time

import numpy as np
import torch

import fasttomo as ft
from benchmarks.scripts.utils.datasets import load_phantom_tilt_series
from benchmarks.scripts.utils.metrics import nmse, ssim
from benchmarks.scripts.utils.torch_radon_wrapper import TorchRadonParallelBeam


# ---------------------------------------------------------------------------
# FastTomo SIRT in a specified dtype
# ---------------------------------------------------------------------------

def _ft_sirt(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    *,
    dtype: torch.dtype,
    n_iter: int,
) -> tuple[torch.Tensor, float]:
    device = sinogram.device
    n_angles, Ny, N_det = sinogram.shape
    sino = sinogram.to(dtype)
    proj = ft.ParallelProjector((Ny, N_det, N_det), angles_deg, device=device, N_det=N_det)

    volume = torch.zeros(Ny, N_det, N_det, dtype=dtype, device=device)
    C_vol = proj.back(torch.ones_like(sino))
    C_vol = torch.where(C_vol > 0, 1.0 / C_vol, torch.zeros_like(C_vol))
    R_sino = proj.forward(torch.ones_like(volume))
    R_sino = torch.where(R_sino > 0, 1.0 / R_sino, torch.zeros_like(R_sino))
    mask = ft.cylindrical_mask(N_det, N_det, device=device).to(dtype).unsqueeze(0)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        residual = (sino - proj.forward(volume)) * R_sino
        volume = (volume + C_vol * proj.back(residual)).clamp_(min=0.0) * mask
    torch.cuda.synchronize(device)
    return volume.float(), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# TorchRadon iterative-FBP in a specified dtype
# ---------------------------------------------------------------------------

def _tr_sirt(
    sinogram: torch.Tensor,
    angles_rad: np.ndarray,
    *,
    dtype: torch.dtype,
    n_iter: int,
    step_size: float,
    smoothing_sigma: float | None,
) -> tuple[torch.Tensor, float, str]:
    """Returns (recon_fp32, elapsed_s, actual_dtype_label)."""
    tr = TorchRadonParallelBeam(sinogram.shape[2], angles_rad)
    sino = sinogram.to(dtype)

    # Probe: one forward pass to see if TorchRadon accepts this dtype.
    # TorchRadon fp16 requires batch (Y) to be a multiple of 4; probe with 4.
    actual_label = {torch.float16: "fp16", torch.float32: "fp32"}[dtype]
    try:
        n_angles, Ny, N_det = sino.shape
        _ = tr.radon.forward(torch.zeros(4, N_det, N_det, dtype=dtype, device=sino.device))
    except (RuntimeError, TypeError, AssertionError):
        # TorchRadon doesn't support this dtype — fall back to fp32.
        sino = sino.float()
        actual_label = "fp32 (fallback)"

    torch.cuda.synchronize(sinogram.device)
    t0 = time.perf_counter()
    rec = tr.sirt(
        sino,
        num_iter=n_iter,
        step_size=step_size,
        positivity=True,
        smoothing_sigma=smoothing_sigma,
    )
    torch.cuda.synchronize(sinogram.device)
    return rec.float(), time.perf_counter() - t0, actual_label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda")
    n_iter = 1000
    tr_step = 0.15
    tr_sigma = 1.0

    print("Loading 200³ phantom + 29-angle tilt series ...")
    phantom, sino, angles_deg = load_phantom_tilt_series(device=device)
    n_angles, Ny, N_det = sino.shape
    angles_rad = (angles_deg.cpu().numpy() * np.pi / 180.0).astype(np.float32)
    print(f"  phantom {tuple(phantom.shape)}, sino {(n_angles, Ny, N_det)}\n")

    rows: list[dict] = []

    # ---- FastTomo FP32 SIRT (reference) ------------------------------------
    print(f"[FastTomo] FP32 SIRT ({n_iter} iter, positivity, mask, cached) ...")
    _ = _ft_sirt(sino, angles_deg, dtype=torch.float32, n_iter=5)  # warm-up
    rec_ft32, t_ft32 = _ft_sirt(sino, angles_deg, dtype=torch.float32, n_iter=n_iter)
    rows.append(dict(method="ft-sirt-fp32", dtype="fp32", n_iter=n_iter,
                     time_s=t_ft32, ssim=ssim(rec_ft32, phantom), nmse=nmse(rec_ft32, phantom)))
    print(f"  {t_ft32:.3f}s  SSIM={rows[-1]['ssim']:.4f}")

    # ---- FastTomo FP16 SIRT ------------------------------------------------
    print(f"[FastTomo] FP16 SIRT ({n_iter} iter, positivity, mask, cached) ...")
    _ = _ft_sirt(sino, angles_deg, dtype=torch.float16, n_iter=5)  # warm-up
    rec_ft16, t_ft16 = _ft_sirt(sino, angles_deg, dtype=torch.float16, n_iter=n_iter)
    rows.append(dict(method="ft-sirt-fp16", dtype="fp16", n_iter=n_iter,
                     time_s=t_ft16, ssim=ssim(rec_ft16, phantom), nmse=nmse(rec_ft16, phantom)))
    print(f"  {t_ft16:.3f}s  SSIM={rows[-1]['ssim']:.4f}")

    # ---- TorchRadon FP16 (probe; falls back to fp32 if unsupported) --------
    print(f"[TorchRadon] FP16 iterative-FBP ({n_iter} iter, "
          f"step={tr_step}, sigma={tr_sigma}, positivity) ...")
    # TorchRadon fp16 requires Ny (batch) % 4 == 0 — warm-up with 4 slices.
    _tr_sirt(sino[:, :4, :], angles_rad, dtype=torch.float16, n_iter=2,
             step_size=tr_step, smoothing_sigma=tr_sigma)
    rec_tr16, t_tr16, tr16_label = _tr_sirt(
        sino, angles_rad, dtype=torch.float16, n_iter=n_iter,
        step_size=tr_step, smoothing_sigma=tr_sigma,
    )
    rows.append(dict(method=f"torch_radon ({tr16_label})", dtype=tr16_label, n_iter=n_iter,
                     time_s=t_tr16, ssim=ssim(rec_tr16, phantom), nmse=nmse(rec_tr16, phantom)))
    print(f"  {t_tr16:.3f}s  SSIM={rows[-1]['ssim']:.4f}  [actual dtype: {tr16_label}]")

    # ---- TorchRadon FP32 baseline ------------------------------------------
    print(f"[TorchRadon] FP32 iterative-FBP ({n_iter} iter, "
          f"step={tr_step}, sigma={tr_sigma}, positivity) ...")
    _tr_sirt(sino[:, :4, :], angles_rad, dtype=torch.float32, n_iter=2,
             step_size=tr_step, smoothing_sigma=tr_sigma)  # warm-up
    rec_tr32, t_tr32, _ = _tr_sirt(
        sino, angles_rad, dtype=torch.float32, n_iter=n_iter,
        step_size=tr_step, smoothing_sigma=tr_sigma,
    )
    rows.append(dict(method="torch_radon-fp32", dtype="fp32", n_iter=n_iter,
                     time_s=t_tr32, ssim=ssim(rec_tr32, phantom), nmse=nmse(rec_tr32, phantom)))
    print(f"  {t_tr32:.3f}s  SSIM={rows[-1]['ssim']:.4f}")

    # ---- Table -------------------------------------------------------------
    print()
    hdr = f"  {'method':<30} {'dtype':>18} {'n_iter':>7} {'time_s':>9} {'it/s':>8} {'SSIM':>8} {'NMSE':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        ips = r["n_iter"] / r["time_s"]
        print(
            f"  {r['method']:<30} {r['dtype']:>18} {r['n_iter']:>7} "
            f"{r['time_s']:>9.3f} {ips:>8.1f} {r['ssim']:>8.4f} {r['nmse']:>10.4e}"
        )
    print()

    ft32 = rows[0]
    ft16 = rows[1]
    print(f"FP16 vs FP32 FastTomo SIRT:")
    print(f"  speedup       : {ft32['time_s'] / ft16['time_s']:.2f}×")
    print(f"  SSIM Δ        : {ft16['ssim'] - ft32['ssim']:+.4f}")
    print(f"  NMSE Δ        : {ft16['nmse'] - ft32['nmse']:+.4f}")
    print()

    tr = rows[2]
    tr32 = rows[3]
    ft16_vs_tr16 = tr["time_s"] / ft16["time_s"]
    ft16_vs_tr32 = tr32["time_s"] / ft16["time_s"]
    print(f"FP16 FastTomo SIRT vs TorchRadon ({tr['dtype']}, step={tr_step}, sigma={tr_sigma}):")
    print(f"  ft16 speedup vs tr-fp16 : {ft16_vs_tr16:.2f}×  ({ft16['time_s']:.3f}s vs {tr['time_s']:.3f}s)")
    print(f"  ft16 speedup vs tr-fp32 : {ft16_vs_tr32:.2f}×  ({ft16['time_s']:.3f}s vs {tr32['time_s']:.3f}s)")
    print(f"  SSIM Δ (ft16 − tr-fp16) : {ft16['ssim'] - tr['ssim']:+.4f}")
    print(f"  SSIM Δ (ft16 − tr-fp32) : {ft16['ssim'] - tr32['ssim']:+.4f}")


if __name__ == "__main__":
    main()
