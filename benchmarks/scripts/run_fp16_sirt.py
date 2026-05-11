"""FP16 vs FP32 microbench — 1000-iter SIRT on the 200³ phantom.

Mirrors the inner loop of ``ft.run_sirt`` (which is fp32-only by design) but
parameterizes the dtype so we can compare. Routes FP/BP through
``ParallelProjector`` exactly as the production ``run_sirt`` does on CUDA.

SIRT has no optimizer state, so the fp16 underflow that kills Adam-fp16 here
isn't a concern — this is exactly the kind of inference loop the fp16 path is
designed for.
"""

from __future__ import annotations

import time

import torch

import fasttomo as ft

from benchmarks.scripts.utils.datasets import load_phantom_tilt_series
from benchmarks.scripts.utils.metrics import ssim, nmse


def sirt_run(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    *,
    dtype: torch.dtype,
    n_iter: int = 1000,
    omega: float = 1.0,
    positivity: bool = False,
    mask: bool = False,
) -> tuple[torch.Tensor, list[float], float]:
    """SIRT in the requested dtype.

    Returns (volume_fp32, losses, elapsed_s). Volume is upcast to fp32 on
    return so SSIM/NMSE comparisons are dtype-neutral.
    """
    device = sinogram.device
    n_angles, Ny, N_det = sinogram.shape
    Nx, Nz = N_det, N_det

    sino = sinogram.to(dtype)
    proj = ft.ParallelProjector((Ny, Nz, Nx), angles_deg, device=device, N_det=N_det)
    _fp = lambda v: proj.forward(v)
    _bp = lambda s: proj.back(s)

    volume = torch.zeros(Ny, Nz, Nx, dtype=dtype, device=device)

    # Normalisers (computed once)
    ones_sino = torch.ones_like(sino)
    C_vol = _bp(ones_sino)
    C_vol = torch.where(C_vol > 0, 1.0 / C_vol, torch.zeros_like(C_vol))

    ones_vol = torch.ones_like(volume)
    R_sino = _fp(ones_vol)
    R_sino = torch.where(R_sino > 0, 1.0 / R_sino, torch.zeros_like(R_sino))

    _mask = None
    if mask:
        _mask = ft.cylindrical_mask(Nz, Nx, device=device).to(dtype).unsqueeze(0)

    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        raw_res = sino - _fp(volume)
        losses.append(float(raw_res.float().pow(2).mean().item()))
        residual = raw_res * R_sino
        volume = volume + omega * C_vol * _bp(residual)
        if positivity:
            volume.clamp_(min=0.0)
        if _mask is not None:
            volume = volume * _mask
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    return volume.float(), losses, elapsed


def main():
    device = torch.device("cuda")
    print("Loading 200³ phantom + 29-angle tilt series ...")
    phantom, sino, angles = load_phantom_tilt_series(device=device)
    print(f"  phantom {tuple(phantom.shape)}, sino {tuple(sino.shape)}, "
          f"angles [{angles.min().item():+.1f}°, {angles.max().item():+.1f}°]")
    print()

    n_iter = 1000
    print(f"Running {n_iter}-iter SIRT (omega=1.0, positivity=True, mask=True) "
          f"through ParallelProjector cache ...")
    print()

    rows = []
    for dtype, label in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        rec, losses, elapsed = sirt_run(
            sino, angles, dtype=dtype, n_iter=n_iter,
            positivity=True, mask=True,
        )
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        s = ssim(rec, phantom)
        n = nmse(rec, phantom)
        rows.append((label, elapsed, peak, losses[-1], s, n))

    print(f"{'dtype':>6} | {'time (s)':>9} | {'ms/iter':>8} | {'peak (GB)':>9} | "
          f"{'final residual':>15} | {'SSIM':>7} | {'NMSE':>7}")
    print("-" * 90)
    for label, elapsed, peak, fl, s, n in rows:
        print(f"{label:>6} | {elapsed:9.3f} | {elapsed / n_iter * 1000:8.3f} | "
              f"{peak:9.3f} | {fl:15.4e} | {s:7.4f} | {n:7.4f}")
    print()
    t_fp32, t_fp16 = rows[0][1], rows[1][1]
    p_fp32, p_fp16 = rows[0][2], rows[1][2]
    print(f"speedup (fp16 vs fp32)   : {t_fp32 / t_fp16:.2f}×")
    print(f"memory ratio (fp16/fp32) : {p_fp16 / p_fp32:.2f}×")
    print(f"SSIM Δ  (fp16 − fp32)    : {rows[1][4] - rows[0][4]:+.4f}")
    print(f"NMSE Δ  (fp16 − fp32)    : {rows[1][5] - rows[0][5]:+.4f}")


if __name__ == "__main__":
    main()
