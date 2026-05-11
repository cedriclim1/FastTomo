"""FP16 vs FP32 microbench — 1000-iter Adam reconstruction on the 200³ phantom.

Mirrors the loop in ``ad_reconstruct.ad_reconstruct_cached`` but parameterizes
the dtype so we can compare fp32 vs fp16 in the same Adam + cosine-LR + TV
recipe. Routes FP/BP through ``ParallelProjector`` so the half-precision path
exercises the cached fp16 texture.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

import fasttomo as ft

from benchmarks.scripts.utils.ad_reconstruct import _tv_charbonnier
from benchmarks.scripts.utils.datasets import load_phantom_tilt_series
from benchmarks.scripts.utils.metrics import ssim, nmse


def ad_run(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    *,
    mode: str,                      # 'fp32' | 'fp16-pure' | 'fp16-mixed'
    n_iter: int = 1000,
    lr: float = 0.05,
    tv_weight: float = 1.0,
) -> tuple[torch.Tensor, list[float], float]:
    """Adam + cosine-LR + TV reconstruction.

    mode='fp32'         — baseline.
    mode='fp16-pure'    — vol, sino, Adam state all fp16. Diverges (NaN) on
                          this recipe; included for honest comparison.
    mode='fp16-mixed'   — fp32 master volume + Adam state; cast to fp16 only
                          for the projector forward (autograd downcasts the
                          gradient back to fp32 through the cast). Mirrors
                          how TorchRadon recommends using half precision in
                          training loops. Fp16 sinogram, fp32 loss.
    """
    device = sinogram.device
    n_angles, Ny, N_det = sinogram.shape
    Nx, Nz = N_det, N_det

    if mode == "fp32":
        vol_dtype = torch.float32
        sino_dtype = torch.float32
        cast_for_fp = False
    elif mode == "fp16-pure":
        vol_dtype = torch.float16
        sino_dtype = torch.float16
        cast_for_fp = False
    elif mode == "fp16-mixed":
        vol_dtype = torch.float32     # master weights stay fp32
        sino_dtype = torch.float16
        cast_for_fp = True            # cast vol to fp16 before proj.forward
    else:
        raise ValueError(f"unknown mode: {mode}")

    sino = sinogram.to(sino_dtype)
    proj = ft.ParallelProjector((Ny, Nz, Nx), angles_deg, device=device, N_det=N_det)

    vol = torch.zeros(Ny, Nz, Nx, device=device, dtype=vol_dtype, requires_grad=True)
    opt = torch.optim.Adam([vol], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter)

    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        opt.zero_grad()
        # fp16-mixed: cast vol to fp16 inside the autograd graph; backward
        # downcasts the gradient on the cast and Adam updates fp32 master.
        vol_proj = vol.half() if cast_for_fp else vol
        pred = proj.forward(vol_proj)
        if cast_for_fp:
            # MSE in fp32 keeps loss precision; gradient flows fp32 → fp16
            # through pred.float() → vol_proj → vol.
            loss_data = F.mse_loss(pred.float(), sino.float())
        else:
            loss_data = F.mse_loss(pred, sino)
        loss = loss_data + tv_weight * _tv_charbonnier(vol)
        losses.append(float(loss.item()))
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            vol.clamp_(min=0)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    return vol.detach().float(), losses, elapsed


def main():
    device = torch.device("cuda")
    print("Loading 200³ phantom + 29-angle tilt series ...")
    phantom, sino, angles = load_phantom_tilt_series(device=device)
    print(f"  phantom {tuple(phantom.shape)}, sino {tuple(sino.shape)}, "
          f"angles [{angles.min().item():+.1f}°, {angles.max().item():+.1f}°]")
    print()

    n_iter = 1000
    print(f"Running {n_iter}-iter Adam + cosine-LR + TV (lr=0.05, tv=1.0) "
          f"through ParallelProjector cache ...")
    print()

    rows = []
    for mode in ("fp32", "fp16-pure", "fp16-mixed"):
        rec, losses, elapsed = ad_run(sino, angles, mode=mode, n_iter=n_iter)
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        s = ssim(rec, phantom)
        n = nmse(rec, phantom)
        rows.append((mode, elapsed, peak, losses[-1], s, n))

    print(f"{'mode':>11} | {'time (s)':>9} | {'ms/iter':>8} | {'peak (GB)':>9} | "
          f"{'final loss':>10} | {'SSIM':>7} | {'NMSE':>7}")
    print("-" * 85)
    for mode, elapsed, peak, fl, s, n in rows:
        loss_str = f"{fl:10.3e}" if not (fl != fl) else "       nan"
        ssim_str = f"{s:7.4f}"   if not (s != s)  else "    nan"
        nmse_str = f"{n:7.4f}"   if not (n != n)  else "    nan"
        print(f"{mode:>11} | {elapsed:9.3f} | {elapsed / n_iter * 1000:8.3f} | "
              f"{peak:9.3f} | {loss_str} | {ssim_str} | {nmse_str}")
    print()
    t_fp32   = rows[0][1]
    t_mixed  = rows[2][1]
    p_fp32   = rows[0][2]
    p_mixed  = rows[2][2]
    print(f"speedup (fp16-mixed vs fp32) : {t_fp32 / t_mixed:.2f}×")
    print(f"memory ratio (fp16-mixed / fp32): {p_mixed / p_fp32:.2f}×")
    print(f"SSIM Δ  (fp16-mixed − fp32)  : {rows[2][4] - rows[0][4]:+.4f}")
    print(f"NMSE Δ  (fp16-mixed − fp32)  : {rows[2][5] - rows[0][5]:+.4f}")


if __name__ == "__main__":
    main()
