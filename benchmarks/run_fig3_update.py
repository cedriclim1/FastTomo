"""Run the two new AD conditions for fig3_ssim and augment presentation_results.json.

New entries:
  ft-ad-pose-notv    : AD + pose, tv_weight=0  (no TV, no SSIM)
  ft-ad-pose-tv-ssim : AD + pose + TV (tv_w=1.0) + SSIM (λ=100), n_iter=1000

The existing ft-ad-with-pose-cached entry (tv_weight=1.0, no SSIM) already represents
AD + Pose + TV — this script only adds the two missing conditions.

Usage:
    conda run -n FastTomo python -m benchmarks.run_fig3_update --device cuda:2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import fasttomo as ft
from fasttomo.autograd import _euler_zyz_to_rotmat
from benchmarks.datasets import load_phantom_tilt_series
from benchmarks.metrics import ssim, nmse


def run_ad_pose(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    *,
    n_iter: int = 1000,
    lr_vol: float = 0.05,
    lr_ang: float = 1.0,
    lr_shf: float = 0.1,
    tv_w: float = 1.0,
    ssim_w: float = 0.0,
    device: torch.device,
) -> torch.Tensor:
    n_angles, Ny, N_det = sinogram.shape
    vol_shape = (Ny, N_det, N_det)

    proj = ft.RotmatProjector(vol_shape, n_angles, device=device, N_det=N_det)

    vol = torch.zeros(vol_shape, device=device, requires_grad=True)
    euler = torch.zeros(n_angles, 3, device=device)
    euler[:, 1] = angles_deg
    euler.requires_grad_(True)
    shifts = torch.zeros(n_angles, 2, device=device, requires_grad=True)

    mask = ft.cylindrical_mask(N_det, N_det, device=device)

    opts = [
        torch.optim.Adam([vol],    lr=lr_vol),
        torch.optim.Adam([euler],  lr=lr_ang),
        torch.optim.Adam([shifts], lr=lr_shf),
    ]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=n_iter)
              for o in opts]

    for i in range(n_iter):
        for o in opts:
            o.zero_grad()
        mats = _euler_zyz_to_rotmat(euler)
        pred = proj.forward(vol, mats)
        pred = ft.apply_shifts_2d(pred, shifts)
        loss = F.mse_loss(pred, sinogram)
        if tv_w:    loss = loss + tv_w   * ft.tv_loss_3d(vol)
        if ssim_w:  loss = loss + ssim_w * ft.tilt_ssim_loss(pred, sinogram)
        loss.backward()
        for o in opts:  o.step()
        for s in scheds: s.step()
        with torch.no_grad():
            vol.clamp_(min=0)
            vol.mul_(mask)
        if (i + 1) % 100 == 0:
            print(f"   iter {i+1}/{n_iter}  loss={loss.item():.4e}")

    return vol.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--ssim-weight", type=float, default=100.0)
    parser.add_argument("--out", default=str(ROOT / "presentation_results.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")

    results_path = Path(args.out)
    data = json.loads(results_path.read_text())
    existing_methods = {r["method"] for r in data["rows"]}

    phantom, sinogram, angles_deg = load_phantom_tilt_series(device=device)
    phantom_np = phantom.cpu().numpy()
    print(f"phantom {tuple(phantom.shape)}, sinogram {tuple(sinogram.shape)}")

    new_rows: list[dict] = []

    conditions = [
        ("ft-ad-pose-notv",    "AD + Pose (no TV)",       0.0,           0.0),
        ("ft-ad-pose-tv-ssim", "AD + Pose + TV + SSIM",   1.0,           args.ssim_weight),
    ]

    for method, label, tv_w, ssim_w in conditions:
        if method in existing_methods:
            print(f"[skip] {method} already in JSON — delete entry to re-run")
            continue
        print(f"\n[run] {label}  tv_w={tv_w}  ssim_w={ssim_w}  n_iter={args.n_iter}")
        t0 = time.perf_counter()
        vol = run_ad_pose(
            sinogram, angles_deg,
            n_iter=args.n_iter,
            tv_w=tv_w,
            ssim_w=ssim_w,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        vol_np = vol.cpu().numpy()
        s = ssim(vol_np, phantom_np)
        n = nmse(vol_np, phantom_np)
        print(f"   {elapsed:.1f}s  SSIM={s:.4f}  NMSE={n:.3e}")
        new_rows.append({
            "method":   method,
            "kind":     "ft",
            "n_iter":   args.n_iter,
            "time_s":   elapsed,
            "it_per_s": args.n_iter / elapsed,
            "ssim":     s,
            "nmse":     n,
        })

    if new_rows:
        data["rows"].extend(new_rows)
        results_path.write_text(json.dumps(data, indent=2))
        print(f"\n[saved] {results_path}  (+{len(new_rows)} rows)")
    else:
        print("\n[no new rows to write]")


if __name__ == "__main__":
    main()
