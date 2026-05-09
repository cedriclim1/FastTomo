#!/usr/bin/env python3
"""Simulate a single-axis tilt series from the FePt phantom.

Angles: -70 to +70 deg, 2 deg steps (71 tilts), no rotational error.
Rotation is about the Y-axis (tilt axis = Ny) using ZYZ Euler convention
with alpha=gamma=0 and beta=tilt_angle — identical to how eval_holdout and
run_combo_sweep project in the AD loop.

Outputs (in data/):
  fept_tilt_series_70deg_2step.npy   float32 [71, 200, 200]
  fept_tilt_angles_70deg_2step.npy   float32 [71]  (degrees)

Run from the repo root:
  conda run -n FastTomo python scripts/make_fept_tilt_series.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import fasttomo as ft
from fasttomo.autograd import _euler_zyz_to_rotmat

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    vol_np = np.load(DATA_DIR / "fept_phantom_200.npy")          # [Ny, Nz, Nx]
    Ny, Nz, Nx = vol_np.shape
    N_det = Nx

    device = torch.device(DEVICE)
    vol = torch.from_numpy(vol_np).to(device)

    angles_deg = np.arange(-70, 71, 2, dtype=np.float32)         # -70,-68,...,70
    n_angles   = len(angles_deg)
    print(f"Phantom : {vol_np.shape}  device={device}")
    print(f"Angles  : {n_angles} tilts  [{angles_deg[0]:.0f}° … {angles_deg[-1]:.0f}°]")

    proj = ft.RotmatProjector(
        vol_shape=(Ny, Nz, Nx),
        n_angles=n_angles,
        device=device,
        N_det=N_det,
    )

    angles_t = torch.from_numpy(angles_deg).to(device)
    eul = torch.zeros(n_angles, 3, device=device)
    eul[:, 1] = angles_t                                          # beta = tilt angle

    with torch.no_grad():
        mats     = _euler_zyz_to_rotmat(eul)
        tilt_series = proj.forward(vol, mats)                     # [n_angles, Ny, N_det]

    ts_np = tilt_series.cpu().numpy()
    print(f"Tilt series: {ts_np.shape}  min={ts_np.min():.4f}  max={ts_np.max():.4f}")

    ts_path  = DATA_DIR / "fept_tilt_series_70deg_2step.npy"
    ang_path = DATA_DIR / "fept_tilt_angles_70deg_2step.npy"
    np.save(ts_path,  ts_np)
    np.save(ang_path, angles_deg)
    print(f"\nSaved tilt series → {ts_path}")
    print(f"Saved angles      → {ang_path}")


if __name__ == "__main__":
    main()
