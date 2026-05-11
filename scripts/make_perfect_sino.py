"""Generate a perfect (zero-error) tilt series from the quantem benchmark phantom.

Loads phantom.npy from the quantem-tutorials data directory, applies the same
axis reorientation used in load_phantom_tilt_series(), forward-projects with 29
angles from −70° to +70° (5° step), and writes
data/quantem_phantom_perfect_sino.npy.

Run once:
    conda run -n FastTomo python scripts/make_perfect_sino.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import fasttomo as ft

REPO_ROOT = Path(__file__).resolve().parent.parent
QUANTEM_DATA_DIR = Path("/home/cedlim/quantem/quantem-tutorials/data")
OUT_PATH = REPO_ROOT / "data" / "quantem_phantom_perfect_sino.npy"


def main() -> None:
    phantom_path = QUANTEM_DATA_DIR / "phantom.npy"
    print(f"Loading phantom: {phantom_path}")
    phantom_np = np.load(phantom_path).astype(np.float32)
    # Same reorientation as load_phantom_tilt_series()
    phantom_np = phantom_np.transpose(2, 0, 1)[::-1].copy()
    print(f"  shape after reorient: {phantom_np.shape}")

    phantom = torch.from_numpy(phantom_np).cuda()
    angles_deg = torch.linspace(-70.0, 70.0, 29, device=phantom.device)
    print(f"  angles: {angles_deg[0].item():.1f}° … {angles_deg[-1].item():.1f}°  "
          f"({len(angles_deg)} projections)")

    print("Forward projecting…")
    with torch.no_grad():
        sino = ft.forward_project(phantom, angles_deg)
    print(f"  sinogram shape: {tuple(sino.shape)}")
    print(f"  value range: [{sino.min().item():.4f}, {sino.max().item():.4f}]")

    sino_np = sino.cpu().numpy()
    np.save(OUT_PATH, sino_np)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
