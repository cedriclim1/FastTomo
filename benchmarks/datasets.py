"""Dataset loaders for benchmarks.

The phantom + tilt-series + tilt-angles triple lives outside the FastTomo repo
(in /home/cedlim/quantem/quantem-tutorials/data/). The tilt series has a 1°
tilt-axis error baked in, which is the headline thing pose optimization should
recover.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

DEFAULT_DATA_DIR = Path("/home/cedlim/quantem/quantem-tutorials/data")


def load_phantom_tilt_series(
    device: torch.device | str = "cuda",
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the (phantom, sinogram, angles_deg) triple as float32 tensors.

    Applies the per-projection ``np.rot90`` alignment from
    ``Notebooks/opt_test.ipynb`` so the tilt axis lines up with FastTomo's Y.

    Returns
    -------
    phantom : [200, 200, 200] float32
    sinogram : [29, 200, 200] float32 — [n_angles, Ny, N_det]
    angles_deg : [29] float32 — −70°…70° step 5°
    """
    data_dir = Path(data_dir)
    phantom_np = np.load(data_dir / "phantom.npy").astype(np.float32)
    ts_np = np.load(data_dir / "tilt_series_1_deg_tilt_axis.npy").astype(np.float32)
    ang_np = np.load(data_dir / "tilt_angles_1_deg_tilt_axis.npy").astype(np.float32)

    # Reorient phantom from quantem axes to FastTomo's [Ny, Nz, Nx].
    phantom_np = phantom_np.transpose(2, 0, 1)
    ts_np = np.stack([np.rot90(img) for img in ts_np])

    phantom = torch.from_numpy(phantom_np).to(device)
    sinogram = torch.from_numpy(ts_np).to(device)
    angles = torch.from_numpy(ang_np).to(device)
    return phantom, sinogram, angles
