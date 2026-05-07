"""TorchRadon parallel-beam wrapper.

TorchRadon is 2D-batched: it takes ``[batch, H, W]`` volumes and returns
``[batch, n_angles, N_det]`` sinograms. Mapping to FastTomo: batch = Y, so
we permute ``[n_angles, Ny, N_det]`` ↔ ``[Ny, n_angles, N_det]``.

TorchRadon 1.0.0's built-in ``filter_sinogram`` calls the removed
``torch.rfft`` API. Rather than reimplementing the filter, we reuse FastTomo's
``ramp_filter`` and ``π / n_angles`` scaling — so the comparison reduces to
TorchRadon's CUDA back-projector vs FastTomo's, with everything else held
equal.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch_radon import Radon

from fasttomo import ramp_filter


class TorchRadonParallelBeam:
    def __init__(self, n_det: int, angles_rad: np.ndarray):
        self.n_det = int(n_det)
        self.angles_rad = np.asarray(angles_rad, dtype=np.float32)
        # clip_to_circle=False keeps the full square reconstruction window,
        # matching ASTRA's and FastTomo's default behaviour.
        self.radon = Radon(n_det, self.angles_rad, clip_to_circle=False)

    def fbp(self, sinogram: torch.Tensor, positivity: bool = False) -> torch.Tensor:
        """sinogram: [n_angles, Ny, N_det] CUDA → recon [Ny, Nz, Nx] CUDA."""
        n_angles = sinogram.shape[0]
        filtered = ramp_filter(sinogram)                       # [n_angles, Ny, N_det]
        sino_b = filtered.permute(1, 0, 2).contiguous()        # [Ny, n_angles, N_det]
        rec = self.radon.backprojection(sino_b)                # [Ny, N_det, N_det]
        rec = rec * (math.pi / n_angles)
        if positivity:
            rec = rec.clamp_(min=0.0)
        return rec

    def sirt(
        self,
        sinogram: torch.Tensor,
        num_iter: int = 100,
        step_size: float = 0.15,
        positivity: bool = True,
    ) -> torch.Tensor:
        """Iterative-FBP-with-positivity, ported from sirt_recon_ex.py.

        Each iteration: filter the projection residual (with FastTomo's ramp
        filter), back-project, accumulate scaled by ``step_size``. ``positivity``
        clamps after every outer iteration. All smoothing is removed.

        ``step_size`` is reduced to 0.15 (the original example used 0.95 with
        Gaussian smoothing as a stabiliser; without smoothing, 0.95 diverges
        on limited-angle sinograms where the per-iter FBP-of-residual amplifies
        missing-wedge artefacts).

        sinogram: [n_angles, Ny, N_det] CUDA → recon [Ny, Nz=N_det, Nx=N_det] CUDA.
        Slices are batched over Y rather than looped (the original code's per-slice
        loop is a Python-overhead artifact, not algorithmic).
        """
        n_angles, Ny, N_det = sinogram.shape
        sino_b = sinogram.permute(1, 0, 2).contiguous()        # [Ny, n_angles, N_det]
        recon = torch.zeros(Ny, N_det, N_det, device=sinogram.device, dtype=sinogram.dtype)
        scale = math.pi / n_angles
        for _ in range(num_iter):
            pred = self.radon.forward(recon)                   # [Ny, n_angles, N_det]
            diff = sino_b - pred
            update = self.radon.backprojection(ramp_filter(diff)) * scale
            recon += step_size * update
            if positivity:
                recon.clamp_(min=0.0)
        return recon
