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
import torch.nn.functional as F
from torch_radon import Radon

from fasttomo import ramp_filter


def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.0) -> torch.Tensor:
    """Normalised 1D Gaussian, ported from sirt_recon_ex.py."""
    radius = int(math.ceil(num_sigmas * sigma))
    support = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.distributions.Normal(loc=0.0, scale=float(sigma)).log_prob(support).exp_()
    return kernel.mul_(1.0 / kernel.sum())


def gaussian_filter_2d_batched(vol: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    """Separable 2D Gaussian over the last two dims of a [B, H, W] tensor.

    The original example loops ``conv2d`` per slice; we keep the kernel separable
    but batch the B axis through the channel dim of ``conv2d`` to avoid the Python
    loop while preserving identical numerics.
    """
    pad = kernel_1d.numel() // 2
    img = vol.unsqueeze(1)  # [B, 1, H, W]
    k = kernel_1d.to(dtype=vol.dtype)
    w_v = k.view(1, 1, -1, 1)
    w_h = k.view(1, 1, 1, -1)
    img = F.conv2d(img, weight=w_v, padding=(pad, 0))
    img = F.conv2d(img, weight=w_h, padding=(0, pad))
    return img.squeeze(1)


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
        step_size: float | None = None,
        positivity: bool = True,
        smoothing_sigma: float | None = 1.0,
    ) -> torch.Tensor:
        """Iterative-FBP-with-positivity, ported from sirt_recon_ex.py.

        Each iteration: filter the projection residual (with FastTomo's ramp
        filter), back-project, accumulate scaled by ``step_size``. ``positivity``
        clamps after every outer iteration.

        ``smoothing_sigma`` enables the Gaussian regulariser from the original
        example (separable conv2d after every inner update). It stabilises the
        per-iter FBP-of-residual against missing-wedge amplification. The
        original ``sirt_recon_ex.py`` paired ``sigma=1.0`` with ``step=0.95``,
        but that combination only converges for the handful of iterations
        used in the example — by ~iter 160 on limited-angle data it diverges
        to NaN even with smoothing. We default to ``sigma=1.0`` with
        ``step=0.5`` instead, which is stable past 1000 iterations on the
        29-angle phantom. Pass ``smoothing_sigma=None`` for the
        unregularised variant (then defaults to ``step_size=0.15``).

        sinogram: [n_angles, Ny, N_det] CUDA → recon [Ny, Nz=N_det, Nx=N_det] CUDA.
        Slices are batched over Y rather than looped (the original code's per-slice
        loop is a Python-overhead artifact, not algorithmic).
        """
        n_angles, Ny, N_det = sinogram.shape
        if step_size is None:
            step_size = 0.5 if smoothing_sigma is not None else 0.15
        sino_b = sinogram.permute(1, 0, 2).contiguous()        # [Ny, n_angles, N_det]
        recon = torch.zeros(Ny, N_det, N_det, device=sinogram.device, dtype=sinogram.dtype)
        scale = math.pi / n_angles
        kernel_1d = (
            gaussian_kernel_1d(smoothing_sigma).to(device=sinogram.device, dtype=torch.float32)
            if smoothing_sigma is not None
            else None
        )
        for _ in range(num_iter):
            pred = self.radon.forward(recon)                   # [Ny, n_angles, N_det]
            diff = sino_b - pred
            update = self.radon.backprojection(ramp_filter(diff)) * scale
            recon += step_size * update
            if kernel_1d is not None:
                recon = gaussian_filter_2d_batched(recon, kernel_1d)
            if positivity:
                recon.clamp_(min=0.0)
        return recon
