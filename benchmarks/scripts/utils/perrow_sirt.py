"""SIRT treatments for continuous-rotation tilt series.

Fast-perp data is exactly separable into one 2-D SIRT problem per tilt-axis
slice because its angle varies along that independent axis.  Fast-par data is
not representable by the scalar-angle Radon operator: its angle varies along
the detector within a projection.  The supported fast-par treatment is
therefore the deliberately approximate per-frame mid-angle baseline when
using :func:`fasttomo.run_sirt`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

import fasttomo as ft

from .perrow_ad import (
    ContinuousDataset,
    generator_to_fasttomo_sinogram,
    load_continuous_dataset,
)


SIRTMode = Literal["perp-exact", "midangle"]


@dataclass(frozen=True)
class SIRTResult:
    """Reconstructed FastTomo-frame volume and global sinogram MSE history."""

    volume: torch.Tensor
    loss_history: torch.Tensor
    mode: SIRTMode


def sirt_continuous(
    dataset: ContinuousDataset | str | Path,
    mode: SIRTMode,
    *,
    n_iter: int = 50,
    omega: float = 1.0,
    device: str | torch.device = "cuda",
    seed: int = 0,
    num_threads: int | None = None,
) -> SIRTResult:
    """Run one of the continuous-rotation SIRT treatments.

    The returned volume uses FastTomo order ``[image_X, z, image_Y]``.  For
    ``perp-exact``, each history entry is the MSE over the complete measured
    tilt series, obtained by averaging the equal-sized slice histories
    returned by :func:`fasttomo.run_sirt`.
    """
    loaded = (
        dataset
        if isinstance(dataset, ContinuousDataset)
        else load_continuous_dataset(dataset)
    )
    _validate_options(mode, loaded, n_iter, omega, num_threads)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    generator_frames = torch.as_tensor(
        loaded.tilt_series, dtype=torch.float32, device=resolved_device
    )
    sinogram = generator_to_fasttomo_sinogram(generator_frames)
    image_y = loaded.frame_shape[1]

    if mode == "midangle":
        angles = torch.as_tensor(
            loaded.angles, dtype=torch.float32, device=resolved_device
        )
        volume, history = ft.run_sirt(
            sinogram,
            angles,
            n_iter=n_iter,
            omega=omega,
            Nz=image_y,
            positivity=True,
            return_loss=True,
            num_threads=num_threads,
        )
        return SIRTResult(volume=volume, loss_history=history, mode=mode)

    per_slice_angles = torch.as_tensor(
        loaded.angles_per_line, dtype=torch.float32, device=resolved_device
    )
    slice_volumes: list[torch.Tensor] = []
    slice_histories: list[torch.Tensor] = []
    for slice_index in range(loaded.frame_shape[2]):
        slab = sinogram[:, slice_index : slice_index + 1, :]
        slab_volume, slab_history = ft.run_sirt(
            slab,
            per_slice_angles[:, slice_index],
            n_iter=n_iter,
            omega=omega,
            Nz=image_y,
            positivity=True,
            return_loss=True,
            num_threads=num_threads,
        )
        slice_volumes.append(slab_volume[0])
        slice_histories.append(slab_history)

    volume = torch.stack(slice_volumes, dim=0)
    history = torch.stack(slice_histories, dim=0).mean(dim=0)
    return SIRTResult(volume=volume, loss_history=history, mode=mode)


def _validate_options(
    mode: str,
    dataset: ContinuousDataset,
    n_iter: int,
    omega: float,
    num_threads: int | None,
) -> None:
    if mode not in ("perp-exact", "midangle"):
        raise ValueError(f"mode must be 'perp-exact' or 'midangle', got {mode!r}")
    if mode == "perp-exact" and dataset.arm != "fast-perp":
        raise ValueError(
            "perp-exact is only valid for fast-perp data; fast-par angles vary "
            "along the detector and cannot be represented by one Radon angle "
            "per projection"
        )
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer")
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be positive and finite")
    if num_threads is not None and num_threads < 1:
        raise ValueError("num_threads must be at least one when supplied")
