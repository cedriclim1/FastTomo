"""SSIM and NMSE for reconstruction-vs-phantom evaluation."""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def ssim(recon: torch.Tensor | np.ndarray, reference: torch.Tensor | np.ndarray) -> float:
    """3D SSIM of recon vs reference.

    ``data_range`` is set to the reference's intensity span (reference.max() -
    reference.min()) so the score reflects how well recon matches the
    ground-truth scale. recon does not need to share that range.
    """
    a = _to_numpy(recon)
    b = _to_numpy(reference)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    data_range = float(b.max() - b.min())
    return float(structural_similarity(b, a, data_range=data_range))


def nmse(recon: torch.Tensor | np.ndarray, reference: torch.Tensor | np.ndarray) -> float:
    """Normalised MSE: ||recon - reference||² / ||reference||²."""
    a = _to_numpy(recon)
    b = _to_numpy(reference)
    num = float(np.sum((a - b) ** 2))
    den = float(np.sum(b ** 2))
    return num / den
