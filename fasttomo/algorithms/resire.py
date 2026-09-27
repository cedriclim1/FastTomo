"""
RESIRE (Pham et al.). FROZEN: moved verbatim from the former
``fasttomo/algorithms.py``; no change to its math, normalisation, edge
handling or numerical output (tests/test_frozen_identity.py).
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from fasttomo.projection import (
    forward_project_resire,
    back_project_resire,
)
from fasttomo.algorithms._common import (
    _deprecated_kwarg,
    _reject_unknown_kwargs,
    _prepare_recon,
    _LossRecorder,
)


def run_resire(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 100,
    *,
    step_size: float = 1.0,
    Nz: int = -1,
    positivity: bool = True,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,   # [Nz, Nx], [1, Nz, Nx] or [Ny, Nz, Nx]
    sparsity: float = 0.0,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — overrides angles_deg
    warm_start: torch.Tensor | None = None,
    return_loss: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
    **deprecated: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """RESIRE, REal Space Iterative REconstruction (Miao group). CUDA, fp32. Frozen numerics.

    Projected gradient descent on ``||A x - b||^2``:
    ``x <- P_C(x + alpha * A^T (b - A x))`` with
    ``alpha = step_size / (n_angles * Nz)``, using the RESIRE voxel-driven
    scatter/gather projectors with 4-point supersampling
    (:func:`forward_project_resire`). Single-axis angles are wrapped as ZYZ
    triples ``(0, angle, 0)``; pass ``euler_zyz_deg`` for dual-axis tilt or
    general orientations. ``P_C``: support, positivity (default on),
    soft-threshold, in that order. Recorded loss: the data MSE before each
    update.

    Parameters
    ----------
    sinogram, angles_deg, n_iter
        fp32 CUDA sinogram, degrees, iterations (default 100).
    step_size : float
        Default 1.0.
    euler_zyz_deg : Tensor ``[n_angles, 3]``, optional
        Overrides ``angles_deg`` for the projectors.
    support_mask : Tensor, optional
        See Conventions (formerly ``support=``, still accepted with a
        DeprecationWarning); mutually exclusive with ``mask=True`` (before,
        ``support`` silently won).
    Nz, positivity, mask, sparsity, warm_start, return_loss, loss_callback,
    progress
        See Conventions of :mod:`fasttomo.algorithms`.

    Returns
    -------
    volume, or ``(volume, loss_history)``; see Conventions.
    """
    support_mask = _deprecated_kwarg(deprecated, "support", "support_mask",
                                     support_mask, "run_resire")
    _reject_unknown_kwargs(deprecated, "run_resire")
    setup = _prepare_recon(
        "run_resire", sinogram, angles_deg, Nz=Nz, n_iter=n_iter, require_cuda=True,
        warm_start=warm_start, positivity=positivity, mask=mask,
        support_mask=support_mask, sparsity=sparsity,
    )
    n_angles, N_det, Nz = setup.n_angles, setup.N_det, setup.Nz
    volume, constraints, n_iter = setup.volume0, setup.constraints, setup.n_iter

    # step size: matches RESIRE's dt = step_size / Num_pjs / dimz
    alpha = step_size / (n_angles * Nz)

    if euler_zyz_deg is None:
        # Wrap scalar tilt angles as ZYZ triples (φ=0, θ=angle, ψ=0)
        euler_zyz_deg = torch.zeros(n_angles, 3, dtype=torch.float32)
        euler_zyz_deg[:, 1] = angles_deg.cpu().float()
    elif tuple(euler_zyz_deg.shape) != (n_angles, 3):
        # A shorter table would broadcast the forward projection over the
        # sinogram in the residual; fail before the first iteration instead.
        raise ValueError(
            f"run_resire: euler_zyz_deg must have shape ({n_angles}, 3) to "
            f"match the sinogram, got {tuple(euler_zyz_deg.shape)}"
        )

    def _fp(v):
        return forward_project_resire(v, euler_zyz_deg, N_det=N_det)

    def _bp(s):
        return back_project_resire(s, euler_zyz_deg, Nz=Nz)

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_resire")

    for it in rec.iterate(n_iter):
        residual = sinogram - _fp(volume)
        if rec.wants_loss:
            rec.record(it, residual.pow(2).mean())
        volume   = volume + alpha * _bp(residual)
        volume   = constraints.apply(volume)

    return rec.finish(volume)
