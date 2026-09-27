"""
Support masks and the post-step constraints of the reconstruction algorithms.

Every algorithm that constrains its iterate applies, after each update and
in this order (the *constraints order* the algorithm docstrings refer to):

1. the support: ``volume * support`` (the inscribed cylinder for
   ``mask=True``, or the user's ``support_mask``; a soft mask in ``[0, 1]``
   scales voxels instead of zeroing them),
2. positivity: ``volume.clamp(min=0)``,
3. the soft-threshold ``sign(v) * max(|v| - t, 0)`` (the proximal step of
   ``t * ||v||_1``, the ``sparsity=`` keyword of the ``run_*`` functions and
   ``ConstraintConfig.soft_threshold``).

With a non-negative support, steps 1 and 2 commute, so ``run_ad_recon``'s
positivity-then-mask order gives the same result.

A binary support leaves the out-of-support voxels of the result exactly 0,
except in GENFIRE (frozen): it applies the support in real space on its
oversampled grid, inside the loop, and multiplies only a user
``support_mask`` into the cropped output. With ``mask=True`` (its default
when no ``support_mask`` is given) voxels outside the cylinder are 0 only up
to FFT round-off (~1e-7 in float32).

Support-mask contract (frozen by the 2026-06 desktop-app handoff): a
``support_mask`` tensor has shape ``[Nz, Nx]``, ``[1, Nz, Nx]`` or
``[Ny, Nz, Nx]`` (the 2-D and leading-1 forms broadcast along the tilt axis),
any real or bool dtype (cast to the compute dtype) and any device (moved to
the volume's); values are expected in ``{0, 1}``. It is mutually exclusive
with ``mask=True``: one support per reconstruction, ValueError otherwise.

This module depends on torch and numpy only.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["cylindrical_mask", "apply_constraints", "apply_constraints_"]


def cylindrical_mask(
    Nz: int,
    Nx: int,
    radius: float | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Inscribed-cylinder support mask for the XZ reconstruction plane.

    Parameters
    ----------
    Nz, Nx : int
        Reconstruction depth and width.
    radius : float, optional
        Cylinder radius in voxels. Default ``min(Nz, Nx) / 2``, which
        inscribes the cylinder in the smaller dimension.
    device : torch.device or str, optional
        Device of the returned tensor (default CPU).

    Returns
    -------
    Tensor, shape ``[Nz, Nx]``, float32
        1.0 inside the cylinder, 0.0 outside. Multiply a ``[Ny, Nz, Nx]``
        volume by ``mask.unsqueeze(0)`` to apply it. Voxel ``i`` sits at
        ``i - (N - 1) / 2``, the centring of the projection kernels.
    """
    r = float(min(Nz, Nx)) / 2.0 if radius is None else radius
    iz = np.linspace(-(Nz - 1) / 2.0, (Nz - 1) / 2.0, Nz, dtype=np.float32)
    ix = np.linspace(-(Nx - 1) / 2.0, (Nx - 1) / 2.0, Nx, dtype=np.float32)
    mask_np = (iz[:, None] ** 2 + ix[None, :] ** 2 <= r ** 2).astype(np.float32)
    mask = torch.from_numpy(mask_np)
    return mask.to(device) if device is not None else mask


def _maybe_cylindrical_mask(
    Nz: int,
    Nx: int,
    device: torch.device,
    apply: bool,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Return a ``[1, Nz, Nx]`` cylindrical mask if ``apply``, else None.

    The mask is cast to ``dtype`` so callers can multiply it against a
    volume in any supported precision (fp16 / fp32) without an upcast.
    """
    if not apply:
        return None
    return cylindrical_mask(Nz, Nx, device=device).to(dtype).unsqueeze(0)


def _resolve_support_mask(
    support_mask: torch.Tensor,
    Ny: int,
    Nz: int,
    Nx: int,
    device: torch.device,
    dtype: torch.dtype,
    op_name: str,
) -> torch.Tensor:
    """Validate and normalise a user-supplied support mask (the contract in
    the module docstring).

    Returns a 3-D ``[1 | Ny, Nz, Nx]`` tensor on ``device`` cast to
    ``dtype``, ready to multiply against the volume. Raises TypeError if
    ``support_mask`` is not a tensor, ValueError on a wrong shape.
    """
    if not isinstance(support_mask, torch.Tensor):
        raise TypeError(
            f"{op_name}: support_mask must be a torch.Tensor, "
            f"got {type(support_mask).__name__}"
        )
    if support_mask.ndim == 2:
        if support_mask.shape != (Nz, Nx):
            raise ValueError(
                f"{op_name}: support_mask must have shape ({Nz}, {Nx}) "
                f"when 2-D; got {tuple(support_mask.shape)}."
            )
        support_mask = support_mask.unsqueeze(0)
    elif support_mask.ndim == 3:
        if support_mask.shape[1:] != (Nz, Nx) or support_mask.shape[0] not in (1, Ny):
            raise ValueError(
                f"{op_name}: support_mask must have shape (1, {Nz}, {Nx}) or "
                f"({Ny}, {Nz}, {Nx}) when 3-D; got {tuple(support_mask.shape)}."
            )
    else:
        raise ValueError(
            f"{op_name}: support_mask must be 2-D or 3-D; "
            f"got {support_mask.ndim}-D."
        )
    return support_mask.to(device=device, dtype=dtype).contiguous()


# dtypes for which ``softshrink`` gives exactly ``sign(v) * max(|v| - t, 0)``:
# it computes ``v -+ t`` in the tensor's dtype with ``t`` rounded to it, the
# same arithmetic as the reference expression for fp32 and fp64 (up to the
# sign of a zero result: +0 where the expression gives -0). fp16/bf16 keep
# the expression, whose fp32 op-math uses the unrounded threshold.
_SOFTSHRINK_DTYPES = (torch.float32, torch.float64)


def _soft_threshold(vol: torch.Tensor, weight: float) -> torch.Tensor:
    """L1 soft-threshold ``sign(v) * max(|v| - weight, 0)`` (the proximal
    operator of ``weight * ||v||_1``), as a new tensor. One elementwise pass
    (``softshrink``) for fp32/fp64."""
    if vol.dtype in _SOFTSHRINK_DTYPES:
        return torch.nn.functional.softshrink(vol, weight)
    return vol.sign() * (vol.abs() - weight).clamp_(min=0.0)


def _soft_threshold_(vol: torch.Tensor, weight: float) -> None:
    """In-place `_soft_threshold`: ``vol`` keeps its storage (leaf tensors,
    CUDA-graph static buffers), with no temporaries for fp32/fp64."""
    if vol.dtype in _SOFTSHRINK_DTYPES:
        torch.ops.aten.softshrink.out(vol, weight, out=vol)
    else:
        vol.copy_(_soft_threshold(vol, weight))


def apply_constraints(
    volume: torch.Tensor,
    *,
    positivity: bool = False,
    support_mask: torch.Tensor | None = None,
    soft_threshold: float = 0.0,
) -> torch.Tensor:
    """Apply the post-step constraints in the constraints order (module
    docstring) and return a new tensor.

    Parameters
    ----------
    volume : Tensor, shape ``[Ny, Nz, Nx]``
    positivity : bool
        Clamp at 0.
    support_mask : Tensor, optional
        Already resolved support, broadcastable against ``volume`` and in
        its dtype and device (see ``_resolve_support_mask``).
    soft_threshold : float
        Soft-threshold ``t >= 0``; 0 disables it.
    """
    if support_mask is not None:
        volume = volume * support_mask
    if positivity:
        volume = volume.clamp(min=0.0)
    if soft_threshold > 0.0:
        volume = _soft_threshold(volume, soft_threshold)
    return volume


def apply_constraints_(
    volume: torch.Tensor,
    *,
    positivity: bool = False,
    support_mask: torch.Tensor | None = None,
    soft_threshold: float = 0.0,
) -> None:
    """In-place `apply_constraints`: ``volume`` keeps its storage, as leaf
    tensors under ``torch.no_grad`` and CUDA-graph static buffers need."""
    if support_mask is not None:
        volume.mul_(support_mask)
    if positivity:
        volume.clamp_(min=0.0)
    if soft_threshold > 0.0:
        _soft_threshold_(volume, soft_threshold)
