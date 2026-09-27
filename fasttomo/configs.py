"""
Configuration dataclasses for FastTomo's reconstruction algorithms.

The dataclasses here are intentionally mutable so notebook callers can
tweak nested fields inline (``cfg.reg.tv_weight = 0.01``). Any function
that accepts one of these should default the argument to ``None`` and
instantiate inside, to avoid the mutable-default footgun.

A note on the dual semantics of ``sparsity``:

* :attr:`RegularizerConfig.sparsity` is an autograd loss term added each
  iteration as ``λ · mean(|f|)`` (used by ``run_ad_recon``).
* :attr:`ConstraintConfig.sparsity` is a post-step soft-threshold
  ``f ← sign(f) · max(|f| − λ, 0)`` applied by :meth:`ConstraintConfig.apply`
  (used by SIRT/SART/RESIRE/FISTA).

The two are different operations sharing the same name because they
serve the same modelling goal in two different optimisation regimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


__all__ = [
    "OptimConfig",
    "RegularizerConfig",
    "PoseRefinementConfig",
    "ConstraintConfig",
    "AdReconConfig",
]


def _apply_sparsity(vol: torch.Tensor, weight: float) -> torch.Tensor:
    """L1 soft-threshold (proximal operator of weight·‖·‖₁)."""
    return vol.sign() * (vol.abs() - weight).clamp_(min=0.0)


@dataclass
class OptimConfig:
    """Optimiser-loop knobs shared across AD reconstructions."""
    n_iter: int = 1000
    lr_volume: float = 0.05
    cosine_schedule: bool = True
    loss_fn: Literal["mse", "l1"] = "mse"


@dataclass
class RegularizerConfig:
    """Loss-augmenting regularisation weights for AD reconstructions.

    All weights default to 0.0 (disabled). See ``run_ad_recon`` for the
    semantics of each term.
    """
    tv_weight: float = 0.0
    tv_xz_weight: float = 0.0
    sparsity: float = 0.0           # loss term: + sparsity * mean(|f|)
    s3im_weight: float = 0.0
    tilt_ssim_weight: float = 0.0


@dataclass
class PoseRefinementConfig:
    """Joint pose / shift refinement knobs for AD reconstructions.

    Pass ``pose=PoseRefinementConfig(...)`` to ``AdReconConfig`` to enable
    refinement, or leave ``pose=None`` (the default) to disable both pose
    and shift refinement entirely.

    The ``refine_pose`` and ``refine_shifts`` flags inside this config are
    independent: either, neither, or both may be true. ``anchor_reference``
    requires at least one of them to be true (validated in ``__post_init__``).

    ``pose_warmup_iters`` delays pose/shift optimiser steps while the volume
    forms. After warmup, a positive ``alternating_period`` alternates
    volume-only and pose-only blocks of that length; zero keeps the historical
    joint update. The optional initialisers are engine-native Euler angles in
    degrees with shape ``[n_views, 3]`` and XY shifts in pixels with shape
    ``[n_views, 2]``.
    """
    refine_pose: bool = False
    refine_shifts: bool = False
    lr_angles: float = 1e-2
    lr_shifts: float = 1e-2
    in_plane_weight: float = 0.0
    tilt_bound_deg: float | None = 1.0
    antisymmetric_z: bool = True
    anchor_reference: bool = False
    pose_warmup_iters: int = 0
    alternating_period: int = 0
    pose_init_euler_deg: Any | None = None
    pose_init_shifts: Any | None = None

    def __post_init__(self) -> None:
        if self.anchor_reference and not (self.refine_pose or self.refine_shifts):
            raise ValueError(
                "anchor_reference=True requires refine_pose=True or "
                "refine_shifts=True; nothing to anchor otherwise."
            )
        for name in ("pose_warmup_iters", "alternating_period"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}"
                )
        if self.alternating_period > 0 and not (
            self.refine_pose or self.refine_shifts
        ):
            raise ValueError(
                "alternating_period > 0 requires refine_pose=True or "
                "refine_shifts=True; there is no pose-only block otherwise."
            )
        if self.pose_init_euler_deg is not None and not self.refine_pose:
            raise ValueError("pose_init_euler_deg requires refine_pose=True.")
        if self.pose_init_shifts is not None and not self.refine_shifts:
            raise ValueError("pose_init_shifts requires refine_shifts=True.")


@dataclass
class ConstraintConfig:
    """Post-step constraints applied each iteration.

    Used by SIRT/SART/RESIRE/FISTA and (via field access) by run_ad_recon.
    Construct once before the iteration loop; for SIRT/SART/etc. callers,
    the ``mask`` field must hold a precomputed ``[1, Nz, Nx]`` tensor.

    For ``run_ad_recon``, ``mask`` may be ``True`` — in which case
    ``run_ad_recon`` materialises a ``cylindrical_mask`` internally — or
    a precomputed tensor, or ``None``/``False`` to disable.

    ``apply`` / ``apply_inplace_`` operate only on the tensor case: a
    bool ``mask`` is treated as a no-op there (callers that want
    masking via ``apply`` must materialise the mask themselves).

    Application order: mask → positivity → sparsity.
    """
    positivity: bool = False
    mask: bool | torch.Tensor | None = None
    sparsity: float = 0.0           # post-step soft-threshold (not an AD loss term)

    def apply(self, volume: torch.Tensor) -> torch.Tensor:
        if isinstance(self.mask, torch.Tensor):
            volume = volume * self.mask
        if self.positivity:
            volume = volume.clamp(min=0.0)
        if self.sparsity > 0.0:
            volume = _apply_sparsity(volume, self.sparsity)
        return volume

    def apply_inplace_(self, volume: torch.Tensor) -> None:
        """In-place variant for leaf tensors and CUDA Graph paths."""
        if isinstance(self.mask, torch.Tensor):
            volume.mul_(self.mask)
        if self.positivity:
            volume.clamp_(min=0.0)
        if self.sparsity > 0.0:
            volume.copy_(_apply_sparsity(volume, self.sparsity))


@dataclass
class AdReconConfig:
    """Top-level config for :func:`fasttomo.run_ad_recon`.

    Nested sub-configs cluster the 26 underlying knobs into four groups:
    optimiser, regularisation, post-step constraints, and pose/shift
    refinement. ``pose=None`` disables refinement entirely.
    """
    optim: OptimConfig = field(default_factory=OptimConfig)
    reg: RegularizerConfig = field(default_factory=RegularizerConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    pose: PoseRefinementConfig | None = None
