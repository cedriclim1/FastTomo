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
  (used by SIRT/SART/RESIRE/FISTA) and, once per iteration after the
  optimiser step, by ``run_ad_recon``.

The two are different operations sharing the same name because they
serve the same modelling goal in two different optimisation regimes.
"""

from __future__ import annotations

import math
import numbers
import operator
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


def _real_or_none(value: object) -> float | None:
    """``value`` as a float if it is a real number: Python and numpy
    scalars (``numbers.Real``) and one-element real tensors count, bools do
    not. None otherwise."""
    if (isinstance(value, torch.Tensor) and value.numel() == 1
            and value.dtype != torch.bool and not value.is_complex()):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


def _check_weight(owner: str, name: str, value: object, *, positive: bool = False) -> None:
    """Raise ValueError unless ``value`` is a finite real number that is
    >= 0 (or > 0 when ``positive``). numpy and one-element tensor scalars
    are accepted (parameter sweeps built with ``np.linspace``)."""
    v = _real_or_none(value)
    ok = v is not None and math.isfinite(v) and (v > 0 if positive else v >= 0)
    if not ok:
        bound = "> 0" if positive else ">= 0"
        raise ValueError(f"{owner}.{name} must be a finite number {bound}, got {value!r}")


def _check_count(owner: str, name: str, value: object) -> None:
    """Raise ValueError unless ``value`` is a non-negative integer (Python
    or numpy integer, or an integer 0-d tensor; bools rejected)."""
    try:
        ok = not isinstance(value, bool) and operator.index(value) >= 0
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f"{owner}.{name} must be a non-negative integer, got {value!r}")


@dataclass
class OptimConfig:
    """Optimiser-loop knobs shared across AD reconstructions."""
    n_iter: int = 1000
    lr_volume: float = 0.05
    cosine_schedule: bool = True
    loss_fn: Literal["mse", "l1"] = "mse"

    def validate(self) -> None:
        """Raise ValueError on an invalid field (called by ``run_ad_recon``)."""
        _check_count("OptimConfig", "n_iter", self.n_iter)
        _check_weight("OptimConfig", "lr_volume", self.lr_volume)
        if self.loss_fn not in ("mse", "l1"):
            raise ValueError(
                f"OptimConfig.loss_fn must be 'mse' or 'l1', got {self.loss_fn!r}")


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

    def validate(self) -> None:
        """Raise ValueError unless every weight is finite and >= 0."""
        for name in ("tv_weight", "tv_xz_weight", "sparsity", "s3im_weight",
                     "tilt_ssim_weight"):
            _check_weight("RegularizerConfig", name, getattr(self, name))


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
    joint update. ``n_iter`` counts every iteration, warm-up included, and
    with ``cosine_schedule=True`` each optimiser's cosine schedule spans only
    the iterations in which it steps, so every learning rate anneals to ~0
    by the end of its own active iterations. Both staging fields require a
    refine flag.

    Pose parameters (Euler convention):
        ``refine_pose`` optimises one row of ZYZ Euler angles per view,
        ``(phi, theta, psi)`` in degrees, with rotation
        ``R = R_Z(psi) · R_Y(theta) · R_Z(phi)`` (the RESIRE convention of
        ``forward_project_3d``). ``theta`` (column 1) is the tilt angle; the
        default initialisation is ``(0, angles_deg[i], 0)``. ``phi`` and
        ``psi`` (columns 0 and 2) are in-plane rotations, penalised by
        ``in_plane_weight * (mean(phi²) + mean(psi²))`` when that is > 0.
        ``refine_shifts`` optimises per-view ``(detector-axis, tilt-axis)``
        shifts in pixels, applied by ``apply_shifts_2d``.

    Constraints applied after every iteration's optimiser step:
        * ``tilt_bound_deg`` clamps ``theta`` to ``angles_deg ± bound``
          around the *nominal* angles passed to ``run_ad_recon``;
        * ``antisymmetric_z`` (default True) sets ``psi = -phi``;
        * ``anchor_reference`` zeroes the pose and shift gradients of the
          view closest to 0 degrees, so that row keeps its initial value.
        These also act on ``pose_init_euler_deg``: the initialiser is used
        unchanged for iteration 0's forward projection, then after
        iteration 0 a ``theta`` outside the bound is clamped and ``psi`` is
        overwritten with ``-phi``. Set ``tilt_bound_deg=None`` and
        ``antisymmetric_z=False`` to keep an initialiser that violates them.

    The optional initialisers are engine-native Euler angles in degrees with
    shape ``[n_views, 3]`` (convention above) and XY shifts in pixels with
    shape ``[n_views, 2]``.

    The dataclass is mutable; ``run_ad_recon`` calls :meth:`validate`
    again, so fields edited after construction are checked too.
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
        self.validate()

    def validate(self) -> None:
        """Raise ValueError on an invalid or inconsistent field. Runs at
        construction and again inside ``run_ad_recon``."""
        refine = self.refine_pose or self.refine_shifts
        if self.anchor_reference and not refine:
            raise ValueError(
                "anchor_reference=True requires refine_pose=True or "
                "refine_shifts=True; nothing to anchor otherwise."
            )
        for name in ("pose_warmup_iters", "alternating_period"):
            _check_count("PoseRefinementConfig", name, getattr(self, name))
        if self.alternating_period > 0 and not refine:
            raise ValueError(
                "alternating_period > 0 requires refine_pose=True or "
                "refine_shifts=True; there is no pose-only block otherwise."
            )
        if self.pose_warmup_iters > 0 and not refine:
            raise ValueError(
                "pose_warmup_iters > 0 requires refine_pose=True or "
                "refine_shifts=True; there is no pose optimiser to delay."
            )
        for name in ("lr_angles", "lr_shifts"):
            _check_weight("PoseRefinementConfig", name, getattr(self, name))
        _check_weight("PoseRefinementConfig", "in_plane_weight", self.in_plane_weight)
        if self.tilt_bound_deg is not None:
            _check_weight("PoseRefinementConfig", "tilt_bound_deg", self.tilt_bound_deg)
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
    # Post-step L1 soft-threshold f <- sign(f) * max(|f| - sparsity, 0), in
    # volume units, applied once per iteration after the mask and positivity
    # (last in apply() and in run_ad_recon; positivity and a non-negative
    # mask commute, so their relative order does not matter). Not a loss
    # term; see RegularizerConfig.sparsity.
    sparsity: float = 0.0

    def validate(self) -> None:
        """Raise ValueError on an invalid field (called by ``run_ad_recon``)."""
        _check_weight("ConstraintConfig", "sparsity", self.sparsity)
        if not (self.mask is None or isinstance(self.mask, (bool, torch.Tensor))):
            raise ValueError("ConstraintConfig.mask must be None, a bool or a "
                             f"tensor, got {type(self.mask).__name__}")

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

    def validate(self) -> None:
        """Validate every sub-config. ``run_ad_recon`` calls this on entry,
        so edits made after construction are checked as well."""
        for name, cls in (("optim", OptimConfig), ("reg", RegularizerConfig),
                          ("constraints", ConstraintConfig)):
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f"AdReconConfig.{name} must be a {cls.__name__}, "
                                 f"got {type(getattr(self, name)).__name__}")
        if self.pose is not None and not isinstance(self.pose, PoseRefinementConfig):
            raise ValueError("AdReconConfig.pose must be a PoseRefinementConfig or None, "
                             f"got {type(self.pose).__name__}")
        self.optim.validate()
        self.reg.validate()
        self.constraints.validate()
        if self.pose is not None:
            self.pose.validate()
