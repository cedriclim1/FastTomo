"""
Configuration dataclasses for FastTomo's reconstruction algorithms.

The dataclasses here are intentionally mutable so notebook callers can
tweak nested fields inline (``cfg.reg.tv_weight = 0.01``). Any function
that accepts one of these should default the argument to ``None`` and
instantiate inside, to avoid the mutable-default footgun.

Two L1 knobs, deliberately with different names:

* :attr:`RegularizerConfig.sparsity` is an autograd loss term added each
  iteration as ``λ · mean(|f|)`` (``run_ad_recon``).
* :attr:`ConstraintConfig.soft_threshold` (formerly ``sparsity``) is a
  post-step soft-threshold ``f ← sign(f) · max(|f| − λ, 0)`` applied by
  :meth:`ConstraintConfig.apply`; ``run_ad_recon`` applies it once per
  iteration in which the volume was stepped. The ``sparsity=`` keyword of
  the classical ``run_*`` functions is this same soft-threshold.

The runtime constraint logic lives in :mod:`fasttomo.constraints`.
"""

from __future__ import annotations

import math
import numbers
import operator
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from fasttomo.constraints import (
    _maybe_cylindrical_mask,
    _soft_threshold,
    apply_constraints,
    apply_constraints_,
)


__all__ = [
    "OptimConfig",
    "RegularizerConfig",
    "PoseRefinementConfig",
    "ConstraintConfig",
    "AdReconConfig",
    "MbirConfig",
]


# Former name of the soft-threshold helper, kept for existing imports.
_apply_sparsity = _soft_threshold


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


_UNSET: Any = object()


def _warn_renamed(old: str, new: str) -> None:
    warnings.warn(
        f"ConstraintConfig.{old} is deprecated, use {new} instead; the old name "
        "will be removed in the next release.",
        DeprecationWarning,
        stacklevel=3,
    )


def _split_mask(mask: object) -> tuple[bool, Any]:
    """Legacy ``ConstraintConfig.mask`` value -> ``(cylinder, support_mask)``."""
    if mask is None or mask is False:
        return False, None
    if mask is True:
        return True, None
    return False, mask      # a tensor, or garbage that validate() rejects


@dataclass(init=False)
class ConstraintConfig:
    """Post-step constraints applied after every iteration of
    ``run_ad_recon`` (the other algorithms build one from their flat
    ``positivity`` / ``mask`` / ``support_mask`` / ``sparsity`` keywords).

    Fields
    ------
    positivity : bool
        Clamp the volume at 0.
    cylinder : bool
        Restrict the volume to the inscribed cylinder of the XZ plane
        (``cylindrical_mask``). Mutually exclusive with ``support_mask``.
    support_mask : Tensor or None
        Arbitrary support, ``[Nz, Nx]``, ``[1, Nz, Nx]`` or
        ``[Ny, Nz, Nx]`` (contract in :mod:`fasttomo.constraints`).
    soft_threshold : float
        Post-step L1 soft-threshold ``f <- sign(f) * max(|f| - t, 0)`` in
        volume units (``>= 0``; 0 disables it). Not a loss term: compare
        ``RegularizerConfig.sparsity``, which adds ``sparsity * mean(|f|)``
        to the AD loss.

    The order is the constraints order of :mod:`fasttomo.constraints`:
    support, positivity, soft-threshold. `apply` honours both support forms
    (``cylinder`` is built for the volume's ``(Nz, Nx)`` on first use).

    Deprecated aliases (one release, DeprecationWarning): ``mask`` (``True``
    -> ``cylinder=True``, a tensor -> ``support_mask``) and ``sparsity``
    (-> ``soft_threshold``), as constructor keywords and as attributes.
    Everything after ``positivity`` is keyword-only.
    """
    positivity: bool = False
    cylinder: bool = False
    support_mask: torch.Tensor | None = None
    soft_threshold: float = 0.0

    def __init__(
        self,
        positivity: bool = False,
        *,
        cylinder: bool = False,
        support_mask: torch.Tensor | None = None,
        soft_threshold: float = 0.0,
        mask: Any = _UNSET,
        sparsity: Any = _UNSET,
    ) -> None:
        if mask is not _UNSET:
            _warn_renamed("mask", "cylinder or support_mask")
            if cylinder or support_mask is not None:
                raise TypeError("ConstraintConfig: pass cylinder/support_mask or the "
                                "deprecated mask, not both.")
            cylinder, support_mask = _split_mask(mask)
        if sparsity is not _UNSET:
            _warn_renamed("sparsity", "soft_threshold")
            if soft_threshold != 0.0:
                raise TypeError("ConstraintConfig: pass soft_threshold or the "
                                "deprecated sparsity, not both.")
            soft_threshold = sparsity
        self.positivity = positivity
        self.cylinder = cylinder
        self.support_mask = support_mask
        self.soft_threshold = soft_threshold

    # ── deprecated attribute aliases ──────────────────────────────────────
    @property
    def mask(self) -> bool | torch.Tensor | None:
        """Deprecated: ``support_mask`` if set, else ``True`` for the
        cylinder, else None."""
        _warn_renamed("mask", "cylinder or support_mask")
        if self.support_mask is not None:
            return self.support_mask
        return True if self.cylinder else None

    @mask.setter
    def mask(self, value: object) -> None:
        _warn_renamed("mask", "cylinder or support_mask")
        self.cylinder, self.support_mask = _split_mask(value)

    @property
    def sparsity(self) -> float:
        """Deprecated alias of ``soft_threshold``."""
        _warn_renamed("sparsity", "soft_threshold")
        return self.soft_threshold

    @sparsity.setter
    def sparsity(self, value: float) -> None:
        _warn_renamed("sparsity", "soft_threshold")
        self.soft_threshold = value

    def validate(self) -> None:
        """Raise ValueError on an invalid field or on both supports set
        (called by ``run_ad_recon``)."""
        _check_weight("ConstraintConfig", "soft_threshold", self.soft_threshold)
        if not isinstance(self.cylinder, bool):
            raise ValueError("ConstraintConfig.cylinder must be a bool, "
                             f"got {type(self.cylinder).__name__}")
        if self.support_mask is not None and not isinstance(self.support_mask, torch.Tensor):
            raise ValueError("ConstraintConfig.support_mask must be None or a tensor, "
                             f"got {type(self.support_mask).__name__}")
        if self.cylinder and self.support_mask is not None:
            raise ValueError("ConstraintConfig: cylinder=True and support_mask are "
                             "mutually exclusive — pass one support per reconstruction.")

    def _support_for(self, volume: torch.Tensor) -> torch.Tensor | None:
        """The support to multiply ``volume`` by: ``support_mask``, or the
        cylinder for the volume's ``(Nz, Nx)`` (cached), or None."""
        if self.support_mask is not None:
            return self.support_mask
        if not self.cylinder:
            return None
        key = (volume.shape[-2], volume.shape[-1], volume.device, volume.dtype)
        cache = self.__dict__.setdefault("_cylinder_cache", {})
        if key not in cache:
            cache[key] = _maybe_cylindrical_mask(key[0], key[1], volume.device, True,
                                                 dtype=volume.dtype)
        return cache[key]

    def apply(self, volume: torch.Tensor) -> torch.Tensor:
        """Return ``volume`` with the constraints applied (new tensor)."""
        return apply_constraints(
            volume, positivity=self.positivity,
            support_mask=self._support_for(volume), soft_threshold=self.soft_threshold,
        )

    def apply_inplace_(self, volume: torch.Tensor) -> None:
        """In-place variant for leaf tensors and CUDA Graph paths."""
        apply_constraints_(
            volume, positivity=self.positivity,
            support_mask=self._support_for(volume), soft_threshold=self.soft_threshold,
        )


@dataclass
class MbirConfig:
    """Optional bundle of :func:`fasttomo.run_mbir`'s tuning keywords.

    ``run_mbir(sino, angles, config=MbirConfig(huber_T=3.0))`` is the same
    call as ``run_mbir(sino, angles, huber_T=3.0)``; the keywords stay the
    primary interface. Passing ``config`` together with a tuning keyword
    that differs from its default raises TypeError. Field meanings and
    defaults are those of ``run_mbir``'s keywords of the same name.
    """
    sigma_f: float = 1.0
    qggmrf_p: float = 1.2
    qggmrf_q: float = 2.0
    qggmrf_c: float = 1e-3
    huber_T: float = float("inf")
    huber_delta: float = 0.5
    estimate_offset: bool = True
    estimate_variance: bool = True
    sigma_init: float | None = None
    sigma_min: float | None = None
    multires_levels: int = 1
    convergence_tol: float = 1e-4
    seed: int = 0
    sigma_f_per_scale: list[float] | None = None
    sweep_kernel: str = "auto"


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
