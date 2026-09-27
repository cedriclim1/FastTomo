"""
Reconstruction algorithms for FastTomo.

All public functions accept and return torch.Tensor (float32).

Volume convention:  [Ny, Nz, Nx]
Sinogram convention: [n_angles, Ny, N_det]
Angles: 1-D tensor, degrees.
"""

from __future__ import annotations

import inspect
import logging
import math
import numbers
import operator
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import fasttomo._core as _core
from fasttomo.autograd import (
    _assert_dtype_match, _check_shapes, _data_ptr, euler_zyz_to_rotmat, _launch,
    _resolve_num_threads,
)
from fasttomo.projection import (
    forward_project,
    back_project,
    forward_project_3d,
    back_project_3d,
    forward_project_resire,
    back_project_resire,
)
from fasttomo.constraints import (
    _maybe_cylindrical_mask,
    _resolve_support_mask,
    _soft_threshold,
    cylindrical_mask,
)
from fasttomo.utilities import apply_shifts_2d, ramp_filter, tv_loss_3d
from fasttomo.regularizers import s3im_loss, tilt_ssim_loss, tv_loss_xz
from fasttomo.projector import ParallelProjector, RotmatProjector, _drain_deferred_handles
from fasttomo.configs import (
    AdReconConfig,
    ConstraintConfig,
    MbirConfig,
    OptimConfig,
    PoseRefinementConfig,
    RegularizerConfig,
)

_log = logging.getLogger(__name__)


# ── module-level constants ─────────────────────────────────────────────────

# GENFIRE resolution extension/suppression schedule control points.
# np.interp maps iteration progress [0,1] through these pivots to produce
# the per-iteration Fourier cutoff fraction.
_GENFIRE_RES_EXTENSION_PIVOTS = (0.0, 0.5, 0.95, 0.5, 0.0)

# FISTA: the power-iteration estimate of L = ||AᵀA|| is multiplied by this
# margin so the step 1/L stays inside the contraction region even when the
# iteration has not fully converged.
_LIPSCHITZ_SAFETY = 1.01

# Added to a norm before dividing by it, where the norm can be exactly 0
# (power iteration from a vector that A maps to zero).
_NORM_EPS = 1e-30

# Dual step of the TV proximal solvers (FGP / Chambolle): 1 / L with
# L = 12 the upper bound of ||grad ∘ div|| in 3-D (8 in 2-D).
_TV_DUAL_STEP = 1.0 / 12.0

# Seed of run_ad_recon's S3IM patch sampler (fixed, so runs are repeatable).
_S3IM_SEED = 42

# Floor of the denominator of relative changes and R-factors
# (refine_angles, MBIR convergence check): a zero reference gives a large
# ratio instead of NaN.
_REL_EPS = 1e-12

# MBIR constants.
# The convergence check compares every _MBIR_CONVERGENCE_STRIDE-th voxel,
# O(Nvox / stride) work, negligible next to a sweep.
_MBIR_CONVERGENCE_STRIDE = 8
# Offset update skipped for a view whose summed surrogate weight is below
# this; also the |r| floor of the Huber tail weight delta*T/|r|.
_MBIR_WEIGHT_EPS = 1e-30
# Default sigma_init is the sinogram std; below this (a blank sinogram) it
# falls back to 1.0.
_MBIR_BLANK_STD = 1e-6
# Default sigma_min = sigma_init * _MBIR_SIGMA_MIN_RATIO (keeps sigma from
# collapsing to 0 when the residual vanishes).
_MBIR_SIGMA_MIN_RATIO = 1e-3


# ── private helpers shared across algorithms ───────────────────────────────


def _ad_recon_step_schedule(
    iteration: int,
    *,
    pose_warmup_iters: int = 0,
    alternating_period: int = 0,
) -> tuple[bool, bool]:
    """Return ``(step_volume, step_pose)`` for one AD iteration.

    Defaults deliberately return the historical joint schedule. Warmup is
    always volume-only. Alternation starts with a volume-only block, then a
    pose-only block, and repeats.
    """
    if iteration < pose_warmup_iters:
        return True, False
    if alternating_period == 0:
        return True, True
    block = (iteration - pose_warmup_iters) // alternating_period
    step_volume = block % 2 == 0
    return step_volume, not step_volume


def _ad_recon_active_steps(
    n_iter: int,
    *,
    pose_warmup_iters: int = 0,
    alternating_period: int = 0,
) -> tuple[int, int]:
    """``(volume_steps, pose_steps)``: how many of the ``n_iter`` iterations
    step the volume optimiser and the pose/shift optimisers. Each cosine
    schedule uses its optimiser's count as ``T_max`` so its learning rate
    reaches ~0 at the end of that optimiser's own active iterations."""
    vol = pose = 0
    for it in range(n_iter):
        step_volume, step_pose = _ad_recon_step_schedule(
            it, pose_warmup_iters=pose_warmup_iters,
            alternating_period=alternating_period)
        vol += step_volume
        pose += step_pose
    return vol, pose


def _prepare_ad_recon_pose_init(
    value: object,
    *,
    expected_shape: tuple[int, int],
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Convert and validate one engine-native AD pose initialiser."""
    try:
        initial = torch.as_tensor(value, dtype=torch.float32, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if tuple(initial.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(initial.shape)}"
        )
    if not bool(torch.isfinite(initial).all()):
        raise ValueError(f"{name} must contain only finite values")
    return initial.detach().clone()


def _assert_warm_start_shape(
    volume: torch.Tensor,
    expected_shape: tuple[int, int, int],
    op_name: str,
) -> None:
    """Validate warm-start volume shape before arithmetic can broadcast it.

    Without this guard, a volume with wrong leading-dim (e.g. ``[1, N, N]``
    when the sinogram has ``Ny > 1``) would silently broadcast inside
    ``volume + omega * C_vol * _bp(...)`` and produce a corrupted
    reconstruction with no warning."""
    if tuple(volume.shape) != expected_shape:
        raise ValueError(
            f"{op_name}: warm-start volume must have shape {expected_shape}, "
            f"got {tuple(volume.shape)}."
        )


def _assert_projector_output_shape(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    tensor_name: str,
    op_name: str,
) -> None:
    """Validate projector outputs before downstream arithmetic can broadcast.

    Cached projectors are user-supplied and may be built for a different
    geometry. Without this guard, singleton dimensions (e.g. Ny=1) can
    silently broadcast across the target tensor and corrupt reconstruction
    updates with no explicit error.
    """
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{op_name}: projector returned {tensor_name} with shape "
            f"{tuple(tensor.shape)}, expected {expected_shape}. "
            "Ensure projector geometry matches the current sinogram/volume."
        )


def _finish_iters(
    volume: torch.Tensor,
    losses: list[float],
    return_loss: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Standardised return for iterative reconstruction algorithms."""
    if return_loss:
        return volume, torch.tensor(losses, dtype=torch.float32)
    return volume


# ── deprecated keyword aliases ─────────────────────────────────────────────
#
# Renamed keywords keep working for one release: the old name is accepted
# through ``**deprecated`` and emits a DeprecationWarning. Everything after
# the core positionals is keyword-only, so a call written for an old
# positional order raises TypeError instead of binding to the wrong
# parameter. CHANGELOG.md lists every rename.


def _is_default(value: object, default: object) -> bool:
    if value is default:
        return True
    if isinstance(value, torch.Tensor) or isinstance(default, torch.Tensor):
        return False
    return type(value) is type(default) and value == default


def _deprecated_kwarg(
    kwargs: dict[str, Any],
    old: str,
    new: str,
    value: object,
    op_name: str,
    *,
    default: object = None,
) -> Any:
    """Resolve keyword ``new`` given its deprecated alias ``old``.

    Pops ``old`` from ``kwargs``. If it was passed, warns (DeprecationWarning
    pointing at the caller) and returns its value; passing both names
    raises TypeError. Otherwise returns ``value`` (the new keyword's).
    """
    if old not in kwargs:
        return value
    old_value = kwargs.pop(old)
    if not _is_default(value, default):
        raise TypeError(
            f"{op_name}: got both '{new}' and its deprecated alias '{old}'; "
            f"pass '{new}' only."
        )
    warnings.warn(
        f"{op_name}: '{old}' is deprecated, use '{new}' instead; the old name "
        "will be removed in the next release.",
        DeprecationWarning,
        stacklevel=3,
    )
    return old_value


def _reject_unknown_kwargs(kwargs: dict[str, Any], op_name: str) -> None:
    """Raise TypeError for keywords left after the deprecated aliases were
    popped (what Python raises for an unknown keyword)."""
    if kwargs:
        name = next(iter(kwargs))
        raise TypeError(f"{op_name}() got an unexpected keyword argument '{name}'")


# ── shared reconstruction setup (ARCH P5) ──────────────────────────────────
#
# Every ``run_*`` starts with the same prologue: validate the sinogram and
# angles, resolve ``Nz``/``Nx``, validate and clone a warm start, resolve the
# support (cylinder flag or ``support_mask`` tensor, mutually exclusive) and
# build the per-iteration ``ConstraintConfig``. `_prepare_recon` is that
# prologue, written once. It depends only on torch and the constraint
# helpers, so it can move to a private module unchanged.


def _check_inputs(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor | None,
    Nz: int,
    *,
    allowed_dtypes: tuple[torch.dtype, ...],
    op_name: str,
    require_cuda: bool = False,
) -> tuple[int, int, int, int, torch.device]:
    """Validate and resolve sinogram/angles inputs for reconstruction algorithms.

    Returns (n_angles, Ny, N_det, Nz_resolved, device). Never coerces.

    Raises:
        RuntimeError  if require_cuda and sinogram is not on CUDA.
        TypeError     if sinogram is not a tensor or its dtype is not in
                      allowed_dtypes.
        ValueError    if sinogram is not 3-D, angles_deg is not 1-D,
                      or sinogram.shape[0] != len(angles_deg).
    """
    if not isinstance(sinogram, torch.Tensor):
        raise TypeError(
            f"{op_name}: sinogram must be a torch.Tensor, got {type(sinogram).__name__}"
        )
    if require_cuda and not sinogram.is_cuda:
        raise RuntimeError(
            f"{op_name} requires a CUDA tensor; got {sinogram.device}"
        )

    if sinogram.dtype not in allowed_dtypes:
        names = " or ".join(str(d) for d in allowed_dtypes)
        raise TypeError(
            f"{op_name} requires {names} sinogram (got {sinogram.dtype}). "
            "Cast with .float() or .half() first."
        )

    if sinogram.ndim != 3:
        raise ValueError(
            f"{op_name}: sinogram must be 3-D [n_angles, Ny, N_det], "
            f"got shape {tuple(sinogram.shape)}"
        )

    n_angles, Ny, N_det = sinogram.shape

    if angles_deg is not None:
        if not isinstance(angles_deg, torch.Tensor):
            raise TypeError(
                f"{op_name}: angles_deg must be a torch.Tensor, "
                f"got {type(angles_deg).__name__}"
            )
        if angles_deg.ndim != 1:
            raise ValueError(
                f"{op_name}: angles_deg must be 1-D [n_angles], "
                f"got shape {tuple(angles_deg.shape)}"
            )
        if len(angles_deg) != n_angles:
            raise ValueError(
                f"{op_name}: sinogram has {n_angles} projections but "
                f"angles_deg has {len(angles_deg)} entries."
            )

    Nz_resolved = N_det if Nz < 0 else Nz
    return n_angles, Ny, N_det, Nz_resolved, sinogram.device


def _as_int(value: object, name: str, op_name: str, *, minimum: int | None = 0) -> int:
    """``value`` as a Python int. Accepts ints, numpy integers and integer
    0-d tensors (anything with ``__index__``); rejects bools, floats and
    values below ``minimum`` (TypeError / ValueError)."""
    if isinstance(value, bool):
        raise TypeError(f"{op_name}: {name} must be an integer, got {value!r}")
    try:
        out = operator.index(value)
    except TypeError:
        raise TypeError(
            f"{op_name}: {name} must be an integer, got {type(value).__name__} {value!r}"
        ) from None
    if minimum is not None and out < minimum:
        raise ValueError(f"{op_name}: {name} must be >= {minimum}, got {out}")
    return out


def _as_float(value: object, name: str, op_name: str, *, minimum: float | None = None) -> float:
    """``value`` as a finite Python float. Accepts Python and numpy reals and
    one-element real tensors; rejects bools, non-finite values and values
    below ``minimum``."""
    if isinstance(value, torch.Tensor) and value.numel() == 1 and not value.is_complex():
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{op_name}: {name} must be a real number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{op_name}: {name} must be finite, got {value!r}")
    if minimum is not None and out < minimum:
        raise ValueError(f"{op_name}: {name} must be >= {minimum}, got {value!r}")
    return out


@dataclass
class _ReconSetup:
    """What `_prepare_recon` resolved for one reconstruction call."""
    n_angles: int
    Ny: int
    N_det: int
    Nz: int
    Nx: int
    device: torch.device
    compute_dtype: torch.dtype
    volume0: torch.Tensor | None    # [Ny, Nz, Nx]: cloned warm start, zeros, or None
    constraints: ConstraintConfig   # support resolved to a tensor (or None)
    n_iter: int | None


def _prepare_recon(
    op_name: str,
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor | None,
    *,
    Nz: int = -1,
    n_iter: int | None = None,
    allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32,),
    require_cuda: bool = False,
    warm_start: torch.Tensor | None = None,
    init_volume: bool = True,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    sparsity: float = 0.0,
) -> _ReconSetup:
    """Validate the inputs of a reconstruction and resolve its setup.

    * sinogram / angles: `_check_inputs` (dtype in ``allowed_dtypes``, 3-D
      sinogram, 1-D angles of matching length, CUDA if ``require_cuda``).
    * ``n_iter`` and ``Nz``: integers (numpy and 0-d tensor integers
      accepted), ``n_iter >= 0``; ``Nz < 0`` means ``N_det``. ``Nx = N_det``.
    * ``warm_start``: a tensor of shape ``[Ny, Nz, Nx]`` and the sinogram's
      dtype, moved to the sinogram's device and cloned, so the caller's
      tensor is never modified. Without it ``volume0`` is zeros in the
      sinogram dtype (or None when ``init_volume=False``).
    * support: ``mask=True`` (inscribed cylinder) and ``support_mask`` are
      mutually exclusive (ValueError). ``support_mask`` is validated and
      broadcast by `_resolve_support_mask`; either becomes a tensor in the
      compute dtype on the compute device.
    * ``ConstraintConfig(positivity, support_mask=<resolved support>,
      soft_threshold=sparsity)``.
    """
    if support_mask is not None and mask:
        raise ValueError(
            f"{op_name}: mask=True and support_mask are mutually exclusive — "
            "pass one support per reconstruction."
        )
    Nz = _as_int(Nz, "Nz", op_name, minimum=None)
    if n_iter is not None:
        n_iter = _as_int(n_iter, "n_iter", op_name)
    sparsity = _as_float(sparsity, "sparsity", op_name, minimum=0.0)
    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=allowed_dtypes, op_name=op_name, require_cuda=require_cuda,
    )
    Nx = N_det
    compute_dtype = sinogram.dtype

    volume0: torch.Tensor | None = None
    if warm_start is not None:
        if not isinstance(warm_start, torch.Tensor):
            raise TypeError(
                f"{op_name}: warm_start must be a torch.Tensor, "
                f"got {type(warm_start).__name__}"
            )
        _assert_warm_start_shape(warm_start, (Ny, Nz, Nx), op_name)
        _assert_dtype_match(sinogram, warm_start, "sinogram", "warm_start", op_name)
        volume0 = warm_start.detach().to(device=device).contiguous().clone()
    elif init_volume:
        volume0 = torch.zeros(Ny, Nz, Nx, dtype=compute_dtype, device=device)

    if support_mask is not None:
        support = _resolve_support_mask(
            support_mask, Ny, Nz, Nx, device, compute_dtype, op_name,
        )
    else:
        support = _maybe_cylindrical_mask(Nz, Nx, device, bool(mask), dtype=compute_dtype)
    constraints = ConstraintConfig(positivity=positivity, support_mask=support,
                                   soft_threshold=sparsity)
    return _ReconSetup(
        n_angles=n_angles, Ny=Ny, N_det=N_det, Nz=Nz, Nx=Nx, device=device,
        compute_dtype=compute_dtype, volume0=volume0, constraints=constraints,
        n_iter=n_iter,
    )


# ── loss-callback helpers (live loss streaming) ────────────────────────────
#
# Contract (app handoff, 2026-06-11): iterative algorithms accept an optional
# ``loss_callback`` kwarg invoked as ``loss_callback(iter, loss)`` with
# 0-based iteration index and a plain CPU float (NaN/Inf passed through).
# Callbacks that declare a ``components`` keyword additionally receive
# ``components={...}`` — per-term CPU floats whose sum ≈ the total loss.
# The kwarg must be a named parameter (introspectable via inspect.signature),
# the callback is display-only (``return_loss`` is unchanged), and a raising
# callback must never abort the reconstruction.


def _callback_accepts_components(loss_callback: Callable) -> bool:
    """True if ``loss_callback`` can receive the ``components=`` kwarg.

    Existing consumers pass plain 2-positional callables; probe the signature
    once so those keep working without the components extension.
    """
    try:
        sig = inspect.signature(loss_callback)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "components" and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _emit_loss(
    loss_callback: Callable,
    send_components: bool,
    iteration: int,
    loss: float,
    components: dict[str, float] | None,
) -> None:
    """Invoke ``loss_callback``, swallowing (but logging) any exception.

    One failed call must not stop subsequent invocations — the caller keeps
    calling every iteration regardless.
    """
    try:
        if send_components:
            loss_callback(iteration, loss, components=components)
        else:
            loss_callback(iteration, loss)
    except Exception:
        _log.warning(
            "loss_callback raised at iteration %d; continuing reconstruction",
            iteration,
            exc_info=True,
        )


def _loss_float(value: object) -> float:
    """A loss value as a plain CPU float (``.item()`` on a tensor)."""
    return value.item() if isinstance(value, torch.Tensor) else float(value)


class _LossRecorder:
    """Per-iteration loss bookkeeping shared by every iterative algorithm
    (ARCH P6). The 2026-06 ``loss_callback`` contract lives here only.

    ``wants_loss`` is True when the loss is needed at all (``return_loss``
    or a callback); algorithms compute the loss, and pay the ``.item()``
    host sync, only then. ``record(it, loss, components)`` stores the value
    for ``loss_history`` and invokes the callback as ``loss_callback(it,
    loss)``, adding ``components=`` for callbacks that declare it (signature
    probed once). Values are converted to plain floats; NaN/Inf pass
    through; a raising callback is logged and still called on the next
    iteration. ``components`` defaults to ``{"fidelity": loss}``, for
    algorithms whose recorded loss is the data term alone.

    ``progress=True`` shows a tqdm bar of ``n_iter`` steps, advanced by
    `iterate` (or `tick`). ``finish(volume)`` closes it and returns
    ``volume`` or ``(volume, loss_history)`` with ``loss_history`` a float32
    CPU tensor.
    """

    def __init__(
        self,
        return_loss: bool,
        loss_callback: Callable | None,
        progress: bool,
        n_iter: int,
        *,
        desc: str | None = None,
    ) -> None:
        if loss_callback is not None and not callable(loss_callback):
            raise TypeError(
                f"loss_callback must be callable, got {type(loss_callback).__name__}"
            )
        self.return_loss = bool(return_loss)
        self.loss_callback = loss_callback
        self.send_components = (
            loss_callback is not None and _callback_accepts_components(loss_callback)
        )
        self.wants_loss = self.return_loss or loss_callback is not None
        self.losses: list[float] = []
        self._bar = tqdm(total=n_iter, desc=desc) if progress else None

    def iterate(self, n: int):
        """``range(n)``, advancing the progress bar after each step."""
        for it in range(n):
            yield it
            self.tick()

    def tick(self) -> None:
        """Advance the progress bar by one iteration (no-op without one)."""
        if self._bar is not None:
            self._bar.update(1)

    def record(
        self,
        it: int,
        loss: float | torch.Tensor,
        components: dict[str, float | torch.Tensor] | None = None,
    ) -> None:
        """Record iteration ``it``'s total loss (and its components)."""
        if not self.wants_loss:
            return
        loss_val = _loss_float(loss)
        if self.return_loss:
            self.losses.append(loss_val)
        if self.loss_callback is not None:
            comps = None
            if self.send_components:
                comps = (
                    {"fidelity": loss_val} if components is None
                    else {k: _loss_float(v) for k, v in components.items()}
                )
            _emit_loss(self.loss_callback, self.send_components, it, loss_val, comps)

    def history(self) -> torch.Tensor:
        """The recorded losses as a float32 CPU tensor."""
        return torch.tensor(self.losses, dtype=torch.float32)

    def close(self) -> None:
        """Close the progress bar (idempotent)."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def finish(self, volume: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Close the bar; return ``volume`` or ``(volume, loss_history)``."""
        self.close()
        return _finish_iters(volume, self.losses, self.return_loss)


def _ad_sparsity_term(volume: torch.Tensor, weight: float) -> torch.Tensor:
    """L1 sparsity penalty for ``run_ad_recon``: ``weight * mean(|volume|)``.

    Must use the magnitude, as documented on ``RegularizerConfig.sparsity``.
    A signed ``mean(volume)`` is a linear term whose gradient is the same
    constant for every voxel, so without positivity it drags the whole
    volume negative instead of promoting sparsity.
    """
    return weight * volume.abs().mean()


# ── exact weighting filter (Wolf et al. 2014) ─────────────────────────────
#
# W-BP = back-project, then deconvolve every XZ slice by the transfer
# function of the back-projection of a centred point (a 1 at detector pixel
# (N_det-1)//2 in every view; Wolf et al. 2014). Two details matter for this
# to be (close to) an inverse of the projector, and both used to be wrong:
#
# * The deconvolution is an FFT, i.e. circular, and on the bare [Nz, Nx]
#   grid wSIRT diverged at omega=1 on 32 of 45 test geometries. Two modes
#   grew: one concentrated in the outermost ~3 voxels of the volume (the
#   circular boundary sits on the volume edge, where the tight-bounds
#   forward projector barely sees the voxels), and, with many views, a
#   low-frequency one next to the missing-wedge edge. The layergram and the
#   PSF are now back-projected onto a grid with a margin of
#   `_wbp_margin(N)` = max(2, N // 16) voxels on each side of Z and X,
#   filtered there and cropped at the margin (symmetric, so exact for odd N).
#   Measured over 9 sizes x 5 tilt schemes: no divergence, and the one-step
#   error is lower than without the margin on every geometry; larger margins
#   (N/8, N/4, 2x) lose accuracy.
# * The filter assumes a shift-invariant back-projection. The CPU
#   `back_project` is ray-driven (the exact transpose of the ray-marching CPU
#   forward projector) and is not: a constant sinogram back-projects to a
#   +-34 % voxel pattern at 45 degrees, which the filter amplifies up to
#   ~40x (CPU W-BP: 83 % error vs 30 % on CUDA). W-BP therefore uses the
#   voxel-driven gather back-projection on both backends: the CUDA
#   `back_project` kernel already is one, and `_bp_gather` is the same
#   operator on CPU with exact fp32 interpolation weights. The CUDA texture
#   unit interpolates with 8-bit fractional weights, so the two differ by
#   ~2e-4 (the same texture floor the forward projector has at ~1.3e-4);
#   the CPU path is the exact reference.

_WBP_CHUNK_VOXELS = 1 << 24     # padded layergram elements per Y-chunk (64 MiB fp32)


def _wbp_margin(n: int) -> int:
    """Voxels of padding on each side of a W-BP axis of length ``n``."""
    return max(2, n // 16)


def _bp_gather(
    sinogram: torch.Tensor,     # [n_angles, Ny, N_det] fp32
    angles_deg: torch.Tensor,   # [n_angles] degrees
    Nz: int,
    Nx: int,
) -> torch.Tensor:
    """Voxel-driven (gather) single-axis back-projection -> [Ny, Nz, Nx].

    Each voxel (x, z), centred coordinates, takes from every view the
    sinogram row value at detector position u = x cos + z sin + (N_det-1)/2,
    linearly interpolated, with zero outside the detector. This is the
    operator of the CUDA ``back_project_parallel_kernel`` (texture fetch at
    u + 0.5 with linear filtering and border addressing), in plain PyTorch
    so the CPU W-BP is the same operator, with exact interpolation weights
    (the texture unit's are 8-bit, ~2e-4 apart)."""
    n_angles, Ny, N_det = sinogram.shape
    device = sinogram.device
    a = angles_deg.detach().to(device=device, dtype=torch.float32).reshape(-1) * (math.pi / 180.0)
    cos_a, sin_a = torch.cos(a), torch.sin(a)
    x = torch.arange(Nx, device=device, dtype=torch.float32) - 0.5 * (Nx - 1)
    z = torch.arange(Nz, device=device, dtype=torch.float32) - 0.5 * (Nz - 1)
    # [N_det + 2, Ny]: one zero row on each side stands for "outside".
    out = torch.zeros(Nz * Nx, Ny, device=device, dtype=torch.float32)
    for k in range(n_angles):
        rows = F.pad(sinogram[k].float().t(), (0, 0, 1, 1))
        u = (x[None, :] * cos_a[k] + z[:, None] * sin_a[k] + 0.5 * (N_det - 1)).reshape(-1)
        u0 = torch.floor(u)
        w = (u - u0).unsqueeze(1)
        i0 = (u0.long() + 1).clamp_(0, N_det + 1)       # +1: the zero row
        i1 = (u0.long() + 2).clamp_(0, N_det + 1)
        out += (1.0 - w) * rows[i0] + w * rows[i1]
    return out.t().reshape(Ny, Nz, Nx).contiguous()


def _wbp_backproject(sinogram: torch.Tensor, angles_deg: torch.Tensor,
                     Nz: int, Nx: int) -> torch.Tensor:
    """The shift-invariant back-projection W-BP needs: the CUDA kernel, or
    the same gather operator on CPU."""
    if sinogram.is_cuda:
        return back_project(sinogram, angles_deg, Nz=Nz, Nx=Nx)
    return _bp_gather(sinogram, angles_deg, Nz, Nx)


def _compute_exact_wf(
    angles_deg: torch.Tensor,   # [n_angles] degrees
    Nz: int, Nx: int,
    N_det: int,
    device: torch.device,
    threshold_scale: float = 1.0,
) -> torch.Tensor:
    """
    Numerically compute the exact weighting filter for W-BP on the padded
    grid ``[Pz, Px] = [Nz, Nx] + 2 * margin`` (`_wbp_margin`).

    PSF: back-project the sinogram of a unit point at the rotation centre:
    a 1 at detector pixel ``(N_det - 1) // 2`` in every view. For even N_det
    that pixel is half a pixel off the centre; the filter uses |TF| only, and
    splitting the point 0.5/0.5 over the two central pixels instead adds a
    blur the data do not have (measured: worse error on every even size and
    divergence without the margin). WF = 1 / |TF(PSF)|, zero where |TF| is
    below ``threshold_scale * TF[0, 0] / n_angles`` (missing wedge).

    Returns real Fourier amplitudes ``[Pz, Px//2+1]`` to multiply against
    ``rfft2`` of a padded layergram slice.
    """
    n_angles = len(angles_deg)
    point_sino = torch.zeros(n_angles, 1, N_det, device=device)
    point_sino[:, 0, (N_det - 1) // 2] = 1.0
    Pz, Px = Nz + 2 * _wbp_margin(Nz), Nx + 2 * _wbp_margin(Nx)
    # The exact gather on both backends: the filter is a hard threshold on
    # |TF|, so a frequency sitting near it would be kept on one backend and
    # zeroed on the other if the PSF carried the CUDA texture's 8-bit
    # weights (up to 5e-2 CPU/CUDA difference measured). One Ny=1 slice,
    # built once per geometry.
    psf = _bp_gather(point_sino, angles_deg.to(device), Pz, Px)[0]   # [Pz, Px]

    TF_amp = torch.fft.rfft2(psf).abs()
    # threshold: avoid division by zero inside missing wedge
    thresh = max(threshold_scale * TF_amp[0, 0].item() / float(n_angles), 1e-6)
    return torch.where(TF_amp > thresh, 1.0 / TF_amp, torch.zeros_like(TF_amp))


def _make_wbp(
    angles_deg: torch.Tensor,
    Nz: int, Nx: int, N_det: int,
    device: torch.device,
    threshold_scale: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``wbp(sinogram) -> [Ny, Nz, Nx]``: the exact weighted
    back-projection shared by `run_weighted_backprojection` and `run_wsirt`
    (the filter is built once per geometry)."""
    WF = _compute_exact_wf(angles_deg, Nz, Nx, N_det, device, threshold_scale)
    oz, ox = _wbp_margin(Nz), _wbp_margin(Nx)
    Pz, Px = Nz + 2 * oz, Nx + 2 * ox       # symmetric margin: the crop is centred for odd N too
    angles_deg = angles_deg.to(device)

    # Y-slices are independent; chunk them so the padded layergram and its
    # spectrum stay around _WBP_CHUNK_VOXELS elements.
    y_chunk = max(1, _WBP_CHUNK_VOXELS // (Pz * Px))

    @torch.no_grad()
    def wbp(sinogram: torch.Tensor) -> torch.Tensor:
        Ny = sinogram.shape[1]
        out = torch.empty(Ny, Nz, Nx, dtype=torch.float32, device=sinogram.device)
        for y0 in range(0, Ny, y_chunk):
            sl = sinogram[:, y0:y0 + y_chunk]
            spec = torch.fft.rfft2(_wbp_backproject(sl, angles_deg, Pz, Px))   # [ny, Pz, Px//2+1]
            spec.mul_(WF)
            filtered = torch.fft.irfft2(spec, s=(Pz, Px))
            del spec
            out[y0:y0 + sl.shape[1]] = filtered[:, oz:oz + Nz, ox:ox + Nx]
            del filtered
        return out

    return wbp


# ── public reconstruction functions ───────────────────────────────────────

def run_fbp(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    *,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    projector: object | None = None,
    num_threads: int | None = None,
) -> torch.Tensor:
    """
    Filtered Back-Projection.

    Applies ramp filter to each sinogram row then back-projects.
    Returns volume [Ny, Nz, Nx].
    mask=True zeros voxels outside the inscribed cylinder (common in ET).

    support_mask: optional precomputed support tensor applied once to the
    final volume (hard constraint: ``volume * support_mask``). Shape
    ``[Nz, Nx]``, ``[1, Nz, Nx]``, or ``[Ny, Nz, Nx]``; binary {0, 1}
    expected (soft [0, 1] values act as a smooth weighting); cast to the
    volume dtype/device internally. Mutually exclusive with ``mask=True``
    — one support per reconstruction.

    On CUDA a :class:`ParallelProjector` is built internally if ``projector``
    is None — the BP runs through a persistent cudaArray + texture handle that
    is torn down when the function returns. Pass an explicit projector to
    amortize the texture-cache allocation across many calls (e.g. a
    tilt-series sweep). Geometry must match: the projector's ``angles_deg``
    replaces the ``angles_deg`` argument for BP. Numerics are bit-identical
    to the functional path. On CPU the functional ``back_project`` is used
    regardless (cached projectors are CUDA-only).

    Input dtype is preserved end-to-end (fp32 / fp16). The ramp filter
    internally runs in fp32 (cuFFT requires power-of-two N_det for fp16 —
    we sidestep that by upcasting around the FFT only) and the BP runs in
    the input dtype. fp16 FBP gets the BP speedup; quality is comparable
    to fp32 on visualization-grade reconstructions but visibly noisier
    on edges — use fp32 for quantitative work. bf16 raises.
    """
    if support_mask is not None and mask:
        raise ValueError(
            "run_fbp: mask=True and support_mask are mutually exclusive — "
            "pass one support per reconstruction."
        )
    # Validation only: the support is resolved below against the actual
    # output shape (a cached projector may reconstruct with Nz != N_det).
    setup = _prepare_recon(
        "run_fbp", sinogram, angles_deg, Nz=Nz,
        allowed_dtypes=(torch.float32, torch.float16), init_volume=False,
    )
    n_angles, Ny, N_det, Nz, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.device)
    filtered = ramp_filter(sinogram)
    if projector is not None:
        volume = projector.back(filtered)
    elif sinogram.is_cuda:
        _proj = ParallelProjector(
            (Ny, Nz, N_det), angles_deg,
            device=device, N_det=N_det,
        )
        volume = _proj.back(filtered)
    else:
        volume = back_project(filtered, angles_deg, Nz=Nz, num_threads=num_threads)
    # angular sampling normalisation: π / n_angles
    volume = volume * (math.pi / n_angles)
    if support_mask is not None:
        # Validate against the actual output shape — a cached projector may
        # reconstruct with Nz != N_det.
        _mask = _resolve_support_mask(
            support_mask,
            volume.shape[0], volume.shape[1], volume.shape[2],
            volume.device, volume.dtype, "run_fbp",
        )
    else:
        _mask = _maybe_cylindrical_mask(
            volume.shape[1],
            volume.shape[2],
            volume.device,
            mask,
            dtype=volume.dtype,
        )
    return ConstraintConfig(positivity=positivity, support_mask=_mask).apply(volume)


def run_weighted_backprojection(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    *,
    Nz: int = -1,
    positivity: bool = False,
    threshold_scale: float = 1.0,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Exact Weighted Back-Projection (Wolf et al. 2014).

    Computes the exact numerical weighting filter from the PSF of a
    central point, then applies it slice-by-slice in Fourier space. The
    back-projection and the filter run on a grid with a max(2, N // 16)
    voxel margin on each side of Z and X (the circular deconvolution
    otherwise destabilises the volume edge) and use the voxel-driven
    back-projection on both backends, so CPU and CUDA give the same result
    up to the CUDA texture interpolation; see the "exact weighting filter"
    notes above `_bp_gather`. Extra memory: one padded layergram and its
    spectrum per Y-chunk (at most ~64 Mi elements).

    Returns volume [Ny, Nz, Nx].
    mask=True zeros voxels outside the inscribed cylinder (common in ET).
    """
    setup = _prepare_recon(
        "run_weighted_backprojection", sinogram, angles_deg, Nz=Nz,
        init_volume=False, positivity=positivity, mask=mask, support_mask=support_mask,
    )
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)

    volume = _make_wbp(angles_deg, Nz, Nx, N_det, device, threshold_scale)(sinogram)

    return setup.constraints.apply(volume)


def run_sirt(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 50,
    *,
    omega: float = 1.0,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,   # precomputed support, enforced each iteration
    sparsity: float = 0.0,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — use 3D projector if set
    projector: object | None = None,            # ParallelProjector / RotmatProjector for cached FP/BP
    warm_start: torch.Tensor | None = None,     # [Ny, Nz, Nx]; its dtype must match the sinogram
    num_threads: int | None = None,             # CPU paths
    return_loss: bool = False,
    loss_callback: Callable | None = None,      # per-iteration loss streaming hook
    progress: bool = False,
    **deprecated: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Simultaneous Iterative Reconstruction Technique.

    x^{k+1} = x^k + ω · C · A^T · R · (b - A·x^k)

    C = 1 / A^T(ones)   (voxel normalisation)
    R = 1 / A(ones)     (ray normalisation)

    euler_zyz_deg: if provided, uses the full 3D rotation-matrix projector
    (forward_project_3d / back_project_3d) instead of the scalar-angle
    single-axis projector. ``num_threads`` is forwarded to the CPU 3D
    kernels (kwarg → OMP_NUM_THREADS → 1).

    projector: on CUDA a cached projector is always used — built internally
    if not supplied. Pass an explicit :class:`ParallelProjector` (or
    :class:`RotmatProjector` when using ``euler_zyz_deg``) to reuse a
    persistent cudaArray + texture handle across multiple ``run_sirt`` calls;
    otherwise the internal projector is torn down on return. Must be sized
    to the volume / sinogram shape; angles passed via the projector replace
    ``angles_deg`` / ``euler_zyz_deg`` for FP/BP. CPU SIRT uses the
    functional API (cached projectors are CUDA-only).

    return_loss: if True, returns (volume, loss_history) where loss_history
    is a float32 CPU tensor of shape [n_iter] containing the sinogram MSE
    ||b - A·x^k||² / numel at each iteration.

    sparsity: L1 soft-threshold applied after each update (proximal operator
    of sparsity·‖x‖₁). 0.0 disables it.

    volume: optional warm-start [Ny, Nz, Nx]. When provided, ``volume.dtype``
    sets the compute dtype (and must match the sinogram). When None, the
    algorithm zero-initialises in ``sinogram.dtype`` (fp32 or fp16). All
    internal tensors (normalisers, residuals, masks) inherit the compute
    dtype so the projector locks consistently. Output preserves the compute
    dtype — fp16 in, fp16 out.

    support_mask: optional precomputed support tensor enforced as a hard
    constraint each iteration (``volume * support_mask`` after every
    update, like ``mask=True`` but with an arbitrary support). Shape
    ``[Nz, Nx]``, ``[1, Nz, Nx]``, or ``[Ny, Nz, Nx]``; binary {0, 1}
    expected (soft [0, 1] values act as a smooth weighting); cast to the
    compute dtype/device internally. Mutually exclusive with ``mask=True``
    — one support per reconstruction.
    loss_callback: optional hook invoked once per iteration as
    ``loss_callback(i, loss)`` — 0-based index and the same plain CPU float
    that lands in ``loss_history[i]`` (NaN/Inf passed through). Callbacks
    whose signature accepts a ``components`` keyword also receive
    ``components={"fidelity": loss}`` (SIRT's loss is pure data fidelity;
    ``sparsity`` acts as a proximal constraint, not a loss term). Exceptions
    raised by the callback are logged and swallowed. Independent of
    ``return_loss``; supplying a callback forces the eager loop (no CUDA
    Graph capture).

    Returns volume [Ny, Nz, Nx].
    mask=True enforces cylindrical support at each iteration (GENFIRE-style).
    """
    warm_start = _deprecated_kwarg(deprecated, "volume", "warm_start", warm_start, "run_sirt")
    _reject_unknown_kwargs(deprecated, "run_sirt")
    setup = _prepare_recon(
        "run_sirt", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        allowed_dtypes=(torch.float32, torch.float16), warm_start=warm_start,
        positivity=positivity, mask=mask, support_mask=support_mask, sparsity=sparsity,
    )
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)
    volume, constraints, n_iter = setup.volume0, setup.constraints, setup.n_iter

    _fp, _bp, _internal_proj = _build_fp_bp(
        sinogram, angles_deg, (Ny, Nz, Nx), N_det, n_angles,
        device, num_threads, projector, euler_zyz_deg,
    )

    # --- normalisation matrices (computed once with the chosen projector) ---
    ones_sino = torch.ones_like(sinogram)
    C_vol     = _bp(ones_sino)
    _assert_projector_output_shape(C_vol, (Ny, Nz, Nx), "back-projection", "run_sirt")
    C_vol     = torch.where(C_vol > 0, 1.0 / C_vol, torch.zeros_like(C_vol))

    ones_vol  = torch.ones_like(volume)
    R_sino    = _fp(ones_vol)
    _assert_projector_output_shape(R_sino, (n_angles, Ny, N_det), "forward-projection", "run_sirt")
    R_sino    = torch.where(R_sino > 0, 1.0 / R_sino, torch.zeros_like(R_sino))

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_sirt")

    # --- CUDA Graph capture path ----------------------------------------
    # Gate: CUDA input, no per-iteration loss tracking (which calls .item()).
    # Rotmat mode (euler_zyz_deg) is kept on eager path because the per-call
    # matrix tensor can vary, while cached graphs assume fixed captured inputs.
    # If capture fails, fall back to the eager loop below.
    _active_proj = projector if projector is not None else _internal_proj
    if sinogram.is_cuda and not rec.wants_loss and n_iter > 1 and euler_zyz_deg is None:
        graph_result = _run_sirt_cuda_graph(
            volume=volume,
            sinogram=sinogram,
            n_iter=n_iter,
            omega=omega,
            C_vol=C_vol,
            R_sino=R_sino,
            _fp=_fp,
            _bp=_bp,
            constraints=constraints,
            projector=_active_proj,
            key=(Ny, Nz, Nx, n_angles, N_det,
                 sinogram.dtype,
                 float(omega),
                 constraints.positivity,
                 float(constraints.soft_threshold),
                 tuple(constraints.support_mask.shape)
                 if constraints.support_mask is not None else None,
                 euler_zyz_deg is not None),
            on_replay=rec.tick,
        )
        if graph_result is not None:
            return rec.finish(graph_result)

    # --- eager iteration loop (fallback / CPU / loss-tracking path) ------
    # The loss is the data MSE before the update (SIRT's loss is pure
    # fidelity; sparsity acts as a proximal constraint, not a loss term).
    for it in rec.iterate(n_iter):
        raw_res  = sinogram - _fp(volume)
        if rec.wants_loss:
            rec.record(it, raw_res.pow(2).mean())
        residual = raw_res * R_sino
        volume   = volume + omega * C_vol * _bp(residual)
        volume   = constraints.apply(volume)

    return rec.finish(volume)


# ── FISTA helpers (Beck & Teboulle 2009) ──────────────────────────────────

def _estimate_lipschitz(
    _fp,
    _bp,
    vol_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    n_iter: int = 10,
) -> float:
    """Spectral norm ||A^T A||_op = L (= Lipschitz constant of ∇(½||Ax-b||²)).

    Power iteration. Returns a Python float with a 1.01 safety margin
    (``_LIPSCHITZ_SAFETY``) so the
    FISTA step size 1/L sits strictly inside the contraction region even if
    the iteration hasn't fully converged.
    """
    v = torch.randn(vol_shape, device=device, dtype=dtype)
    v = v / (v.norm() + _NORM_EPS)
    L = 0.0
    for _ in range(n_iter):
        AtAv = _bp(_fp(v))
        L = float(AtAv.norm())
        v = AtAv / (L + _NORM_EPS)
    return L * _LIPSCHITZ_SAFETY


def _tv_grad(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward differences along (Ny, Nz, Nx) — discrete L = grad operator."""
    return (
        v[1:, :, :] - v[:-1, :, :],
        v[:, 1:, :] - v[:, :-1, :],
        v[:, :, 1:] - v[:, :, :-1],
    )


def _tv_div(
    p_y: torch.Tensor, p_z: torch.Tensor, p_x: torch.Tensor,
    Ny: int, Nz: int, Nx: int,
) -> torch.Tensor:
    """Positive adjoint of _tv_grad: ⟨grad(v), p⟩ = ⟨v, _tv_div(p)⟩.

    Used as ``v* = z − _tv_div(p, ...)`` for the unconstrained TV-l2 prox
    primal recovery. Zero-padded at the boundary (Neumann).
    """
    out = torch.zeros(Ny, Nz, Nx, dtype=p_y.dtype, device=p_y.device)
    if p_y.numel() > 0:
        out[1:, :, :] += p_y
        out[:-1, :, :] -= p_y
    if p_z.numel() > 0:
        out[:, 1:, :] += p_z
        out[:, :-1, :] -= p_z
    if p_x.numel() > 0:
        out[:, :, 1:] += p_x
        out[:, :, :-1] -= p_x
    return out


def _project_dual_ball(
    q_y: torch.Tensor, q_z: torch.Tensor, q_x: torch.Tensor, lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Isotropic per-voxel projection onto the ℓ₂ ball of radius ``lam``.

    The three q tensors have mismatched shapes (each one short by 1 along its
    own diff axis); zero-pad each to the full grid before computing the
    per-voxel norm, divide, then slice back.
    """
    Ny = q_z.shape[0]
    Nz = q_y.shape[1]
    Nx = q_y.shape[2]
    full_y = F.pad(q_y, (0, 0, 0, 0, 0, 1))  # pad +1 at end of dim -3 (Ny)
    full_z = F.pad(q_z, (0, 0, 0, 1, 0, 0))  # pad +1 at end of dim -2 (Nz)
    full_x = F.pad(q_x, (0, 1, 0, 0, 0, 0))  # pad +1 at end of dim -1 (Nx)
    norm = (full_y * full_y + full_z * full_z + full_x * full_x).sqrt()
    denom = (norm / lam).clamp(min=1.0)
    full_y = full_y / denom
    full_z = full_z / denom
    full_x = full_x / denom
    return full_y[:-1, :, :], full_z[:, :-1, :], full_x[:, :, :-1]


def _tv_prox_3d(
    z: torch.Tensor,
    lam: float,
    positivity: bool,
    mask: torch.Tensor | None,
    n_iter: int = 20,
    mode: str = "fgp",
) -> torch.Tensor:
    """3-D isotropic TV proximal operator with optional box constraint.

    Solves:

        argmin_v  (1/2) ‖v − z‖²  +  lam · TV(v)  +  ι_C(v)

    where ``C = {v ≥ 0 if positivity} ∩ {v · mask = v}``. TV is the discrete
    isotropic total variation built from forward differences.

    ``mode="fgp"`` uses Beck-Teboulle 2009b Fast Gradient Projection
    (Nesterov-accelerated dual projection, O(1/k²) on the dual). ``mode=
    "chambolle"`` uses Chambolle 2004's first-order dual projection
    (O(1/k)). Dual step size 1/12 — the upper bound on ||grad ∘ div||_op in
    3-D (8 in 2-D, 12 in 3-D).
    """
    if mode not in ("fgp", "chambolle"):
        raise ValueError(
            f"_tv_prox_3d: unknown mode {mode!r}; expected 'fgp' or 'chambolle'."
        )
    Ny, Nz_, Nx = z.shape
    device, dtype = z.device, z.dtype

    p_y = torch.zeros(max(Ny - 1, 0), Nz_, Nx, device=device, dtype=dtype)
    p_z = torch.zeros(Ny, max(Nz_ - 1, 0), Nx, device=device, dtype=dtype)
    p_x = torch.zeros(Ny, Nz_, max(Nx - 1, 0), device=device, dtype=dtype)
    p_prev_y, p_prev_z, p_prev_x = p_y, p_z, p_x
    r_y, r_z, r_x = p_y, p_z, p_x
    t = 1.0
    step = _TV_DUAL_STEP

    for _ in range(n_iter):
        v = z - _tv_div(r_y, r_z, r_x, Ny, Nz_, Nx)
        if positivity:
            v = v.clamp(min=0.0)
        if mask is not None:
            v = v * mask

        gy, gz, gx = _tv_grad(v)
        q_y = r_y + step * gy
        q_z = r_z + step * gz
        q_x = r_x + step * gx
        p_y_new, p_z_new, p_x_new = _project_dual_ball(q_y, q_z, q_x, lam)

        if mode == "fgp":
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new
            r_y = p_y_new + beta * (p_y_new - p_prev_y)
            r_z = p_z_new + beta * (p_z_new - p_prev_z)
            r_x = p_x_new + beta * (p_x_new - p_prev_x)
            t = t_new
        else:  # chambolle
            r_y, r_z, r_x = p_y_new, p_z_new, p_x_new

        p_prev_y, p_prev_z, p_prev_x = p_y_new, p_z_new, p_x_new

    v = z - _tv_div(p_prev_y, p_prev_z, p_prev_x, Ny, Nz_, Nx)
    if positivity:
        v = v.clamp(min=0.0)
    if mask is not None:
        v = v * mask
    return v


def _fista_loop(
    volume: torch.Tensor,
    sinogram: torch.Tensor,
    _fp,
    _bp,
    n_iter: int,
    L: float,
    prox_fn,
    rec: _LossRecorder,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Shared Beck-Teboulle FISTA outer loop.

    ``prox_fn(u)`` is called once per iteration on ``u = y_k - (1/L) ∇f(y_k)``
    and must return the next iterate ``x_{k+1}``. For pure FISTA it's just the
    constraint projection; for FISTA-TV it's the TV+constraint prox.

    The recorded loss is ``||b − A·y_k||²`` (data-fit MSE at the momentum
    point, which is free) at each iteration — a tight surrogate for the loss
    at ``x_k`` — and it is the data-fit term only (components
    ``{"fidelity": loss}``), regardless of prox regularisation.
    """
    inv_L = 1.0 / L
    t = 1.0
    x_prev = volume.clone()
    y = volume.clone()
    for it in rec.iterate(n_iter):
        fp_y = _fp(y)
        if rec.wants_loss:
            rec.record(it, (sinogram - fp_y).pow(2).mean())
        grad = _bp(fp_y - sinogram)
        x_new = prox_fn(y - inv_L * grad)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        beta = (t - 1.0) / t_new
        y = x_new + beta * (x_new - x_prev)
        x_prev = x_new
        t = t_new
    return rec.finish(x_prev)


def _build_fp_bp(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    vol_shape: tuple[int, int, int],
    N_det: int,
    n_angles: int,
    device: torch.device,
    num_threads: int | None,
    projector: object | None,
    euler_zyz_deg: torch.Tensor | None,
):
    """Build (_fp, _bp, _internal_proj) closures for iterative algorithms.

    Caller must hold ``_internal_proj`` until the algorithm finishes so the
    texture cache stays alive.
    """
    Ny, Nz, Nx = vol_shape
    _internal_proj = None
    if projector is not None:
        if euler_zyz_deg is not None:
            mats = euler_zyz_to_rotmat(euler_zyz_deg).to(device=device)
            _fp = lambda v: projector.forward(v, mats)
            _bp = lambda s: projector.back(s, mats)
        else:
            _fp = lambda v: projector.forward(v)
            _bp = lambda s: projector.back(s)
    elif sinogram.is_cuda:
        if euler_zyz_deg is not None:
            mats = euler_zyz_to_rotmat(euler_zyz_deg).to(device=device)
            _internal_proj = RotmatProjector(
                (Ny, Nz, Nx), n_angles=n_angles, device=device, N_det=N_det,
            )
            _fp = lambda v: _internal_proj.forward(v, mats)
            _bp = lambda s: _internal_proj.back(s, mats)
        else:
            _internal_proj = ParallelProjector(
                (Ny, Nz, Nx), angles_deg, device=device, N_det=N_det,
            )
            _fp = lambda v: _internal_proj.forward(v)
            _bp = lambda s: _internal_proj.back(s)
    elif euler_zyz_deg is not None:
        _fp = lambda v: forward_project_3d(v, euler_zyz_deg, N_det=N_det, num_threads=num_threads)
        _bp = lambda s: back_project_3d(s, euler_zyz_deg, Nz=Nz, num_threads=num_threads)
    else:
        _fp = lambda v: forward_project(v, angles_deg, N_det=N_det, num_threads=num_threads)
        _bp = lambda s: back_project(s, angles_deg, Nz=Nz, num_threads=num_threads)
    return _fp, _bp, _internal_proj


def run_fista(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 50,
    *,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    euler_zyz_deg: torch.Tensor | None = None,
    n_power_iter: int = 10,
    L: float | None = None,
    projector: object | None = None,
    warm_start: torch.Tensor | None = None,
    num_threads: int | None = None,
    return_loss: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
    **deprecated: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    FISTA (Beck & Teboulle 2009) on the smooth least-squares problem
    ``f(x) = ½ ‖A·x − b‖²``.

        y_k     = x_k + ((t_{k-1} - 1) / t_k) · (x_k − x_{k-1})  (momentum)
        x_{k+1} = P_C( y_k − (1/L) · A^T (A·y_k − b) )
        t_{k+1} = (1 + √(1 + 4·t_k²)) / 2

    where ``L = ||A^T A||_op`` is estimated by power iteration and ``P_C`` is
    the projection onto positivity and/or cylindrical mask. Achieves
    O(1/k²) convergence on the data-fit objective vs. SIRT's O(1/k).

    Args:
        sinogram:     [n_angles, Ny, N_det] — measured tilt series.
        angles_deg:   [n_angles] — single-axis tilt angles.
        n_iter:       outer FISTA iterations.
        Nz:           reconstruction depth; defaults to N_det.
        positivity:   project onto ``x ≥ 0`` each iteration.
        mask:         apply inscribed-cylinder support each iteration.
        euler_zyz_deg: [n_angles, 3] — switches to the 3-D rotmat projector
                      (matches ``run_sirt``'s ZYZ branch).
        return_loss:  if True, also return a float32 CPU tensor of shape
                      [n_iter] containing ``||b − A·y_k||²`` (data-fit MSE at
                      the momentum point — zero extra cost; a tight surrogate
                      for the MSE at ``x_k``).
        n_power_iter: power-iteration steps for the Lipschitz estimate.
        L:            user-supplied Lipschitz constant ``||A^T A||_op``;
                      ``None`` triggers power iteration.
        num_threads:  CPU 3-D path only.
        projector:    optional cached :class:`ParallelProjector` /
                      :class:`RotmatProjector` (CUDA reuse).
        volume:       optional warm-start ``[Ny, Nz, Nx]``.
        loss_callback: optional per-iteration hook ``loss_callback(i, loss)``
                      streaming the same value recorded in ``loss_history``
                      (see :func:`run_sirt` for the full contract).

    Returns the reconstructed volume (or ``(volume, loss_history)`` when
    ``return_loss=True``). Eager CUDA only — no CUDA Graph capture because
    ``t_k`` changes every iteration.
    """
    warm_start = _deprecated_kwarg(deprecated, "volume", "warm_start", warm_start, "run_fista")
    _reject_unknown_kwargs(deprecated, "run_fista")
    setup = _prepare_recon(
        "run_fista", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        allowed_dtypes=(torch.float32, torch.float16), warm_start=warm_start,
        positivity=positivity, mask=mask, support_mask=support_mask,
    )
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)
    volume, compute_dtype, n_iter = setup.volume0, setup.compute_dtype, setup.n_iter

    _fp, _bp, _internal_proj = _build_fp_bp(
        sinogram, angles_deg, (Ny, Nz, Nx), N_det, n_angles,
        device, num_threads, projector, euler_zyz_deg,
    )

    L_eff = (
        float(L) if L is not None
        else _estimate_lipschitz(_fp, _bp, (Ny, Nz, Nx), device, compute_dtype, n_power_iter)
    )

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_fista")
    return _fista_loop(
        volume, sinogram, _fp, _bp, n_iter, L_eff, setup.constraints.apply, rec,
    )


def run_fista_tv(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 50,
    *,
    tv_weight: float,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    euler_zyz_deg: torch.Tensor | None = None,
    n_power_iter: int = 10,
    L: float | None = None,
    tv_inner_iter: int = 20,
    tv_inner: str = "fgp",
    projector: object | None = None,
    warm_start: torch.Tensor | None = None,
    num_threads: int | None = None,
    return_loss: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
    **deprecated: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    FISTA-TV (Beck & Teboulle 2009 + 2009b) on the composite problem
    ``½ ‖A·x − b‖² + tv_weight · TV(x) + ι_C(x)``.

    Same outer Nesterov loop as :func:`run_fista`; the projection prox is
    swapped for the TV-l2 proximal operator (:func:`_tv_prox_3d`), solved on
    the dual by either Beck-Teboulle 2009b Fast Gradient Projection
    (``tv_inner="fgp"``) or Chambolle 2004 (``tv_inner="chambolle"``).
    Positivity and cylindrical mask are baked into the prox itself so the
    composite-convex theory applies.

    Args:
        sinogram, angles_deg, n_iter, Nz, positivity, mask, euler_zyz_deg,
        return_loss, n_power_iter, L, num_threads, projector, volume,
        loss_callback:
                      see :func:`run_fista`.
        tv_weight:    isotropic TV regularisation weight (required, > 0).
                      Pass ``tv_weight=0`` and use :func:`run_fista` directly.
        tv_inner_iter: inner dual-projection iterations (default 20).
        tv_inner:     ``"fgp"`` (default; O(1/k²) inner) or ``"chambolle"``
                      (O(1/k) inner).

    Returns same shape/format as :func:`run_fista`. Loss history records
    ``||b − A·y_k||²``, *not* the composite objective.
    """
    warm_start = _deprecated_kwarg(deprecated, "volume", "warm_start", warm_start, "run_fista_tv")
    _reject_unknown_kwargs(deprecated, "run_fista_tv")
    if tv_weight <= 0.0:
        raise ValueError(
            f"run_fista_tv requires tv_weight > 0 (got {tv_weight}). "
            "Use run_fista for the unregularised problem."
        )

    setup = _prepare_recon(
        "run_fista_tv", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        allowed_dtypes=(torch.float32, torch.float16), warm_start=warm_start,
        positivity=positivity, mask=mask, support_mask=support_mask,
    )
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)
    volume, compute_dtype, n_iter = setup.volume0, setup.compute_dtype, setup.n_iter

    _fp, _bp, _internal_proj = _build_fp_bp(
        sinogram, angles_deg, (Ny, Nz, Nx), N_det, n_angles,
        device, num_threads, projector, euler_zyz_deg,
    )

    L_eff = (
        float(L) if L is not None
        else _estimate_lipschitz(_fp, _bp, (Ny, Nz, Nx), device, compute_dtype, n_power_iter)
    )

    _mask = setup.constraints.support_mask
    lam = tv_weight / L_eff
    prox = lambda u: _tv_prox_3d(u, lam, positivity, _mask, tv_inner_iter, tv_inner)

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_fista_tv")
    return _fista_loop(volume, sinogram, _fp, _bp, n_iter, L_eff, prox, rec)


def run_sart(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 5,
    *,
    subset_size: int = 1,
    omega: float = 1.0,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    sparsity: float = 0.0,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — use 3D projector if set
    warm_start: torch.Tensor | None = None,     # [Ny, Nz, Nx]; its dtype must match the sinogram
    num_threads: int | None = None,             # CPU paths
    return_loss: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
    **deprecated: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Simultaneous Algebraic Reconstruction Technique (SART / OS-SART).

    Per-subset ordered-subsets update:

        for each subset S of size subset_size:
            r_S  = b_S - A_S · x
            x   += ω · A_Sᵀ(r_S / R_S) · C_S

    where:
        R_S = A_S(1)    (row normalisation per subset — safeguarded against /0)
        C_S = 1/A_Sᵀ(1) (column normalisation per subset — safeguarded against /0)

    With subset_size=1 this is plain SART (Andersen & Kak 1984).
    With subset_size=n_angles this is equivalent to one SIRT outer step
    (Wang & Yu 2007 OS-SART degenerate case).

    Subsets are formed by sequential splitting of the angle index:
    [0:subset_size], [subset_size:2*subset_size], …, remainder.
    The last subset may be smaller than subset_size.

    sparsity / positivity / mask are applied once per outer iteration
    (after all subsets have been processed), mirroring run_sirt.

    return_loss: if True, returns (volume, loss_history) where loss_history
    is a float32 CPU tensor of shape [n_iter] containing the sinogram MSE
    ||b - A·x||² / numel evaluated over the *full* sinogram after each
    outer iteration.

    euler_zyz_deg: if provided, uses the full 3D rotation-matrix projector
    (forward_project_3d / back_project_3d) instead of the scalar-angle
    single-axis projector. num_threads is forwarded to the CPU 3D kernels.

    This is a frozen baseline — no projector caching, no CUDA-Graph capture.
    Use run_sirt (or the AD path) for performance-critical workloads.

    volume: optional warm-start [Ny, Nz, Nx]. When provided, ``volume.dtype``
    sets the compute dtype (and must match the sinogram). When None, the
    algorithm zero-initialises in ``sinogram.dtype`` (fp32 or fp16). Output
    preserves the compute dtype.

    Returns volume [Ny, Nz, Nx].
    """
    warm_start = _deprecated_kwarg(deprecated, "volume", "warm_start", warm_start, "run_sart")
    _reject_unknown_kwargs(deprecated, "run_sart")
    setup = _prepare_recon(
        "run_sart", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        allowed_dtypes=(torch.float32, torch.float16), warm_start=warm_start,
        positivity=positivity, mask=mask, support_mask=support_mask, sparsity=sparsity,
    )
    subset_size = _as_int(subset_size, "subset_size", "run_sart", minimum=None)
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)
    volume, compute_dtype, n_iter = setup.volume0, setup.compute_dtype, setup.n_iter
    constraints = setup.constraints

    # Build functional closures that accept an index list so we can slice
    # both the angles tensor and the sinogram for each subset.
    if euler_zyz_deg is not None:
        def _fp_subset(v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return forward_project_3d(v, euler_zyz_deg[idx], N_det=N_det, num_threads=num_threads)

        def _bp_subset(s: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return back_project_3d(s, euler_zyz_deg[idx], Nz=Nz, num_threads=num_threads)

        def _fp_full(v: torch.Tensor) -> torch.Tensor:
            return forward_project_3d(v, euler_zyz_deg, N_det=N_det, num_threads=num_threads)
    else:
        def _fp_subset(v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return forward_project(v, angles_deg[idx], N_det=N_det, num_threads=num_threads)

        def _bp_subset(s: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return back_project(s, angles_deg[idx], Nz=Nz, num_threads=num_threads)

        def _fp_full(v: torch.Tensor) -> torch.Tensor:
            return forward_project(v, angles_deg, N_det=N_det, num_threads=num_threads)

    # Build subsets: list of index tensors
    clamp = max(1, min(subset_size, n_angles))
    all_idx = torch.arange(n_angles, dtype=torch.long)
    subsets = list(torch.split(all_idx, clamp))

    # Pre-compute per-subset normalisations (done once, outside the outer loop).
    ones_vol = torch.ones(Ny, Nz, Nx, dtype=compute_dtype, device=device)
    R_list: list[torch.Tensor] = []  # row normalisers [|S|, Ny, N_det]
    C_list: list[torch.Tensor] = []  # col normalisers [Ny, Nz, Nx]
    for idx in subsets:
        R_s = _fp_subset(ones_vol, idx)
        R_s = torch.where(R_s > 0, 1.0 / R_s, torch.zeros_like(R_s))
        R_list.append(R_s)

        # ones_sino for this subset
        ones_sino_s = torch.ones(len(idx), Ny, N_det, dtype=compute_dtype, device=device)
        C_s = _bp_subset(ones_sino_s, idx)
        C_s = torch.where(C_s > 0, 1.0 / C_s, torch.zeros_like(C_s))
        C_list.append(C_s)

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_sart")

    # --- outer iteration loop ---
    for it in rec.iterate(n_iter):
        # --- inner subset loop ---
        for idx, R_s, C_s in zip(subsets, R_list, C_list):
            sino_s = sinogram[idx]                   # [|S|, Ny, N_det]
            pred_s = _fp_subset(volume, idx)         # [|S|, Ny, N_det]
            r_s    = sino_s - pred_s                 # [|S|, Ny, N_det]
            volume = volume + omega * C_s * _bp_subset(r_s * R_s, idx)

        # Loss over the full sinogram after the subset sweep, before the
        # constraints (costs one extra full forward projection).
        if rec.wants_loss:
            with torch.no_grad():
                raw_res = sinogram - _fp_full(volume)
                rec.record(it, raw_res.pow(2).mean())

        volume = constraints.apply(volume)

    return rec.finish(volume)


@dataclass
class _AdRun:
    """State of one `run_ad_recon` call that its loop reads and updates."""
    sinogram: torch.Tensor
    angles_deg: torch.Tensor              # nominal tilt angles, on the device
    volume: torch.Tensor                  # leaf tensor, requires_grad
    proj: Any                             # ParallelProjector | RotmatProjector
    tilt_angle_opt: torch.Tensor | None   # [n_angles, 3] ZYZ Euler, requires_grad
    shifts: torch.Tensor | None           # [n_angles, 2], requires_grad
    optimisers: list                      # [volume, (angles), (shifts)]
    schedulers: list
    optim_cfg: OptimConfig
    reg_cfg: RegularizerConfig
    constraints_cfg: ConstraintConfig
    pose_cfg: PoseRefinementConfig | None
    refine_pose: bool
    refine_shifts: bool
    ref_idx: int | None                   # anchored view (anchor_reference)
    support: torch.Tensor | None          # resolved support, [1|Ny, Nz, Nx]
    s3im_gen: torch.Generator | None
    default_step_schedule: bool
    pose_warmup_iters: int
    alternating_period: int
    pred_shape: tuple[int, int, int]      # (n_angles, Ny, N_det)


def _ad_loss_terms(run: _AdRun, pred: torch.Tensor) -> dict[str, torch.Tensor]:
    """Data-fit and regularisation terms, keyed for the loss_callback
    components contract (only the active ones). Insertion order is the
    summation order of the total loss (float addition is not associative),
    kept identical to the historical loop."""
    optim_cfg, reg_cfg, pose_cfg = run.optim_cfg, run.reg_cfg, run.pose_cfg
    volume, sinogram, tilt = run.volume, run.sinogram, run.tilt_angle_opt
    terms: dict[str, torch.Tensor] = {}
    if optim_cfg.loss_fn == "mse":
        terms["fidelity"] = F.mse_loss(pred, sinogram)
    else:
        terms["fidelity"] = F.l1_loss(pred, sinogram)
    if reg_cfg.tv_weight > 0.0:
        terms["tv"] = reg_cfg.tv_weight * tv_loss_3d(volume)
    if reg_cfg.tv_xz_weight > 0.0:
        terms["tv_xz"] = reg_cfg.tv_xz_weight * tv_loss_xz(volume)
    if reg_cfg.sparsity > 0.0:
        terms["sparsity"] = _ad_sparsity_term(volume, reg_cfg.sparsity)
    if pose_cfg is not None and pose_cfg.in_plane_weight > 0.0 and tilt is not None:
        terms["in_plane"] = pose_cfg.in_plane_weight * (
            tilt[:, 0].pow(2).mean()
            + tilt[:, 2].pow(2).mean()
        )
    if reg_cfg.s3im_weight > 0.0:
        terms["s3im"] = reg_cfg.s3im_weight * s3im_loss(pred, sinogram, generator=run.s3im_gen)
    if reg_cfg.tilt_ssim_weight > 0.0:
        terms["tilt_ssim"] = reg_cfg.tilt_ssim_weight * tilt_ssim_loss(pred, sinogram)
    return terms


def _ad_step(run: _AdRun, it: int, rec: _LossRecorder) -> bool:
    """One `run_ad_recon` iteration up to the optimiser steps: forward
    projection, loss, backward, the pose-gradient masks and the scheduled
    optimiser/scheduler steps. Returns whether the volume was stepped."""
    for opt in run.optimisers:
        opt.zero_grad()

    # forward project
    tilt, shifts = run.tilt_angle_opt, run.shifts
    if run.refine_pose and tilt is not None:
        mats = euler_zyz_to_rotmat(tilt)
        pred = run.proj.forward(run.volume, mats)
    else:
        pred = run.proj.forward(run.volume)

    # A user-supplied projector built for another geometry would make
    # the loss broadcast silently (e.g. [1, Ny, N] against the sinogram).
    # Checked before the shift correction, whose own error ("shifts has
    # N entries but sinogram has M angles") would name the wrong culprit.
    _assert_projector_output_shape(
        pred, run.pred_shape, "forward-projection", "run_ad_recon",
    )

    # optional shift correction
    if run.refine_shifts and shifts is not None:
        pred = apply_shifts_2d(pred, shifts)

    terms = _ad_loss_terms(run, pred)
    loss = terms["fidelity"]
    for key, term in terms.items():
        if key != "fidelity":
            loss = loss + term

    # .item() (a host sync) only when the loss is returned or streamed
    if rec.wants_loss:
        rec.record(it, loss, terms)
    loss.backward()

    if run.ref_idx is not None:
        if tilt is not None and tilt.grad is not None:
            tilt.grad[run.ref_idx] = 0.0
        if shifts is not None and shifts.grad is not None:
            shifts.grad[run.ref_idx] = 0.0

    antisymmetric_z = run.pose_cfg is not None and run.pose_cfg.antisymmetric_z
    if antisymmetric_z and run.refine_pose and tilt is not None:
        if tilt.grad is not None:
            tilt.grad[:, 2] = 0.0

    step_volume = True
    if run.default_step_schedule:
        # Keep the default path exactly identical to the historical loop.
        for opt in run.optimisers:
            opt.step()
        for sched in run.schedulers:
            sched.step()
    else:
        step_volume, step_pose = _ad_recon_step_schedule(
            it,
            pose_warmup_iters=run.pose_warmup_iters,
            alternating_period=run.alternating_period,
        )
        step_flags = [step_volume] + [step_pose] * (len(run.optimisers) - 1)
        for opt, should_step in zip(run.optimisers, step_flags):
            if should_step:
                opt.step()
        # An inactive optimiser's LR must not decay before its stage runs.
        for sched, should_step in zip(run.schedulers, step_flags):
            if should_step:
                sched.step()
    return step_volume


def _ad_project_constraints(run: _AdRun, step_volume: bool) -> None:
    """The post-step constraints of `run_ad_recon`, inside ``no_grad`` on
    the leaf tensors: volume positivity, support, soft-threshold (only if
    the volume was stepped), then the tilt bound and ``psi = -phi``."""
    volume, tilt = run.volume, run.tilt_angle_opt
    constraints_cfg, pose_cfg = run.constraints_cfg, run.pose_cfg
    with torch.no_grad():
        if constraints_cfg.positivity:
            volume.clamp_(min=0.0)
        if run.support is not None:
            volume.mul_(run.support)
        if constraints_cfg.soft_threshold > 0.0 and step_volume:
            # Post-step L1 soft-threshold (ConstraintConfig.soft_threshold),
            # last: positivity -> support mask -> soft-threshold. Positivity
            # and a non-negative mask commute, so this is the constraints
            # order of fasttomo.constraints (mask -> positivity -> threshold).
            # Skipped in pose-only iterations: unlike positivity and the
            # mask it is not idempotent and would shrink a frozen volume.
            volume.copy_(_soft_threshold(volume, constraints_cfg.soft_threshold))
        tilt_bound_deg = pose_cfg.tilt_bound_deg if pose_cfg is not None else None
        if tilt_bound_deg is not None and run.refine_pose and tilt is not None:
            tilt[:, 1].clamp_(
                min=run.angles_deg - tilt_bound_deg,
                max=run.angles_deg + tilt_bound_deg,
            )
        antisymmetric_z = pose_cfg is not None and pose_cfg.antisymmetric_z
        if antisymmetric_z and run.refine_pose and tilt is not None:
            tilt[:, 2] = -tilt[:, 0]


def run_ad_recon(
    sinogram: torch.Tensor,            # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,          # [n_angles] degrees
    n_iter: int | None = None,         # None: config.optim.n_iter
    *,
    Nz: int = -1,
    config: AdReconConfig | None = None,
    support_mask: torch.Tensor | None = None,
    projector: object | None = None,
    warm_start: torch.Tensor | None = None,
    return_loss: bool = False,
    return_pose: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
) -> torch.Tensor | tuple:
    """
    Autograd-based reconstruction with optional pose and shift refinement.

    Library entry point for the AD reconstruction loop demonstrated in
    ``Notebooks/test.ipynb``. Knobs live on :class:`AdReconConfig` and its
    sub-configs (:class:`OptimConfig`, :class:`RegularizerConfig`,
    :class:`ConstraintConfig`, :class:`PoseRefinementConfig`); see those
    classes for per-field semantics.

    Args:
        sinogram:        [n_angles, Ny, N_det] float32 CUDA tensor.
        angles_deg:      [n_angles] tilt angles in degrees.
        Nz:              Reconstruction depth; defaults to ``N_det``.
        projector:       Optional pre-built :class:`RotmatProjector` (when
                         ``config.pose.refine_pose=True``) or
                         :class:`ParallelProjector` (otherwise). A projector
                         whose class does not match the requested mode raises
                         ``ValueError``. If ``None``, one is built internally
                         and torn down on return.
        config:          :class:`AdReconConfig` bundling optimiser,
                         regulariser, constraint, and pose-refinement knobs.
                         ``None`` (default) constructs ``AdReconConfig()``
                         which runs vanilla AD volume-only refinement.
        support_mask:    Optional precomputed support tensor enforced as a
                         hard constraint each iteration (``volume *=
                         support_mask`` after every optimiser step). Shape
                         ``[Nz, Nx]``, ``[1, Nz, Nx]``, or ``[Ny, Nz, Nx]``;
                         binary {0, 1} expected. Convenience alias for
                         ``config.constraints.mask`` — passing both raises
                         ``ValueError`` (one support per reconstruction).
        return_loss:     If True, also return the per-iteration loss history.
        return_pose:     If True, also return ``(refined_euler, refined_shifts)``
                         where each is ``None`` if the corresponding refinement
                         was disabled.
        loss_callback:   Optional hook invoked once per iteration as
                         ``loss_callback(i, loss)`` — 0-based index and the
                         same plain CPU float that lands in ``loss_history[i]``
                         (NaN/Inf passed through). Callbacks whose signature
                         accepts a ``components`` keyword also receive
                         ``components={...}``: per-term CPU floats keyed
                         ``fidelity`` plus only the regularisers active this
                         run (``tv``, ``tv_xz``, ``sparsity``, ``in_plane``,
                         ``s3im``, ``tilt_ssim``), each already weighted so
                         ``sum(components.values()) ≈ loss``. Exceptions
                         raised by the callback are logged and swallowed.

    Returns:
        * ``volume``               — float32 CUDA tensor [Ny, Nz, Nx]
        * if ``return_loss`` only  — ``(volume, loss_history)``
        * if ``return_pose`` only  — ``(volume, euler_or_None, shifts_or_None)``
        * if both                  — ``(volume, loss_history, euler_or_None, shifts_or_None)``

    Validation:
        ``config.validate()`` runs on entry, so fields edited after the
        dataclasses were constructed are checked too (for example an
        ``alternating_period`` or ``pose_init_shifts`` set without the
        matching refine flag, a negative weight, or an unknown ``loss_fn``).

    Staging:
        ``PoseRefinementConfig.pose_warmup_iters`` and ``alternating_period``
        schedule volume-only, pose-only and joint iterations; ``n_iter``
        counts all of them, and each cosine schedule spans only its
        optimiser's active iterations. See :class:`PoseRefinementConfig`
        for the Euler convention and how the pose constraints treat a pose
        initialiser.

    Constraints:
        After every optimiser step, inside ``torch.no_grad``, the volume gets
        ``ConstraintConfig.positivity`` (clamp at 0), then the support mask
        (multiplied in, so a soft mask scales voxels), then
        ``ConstraintConfig.sparsity`` as a soft-threshold
        ``f <- sign(f) * max(|f| - sparsity, 0)`` (the proximal step of an L1
        penalty, in volume units, once per iteration in which the volume
        was stepped, so not in pose-only iterations). It is independent of
        ``RegularizerConfig.sparsity``, which adds ``sparsity * mean(|f|)`` to
        the loss and acts through the gradient instead; both may be set.

    Note:
        This function requires a CUDA device. The underlying projectors are
        CUDA-only; passing CPU tensors will raise a ``RuntimeError`` from
        the projector constructor.
    """
    if config is None:
        config = AdReconConfig()
    # The dataclasses are mutable, so construction-time checks can be
    # bypassed by later edits; validate the whole tree here.
    config.validate()

    optim_cfg = config.optim
    reg_cfg = config.reg
    constraints_cfg = config.constraints
    pose_cfg = config.pose

    refine_pose = pose_cfg is not None and pose_cfg.refine_pose
    refine_shifts = pose_cfg is not None and pose_cfg.refine_shifts
    anchor_reference = pose_cfg is not None and pose_cfg.anchor_reference

    # Projector class/mode mismatch validation
    if projector is not None:
        if refine_pose and not isinstance(projector, RotmatProjector):
            raise ValueError(
                f"refine_pose=True requires a RotmatProjector, but got "
                f"{type(projector).__name__}. Either pass a RotmatProjector "
                f"or set refine_pose=False."
            )
        if not refine_pose and not isinstance(projector, ParallelProjector):
            raise ValueError(
                f"refine_pose=False requires a ParallelProjector, but got "
                f"{type(projector).__name__}. Either pass a ParallelProjector "
                f"or set refine_pose=True."
            )

    # ── support: ConstraintConfig.mask (bool True → cylinder; tensor → that
    # support) or the support_mask kwarg, one support per reconstruction ──
    if support_mask is not None:
        if constraints_cfg.cylinder or constraints_cfg.support_mask is not None:
            raise ValueError(
                "run_ad_recon: support_mask and ConstraintConfig.cylinder/support_mask "
                "are mutually exclusive — pass one support per reconstruction."
            )
    else:
        support_mask = constraints_cfg.support_mask

    setup = _prepare_recon(
        "run_ad_recon", sinogram, angles_deg, Nz=Nz,
        n_iter=optim_cfg.n_iter if n_iter is None else n_iter,
        require_cuda=True, warm_start=warm_start,
        mask=constraints_cfg.cylinder, support_mask=support_mask,
    )
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)
    n_iter = setup.n_iter
    angles_deg = angles_deg.float().to(device)

    ref_idx: int | None = None
    if anchor_reference:
        ref_idx = int(angles_deg.abs().argmin())

    # ── initialise volume ──────────────────────────────────────────────────
    volume = setup.volume0.requires_grad_(True)

    # ── build projector ────────────────────────────────────────────────────
    _internal_proj: object | None = None
    if projector is not None:
        _proj = projector
    elif refine_pose:
        _internal_proj = RotmatProjector(
            (Ny, Nz, Nx), n_angles=n_angles, device=device, N_det=N_det,
        )
        _proj = _internal_proj
    else:
        _internal_proj = ParallelProjector(
            (Ny, Nz, Nx), angles_deg, device=device, N_det=N_det,
        )
        _proj = _internal_proj

    # ── initialise pose / shift parameters ────────────────────────────────
    tilt_angle_opt: torch.Tensor | None = None
    shifts: torch.Tensor | None = None

    if refine_pose:
        if pose_cfg is not None and pose_cfg.pose_init_euler_deg is not None:
            tilt_angle_opt = _prepare_ad_recon_pose_init(
                pose_cfg.pose_init_euler_deg,
                expected_shape=(n_angles, 3),
                device=device,
                name="pose_init_euler_deg",
            )
        else:
            tilt_angle_opt = torch.zeros(
                n_angles, 3, dtype=torch.float32, device=device
            )
            tilt_angle_opt[:, 1] = angles_deg
        tilt_angle_opt.requires_grad_(True)

    if refine_shifts:
        if pose_cfg is not None and pose_cfg.pose_init_shifts is not None:
            shifts = _prepare_ad_recon_pose_init(
                pose_cfg.pose_init_shifts,
                expected_shape=(n_angles, 2),
                device=device,
                name="pose_init_shifts",
            )
            shifts.requires_grad_(True)
        else:
            shifts = torch.zeros(
                n_angles, 2, dtype=torch.float32, device=device,
                requires_grad=True,
            )

    # ── build optimiser list ───────────────────────────────────────────────
    opt_vol = torch.optim.Adam([volume], lr=optim_cfg.lr_volume)
    optimisers = [opt_vol]

    opt_angles: torch.optim.Optimizer | None = None
    opt_shifts: torch.optim.Optimizer | None = None

    if refine_pose and tilt_angle_opt is not None and pose_cfg is not None:
        opt_angles = torch.optim.Adam([tilt_angle_opt], lr=pose_cfg.lr_angles)
        optimisers.append(opt_angles)
    if refine_shifts and shifts is not None and pose_cfg is not None:
        opt_shifts = torch.optim.Adam([shifts], lr=pose_cfg.lr_shifts)
        optimisers.append(opt_shifts)

    pose_warmup_iters = pose_cfg.pose_warmup_iters if pose_cfg is not None else 0
    alternating_period = pose_cfg.alternating_period if pose_cfg is not None else 0
    default_step_schedule = pose_warmup_iters == 0 and alternating_period == 0
    if (refine_pose or refine_shifts) and pose_warmup_iters >= n_iter > 0:
        warnings.warn(
            f"run_ad_recon: pose_warmup_iters={pose_warmup_iters} >= "
            f"n_iter={n_iter}, so the pose/shift parameters never step.",
            stacklevel=2,
        )

    # ── cosine LR schedulers ───────────────────────────────────────────────
    # Each scheduler steps only when its optimiser steps, so T_max is that
    # optimiser's number of active iterations (n_iter on the default joint
    # schedule); every LR then anneals to ~0 by the end of its own stage.
    schedulers: list = []
    if optim_cfg.cosine_schedule:
        vol_steps, pose_steps = _ad_recon_active_steps(
            n_iter,
            pose_warmup_iters=pose_warmup_iters,
            alternating_period=alternating_period,
        )
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, vol_steps if opt is opt_vol else pose_steps))
            for opt in optimisers
        ]

    # ── S3IM generator (seeded once; reused each iteration) ───────────────
    _s3im_gen: torch.Generator | None = None
    if reg_cfg.s3im_weight > 0.0:
        _s3im_gen = torch.Generator(device=device).manual_seed(_S3IM_SEED)

    # ── iteration ─────────────────────────────────────────────────────────
    run = _AdRun(
        sinogram=sinogram, angles_deg=angles_deg, volume=volume, proj=_proj,
        tilt_angle_opt=tilt_angle_opt, shifts=shifts,
        optimisers=optimisers, schedulers=schedulers,
        optim_cfg=optim_cfg, reg_cfg=reg_cfg, constraints_cfg=constraints_cfg,
        pose_cfg=pose_cfg, refine_pose=refine_pose, refine_shifts=refine_shifts,
        ref_idx=ref_idx, support=setup.constraints.support_mask, s3im_gen=_s3im_gen,
        default_step_schedule=default_step_schedule,
        pose_warmup_iters=pose_warmup_iters, alternating_period=alternating_period,
        pred_shape=(n_angles, Ny, N_det),
    )
    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_ad_recon")
    for it in rec.iterate(n_iter):
        step_volume = _ad_step(run, it, rec)
        _ad_project_constraints(run, step_volume)
    rec.close()

    out = volume.detach()

    # ── build return tuple ─────────────────────────────────────────────────
    if not return_loss and not return_pose:
        return out

    result: list = [out]
    if return_loss:
        result.append(rec.history())
    if return_pose:
        refined_euler = tilt_angle_opt.detach().cpu() if tilt_angle_opt is not None else None
        refined_shifts = shifts.detach().cpu() if shifts is not None else None
        result.append(refined_euler)
        result.append(refined_shifts)

    return tuple(result)


# ---------------------------------------------------------------------------
# CUDA Graph helper for run_sirt
# ---------------------------------------------------------------------------

def _run_sirt_cuda_graph(
    volume: torch.Tensor,
    sinogram: torch.Tensor,
    n_iter: int,
    omega: float,
    C_vol: torch.Tensor,
    R_sino: torch.Tensor,
    _fp,
    _bp,
    constraints: ConstraintConfig,
    projector,
    key: tuple,
    on_replay: Callable[[], None] | None = None,
) -> torch.Tensor | None:
    """
    Run ``n_iter`` SIRT iterations via a captured CUDA Graph.

    Returns the reconstructed volume tensor, or ``None`` if capture was
    skipped / failed (caller should fall back to the eager loop).

    The graph is cached on ``projector._sirt_graph_cache[key]``; keys whose
    capture failed are kept in ``projector._sirt_graph_bad_keys`` so later
    calls with that projector skip capture. Both are declared by the
    projector base class; a projector without them (a duck-typed object)
    runs the eager loop.
    """
    cache = getattr(projector, "_sirt_graph_cache", None)
    bad_keys = getattr(projector, "_sirt_graph_bad_keys", None)
    if cache is None or bad_keys is None or key in bad_keys:
        return None

    # CUDA Graph capture is rooted on the *current* CUDA device. When the
    # caller's tensors live on a non-default device (e.g. cuda:1 while the
    # process default is cuda:0), torch.cuda.graph(g) captures on the wrong
    # device's stream — the graph captures nothing and the failed capture
    # leaves the runtime in a state that crashes the eager fallback. Pin
    # the current device to the tensor's device for capture and replay.
    target_device = volume.device

    if key not in cache:
        # --- build a new graph for this key ---
        try:
            # Allocate persistent static buffers. The constraint mask is a
            # graph input too (captured by pointer inside apply_inplace_),
            # so it gets its own static buffer refreshed on every replay —
            # otherwise a later call with a different mask of the same shape
            # would silently replay the first call's mask.
            vol_static = volume.clone()
            sino_static = sinogram.clone()
            C_static = C_vol.clone()
            R_static = R_sino.clone()
            mask_static = (
                constraints.support_mask.clone()
                if constraints.support_mask is not None else None
            )
            static_constraints = ConstraintConfig(
                positivity=constraints.positivity,
                support_mask=mask_static,
                soft_threshold=constraints.soft_threshold,
            )

            # Warmup: one eager iteration with the static buffer.
            # Populates cuFFT plans, allocator pool, projector texture caches.
            with torch.no_grad():
                _sirt_iter_inplace(
                    vol_static, sino_static, C_static, R_static, omega,
                    _fp, _bp, static_constraints,
                )

            # Capture the iteration body on the correct device. Use an
            # explicit Stream pinned to the target device so torch.cuda.graph
            # captures on the right device — the implicit side stream is
            # created lazily on the first call's device and reused for later
            # calls, which corrupts cross-device captures.
            try:
                with torch.cuda.device(target_device):
                    torch.cuda.synchronize(target_device)
                    capture_stream = torch.cuda.Stream(device=target_device)
                    g = torch.cuda.CUDAGraph()
                    with torch.no_grad(), torch.cuda.graph(g, stream=capture_stream):
                        _sirt_iter_inplace(
                            vol_static, sino_static, C_static, R_static, omega,
                            _fp, _bp, static_constraints,
                        )
                    torch.cuda.synchronize(target_device)
            finally:
                # Free texture caches of projectors collected during the
                # capture (their release was deferred, see projector.py).
                _drain_deferred_handles()

            cache[key] = (g, vol_static, sino_static, C_static, R_static,
                          mask_static)

        except Exception as exc:
            _log.warning(
                "run_sirt: CUDA Graph capture failed for key %s (%s). "
                "Falling back to eager loop for this and future calls "
                "with this configuration and projector.",
                key, exc,
            )
            bad_keys.add(key)
            return None

    g, vol_static, sino_static, C_static, R_static, mask_static = cache[key]

    # Reset the captured static buffer to the caller's starting volume so the
    # warmup iteration that ran inside the cache-build branch above does not
    # advance the reconstruction. After this copy each replay applies one
    # SIRT iteration, yielding exactly n_iter updates for both first-call
    # and cache-hit paths.
    vol_static.copy_(volume)
    # Keep graph inputs fresh on cache hits; captured CUDA graphs replay with
    # pointer-stable tensors, so dynamic call-time tensors must be copied into
    # those static buffers before replay.
    sino_static.copy_(sinogram)
    C_static.copy_(C_vol)
    R_static.copy_(R_sino)
    if mask_static is not None:
        mask_static.copy_(constraints.support_mask)

    with torch.cuda.device(target_device):
        for _ in range(n_iter):
            g.replay()
            if on_replay is not None:
                on_replay()
        torch.cuda.synchronize(target_device)

    return vol_static.clone()


def _sirt_iter_inplace(
    vol_static: torch.Tensor,
    sinogram: torch.Tensor,
    C_vol: torch.Tensor,
    R_sino: torch.Tensor,
    omega: float,
    _fp,
    _bp,
    constraints: ConstraintConfig,
) -> None:
    """
    One in-place SIRT iteration on ``vol_static``.

    All writes go through in-place ops (``add_``, ``mul_``, ``clamp_``,
    ``copy_``) so that ``vol_static``'s memory address is stable —
    a requirement for CUDA Graph capture to replay correctly.
    """
    raw_res  = sinogram - _fp(vol_static)
    residual = raw_res * R_sino
    update   = _bp(residual)
    # Avoid creating a temporary for `update * C_vol` to keep allocations
    # inside the graph's memory pool.
    update.mul_(C_vol)
    vol_static.add_(update, alpha=omega)
    constraints.apply_inplace_(vol_static)


def _genfire_hermitian_sym(
    grid_real: torch.Tensor,   # [Ny, bigN, bigN]
    grid_imag: torch.Tensor,
    known_2d: torch.Tensor,    # [bigN, bigN] bool
    bigN: int,
) -> tuple:
    """
    Fill the conjugate-symmetric half of the Fourier grid.

    For a real volume F(-k) = conj(F(k)).  In fftshifted convention
    (DC at index bigN//2) the conjugate of index i is (bigN - i) % bigN.
    Fills positions where the conjugate is known but the position itself
    is not; returns updated (grid_real, grid_imag, known_2d).
    """
    flip = np.mod(-np.arange(bigN), bigN).astype(np.int64)
    flip_t = torch.from_numpy(flip).to(grid_real.device)

    conj_r =  grid_real[:, flip_t][:, :, flip_t]
    conj_i = -grid_imag[:, flip_t][:, :, flip_t]
    known_flip = known_2d[flip_t][:, flip_t]

    fill = known_flip & (~known_2d)
    if fill.any():
        grid_real = grid_real.clone()
        grid_imag = grid_imag.clone()
        grid_real[:, fill] = conj_r[:, fill]
        grid_imag[:, fill] = conj_i[:, fill]
        known_2d = known_2d | fill
    return grid_real, grid_imag, known_2d


def run_genfire(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    *,
    n_iter: int = 100,
    Nz: int = -1,
    positivity: bool = True,
    mask: bool | None = None,
    support_mask: torch.Tensor | None = None,
    sparsity: float = 0.0,
    oversampling_ratio: float = 2.0,
    cutoff_dist: float = 0.5,
    resolution_extension: bool = True,
    warm_start: torch.Tensor | None = None,
    return_loss: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    GENFIRE reconstruction (Pryor et al. 2017).

    Assembles a 2D Fourier grid per Y-slice via inverse-distance weighted
    gridding of oversampled 1D FFT projections, then iterates between
    Fourier-space data enforcement and real-space constraints (support,
    positivity).  Optional resolution extension/suppression schedule
    gradually widens then narrows the enforced frequency band to improve
    convergence under limited tilt range.

    return_loss: if True, returns (volume, loss_history) where loss_history
    is a float32 CPU tensor of shape [n_iter] containing the mean L1 Fourier
    error on known grid points at each iteration.

    sparsity: L1 soft-threshold applied in real space after each IFFT step
    (proximal operator of sparsity·‖x‖₁). 0.0 disables it.

    Returns volume [Ny, Nz, Nx].
    mask=True inscribes a cylindrical support in the reconstruction plane.

    fp32 only — the GENFIRE gridding kernel (`csrc/cuda/genfire.cu`) has no
    half-precision path. Cast your sinogram to ``torch.float32`` first.
    """
    # mask=None: the cylinder unless a support_mask is given (the default
    # stays the historical mask=True for calls without support_mask).
    if mask is None:
        mask = support_mask is None
    # Validation only (frozen algorithm): GENFIRE builds its support and
    # constraints on the oversampled grid below.
    setup = _prepare_recon(
        "run_genfire", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        init_volume=False, warm_start=warm_start, mask=mask,
        support_mask=support_mask, sparsity=sparsity,
    )
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)
    n_iter = setup.n_iter

    # oversampled grid size (must be even)
    bigN = int(round(N_det * oversampling_ratio))
    if bigN % 2 != 0:
        bigN += 1

    angles_np = (angles_deg.detach().cpu().float() * (math.pi / 180.0)).numpy()
    cos_a = np.cos(angles_np).astype(np.float32)
    sin_a = np.sin(angles_np).astype(np.float32)

    # ── 1D FFT of zero-padded projections (centred convention) ─────────────
    pad_l = (bigN - N_det) // 2
    pad_r = bigN - N_det - pad_l
    proj_pad = F.pad(sinogram, (pad_l, pad_r))            # [n_ang, Ny, bigN]
    # The pulse is centred at array index bigN/2, not at 0.  Without an
    # ifftshift before the FFT, fft(proj_pad)[ξ] picks up a (-1)^ξ phase that
    # systematically flips signs at off-axis grid cells (where ξ ≠ ikz+ikx
    # mod 2 after rounding ξ·cosθ, ξ·sinθ).  The *centred* FT removes that
    # phase, giving the literal Fourier-slice-theorem identity
    # fk_shifted[half+ξ] = F_2D(ξ·cosθ/bigN, ξ·sinθ/bigN).
    fk_shifted = torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(proj_pad, dim=-1), dim=-1),
        dim=-1,
    )                                                     # DC at bigN//2
    fk_r = fk_shifted.real.contiguous()
    fk_i = fk_shifted.imag.contiguous()

    # ── Fourier gridding ────────────────────────────────────────────────────
    if sinogram.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        grid_r = sinogram.new_zeros(Ny, bigN, bigN)
        grid_i = sinogram.new_zeros(Ny, bigN, bigN)
        grid_w = sinogram.new_zeros(Ny, bigN, bigN)
        cos_t = torch.from_numpy(cos_a).to(device)
        sin_t = torch.from_numpy(sin_a).to(device)
        with torch.cuda.device(device):
            _core.genfire_gridding_2d_cuda(
                fk_r.data_ptr(), fk_i.data_ptr(),
                cos_t.data_ptr(), sin_t.data_ptr(),
                grid_r.data_ptr(), grid_i.data_ptr(), grid_w.data_ptr(),
                bigN, Ny, n_angles, cutoff_dist,
                torch.cuda.current_stream(device).cuda_stream,
            )
    else:
        grid_r = torch.zeros(Ny, bigN, bigN, dtype=torch.float32)
        grid_i = torch.zeros(Ny, bigN, bigN, dtype=torch.float32)
        grid_w = torch.zeros(Ny, bigN, bigN, dtype=torch.float32)
        cos_t = torch.from_numpy(cos_a)
        sin_t = torch.from_numpy(sin_a)
        _core.genfire_gridding_2d_cpu(
            _data_ptr(fk_r, "fk_real"), _data_ptr(fk_i, "fk_imag"),
            _data_ptr(cos_t, "cos_a"), _data_ptr(sin_t, "sin_a"),
            grid_r.data_ptr(), grid_i.data_ptr(), grid_w.data_ptr(),
            bigN, Ny, n_angles, cutoff_dist,
        )

    # ── normalize by accumulated weights ────────────────────────────────────
    known_2d = grid_w[0] > 0                        # [bigN, bigN]
    w_norm = grid_w[0].clamp(min=1e-12)
    grid_r = grid_r / w_norm.unsqueeze(0)
    grid_i = grid_i / w_norm.unsqueeze(0)

    # ── Hermitian symmetry — fills conjugate half of grid ───────────────────
    grid_r, grid_i, known_2d = _genfire_hermitian_sym(grid_r, grid_i, known_2d, bigN)

    # ── Convert to torch-native FFT convention (DC at [0,0]) ────────────────
    measured_K = torch.fft.ifftshift(
        grid_r + 1j * grid_i, dim=(-2, -1)
    )                                                # [Ny, bigN, bigN] complex
    known_us = torch.fft.ifftshift(
        known_2d.float(), dim=(-2, -1)
    ).bool()                                         # [bigN, bigN]

    # ── real-space support mask ─────────────────────────────────────────────
    z0 = (bigN - Nz) // 2;  z1 = z0 + Nz
    x0 = (bigN - Nx) // 2;  x1 = x0 + Nx
    if support_mask is not None:
        # [1|Ny, Nz, Nx] support, zero on the oversampling margin
        user_support = setup.constraints.support_mask
        support = sinogram.new_zeros(user_support.shape[0], bigN, bigN)
        support[:, z0:z1, x0:x1] = user_support
    else:
        support = sinogram.new_zeros(bigN, bigN)
        support[z0:z1, x0:x1] = (
            cylindrical_mask(Nz, Nx, device=device) if mask
            else sinogram.new_ones(Nz, Nx)
        )
        support = support.unsqueeze(0)

    # ── resolution extension/suppression schedule ───────────────────────────
    cutoffs: np.ndarray = np.zeros(n_iter, dtype=np.float32)
    res_ind_us: torch.Tensor = sinogram.new_zeros(bigN, bigN)
    if resolution_extension:
        delay_pts = np.array(_GENFIRE_RES_EXTENSION_PIVOTS, dtype=np.float32)
        cutoffs = np.interp(
            np.linspace(0.0, 1.0, n_iter),
            np.linspace(0.0, 1.0, len(delay_pts)),
            delay_pts,
        ).astype(np.float32)

        iz_c = (np.arange(bigN) - bigN // 2).astype(np.float32)
        ix_c = (np.arange(bigN) - bigN // 2).astype(np.float32)
        R_np = np.minimum(
            np.sqrt(iz_c[:, None] ** 2 + ix_c[None, :] ** 2) / (bigN // 2),
            1.0,
        ).astype(np.float32)
        res_ind = torch.from_numpy(1.0 - R_np).to(device)      # [bigN, bigN] shifted
        res_ind_us = torch.fft.ifftshift(res_ind)               # unshifted

    # ── iteration ───────────────────────────────────────────────────────────
    _inner_constraints = ConstraintConfig(
        positivity=positivity, support_mask=support, soft_threshold=sparsity,
    )
    current_F = measured_K.clone()
    if setup.volume0 is not None:
        # Warm start: the Fourier transform of the given volume (embedded in
        # the oversampled grid) with the measured values on the known grid
        # points, so the first real-space iterate is the data-consistent
        # version of warm_start instead of the zero-filled data.
        v0 = sinogram.new_zeros(Ny, bigN, bigN)
        v0[:, z0:z1, x0:x1] = setup.volume0
        current_F = torch.fft.fftn(torch.fft.ifftshift(v0, dim=(-2, -1)), dim=(-2, -1))
        current_F[:, known_us] = measured_K[:, known_us]
    best_F = current_F
    best_err = float("inf")
    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_genfire")

    for i_iter in rec.iterate(n_iter):
        # build enforce mask for this iteration
        if resolution_extension:
            thresh = float(cutoffs[i_iter])
            enforce_us = known_us & (res_ind_us > thresh)
        else:
            enforce_us = known_us                               # [bigN, bigN]

        # IFFT → real-space volume (centred convention: signal at array index half)
        vol = torch.fft.fftshift(
            torch.fft.ifftn(current_F, dim=(-2, -1)), dim=(-2, -1),
        ).real                                                # [Ny, bigN, bigN]

        # real-space constraints: support → positivity → sparsity
        vol = _inner_constraints.apply(vol)

        # FFT back to Fourier space (centred convention: ifftshift before fft)
        F_cur = torch.fft.fftn(
            torch.fft.ifftshift(vol, dim=(-2, -1)), dim=(-2, -1),
        )

        # track best-error iterate (L1 on all known voxels, not just enforce_us)
        err = (F_cur[:, known_us] - measured_K[:, known_us]).abs().mean().item()
        rec.record(i_iter, err)
        if err < best_err:
            best_err = err
            best_F = F_cur.clone()

        # enforce Fourier constraint on known voxels
        F_cur[:, enforce_us] = measured_K[:, enforce_us]
        current_F = F_cur

    # final IFFT of best iterate (centred convention)
    vol = torch.fft.fftshift(
        torch.fft.ifftn(best_F, dim=(-2, -1)), dim=(-2, -1),
    ).real                                                      # [Ny, bigN, bigN]
    vol = ConstraintConfig(positivity=positivity).apply(vol)
    if support_mask is not None:
        # The best iterate is support-constrained only up to FFT round-off;
        # a user support is a hard constraint, so it is exact in the output.
        vol = vol * support

    # crop to [Ny, Nz, Nx]
    vol = vol[:, z0:z1, x0:x1]
    return rec.finish(vol)


def run_wsirt(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    n_iter: int = 50,
    *,
    omega: float = 1.0,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    sparsity: float = 0.0,
    threshold_scale: float = 1.0,
    warm_start: torch.Tensor | None = None,
    num_threads: int | None = None,
    return_loss: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted SIRT (Wolf et al. 2014).

    Replaces the simple back-projection at each SIRT step with the
    exact-filter W-BP. Converges faster and gives higher fidelity than
    SIRT, especially under limited tilt range.

    f^k = f^{k-1} + ω · W-BP(b - A·f^{k-1})

    W-BP is the exact weighted back-projection of
    `run_weighted_backprojection` (edge margin, voxel-driven on both
    backends). With it wSIRT does not diverge at the default ``omega=1``;
    without the margin it diverged on most geometries tested. CPU and CUDA
    results agree to ~2e-4, the floor set by the CUDA texture unit's 8-bit
    interpolation weights in the forward and back projectors.

    return_loss: if True, returns (volume, loss_history) where loss_history
    is a float32 CPU tensor of shape [n_iter] containing the sinogram MSE
    ||b - A·f^k||² / numel at each iteration.

    sparsity: L1 soft-threshold applied after each update (proximal operator
    of sparsity·‖x‖₁). 0.0 disables it.

    mask=True enforces cylindrical support at each iteration (GENFIRE-style).

    fp32 only — the exact WF construction goes through `_compute_exact_wf`
    which builds the PSF in fp32 and applies a 2-D rfft, both fp32-pinned.
    Cast your sinogram to ``torch.float32`` first.
    """
    setup = _prepare_recon(
        "run_wsirt", sinogram, angles_deg, Nz=Nz, n_iter=n_iter, warm_start=warm_start,
        positivity=positivity, mask=mask, support_mask=support_mask, sparsity=sparsity,
    )
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)
    volume, constraints, n_iter = setup.volume0, setup.constraints, setup.n_iter

    # exact W-BP operator, built once per tilt geometry
    _wbp = _make_wbp(angles_deg, Nz, Nx, N_det, device, threshold_scale)

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_wsirt")

    for it in rec.iterate(n_iter):
        proj     = forward_project(volume, angles_deg, N_det=N_det, num_threads=num_threads)
        residual = sinogram - proj
        if rec.wants_loss:
            rec.record(it, residual.pow(2).mean())
        volume   = volume + omega * _wbp(residual)
        volume   = constraints.apply(volume)

    return rec.finish(volume)


def refine_angles(
    volume: torch.Tensor,                       # [Ny, Nz, Nx]
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor | None = None,     # [n_angles] — single-axis mode
    *,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — ZYZ mode
    search_range_deg: float = 3.0,
    n_steps: int = 31,
    num_threads: int | None = None,             # CPU paths
) -> torch.Tensor:
    """
    RESIRE angular refinement — brute-force grid search per projection.

    **Single-axis mode** (angles_deg: [n_angles]):
      For each projection j, searches n_steps candidates in
      [θ_j ± search_range_deg] and picks the one minimising the L1 R-factor.
      Returns refined angles [n_angles] on CPU.

    **ZYZ mode** (euler_zyz_deg: [n_angles, 3]):
      Coordinate descent over three Euler angles (φ, θ, ψ) per projection.
      For each projection j, iterates through axes 0→1→2, fixing the other
      two and searching n_steps candidates along the active axis. One call
      performs one full coordinate-descent round per projection.
      Uses forward_project_3d (CUDA only).
      Returns refined euler_zyz_deg [n_angles, 3] on CPU.

    Typical usage:

        angles = initial_angles
        for _ in range(n_outer):
            vol    = ft.run_resire(sino, angles, n_iter=100)
            angles = ft.refine_angles(vol, sino, angles_deg=angles)

        # ZYZ version:
        euler = torch.zeros(n_angles, 3); euler[:, 1] = tilt_angles
        for _ in range(n_outer):
            vol   = ft.run_resire(sino, None, n_iter=100, euler_zyz_deg=euler)
            euler = ft.refine_angles(vol, sino, euler_zyz_deg=euler)
    """
    _assert_dtype_match(sinogram, volume, "sinogram", "volume", "refine_angles")

    if angles_deg is not None and euler_zyz_deg is None:
        # ── single-axis mode ──────────────────────────────────────────────
        n_angles, Ny, N_det = sinogram.shape
        refined = angles_deg.detach().cpu().clone()

        for j in range(n_angles):
            theta_j = refined[j].item()
            candidates = torch.linspace(
                theta_j - search_range_deg,
                theta_j + search_range_deg,
                n_steps,
            )
            projs = forward_project(volume, candidates.to(sinogram.device), N_det=N_det,
                                    num_threads=num_threads)
            measured_j = sinogram[j]
            residuals  = (projs - measured_j.unsqueeze(0)).abs().sum(dim=(-2, -1))
            r_factors  = residuals / (measured_j.abs().sum().clamp(min=_REL_EPS))
            refined[j] = candidates[r_factors.argmin()]

        return refined

    elif euler_zyz_deg is not None and angles_deg is None:
        # ── ZYZ coordinate-descent mode ───────────────────────────────────
        n_angles, Ny, N_det = sinogram.shape
        refined = euler_zyz_deg.detach().cpu().clone()  # [n_angles, 3]

        for j in range(n_angles):
            measured_j  = sinogram[j]                              # [Ny, N_det]
            normalizer  = measured_j.abs().sum().clamp(min=_REL_EPS)

            for axis in range(3):  # φ, θ, ψ in sequence
                current_val = refined[j, axis].item()
                candidates  = torch.linspace(
                    current_val - search_range_deg,
                    current_val + search_range_deg,
                    n_steps,
                )  # [n_steps]

                # build [n_steps, 3] euler batch — vary axis, hold others fixed
                euler_batch = refined[j].unsqueeze(0).expand(n_steps, -1).clone()
                euler_batch[:, axis] = candidates

                # forward project at all n_steps orientations → [n_steps, Ny, N_det]
                projs = forward_project_3d(
                    volume, euler_batch.to(sinogram.device), N_det=N_det,
                    num_threads=num_threads,
                )
                residuals = (projs - measured_j.unsqueeze(0)).abs().sum(dim=(-2, -1))
                refined[j, axis] = candidates[residuals.div(normalizer).argmin()]

        return refined

    else:
        raise ValueError(
            "Provide exactly one of angles_deg (single-axis) or "
            "euler_zyz_deg (ZYZ 3D), not both or neither."
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
    """
    REal Space Iterative REconstruction (RESIRE, Miao group).  CUDA only.

    Projected gradient descent on ||Ax - b||²:

        x^{k+1} = P_C(x^k + α · A^T(b − A·x^k))

    where α = step_size / (n_angles · Nz).

    Uses the Miao group voxel-driven scatter/gather kernels with 4-point
    supersampling (forward_project_resire / back_project_resire).  For a
    single-axis tilt series, angles_deg is wrapped as ZYZ triples (φ=0,
    θ=angle, ψ=0) internally.  Pass euler_zyz_deg directly for dual-axis
    tilt or general orientations.

    support: explicit float mask [Ny, Nz, Nx] or [1, Nz, Nx] applied each
    iteration.  If None and mask=True, uses the inscribed cylindrical mask.

    return_loss: if True, returns (volume, loss_history) where loss_history
    is a float32 CPU tensor of shape [n_iter] containing the sinogram MSE
    ||b - A·x^k||² / numel at each iteration.

    sparsity: L1 soft-threshold applied after each gradient step (proximal
    operator of sparsity·‖x‖₁, as noted in Algorithm 1 of the RESIRE paper).
    0.0 disables it.

    fp32 only — the RESIRE FP/BP kernels (`csrc/cuda/projection_resire.cu`)
    have no half-precision path. Cast your sinogram to ``torch.float32`` first.

    Returns volume [Ny, Nz, Nx].
    """
    support_mask = _deprecated_kwarg(deprecated, "support", "support_mask",
                                     support_mask, "run_resire")
    _reject_unknown_kwargs(deprecated, "run_resire")
    setup = _prepare_recon(
        "run_resire", sinogram, angles_deg, Nz=Nz, n_iter=n_iter, require_cuda=True,
        warm_start=warm_start, positivity=positivity, mask=mask,
        support_mask=support_mask, sparsity=sparsity,
    )
    n_angles, Ny, N_det, Nz, Nx = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx)
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

    _fp = lambda v: forward_project_resire(v, euler_zyz_deg, N_det=N_det)
    _bp = lambda s: back_project_resire(s, euler_zyz_deg, Nz=Nz)

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_resire")

    for it in rec.iterate(n_iter):
        residual = sinogram - _fp(volume)
        if rec.wants_loss:
            rec.record(it, residual.pow(2).mean())
        volume   = volume + alpha * _bp(residual)
        volume   = constraints.apply(volume)

    return rec.finish(volume)


# ── autograd-based reference reconstruction ──────────────────────────────

def run_grad_descent(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor | None = None,     # [n_angles] — single-axis
    n_iter: int = 200,
    *,
    lr: float = 1e-2,
    Nz: int = -1,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    tv_weight: float = 0.0,
    tv_eps: float = 1e-8,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — 3D mode
    use_resire_projector: bool = False,
    warm_start: torch.Tensor | None = None,
    num_threads: int | None = None,             # CPU paths
    return_loss: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Reference autograd-based reconstruction.

    Demonstrates the canonical pattern using FastTomo's differentiable
    projectors: a torch.optim optimizer drives ``volume.requires_grad_(True)``
    by calling ``loss.backward()`` each iteration. Equivalent to Landweber
    iteration when ``optimizer_cls=torch.optim.SGD``; with ``Adam`` (default)
    the adaptive step size loosely substitutes for SIRT's diagonal
    preconditioners.

    This is *not* a SIRT/RESIRE replacement — those remain the
    production-fast iterative entry points. Use this when you need a custom
    differentiable loss, learnable prior, or any reconstruction that benefits
    from being part of a larger autograd graph.

    Args:
        sinogram:       [n_angles, Ny, N_det] — measured tilt series
        angles_deg:     [n_angles] — single-axis tilt angles, mutually
                        exclusive with ``euler_zyz_deg``
        n_iter:         number of optimizer steps
        lr:             optimizer learning rate
        Nz:             reconstruction depth; defaults to N_det
        optimizer_cls:  any torch.optim class (Adam, SGD, LBFGS, …)
        positivity:     project volume onto [0, ∞) after each step
        mask:           apply inscribed-cylinder mask after each step
        tv_weight:      3-D isotropic TV regularisation weight; the
                        penalty is ``mean(sqrt(∂y² + ∂z² + ∂x² + tv_eps))``
                        across the inner-corner grid of the volume.
                        Dispatches to a fused CUDA kernel via
                        ``fasttomo.tv_loss_3d`` (CPU falls back to pure
                        PyTorch). 0.0 disables the regularizer.
        euler_zyz_deg:  [n_angles, 3] — switches to the 3D rotmat projector
                        (CUDA only). Mutually exclusive with ``angles_deg``.
        use_resire_projector: with ``euler_zyz_deg``, use the RESIRE 4-pt
                        scatter projector instead of the rotmat tex3D one.
        return_loss:    if True, also return the per-iteration loss history.
        tv_eps:         smoothing constant inside the TV sqrt (default
                        1e-8); only used when ``tv_weight > 0``.

    Constraints (positivity, mask) are applied **inside torch.no_grad()** on
    the leaf tensor — the autograd-correct way to project. Users hand-rolling
    similar loops should mimic this; in-place ops on a graph-tracked tensor
    raise "leaf Variable that requires grad ... in-place operation".

    Memory note: peak ~7 N³ vs SIRT's ~5 N³ because the BP backward must
    materialise both the sinogram (forward output) and the volume gradient.

    LBFGS is supported — it calls the closure multiple times per ``step()``
    and rebuilds the graph each call.
    """
    if (angles_deg is None) == (euler_zyz_deg is None):
        raise ValueError(
            "Provide exactly one of angles_deg (single-axis) or "
            "euler_zyz_deg (3D ZYZ), not both or neither."
        )

    setup = _prepare_recon(
        "run_grad_descent", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        warm_start=warm_start, positivity=positivity, mask=mask, support_mask=support_mask,
    )
    n_angles, Ny, N_det, Nz, Nx = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx)
    constraints, n_iter = setup.constraints, setup.n_iter

    volume = setup.volume0.requires_grad_(True)
    opt = optimizer_cls([volume], lr=lr)

    if euler_zyz_deg is not None:
        if use_resire_projector:
            _fp = lambda v: forward_project_resire(v, euler_zyz_deg, N_det=N_det)
        else:
            _fp = lambda v: forward_project_3d(v, euler_zyz_deg, N_det=N_det, num_threads=num_threads)
    else:
        _fp = lambda v: forward_project(v, angles_deg, N_det=N_det, num_threads=num_threads)

    is_lbfgs = isinstance(opt, torch.optim.LBFGS)
    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_grad_descent")

    # Loss terms of the first closure evaluation of each step: the loss
    # LBFGS.step returns is that evaluation's (it may evaluate several times).
    terms: dict[str, torch.Tensor] = {}

    def closure():
        opt.zero_grad()
        sino_pred = _fp(volume)
        fidelity = ((sino_pred - sinogram) ** 2).mean()
        loss = fidelity
        tv = None
        if tv_weight > 0.0:
            tv = tv_weight * tv_loss_3d(volume, eps=tv_eps)
            loss = loss + tv
        if not terms:
            terms["fidelity"] = fidelity.detach()
            if tv is not None:
                terms["tv"] = tv.detach()
        loss.backward()
        return loss

    for it in rec.iterate(n_iter):
        terms.clear()
        if is_lbfgs:
            loss_val = opt.step(closure)
        else:
            loss_val = closure()
            opt.step()
        if rec.wants_loss:
            rec.record(it, loss_val.detach(), terms)
        with torch.no_grad():
            constraints.apply_inplace_(volume)

    return rec.finish(volume.detach())


# ── MBIR ───────────────────────────────────────────────────────────────────
# Classical Model-Based Iterative Reconstruction for single-axis parallel-beam
# tomography. Reference: Venkatakrishnan et al. 2015 (IEEE TCI 1(1)). The
# kernels are in csrc/{cuda,cpu}/mbir_icd.{cu,c}; see
# fasttomo-wiki/wiki/algorithms/mbir.md for the surrogate-MM derivation.


@dataclass
class _MbirState:
    """One MBIR optimization level's running state.

    Maintained by `_run_mbir_single_scale` and passed across multires
    levels so the per-tilt offset survives upsampling. (σ is re-initialized
    per scale; see the docstring of `run_mbir` for why.)
    """
    volume: torch.Tensor          # [Ny, Nz, Nx] float32
    d_offsets: torch.Tensor       # [n_angles] float32
    sigma: float
    e_sino: torch.Tensor | None   # [n_angles, Ny, N_det] float32 — residual (None: not yet computed)
    n_done: int = 0               # outer iterations run at this scale


def _huber_surr_weight(r_abs: torch.Tensor, T: float, delta: float) -> torch.Tensor:
    """Per-element Huber surrogate weight matching `huber_surr_weight_dev` in
    the CUDA kernel and `huber_surr_weight` in the CPU reference. When
    T = +inf the weight collapses to ones (conventional MBIR, no anomaly
    suppression)."""
    if not math.isfinite(T):
        return torch.ones_like(r_abs)
    inside = r_abs < T
    outside_w = delta * T / r_abs.clamp(min=_MBIR_WEIGHT_EPS)
    return torch.where(inside, torch.ones_like(r_abs), outside_w)


def _mbir_update_offset_variance(
    e_sino: torch.Tensor,
    d_offsets: torch.Tensor,
    sigma: float,
    sigma_min: float,
    T: float,
    delta: float,
    lambdas: torch.Tensor | None,
    estimate_offset: bool,
    estimate_variance: bool,
) -> tuple[torch.Tensor, float]:
    """Closed-form updates for d_k and σ under the quadratic surrogate.

    The only implementation: the CPU and CUDA sweeps run the ICD voxel
    updates only, and `_run_mbir_single_scale` calls this afterwards on the
    live e_sino for both backends (the C copy was removed, ARCH P12).

        Δd_k = Σ_i b·Λ·e / Σ_i b·Λ   (per tilt k; skipped if Σ b·Λ < 1e-30)
        σ    = sqrt( Σ b·Λ·e² / (n_angles·Ny·N_det) ), floored at sigma_min

    with b the Huber surrogate weight of r = e·√Λ/σ, recomputed after the
    offset update. Element products are fp32; the sums accumulate in fp64
    on both backends (large M·K would otherwise lose digits).

    Returns (d_offsets_new, sigma_new) and mutates e_sino in place to
    maintain the invariant e = g - A·f - d.
    """
    if not (estimate_offset or estimate_variance):
        return d_offsets, sigma
    n_angles, Ny, N_det = e_sino.shape
    if lambdas is None:
        r = e_sino / sigma
    else:
        r = e_sino * lambdas.sqrt() / sigma
    r_abs = r.abs()
    bw = _huber_surr_weight(r_abs, T, delta)
    lam = lambdas if lambdas is not None else torch.ones_like(e_sino)

    if estimate_offset:
        num = (bw * lam * e_sino).sum(dim=(1, 2), dtype=torch.float64)
        den = (bw * lam).sum(dim=(1, 2), dtype=torch.float64)
        delta_d = torch.where(den >= _MBIR_WEIGHT_EPS, num / den.clamp(min=_MBIR_WEIGHT_EPS),
                              torch.zeros_like(num)).float()
        d_offsets = d_offsets + delta_d
        e_sino.sub_(delta_d.view(n_angles, 1, 1))
        # recompute weights for the variance update on the updated residual
        if estimate_variance:
            if lambdas is None:
                r = e_sino / sigma
            else:
                r = e_sino * lambdas.sqrt() / sigma
            r_abs = r.abs()
            bw = _huber_surr_weight(r_abs, T, delta)

    sigma_new = sigma
    if estimate_variance:
        total = n_angles * Ny * N_det
        s2 = (bw * lam * e_sino * e_sino).sum(dtype=torch.float64) / total
        sigma_new = max(sigma_min, float(s2.clamp(min=0.0).sqrt().float().item()))

    return d_offsets, sigma_new


def _generate_permutation(Ny: int, Nvox: int, rng: np.random.Generator) -> np.ndarray:
    """Per-Y-slice independent random permutation, returned as int32
    [Ny, Nvox]. Distinct shuffles across slices match the paper's
    randomized ICD ordering. Reproducible for a given seeded rng."""
    perm = np.empty((Ny, Nvox), dtype=np.int32)
    for iy in range(Ny):
        perm[iy] = rng.permutation(Nvox).astype(np.int32, copy=False)
    return perm


def _mbir_forward_project(
    volume: torch.Tensor,
    N_det: int,
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
    num_threads: int = 1,
) -> torch.Tensor:
    """A_MBIR · f: the voxel-driven scatter FP that matches the column-A
    model in the ICD kernel. Returns ``[n_angles, Ny, N_det]`` fp32."""
    Ny, Nz, Nx = volume.shape
    n_angles = cos_a.shape[0]
    Af = torch.empty(n_angles, Ny, N_det, dtype=torch.float32, device=volume.device)
    vol = volume.detach().float().contiguous()
    _launch("mbir_forward_project", Af,
            vol, Af, cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det,
            shapes=[("volume", (Ny, Nz, Nx)), ("Af", (n_angles, Ny, N_det)),
                    ("cos", (n_angles,)), ("sin", (n_angles,))],
            num_threads=num_threads)
    return Af


def _mbir_initial_residual(
    volume: torch.Tensor,
    sinogram: torch.Tensor,
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
    num_threads: int = 1,
) -> torch.Tensor:
    """e = g - A_MBIR · f at the start of an MBIR level. Falls back to
    `e = g` when `volume` is exactly zero (a common warm-start case) to
    skip a kernel launch."""
    if volume.abs().sum().item() == 0.0:
        return sinogram.clone()
    return sinogram - _mbir_forward_project(volume, sinogram.shape[2], cos_a, sin_a,
                                            num_threads)


def _mbir_icd_sweep_cpu(
    volume: torch.Tensor,             # [Ny, Nz, Nx] fp32, updated in place
    e_sino: torch.Tensor,             # [n_angles, Ny, N_det] fp32, updated in place
    lambdas: torch.Tensor | None,     # [n_angles, Ny, N_det] fp32
    vol_snap: torch.Tensor,           # [Ny, Nz, Nx] fp32
    perm: torch.Tensor,               # [Ny, Nz*Nx] int32
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
    *,
    qggmrf_p: float, qggmrf_q: float, qggmrf_c: float, sigma_f: float,
    huber_T: float, huber_delta: float, sigma: float, num_threads: int,
) -> None:
    """One CPU ICD sweep (voxel updates only; the offset/variance updates
    are `_mbir_update_offset_variance`, shared with CUDA). The C code updates
    volume / e_sino through raw pointers, so every buffer must be a dense
    CPU tensor of the right dtype: a silent copy would drop the in-place
    writes, so strided buffers raise instead (#58)."""
    if volume.ndim != 3 or e_sino.ndim != 3:
        raise ValueError("_mbir_icd_sweep_cpu: volume and e_sino must be 3-D")
    n_angles, Ny, N_det = e_sino.shape
    _, Nz, Nx = volume.shape
    for name, t, dt in (("volume", volume, torch.float32),
                        ("e_sino", e_sino, torch.float32),
                        ("volume_snapshot", vol_snap, torch.float32),
                        ("perm", perm, torch.int32),
                        ("lambdas", lambdas, torch.float32)):
        if t is not None and (t.dtype != dt or t.device.type != "cpu"):
            raise TypeError(f"_mbir_icd_sweep_cpu: {name} must be a CPU {dt} tensor")
    # One place for every size the C sweep trusts (#58 checks included).
    _check_shapes("_mbir_icd_sweep_cpu", [
        ("volume", volume, (Ny, Nz, Nx)),            # e_sino Ny must match volume
        ("e_sino", e_sino, (n_angles, Ny, N_det)),
        ("volume_snapshot", vol_snap, (Ny, Nz, Nx)),
        ("perm", perm, (Ny, Nz * Nx)),
        ("lambdas", lambdas, (n_angles, Ny, N_det)),
        ("cos_a (angles)", cos_a, (n_angles,)),
        ("sin_a (angles)", sin_a, (n_angles,)),
    ])
    _core.mbir_icd_sweep_cpu(
        vol_ptr=_data_ptr(volume, "volume"),
        e_sino_ptr=_data_ptr(e_sino, "e_sino"),
        vol_snapshot_ptr=_data_ptr(vol_snap, "volume_snapshot"),
        lambdas_ptr=_data_ptr(lambdas, "lambdas") if lambdas is not None else 0,
        perm_ptr=_data_ptr(perm, "perm"),
        cos_ptr=_data_ptr(cos_a, "cos_a"), sin_ptr=_data_ptr(sin_a, "sin_a"),
        Ny=Ny, Nz=Nz, Nx=Nx, n_angles=n_angles, N_det=N_det,
        qggmrf_p=qggmrf_p, qggmrf_q=qggmrf_q, qggmrf_c=qggmrf_c, sigma_f=sigma_f,
        huber_T=huber_T, huber_delta=huber_delta, sigma=sigma,
        num_threads=num_threads,
    )


def _run_mbir_single_scale(
    sinogram: torch.Tensor,       # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,     # [n_angles]
    *,
    sigma_f: float,
    qggmrf_p: float,
    qggmrf_q: float,
    qggmrf_c: float,
    huber_T: float,
    huber_delta: float,
    sigma: float,
    sigma_min: float,
    estimate_offset: bool,
    estimate_variance: bool,
    outer_iters: int,
    convergence_tol: float,
    seed: int,
    Nz: int,
    Nx: int,
    init_state: _MbirState | None,
    lambdas: torch.Tensor | None,
    return_history: bool,
    rec: _LossRecorder,
    it_offset: int = 0,
    num_threads: int = 1,
    support: torch.Tensor | None = None,
) -> tuple[_MbirState, list[float]]:
    """Inner loop for one resolution scale. The public `run_mbir` wraps
    this in a coarse-to-fine multiresolution loop.

    Convergence is measured by the relative L2 change in the volume
    between successive outer iterations. The check uses every-N-th voxel
    sampling to keep the comparison cheap relative to the kernel.

    ``rec`` records, after every outer iteration, the data MSE
    ``mean(e²)`` of the residual ``e = g - A·f - d`` at this scale, under
    the global iteration index ``it_offset + it``.

    ``support`` (``[1|Ny, Nz, Nx]`` on this scale's grid) multiplies the
    initial volume and the volume after every ICD sweep. The part it
    removes, ``r = f·(1 - support)``, is forward-projected and added back
    to the error sinogram (``e += A·r``), so the invariant
    ``e = g - A·f - d`` holds for the masked volume before the offset and
    variance updates and the next sweep read it."""
    device = sinogram.device
    n_angles, Ny, N_det = sinogram.shape

    angles_rad = (angles_deg.float().to(device) * (math.pi / 180.0)).contiguous()
    cos_a = torch.cos(angles_rad).contiguous()
    sin_a = torch.sin(angles_rad).contiguous()

    if init_state is None:
        volume = torch.zeros((Ny, Nz, Nx), device=device, dtype=torch.float32)
        d_offsets = torch.zeros(n_angles, device=device, dtype=torch.float32)
    else:
        volume = init_state.volume.clone().contiguous()
        d_offsets = init_state.d_offsets.clone().contiguous()
    if support is not None:
        volume.mul_(support)

    # Residual e = g - A·f - d. d is rolled into A·f via the offset
    # term (we maintain e ≡ g - A·f - d, so initial e adjusts for both).
    e_sino = _mbir_initial_residual(volume, sinogram, cos_a, sin_a, num_threads).contiguous()
    if d_offsets.abs().sum().item() > 0.0:
        e_sino.sub_(d_offsets.view(n_angles, 1, 1))

    rng = np.random.default_rng(seed)
    Nvox = Nz * Nx
    history: list[float] = []
    sigma_cur = float(sigma)
    n_done = 0

    for it in range(outer_iters):
        n_done = it + 1
        vol_snap = volume.clone().contiguous()
        perm = _generate_permutation(Ny, Nvox, rng)

        if device.type == "cuda":
            perm_dev = torch.from_numpy(perm).to(device).contiguous()
            stream_ptr = torch.cuda.current_stream(device).cuda_stream
            with torch.cuda.device(device):   # D7: launch on the data's device
                _core.mbir_icd_sweep_cuda(
                    vol_ptr=volume.data_ptr(),
                    e_sino_ptr=e_sino.data_ptr(),
                    vol_snapshot_ptr=vol_snap.data_ptr(),
                    lambdas_ptr=lambdas.data_ptr() if lambdas is not None else 0,
                    perm_ptr=perm_dev.data_ptr(),
                    cos_ptr=cos_a.data_ptr(),
                    sin_ptr=sin_a.data_ptr(),
                    Ny=Ny, Nz=Nz, Nx=Nx, n_angles=n_angles, N_det=N_det,
                    qggmrf_p=qggmrf_p, qggmrf_q=qggmrf_q,
                    qggmrf_c=qggmrf_c, sigma_f=sigma_f,
                    huber_T=huber_T, huber_delta=huber_delta,
                    sigma=sigma_cur,
                    stream_ptr=stream_ptr,
                )
        else:
            _mbir_icd_sweep_cpu(
                volume, e_sino,
                lambdas.contiguous() if lambdas is not None else None,
                vol_snap, torch.from_numpy(perm), cos_a, sin_a,
                qggmrf_p=qggmrf_p, qggmrf_q=qggmrf_q, qggmrf_c=qggmrf_c,
                sigma_f=sigma_f, huber_T=huber_T, huber_delta=huber_delta,
                sigma=sigma_cur, num_threads=num_threads,
            )
        if support is not None:
            # Hard support after the sweep; keep e = g - A·f - d consistent.
            removed = volume * (1.0 - support)
            volume.sub_(removed)
            e_sino.add_(_mbir_forward_project(removed, N_det, cos_a, sin_a, num_threads))
        # Closed-form updates for d_k and σ: one implementation, both backends.
        d_offsets, sigma_cur = _mbir_update_offset_variance(
            e_sino, d_offsets, sigma_cur, sigma_min,
            huber_T, huber_delta, lambdas,
            estimate_offset, estimate_variance,
        )
        if rec.wants_loss:
            rec.record(it_offset + it, e_sino.pow(2).mean())
        rec.tick()

        # Convergence check: relative L2 change on a strided sample so it
        # scales as O(Nvox / 8). Cheap vs the per-iter compute.
        with torch.no_grad():
            stride = _MBIR_CONVERGENCE_STRIDE
            f_new = volume.flatten()[::stride]
            f_old = vol_snap.flatten()[::stride]
            denom = f_old.norm().clamp(min=_REL_EPS)
            rel_change = float((f_new - f_old).norm() / denom)
        if return_history:
            history.append(rel_change)
        if rel_change < convergence_tol:
            break

    state = _MbirState(volume=volume, d_offsets=d_offsets, sigma=sigma_cur,
                       e_sino=e_sino, n_done=n_done)
    return state, history


def _downsample_sino_det(sino: torch.Tensor, factor: int) -> torch.Tensor:
    """Avg-pool the sinogram in the detector axis only. Y (tilt-axis row)
    is preserved because each Y-slice is an independent MBIR sub-problem
    under single-axis parallel-beam geometry."""
    if factor == 1:
        return sino
    K, Y, M = sino.shape
    s = sino.reshape(K * Y, 1, M)
    s = F.avg_pool1d(s, kernel_size=factor, stride=factor)
    return s.reshape(K, Y, s.shape[-1])


def _resize_volume_zx(vol: torch.Tensor, target_nz: int, target_nx: int) -> torch.Tensor:
    """Trilinear-style resize in (Z, X) only. Y is preserved."""
    Ny, Nz, Nx = vol.shape
    if Nz == target_nz and Nx == target_nx:
        return vol
    v = vol.unsqueeze(0).unsqueeze(0)              # [1,1,Ny,Nz,Nx]
    v = F.interpolate(
        v,
        size=(Ny, target_nz, target_nx),
        mode="trilinear",
        align_corners=False,
    )
    return v.squeeze(0).squeeze(0).contiguous()


def run_mbir(
    sinogram: torch.Tensor,           # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,         # [n_angles] degrees
    n_iter: int = 100,                # max outer iterations per scale
    *,
    Nz: int = -1,
    sigma_f: float = 1.0,
    qggmrf_p: float = 1.2,
    qggmrf_q: float = 2.0,
    qggmrf_c: float = 1e-3,
    huber_T: float = float("inf"),
    huber_delta: float = 0.5,
    estimate_offset: bool = True,
    estimate_variance: bool = True,
    sigma_init: float | None = None,
    sigma_min: float | None = None,
    multires_levels: int = 1,
    convergence_tol: float = 1e-4,
    seed: int = 0,
    lambdas: torch.Tensor | None = None,
    sigma_f_per_scale: list[float] | None = None,
    config: MbirConfig | None = None,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    warm_start: torch.Tensor | None = None,
    num_threads: int | None = None,
    return_loss: bool = False,
    return_state: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
    **deprecated: Any,
) -> torch.Tensor | tuple:
    """Classical MBIR (Model-Based Iterative Reconstruction) for single-axis
    parallel-beam tomography.

    Implements MBIR-AM-PE (Venkatakrishnan et al. 2015): joint MAP
    reconstruction of the volume `f`, per-tilt offset `d_k`, and noise
    variance `σ` under a generalized-Huber likelihood + qGGMRF prior.

    Parameters
    ----------
    sinogram : [n_angles, Ny, N_det] float32 tensor
    angles_deg : [n_angles] float32 tensor, degrees
    Nz : reconstruction depth (default = N_det)
    sigma_f, qggmrf_p, qggmrf_q, qggmrf_c : qGGMRF prior parameters.
        Defaults `p=1.2, q=2.0, c=1e-3` follow the paper. σ_f sets the
        prior strength.
    huber_T : threshold for the generalized Huber likelihood. `inf`
        (default) recovers conventional MBIR (no anomaly modeling).
        `T=3` is the paper's recommendation for BF-TEM with Bragg scatter.
    huber_delta : slope of the Huber tail. `δ=0.5` is the recommendation.
    estimate_offset, estimate_variance : enable closed-form updates of
        `d_k` and `σ` between outer iterations (MBIR-AM → MBIR-AM-PE).
    sigma_init : initial noise std. Defaults to the std of the sinogram.
    sigma_min : lower clamp on σ to prevent collapse (default
        `sigma_init * 1e-3`).
    multires_levels : coarse-to-fine levels. 1 = single-scale (default).
    outer_iters : maximum outer iterations per scale.
    convergence_tol : stop when the relative L2 change in the volume
        between iterations falls below this.
    seed : RNG seed for the per-Y-slice voxel permutation.
    lambdas : optional [n_angles, Ny, N_det] per-pixel data weights (Λ).
        `None` → uniform 1 (standard Poisson approximation).
    sigma_f_per_scale : optional list of σ_f overrides per multires level,
        coarse → fine. If None, `σ_f * 2**level` is used (prior weight
        scales with voxel size).
    return_history : if True, also returns the per-iteration convergence
        history (relative L2 change).

    Returns
    -------
    volume : [Ny, Nz, Nx] float32 tensor.
    If `return_history=True`, also returns (d_offsets, sigma, history).
    """
    n_iter = _deprecated_kwarg(deprecated, "outer_iters", "n_iter", n_iter, "run_mbir",
                               default=100)
    return_history = deprecated.pop("return_history", False)
    _reject_unknown_kwargs(deprecated, "run_mbir")
    if return_history:
        if return_loss or return_state:
            raise TypeError(
                "run_mbir: return_history (deprecated) cannot be combined with "
                "return_loss or return_state."
            )
        warnings.warn(
            "run_mbir: return_history is deprecated; use return_loss=True for the "
            "loss history and return_state=True for (d_offsets, sigma). It will "
            "be removed in the next release.",
            DeprecationWarning, stacklevel=2,
        )
    if config is not None:
        if not isinstance(config, MbirConfig):
            raise TypeError(f"run_mbir: config must be an MbirConfig, "
                            f"got {type(config).__name__}")
        tuning = dict(
            sigma_f=sigma_f, qggmrf_p=qggmrf_p, qggmrf_q=qggmrf_q, qggmrf_c=qggmrf_c,
            huber_T=huber_T, huber_delta=huber_delta, estimate_offset=estimate_offset,
            estimate_variance=estimate_variance, sigma_init=sigma_init,
            sigma_min=sigma_min, multires_levels=multires_levels,
            convergence_tol=convergence_tol, seed=seed,
            sigma_f_per_scale=sigma_f_per_scale,
        )
        defaults = MbirConfig()
        clash = [k for k, v in tuning.items() if not _is_default(v, getattr(defaults, k))]
        if clash:
            raise TypeError(
                f"run_mbir: pass the tuning options either through config or as "
                f"keywords, not both (got config and {', '.join(clash)})."
            )
        (sigma_f, qggmrf_p, qggmrf_q, qggmrf_c, huber_T, huber_delta,
         estimate_offset, estimate_variance, sigma_init, sigma_min,
         multires_levels, convergence_tol, seed, sigma_f_per_scale) = (
            getattr(config, k) for k in tuning)
    if multires_levels < 1:
        raise ValueError("multires_levels must be >= 1")
    # fp32 only, like every other fp32 algorithm: other dtypes raise instead
    # of being cast silently. The ICD kernels are fp32-only.
    setup = _prepare_recon(
        "run_mbir", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        init_volume=False, warm_start=warm_start, mask=mask, support_mask=support_mask,
    )
    support = setup.constraints.support_mask     # [1|Ny, Nz, Nx] fp32 or None
    n_angles, Ny, N_det, Nz, Nx = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx)
    n_iter = setup.n_iter
    nt = _resolve_num_threads(num_threads)
    multires_levels = _as_int(multires_levels, "multires_levels", "run_mbir", minimum=1)
    angles_deg = angles_deg.float()
    # A NaN/inf pixel would otherwise surface as "sigma_init must be finite"
    # (sinogram.std() is NaN) even though the caller never passed sigma_init,
    # or poison every voxel if sigma_init is given.
    if not torch.isfinite(sinogram).all():
        raise ValueError("run_mbir: sinogram contains NaN or inf values")

    if lambdas is not None:
        if lambdas.dim() != 3 or tuple(lambdas.shape) != (n_angles, Ny, N_det):
            raise ValueError(
                "lambdas must match sinogram shape "
                f"[n_angles, Ny, N_det]={sinogram.shape}, got {tuple(lambdas.shape)}"
            )
        _assert_dtype_match(sinogram, lambdas, "sinogram", "lambdas", "run_mbir")
        if lambdas.device != sinogram.device:
            lambdas = lambdas.to(sinogram.device)
        # The MBIR sweeps read lambdas through a raw linear pointer (CUDA) or
        # a numpy buffer (CPU), so a strided view such as one produced by
        # ``expand`` would be indexed as if it were dense. Materialise it.
        lambdas = lambdas.contiguous()
        # Negative weights make lambdas.sqrt() NaN inside the variance update;
        # non-finite weights poison every voxel they touch.
        if not torch.isfinite(lambdas).all():
            raise ValueError("run_mbir: lambdas must contain only finite values")
        if (lambdas < 0).any():
            raise ValueError("run_mbir: lambdas must be non-negative")

    if sigma_init is None:
        sigma_init = float(sinogram.std().item())
        if sigma_init < _MBIR_BLANK_STD:
            sigma_init = 1.0
    # sigma divides the data term, so zero or negative values blow up.
    if not math.isfinite(float(sigma_init)) or float(sigma_init) <= 0.0:
        raise ValueError(f"run_mbir: sigma_init must be finite and > 0, got {sigma_init}")
    if sigma_min is None:
        sigma_min = sigma_init * _MBIR_SIGMA_MIN_RATIO
    if not math.isfinite(float(sigma_min)) or float(sigma_min) <= 0.0:
        raise ValueError(f"run_mbir: sigma_min must be finite and > 0, got {sigma_min}")

    # Multires schedule: factors = [2**(L-1), ..., 2, 1] for L levels.
    factors = [2 ** (multires_levels - 1 - lvl) for lvl in range(multires_levels)]
    coarse_factor = factors[0]
    if coarse_factor > 1 and (
        N_det % coarse_factor != 0
        or Nx % coarse_factor != 0
        or Nz % coarse_factor != 0
    ):
        raise ValueError(
            "run_mbir multires_levels requires detector and reconstruction "
            f"dims divisible by {coarse_factor}; got N_det={N_det}, "
            f"Nx={Nx}, Nz={Nz}"
        )
    if sigma_f_per_scale is None:
        sigma_f_per_scale = [sigma_f * float(f) for f in factors]
    elif len(sigma_f_per_scale) != multires_levels:
        raise ValueError(
            "sigma_f_per_scale must have length multires_levels"
        )

    history_all: list[list[float]] = []
    state: _MbirState | None = None
    if setup.volume0 is not None:
        # warm start: resized to the coarsest grid; offsets start at zero
        state = _MbirState(
            volume=_resize_volume_zx(setup.volume0, max(Nz // coarse_factor, 1),
                                     max(Nx // coarse_factor, 1)),
            d_offsets=torch.zeros(n_angles, device=sinogram.device, dtype=torch.float32),
            sigma=float(sigma_init), e_sino=None,
        )
    rec = _LossRecorder(return_loss, loss_callback, progress,
                        n_iter * multires_levels, desc="run_mbir")
    it_offset = 0

    for lvl, f in enumerate(factors):
        sino_lvl = _downsample_sino_det(sinogram, f)
        # Scale lambdas the same way if provided.
        if lambdas is not None and f > 1:
            lam_lvl = _downsample_sino_det(lambdas, f)
        else:
            lam_lvl = lambdas

        Nz_lvl = max(Nz // f, 1)
        Nx_lvl = max(Nx // f, 1)
        # The support on this level's grid: a coarse voxel keeps the largest
        # weight of the f x f fine voxels it covers (max-pool; f divides Nz
        # and Nx, checked above), so a coarse level never cuts into the
        # support. Exact at f = 1.
        support_lvl = (support if support is None or f == 1
                       else F.max_pool2d(support, kernel_size=f, stride=f))

        # First iteration at the coarsest scale: force T=inf so the
        # anomaly modelling is disabled until the next finer scale. This
        # matches the paper's multires + AM-PE bootstrap.
        if multires_levels > 1 and lvl == 0:
            level_T = float("inf")
        else:
            level_T = huber_T

        # Carry d_offsets across scales unchanged; re-initialize σ from
        # the current sinogram's std (intensity statistics shift with
        # the detector binning).
        if lvl == 0:
            level_sigma = sigma_init
        else:
            level_sigma = max(sigma_min, float(sino_lvl.std().item()))
            state.sigma = level_sigma

        state, hist = _run_mbir_single_scale(
            sino_lvl, angles_deg,
            sigma_f=sigma_f_per_scale[lvl],
            qggmrf_p=qggmrf_p, qggmrf_q=qggmrf_q, qggmrf_c=qggmrf_c,
            huber_T=level_T, huber_delta=huber_delta,
            sigma=level_sigma, sigma_min=sigma_min,
            estimate_offset=estimate_offset,
            estimate_variance=estimate_variance,
            outer_iters=n_iter,
            convergence_tol=convergence_tol,
            seed=seed + lvl,
            Nz=Nz_lvl, Nx=Nx_lvl,
            init_state=state,
            lambdas=lam_lvl,
            return_history=return_history,
            rec=rec, it_offset=it_offset, num_threads=nt, support=support_lvl,
        )
        history_all.append(hist)
        it_offset += state.n_done

        # Carry the current solution to the next scale with an exact resize.
        # Using a fixed ×2 upsample breaks odd detector sizes (e.g. 65 → 32 → 64),
        # causing shape mismatches at the next level.
        if lvl + 1 < len(factors):
            next_f = factors[lvl + 1]
            next_nz = max(Nz // next_f, 1)
            next_nx = max(Nx // next_f, 1)
            state.volume = _resize_volume_zx(state.volume, next_nz, next_nx)

    rec.close()
    if return_history:
        return state.volume, state.d_offsets, state.sigma, history_all
    result: list = [state.volume]
    if return_loss:
        result.append(rec.history())
    if return_state:
        result += [state.d_offsets, state.sigma]
    return result[0] if len(result) == 1 else tuple(result)
