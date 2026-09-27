"""
Shared machinery of the reconstruction algorithms.

Module constants, the reconstruction prologue `_prepare_recon` (ARCH P5),
the loss recorder `_LossRecorder` (ARCH P6, owns the frozen 2026-06-11
``loss_callback`` contract), deprecated-keyword handling and the FP/BP
closure builder `_build_fp_bp` shared by SIRT and FISTA (a projection
operator from `fasttomo._operator._make_operator` plus two closures). The conventions
every ``run_*`` follows are documented in :mod:`fasttomo.algorithms`.
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

import torch
from tqdm.auto import tqdm

from fasttomo._backend import _assert_dtype_match, _check_finite_geometry
from fasttomo._operator import Geometry, _make_operator
from fasttomo.autograd import euler_zyz_to_rotmat
from fasttomo.constraints import (
    _maybe_cylindrical_mask,
    _resolve_support_mask,
)
from fasttomo.configs import ConstraintConfig


# Named explicitly so the logger keeps the pre-split name
# ("fasttomo.algorithms") that users may have configured.
_log = logging.getLogger("fasttomo.algorithms")


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

# W-BP: absolute floor of the weighting filter's missing-wedge threshold, so
# a degenerate transfer function (all ~0) never divides by ~0.
_WBP_MIN_THRESHOLD = 1e-6

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


def _reciprocal_or_zero(x: torch.Tensor) -> torch.Tensor:
    """``1 / x`` where ``x > 0``, else 0: the SIRT / SART normalisers
    ``1 / BP(1)`` and ``1 / FP(1)``, in ``x``'s dtype.

    fp16 also zeroes the entries whose reciprocal would overflow (``x`` below
    ``1 / 65504``). Those are rays that graze the volume (a path length of
    ~1e-5 voxels, which the CPU rotation-matrix projector produces with its
    exact interpolation weights); fp32 gives them a finite weight of ~1e5, the
    fp16 division gave ``inf``, and ``residual * inf`` is ``inf`` or NaN.
    A zero weight drops those rays from the update; for consistent data their
    residual is ~1e-5 of a voxel value anyway. fp32 is unchanged (``x > 0``).
    """
    floor = 1.0 / torch.finfo(torch.float16).max if x.dtype == torch.float16 else 0.0
    return torch.where(x > floor, 1.0 / x, torch.zeros_like(x))


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
                      sinogram.shape[0] != len(angles_deg), or host-side
                      angles_deg holds a NaN or Inf.
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
        _check_finite_geometry(angles_deg, f"{op_name}: angles_deg")   # host angles only

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
    Nz: int | None = None,
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
      accepted), ``n_iter >= 0``; ``Nz=None`` (or any negative value, the
      former ``-1`` default) means ``N_det``. ``Nx = N_det``.
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
    # None (the default) or any negative value: Nz = N_det. -1 is what the
    # raw kernel layer uses for "from the input"; None is the Python API's.
    Nz = -1 if Nz is None else _as_int(Nz, "Nz", op_name, minimum=None)
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


class _DeviceLossRecorder(_LossRecorder):
    """`_LossRecorder` that keeps tensor losses on the device (AD audit
    A6-8, solver audit P5).

    Same values and callback contract, fewer host syncs per iteration:

    * ``return_loss`` alone: each loss is copied into a device buffer (no
      sync); `history` / `finish` move it to the host once, at the end.
    * with a ``loss_callback``: the total and its components go to the host
      in one transfer (one sync instead of one per value), and the callback
      still gets plain floats for the true iteration index. The history is
      then kept from those floats.

    Values are the losses the base class records: an fp32 or fp16 0-d
    tensor is widened to fp32 exactly, and ``tolist`` / ``item`` give the
    same Python float for it. Other values (Python floats, fp64 tensors,
    mixed devices) fall back to the base class.
    """

    def __init__(self, return_loss, loss_callback, progress, n_iter, *, desc=None) -> None:
        super().__init__(return_loss, loss_callback, progress, n_iter, desc=desc)
        self._n_iter = max(int(n_iter), 1)
        self._buf: torch.Tensor | None = None
        self._n = 0

    def record(self, it, loss, components=None) -> None:
        if not self.wants_loss:
            return
        values = [loss]
        if self.send_components and components is not None:
            values += list(components.values())
        if not all(isinstance(v, torch.Tensor) and v.numel() == 1
                   and v.dtype in (torch.float32, torch.float16)
                   and v.device == loss.device for v in values):
            self._flush()
            super().record(it, loss, components)
            return
        if self.loss_callback is None:
            if self._buf is not None and self._n == self._buf.numel():
                self._flush()
            if self._buf is None:
                self._buf = torch.empty(self._n_iter, dtype=torch.float32,
                                        device=loss.device)
            self._buf[self._n].copy_(loss.detach().reshape(()))
            self._n += 1
            return
        host = torch.stack([v.detach().reshape(()).float() for v in values]).tolist()
        loss_val = host[0]
        if self.return_loss:
            self.losses.append(loss_val)
        comps = None
        if self.send_components:
            comps = ({"fidelity": loss_val} if components is None
                     else dict(zip(components.keys(), host[1:])))
        _emit_loss(self.loss_callback, self.send_components, it, loss_val, comps)

    def _flush(self) -> None:
        if self._buf is not None:
            self.losses.extend(self._buf[:self._n].tolist())
            self._buf = None
            self._n = 0

    def history(self) -> torch.Tensor:
        self._flush()
        return super().history()

    def finish(self, volume):
        self._flush()
        return super().finish(volume)


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
    """Build ``(_fp, _bp, op)`` for an iterative algorithm.

    ``op`` is the reconstruction's projection operator (`_make_operator`):
    the caller's ``projector`` if given, else one for this geometry (CUDA:
    cached textures reused by every iteration; CPU: the OpenMP kernels).
    ``euler_zyz_deg`` selects the rotation-matrix geometry; its matrices are
    computed once. ``_fp``/``_bp`` are differentiable. Hold ``op`` until the
    algorithm finishes: it owns the textures and ``run_sirt``'s CUDA graphs.
    """
    rotmat = euler_zyz_deg is not None
    op = _make_operator(
        Geometry("rotmat" if rotmat else "parallel", vol_shape, N_det, n_angles,
                 None if rotmat else angles_deg),
        device, sinogram.dtype, projector, num_threads=num_threads)
    geom = euler_zyz_to_rotmat(euler_zyz_deg).to(device=device) if rotmat else None

    def _fp(v):
        return op.project(v, geom)

    def _bp(s):
        return op.backproject(s, geom)
    return _fp, _bp, op
