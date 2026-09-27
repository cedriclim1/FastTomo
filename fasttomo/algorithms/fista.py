"""
FISTA (Beck & Teboulle 2009) and FISTA-TV, with the Lipschitz estimate and
the 3-D TV proximal operator.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

import fasttomo._core as _core
from fasttomo._backend import _fp16_overflow_checked, _stream_ptr
from fasttomo.algorithms._common import (
    _LIPSCHITZ_SAFETY,
    _NORM_EPS,
    _TV_DUAL_STEP,
    _as_int,
    _deprecated_kwarg,
    _reject_unknown_kwargs,
    _prepare_recon,
    _DeviceLossRecorder,
    _LossRecorder,
    _build_fp_bp,
)


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

    The norms are fp32 whatever the volume dtype: an fp16 norm overflows to
    inf once L exceeds 65504 (parallel beam: L ~ n_views x N, so a 512-pixel
    detector with 180 views), which zeroed the next iterate and made
    ``1 / L`` raise ZeroDivisionError. L stays on the device between steps
    (one host sync at the end instead of one per step); the normalisation
    by ``L + eps`` rounds exactly as the former division by the Python float
    did (`_normalise`), so fp32 runs are unchanged bit for bit.
    """
    v = torch.randn(vol_shape, device=device, dtype=dtype)
    if dtype == torch.float32:
        v = v / (v.norm() + _NORM_EPS)
    else:
        v = (v.float() / (torch.linalg.vector_norm(v, dtype=torch.float32) + _NORM_EPS)).to(dtype)
    nrm = None
    for _ in range(n_iter):
        AtAv = _bp(_fp(v))
        nrm = torch.linalg.vector_norm(AtAv, dtype=torch.float32)
        v = _normalise(AtAv, nrm)
    L = 0.0 if nrm is None else float(nrm)
    return L * _LIPSCHITZ_SAFETY


def _normalise(t: torch.Tensor, nrm: torch.Tensor) -> torch.Tensor:
    """``t / (float(nrm) + _NORM_EPS)`` for a 0-d fp32 device tensor
    ``nrm``, without a host sync and rounded as ATen rounds that division by
    a Python float: the divisor ``d = double(nrm) + eps`` is formed in fp64;
    on CUDA ATen multiplies by ``fp32(1.0 / d)`` (reciprocal in double,
    rounded once), on CPU it divides by ``fp32(d)`` (checked on torch 2.14
    over random divisors, fp32 and fp16). fp16 ``t`` is computed in fp32 and
    rounded once (casting the divisor or its reciprocal to fp16 first would
    lose it to the subnormal range)."""
    d = nrm.double() + _NORM_EPS
    t32 = t if t.dtype == torch.float32 else t.float()
    out = t32 * torch.reciprocal(d).float() if t.is_cuda else t32 / d.float()
    return out if t.dtype == torch.float32 else out.to(t.dtype)


def _fista_texture_operator(op, sinogram: torch.Tensor):
    """The cached CUDA operator whose volume texture the fused FISTA update
    may write (`algebraic._texture_operator`), or None."""
    if not sinogram.is_cuda:
        return None
    from fasttomo.algorithms.algebraic import _texture_operator
    return _texture_operator(op)


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

    On CUDA (fp32 / fp16) this runs as the two kernels per inner iteration
    of ``csrc/cuda/tv_prox.cu`` (`_tv_prox_3d_cuda`), bit for bit the
    PyTorch code below, which stays the CPU path and the reference.
    """
    if mode not in ("fgp", "chambolle"):
        raise ValueError(
            f"_tv_prox_3d: unknown mode {mode!r}; expected 'fgp' or 'chambolle'."
        )
    out = _tv_prox_3d_cuda(z, lam, positivity, mask, n_iter, mode)
    if out is not None:
        return out
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


def _fgp_betas(n_iter: int) -> list[float]:
    """The FGP momenta ``(t_k - 1) / t_{k+1}`` of `_tv_prox_3d`'s loop."""
    t, betas = 1.0, []
    for _ in range(n_iter):
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        betas.append((t - 1.0) / t_new)
        t = t_new
    return betas


def _tv_prox_3d_cuda(
    z: torch.Tensor,
    lam: float,
    positivity: bool,
    mask: torch.Tensor | None,
    n_iter: int,
    mode: str,
) -> torch.Tensor | None:
    """`_tv_prox_3d` as the fused CUDA kernels (solver audit P1), or None
    when not applicable (CPU, another dtype, an unexpected mask shape): the
    caller then runs the PyTorch reference. Memory: the output plus 6 (FGP)
    or 3 (Chambolle) dual volumes, no per-iteration allocation."""
    if not (z.is_cuda and z.dtype in (torch.float32, torch.float16) and z.ndim == 3
            and hasattr(_core, "tv_prox_cuda")):
        return None
    Ny, Nz_, Nx = z.shape
    if mask is not None:
        if not (mask.device == z.device and mask.dtype == z.dtype and mask.ndim == 3
                and tuple(mask.shape[1:]) == (Nz_, Nx) and mask.shape[0] in (1, Ny)):
            return None
        mask = mask.contiguous()
    z = z.contiguous()
    fgp = mode == "fgp"
    v = torch.empty_like(z)
    work = torch.empty((6 if fgp else 3) * z.numel(), dtype=z.dtype, device=z.device)
    betas = torch.tensor(_fgp_betas(n_iter) if fgp else [0.0] * n_iter, dtype=torch.float32)
    # CUDA ATen divides by a Python scalar as a product with its reciprocal
    # taken in double and rounded once to fp32 (torch 2.14: fp32(1) /
    # fp32(lam) differs for e.g. lam = 1e-3); pinned by
    # test_tv_prox_cuda_bit_identical
    inv_lam = float(np.float32(1.0 / float(lam)))
    with torch.cuda.device(z.device):
        _core.tv_prox_cuda(
            0 if z.dtype == torch.float32 else 1, int(fgp),
            z.data_ptr(), v.data_ptr(), work.data_ptr(),
            0 if mask is None else mask.data_ptr(),
            int(mask is not None and mask.shape[0] == Ny and Ny > 1),
            int(bool(positivity)), float(_TV_DUAL_STEP), inv_lam,
            betas.data_ptr(), int(n_iter), Ny, Nz_, Nx, _stream_ptr(z.device))
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
    *,
    constraints=None,
    tex_op=None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Shared Beck-Teboulle FISTA outer loop.

    ``prox_fn(u)`` is called once per iteration on ``u = y_k - (1/L) ∇f(y_k)``
    and must return the next iterate ``x_{k+1}``. For pure FISTA it's just the
    constraint projection; for FISTA-TV it's the TV+constraint prox.

    The recorded loss is ``||b − A·y_k||²`` (data-fit MSE at the momentum
    point, which is free) at each iteration — a tight surrogate for the loss
    at ``x_k`` — and it is the data-fit term only (components
    ``{"fidelity": loss}``), regardless of prox regularisation.

    On CUDA the elementwise steps run as the fused kernels of
    ``fista_update.cu`` (solver audit P4), bit for bit the PyTorch chain:
    with ``constraints`` (the caller's ``prox_fn`` is exactly
    ``constraints.apply``) the step, the constraints and the momentum are one
    kernel updating ``x_prev`` and ``y`` in place; otherwise the step and
    the momentum around ``prox_fn`` are one kernel each. The new ``y`` also
    goes into ``tex_op``'s volume texture when possible, so the next forward
    projection skips its upload.
    """
    inv_L = 1.0 / L
    t = 1.0
    x_prev = volume.clone()
    y = volume.clone()
    try:
        for it in rec.iterate(n_iter):
            fp_y = _fp(y)
            res = fp_y - sinogram
            if rec.wants_loss:
                # mean((b - A y)^2): the same bits as squaring -res
                rec.record(it, res.pow(2).mean())
            grad = _bp(res)
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new
            if constraints is not None and _fista_update(
                    0, y, grad, x_prev, x_prev, y, inv_L, beta, constraints, tex_op):
                pass                                    # x_prev <- x_new, y <- y_new
            else:
                u = _fista_step(y, grad, inv_L)
                x_new = prox_fn(u)
                if not _fista_update(2, x_new, None, x_prev, None, y, inv_L, beta,
                                     None, tex_op):
                    y = x_new + beta * (x_new - x_prev)
                x_prev = x_new
            t = t_new
    finally:
        if tex_op is not None:
            tex_op._forget_texture_writes()
    return rec.finish(x_prev)


def _fista_step(y: torch.Tensor, grad: torch.Tensor, inv_L: float) -> torch.Tensor:
    """``y - inv_L * grad`` (fused on CUDA, same bits)."""
    u = torch.empty_like(y, memory_format=torch.contiguous_format)
    if _fista_update(1, y, grad, None, u, None, inv_L, 0.0, None, None):
        return u
    return y - inv_L * grad


def _fista_update(mode: int, a, grad, xp, x_out, y_out, inv_L: float, beta: float,
                  constraints, tex_op) -> bool:
    """Run ``fista_update_cuda`` (modes of ``csrc/cuda/fista_update.cu``) and
    return True, or return False when the tensors are not fusable (CPU, a
    duck-typed projector's dtype, non-contiguous, a soft threshold, an
    unexpected mask shape) so the caller runs the PyTorch chain. ``y_out``
    (modes 0, 2) is also written into ``tex_op``'s volume texture when
    possible."""
    ts = [t for t in (a, grad, xp, x_out, y_out) if t is not None]
    t0 = ts[0]
    if not (t0.is_cuda and t0.dtype in (torch.float32, torch.float16) and t0.ndim == 3
            and hasattr(_core, "fista_update_cuda")
            and all(t.device == t0.device and t.dtype == t0.dtype and t.shape == t0.shape
                    and t.is_contiguous() for t in ts)):
        return False
    Ny, Nz, Nx = t0.shape
    mask = None
    positivity = False
    if constraints is not None:
        if constraints.soft_threshold > 0.0:
            return False
        positivity = bool(constraints.positivity)
        mask = constraints._support_for(t0)
        if mask is not None:
            if not (mask.is_cuda and mask.device == t0.device and mask.dtype == t0.dtype
                    and mask.ndim == 3 and tuple(mask.shape[1:]) == (Nz, Nx)
                    and mask.shape[0] in (1, Ny)):
                return False
            mask = mask.contiguous()
    handle = None
    if tex_op is not None and y_out is not None:
        from fasttomo.algorithms.algebraic import _tex_target
        handle = _tex_target(tex_op, y_out, "vol")

    def ptr(t):
        return 0 if t is None else t.data_ptr()

    with torch.cuda.device(t0.device):
        wrote = _core.fista_update_cuda(
            0 if t0.dtype == torch.float32 else 1, mode,
            ptr(a), ptr(grad), ptr(xp), ptr(mask),
            int(mask is not None and mask.shape[0] == Ny and Ny > 1),
            ptr(x_out), ptr(y_out), float(inv_L), float(beta), int(positivity),
            Ny, Nz, Nx, _stream_ptr(t0.device),
            vol_cache_ptr=0 if handle is None else handle)
    if wrote:
        tex_op._mark_vol_written(y_out)
    return True


@_fp16_overflow_checked
def run_fista(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 50,
    *,
    Nz: int | None = None,
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
    """FISTA (Beck & Teboulle 2009) on ``f(x) = 1/2 ||A x - b||^2``.

    ::

        y_k     = x_k + ((t_{k-1} - 1) / t_k) (x_k - x_{k-1})
        x_{k+1} = P_C(y_k - (1/L) A^T (A y_k - b))
        t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2

    ``L = ||A^T A||`` is estimated by power iteration (times a 1.01 safety
    margin) unless given; ``P_C`` applies the support and positivity. O(1/k^2)
    on the data term versus SIRT's O(1/k). Recorded loss: the data MSE at
    the momentum point ``y_k`` (free, a tight surrogate for ``x_k``). No
    CUDA graph (``t_k`` changes every iteration).

    Long runs on CUDA: the CUDA back-projection (voxel-driven texture
    gather) is not the exact adjoint of the CUDA forward projection
    (ray-marching): ``||B - A^T||_F / ||A||_F = 0.24`` (the CPU pair is
    exact), so ``B A`` has eigenvalues with small negative real parts and
    FISTA's momentum eventually amplifies them (solver audit A2). Relative
    error to the phantom, 64^2 slice, 61 views over +-60 degrees, after
    20 / 200 / 1000 / 3000 iterations: CUDA noise-free 0.24 / 0.185 / 0.21
    / 14.1, CPU 0.23 / 0.176 / 0.149 / 0.131; with 1 % noise CUDA 0.25 /
    0.31 / 2.2 / 384 (CPU 0.24 / 0.34 / 1.3 / 3.0, the noise amplification
    of unregularised least squares). ``positivity=True`` delays it (CUDA
    noise-free 0.051 at 1000, 0.35 at 3000) but does not prevent it. SIRT
    is not affected at these lengths. On CUDA keep ``n_iter`` at about 200
    to 300 (with noise the error is lowest well before that anyway), or
    regularise (:func:`run_fista_tv`) and watch the loss for longer runs.
    The same holds for :func:`run_fista_tv`.

    Parameters
    ----------
    sinogram, angles_deg, n_iter
        fp32/fp16 sinogram, degrees, iterations (default 50).
    Nz, positivity, mask, support_mask, warm_start, num_threads, return_loss,
    loss_callback, progress
        See Conventions of :mod:`fasttomo.algorithms`.
    euler_zyz_deg, projector
        As in :func:`run_sirt`.
    n_power_iter : int
        Power-iteration steps for the Lipschitz estimate (default 10).
    L : float, optional
        Lipschitz constant ``||A^T A||``; skips the power iteration.

    Returns
    -------
    volume, or ``(volume, loss_history)``; see Conventions.
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

    _fp, _bp, _op = _build_fp_bp(
        sinogram, angles_deg, (Ny, Nz, Nx), N_det, n_angles,
        device, num_threads, projector, euler_zyz_deg,
    )

    L_eff = (
        float(L) if L is not None
        else _estimate_lipschitz(_fp, _bp, (Ny, Nz, Nx), device, compute_dtype, n_power_iter)
    )

    rec = _DeviceLossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_fista")
    return _fista_loop(
        volume, sinogram, _fp, _bp, n_iter, L_eff, setup.constraints.apply, rec,
        constraints=setup.constraints, tex_op=_fista_texture_operator(_op, sinogram),
    )


@_fp16_overflow_checked
def run_fista_tv(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 50,
    *,
    tv_weight: float,
    Nz: int | None = None,
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
    """FISTA-TV: ``1/2 ||A x - b||^2 + tv_weight * TV(x) + i_C(x)``.

    The outer loop of :func:`run_fista` with the TV-L2 proximal operator,
    solved on the dual by Fast Gradient Projection (``tv_inner="fgp"``, Beck
    & Teboulle 2009b) or Chambolle 2004 (``"chambolle"``). The support and
    positivity sit inside the prox, so the composite convex theory applies.
    Recorded loss: the data MSE at ``y_k``, not the composite objective.

    Parameters
    ----------
    sinogram, angles_deg, n_iter
        fp32/fp16 sinogram, degrees, iterations (default 50).
    tv_weight : float
        Isotropic TV weight, required and > 0 (use :func:`run_fista` for 0).
        Keyword-only (it used to be the third positional). ``TV(x)`` is the
        *sum* over all voxels of the isotropic gradient norm
        ``sqrt(dy^2 + dz^2 + dx^2)`` from forward differences, zero across
        the far boundary (for ``Ny = 1`` it is the 2-D TV of the slice), and
        the data term the sum ``1/2 ||A x - b||^2``. The gradient-descent
        solvers (:func:`run_grad_descent`, :func:`run_ad_recon`) scale both
        differently: ``mean((A x - b)^2) + tv_weight * tv_loss_3d(x)``, with
        :func:`fasttomo.tv_loss_3d` the *mean* over the ``Nc = (Ny-1)(Nz-1)
        (Nx-1)`` inner voxels (with an ``eps`` in the root; 0 for ``Ny = 1``).
        So the same ``tv_weight`` means different things; the same trade-off
        (up to the boundary voxels) needs ``tv_weight_fista ~= tv_weight_ad *
        M / (2 Nc)`` with ``M = n_angles * Ny * N_det`` the sinogram size
        (61 views, 128^3: ``0.24 * tv_weight_ad``).
    tv_inner_iter : int
        Inner dual iterations of the TV prox per outer iteration (default 20,
        >= 0). The prox is solved inexactly, the more so the stronger the
        prox weight ``lam = tv_weight / L``: at the default, the relative
        duality gap of the prox is 1-3 % at ``lam = 0.02`` and 13-24 % at
        ``lam = 0.1`` (prox error 0.4-0.5 % and 2.1-2.7 %; fp64 gap on the
        prox inputs of a 96^3 run, solver audit P1b). End to end at
        ``lam = 0.1`` (96^3, 2 % noise) the objective after 50 outer
        iterations was 7.69e6 with 20, 7.16e6 with 50 and 7.00e6 with 200
        inner iterations (relative error to the phantom 0.225 / 0.178 /
        0.169), so raise it for strong regularisation. On CUDA an inner
        iteration is two kernels (~0.8 ms at 256^3), so 100 inner iterations
        now cost about what 20 did with the former PyTorch prox. The dual
        restarts from zero every outer iteration (warm-starting it across
        outer iterations was measured to raise the objective).
    tv_inner : {"fgp", "chambolle"}
        Dual solver of the prox (default ``"fgp"``, O(1/k^2); ``"chambolle"``
        is O(1/k)).
    Nz, positivity, mask, support_mask, warm_start, num_threads, return_loss,
    loss_callback, progress
        See Conventions of :mod:`fasttomo.algorithms`.
    euler_zyz_deg, projector, n_power_iter, L
        As in :func:`run_fista`.

    Returns
    -------
    volume, or ``(volume, loss_history)``; see Conventions.
    """
    warm_start = _deprecated_kwarg(deprecated, "volume", "warm_start", warm_start, "run_fista_tv")
    _reject_unknown_kwargs(deprecated, "run_fista_tv")
    if tv_weight <= 0.0:
        raise ValueError(
            f"run_fista_tv requires tv_weight > 0 (got {tv_weight}). "
            "Use run_fista for the unregularised problem."
        )
    tv_inner_iter = _as_int(tv_inner_iter, "tv_inner_iter", "run_fista_tv")
    if tv_inner not in ("fgp", "chambolle"):
        raise ValueError(
            f"run_fista_tv: unknown mode tv_inner={tv_inner!r}; expected 'fgp' or 'chambolle'."
        )

    setup = _prepare_recon(
        "run_fista_tv", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        allowed_dtypes=(torch.float32, torch.float16), warm_start=warm_start,
        positivity=positivity, mask=mask, support_mask=support_mask,
    )
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)
    volume, compute_dtype, n_iter = setup.volume0, setup.compute_dtype, setup.n_iter

    _fp, _bp, _op = _build_fp_bp(
        sinogram, angles_deg, (Ny, Nz, Nx), N_det, n_angles,
        device, num_threads, projector, euler_zyz_deg,
    )

    L_eff = (
        float(L) if L is not None
        else _estimate_lipschitz(_fp, _bp, (Ny, Nz, Nx), device, compute_dtype, n_power_iter)
    )

    _mask = setup.constraints.support_mask
    lam = tv_weight / L_eff

    def prox(u):
        return _tv_prox_3d(u, lam, positivity, _mask, tv_inner_iter, tv_inner)

    rec = _DeviceLossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_fista_tv")
    return _fista_loop(volume, sinogram, _fp, _bp, n_iter, L_eff, prox, rec,
                       tex_op=_fista_texture_operator(_op, sinogram))
