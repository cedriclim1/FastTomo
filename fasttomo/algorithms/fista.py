"""
FISTA (Beck & Teboulle 2009) and FISTA-TV, with the Lipschitz estimate and
the 3-D TV proximal operator.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

from fasttomo._backend import _fp16_overflow_checked
from fasttomo.algorithms._common import (
    _LIPSCHITZ_SAFETY,
    _NORM_EPS,
    _TV_DUAL_STEP,
    _deprecated_kwarg,
    _reject_unknown_kwargs,
    _prepare_recon,
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
        v = _normalise(AtAv, nrm + _NORM_EPS)
    L = 0.0 if nrm is None else float(nrm)
    return L * _LIPSCHITZ_SAFETY


def _normalise(t: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """``t / den`` for a 0-d fp32 device tensor ``den``, rounded as ATen
    rounds ``t / float(den)``: on CUDA a Python-scalar divisor becomes a
    multiplication by its fp32 reciprocal, on CPU an fp32 true division.
    fp16 ``t`` is computed in fp32 and rounded once (casting ``den`` or its
    reciprocal to fp16 first would lose it to the subnormal range)."""
    t32 = t if t.dtype == torch.float32 else t.float()
    out = t32 * torch.reciprocal(den) if t.is_cuda else t32 / den
    return out if t.dtype == torch.float32 else out.to(t.dtype)


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

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_fista")
    return _fista_loop(
        volume, sinogram, _fp, _bp, n_iter, L_eff, setup.constraints.apply, rec,
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
        Keyword-only (it used to be the third positional).
    tv_inner_iter : int
        Inner dual iterations (default 20).
    tv_inner : {"fgp", "chambolle"}
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

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_fista_tv")
    return _fista_loop(volume, sinogram, _fp, _bp, n_iter, L_eff, prox, rec)
