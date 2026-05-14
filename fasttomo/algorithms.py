"""
Reconstruction algorithms for FastTomo.

All public functions accept and return torch.Tensor (float32).

Volume convention:  [Ny, Nz, Nx]
Sinogram convention: [n_angles, Ny, N_det]
Angles: 1-D tensor, degrees.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm  # noqa: F401 — kept for downstream scripts that import from here

import fasttomo._core as _core
from fasttomo.autograd import _assert_dtype_match, _check_inputs, _euler_zyz_to_rotmat
from fasttomo.projection import (
    forward_project,
    back_project,
    forward_project_3d,
    back_project_3d,
    forward_project_resire,
    back_project_resire,
)
from fasttomo.utilities import apply_shifts_2d, cylindrical_mask, ramp_filter, tv_loss_3d
from fasttomo.regularizers import s3im_loss
from fasttomo.projector import ParallelProjector, RotmatProjector

_log = logging.getLogger(__name__)


# ── module-level constants ─────────────────────────────────────────────────

# GENFIRE resolution extension/suppression schedule control points.
# np.interp maps iteration progress [0,1] through these pivots to produce
# the per-iteration Fourier cutoff fraction.
_GENFIRE_RES_EXTENSION_PIVOTS = (0.0, 0.5, 0.95, 0.5, 0.0)


# ── private helpers shared across algorithms ───────────────────────────────

def _apply_sparsity(vol: torch.Tensor, weight: float) -> torch.Tensor:
    """L1 soft-threshold (proximal operator of weight·‖·‖₁)."""
    return vol.sign() * (vol.abs() - weight).clamp_(min=0.0)


def _maybe_cylindrical_mask(
    Nz: int,
    Nx: int,
    device: torch.device,
    apply: bool,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Return a [1, Nz, Nx] cylindrical mask if apply=True, else None.

    The mask is cast to ``dtype`` so callers can multiply it against a
    volume in any supported precision (fp16 / fp32) without an upcast.
    """
    if not apply:
        return None
    return cylindrical_mask(Nz, Nx, device=device).to(dtype).unsqueeze(0)


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
            f"{op_name}: volume must have shape {expected_shape}, "
            f"got {tuple(volume.shape)}."
        )


@dataclass
class ConstraintSet:
    """Reconstruction constraints applied once per iteration.

    Construct once before the iteration loop so the mask tensor is precomputed.
    ``apply`` is out-of-place (returns new tensor); ``apply_inplace_`` mutates
    in place for leaf tensors and CUDA Graph paths.

    Application order: mask → positivity → sparsity.
    """
    positivity: bool = False
    mask: torch.Tensor | None = None   # precomputed [1, Nz, Nx] or None
    sparsity: float = 0.0

    def apply(self, volume: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            volume = volume * self.mask
        if self.positivity:
            volume = volume.clamp(min=0.0)
        if self.sparsity > 0.0:
            volume = _apply_sparsity(volume, self.sparsity)
        return volume

    def apply_inplace_(self, volume: torch.Tensor) -> None:
        """In-place variant for leaf tensors and CUDA Graph paths."""
        if self.mask is not None:
            volume.mul_(self.mask)
        if self.positivity:
            volume.clamp_(min=0.0)
        if self.sparsity > 0.0:
            volume.copy_(_apply_sparsity(volume, self.sparsity))


def _finish_iters(
    volume: torch.Tensor,
    losses: list[float],
    return_loss: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Standardised return for iterative reconstruction algorithms."""
    if return_loss:
        return volume, torch.tensor(losses, dtype=torch.float32)
    return volume


# ── exact weighting filter (Wolf et al. 2014) ─────────────────────────────

def _compute_exact_wf(
    angles_deg: torch.Tensor,   # [n_angles] degrees
    Nz: int, Nx: int,
    N_det: int,
    device: torch.device,
    threshold_scale: float = 1.0,
) -> torch.Tensor:
    """
    Numerically compute the exact weighting filter for W-BP.

    Construct PSF: back-project a sinogram with a 1 at t=0 for every angle
    (i.e., the sinogram of a unit point at the volume centre).
    WF = |g| / |TF|  (thresholded).

    Returns real 2D filter [Nz, Nx//2+1] Fourier amplitudes ready to
    multiply against rfft2 of a volume slice.
    """
    # sinogram of a unit point at the origin: each row has a 1 at t=0
    n_angles = len(angles_deg)
    point_sino = torch.zeros(n_angles, 1, N_det, device=device)
    center_det = (N_det - 1) // 2
    point_sino[:, 0, center_det] = 1.0

    # back-project to get PSF of the S-BP [1, Nz, Nx]
    psf = back_project(point_sino, angles_deg, Nz=Nz).squeeze(0)  # [Nz, Nx]

    # transfer function
    TF  = torch.fft.rfft2(psf)
    TF_amp = TF.abs()

    # threshold: avoid division by zero inside missing wedge
    n_alpha = float(n_angles)
    thresh  = threshold_scale * TF_amp[0, 0].item() / n_alpha
    thresh  = max(thresh, 1e-6)

    # WF = 1 / TF  (thresholded)
    WF_amp = torch.where(TF_amp > thresh, 1.0 / TF_amp, torch.zeros_like(TF_amp))

    # return as real-valued Fourier amplitudes (apply via rfft2 multiply)
    return WF_amp  # [Nz, Nx//2+1] complex magnitudes


# ── public reconstruction functions ───────────────────────────────────────

def run_fbp(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    projector: object | None = None,
) -> torch.Tensor:
    """
    Filtered Back-Projection.

    Applies ramp filter to each sinogram row then back-projects.
    Returns volume [Ny, Nz, Nx].
    mask=True zeros voxels outside the inscribed cylinder (common in ET).

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
    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=(torch.float32, torch.float16),
        op_name="run_fbp",
    )
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
        volume = back_project(filtered, angles_deg, Nz=Nz)
    # angular sampling normalisation: π / n_angles
    volume = volume * (math.pi / n_angles)
    _mask = _maybe_cylindrical_mask(
        volume.shape[1],
        volume.shape[2],
        volume.device,
        mask,
        dtype=volume.dtype,
    )
    return ConstraintSet(positivity=positivity, mask=_mask).apply(volume)


def run_weighted_backprojection(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    Nz: int = -1,
    positivity: bool = False,
    threshold_scale: float = 1.0,
    mask: bool = False,
) -> torch.Tensor:
    """
    Exact Weighted Back-Projection (Wolf et al. 2014).

    Computes the exact numerical weighting filter from the PSF of a
    central point, then applies it slice-by-slice in Fourier space.

    Returns volume [Ny, Nz, Nx].
    mask=True zeros voxels outside the inscribed cylinder (common in ET).
    """
    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=(torch.float32,),
        op_name="run_weighted_backprojection",
    )
    Nx = N_det

    # compute WF once for this tilt geometry [Nz, Nx//2+1] (amplitudes)
    WF_amp = _compute_exact_wf(
        angles_deg, Nz, Nx, N_det, device, threshold_scale
    )

    # simple back-project first → [Ny, Nz, Nx]
    layergram = back_project(sinogram, angles_deg, Nz=Nz)

    # apply WF slice-by-slice via rfft2
    volume = torch.empty_like(layergram)
    for iy in range(Ny):
        F_slice = torch.fft.rfft2(layergram[iy])     # [Nz, Nx//2+1]
        # multiply by WF (phase-preserving, magnitude correction)
        phase   = F_slice / (F_slice.abs() + 1e-12)
        F_out   = phase * (F_slice.abs() * WF_amp)
        volume[iy] = torch.fft.irfft2(F_out, s=(Nz, Nx))

    _mask = _maybe_cylindrical_mask(Nz, Nx, device, mask)
    return ConstraintSet(positivity=positivity, mask=_mask).apply(volume)


def run_sirt(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 50,
    omega: float = 1.0,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — use 3D projector if set
    return_loss: bool = False,
    sparsity: float = 0.0,
    num_threads: int | None = None,             # CPU 3D path only
    projector: object | None = None,            # ParallelProjector / RotmatProjector for cached FP/BP
    volume: torch.Tensor | None = None,         # optional warm-start [Ny, Nz, Nx]; controls dtype
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

    Returns volume [Ny, Nz, Nx].
    mask=True enforces cylindrical support at each iteration (GENFIRE-style).
    """
    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=(torch.float32, torch.float16),
        op_name="run_sirt",
    )
    Nx = N_det
    if volume is None:
        volume = torch.zeros(Ny, Nz, Nx, dtype=sinogram.dtype, device=device)
    else:
        _assert_warm_start_shape(volume, (Ny, Nz, Nx), "run_sirt")
        _assert_dtype_match(sinogram, volume, "sinogram", "volume", "run_sirt")
        volume = volume.to(device=device).contiguous().clone()
    compute_dtype = volume.dtype

    _fp, _bp, _internal_proj = _build_fp_bp(
        sinogram, angles_deg, (Ny, Nz, Nx), N_det, n_angles,
        device, num_threads, projector, euler_zyz_deg,
    )

    # --- normalisation matrices (computed once with the chosen projector) ---
    ones_sino = torch.ones_like(sinogram)
    C_vol     = _bp(ones_sino)
    C_vol     = torch.where(C_vol > 0, 1.0 / C_vol, torch.zeros_like(C_vol))

    ones_vol  = torch.ones_like(volume)
    R_sino    = _fp(ones_vol)
    R_sino    = torch.where(R_sino > 0, 1.0 / R_sino, torch.zeros_like(R_sino))

    _mask = _maybe_cylindrical_mask(Nz, Nx, device, mask, dtype=compute_dtype)
    constraints = ConstraintSet(positivity=positivity, mask=_mask, sparsity=sparsity)
    losses: list[float] = []

    # --- CUDA Graph capture path ----------------------------------------
    # Gate: CUDA input, no per-iteration loss tracking (which calls .item()).
    # Rotmat mode (euler_zyz_deg) is kept on eager path because the per-call
    # matrix tensor can vary, while cached graphs assume fixed captured inputs.
    # If capture fails, fall back to the eager loop below.
    _active_proj = projector if projector is not None else _internal_proj
    if sinogram.is_cuda and not return_loss and n_iter > 1 and euler_zyz_deg is None:
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
                 float(constraints.sparsity),
                 constraints.mask is not None,
                 euler_zyz_deg is not None),
        )
        if graph_result is not None:
            return graph_result

    # --- eager iteration loop (fallback / CPU / return_loss path) --------
    for _ in range(n_iter):
        raw_res  = sinogram - _fp(volume)
        if return_loss:
            losses.append(raw_res.pow(2).mean().item())
        residual = raw_res * R_sino
        volume   = volume + omega * C_vol * _bp(residual)
        volume   = constraints.apply(volume)

    return _finish_iters(volume, losses, return_loss)


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

    Power iteration. Returns a Python float with a 1.01 safety margin so the
    FISTA step size 1/L sits strictly inside the contraction region even if
    the iteration hasn't fully converged.
    """
    v = torch.randn(vol_shape, device=device, dtype=dtype)
    v = v / (v.norm() + 1e-30)
    L = 0.0
    for _ in range(n_iter):
        AtAv = _bp(_fp(v))
        L = float(AtAv.norm())
        v = AtAv / (L + 1e-30)
    return L * 1.01


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
    step = 1.0 / 12.0

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
    return_loss: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Shared Beck-Teboulle FISTA outer loop.

    ``prox_fn(u)`` is called once per iteration on ``u = y_k - (1/L) ∇f(y_k)``
    and must return the next iterate ``x_{k+1}``. For pure FISTA it's just the
    constraint projection; for FISTA-TV it's the TV+constraint prox.

    ``return_loss`` records ``||b − A·y_k||²`` (data-fit MSE at the momentum
    point, which is free) at each iteration — a tight surrogate for the loss
    at ``x_k``.
    """
    inv_L = 1.0 / L
    t = 1.0
    x_prev = volume.clone()
    y = volume.clone()
    losses: list[float] = []
    for _ in range(n_iter):
        fp_y = _fp(y)
        if return_loss:
            losses.append((sinogram - fp_y).pow(2).mean().item())
        grad = _bp(fp_y - sinogram)
        x_new = prox_fn(y - inv_L * grad)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        beta = (t - 1.0) / t_new
        y = x_new + beta * (x_new - x_prev)
        x_prev = x_new
        t = t_new
    return _finish_iters(x_prev, losses, return_loss)


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
            mats = _euler_zyz_to_rotmat(euler_zyz_deg).to(device=device)
            _fp = lambda v: projector.forward(v, mats)
            _bp = lambda s: projector.back(s, mats)
        else:
            _fp = lambda v: projector.forward(v)
            _bp = lambda s: projector.back(s)
    elif sinogram.is_cuda:
        if euler_zyz_deg is not None:
            mats = _euler_zyz_to_rotmat(euler_zyz_deg).to(device=device)
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
        _fp = lambda v: forward_project(v, angles_deg, N_det=N_det)
        _bp = lambda s: back_project(s, angles_deg, Nz=Nz)
    return _fp, _bp, _internal_proj


def run_fista(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 50,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    euler_zyz_deg: torch.Tensor | None = None,
    return_loss: bool = False,
    n_power_iter: int = 10,
    L: float | None = None,
    num_threads: int | None = None,
    projector: object | None = None,
    volume: torch.Tensor | None = None,
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

    Returns the reconstructed volume (or ``(volume, loss_history)`` when
    ``return_loss=True``). Eager CUDA only — no CUDA Graph capture because
    ``t_k`` changes every iteration.
    """
    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=(torch.float32, torch.float16),
        op_name="run_fista",
    )
    Nx = N_det
    if volume is None:
        volume = torch.zeros(Ny, Nz, Nx, dtype=sinogram.dtype, device=device)
    else:
        _assert_warm_start_shape(volume, (Ny, Nz, Nx), "run_fista")
        _assert_dtype_match(sinogram, volume, "sinogram", "volume", "run_fista")
        volume = volume.to(device=device).contiguous().clone()
    compute_dtype = volume.dtype

    _fp, _bp, _internal_proj = _build_fp_bp(
        sinogram, angles_deg, (Ny, Nz, Nx), N_det, n_angles,
        device, num_threads, projector, euler_zyz_deg,
    )

    L_eff = (
        float(L) if L is not None
        else _estimate_lipschitz(_fp, _bp, (Ny, Nz, Nx), device, compute_dtype, n_power_iter)
    )

    _mask = _maybe_cylindrical_mask(Nz, Nx, device, mask, dtype=compute_dtype)
    constraints = ConstraintSet(positivity=positivity, mask=_mask)

    return _fista_loop(volume, sinogram, _fp, _bp, n_iter, L_eff, constraints.apply, return_loss)


def run_fista_tv(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    tv_weight: float,
    n_iter: int = 50,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    euler_zyz_deg: torch.Tensor | None = None,
    return_loss: bool = False,
    n_power_iter: int = 10,
    L: float | None = None,
    tv_inner_iter: int = 20,
    tv_inner: str = "fgp",
    num_threads: int | None = None,
    projector: object | None = None,
    volume: torch.Tensor | None = None,
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
        return_loss, n_power_iter, L, num_threads, projector, volume:
                      see :func:`run_fista`.
        tv_weight:    isotropic TV regularisation weight (required, > 0).
                      Pass ``tv_weight=0`` and use :func:`run_fista` directly.
        tv_inner_iter: inner dual-projection iterations (default 20).
        tv_inner:     ``"fgp"`` (default; O(1/k²) inner) or ``"chambolle"``
                      (O(1/k) inner).

    Returns same shape/format as :func:`run_fista`. Loss history records
    ``||b − A·y_k||²``, *not* the composite objective.
    """
    if tv_weight <= 0.0:
        raise ValueError(
            f"run_fista_tv requires tv_weight > 0 (got {tv_weight}). "
            "Use run_fista for the unregularised problem."
        )

    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=(torch.float32, torch.float16),
        op_name="run_fista_tv",
    )
    Nx = N_det
    if volume is None:
        volume = torch.zeros(Ny, Nz, Nx, dtype=sinogram.dtype, device=device)
    else:
        _assert_warm_start_shape(volume, (Ny, Nz, Nx), "run_fista_tv")
        _assert_dtype_match(sinogram, volume, "sinogram", "volume", "run_fista_tv")
        volume = volume.to(device=device).contiguous().clone()
    compute_dtype = volume.dtype

    _fp, _bp, _internal_proj = _build_fp_bp(
        sinogram, angles_deg, (Ny, Nz, Nx), N_det, n_angles,
        device, num_threads, projector, euler_zyz_deg,
    )

    L_eff = (
        float(L) if L is not None
        else _estimate_lipschitz(_fp, _bp, (Ny, Nz, Nx), device, compute_dtype, n_power_iter)
    )

    _mask = _maybe_cylindrical_mask(Nz, Nx, device, mask, dtype=compute_dtype)
    lam = tv_weight / L_eff
    prox = lambda u: _tv_prox_3d(u, lam, positivity, _mask, tv_inner_iter, tv_inner)

    return _fista_loop(volume, sinogram, _fp, _bp, n_iter, L_eff, prox, return_loss)


def run_sart(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 5,
    subset_size: int = 1,
    omega: float = 1.0,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — use 3D projector if set
    return_loss: bool = False,
    sparsity: float = 0.0,
    num_threads: int | None = None,             # CPU 3D path only
    volume: torch.Tensor | None = None,         # optional warm-start [Ny, Nz, Nx]; controls dtype
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
    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=(torch.float32, torch.float16),
        op_name="run_sart",
    )
    Nx = N_det
    if volume is None:
        volume = torch.zeros(Ny, Nz, Nx, dtype=sinogram.dtype, device=device)
    else:
        _assert_warm_start_shape(volume, (Ny, Nz, Nx), "run_sart")
        _assert_dtype_match(sinogram, volume, "sinogram", "volume", "run_sart")
        volume = volume.to(device=device).contiguous().clone()
    compute_dtype = volume.dtype

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
            return forward_project(v, angles_deg[idx], N_det=N_det)

        def _bp_subset(s: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return back_project(s, angles_deg[idx], Nz=Nz)

        def _fp_full(v: torch.Tensor) -> torch.Tensor:
            return forward_project(v, angles_deg, N_det=N_det)

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

    _mask = _maybe_cylindrical_mask(Nz, Nx, device, mask, dtype=compute_dtype)
    constraints = ConstraintSet(positivity=positivity, mask=_mask, sparsity=sparsity)
    losses: list[float] = []

    # --- outer iteration loop ---
    for _ in range(n_iter):
        # --- inner subset loop ---
        for idx, R_s, C_s in zip(subsets, R_list, C_list):
            sino_s = sinogram[idx]                   # [|S|, Ny, N_det]
            pred_s = _fp_subset(volume, idx)         # [|S|, Ny, N_det]
            r_s    = sino_s - pred_s                 # [|S|, Ny, N_det]
            volume = volume + omega * C_s * _bp_subset(r_s * R_s, idx)

        if return_loss:
            with torch.no_grad():
                raw_res = sinogram - _fp_full(volume)
                losses.append(raw_res.pow(2).mean().item())

        volume = constraints.apply(volume)

    return _finish_iters(volume, losses, return_loss)


def run_ad_recon(
    sinogram: torch.Tensor,            # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,          # [n_angles] degrees
    *,
    n_iter: int = 1000,
    refine_pose: bool = False,
    refine_shifts: bool = False,
    Nz: int = -1,
    lr_volume: float = 0.05,
    lr_angles: float = 1e-2,
    lr_shifts: float = 1e-2,
    tv_weight: float = 0.0,
    sparsity: float = 0.0,
    in_plane_weight: float = 0.0,
    s3im_weight: float = 0.0,
    positivity: bool = False,
    mask: bool = False,
    loss_fn: str = "mse",
    cosine_schedule: bool = True,
    return_loss: bool = False,
    return_pose: bool = False,
    projector: object | None = None,
    anchor_reference: bool = False,
    tilt_bound_deg: float | None = 1.0,
    antisymmetric_z: bool = True,
) -> torch.Tensor | tuple:
    """
    Autograd-based reconstruction with optional pose and shift refinement.

    ``run_ad_recon`` packages the autograd reconstruction loop demonstrated in
    ``Notebooks/test.ipynb`` (cells 5d13fa07, acadca5a). It is the AD path's
    library entry point; it is not a new algorithm.

    The loop optimises a volume (and optionally ZYZ Euler angles and/or 2-D
    per-tilt shifts) by minimising the sinogram data-fit loss with Adam.  A
    persistent :class:`ParallelProjector` or :class:`RotmatProjector` is built
    once before the loop so the underlying ``TextureCache`` is reused across all
    iterations — exactly as in the notebook.

    Args:
        sinogram:        [n_angles, Ny, N_det] float32 CUDA tensor.
        angles_deg:      [n_angles] tilt angles in degrees.
        n_iter:          Number of Adam steps (default 1000).
        refine_pose:     If True, also optimise ZYZ Euler angles (requires
                         RotmatProjector internally). Default False.
        refine_shifts:   If True, also optimise per-tilt 2-D shifts via
                         ``apply_shifts_2d`` after each FP. Independent of
                         ``refine_pose``; both may be True simultaneously.
        Nz:              Reconstruction depth; defaults to N_det.
        lr_volume:       Adam learning rate for the volume (default 0.05).
        lr_angles:       Adam learning rate for Euler angles (default 1e-2).
        lr_shifts:       Adam learning rate for shifts (default 1e-2).
        tv_weight:       3-D isotropic TV regularisation weight; added as
                         ``tv_weight * tv_loss_3d(volume)`` each iteration.
                         0.0 disables it.
        sparsity:        Sparsity weight; adds ``sparsity * volume.mean()``
                         to the loss each iteration (L1-style, as used in the
                         notebook for atomic-resolution data). 0.0 disables it.
        in_plane_weight: Soft L2 regularisation weight on the in-plane Euler
                         columns α (col0) and γ (col2); adds
                         ``in_plane_weight * (α² + γ²).mean()`` (deg²) to the
                         loss each iteration. Only active when
                         ``refine_pose=True`` and the weight is > 0. With
                         ``antisymmetric_z=True`` (default), γ = −α so this is
                         effectively ``2 · α².mean()``. With
                         ``antisymmetric_z=False`` it applies independently to
                         both columns. Off by default (0.0); typical values for
                         single-tilt data are 1e-4 to 1e-2.
        s3im_weight:     Stochastic Structural Similarity (S3IM) regularisation
                         weight; adds ``s3im_weight * s3im_loss(pred, sinogram)``
                         to the loss each iteration. S3IM captures non-local
                         sinogram structure that MSE/L1 misses, and empirically
                         suppresses missing-wedge streaks without the ring
                         artefact that TV can introduce at high weights.
                         0.0 disables it (default). Typical values: 1e-5–1e-3.
        positivity:      If True, clamps ``volume`` to ``[0, ∞)`` after each
                         step inside ``torch.no_grad()``.
        mask:            If True, builds ``cylindrical_mask(Nz, Nx)`` once and
                         multiplies ``volume`` by it inside ``torch.no_grad()``
                         after each step.
        loss_fn:         ``"mse"`` (default) or ``"l1"`` — data-fit loss
                         between predicted and measured sinogram.
        cosine_schedule: If True, attaches a
                         :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`
                         to each Adam optimiser (mirrors notebook defaults).
        return_loss:     If True, also return the per-iteration loss history
                         as ``torch.tensor(losses)``.
        return_pose:     If True, also return the refined pose parameters as
                         ``(euler_zyz_deg, shifts)`` where each is ``None``
                         if the corresponding refinement was disabled.
        projector:       Optional pre-built :class:`RotmatProjector` (when
                         ``refine_pose=True``) or :class:`ParallelProjector`
                         (when ``refine_pose=False``). Passing a projector
                         whose class does not match the requested mode raises
                         ``ValueError``. If ``None``, a projector is built
                         internally and torn down on return.
        anchor_reference: If True, the projection whose tilt angle is closest
                         to 0° is used as a gauge anchor — its pose row is
                         frozen at ``(0, θ_ref, 0)`` and its shift row at
                         ``(0, 0)`` for the entire run by zeroing that row's
                         gradient before every Adam step. Requires at least
                         one of ``refine_pose`` or ``refine_shifts`` to be
                         True (raises ``ValueError`` otherwise). Default False.
        tilt_bound_deg:  If not None, clamps the Y-Euler column (tilt axis)
                         to ``[angles_deg ± tilt_bound_deg]`` after each Adam
                         step. Only active when ``refine_pose=True``. Pass
                         ``None`` to disable and let Y optimise freely.
                         Default 1.0.
        antisymmetric_z: If True (default), enforces ``γ = −α`` (ZY(-Z) form)
                         after every Adam step — the third ZYZ angle is always
                         the negative of the first. This removes a degree of
                         freedom per projection and prevents unrealistic
                         in-plane rotation drift on single-tilt data. Only
                         active when ``refine_pose=True``; silently ignored
                         otherwise. Pass ``False`` to let α and γ optimise
                         independently.

    Returns:
        * ``volume``               — float32 CUDA tensor [Ny, Nz, Nx]
        * if ``return_loss`` only  — ``(volume, loss_history)``
        * if ``return_pose`` only  — ``(volume, euler_or_None, shifts_or_None)``
        * if both                  — ``(volume, loss_history, euler_or_None, shifts_or_None)``

    Note:
        This function requires a CUDA device.  The underlying projectors are
        CUDA-only; passing CPU tensors will raise a ``RuntimeError`` from the
        projector constructor.
    """
    if loss_fn not in ("mse", "l1"):
        raise ValueError(f"loss_fn must be 'mse' or 'l1', got {loss_fn!r}")

    if anchor_reference and not (refine_pose or refine_shifts):
        raise ValueError(
            "anchor_reference=True requires refine_pose=True or "
            "refine_shifts=True; nothing to anchor otherwise."
        )

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

    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=(torch.float32,),
        op_name="run_ad_recon",
        require_cuda=True,
    )
    Nx = N_det
    angles_deg = angles_deg.float().to(device)

    ref_idx: int | None = None
    if anchor_reference:
        ref_idx = int(angles_deg.abs().argmin())

    # ── initialise volume ──────────────────────────────────────────────────
    volume = torch.zeros(Ny, Nz, Nx, dtype=torch.float32, device=device,
                         requires_grad=True)

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
        tilt_angle_opt = torch.zeros(n_angles, 3, dtype=torch.float32, device=device)
        tilt_angle_opt[:, 1] = angles_deg
        tilt_angle_opt.requires_grad_(True)

    if refine_shifts:
        shifts = torch.zeros(n_angles, 2, dtype=torch.float32, device=device,
                             requires_grad=True)

    # ── build optimiser list ───────────────────────────────────────────────
    opt_vol = torch.optim.Adam([volume], lr=lr_volume)
    optimisers = [opt_vol]

    opt_angles: torch.optim.Optimizer | None = None
    opt_shifts: torch.optim.Optimizer | None = None

    if refine_pose and tilt_angle_opt is not None:
        opt_angles = torch.optim.Adam([tilt_angle_opt], lr=lr_angles)
        optimisers.append(opt_angles)
    if refine_shifts and shifts is not None:
        opt_shifts = torch.optim.Adam([shifts], lr=lr_shifts)
        optimisers.append(opt_shifts)

    # ── cosine LR schedulers ───────────────────────────────────────────────
    schedulers: list = []
    if cosine_schedule:
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
            for opt in optimisers
        ]

    # ── cylindrical mask (built once) ─────────────────────────────────────
    _cmask: torch.Tensor | None = None
    if mask:
        _cmask = cylindrical_mask(Nz, Nx, device=device)  # [Nz, Nx]

    # ── S3IM generator (seeded once; reused each iteration) ───────────────
    _s3im_gen: torch.Generator | None = None
    if s3im_weight > 0.0:
        _s3im_gen = torch.Generator(device=device).manual_seed(42)

    # ── iteration ─────────────────────────────────────────────────────────
    losses: list[float] = []

    for _ in range(n_iter):
        for opt in optimisers:
            opt.zero_grad()

        # forward project
        if refine_pose and tilt_angle_opt is not None:
            mats = _euler_zyz_to_rotmat(tilt_angle_opt)
            pred = _proj.forward(volume, mats)  # type: ignore[arg-type]
        else:
            pred = _proj.forward(volume)  # type: ignore[arg-type]

        # optional shift correction
        if refine_shifts and shifts is not None:
            pred = apply_shifts_2d(pred, shifts)

        # data-fit loss
        if loss_fn == "mse":
            loss = F.mse_loss(pred, sinogram)
        else:
            loss = F.l1_loss(pred, sinogram)

        # regularisation
        if tv_weight > 0.0:
            loss = loss + tv_weight * tv_loss_3d(volume)
        if sparsity > 0.0:
            loss = loss + sparsity * volume.mean()
        if in_plane_weight > 0.0 and tilt_angle_opt is not None:
            loss = loss + in_plane_weight * (
                tilt_angle_opt[:, 0].pow(2).mean()
                + tilt_angle_opt[:, 2].pow(2).mean()
            )
        if s3im_weight > 0.0:
            loss = loss + s3im_weight * s3im_loss(pred, sinogram, generator=_s3im_gen)

        losses.append(loss.item())
        loss.backward()

        if ref_idx is not None:
            if tilt_angle_opt is not None and tilt_angle_opt.grad is not None:
                tilt_angle_opt.grad[ref_idx] = 0.0
            if shifts is not None and shifts.grad is not None:
                shifts.grad[ref_idx] = 0.0

        if antisymmetric_z and refine_pose and tilt_angle_opt is not None:
            if tilt_angle_opt.grad is not None:
                tilt_angle_opt.grad[:, 2] = 0.0

        for opt in optimisers:
            opt.step()
        for sched in schedulers:
            sched.step()

        # constraints applied inside no_grad (leaf-tensor safe)
        with torch.no_grad():
            if positivity:
                volume.clamp_(min=0.0)
            if _cmask is not None:
                volume.mul_(_cmask.unsqueeze(0))
            if tilt_bound_deg is not None and refine_pose and tilt_angle_opt is not None:
                tilt_angle_opt[:, 1].clamp_(
                    min=angles_deg - tilt_bound_deg,
                    max=angles_deg + tilt_bound_deg,
                )
            if antisymmetric_z and refine_pose and tilt_angle_opt is not None:
                tilt_angle_opt[:, 2] = -tilt_angle_opt[:, 0]

    out = volume.detach()

    # ── build return tuple ─────────────────────────────────────────────────
    if not return_loss and not return_pose:
        return out

    result: list = [out]
    if return_loss:
        result.append(torch.tensor(losses, dtype=torch.float32))
    if return_pose:
        refined_euler = tilt_angle_opt.detach().cpu() if tilt_angle_opt is not None else None
        refined_shifts = shifts.detach().cpu() if shifts is not None else None
        result.append(refined_euler)
        result.append(refined_shifts)

    return tuple(result)


# ---------------------------------------------------------------------------
# CUDA Graph helper for run_sirt
# ---------------------------------------------------------------------------

# Set of keys known to fail graph capture — skip recapture on subsequent calls.
_sirt_graph_bad_keys: set = set()


def _run_sirt_cuda_graph(
    volume: torch.Tensor,
    sinogram: torch.Tensor,
    n_iter: int,
    omega: float,
    C_vol: torch.Tensor,
    R_sino: torch.Tensor,
    _fp,
    _bp,
    constraints: ConstraintSet,
    projector,
    key: tuple,
) -> torch.Tensor | None:
    """
    Run ``n_iter`` SIRT iterations via a captured CUDA Graph.

    Returns the reconstructed volume tensor, or ``None`` if capture was
    skipped / failed (caller should fall back to the eager loop).

    The graph is cached on ``projector._sirt_graph_cache[key]``.

    Keys that previously failed capture are remembered in
    ``_sirt_graph_bad_keys`` so subsequent calls skip capture directly.
    """
    global _sirt_graph_bad_keys

    if key in _sirt_graph_bad_keys:
        return None

    if projector is None:
        return None
    if not hasattr(projector, "_sirt_graph_cache"):
        projector._sirt_graph_cache = {}
    cache = projector._sirt_graph_cache

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
            # Allocate persistent static buffers.
            vol_static = volume.clone()
            sino_static = sinogram.clone()
            C_static = C_vol.clone()
            R_static = R_sino.clone()

            # Warmup: one eager iteration with the static buffer.
            # Populates cuFFT plans, allocator pool, projector texture caches.
            with torch.no_grad():
                _sirt_iter_inplace(
                    vol_static, sino_static, C_static, R_static, omega,
                    _fp, _bp, constraints,
                )

            # Capture the iteration body on the correct device. Use an
            # explicit Stream pinned to the target device so torch.cuda.graph
            # captures on the right device — the implicit side stream is
            # created lazily on the first call's device and reused for later
            # calls, which corrupts cross-device captures.
            with torch.cuda.device(target_device):
                torch.cuda.synchronize(target_device)
                capture_stream = torch.cuda.Stream(device=target_device)
                g = torch.cuda.CUDAGraph()
                with torch.no_grad(), torch.cuda.graph(g, stream=capture_stream):
                    _sirt_iter_inplace(
                        vol_static, sino_static, C_static, R_static, omega,
                        _fp, _bp, constraints,
                    )
                torch.cuda.synchronize(target_device)

            cache[key] = (g, vol_static, sino_static, C_static, R_static)

        except Exception as exc:
            _log.warning(
                "run_sirt: CUDA Graph capture failed for key %s (%s). "
                "Falling back to eager loop for this and future calls "
                "with this configuration.",
                key, exc,
            )
            _sirt_graph_bad_keys.add(key)
            return None

    g, vol_static, sino_static, C_static, R_static = cache[key]

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

    with torch.cuda.device(target_device):
        for _ in range(n_iter):
            g.replay()
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
    constraints: ConstraintSet,
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
    flip_t = torch.LongTensor(flip).to(grid_real.device)

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
    Nz: int = -1,
    n_iter: int = 100,
    oversampling_ratio: float = 2.0,
    cutoff_dist: float = 0.5,
    positivity: bool = True,
    mask: bool = True,
    resolution_extension: bool = True,
    return_loss: bool = False,
    sparsity: float = 0.0,
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
    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=(torch.float32,),
        op_name="run_genfire",
    )
    Nx = N_det

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
        cos_t = torch.Tensor(cos_a).to(device)
        sin_t = torch.Tensor(sin_a).to(device)
        with torch.cuda.device(device):
            _core.genfire_gridding_2d_cuda(
                fk_r.data_ptr(), fk_i.data_ptr(),
                cos_t.data_ptr(), sin_t.data_ptr(),
                grid_r.data_ptr(), grid_i.data_ptr(), grid_w.data_ptr(),
                bigN, Ny, n_angles, cutoff_dist,
                torch.cuda.current_stream(device).cuda_stream,
            )
    else:
        gr_np, gi_np, gw_np = _core.genfire_gridding_2d_cpu(
            fk_r.numpy(), fk_i.numpy(), cos_a, sin_a, cutoff_dist,
        )
        grid_r = torch.Tensor(gr_np)
        grid_i = torch.Tensor(gi_np)
        grid_w = torch.Tensor(gw_np)

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
    support = sinogram.new_zeros(bigN, bigN)
    z0 = (bigN - Nz) // 2;  z1 = z0 + Nz
    x0 = (bigN - Nx) // 2;  x1 = x0 + Nx
    support[z0:z1, x0:x1] = (
        cylindrical_mask(Nz, Nx, device=device) if mask
        else sinogram.new_ones(Nz, Nx)
    )

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
        res_ind = torch.Tensor(1.0 - R_np).to(device)          # [bigN, bigN] shifted
        res_ind_us = torch.fft.ifftshift(res_ind)               # unshifted

    # ── iteration ───────────────────────────────────────────────────────────
    _inner_constraints = ConstraintSet(
        positivity=positivity, mask=support.unsqueeze(0), sparsity=sparsity,
    )
    current_F = measured_K.clone()
    best_F = current_F
    best_err = float("inf")
    losses: list[float] = []

    for i_iter in range(n_iter):
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
        losses.append(err)
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
    vol = ConstraintSet(positivity=positivity).apply(vol)

    # crop to [Ny, Nz, Nx]
    vol = vol[:, z0:z1, x0:x1]
    return _finish_iters(vol, losses, return_loss)


def run_wsirt(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    n_iter: int = 50,
    omega: float = 1.0,
    Nz: int = -1,
    positivity: bool = False,
    threshold_scale: float = 1.0,
    mask: bool = False,
    return_loss: bool = False,
    sparsity: float = 0.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted SIRT (Wolf et al. 2014).

    Replaces the simple back-projection at each SIRT step with the
    exact-filter W-BP. Converges faster and gives higher fidelity than
    SIRT, especially under limited tilt range.

    f^k = f^{k-1} + ω · W-BP(b - A·f^{k-1})

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
    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=(torch.float32,),
        op_name="run_wsirt",
    )
    Nx = N_det
    volume = torch.zeros(Ny, Nz, Nx, dtype=torch.float32, device=device)

    # pre-compute exact WF (once per tilt geometry)
    WF_amp = _compute_exact_wf(
        angles_deg, Nz, Nx, N_det, device, threshold_scale
    )

    def _wbp(sino: torch.Tensor) -> torch.Tensor:
        """W-BP: back-project then apply exact WF in Fourier space."""
        lg = back_project(sino, angles_deg, Nz=Nz)  # [Ny, Nz, Nx]
        out = torch.empty_like(lg)
        for iy in range(Ny):
            F  = torch.fft.rfft2(lg[iy])
            ph = F / (F.abs() + 1e-12)
            out[iy] = torch.fft.irfft2(ph * (F.abs() * WF_amp), s=(Nz, Nx))
        return out

    _mask = _maybe_cylindrical_mask(Nz, Nx, device, mask)
    constraints = ConstraintSet(positivity=positivity, mask=_mask, sparsity=sparsity)
    losses: list[float] = []

    for _ in range(n_iter):
        proj     = forward_project(volume, angles_deg, N_det=N_det)
        residual = sinogram - proj
        if return_loss:
            losses.append(residual.pow(2).mean().item())
        volume   = volume + omega * _wbp(residual)
        volume   = constraints.apply(volume)

    return _finish_iters(volume, losses, return_loss)


def refine_angles(
    volume: torch.Tensor,                       # [Ny, Nz, Nx]
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor | None = None,     # [n_angles] — single-axis mode
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — ZYZ mode (CUDA)
    search_range_deg: float = 3.0,
    n_steps: int = 31,
    num_threads: int | None = None,             # CPU 3D path only
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
            projs = forward_project(volume, candidates.to(sinogram.device), N_det=N_det)
            measured_j = sinogram[j]
            residuals  = (projs - measured_j.unsqueeze(0)).abs().sum(dim=(-2, -1))
            r_factors  = residuals / (measured_j.abs().sum().clamp(min=1e-12))
            refined[j] = candidates[r_factors.argmin()]

        return refined

    elif euler_zyz_deg is not None and angles_deg is None:
        # ── ZYZ coordinate-descent mode ───────────────────────────────────
        n_angles, Ny, N_det = sinogram.shape
        refined = euler_zyz_deg.detach().cpu().clone()  # [n_angles, 3]

        for j in range(n_angles):
            measured_j  = sinogram[j]                              # [Ny, N_det]
            normalizer  = measured_j.abs().sum().clamp(min=1e-12)

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
    step_size: float = 1.0,
    Nz: int = -1,
    positivity: bool = True,
    mask: bool = False,
    support: torch.Tensor | None = None,        # [Ny, Nz, Nx] or [1, Nz, Nx] float mask
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — overrides angles_deg
    return_loss: bool = False,
    sparsity: float = 0.0,
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
    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=(torch.float32,),
        op_name="run_resire",
        require_cuda=True,
    )
    Nx = N_det
    volume = torch.zeros(Ny, Nz, Nx, dtype=torch.float32, device=device)

    # step size: matches RESIRE's dt = step_size / Num_pjs / dimz
    alpha = step_size / (n_angles * Nz)

    if euler_zyz_deg is None:
        # Wrap scalar tilt angles as ZYZ triples (φ=0, θ=angle, ψ=0)
        euler_zyz_deg = torch.zeros(n_angles, 3, dtype=torch.float32)
        euler_zyz_deg[:, 1] = angles_deg.cpu().float()

    _fp = lambda v: forward_project_resire(v, euler_zyz_deg, N_det=N_det)
    _bp = lambda s: back_project_resire(s, euler_zyz_deg, Nz=Nz)

    if support is not None:
        _mask = support.to(device=device, dtype=torch.float32)
    elif mask:
        _mask = cylindrical_mask(Nz, Nx, device=device).unsqueeze(0)  # [1, Nz, Nx]
    else:
        _mask = None
    constraints = ConstraintSet(positivity=positivity, mask=_mask, sparsity=sparsity)
    losses: list[float] = []

    for _ in range(n_iter):
        residual = sinogram - _fp(volume)
        if return_loss:
            losses.append(residual.pow(2).mean().item())
        volume   = volume + alpha * _bp(residual)
        volume   = constraints.apply(volume)

    return _finish_iters(volume, losses, return_loss)


# ── autograd-based reference reconstruction ──────────────────────────────

def run_grad_descent(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor | None = None,     # [n_angles] — single-axis
    n_iter: int = 200,
    lr: float = 1e-2,
    Nz: int = -1,
    optimizer_cls=torch.optim.Adam,
    positivity: bool = False,
    mask: bool = False,
    tv_weight: float = 0.0,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — 3D mode (CUDA)
    use_resire_projector: bool = False,
    return_loss: bool = False,
    num_threads: int | None = None,             # CPU 3D path only
    tv_eps: float = 1e-8,
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

    n_angles, Ny, N_det, Nz, device = _check_inputs(
        sinogram, angles_deg, Nz,
        allowed_dtypes=(torch.float32,),
        op_name="run_grad_descent",
    )
    Nx = N_det

    volume = torch.zeros(Ny, Nz, Nx, dtype=torch.float32, device=device,
                         requires_grad=True)
    opt = optimizer_cls([volume], lr=lr)

    if euler_zyz_deg is not None:
        if use_resire_projector:
            _fp = lambda v: forward_project_resire(v, euler_zyz_deg, N_det=N_det)
        else:
            _fp = lambda v: forward_project_3d(v, euler_zyz_deg, N_det=N_det, num_threads=num_threads)
    else:
        _fp = lambda v: forward_project(v, angles_deg, N_det=N_det)

    _mask = _maybe_cylindrical_mask(Nz, Nx, device, mask)
    constraints = ConstraintSet(positivity=positivity, mask=_mask)
    losses: list[float] = []

    is_lbfgs = isinstance(opt, torch.optim.LBFGS)

    def closure():
        opt.zero_grad()
        sino_pred = _fp(volume)
        loss = ((sino_pred - sinogram) ** 2).mean()
        if tv_weight > 0.0:
            loss = loss + tv_weight * tv_loss_3d(volume, eps=tv_eps)
        loss.backward()
        return loss

    for _ in range(n_iter):
        if is_lbfgs:
            loss_val = opt.step(closure)
        else:
            loss_val = closure()
            opt.step()
        losses.append(float(loss_val.detach()))
        with torch.no_grad():
            constraints.apply_inplace_(volume)

    out = volume.detach()
    return _finish_iters(out, losses, return_loss)
