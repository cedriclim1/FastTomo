"""
PyTorch-level wrappers around FastTomo's C/CUDA projection kernels.

All public functions accept and return torch.Tensor (float32).

Volume convention:  [Ny, Nz, Nx]
  Ny  — tilt axis (independent slices)
  Nz  — depth (beam direction at θ=0)
  Nx  — lateral (detector direction)

Sinogram convention: [n_angles, Ny, N_det]

Angles: 1-D tensor, degrees.

For a 2-D single-slice problem wrap your [Nz, Nx] volume as [1, Nz, Nx]
and squeeze the result.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import fasttomo._core as _core
from fasttomo.autograd import (
    _FP2D, _BP2D, _FP3D, _BP3D, _FPResire, _BPResire,
    _euler_zyz_to_rotmat, _resolve_num_threads,
)


# ── primitive projection ops ───────────────────────────────────────────────
#
# These are thin shims around torch.autograd.Function subclasses defined in
# fasttomo/autograd.py. When inputs do not require grad, Function.apply
# short-circuits and behaves identically to a direct kernel call. When a
# volume (or sinogram) has requires_grad=True, loss.backward() flows the
# adjoint partner kernel back into .grad. See fasttomo/autograd.py and
# fasttomo-wiki/wiki/concepts/autograd-projectors.md.

def forward_project(
    volume: torch.Tensor,          # [Ny, Nz, Nx]
    angles_deg: torch.Tensor,      # [n_angles] degrees
    N_det: int = -1,
) -> torch.Tensor:
    """
    Forward-project volume → sinogram.

    Returns [n_angles, Ny, N_det] float32 on the same device as volume.
    N_det defaults to Nx.

    Differentiable w.r.t. ``volume`` and ``angles_deg``.  Volume gradient
    calls back_project (the discrete adjoint). Angle gradient uses
    ∂sino/∂θ = ∂vol/∂x·(−z) + ∂vol/∂z·x at each ray step (CPU + CUDA).
    """
    if volume.ndim == 2:
        volume = volume.unsqueeze(0)
    return _FP2D.apply(volume, angles_deg, N_det)


def back_project(
    sinogram: torch.Tensor,        # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,      # [n_angles] degrees
    Nz: int = -1,
) -> torch.Tensor:
    """
    Back-project sinogram → volume [Ny, Nz, Nx].

    Nz defaults to N_det (square reconstruction).

    Differentiable w.r.t. ``sinogram``: backward calls forward_project.
    """
    if sinogram.ndim == 2:
        sinogram = sinogram.unsqueeze(1)
    return _BP2D.apply(sinogram, angles_deg, Nz)


# ── 3D rotation-matrix projection (CPU + CUDA) ───────────────────────────

def forward_project_3d(
    volume: torch.Tensor,                       # [Ny, Nz, Nx]
    euler_zyz_deg: torch.Tensor,                # [n_angles, 3] — (φ, θ, ψ) degrees
    N_det: int = -1,
    shifts: torch.Tensor | None = None,         # [n_angles, 2] — optional per-tilt detector-plane shifts (px)
    num_threads: int | None = None,             # CPU only; resolution: kwarg → OMP_NUM_THREADS → 1
) -> torch.Tensor:
    """
    3D rotation-matrix forward projection (CPU + CUDA).

    Each projection uses a full 3×3 ZYZ rotation matrix, so rays can cross
    Y-slice boundaries. This is required for general orientations (dual-axis,
    in-plane rotation, azimuthal misalignment).

    euler_zyz_deg columns: (φ, θ, ψ) — ZYZ Euler angles in degrees.
    For a pure single-axis tilt series: φ=0, ψ=0, θ=tilt_angle.

    If ``shifts`` is provided, a per-tilt sub-pixel translation is applied to
    the predicted sinogram via :func:`apply_shifts_2d` (bilinear). This makes
    the operator a full pose model — rotation + detector-plane translation —
    so a learnable ``[n_angles, 2]`` shifts tensor can be optimized jointly
    with ``volume`` and ``euler_zyz_deg`` under autograd.

    Returns sinogram [n_angles, Ny, N_det] on the same device as ``volume``.
    Same shape convention as forward_project so algorithms can swap projectors
    transparently. The CPU path uses explicit trilinear interpolation; pass
    ``num_threads=N`` to opt into OpenMP parallelism. Default behavior is
    serial (single-threaded) so reconstructions don't silently grab cores.

    ``num_threads`` resolution: explicit kwarg → ``OMP_NUM_THREADS`` env →
    fall back to 1. The CUDA path ignores ``num_threads`` (the kernel
    manages its own grid).

    Differentiable w.r.t. ``volume``, ``euler_zyz_deg``, and ``shifts`` on
    both devices: backward calls back_project_3d for the volume gradient and
    a dedicated grad_rotmat kernel for the rotation-matrix gradient. The
    euler→rotmat conversion lives in pure PyTorch so autograd chains ∂L/∂M
    into ∂L/∂euler; ``grid_sample`` handles ∂L/∂shifts.
    """
    if volume.ndim == 2:
        volume = volume.unsqueeze(0)
    nt = _resolve_num_threads(num_threads)
    mats = _euler_zyz_to_rotmat(euler_zyz_deg)   # [n_angles, 3, 3]; autograd traces through
    sino = _FP3D.apply(volume, mats, N_det, nt)
    if shifts is not None:
        sino = apply_shifts_2d(sino, shifts)
    return sino


def back_project_3d(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    euler_zyz_deg: torch.Tensor,    # [n_angles, 3] — (φ, θ, ψ) degrees
    Nz: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    """
    3D rotation-matrix back-projection (CPU + CUDA).

    Adjoint of forward_project_3d. Voxel-driven gather: each voxel samples
    the sinogram at its rotated detector position for every projection.

    Returns volume [Ny, Nz, Nx] on the same device as ``sinogram``.
    Pass ``num_threads=N`` to opt into OpenMP parallelism on the CPU path
    (default 1 — serial). Resolution order: kwarg → OMP_NUM_THREADS → 1.

    Differentiable w.r.t. ``sinogram``: backward calls forward_project_3d.
    """
    if sinogram.ndim == 2:
        sinogram = sinogram.unsqueeze(1)
    nt = _resolve_num_threads(num_threads)
    return _BP3D.apply(sinogram, euler_zyz_deg, Nz, nt)


def forward_project_resire(
    volume: torch.Tensor,           # [Ny, Nz, Nx]
    euler_zyz_deg: torch.Tensor,    # [n_angles, 3] — (φ, θ, ψ) degrees
    N_det: int = -1,
) -> torch.Tensor:
    """
    RESIRE voxel-driven forward projection (CUDA only).

    Implements the Miao group scatter kernel: each voxel is projected to
    the detector using a full 3×3 rotation matrix with 4-point supersampling
    (±0.25 px in volume XZ space). Uses atomicAdd; output is normalised by
    the supersampling factor (÷4) so it is on the same scale as
    forward_project_3d.

    euler_zyz_deg: [n_angles, 3] — (φ, θ, ψ) ZYZ Euler angles in degrees.

    Returns sinogram [n_angles, Ny, N_det].

    Differentiable w.r.t. ``volume``: backward calls back_project_resire.
    Not differentiable w.r.t. ``euler_zyz_deg`` — use ``refine_angles`` for
    angular refinement (brute-force grid search) outside the reconstruction
    loop. For autograd-driven angle optimisation, prefer the texture-based
    ``forward_project_3d`` (or the single-axis ``forward_project``).
    """
    if volume.ndim == 2:
        volume = volume.unsqueeze(0)
    return _FPResire.apply(volume, euler_zyz_deg, N_det)


def back_project_resire(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    euler_zyz_deg: torch.Tensor,    # [n_angles, 3] — (φ, θ, ψ) degrees
    Nz: int = -1,
) -> torch.Tensor:
    """
    RESIRE voxel-driven back-projection (CUDA only).

    Adjoint of forward_project_resire. Each voxel gathers from the sinogram
    at its rotated detector position with 4-point supersampling. Output is
    normalised by the supersampling factor (÷4).

    Returns volume [Ny, Nz, Nx].

    Differentiable w.r.t. ``sinogram``: backward calls forward_project_resire.
    """
    if sinogram.ndim == 2:
        sinogram = sinogram.unsqueeze(1)
    return _BPResire.apply(sinogram, euler_zyz_deg, Nz)


# ── cylindrical support mask ──────────────────────────────────────────────

def cylindrical_mask(
    Nz: int,
    Nx: int,
    radius: float | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Inscribed-cylinder support mask for the XZ reconstruction plane.

    Returns [Nz, Nx] float32 tensor — 1.0 inside the cylinder, 0.0 outside.
    Multiply a volume [Ny, Nz, Nx] by ``mask.unsqueeze(0)`` to apply.

    Default radius = min(Nz, Nx) / 2, which inscribes the cylinder in the
    smaller dimension.  Uses the same centering as the projection kernels:
    pixel index ``i`` maps to coordinate ``i - (N-1)/2``.
    """
    r = float(min(Nz, Nx)) / 2.0 if radius is None else radius
    iz = np.linspace(-(Nz - 1) / 2.0, (Nz - 1) / 2.0, Nz, dtype=np.float32)
    ix = np.linspace(-(Nx - 1) / 2.0, (Nx - 1) / 2.0, Nx, dtype=np.float32)
    mask_np = (iz[:, None] ** 2 + ix[None, :] ** 2 <= r ** 2).astype(np.float32)
    mask = torch.Tensor(mask_np)  # CPU float32; torch.Tensor is a class, not a factory
    return mask.to(device) if device is not None else mask


# ── ramp filter (PyTorch, runs on same device as input) ───────────────────

# Ramp-filter |ξ| is shape-only; cache per (N_det, device, dtype) so tight
# loops (FBP per slice, SIRT-style residual filtering) avoid the rfftfreq +
# abs allocation each call.
_RAMP_FILTER_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _ramp_filter_coeffs(N_det: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (int(N_det), device, dtype)
    coeffs = _RAMP_FILTER_CACHE.get(key)
    if coeffs is None:
        coeffs = torch.fft.rfftfreq(N_det, device=device, dtype=dtype).abs().contiguous()
        _RAMP_FILTER_CACHE[key] = coeffs
    return coeffs


def ramp_filter(sinogram: torch.Tensor) -> torch.Tensor:
    """
    Apply ramp filter |ξ| to each detector row of sinogram.

    sinogram: [n_angles, Ny, N_det]
    Returns same shape, same device.
    """
    N_det  = sinogram.shape[-1]
    sino_f = sinogram.float()
    f      = torch.fft.rfft(sino_f, n=N_det, dim=-1)
    f      = f * _ramp_filter_coeffs(N_det, sino_f.device, torch.float32)
    return torch.fft.irfft(f, n=N_det, dim=-1)


def apply_shifts(
    sinogram: torch.Tensor,   # [n_angles, Ny, N_det]
    shifts: torch.Tensor,     # [n_angles] detector-axis shifts in pixels
) -> torch.Tensor:            # [n_angles, Ny, N_det], same device as sinogram
    """
    Sub-pixel shift each projection row along the detector axis via Fourier
    phase multiplication.  Positive shift moves the signal to the right.

    shifts may be on CPU even when sinogram is on CUDA.
    """
    N_det = sinogram.shape[-1]
    s = shifts.to(device=sinogram.device, dtype=torch.float32)
    freqs = torch.fft.rfftfreq(N_det, device=sinogram.device)          # [N_freq]
    phase_arg = -2.0 * math.pi * freqs.view(1, 1, -1) * s.view(-1, 1, 1)
    phase_mul = torch.polar(torch.ones_like(phase_arg), phase_arg)
    F = torch.fft.rfft(sinogram.float(), n=N_det, dim=-1)
    return torch.fft.irfft(F * phase_mul, n=N_det, dim=-1)


def apply_shifts_2d(
    sinogram: torch.Tensor,   # [n_angles, Ny, N_det]
    shifts: torch.Tensor,     # [n_angles, 2] — (det-axis, tilt-axis) pixel shifts
) -> torch.Tensor:            # [n_angles, Ny, N_det], same device as sinogram
    """
    Per-tilt sub-pixel translation of the sinogram in the detector plane via
    bilinear ``grid_sample``.  Differentiable w.r.t. both ``sinogram`` and
    ``shifts`` — autograd flows through ``grid_sample`` natively.

    ``shifts[:, 0]`` shifts along the detector axis (last dim, ``N_det``);
    ``shifts[:, 1]`` shifts along the tilt axis (``Ny``).  Sign convention
    matches ``apply_shifts``: positive shift moves the signal toward larger
    indices (right / down).

    Intended for joint optimization with the volume and pose: drop in after
    ``forward_project_3d`` (or pass ``shifts=`` to it directly) so per-tilt
    translational alignment becomes a learnable parameter.
    """
    if shifts.ndim != 2 or shifts.shape[-1] != 2:
        raise ValueError(f"shifts must have shape [n_angles, 2], got {tuple(shifts.shape)}")
    n_a, Ny, Nd = sinogram.shape
    if shifts.shape[0] != n_a:
        raise ValueError(
            f"shifts has {shifts.shape[0]} entries but sinogram has {n_a} angles"
        )

    s = shifts.to(device=sinogram.device, dtype=torch.float32)

    ys = torch.linspace(-1.0, 1.0, Ny, device=sinogram.device, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, Nd, device=sinogram.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")                  # [Ny, Nd]
    base = torch.stack((gx, gy), dim=-1)                            # [Ny, Nd, 2]
    base = base.unsqueeze(0).expand(n_a, -1, -1, -1)                # [n_a, Ny, Nd, 2]

    # Sample from (x - sx, y - sy) so positive shift moves the signal +x / +y.
    # Normalized coords with align_corners=True: 1 pixel = 2/(N-1).
    norm_x = 2.0 / max(Nd - 1, 1)
    norm_y = 2.0 / max(Ny - 1, 1)
    off = torch.stack((s[:, 0] * norm_x, s[:, 1] * norm_y), dim=-1) # [n_a, 2]
    grid = base - off.view(n_a, 1, 1, 2)                            # [n_a, Ny, Nd, 2]

    inp = sinogram.unsqueeze(1).float()                             # [n_a, 1, Ny, Nd]
    out = torch.nn.functional.grid_sample(
        inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return out.squeeze(1)


def estimate_shifts_xcorr(
    volume: torch.Tensor,                       # [Ny, Nz, Nx]
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    *,
    upsample_factor: int = 10,                  # reserved; parabolic fit used today
    max_shift: float | None = None,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — CUDA only
    num_threads: int | None = None,             # CPU 3D path only
) -> torch.Tensor:                              # [n_angles] float32, CPU
    """
    Estimate the translational shift of each projection (along the detector
    axis) by 1-D phase cross-correlation between the forward projection of
    *volume* and the corresponding row of *sinogram*.

    Returns the correction shift to apply so that::

        aligned_sino = ft.apply_shifts(sinogram, shifts)

    brings sinogram into alignment with the volume (no negation needed at the
    call site).

    Sub-pixel precision is achieved via 3-point parabolic interpolation around
    the cross-correlation peak.  upsample_factor is accepted for API
    compatibility but currently unused.

    Typical outer-loop usage::

        for _ in range(n_outer):
            vol    = ft.run_resire(aligned_sino, angles, n_iter=100)
            shifts = ft.estimate_shifts_xcorr(vol, aligned_sino, angles)
            aligned_sino = ft.apply_shifts(aligned_sino, shifts)
    """
    N_det = sinogram.shape[-1]

    if euler_zyz_deg is not None:
        pred = forward_project_3d(volume, euler_zyz_deg, num_threads=num_threads)
    else:
        pred = forward_project(volume, angles_deg)

    F_pred = torch.fft.rfft(pred.float(),      n=N_det, dim=-1)  # [n_a, Ny, N_freq]
    F_meas = torch.fft.rfft(sinogram.float(),  n=N_det, dim=-1)

    # Phase cross-power spectrum: peaks at +s when sinogram is shifted +s vs pred
    cross   = F_pred * F_meas.conj()
    cross_n = cross / cross.abs().clamp(min=1e-12)

    # Coherent sum over Ny before IRFFT (SNR ∝ sqrt(Ny))
    cross_sum = cross_n.sum(dim=1)                                  # [n_a, N_freq]
    ccf       = torch.fft.irfft(cross_sum, n=N_det, dim=-1)        # [n_a, N_det]

    # Integer peak → signed shift
    peak_idx  = ccf.argmax(dim=-1)                                  # [n_a]
    shifts_int = torch.where(
        peak_idx <= N_det // 2,
        peak_idx.float(),
        peak_idx.float() - N_det,
    )

    # Parabolic sub-pixel refinement with circular neighbour wrap
    idx_m = (peak_idx - 1) % N_det
    idx_p = (peak_idx + 1) % N_det
    y_m   = ccf.gather(1, idx_m.unsqueeze(1)).squeeze(1)
    y_0   = ccf.gather(1, peak_idx.unsqueeze(1)).squeeze(1)
    y_p   = ccf.gather(1, idx_p.unsqueeze(1)).squeeze(1)
    denom = y_m - 2.0 * y_0 + y_p
    delta = torch.where(
        denom.abs() > 1e-9,
        0.5 * (y_m - y_p) / denom,
        torch.zeros_like(denom),
    )

    # Negate: F_pred * conj(F_meas) peaks at +s when sino is +s vs pred;
    # returning -s so that apply_shifts(sino, shifts) corrects the alignment.
    shifts_out = -(shifts_int + delta)

    if max_shift is not None:
        shifts_out = shifts_out.clamp(-max_shift, max_shift)

    return shifts_out.cpu()


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

    Returns real 2D filter [Nz, Nx] in Fourier space ready to multiply
    against rfft2 of a volume slice.  Actually returns the spatial-domain
    WF [Nz, Nx] so it can be applied via fft2 multiplication.
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

    # return as real-valued spatial filter (apply via fft2 multiply)
    return WF_amp  # [Nz, Nx//2+1] complex magnitudes


# ── public reconstruction functions ───────────────────────────────────────

def run_fbp(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
) -> torch.Tensor:
    """
    Filtered Back-Projection.

    Applies ramp filter to each sinogram row then back-projects.
    Returns volume [Ny, Nz, Nx].
    mask=True zeros voxels outside the inscribed cylinder (common in ET).
    """
    filtered = ramp_filter(sinogram)
    volume   = back_project(filtered, angles_deg, Nz=Nz)
    # angular sampling normalisation: π / n_angles
    volume   = volume * (math.pi / len(angles_deg))
    if positivity:
        volume.clamp_(min=0.0)
    if mask:
        _mask = cylindrical_mask(volume.shape[1], volume.shape[2], device=volume.device)
        volume = volume * _mask.unsqueeze(0)
    return volume


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
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    device = sinogram.device

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

    if positivity:
        volume.clamp_(min=0.0)
    if mask:
        _mask = cylindrical_mask(Nz, Nx, device=device)
        volume = volume * _mask.unsqueeze(0)
    return volume


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

    projector: pass a :class:`ParallelProjector` (or :class:`RotmatProjector`
    when using ``euler_zyz_deg``) to reuse a persistent cudaArray + texture
    handle across all FP/BP calls. Same numerics, ~1.5–1.9× faster on the
    inner loop. Must be sized to the volume / sinogram shape; angles passed
    via the projector replace ``angles_deg`` / ``euler_zyz_deg`` for FP/BP.

    return_loss: if True, returns (volume, loss_history) where loss_history
    is a float32 CPU tensor of shape [n_iter] containing the sinogram MSE
    ||b - A·x^k||² / numel at each iteration.

    sparsity: L1 soft-threshold applied after each update (proximal operator
    of sparsity·‖x‖₁). 0.0 disables it.

    Returns volume [Ny, Nz, Nx].
    mask=True enforces cylindrical support at each iteration (GENFIRE-style).
    """
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    device = sinogram.device
    volume = torch.zeros(Ny, Nz, Nx, dtype=torch.float32, device=device)

    if projector is not None:
        if euler_zyz_deg is not None:
            # RotmatProjector path: matrices are fixed for SIRT, baked into a closure
            from fasttomo.autograd import _euler_zyz_to_rotmat
            mats = _euler_zyz_to_rotmat(euler_zyz_deg).to(device=device)
            _fp = lambda v: projector.forward(v, mats)
            _bp = lambda s: projector.back(s, mats)
        else:
            _fp = lambda v: projector.forward(v)
            _bp = lambda s: projector.back(s)
    elif euler_zyz_deg is not None:
        _fp = lambda v: forward_project_3d(v, euler_zyz_deg, N_det=N_det, num_threads=num_threads)
        _bp = lambda s: back_project_3d(s, euler_zyz_deg, Nz=Nz, num_threads=num_threads)
    else:
        _fp = lambda v: forward_project(v, angles_deg, N_det=N_det)
        _bp = lambda s: back_project(s, angles_deg, Nz=Nz)

    # --- normalisation matrices (computed once with the chosen projector) ---
    ones_sino = torch.ones_like(sinogram)
    C_vol     = _bp(ones_sino)
    C_vol     = torch.where(C_vol > 0, 1.0 / C_vol, torch.zeros_like(C_vol))

    ones_vol  = torch.ones_like(volume)
    R_sino    = _fp(ones_vol)
    R_sino    = torch.where(R_sino > 0, 1.0 / R_sino, torch.zeros_like(R_sino))

    _mask = cylindrical_mask(Nz, Nx, device=device).unsqueeze(0) if mask else None
    losses: list[float] = []

    # --- iteration loop ---
    for _ in range(n_iter):
        raw_res  = sinogram - _fp(volume)
        if return_loss:
            losses.append(raw_res.pow(2).mean().item())
        residual = raw_res * R_sino
        volume   = volume + omega * C_vol * _bp(residual)
        if sparsity > 0.0:
            volume = volume.sign() * (volume.abs() - sparsity).clamp_(min=0.0)
        if positivity:
            volume.clamp_(min=0.0)
        if _mask is not None:
            volume = volume * _mask

    if return_loss:
        return volume, torch.tensor(losses, dtype=torch.float32)
    return volume


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
    """
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    device = sinogram.device

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
    proj_pad = F.pad(sinogram.float(), (pad_l, pad_r))   # [n_ang, Ny, bigN]
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
        delay_pts = np.array([0.0, 0.5, 0.95, 0.5, 0.0], dtype=np.float32)
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

        # real-space constraints
        if sparsity > 0.0:
            vol = vol.sign() * (vol.abs() - sparsity).clamp_(min=0.0)
        if positivity:
            vol.clamp_(min=0.0)
        vol = vol * support.unsqueeze(0)

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
    if positivity:
        vol.clamp_(min=0.0)

    # crop to [Ny, Nz, Nx]
    vol = vol[:, z0:z1, x0:x1]
    if return_loss:
        return vol, torch.tensor(losses, dtype=torch.float32)
    return vol


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
    """
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    device = sinogram.device
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

    _mask = cylindrical_mask(Nz, Nx, device=device).unsqueeze(0) if mask else None
    losses: list[float] = []

    for _ in range(n_iter):
        proj     = forward_project(volume, angles_deg, N_det=N_det)
        residual = sinogram - proj
        if return_loss:
            losses.append(residual.pow(2).mean().item())
        volume   = volume + omega * _wbp(residual)
        if sparsity > 0.0:
            volume = volume.sign() * (volume.abs() - sparsity).clamp_(min=0.0)
        if positivity:
            volume.clamp_(min=0.0)
        if _mask is not None:
            volume = volume * _mask

    if return_loss:
        return volume, torch.tensor(losses, dtype=torch.float32)
    return volume


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

    Returns volume [Ny, Nz, Nx].
    """
    if not sinogram.is_cuda:
        raise RuntimeError("run_resire requires CUDA — move sinogram to GPU first")

    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    device = sinogram.device
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

    losses: list[float] = []

    for _ in range(n_iter):
        residual = sinogram - _fp(volume)
        if return_loss:
            losses.append(residual.pow(2).mean().item())
        volume   = volume + alpha * _bp(residual)
        if sparsity > 0.0:
            volume = volume.sign() * (volume.abs() - sparsity).clamp_(min=0.0)
        if positivity:
            volume.clamp_(min=0.0)
        if _mask is not None:
            volume = volume * _mask

    if return_loss:
        return volume, torch.tensor(losses, dtype=torch.float32)
    return volume


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
        tv_weight:      3-D anisotropic TV regularisation weight; the
                        penalty is mean(|∂y f|) + mean(|∂z f|) + mean(|∂x f|)
                        across the full [Ny, Nz, Nx] volume (0.0 disables)
        euler_zyz_deg:  [n_angles, 3] — switches to the 3D rotmat projector
                        (CUDA only). Mutually exclusive with ``angles_deg``.
        use_resire_projector: with ``euler_zyz_deg``, use the RESIRE 4-pt
                        scatter projector instead of the rotmat tex3D one.
        return_loss:    if True, also return the per-iteration loss history.

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

    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det
    device = sinogram.device

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

    _mask = (
        cylindrical_mask(Nz, Nx, device=device).unsqueeze(0) if mask else None
    )
    losses: list[float] = []

    is_lbfgs = isinstance(opt, torch.optim.LBFGS)

    def closure():
        opt.zero_grad()
        sino_pred = _fp(volume)
        loss = ((sino_pred - sinogram) ** 2).mean()
        if tv_weight > 0.0:
            dy = volume[1:, :, :] - volume[:-1, :, :]
            dz = volume[:, 1:, :] - volume[:, :-1, :]
            dx = volume[:, :, 1:] - volume[:, :, :-1]
            loss = loss + tv_weight * (
                dy.abs().mean() + dz.abs().mean() + dx.abs().mean()
            )
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
            if positivity:
                volume.clamp_(min=0.0)
            if _mask is not None:
                volume.mul_(_mask)

    out = volume.detach()
    if return_loss:
        return out, torch.tensor(losses, dtype=torch.float32)
    return out
