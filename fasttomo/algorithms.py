"""
Reconstruction algorithms for FastTomo.

All public functions accept and return torch.Tensor (float32).

Volume convention:  [Ny, Nz, Nx]
Sinogram convention: [n_angles, Ny, N_det]
Angles: 1-D tensor, degrees.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm  # noqa: F401 — kept for downstream scripts that import from here

import fasttomo._core as _core
from fasttomo.autograd import _euler_zyz_to_rotmat
from fasttomo.projection import (
    forward_project,
    back_project,
    forward_project_3d,
    back_project_3d,
    forward_project_resire,
    back_project_resire,
)
from fasttomo.utilities import cylindrical_mask, ramp_filter, tv_loss_3d
from fasttomo.projector import ParallelProjector, RotmatProjector


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
) -> torch.Tensor | None:
    """Return a [1, Nz, Nx] cylindrical mask if apply=True, else None."""
    if not apply:
        return None
    return cylindrical_mask(Nz, Nx, device=device).unsqueeze(0)


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
    filtered = ramp_filter(sinogram)
    if projector is not None:
        volume = projector.back(filtered)
    elif sinogram.is_cuda:
        n_angles, Ny, N_det = sinogram.shape
        Nz_resolved = N_det if Nz < 0 else Nz
        _proj = ParallelProjector(
            (Ny, Nz_resolved, N_det), angles_deg,
            device=sinogram.device, N_det=N_det,
        )
        volume = _proj.back(filtered)
    else:
        volume = back_project(filtered, angles_deg, Nz=Nz)
    # angular sampling normalisation: π / n_angles
    volume   = volume * (math.pi / len(angles_deg))
    if positivity:
        volume.clamp_(min=0.0)
    if mask:
        _mask = cylindrical_mask(volume.shape[1], volume.shape[2],
                                 device=volume.device).to(volume.dtype)
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

    Returns volume [Ny, Nz, Nx].
    mask=True enforces cylindrical support at each iteration (GENFIRE-style).
    """
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    device = sinogram.device
    volume = torch.zeros(Ny, Nz, Nx, dtype=torch.float32, device=device)

    # Build closures _fp / _bp. On CUDA we always go through a cached projector
    # (built locally if the caller didn't pass one); on CPU we use the
    # functional API. Numerics match across both paths.
    _internal_proj = None  # held to keep texture caches alive across iterations
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

    # --- normalisation matrices (computed once with the chosen projector) ---
    ones_sino = torch.ones_like(sinogram)
    C_vol     = _bp(ones_sino)
    C_vol     = torch.where(C_vol > 0, 1.0 / C_vol, torch.zeros_like(C_vol))

    ones_vol  = torch.ones_like(volume)
    R_sino    = _fp(ones_vol)
    R_sino    = torch.where(R_sino > 0, 1.0 / R_sino, torch.zeros_like(R_sino))

    _mask = _maybe_cylindrical_mask(Nz, Nx, device, mask)
    losses: list[float] = []

    # --- iteration loop ---
    for _ in range(n_iter):
        raw_res  = sinogram - _fp(volume)
        if return_loss:
            losses.append(raw_res.pow(2).mean().item())
        residual = raw_res * R_sino
        volume   = volume + omega * C_vol * _bp(residual)
        if sparsity > 0.0:
            volume = _apply_sparsity(volume, sparsity)
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
            vol = _apply_sparsity(vol, sparsity)
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

    _mask = _maybe_cylindrical_mask(Nz, Nx, device, mask)
    losses: list[float] = []

    for _ in range(n_iter):
        proj     = forward_project(volume, angles_deg, N_det=N_det)
        residual = sinogram - proj
        if return_loss:
            losses.append(residual.pow(2).mean().item())
        volume   = volume + omega * _wbp(residual)
        if sparsity > 0.0:
            volume = _apply_sparsity(volume, sparsity)
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
            volume = _apply_sparsity(volume, sparsity)
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
    tv_eps: float = 1e-8,
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
        tv_weight:      3-D isotropic TV regularisation weight; the
                        penalty is ``mean(sqrt(∂y² + ∂z² + ∂x² + tv_eps))``
                        across the inner-corner grid of the volume.
                        Dispatches to a fused CUDA kernel via
                        ``fasttomo.tv_loss_3d`` (CPU falls back to pure
                        PyTorch). 0.0 disables the regularizer.
        tv_eps:         smoothing constant inside the TV sqrt (default
                        1e-8); only used when ``tv_weight > 0``.
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

    _mask = _maybe_cylindrical_mask(Nz, Nx, device, mask)
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
            if positivity:
                volume.clamp_(min=0.0)
            if _mask is not None:
                volume.mul_(_mask)

    out = volume.detach()
    if return_loss:
        return out, torch.tensor(losses, dtype=torch.float32)
    return out
