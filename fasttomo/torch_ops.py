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


# ── primitive projection ops ───────────────────────────────────────────────

def forward_project(
    volume: torch.Tensor,          # [Ny, Nz, Nx]
    angles_deg: torch.Tensor,      # [n_angles] degrees
    N_det: int = -1,
) -> torch.Tensor:
    """
    Forward-project volume → sinogram.

    Returns [n_angles, Ny, N_det] float32 on the same device as volume.
    N_det defaults to Nx.
    """
    if volume.ndim == 2:
        volume = volume.unsqueeze(0)

    angles_np = (angles_deg.detach().cpu().float() * (math.pi / 180.0)).numpy()

    if volume.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        Ny, Nz, Nx = volume.shape
        n_angles    = len(angles_np)
        if N_det < 0:
            N_det = Nx
        vol = volume.contiguous().float()
        sino = torch.empty(n_angles, Ny, N_det,
                           dtype=torch.float32, device=volume.device)
        _core.forward_project_cuda(
            vol.data_ptr(), sino.data_ptr(),
            angles_np, Ny, Nz, Nx, n_angles, N_det,
        )
        return sino
    else:
        vol_np = volume.detach().float().numpy()
        sino_np = _core.forward_project_cpu(vol_np, angles_np, N_det)
        return torch.from_numpy(sino_np)


def back_project(
    sinogram: torch.Tensor,        # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,      # [n_angles] degrees
    Nz: int = -1,
) -> torch.Tensor:
    """
    Back-project sinogram → volume [Ny, Nz, Nx].

    Nz defaults to N_det (square reconstruction).
    """
    if sinogram.ndim == 2:
        sinogram = sinogram.unsqueeze(1)

    angles_np = (angles_deg.detach().cpu().float() * (math.pi / 180.0)).numpy()

    if sinogram.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        n_angles, Ny, N_det = sinogram.shape
        Nx = N_det
        if Nz < 0:
            Nz = N_det
        sino = sinogram.contiguous().float()
        vol = torch.empty(Ny, Nz, Nx,
                          dtype=torch.float32, device=sinogram.device)
        _core.back_project_cuda(
            sino.data_ptr(), vol.data_ptr(),
            angles_np, Ny, Nz, Nx, n_angles, N_det,
        )
        return vol
    else:
        sino_np = sinogram.detach().float().numpy()
        vol_np  = _core.back_project_cpu(sino_np, angles_np, Nz)
        return torch.from_numpy(vol_np)


# ── 3D rotation-matrix projection (CUDA only) ────────────────────────────

def forward_project_3d(
    volume: torch.Tensor,           # [Ny, Nz, Nx]
    euler_zyz_deg: torch.Tensor,    # [n_angles, 3] — (φ, θ, ψ) degrees
    N_det: int = -1,
) -> torch.Tensor:
    """
    3D rotation-matrix forward projection (CUDA only).

    Each projection uses a full 3×3 ZYZ rotation matrix, so rays can cross
    Y-slice boundaries. This is required for general orientations (dual-axis,
    in-plane rotation, azimuthal misalignment).

    euler_zyz_deg columns: (φ, θ, ψ) — ZYZ Euler angles in degrees.
    For a pure single-axis tilt series: φ=0, ψ=0, θ=tilt_angle.

    Returns sinogram [n_angles, Ny, N_det] on CUDA — same shape convention
    as forward_project so algorithms can swap projectors transparently.
    """
    if not volume.is_cuda:
        raise RuntimeError(
            "forward_project_3d requires CUDA — move volume to GPU first"
        )
    if not _core.cuda_available:
        raise RuntimeError("fasttomo was compiled without CUDA support")

    if volume.ndim == 2:
        volume = volume.unsqueeze(0)

    Ny, Nz, Nx = volume.shape
    n_angles = euler_zyz_deg.shape[0]
    if N_det < 0:
        N_det = Nx

    vol  = volume.contiguous().float()
    sino = torch.empty(n_angles, Ny, N_det, dtype=torch.float32, device=vol.device)

    mats = _euler_zyz_to_rotmat(euler_zyz_deg).to(device=vol.device)  # [n_angles, 3, 3]
    mats_flat = mats.reshape(n_angles, 9).contiguous()                 # [n_angles, 9]

    _core.forward_project_rotmat_cuda(
        vol.data_ptr(), sino.data_ptr(), mats_flat.data_ptr(),
        Ny, Nz, Nx, n_angles, N_det,
    )
    return sino


def back_project_3d(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    euler_zyz_deg: torch.Tensor,    # [n_angles, 3] — (φ, θ, ψ) degrees
    Nz: int = -1,
) -> torch.Tensor:
    """
    3D rotation-matrix back-projection (CUDA only).

    Adjoint of forward_project_3d. Voxel-driven gather: each voxel samples
    the sinogram at its rotated detector position for every projection.

    Returns volume [Ny, Nz, Nx].
    """
    if not sinogram.is_cuda:
        raise RuntimeError(
            "back_project_3d requires CUDA — move sinogram to GPU first"
        )
    if not _core.cuda_available:
        raise RuntimeError("fasttomo was compiled without CUDA support")

    if sinogram.ndim == 2:
        sinogram = sinogram.unsqueeze(1)

    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    sino = sinogram.contiguous().float()
    vol  = torch.empty(Ny, Nz, Nx, dtype=torch.float32, device=sino.device)

    mats = _euler_zyz_to_rotmat(euler_zyz_deg).to(device=sino.device)
    mats_flat = mats.reshape(n_angles, 9).contiguous()

    _core.back_project_rotmat_cuda(
        sino.data_ptr(), vol.data_ptr(), mats_flat.data_ptr(),
        Ny, Nz, Nx, n_angles, N_det,
    )
    return vol


# ── ZYZ Euler angles → rotation matrix ───────────────────────────────────

def _euler_zyz_to_rotmat(euler_zyz_deg: torch.Tensor) -> torch.Tensor:
    """
    Convert ZYZ Euler angles to 3×3 rotation matrices (row-major).

    euler_zyz_deg: [n_angles, 3]  — columns are (φ, θ, ψ) in degrees.
    Convention: R = R_Z(ψ) · R_Y(θ) · R_Z(φ)  (same as RESIRE reference).

    Returns [n_angles, 3, 3] float32 on the same device as input.

    Consistency: _euler_zyz_to_rotmat([[0, θ, 0]]) equals the standard
    Y-rotation matrix [[cosθ, 0, sinθ], [0,1,0], [−sinθ, 0, cosθ]], which
    is the rotation used by FastTomo's scalar-angle projection kernels.
    """
    r   = math.pi / 180.0
    phi   = euler_zyz_deg[:, 0].float() * r
    theta = euler_zyz_deg[:, 1].float() * r
    psi   = euler_zyz_deg[:, 2].float() * r

    cp, sp = torch.cos(phi),   torch.sin(phi)
    ct, st = torch.cos(theta), torch.sin(theta)
    cu, su = torch.cos(psi),   torch.sin(psi)

    # R_ZYZ rows (each [n_angles]):
    r00 =  cu*ct*cp - su*sp;  r01 = -cu*ct*sp - su*cp;  r02 = cu*st
    r10 =  su*ct*cp + cu*sp;  r11 = -su*ct*sp + cu*cp;  r12 = su*st
    r20 = -st*cp;             r21 =  st*sp;              r22 = ct

    row0 = torch.stack([r00, r01, r02], dim=1)   # [n_angles, 3]
    row1 = torch.stack([r10, r11, r12], dim=1)
    row2 = torch.stack([r20, r21, r22], dim=1)
    return torch.stack([row0, row1, row2], dim=1)  # [n_angles, 3, 3]


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

def ramp_filter(sinogram: torch.Tensor) -> torch.Tensor:
    """
    Apply ramp filter |ξ| to each detector row of sinogram.

    sinogram: [n_angles, Ny, N_det]
    Returns same shape, same device.
    """
    N_det  = sinogram.shape[-1]
    f      = torch.fft.rfft(sinogram.float(), n=N_det, dim=-1)
    freqs  = torch.fft.rfftfreq(N_det, device=sinogram.device, dtype=torch.float32)
    f      = f * freqs.abs()
    return torch.fft.irfft(f, n=N_det, dim=-1)


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
) -> torch.Tensor:
    """
    Simultaneous Iterative Reconstruction Technique.

    x^{k+1} = x^k + ω · C · A^T · R · (b - A·x^k)

    C = 1 / A^T(ones)   (voxel normalisation)
    R = 1 / A(ones)     (ray normalisation)

    euler_zyz_deg: if provided, uses the full 3D rotation-matrix projector
    (forward_project_3d / back_project_3d) instead of the scalar-angle
    single-axis projector. CUDA only when set.

    Returns volume [Ny, Nz, Nx].
    mask=True enforces cylindrical support at each iteration (GENFIRE-style).
    """
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    device = sinogram.device
    volume = torch.zeros(Ny, Nz, Nx, dtype=torch.float32, device=device)

    if euler_zyz_deg is not None:
        _fp = lambda v: forward_project_3d(v, euler_zyz_deg, N_det=N_det)
        _bp = lambda s: back_project_3d(s, euler_zyz_deg, Nz=Nz)
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

    # --- iteration loop ---
    for _ in range(n_iter):
        residual = (sinogram - _fp(volume)) * R_sino
        volume   = volume + omega * C_vol * _bp(residual)
        if positivity:
            volume.clamp_(min=0.0)
        if _mask is not None:
            volume = volume * _mask

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
) -> torch.Tensor:
    """
    GENFIRE reconstruction (Pryor et al. 2017).

    Assembles a 2D Fourier grid per Y-slice via inverse-distance weighted
    gridding of oversampled 1D FFT projections, then iterates between
    Fourier-space data enforcement and real-space constraints (support,
    positivity).  Optional resolution extension/suppression schedule
    gradually widens then narrows the enforced frequency band to improve
    convergence under limited tilt range.

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

    # ── 1D FFT of zero-padded projections (fftshifted) ─────────────────────
    pad_l = (bigN - N_det) // 2
    pad_r = bigN - N_det - pad_l
    proj_pad = F.pad(sinogram.float(), (pad_l, pad_r))   # [n_ang, Ny, bigN]
    # fftshift(fft(x))[half+ξ] = fft(x)[ξ], which equals F_2D[ξ·sinθ, ξ·cosθ]
    # by the Fourier slice theorem — correct convention for the C gridding kernel.
    fk_shifted = torch.fft.fftshift(
        torch.fft.fft(proj_pad, dim=-1),
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

    for i_iter in range(n_iter):
        # build enforce mask for this iteration
        if resolution_extension:
            thresh = float(cutoffs[i_iter])
            enforce_us = known_us & (res_ind_us > thresh)
        else:
            enforce_us = known_us                               # [bigN, bigN]

        # IFFT → real space volume
        vol = torch.fft.ifftn(current_F, dim=(-2, -1)).real    # [Ny, bigN, bigN]

        # real-space constraints
        if positivity:
            vol.clamp_(min=0.0)
        vol = vol * support.unsqueeze(0)

        # FFT back to Fourier space
        F_cur = torch.fft.fftn(vol, dim=(-2, -1))

        # track best-error iterate (L1 on all known voxels, not just enforce_us)
        err = (F_cur[:, known_us] - measured_K[:, known_us]).abs().mean().item()
        if err < best_err:
            best_err = err
            best_F = F_cur.clone()

        # enforce Fourier constraint on known voxels
        F_cur[:, enforce_us] = measured_K[:, enforce_us]
        current_F = F_cur

    # final IFFT of best iterate
    vol = torch.fft.ifftn(best_F, dim=(-2, -1)).real           # [Ny, bigN, bigN]
    if positivity:
        vol.clamp_(min=0.0)

    # crop to [Ny, Nz, Nx]
    return vol[:, z0:z1, x0:x1]


def run_wsirt(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    n_iter: int = 50,
    omega: float = 1.0,
    Nz: int = -1,
    positivity: bool = False,
    threshold_scale: float = 1.0,
    mask: bool = False,
) -> torch.Tensor:
    """
    Weighted SIRT (Wolf et al. 2014).

    Replaces the simple back-projection at each SIRT step with the
    exact-filter W-BP. Converges faster and gives higher fidelity than
    SIRT, especially under limited tilt range.

    f^k = f^{k-1} + ω · W-BP(b - A·f^{k-1})
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

    for _ in range(n_iter):
        proj     = forward_project(volume, angles_deg, N_det=N_det)
        residual = sinogram - proj
        volume   = volume + omega * _wbp(residual)
        if positivity:
            volume.clamp_(min=0.0)
        if _mask is not None:
            volume = volume * _mask

    return volume


def refine_angles(
    volume: torch.Tensor,                       # [Ny, Nz, Nx]
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor | None = None,     # [n_angles] — single-axis mode
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — ZYZ mode (CUDA)
    search_range_deg: float = 3.0,
    n_steps: int = 31,
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
                    volume, euler_batch.to(sinogram.device), N_det=N_det
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
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — use 3D projector if set
) -> torch.Tensor:
    """
    REal Space Iterative REconstruction (RESIRE, Miao group).

    Projected gradient descent on ||Ax - b||²:

        x^{k+1} = P_C(x^k + α · A^T(b − A·x^k))

    where α = step_size / (n_angles · Nz).  Unlike SIRT there are no
    per-voxel / per-ray normalisation matrices; the step size plays that
    role.  Convergence is controlled by step_size (default 1.0 is safe for
    most geometries).

    euler_zyz_deg: if provided, uses the full 3D rotation-matrix projector
    instead of the scalar-angle single-axis projector. CUDA only when set.

    support: explicit float mask [Ny, Nz, Nx] or [1, Nz, Nx] applied each
    iteration.  If None and mask=True, uses the inscribed cylindrical mask.
    Returns volume [Ny, Nz, Nx].
    """
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    device = sinogram.device
    volume = torch.zeros(Ny, Nz, Nx, dtype=torch.float32, device=device)

    # step size: matches RESIRE's dt = step_size / Num_pjs / dimz
    alpha = step_size / (n_angles * Nz)

    if euler_zyz_deg is not None:
        _fp = lambda v: forward_project_3d(v, euler_zyz_deg, N_det=N_det)
        _bp = lambda s: back_project_3d(s, euler_zyz_deg, Nz=Nz)
    else:
        _fp = lambda v: forward_project(v, angles_deg, N_det=N_det)
        _bp = lambda s: back_project(s, angles_deg, Nz=Nz)

    if support is not None:
        _mask = support.to(device=device, dtype=torch.float32)
    elif mask:
        _mask = cylindrical_mask(Nz, Nx, device=device).unsqueeze(0)  # [1, Nz, Nx]
    else:
        _mask = None

    for _ in range(n_iter):
        residual = sinogram - _fp(volume)
        volume   = volume + alpha * _bp(residual)
        if positivity:
            volume.clamp_(min=0.0)
        if _mask is not None:
            volume = volume * _mask

    return volume
