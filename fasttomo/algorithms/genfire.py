"""
GENFIRE (Pryor et al. 2017). FROZEN: moved verbatim from the former
``fasttomo/algorithms.py``; no change to its math, normalisation, edge
handling or numerical output (tests/test_frozen_identity.py).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

import fasttomo._core as _core
from fasttomo._backend import _data_ptr
from fasttomo.constraints import cylindrical_mask
from fasttomo.configs import ConstraintConfig
from fasttomo.algorithms._common import (
    _GENFIRE_RES_EXTENSION_PIVOTS,
    _prepare_recon,
    _LossRecorder,
    _emit_volume,
)


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
    Nz: int | None = None,
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
    volume_callback: Callable[[int, torch.Tensor], None] | None = None,
    progress: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """GENFIRE (Pryor et al. 2017). fp32, CPU and CUDA. Frozen numerics.

    Grids the oversampled 1-D FFTs of the projections onto a 2-D Fourier grid
    per Y-slice (inverse-distance weighting), then alternates Fourier-space
    data enforcement and real-space constraints (support, positivity,
    soft-threshold). The resolution extension/suppression schedule widens
    then narrows the enforced frequency band. Returns the best iterate, not
    the last. Recorded loss: the mean L1 Fourier error on the known grid
    points of each iterate (computed every iteration anyway).

    Parameters
    ----------
    sinogram, angles_deg
        ``[n_angles, Ny, N_det]`` fp32 and ``[n_angles]`` degrees.
    n_iter : int
        Iterations (default 100). Keyword-only for one release: the third
        positional used to be ``Nz``.
    mask : bool, optional
        Inscribed cylinder support. ``None`` (default) means the cylinder
        unless ``support_mask`` is given (the historical default was
        ``mask=True``).
    support_mask : Tensor, optional
        User support (Conventions), applied in real space every iteration and
        to the output; mutually exclusive with ``mask=True``.
    positivity : bool
        Default True.
    oversampling_ratio : float
        Fourier grid size ``N_det * oversampling_ratio`` (default 2.0).
    cutoff_dist : float
        Gridding radius in grid units (default 0.5).
    resolution_extension : bool
        Use the resolution extension/suppression schedule (default True).
    warm_start : Tensor, optional
        Initial volume: its Fourier transform, with the measured values on the
        known grid points, seeds the iteration instead of the zero-filled
        data.
    Nz, sparsity, return_loss, loss_callback, progress
        See Conventions of :mod:`fasttomo.algorithms`.
    volume_callback : callable, optional
        Optional hook invoked once per iteration as ``volume_callback(i,
        volume)`` — 0-based index and that iteration's real-space iterate
        after the support/positivity/sparsity constraints (the iterate whose
        Fourier error lands in ``loss_history[i]``), cropped to ``[Ny, Nz,
        Nx]`` float32 on its device. The returned volume is the lowest-error
        iterate re-synthesised from its Fourier transform, so it matches the
        callback volume at ``argmin(loss_history)`` (up to FFT round-off),
        which is the last one only when the final iterate is the best. The
        tensor is a view into a per-iteration buffer: the callback must not
        modify it and must ``.clone()`` it to keep it. Exceptions raised by
        the callback are logged and swallowed.

    Returns
    -------
    volume, or ``(volume, loss_history)``; see Conventions.
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
    z0 = (bigN - Nz) // 2
    z1 = z0 + Nz
    x0 = (bigN - Nx) // 2
    x1 = x0 + Nx
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
        if volume_callback is not None:
            _emit_volume(volume_callback, i_iter, vol[:, z0:z1, x0:x1])

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
