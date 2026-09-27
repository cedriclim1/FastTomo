"""
Post-hoc atom tracer for atomic-resolution electron tomography.

Operates on a converged reconstruction volume (e.g. the output of
``run_ad_recon`` with TV + SSIM regularisation) and returns a list of
sub-voxel atom positions, amplitudes, isotropic widths, and species
labels. Lattice-agnostic; the species classifier is a 1-D Gaussian
mixture on log integrated mass and generalises to K species.

Pipeline
--------
1. Multi-scale Difference-of-Gaussians (DoG) response stack; per-voxel
   argmax across scales gives a scale-normalised peak response and the
   characteristic σ at every voxel.
2. Local-maximum NMS on the response → peak voxel mask.
3. Sub-voxel localisation by closed-form 3-D parabolic fit on the DoG
   response neighbourhood.
4. Per-peak 3-D isotropic Gaussian least-squares fit on the original
   volume to recover (μ, σ, A, b).
5. Bond-distance / nearest-neighbour cull (reuses
   ``regularizers._cull_isolated_peaks``).
6. K-component Gaussian-mixture classifier on log integrated mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import math
import numpy as np
import torch

from fasttomo.regularizers import (
    _cull_isolated_peaks,
    _detect_atom_peaks_3d,
    _gaussian_blur_3d,
    _gaussian_kernel_3d,
)
import torch.nn.functional as F


# ── Public dataclass ──────────────────────────────────────────────────────


@dataclass
class AtomList:
    """Result of :func:`trace_atoms`.

    Attributes
    ----------
    positions_vox:
        ``(N, 3)`` float32 sub-voxel coordinates in ``[Ny, Nz, Nx]`` order.
    amplitudes:
        ``(N,)`` float32 fitted Gaussian peak amplitude (foreground only,
        background subtracted).
    sigmas:
        ``(N,)`` float32 fitted isotropic σ (voxels).
    backgrounds:
        ``(N,)`` float32 fitted local background offset.
    integrated_mass:
        ``(N,)`` float32 ``A · σ³ · (2π)^(3/2)`` — the species classifier
        feature.
    species_idx:
        ``(N,)`` int8 mixture-component index in ``[0, K)``. Components
        are ordered by ascending mean log-mass: ``species_idx == 0`` is
        the lightest component.
    species_prob:
        ``(N, K)`` float32 posterior probabilities from the GMM.
    component_log_mass:
        ``(K,)`` float32 component means in log-mass space (sorted
        ascending).
    coord_number:
        ``(N,)`` int16 nearest-neighbour count inside the lattice bond
        shell (``[bond_distance_min_frac, bond_distance_max_frac] · d_NN``).
        Zero when physical filters are disabled.
    filter_rejections:
        Diagnostic counts of peaks rejected by each physical filter
        across the adaptive-detection loop (split-peak, lone, low-coord,
        count-gate retries). Excluded from the npz dump.
    """

    positions_vox: np.ndarray
    amplitudes: np.ndarray
    sigmas: np.ndarray
    backgrounds: np.ndarray
    integrated_mass: np.ndarray
    species_idx: np.ndarray
    species_prob: np.ndarray
    component_log_mass: np.ndarray
    coord_number: np.ndarray = None  # type: ignore[assignment]
    filter_rejections: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = int(self.positions_vox.shape[0])
        if self.coord_number is None:
            self.coord_number = np.zeros(n, dtype=np.int16)
        if self.filter_rejections is None:
            self.filter_rejections = {}

    def __len__(self) -> int:
        return int(self.positions_vox.shape[0])

    def to_npz_dict(self) -> dict:
        return {
            "positions_vox": self.positions_vox,
            "amplitudes": self.amplitudes,
            "sigmas": self.sigmas,
            "backgrounds": self.backgrounds,
            "integrated_mass": self.integrated_mass,
            "species_idx": self.species_idx,
            "species_prob": self.species_prob,
            "component_log_mass": self.component_log_mass,
            "coord_number": self.coord_number,
        }


# ── Stage 1: multi-scale DoG response ─────────────────────────────────────

# Lowe's SIFT scale ratio.
_DOG_K = 1.6


def _dog_response_stack(
    volume: torch.Tensor, sigmas: Iterable[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-voxel argmax DoG response across a σ stack.

    Returns
    -------
    response: ``[Ny, Nz, Nx]`` float32 — scale-normalised response at the
        winning σ.
    sigma_at: ``[Ny, Nz, Nx]`` float32 — winning σ per voxel.
    """
    sigmas = list(sigmas)
    if not sigmas:
        raise ValueError("sigmas must be non-empty")

    response = torch.full_like(volume, fill_value=-float("inf"))
    sigma_at = torch.zeros_like(volume)
    for sigma in sigmas:
        s1 = _gaussian_blur_3d(volume, sigma)
        s2 = _gaussian_blur_3d(volume, sigma * _DOG_K)
        # Scale-normalised LoG approximation; multiply by σ² so peaks at
        # different scales compete fairly. Clamp to ≥ 0 — DoG shoulders of
        # bright peaks are *negative* and would otherwise suppress dim
        # atoms in adjacent layers (the L1₀ Fe-near-Pt failure mode).
        dog = ((s1 - s2) * (sigma * sigma)).clamp(min=0.0)
        win = dog > response
        response = torch.where(win, dog, response)
        sigma_at = torch.where(win, torch.full_like(sigma_at, sigma), sigma_at)
    return response, sigma_at


# ── Stage 1b: iterative CLEAN-style multi-pass detection ─────────────────


@torch.no_grad()
def _render_gaussians_subvoxel(
    shape: tuple[int, int, int],
    coords_float: torch.Tensor,
    amplitudes: torch.Tensor,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Render a sum of isotropic Gaussians at sub-voxel positions.

    Each peak is deposited as a trilinearly-weighted spike across the
    8 nearest integer voxels, then convolved with a peak-normalised
    Gaussian. The trilinear deposit shifts the resulting Gaussian's
    centre to the sub-voxel coordinate to within ~σ/4 accuracy — good
    enough for residual subtraction inside the iterative CLEAN.
    """
    spike = torch.zeros(shape, device=device, dtype=dtype)
    if coords_float.shape[0] == 0:
        return spike

    Ny, Nz, Nx = shape
    floor = torch.floor(coords_float)
    frac = coords_float - floor
    fy, fz, fx = frac[:, 0], frac[:, 1], frac[:, 2]
    iy0 = floor[:, 0].to(torch.long)
    iz0 = floor[:, 1].to(torch.long)
    ix0 = floor[:, 2].to(torch.long)

    for dy_i in (0, 1):
        for dz_i in (0, 1):
            for dx_i in (0, 1):
                yi = iy0 + dy_i
                zi = iz0 + dz_i
                xi = ix0 + dx_i
                wy = fy if dy_i == 1 else (1.0 - fy)
                wz = fz if dz_i == 1 else (1.0 - fz)
                wx = fx if dx_i == 1 else (1.0 - fx)
                w = wy * wz * wx
                in_bounds = (
                    (yi >= 0) & (yi < Ny)
                    & (zi >= 0) & (zi < Nz)
                    & (xi >= 0) & (xi < Nx)
                )
                if not bool(in_bounds.any()):
                    continue
                yi_b = yi[in_bounds]
                zi_b = zi[in_bounds]
                xi_b = xi[in_bounds]
                spike.index_put_(
                    (yi_b, zi_b, xi_b),
                    amplitudes[in_bounds] * w[in_bounds],
                    accumulate=True,
                )

    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel = _gaussian_kernel_3d(sigma, radius=radius, device=device, dtype=dtype)
    x = spike[None, None]
    x = F.pad(x, (radius,) * 6, mode="constant", value=0.0)
    return F.conv3d(x, kernel)[0, 0]


@torch.no_grad()
def _iterative_clean_detect(
    volume: torch.Tensor,
    sigmas_vox: list[float],
    threshold_rel: float,
    min_distance: int,
    n_passes: int,
    render_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-pass DoG + NMS detection with peak subtraction between passes.

    Each pass:
        1. Compute DoG-stack response on the (residual) volume.
        2. NMS local-max on the response, threshold = ``threshold_rel ·
           response_max_first_pass`` (held constant across passes so dim
           atoms aren't filtered out as the residual shrinks).
        3. Add new peaks to the running coord set.
        4. Subtract a Gaussian render of all detected peaks from the
           original volume to produce the next residual.

    Returns
    -------
    coords_int : ``(N, 3)`` int64 voxel coords (concatenated across passes).
    response_first : ``[Ny, Nz, Nx]`` float32 response from pass 1
        (used for sub-voxel parabolic fits).
    sigma_at_first : ``[Ny, Nz, Nx]`` float32 winning σ from pass 1.
    """
    use_dog = bool(sigmas_vox)
    if use_dog:
        response_first, sigma_at_first = _dog_response_stack(volume, sigmas_vox)
    else:
        # Raw-volume mode — used when the input already carries an atom-
        # scale Gaussian profile (typical reconstructions). Each pass
        # detects on the residual volume directly.
        response_first = volume
        sigma_at_first = torch.full_like(volume, render_sigma)
    rmax = float(response_first.max())
    if not math.isfinite(rmax) or rmax <= 0.0:
        empty = torch.zeros((0, 3), dtype=torch.int64, device=volume.device)
        return empty, response_first, sigma_at_first
    threshold = threshold_rel * rmax

    coords_accum: list[torch.Tensor] = []
    amps_accum: list[torch.Tensor] = []

    # Cumulative exclusion mask: any voxel within ``min_distance`` of an
    # already-detected peak is forbidden in subsequent passes. Robust
    # against the subtraction residual leaving a sub-pixel ghost behind.
    forbidden = torch.zeros_like(volume, dtype=torch.bool)
    nms_kernel = 2 * min_distance + 1

    residual = volume.clone()
    for p in range(n_passes):
        if p == 0:
            response = response_first
        elif use_dog:
            response, _ = _dog_response_stack(residual, sigmas_vox)
        else:
            response = residual
        new_peaks = _detect_atom_peaks_3d(response, min_distance, threshold)
        new_peaks = new_peaks & ~forbidden

        if not bool(new_peaks.any()):
            break
        new_coords = torch.nonzero(new_peaks)
        # Update the forbidden mask via max-pool dilation.
        new_mask = torch.zeros_like(volume, dtype=torch.float32)
        new_mask[new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]] = 1.0
        dilated = F.max_pool3d(
            new_mask[None, None], kernel_size=nms_kernel,
            stride=1, padding=min_distance,
        )[0, 0]
        forbidden = forbidden | (dilated > 0.5)

        coords_accum.append(new_coords)
        # Per-peak amplitude: integer-voxel value of the *current* residual
        # is the amplitude scale of what's still to be subtracted.
        new_amps = residual[new_coords[:, 0], new_coords[:, 1],
                            new_coords[:, 2]].clamp(min=0)
        amps_accum.append(new_amps)

        if p < n_passes - 1:
            # Sub-voxel offsets via parabolic fit on the response — these
            # let us subtract the Gaussian at the true atom centre instead
            # of the integer-snapped one, killing residual ghosts that
            # otherwise pollute pass 2+.
            response_np_local = response.cpu().numpy()
            new_coords_np = new_coords.cpu().numpy().astype(np.int64)
            offs, _keep = _parabolic_subvoxel(response_np_local, new_coords_np)
            offs_t = torch.from_numpy(offs).to(volume.device).to(volume.dtype)
            new_coords_float = new_coords.to(volume.dtype) + offs_t

            # Boost amplitude to compensate the integer-voxel under-sample
            # at sub-voxel offsets — V_int = A_peak · exp(-|δ|² / (2σ²)).
            d2 = (offs_t * offs_t).sum(dim=1)
            boost = torch.exp(d2 / (2.0 * render_sigma * render_sigma))
            new_amps_corr = new_amps * boost

            # Update accumulators with sub-voxel coords / corrected amps for
            # the rendered subtraction.
            if not hasattr(_iterative_clean_detect, "_subvoxel_accum"):
                pass  # placeholder for static analysis
            # Re-render the full set so the subtraction uses every detected
            # peak (cheap relative to DoG).
            all_coords_int = torch.cat(coords_accum, dim=0)
            # Re-derive sub-voxel coords for ALL accumulated passes by
            # re-computing parabolic offsets on the pass-1 response (the
            # ground-truth response field — later passes' responses are
            # contaminated by subtraction artefacts).
            all_coords_np = all_coords_int.cpu().numpy().astype(np.int64)
            all_offs, _ = _parabolic_subvoxel(
                response_first.cpu().numpy(), all_coords_np,
            )
            all_offs_t = torch.from_numpy(all_offs).to(
                volume.device).to(volume.dtype)
            all_coords_float = (all_coords_int.to(volume.dtype) + all_offs_t)
            all_amps = torch.cat(amps_accum, dim=0)
            all_d2 = (all_offs_t * all_offs_t).sum(dim=1)
            all_boost = torch.exp(
                all_d2 / (2.0 * render_sigma * render_sigma))
            rendered = _render_gaussians_subvoxel(
                volume.shape, all_coords_float, all_amps * all_boost,
                render_sigma,
                device=volume.device, dtype=volume.dtype,
            )
            residual = (volume - rendered).clamp(min=0)

    if not coords_accum:
        empty = torch.zeros((0, 3), dtype=torch.int64, device=volume.device)
        return empty, response_first, sigma_at_first
    return (torch.cat(coords_accum, dim=0),
            response_first, sigma_at_first)


# ── Stage 2: parabolic sub-voxel offset ───────────────────────────────────


def _parabolic_subvoxel(
    response_np: np.ndarray, peaks_yzx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form 3-D parabolic fit (Lowe SIFT, eq. 2).

    Returns ``(offsets [N,3], keep_mask [N])``. Offsets outside ``±0.6``
    voxels in any axis flag the peak as ill-conditioned and are filtered.
    """
    Ny, Nz, Nx = response_np.shape
    yy = peaks_yzx[:, 0]
    zz = peaks_yzx[:, 1]
    xx = peaks_yzx[:, 2]

    interior = (
        (yy > 0) & (yy < Ny - 1)
        & (zz > 0) & (zz < Nz - 1)
        & (xx > 0) & (xx < Nx - 1)
    )

    offsets = np.zeros((peaks_yzx.shape[0], 3), dtype=np.float32)
    keep = np.zeros(peaks_yzx.shape[0], dtype=bool)
    if not interior.any():
        return offsets, keep

    yi = yy[interior]
    zi = zz[interior]
    xi = xx[interior]

    R = response_np
    # First derivatives (central differences).
    gy = 0.5 * (R[yi + 1, zi, xi] - R[yi - 1, zi, xi])
    gz = 0.5 * (R[yi, zi + 1, xi] - R[yi, zi - 1, xi])
    gx = 0.5 * (R[yi, zi, xi + 1] - R[yi, zi, xi - 1])

    c = R[yi, zi, xi]
    # Second derivatives.
    Hyy = R[yi + 1, zi, xi] - 2 * c + R[yi - 1, zi, xi]
    Hzz = R[yi, zi + 1, xi] - 2 * c + R[yi, zi - 1, xi]
    Hxx = R[yi, zi, xi + 1] - 2 * c + R[yi, zi, xi - 1]
    Hyz = 0.25 * (
        R[yi + 1, zi + 1, xi] - R[yi + 1, zi - 1, xi]
        - R[yi - 1, zi + 1, xi] + R[yi - 1, zi - 1, xi]
    )
    Hyx = 0.25 * (
        R[yi + 1, zi, xi + 1] - R[yi + 1, zi, xi - 1]
        - R[yi - 1, zi, xi + 1] + R[yi - 1, zi, xi - 1]
    )
    Hzx = 0.25 * (
        R[yi, zi + 1, xi + 1] - R[yi, zi + 1, xi - 1]
        - R[yi, zi - 1, xi + 1] + R[yi, zi - 1, xi - 1]
    )

    n = yi.shape[0]
    H = np.empty((n, 3, 3), dtype=np.float32)
    H[:, 0, 0] = Hyy
    H[:, 1, 1] = Hzz
    H[:, 2, 2] = Hxx
    H[:, 0, 1] = H[:, 1, 0] = Hyz
    H[:, 0, 2] = H[:, 2, 0] = Hyx
    H[:, 1, 2] = H[:, 2, 1] = Hzx

    g = np.stack([gy, gz, gx], axis=1).astype(np.float32)

    # Per-peak solve: δ = -H⁻¹ g. Singular Hessians or wild offsets fall
    # back to δ = 0 — keep the integer peak position rather than discard.
    det = np.linalg.det(H)
    safe = np.abs(det) > 1e-6
    delta_safe = np.zeros_like(g)
    if safe.any():
        sol = np.linalg.solve(H[safe], g[safe][..., None])  # (k, 3, 1)
        delta_safe[safe] = -sol[..., 0]
    bad = np.any(np.abs(delta_safe) > 0.6, axis=1) | ~safe
    delta_safe[bad] = 0.0

    interior_idx = np.where(interior)[0]
    keep[interior_idx] = True
    offsets[interior_idx] = delta_safe
    return offsets, keep


# ── Stage 3: per-peak 3-D Gaussian LS fit ─────────────────────────────────


def _gaussian_3d_residuals(params, coords, target):
    """Residual vector for ``A * exp(-r² / 2σ²) + b - target``."""
    mu_y, mu_z, mu_x, sigma, A, b = params
    if sigma <= 0:
        return np.full(target.shape[0], 1e6, dtype=np.float64)
    dy = coords[:, 0] - mu_y
    dz = coords[:, 1] - mu_z
    dx = coords[:, 2] - mu_x
    r2 = dy * dy + dz * dz + dx * dx
    model = A * np.exp(-r2 / (2.0 * sigma * sigma)) + b
    return model - target


def _gaussian_3d_residuals_fixed_mu(params, coords, target):
    """Residual for a Gaussian whose centre is fixed (μ baked into coords).

    ``coords[:, k] = (k-th voxel offset) − (k-th μ)`` so the residual reduces
    to ``A · exp(−r² / 2σ²) + b − target``. Used for the fixed-position
    refit that extracts amplitude / σ / background after parabolic
    sub-voxel localisation — letting the LS fit relocalise μ in dense
    crystals like FePt drifts atoms toward heavier neighbours' tails.
    """
    sigma, A, b = params
    if sigma <= 0:
        return np.full(target.shape[0], 1e6, dtype=np.float64)
    r2 = (coords ** 2).sum(axis=1)
    model = A * np.exp(-r2 / (2.0 * sigma * sigma)) + b
    return model - target


def _local_gaussian_fit(
    volume_np: np.ndarray,
    seeds_yzx: np.ndarray,
    *,
    half_window: int,
    sigma_init: np.ndarray,
    sigma_bounds: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Fit a 3-D isotropic Gaussian + offset around each seed.

    seeds_yzx may be sub-voxel; the window is centred on ``round(seed)``.
    Fit uses ``scipy.optimize.least_squares`` (Levenberg–Marquardt-flavoured
    Trust Region Reflective). Failures are flagged via ``ok``.

    Returns a dict with arrays of length N: ``positions_vox``, ``sigmas``,
    ``amplitudes``, ``backgrounds``, ``ok``.
    """
    from scipy.optimize import least_squares  # local import keeps cold-start fast

    Ny, Nz, Nx = volume_np.shape
    R = half_window
    n = seeds_yzx.shape[0]

    positions = seeds_yzx.astype(np.float32).copy()
    sigmas = np.zeros(n, dtype=np.float32)
    amplitudes = np.zeros(n, dtype=np.float32)
    backgrounds = np.zeros(n, dtype=np.float32)
    ok = np.zeros(n, dtype=bool)

    # Pre-compute the relative-coord cube (centred at 0).
    ax = np.arange(-R, R + 1, dtype=np.float32)
    cy, cz, cx = np.meshgrid(ax, ax, ax, indexing="ij")
    rel = np.stack([cy.ravel(), cz.ravel(), cx.ravel()], axis=1)  # (W, 3)

    coords = rel.astype(np.float64)

    for i in range(n):
        # Patch is centred on the rounded seed; coords are integer-voxel
        # offsets from that centre.
        yi, zi, xi = seeds_yzx[i].round().astype(int)
        if (yi - R < 0 or yi + R >= Ny
                or zi - R < 0 or zi + R >= Nz
                or xi - R < 0 or xi + R >= Nx):
            continue
        patch = volume_np[
            yi - R: yi + R + 1, zi - R: zi + R + 1, xi - R: xi + R + 1
        ].astype(np.float64).ravel()

        b0 = float(np.percentile(patch, 10))
        A0 = float(patch.max() - b0)
        if A0 <= 0:
            continue
        s0 = float(sigma_init[i])

        # Initialise mu at the parabolic-fit sub-voxel offset relative to
        # the integer-centred patch.
        mu0_y = float(seeds_yzx[i, 0]) - yi
        mu0_z = float(seeds_yzx[i, 1]) - zi
        mu0_x = float(seeds_yzx[i, 2]) - xi
        x0 = np.array([mu0_y, mu0_z, mu0_x, s0, A0, b0], dtype=np.float64)
        lb = np.array([-1.5, -1.5, -1.5, sigma_bounds[0], 0.0, -np.inf])
        ub = np.array([+1.5, +1.5, +1.5, sigma_bounds[1], np.inf, np.inf])
        try:
            res = least_squares(
                _gaussian_3d_residuals, x0,
                args=(coords, patch), bounds=(lb, ub),
                method="trf", max_nfev=40, xtol=1e-4, ftol=1e-4,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not res.success and res.status <= 0:
            continue
        mu_y, mu_z, mu_x, sig, A, b = res.x
        positions[i] = np.array(
            [yi + mu_y, zi + mu_z, xi + mu_x], dtype=np.float32,
        )
        sigmas[i] = float(sig)
        amplitudes[i] = float(A)
        backgrounds[i] = float(b)
        ok[i] = True

    return {
        "positions_vox": positions,
        "sigmas": sigmas,
        "amplitudes": amplitudes,
        "backgrounds": backgrounds,
        "ok": ok,
    }


def _fixed_position_fit(
    volume_np: np.ndarray,
    positions: np.ndarray,
    *,
    half_window: int,
    sigma_init: np.ndarray,
    sigma_bounds: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Fit (σ, A, b) at fixed sub-voxel positions.

    Used when a prior stage (parabolic SIFT-style refine) has produced
    high-quality positions and we only need amplitude / width for the
    species classifier. Holding μ fixed prevents the optimiser from
    drifting atoms toward bright neighbours' tails (the FePt L1₀ failure
    mode of the free-μ fit).
    """
    from scipy.optimize import least_squares

    Ny, Nz, Nx = volume_np.shape
    R = half_window
    n = positions.shape[0]

    sigmas = np.zeros(n, dtype=np.float32)
    amplitudes = np.zeros(n, dtype=np.float32)
    backgrounds = np.zeros(n, dtype=np.float32)
    ok = np.zeros(n, dtype=bool)

    ax = np.arange(-R, R + 1, dtype=np.float64)
    rel = np.stack(
        [g.ravel() for g in np.meshgrid(ax, ax, ax, indexing="ij")],
        axis=1,
    )

    for i in range(n):
        yi, zi, xi = np.round(positions[i]).astype(int)
        if (yi - R < 0 or yi + R >= Ny
                or zi - R < 0 or zi + R >= Nz
                or xi - R < 0 or xi + R >= Nx):
            continue
        patch = volume_np[
            yi - R: yi + R + 1, zi - R: zi + R + 1, xi - R: xi + R + 1
        ].astype(np.float64).ravel()
        # Coords relative to the *true* (sub-voxel) centre:
        # rel - (μ − rounded_μ) = coords with origin at μ.
        offset = (positions[i] - np.array([yi, zi, xi])).astype(np.float64)
        coords = rel - offset

        b0 = float(np.percentile(patch, 10))
        A0 = float(patch.max() - b0)
        if A0 <= 0:
            continue
        s0 = float(sigma_init[i])

        x0 = np.array([s0, A0, b0], dtype=np.float64)
        lb = np.array([sigma_bounds[0], 0.0, -np.inf])
        ub = np.array([sigma_bounds[1], np.inf, np.inf])
        try:
            res = least_squares(
                _gaussian_3d_residuals_fixed_mu, x0,
                args=(coords, patch), bounds=(lb, ub),
                method="trf", max_nfev=30, xtol=1e-4, ftol=1e-4,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if res.status <= 0:
            continue
        sig, A, b = res.x
        sigmas[i] = float(sig)
        amplitudes[i] = float(A)
        backgrounds[i] = float(b)
        ok[i] = True

    return {
        "sigmas": sigmas,
        "amplitudes": amplitudes,
        "backgrounds": backgrounds,
        "ok": ok,
    }


# ── Stage 5: GMM species classifier ───────────────────────────────────────


def _classify_gmm(
    integrated_mass: np.ndarray, n_components: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 1-D K-component GMM on log integrated mass.

    Returns ``(species_idx [N], species_prob [N, K], component_means [K])``,
    where components are sorted by ascending mean.
    """
    from sklearn.mixture import GaussianMixture

    n = integrated_mass.shape[0]
    if n == 0:
        return (np.zeros(0, dtype=np.int8),
                np.zeros((0, n_components), dtype=np.float32),
                np.zeros(n_components, dtype=np.float32))

    log_mass = np.log(np.clip(integrated_mass, 1e-12, None))[:, None]
    if n < n_components or np.unique(log_mass).size < n_components:
        # Degenerate input: collapse to a single class.
        species_idx = np.zeros(n, dtype=np.int8)
        prob = np.zeros((n, n_components), dtype=np.float32)
        prob[:, 0] = 1.0
        means = np.full(n_components, log_mass.mean(), dtype=np.float32)
        return species_idx, prob, means

    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        random_state=seed, max_iter=200, n_init=4,
    )
    gmm.fit(log_mass)
    raw_means = gmm.means_.ravel()
    order = np.argsort(raw_means)
    sorted_means = raw_means[order].astype(np.float32)

    raw_prob = gmm.predict_proba(log_mass).astype(np.float32)
    prob = raw_prob[:, order]
    species_idx = prob.argmax(axis=1).astype(np.int8)
    return species_idx, prob, sorted_means


# ── Stage 4: physical filters (bond-distance, coordination) ───────────────


def _fept_l10_nn_distance_ang(a: float, c: float) -> float:
    """First-shell NN distance in L1₀ FePt.

    L1₀ (CuAu-type, FCT) has 12 NN: 8 different-species at the
    (a/2, 0, c/2) face-center vector and 4 same-species at the
    (a/2, a/2, 0) basal face-center vector. The minimum of the two is
    the canonical NN distance (≈2.68 Å for a=3.86, c=3.71).
    """
    d_diff = math.sqrt((a / 2.0) ** 2 + (c / 2.0) ** 2)
    d_same = a / math.sqrt(2.0)
    return float(min(d_diff, d_same))


def _apply_physical_filters(
    positions_vox: np.ndarray,
    amplitudes: np.ndarray,
    *,
    voxel_size_angstrom: float,
    d_nn_angstrom: float,
    bond_distance_min_frac: float,
    bond_distance_max_frac: float,
    coord_min: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Bidirectional bond + coordination filter.

    Split-peak suppression is amplitude-aware: when two peaks are inside
    ``d_min``, the dimmer one is dropped (greedy non-max suppression).
    Lone / low-coord rejection is symmetric.

    Returns
    -------
    keep : ``(N,) bool`` — atoms passing all enabled filters.
    coord : ``(N,) int16`` — neighbour count inside the bond shell
        ``[d_min, d_max]`` (computed on the survivors).
    rejections : per-filter rejection counts (split-peak, lone, low-coord).
    """
    from scipy.spatial import cKDTree

    n = positions_vox.shape[0]
    coord = np.zeros(n, dtype=np.int16)
    rej = {"split_peak": 0, "lone": 0, "low_coord": 0}
    if n == 0:
        return np.zeros(0, dtype=bool), coord, rej

    d_nn_vox = d_nn_angstrom / voxel_size_angstrom
    d_min = bond_distance_min_frac * d_nn_vox
    d_max = bond_distance_max_frac * d_nn_vox

    pts = positions_vox.astype(np.float64)
    tree = cKDTree(pts)

    # ── Stage 1: amplitude-aware split-peak suppression ──────────────
    # For each pair within d_min, drop the dimmer member. Greedy: walk
    # peaks in descending amplitude, suppress any later peak inside d_min
    # of an already-kept brighter one.
    survives_split = np.ones(n, dtype=bool)
    if d_min > 0:
        order = np.argsort(-amplitudes)  # brightest first
        for idx in order:
            if not survives_split[idx]:
                continue
            close = tree.query_ball_point(pts[idx], r=d_min)
            for j in close:
                if j == idx or not survives_split[j]:
                    continue
                if amplitudes[j] < amplitudes[idx]:
                    survives_split[j] = False

    rej["split_peak"] = int((~survives_split).sum())

    # ── Stage 2: lone / low-coord on survivors ───────────────────────
    surv_idx = np.where(survives_split)[0]
    surv_pts = pts[surv_idx]
    surv_tree = cKDTree(surv_pts)
    m = surv_pts.shape[0]
    coord_surv = np.zeros(m, dtype=np.int16)
    is_lone_surv = np.ones(m, dtype=bool)
    if m > 1:
        k = min(13, m)
        dists, _ = surv_tree.query(surv_pts, k=k)
        if dists.ndim == 1:
            dists = dists[:, None]
        nn = dists[:, 1:]
        in_shell = (nn >= d_min) & (nn <= d_max)
        coord_surv = in_shell.sum(axis=1).astype(np.int16)
        is_lone_surv = ~(nn <= d_max).any(axis=1)
    is_low_coord_surv = coord_surv < coord_min

    rej["lone"] = int(is_lone_surv.sum())
    rej["low_coord"] = int((is_low_coord_surv & ~is_lone_surv).sum())

    keep_surv = ~(is_lone_surv | is_low_coord_surv)

    # Reassemble keep + coord on the full index space.
    keep = np.zeros(n, dtype=bool)
    keep[surv_idx[keep_surv]] = True
    coord[surv_idx] = coord_surv
    return keep, coord, rej


def _estimate_n_expected(
    volume_np: np.ndarray,
    *,
    voxel_size_angstrom: float,
    lattice_a: float,
    lattice_c: float,
    atoms_per_cell: int,
    vol_segment_threshold_rel: float,
) -> int:
    """Estimate atom count from segmented support volume + unit-cell density.

    ``N_exp = V_seg / V_uc · atoms_per_cell``. Yang 2017 (Nature) validated
    counts against bulk density rather than a closed-form formula — this is
    the first-principles equivalent.
    """
    vmax = float(volume_np.max())
    if not math.isfinite(vmax) or vmax <= 0.0:
        return 0
    thr = vol_segment_threshold_rel * vmax
    n_seg_vox = int((volume_np > thr).sum())
    v_seg_ang3 = n_seg_vox * (voxel_size_angstrom ** 3)
    v_uc_ang3 = float(lattice_a) ** 2 * float(lattice_c)
    return int(round(v_seg_ang3 / v_uc_ang3 * atoms_per_cell))


# ── Public entry point ────────────────────────────────────────────────────


def trace_atoms(
    volume: torch.Tensor,
    *,
    sigmas_vox: Iterable[float] = (),
    threshold_rel: float = 0.02,
    min_distance: int = 1,
    fit_window: int = 2,
    sigma_bounds: tuple[float, float] = (0.6, 2.0),
    cull_max_neighbor_dist_vox: float | None = None,
    voxel_size_angstrom: float | None = None,
    max_nn_dist_angstrom: float | None = None,
    n_species: int = 2,
    n_clean_passes: int = 2,
    clean_render_sigma: float | None = None,
    lattice_constants_angstrom: tuple[float, float] | None = None,
    bond_distance_min_frac: float = 0.6,
    bond_distance_max_frac: float = 1.4,
    coord_min: int = 2,
    enable_count_prior: bool = False,
    count_prior_n_expected: int | None = None,
    count_prior_tol: float = 0.15,
    count_prior_atoms_per_cell: int = 4,
    count_prior_max_retries: int = 3,
    count_prior_threshold_step: float = 1.5,
    count_prior_segment_thr_rel: float = 0.02,
    ghost_amp_frac: float = 0.05,
    seed: int = 0,
) -> AtomList:
    """Extract a sub-voxel atom list from a reconstruction volume.

    Parameters
    ----------
    volume:
        ``[Ny, Nz, Nx]`` float32 tensor on CPU or CUDA. Stages 1–2 run on
        the input device; stages 3–5 move to numpy on the CPU.
    sigmas_vox:
        DoG scale stack (voxels). Empty (default) means **skip DoG and
        detect on the raw volume**, which is the right choice when the
        input already carries an atom-scale Gaussian profile (typical
        reconstructions and the FePt σ≈1.2-vox phantom). Pass a non-empty
        tuple to enable scale-normalised DoG (clamped ≥ 0) — useful for
        unsmoothed or noisy inputs where dim-atom shoulders need
        amplification across scales.
    threshold_rel:
        DoG-response peak threshold as a fraction of the response max.
    min_distance:
        Local-max NMS half-width (voxels).
    fit_window:
        Half-width of the cube used for the local Gaussian LS fit (5³
        cube when ``fit_window=2``).
    sigma_bounds:
        Acceptable σ range for the LS fit; out-of-range fits are dropped.
    cull_max_neighbor_dist_vox:
        Direct voxel-units NN cull. If both ``voxel_size_angstrom`` and
        ``max_nn_dist_angstrom`` are set, the physical-units value
        overrides this.
    voxel_size_angstrom, max_nn_dist_angstrom:
        Physical-units NN cull (equivalent to the
        ``cull_max_neighbor_dist`` argument of
        :func:`fasttomo.gaussian_atom_project`).
    lattice_constants_angstrom:
        ``(a, c)`` lattice constants in Å. Required (with
        ``voxel_size_angstrom``) to enable bond-distance / coordination /
        count-prior filters. Pass ``None`` (default) for HEA / unknown
        lattices to leave physical filters off.
    bond_distance_min_frac, bond_distance_max_frac:
        Bond-shell window as fractions of the lattice nearest-neighbour
        distance. Atoms with any neighbour closer than ``min_frac · d_NN``
        are rejected as split-peak FPs; atoms with no neighbour inside
        ``max_frac · d_NN`` are rejected as lone noise.
    coord_min:
        Minimum neighbour count inside the bond shell. Atoms with fewer
        are rejected as low-coordination noise. Default 2 keeps real
        edge / corner / kink surface atoms.
    enable_count_prior:
        Opt in to the adaptive count-prior gate (default off). Over-
        escalation can kill dim / surface atoms; enable only when the
        prior is reliable.
    count_prior_n_expected:
        Optional explicit atom-count upper-bound. If ``None`` and the gate
        is enabled with lattice constants, derived as
        ``V_segmented / V_uc · atoms_per_cell``.
    count_prior_tol:
        Allow up to ``(1 + tol) · N_expected`` detections before triggering
        threshold escalation. Default 0.15 (15 %) is loose enough to keep
        legitimate dim atoms.
    count_prior_atoms_per_cell:
        Atoms per conventional unit cell (4 for FePt L1₀).
    count_prior_max_retries, count_prior_threshold_step:
        Adaptive-detection loop: each retry multiplies ``threshold_rel``
        by ``count_prior_threshold_step`` (default 1.5 ×) for up to
        ``count_prior_max_retries`` extra passes.
    count_prior_segment_thr_rel:
        Volume-fraction threshold for segmenting the support
        (``volume > count_prior_segment_thr_rel · vmax``).
    n_species:
        Number of GMM components for the species classifier.
    seed:
        Random seed for the GMM fit.

    Returns
    -------
    :class:`AtomList`
    """
    if volume.dtype != torch.float32:
        raise TypeError(f"trace_atoms is fp32-only (got {volume.dtype}).")
    if volume.ndim != 3:
        raise ValueError(
            f"trace_atoms expects [Ny, Nz, Nx], got {tuple(volume.shape)}"
        )

    sigmas_list = [float(s) for s in sigmas_vox] if sigmas_vox else []
    use_dog = len(sigmas_list) > 0

    # Stage 1 (response build, threshold-independent) — done once.
    if use_dog:
        response, sigma_at = _dog_response_stack(volume, sigmas_list)
    else:
        response = volume
        # Use a constant-σ stub so downstream code works unchanged.
        sigma_at = torch.full_like(volume, 1.0)

    rmax = float(response.max())
    if not math.isfinite(rmax) or rmax <= 0.0:
        return _empty_atom_list(n_species)

    response_np = response.detach().cpu().numpy()
    sigma_at_np = sigma_at.detach().cpu().numpy()
    volume_np = volume.detach().cpu().numpy()
    vmax = float(volume_np.max())

    # Resolve NN cull radius in voxels.
    cull_vox = cull_max_neighbor_dist_vox
    if (voxel_size_angstrom is not None and max_nn_dist_angstrom is not None
            and voxel_size_angstrom > 0 and max_nn_dist_angstrom > 0):
        cull_vox = max_nn_dist_angstrom / voxel_size_angstrom

    # Decide whether physical filters are enabled.
    use_physical = (
        lattice_constants_angstrom is not None
        and voxel_size_angstrom is not None
        and voxel_size_angstrom > 0
    )
    d_nn_ang = (
        _fept_l10_nn_distance_ang(*lattice_constants_angstrom)
        if use_physical else None
    )

    # Resolve count-prior target. The count gate is *opt-in* —
    # over-escalation of the detection threshold can kill dim atoms
    # (low-Z species, surface atoms in dim shoulders). Enable only when
    # you know the prior is reliable (clean phantom, well-segmented
    # support, lattice constants known).
    n_expected = count_prior_n_expected
    if (enable_count_prior and n_expected is None and use_physical):
        n_expected = _estimate_n_expected(
            volume_np,
            voxel_size_angstrom=voxel_size_angstrom,
            lattice_a=lattice_constants_angstrom[0],
            lattice_c=lattice_constants_angstrom[1],
            atoms_per_cell=count_prior_atoms_per_cell,
            vol_segment_threshold_rel=count_prior_segment_thr_rel,
        )
    use_count_gate = (
        enable_count_prior and n_expected is not None and n_expected > 0
    )

    rejections_total: dict[str, int] = {
        "ghost_amp": 0, "split_peak": 0, "lone": 0, "low_coord": 0,
        "count_gate_retries": 0,
    }

    best: dict | None = None
    cur_thr_rel = threshold_rel
    for attempt in range(count_prior_max_retries + 1):
        out = _run_detect_and_fit(
            volume,
            volume_np=volume_np,
            response=response,
            response_np=response_np,
            sigma_at_np=sigma_at_np,
            threshold_rel=cur_thr_rel,
            min_distance=min_distance,
            fit_window=fit_window,
            sigma_bounds=sigma_bounds,
            cull_vox=cull_vox,
            vmax=vmax,
            rmax=rmax,
            n_clean_passes=n_clean_passes,
            clean_render_sigma=clean_render_sigma,
            sigmas_list=sigmas_list,
        )
        positions, sigmas_fit, amps, bgs = out

        # Ghost-amplitude filter: iterative-CLEAN can leave near-zero-amp
        # detections that pass the integer-voxel amp_floor. Drop fits whose
        # amplitude is below ``ghost_amp_frac × median(amplitudes)``. Real
        # bimodal lattices (Fe at ~0.2× median, Pt at ~1× median) survive
        # at the default 0.05.
        if ghost_amp_frac > 0 and positions.shape[0] > 0:
            med_amp = float(np.median(amps))
            if med_amp > 0:
                ghost = amps < (ghost_amp_frac * med_amp)
                if ghost.any():
                    rejections_total["ghost_amp"] += int(ghost.sum())
                    keep_g = ~ghost
                    positions = positions[keep_g]
                    sigmas_fit = sigmas_fit[keep_g]
                    amps = amps[keep_g]
                    bgs = bgs[keep_g]

        # Apply physical filters before evaluating the count gate.
        if use_physical and positions.shape[0] > 0:
            keep, coord, rej = _apply_physical_filters(
                positions, amps,
                voxel_size_angstrom=voxel_size_angstrom,
                d_nn_angstrom=d_nn_ang,
                bond_distance_min_frac=bond_distance_min_frac,
                bond_distance_max_frac=bond_distance_max_frac,
                coord_min=coord_min,
            )
            for k in ("split_peak", "lone", "low_coord"):
                rejections_total[k] += rej[k]
            positions = positions[keep]
            sigmas_fit = sigmas_fit[keep]
            amps = amps[keep]
            bgs = bgs[keep]
            coord = coord[keep]
        else:
            coord = np.zeros(positions.shape[0], dtype=np.int16)

        n_det = positions.shape[0]
        best = {
            "positions": positions, "sigmas": sigmas_fit,
            "amps": amps, "bgs": bgs, "coord": coord,
            "thr_rel": cur_thr_rel,
        }

        if not use_count_gate or n_det <= int((1.0 + count_prior_tol) * n_expected):
            break
        if attempt >= count_prior_max_retries:
            break
        cur_thr_rel *= count_prior_threshold_step
        rejections_total["count_gate_retries"] += 1

    positions = best["positions"]
    sigmas_fit = best["sigmas"]
    amps = best["amps"]
    bgs = best["bgs"]
    coord = best["coord"]

    if positions.shape[0] == 0:
        out = _empty_atom_list(n_species)
        out.filter_rejections = rejections_total
        return out

    # Stage 5 — GMM on log integrated mass.
    integrated_mass = (amps * sigmas_fit ** 3
                       * (2.0 * math.pi) ** 1.5).astype(np.float32)
    species_idx, species_prob, comp_means = _classify_gmm(
        integrated_mass, n_components=n_species, seed=seed,
    )

    return AtomList(
        positions_vox=positions.astype(np.float32),
        amplitudes=amps.astype(np.float32),
        sigmas=sigmas_fit.astype(np.float32),
        backgrounds=bgs.astype(np.float32),
        integrated_mass=integrated_mass,
        species_idx=species_idx,
        species_prob=species_prob,
        component_log_mass=comp_means,
        coord_number=coord.astype(np.int16),
        filter_rejections=rejections_total,
    )


def _run_detect_and_fit(
    volume: torch.Tensor,
    *,
    volume_np: np.ndarray,
    response: torch.Tensor,
    response_np: np.ndarray,
    sigma_at_np: np.ndarray,
    threshold_rel: float,
    min_distance: int,
    fit_window: int,
    sigma_bounds: tuple[float, float],
    cull_vox: float | None,
    vmax: float,
    rmax: float,
    n_clean_passes: int = 1,
    clean_render_sigma: float | None = None,
    sigmas_list: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stages 1–3 at a given ``threshold_rel``. Returns (positions, σ, A, b).

    When ``n_clean_passes >= 2``, runs the iterative-CLEAN detector
    (peak subtraction between passes) so dim atoms shadowed by bright
    neighbours surface in pass 2+. Falls back to single-pass NMS otherwise.
    """
    threshold = threshold_rel * rmax
    if n_clean_passes >= 2:
        rs = clean_render_sigma if clean_render_sigma is not None else 1.2
        coords_int_t, _resp_first, _sigma_at = _iterative_clean_detect(
            volume,
            sigmas_list or [],
            threshold_rel=threshold_rel,
            min_distance=min_distance,
            n_passes=n_clean_passes,
            render_sigma=rs,
        )
        if coords_int_t.shape[0] == 0:
            z = np.zeros((0,), dtype=np.float32)
            return np.zeros((0, 3), dtype=np.float32), z, z, z
    else:
        peaks_bool = _detect_atom_peaks_3d(response, min_distance, threshold)
        if not bool(peaks_bool.any()):
            z = np.zeros((0,), dtype=np.float32)
            return np.zeros((0, 3), dtype=np.float32), z, z, z
        coords_int_t = torch.nonzero(peaks_bool)

    if cull_vox is not None and cull_vox > 0:
        peaks_bool = torch.zeros_like(volume, dtype=torch.bool)
        peaks_bool[coords_int_t[:, 0], coords_int_t[:, 1],
                   coords_int_t[:, 2]] = True
        peaks_bool = _cull_isolated_peaks(peaks_bool, cull_vox)
        if not bool(peaks_bool.any()):
            z = np.zeros((0,), dtype=np.float32)
            return np.zeros((0, 3), dtype=np.float32), z, z, z
        coords_int_t = torch.nonzero(peaks_bool)

    peaks_yzx = coords_int_t.cpu().numpy().astype(np.int64)

    offsets, keep_subvox = _parabolic_subvoxel(response_np, peaks_yzx)
    positions = peaks_yzx[keep_subvox].astype(np.float32) + offsets[keep_subvox]
    sigma_init = sigma_at_np[
        peaks_yzx[keep_subvox, 0],
        peaks_yzx[keep_subvox, 1],
        peaks_yzx[keep_subvox, 2],
    ]
    if positions.shape[0] == 0:
        z = np.zeros((0,), dtype=np.float32)
        return np.zeros((0, 3), dtype=np.float32), z, z, z

    fit = _fixed_position_fit(
        volume_np, positions,
        half_window=fit_window, sigma_init=sigma_init,
        sigma_bounds=sigma_bounds,
    )
    ok = fit["ok"]
    amp_floor = threshold_rel * vmax
    ok &= fit["amplitudes"] >= amp_floor

    return (positions[ok],
            fit["sigmas"][ok],
            fit["amplitudes"][ok],
            fit["backgrounds"][ok])


def _empty_atom_list(n_species: int) -> AtomList:
    return AtomList(
        positions_vox=np.zeros((0, 3), dtype=np.float32),
        amplitudes=np.zeros(0, dtype=np.float32),
        sigmas=np.zeros(0, dtype=np.float32),
        backgrounds=np.zeros(0, dtype=np.float32),
        integrated_mass=np.zeros(0, dtype=np.float32),
        species_idx=np.zeros(0, dtype=np.int8),
        species_prob=np.zeros((0, n_species), dtype=np.float32),
        component_log_mass=np.zeros(n_species, dtype=np.float32),
    )
