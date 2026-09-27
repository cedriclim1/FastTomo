"""
Tilt-angle refinement (``refine_angles``) against a reconstructed volume.
Shift estimation lives in :mod:`fasttomo.utilities`
(``estimate_shifts_xcorr``).
"""

from __future__ import annotations

import torch

from fasttomo._backend import _assert_dtype_match
from fasttomo.projection import (
    forward_project,
    forward_project_3d,
)
from fasttomo.algorithms._common import _REL_EPS


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
    """Angular refinement by brute-force search (RESIRE style). Not a reconstruction.

    Single-axis mode (``angles_deg``): for each projection ``j``, tries
    ``n_steps`` angles in ``angle_j +- search_range_deg`` and keeps the one
    with the lowest L1 R-factor against ``sinogram[j]``. Returns refined
    angles ``[n_angles]`` on CPU.

    ZYZ mode (``euler_zyz_deg``): one round of coordinate descent over
    ``(phi, theta, psi)`` per projection, each axis searched like above with
    the others fixed. Returns refined ``[n_angles, 3]`` on CPU.

    Parameters
    ----------
    volume : Tensor ``[Ny, Nz, Nx]``
        Current reconstruction (same dtype as ``sinogram``).
    sinogram : Tensor ``[n_angles, Ny, N_det]``
    angles_deg : Tensor ``[n_angles]``, optional
        Single-axis angles; exactly one of ``angles_deg`` and
        ``euler_zyz_deg`` must be given.
    euler_zyz_deg : Tensor ``[n_angles, 3]``, optional (keyword-only)
    search_range_deg : float
        Half-width of the search window (default 3.0).
    n_steps : int
        Candidates per window (default 31).
    num_threads : int, optional
        CPU threads for the forward projections (Conventions).

    Examples
    --------
    ::

        angles = initial_angles
        for _ in range(n_outer):
            vol = ft.run_resire(sino, angles, 100)
            angles = ft.refine_angles(vol, sino, angles)

        euler = torch.zeros(n_angles, 3); euler[:, 1] = tilt_angles
        for _ in range(n_outer):
            vol = ft.run_resire(sino, tilt_angles, 100, euler_zyz_deg=euler)
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
