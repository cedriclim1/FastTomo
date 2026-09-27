"""
Tilt-angle refinement (``refine_angles``) against a reconstructed volume.
Shift estimation lives in :mod:`fasttomo.utilities`
(``estimate_shifts_xcorr``).
"""

from __future__ import annotations

import torch

from fasttomo._backend import _assert_dtype_match
from fasttomo._operator import Geometry, _make_operator
from fasttomo.autograd import euler_zyz_to_rotmat
from fasttomo.algorithms._common import _REL_EPS


# Upper bound on one batch of candidate projections (bytes): every view's
# n_steps candidates go through one projection call, chunked over whole views
# so that chunk * n_steps * Ny * N_det elements stay below this (at 512^3
# with 61 views x 31 steps the unchunked batch would be 2 GB).
_REFINE_BATCH_BYTES = 256 << 20


def _best_candidates(op, volume, sinogram, geom_of, n_steps, normalizers):
    """Index of the lowest L1 R-factor candidate for every view, as a CPU
    tensor ``[n_views]``.

    ``geom_of(j0, j1)`` is the geometry of the candidates of views
    ``j0 .. j1-1`` (``n_steps`` consecutive entries per view);
    ``normalizers`` is ``[n_views, 1]``. The projections of many views run
    in one call. Each view's residual sum is then reduced with the same
    torch ops on a tensor of the same shape as the former per-view loop
    (``[n_steps, Ny, N_det]``, freshly allocated), so the reduction runs
    with the same configuration and gives the same bits; the division by
    the normaliser (elementwise) and the argmin (exact comparisons, first
    minimum) are batched. The chosen candidates are therefore bit-identical
    to the per-view loop, and the argmins stay on the device: one host
    transfer per call instead of one per view.
    """
    n_views, Ny, N_det = sinogram.shape
    per_view = max(1, n_steps * Ny * N_det * sinogram.element_size())
    chunk = max(1, min(n_views, _REFINE_BATCH_BYTES // per_view))
    residuals = []
    for j0 in range(0, n_views, chunk):
        j1 = min(n_views, j0 + chunk)
        projs = op.project(volume, geom_of(j0, j1)).reshape(j1 - j0, n_steps, Ny, N_det)
        for j in range(j0, j1):
            residuals.append(
                (projs[j - j0] - sinogram[j].unsqueeze(0)).abs().sum(dim=(-2, -1)))
    return torch.stack(residuals).div(normalizers).argmin(dim=1).cpu()


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

    The candidates of all projections are forward-projected in one batch
    (one per axis in ZYZ mode; split into chunks of at most 256 MB of
    projections), without autograd, and the result is copied to the host
    once per batch. The chosen angles are bit-identical to searching one
    projection at a time.

    Convergence (noise-free synthetic data, the true volume, 61 views
    at 128^3, 1 deg rms start error; phase1b audit R7): single-axis mode
    reaches 0.060 deg rms after one pass (grid step 0.2 deg) and 5.9e-5 deg
    rms with a bias of 2e-6 deg after four passes of shrinking range (3,
    0.3, 0.03, 0.003 deg). ZYZ mode converges slowly where phi and psi are
    nearly degenerate (small tilts, the worst views at 10-22 deg): 0.25 deg
    rms geodesic error after one pass (range 2 deg, 21 steps) and still
    0.092 deg rms (max 0.26 deg, 21 of 61 views above 0.05 deg) after six
    passes of halving range. A gradient-based polish (Adam on the Euler
    angles only, through ``RotmatProjector`` and its analytic dL/dM, same
    true volume) reached 0.034 deg rms in the same study; for accurate ZYZ
    poses near small tilts, prefer gradient-based pose refinement.

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
    if int(n_steps) < 1:
        raise ValueError(f"refine_angles: n_steps must be at least 1, got {n_steps}")
    n_steps = int(n_steps)
    if volume.ndim == 2:
        volume = volume.unsqueeze(0)
    if (angles_deg is None) == (euler_zyz_deg is None):
        raise ValueError(
            "Provide exactly one of angles_deg (single-axis) or "
            "euler_zyz_deg (ZYZ 3D), not both or neither."
        )

    def _operator(kind):
        # one operator (one volume texture on CUDA) for every candidate batch
        return _make_operator(
            Geometry(kind, tuple(volume.shape), sinogram.shape[-1], n_steps),
            volume.device, volume.dtype, num_threads=num_threads, name="refine_angles")

    def _window(center):
        # the candidate grid of one view: exactly the former per-view call
        return torch.linspace(center - search_range_deg, center + search_range_deg, n_steps)

    n_angles = sinogram.shape[0]
    if n_angles == 0:
        # nothing to refine: an empty [0] / [0, 3] CPU tensor, as before
        return (angles_deg if angles_deg is not None else euler_zyz_deg).detach().cpu().clone()
    # No gradients are needed: the search only compares residuals.
    with torch.no_grad():
        # per-view L1 normalisers [n_angles, 1]: each sum reduced exactly as
        # the former per-view loop did, the clamp batched (elementwise)
        normalizers = torch.stack([sinogram[j].abs().sum() for j in range(n_angles)])
        normalizers = normalizers.clamp(min=_REL_EPS).unsqueeze(1)

        if angles_deg is not None:
            # ── single-axis mode ──────────────────────────────────────────
            refined = angles_deg.detach().cpu().clone()
            cand = torch.stack([_window(refined[j].item()) for j in range(n_angles)])
            flat = cand.reshape(-1).to(sinogram.device)
            best = _best_candidates(
                _operator("parallel"), volume, sinogram,
                lambda j0, j1: flat[j0 * n_steps:j1 * n_steps], n_steps, normalizers)
            refined.copy_(cand[torch.arange(n_angles), best])
            return refined

        # ── ZYZ coordinate-descent mode ───────────────────────────────────
        # One round of coordinate descent per view: phi, then theta, then
        # psi, each searched with the others fixed at their latest value.
        # Views are independent, so every view's search along one axis runs
        # in one batch (3 batches per call instead of 3 per view).
        refined = euler_zyz_deg.detach().cpu().clone()  # [n_angles, 3]
        op = _operator("rotmat")
        for axis in range(3):  # φ, θ, ψ in sequence
            cand = torch.stack([_window(refined[j, axis].item()) for j in range(n_angles)])
            # [n_angles, n_steps, 3]: vary this axis, hold the others fixed
            euler_batch = refined.unsqueeze(1).expand(n_angles, n_steps, 3).clone()
            euler_batch[:, :, axis] = cand
            euler_dev = euler_batch.reshape(-1, 3).to(sinogram.device)
            best = _best_candidates(
                op, volume, sinogram,
                lambda j0, j1: euler_zyz_to_rotmat(euler_dev[j0 * n_steps:j1 * n_steps]),
                n_steps, normalizers)
            refined[:, axis] = cand[torch.arange(n_angles), best]
        return refined
