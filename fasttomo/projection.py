"""
Projection primitives for FastTomo.

Thin shims around torch.autograd.Function subclasses defined in
fasttomo/autograd.py. When inputs do not require grad, Function.apply
short-circuits and behaves identically to a direct kernel call. When a
volume (or sinogram) has requires_grad=True, loss.backward() flows the
adjoint partner kernel back into .grad.

Every op raises ValueError, on CPU and CUDA alike, when a dimension of its
input or output (n_angles, Ny, Nz, Nx, N_det) is zero.

See fasttomo/autograd.py and fasttomo-wiki/wiki/concepts/autograd-projectors.md.
"""

from __future__ import annotations

import torch

from fasttomo.autograd import (
    _FPParallel, _BPParallel, _FPRotmat, _BPRotmat, _FPResire, _BPResire,
    _euler_zyz_to_rotmat, _resolve_num_threads,
)
from fasttomo.utilities import apply_shifts_2d


def forward_project(
    volume: torch.Tensor,          # [Ny, Nz, Nx]
    angles_deg: torch.Tensor,      # [n_angles] degrees
    N_det: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    """
    Forward-project volume → sinogram.

    Returns [n_angles, Ny, N_det] float32 on the same device as volume.
    N_det defaults to Nx.

    The CPU path is OpenMP-parallel; pass ``num_threads=N`` to opt in. Default
    is serial. Resolution order: kwarg → ``OMP_NUM_THREADS`` env → 1. The CUDA
    path ignores ``num_threads`` (the kernel manages its own grid).

    Differentiable w.r.t. ``volume`` and ``angles_deg``.  Volume gradient
    calls back_project (the discrete adjoint). Angle gradient uses
    ∂sino/∂θ = ∂vol/∂x·(−z) + ∂vol/∂z·x at each ray step (CPU + CUDA).
    """
    if volume.ndim == 2:
        volume = volume.unsqueeze(0)
    nt = _resolve_num_threads(num_threads)
    return _FPParallel.apply(volume, angles_deg, N_det, nt)


def back_project(
    sinogram: torch.Tensor,        # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,      # [n_angles] degrees
    Nz: int = -1,
    num_threads: int | None = None,
    *,
    Nx: int = -1,
) -> torch.Tensor:
    """
    Back-project sinogram → volume [Ny, Nz, Nx].

    Nz and Nx default to N_det (square reconstruction). Pass ``Nx`` (and
    ``Nz``) to back-project onto a volume whose width differs from the
    detector; this is the adjoint of ``forward_project(volume, angles, N_det)``
    for a ``[Ny, Nz, Nx]`` volume, and the backward of ``forward_project``
    uses it automatically.

    The CPU path is OpenMP-parallel; pass ``num_threads=N`` to opt in (default
    serial; resolution order kwarg → ``OMP_NUM_THREADS`` env → 1).

    Differentiable w.r.t. ``sinogram``: backward calls forward_project.
    """
    if sinogram.ndim == 2:
        sinogram = sinogram.unsqueeze(1)
    nt = _resolve_num_threads(num_threads)
    return _BPParallel.apply(sinogram, angles_deg, Nz, nt, Nx)


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
    sino = _FPRotmat.apply(volume, mats, N_det, nt)
    if shifts is not None:
        sino = apply_shifts_2d(sino, shifts)
    return sino


def back_project_3d(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    euler_zyz_deg: torch.Tensor,    # [n_angles, 3] — (φ, θ, ψ) degrees
    Nz: int = -1,
    num_threads: int | None = None,
    *,
    Nx: int = -1,
) -> torch.Tensor:
    """
    3D rotation-matrix back-projection (CPU + CUDA).

    Adjoint of forward_project_3d. Voxel-driven gather: each voxel samples
    the sinogram at its rotated detector position for every projection.

    Returns volume [Ny, Nz, Nx] on the same device as ``sinogram``; ``Nz``
    and ``Nx`` default to N_det (pass ``Nx`` when the volume is not as wide
    as the detector). Pass ``num_threads=N`` to opt into OpenMP parallelism on the CPU path
    (default 1 — serial). Resolution order: kwarg → OMP_NUM_THREADS → 1.

    Differentiable w.r.t. ``sinogram``: backward calls forward_project_3d.
    """
    if sinogram.ndim == 2:
        sinogram = sinogram.unsqueeze(1)
    nt = _resolve_num_threads(num_threads)
    return _BPRotmat.apply(sinogram, euler_zyz_deg, Nz, nt, Nx)


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
    *,
    Nx: int = -1,
) -> torch.Tensor:
    """
    RESIRE voxel-driven back-projection (CUDA only).

    Adjoint of forward_project_resire. Each voxel gathers from the sinogram
    at its rotated detector position with 4-point supersampling. Output is
    normalised by the supersampling factor (÷4).

    Returns volume [Ny, Nz, Nx]; ``Nz`` and ``Nx`` default to N_det. Pass
    ``Nx`` when the volume is not as wide as the detector (the backward of
    ``forward_project_resire`` does this automatically).

    Differentiable w.r.t. ``sinogram``: backward calls forward_project_resire.
    """
    if sinogram.ndim == 2:
        sinogram = sinogram.unsqueeze(1)
    return _BPResire.apply(sinogram, euler_zyz_deg, Nz, Nx)
