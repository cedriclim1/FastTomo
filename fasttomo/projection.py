"""
Projection primitives for FastTomo.

Each op builds a transient projection operator for its inputs' geometry
(`fasttomo._operator._make_operator`, uncached adapter: CPU kernels, or a
one-shot CUDA texture) and calls its differentiable ``project`` /
``backproject``. When inputs do not require grad this behaves identically
to a direct kernel call; when a volume (or sinogram, or the angles) has
requires_grad=True, loss.backward() flows the adjoint partner kernel back
into .grad.

Every op raises ValueError, on CPU and CUDA alike, when a dimension of its
input or output (n_angles, Ny, Nz, Nx, N_det) is zero.

See fasttomo/autograd.py and fasttomo-wiki/wiki/concepts/autograd-projectors.md.
"""

from __future__ import annotations

import math

import torch

from fasttomo._operator import Geometry, _make_operator
from fasttomo.autograd import (
    _FPResire, _BPResire, euler_zyz_to_rotmat, _resolve_num_threads,
)

__all__ = [
    "forward_project",
    "back_project",
    "forward_project_3d",
    "back_project_3d",
    "forward_project_resire",
    "back_project_resire",
    "euler_zyz_to_rotmat",
]


# ── detector-plane shifts (used by forward_project_3d's ``shifts``) ─────────
#
# Moved from fasttomo.utilities (ARCH P8), which re-exports them.
#
# One contract for both functions (they differ only in the interpolation):
#
# * ``sinogram``: ``[n_angles, Ny, N_det]``; the result has the same shape,
#   device and dtype (fp16 in, fp16 out; the arithmetic runs in fp32, or in
#   fp64 for fp64 input).
# * ``shifts``: ``[n_angles]`` (detector axis only) or ``[n_angles, 2]`` with
#   columns ``(detector axis, tilt axis)``, i.e. ``(N_det, Ny)``, in pixels;
#   it may live on another device than the sinogram.
# * Sign: a positive shift moves the signal toward larger indices (right
#   along the detector, down along the tilt axis):
#   ``out[k, y, u] = sinogram[k, y - shifts[k, 1], u - shifts[k, 0]]``.
# * ``apply_shifts``: Fourier phase shift, band-limited and circular (the
#   signal wraps around). ``apply_shifts_2d``: bilinear ``grid_sample`` with
#   zero fill (nothing wraps), differentiable w.r.t. sinogram and shifts,
#   the pose model of ``forward_project_3d(shifts=)``. Both agree on integer
#   shifts away from the edges.

def _shift_table(sinogram: torch.Tensor, shifts: torch.Tensor, name: str) -> torch.Tensor:
    """Validate the shared contract; return ``shifts`` on the sinogram's
    device as ``[n_angles]`` or ``[n_angles, 2]`` in the compute dtype."""
    if sinogram.ndim != 3:
        raise ValueError(f"{name}: sinogram must be [n_angles, Ny, N_det], "
                         f"got shape {tuple(sinogram.shape)}")
    if not (shifts.ndim == 1 or (shifts.ndim == 2 and shifts.shape[-1] == 2)):
        raise ValueError(f"{name}: shifts must have shape [n_angles] or "
                         f"[n_angles, 2], got {tuple(shifts.shape)}")
    n_a = sinogram.shape[0]
    if shifts.shape[0] != n_a:
        raise ValueError(f"{name}: shifts has {shifts.shape[0]} entries but "
                         f"sinogram has {n_a} angles")
    return shifts.to(device=sinogram.device, dtype=_shift_compute_dtype(sinogram))


def _shift_compute_dtype(sinogram: torch.Tensor) -> torch.dtype:
    """fp32 arithmetic for fp16/fp32 sinograms, fp64 for fp64 ones."""
    return torch.float64 if sinogram.dtype == torch.float64 else torch.float32


def apply_shifts(
    sinogram: torch.Tensor,   # [n_angles, Ny, N_det]
    shifts: torch.Tensor,     # [n_angles] or [n_angles, 2] (det-axis, tilt-axis) pixels
) -> torch.Tensor:            # [n_angles, Ny, N_det], sinogram's device and dtype
    """
    Sub-pixel shift of every projection by Fourier phase multiplication
    (band-limited, circular: what leaves one edge re-enters at the other).

    ``shifts`` is ``[n_angles]`` (detector axis) or ``[n_angles, 2]``
    (detector axis, tilt axis) in pixels; a positive shift moves the signal
    toward larger indices. Same contract as :func:`apply_shifts_2d` (see the
    comment above both); returns the sinogram's shape, device and dtype.
    ``shifts`` may be on CPU even when ``sinogram`` is on CUDA.
    """
    s = _shift_table(sinogram, shifts, "apply_shifts")
    dt = _shift_compute_dtype(sinogram)
    s_det = s if s.ndim == 1 else s[:, 0]
    N_det = sinogram.shape[-1]
    freqs = torch.fft.rfftfreq(N_det, device=sinogram.device, dtype=dt)   # [N_freq]
    phase_arg = -2.0 * math.pi * freqs.view(1, 1, -1) * s_det.view(-1, 1, 1)
    phase_mul = torch.polar(torch.ones_like(phase_arg), phase_arg)
    F = torch.fft.rfft(sinogram.to(dt), n=N_det, dim=-1)
    out = torch.fft.irfft(F * phase_mul, n=N_det, dim=-1)
    if s.ndim == 2:
        # tilt axis: complex FFT along Ny (rfft needs the last dim)
        Ny = sinogram.shape[1]
        fy = torch.fft.fftfreq(Ny, device=sinogram.device, dtype=dt)       # [Ny]
        arg_y = -2.0 * math.pi * fy.view(1, -1, 1) * s[:, 1].view(-1, 1, 1)
        Fy = torch.fft.fft(out, dim=1) * torch.polar(torch.ones_like(arg_y), arg_y)
        out = torch.fft.ifft(Fy, dim=1).real
    return out.to(sinogram.dtype)


def apply_shifts_2d(
    sinogram: torch.Tensor,   # [n_angles, Ny, N_det]
    shifts: torch.Tensor,     # [n_angles, 2] (det-axis, tilt-axis) or [n_angles] pixels
) -> torch.Tensor:            # [n_angles, Ny, N_det], sinogram's device and dtype
    """
    Per-tilt sub-pixel translation of the sinogram in the detector plane via
    bilinear ``grid_sample`` with zero fill.  Differentiable w.r.t. both
    ``sinogram`` and ``shifts`` — autograd flows through ``grid_sample``
    natively.

    ``shifts[:, 0]`` shifts along the detector axis (last dim, ``N_det``);
    ``shifts[:, 1]`` shifts along the tilt axis (``Ny``); a 1-D ``[n_angles]``
    tensor shifts along the detector axis only.  Positive shift moves the
    signal toward larger indices (right / down).  Same contract as
    :func:`apply_shifts` (see the comment above both); returns the
    sinogram's shape, device and dtype.

    Intended for joint optimization with the volume and pose: drop in after
    ``forward_project_3d`` (or pass ``shifts=`` to it directly) so per-tilt
    translational alignment becomes a learnable parameter.
    """
    s = _shift_table(sinogram, shifts, "apply_shifts_2d")
    if s.ndim == 1:
        s = torch.stack((s, torch.zeros_like(s)), dim=-1)
    n_a, Ny, Nd = sinogram.shape
    dt = _shift_compute_dtype(sinogram)

    ys = torch.linspace(-1.0, 1.0, Ny, device=sinogram.device, dtype=dt)
    xs = torch.linspace(-1.0, 1.0, Nd, device=sinogram.device, dtype=dt)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")                  # [Ny, Nd]
    base = torch.stack((gx, gy), dim=-1)                            # [Ny, Nd, 2]
    base = base.unsqueeze(0).expand(n_a, -1, -1, -1)                # [n_a, Ny, Nd, 2]

    # Sample from (x - sx, y - sy) so positive shift moves the signal +x / +y.
    # Normalized coords with align_corners=True: 1 pixel = 2/(N-1).
    norm_x = 2.0 / max(Nd - 1, 1)
    norm_y = 2.0 / max(Ny - 1, 1)
    off = torch.stack((s[:, 0] * norm_x, s[:, 1] * norm_y), dim=-1) # [n_a, 2]
    grid = base - off.view(n_a, 1, 1, 2)                            # [n_a, Ny, Nd, 2]

    inp = sinogram.unsqueeze(1).to(dt)                              # [n_a, 1, Ny, Nd]
    out = torch.nn.functional.grid_sample(
        inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return out.squeeze(1).to(sinogram.dtype)


# ── projection ops ─────────────────────────────────────────────────────────

def _size(value: int | None) -> int:
    """A size argument of the public API as the raw kernel layer takes it:
    ``None`` (the default, "derive it from the input") becomes ``-1``, the
    binding-level sentinel; negative values are accepted with the same
    meaning (the former ``-1`` defaults)."""
    return -1 if value is None else int(value)


def _transient(kind, shape, N_det, device, num_threads, angles_deg=None, *, name):
    """The one-shot operator behind the public function ``name``: uncached,
    with the volume dims the call knows (``-1``: the raw kernels derive them
    from the input, as they always did). ``name`` heads its error messages."""
    Nz, Nx = int(shape[-2]), int(shape[-1])
    return _make_operator(
        Geometry(kind, (-1, Nz, Nx), int(N_det), -1, angles_deg),
        device, cached=False, num_threads=_resolve_num_threads(num_threads), name=name)


def forward_project(
    volume: torch.Tensor,          # [Ny, Nz, Nx]
    angles_deg: torch.Tensor,      # [n_angles] degrees
    N_det: int | None = None,
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
    op = _transient("parallel", volume.shape, _size(N_det), volume.device, num_threads,
                    angles_deg=angles_deg, name="forward_project")
    return op.project(volume, angles_deg)


def back_project(
    sinogram: torch.Tensor,        # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,      # [n_angles] degrees
    Nz: int | None = None,
    num_threads: int | None = None,
    *,
    Nx: int | None = None,
) -> torch.Tensor:
    """
    Back-project sinogram → volume [Ny, Nz, Nx].

    Nz and Nx default to N_det (square reconstruction). Pass ``Nx`` (and
    ``Nz``) to back-project onto a volume whose width differs from the
    detector; the backward of ``forward_project(volume, angles, N_det)``
    uses it automatically for a ``[Ny, Nz, Nx]`` volume.

    The two backends implement different back-projections:

    * CPU: ray-driven scatter, the exact transpose of the CPU
      ``forward_project`` (inner-product test to fp32 round-off). It is not
      shift-invariant: a constant sinogram back-projects to a voxel pattern
      of up to +-34 % in a single 45 degree view.
    * CUDA: voxel-driven gather; every voxel linearly interpolates each
      view's sinogram row at its detector position (texture unit, 8-bit
      interpolation weights). Shift-invariant, but only an approximate
      adjoint of the CUDA ``forward_project`` (~1 % inner-product error on
      smooth data, more on noise).

    The CPU path is OpenMP-parallel; pass ``num_threads=N`` to opt in (default
    serial; resolution order kwarg → ``OMP_NUM_THREADS`` env → 1).

    Differentiable w.r.t. ``sinogram``: backward calls forward_project.
    """
    if sinogram.ndim == 2:
        sinogram = sinogram.unsqueeze(1)
    op = _transient("parallel", (-1, _size(Nz), _size(Nx)), sinogram.shape[-1],
                    sinogram.device, num_threads, angles_deg=angles_deg, name="back_project")
    return op.backproject(sinogram, angles_deg)


def forward_project_3d(
    volume: torch.Tensor,                       # [Ny, Nz, Nx]
    euler_zyz_deg: torch.Tensor,                # [n_angles, 3] — (φ, θ, ψ) degrees
    N_det: int | None = None,
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
    mats = euler_zyz_to_rotmat(euler_zyz_deg)   # [n_angles, 3, 3]; autograd traces through
    op = _transient("rotmat", volume.shape, _size(N_det), volume.device, num_threads,
                    name="forward_project_3d")
    sino = op.project(volume, mats)
    if shifts is not None:
        sino = apply_shifts_2d(sino, shifts)
    return sino


def back_project_3d(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    euler_zyz_deg: torch.Tensor,    # [n_angles, 3] — (φ, θ, ψ) degrees
    Nz: int | None = None,
    num_threads: int | None = None,
    *,
    Nx: int | None = None,
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
    # no gradient w.r.t. the angles through a back-projection
    mats = euler_zyz_to_rotmat(euler_zyz_deg.detach())
    op = _transient("rotmat", (-1, _size(Nz), _size(Nx)), sinogram.shape[-1],
                    sinogram.device, num_threads, name="back_project_3d")
    return op.backproject(sinogram, mats)


def forward_project_resire(
    volume: torch.Tensor,           # [Ny, Nz, Nx]
    euler_zyz_deg: torch.Tensor,    # [n_angles, 3] — (φ, θ, ψ) degrees
    N_det: int | None = None,
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
    return _FPResire.apply(volume, euler_zyz_deg, _size(N_det))


def back_project_resire(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    euler_zyz_deg: torch.Tensor,    # [n_angles, 3] — (φ, θ, ψ) degrees
    Nz: int | None = None,
    *,
    Nx: int | None = None,
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
    return _BPResire.apply(sinogram, euler_zyz_deg, _size(Nz), _size(Nx))
