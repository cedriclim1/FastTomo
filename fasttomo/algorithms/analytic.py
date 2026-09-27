"""
Direct (non-iterative) reconstructions: FBP and weighted back-projection
(W-BP, exact weighting filter), plus the W-BP operator that wSIRT reuses.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn.functional as F

from fasttomo._backend import _check_fp16_range
from fasttomo._operator import Geometry, UncachedOperator, _make_operator
from fasttomo.autograd import _back_project_gather_parallel_raw, _forward_project_parallel_raw
from fasttomo.constraints import (
    _maybe_cylindrical_mask,
    _resolve_support_mask,
)
from fasttomo.utilities import _ramp_filter_coeffs, _ramp_filter_fp32
from fasttomo.configs import ConstraintConfig
from fasttomo.algorithms._common import (
    _WBP_MIN_THRESHOLD,
    _prepare_recon,
)


# ── exact weighting filter (Wolf et al. 2014) ─────────────────────────────
#
# W-BP = back-project, then deconvolve every XZ slice by the transfer
# function of the back-projection of a centred point (a 1 at detector pixel
# (N_det-1)//2 in every view; Wolf et al. 2014). Two details matter for this
# to be (close to) an inverse of the projector, and both used to be wrong:
#
# * The deconvolution is an FFT, i.e. circular, and on the bare [Nz, Nx]
#   grid wSIRT diverged at omega=1 on 32 of 45 test geometries. Two modes
#   grew: one concentrated in the outermost ~3 voxels of the volume (the
#   circular boundary sits on the volume edge, where the tight-bounds
#   forward projector barely sees the voxels), and, with many views, a
#   low-frequency one next to the missing-wedge edge. The layergram and the
#   PSF are now back-projected onto a grid with a margin of
#   `_wbp_margin(N)` = max(2, N // 16) voxels on each side of Z and X,
#   filtered there and cropped at the margin (symmetric, so exact for odd N).
#   Measured over 9 sizes x 5 tilt schemes: no divergence, and the one-step
#   error is lower than without the margin on every geometry; larger margins
#   (N/8, N/4, 2x) lose accuracy.
# * The filter assumes a shift-invariant back-projection. The CPU
#   `back_project` is ray-driven (the exact transpose of the ray-marching CPU
#   forward projector) and is not: a constant sinogram back-projects to a
#   +-34 % voxel pattern at 45 degrees, which the filter amplifies up to
#   ~40x (CPU W-BP: 83 % error vs 30 % on CUDA). W-BP therefore uses the
#   voxel-driven gather back-projection on both backends: the CUDA
#   `back_project` kernel already is one, and `_bp_gather` is the same
#   operator on CPU with exact fp32 interpolation weights (the CPU W-BP runs
#   it as the C/OpenMP kernel `back_project_gather_parallel_cpu`, bit for bit
#   `_bp_gather`, which stays as the device-generic reference and builds the
#   PSF). The CUDA texture unit interpolates with 8-bit fractional weights,
#   so the two differ by ~2e-4 (the same texture floor the forward projector
#   has at ~1.3e-4); the CPU path is the exact reference.

_WBP_CHUNK_VOXELS = 1 << 24     # padded layergram elements per Y-chunk (64 MiB fp32)


def _wbp_margin(n: int) -> int:
    """Voxels of padding on each side of a W-BP axis of length ``n``."""
    return max(2, n // 16)


def _bp_gather(
    sinogram: torch.Tensor,     # [n_angles, Ny, N_det] fp32
    angles_deg: torch.Tensor,   # [n_angles] degrees
    Nz: int,
    Nx: int,
) -> torch.Tensor:
    """Voxel-driven (gather) single-axis back-projection -> [Ny, Nz, Nx].

    Each voxel (x, z), centred coordinates, takes from every view the
    sinogram row value at detector position u = x cos + z sin + (N_det-1)/2,
    linearly interpolated, with zero outside the detector. This is the
    operator of the CUDA ``back_project_parallel_kernel`` (texture fetch at
    u + 0.5 with linear filtering and border addressing), in plain PyTorch
    so the CPU W-BP is the same operator, with exact interpolation weights
    (the texture unit's are 8-bit, ~2e-4 apart)."""
    n_angles, Ny, N_det = sinogram.shape
    device = sinogram.device
    a = angles_deg.detach().to(device=device, dtype=torch.float32).reshape(-1) * (math.pi / 180.0)
    cos_a, sin_a = torch.cos(a), torch.sin(a)
    x = torch.arange(Nx, device=device, dtype=torch.float32) - 0.5 * (Nx - 1)
    z = torch.arange(Nz, device=device, dtype=torch.float32) - 0.5 * (Nz - 1)
    # [N_det + 2, Ny]: one zero row on each side stands for "outside".
    out = torch.zeros(Nz * Nx, Ny, device=device, dtype=torch.float32)
    for k in range(n_angles):
        rows = F.pad(sinogram[k].float().t(), (0, 0, 1, 1))
        u = (x[None, :] * cos_a[k] + z[:, None] * sin_a[k] + 0.5 * (N_det - 1)).reshape(-1)
        u0 = torch.floor(u)
        w = (u - u0).unsqueeze(1)
        i0 = (u0.long() + 1).clamp_(0, N_det + 1)       # +1: the zero row
        i1 = (u0.long() + 2).clamp_(0, N_det + 1)
        out += (1.0 - w) * rows[i0] + w * rows[i1]
    return out.t().reshape(Ny, Nz, Nx).contiguous()


def _wbp_backproject(sinogram: torch.Tensor, angles_deg: torch.Tensor,
                     Nz: int, Nx: int, ops: dict | None = None,
                     num_threads: int | None = None) -> torch.Tensor:
    """The shift-invariant back-projection W-BP needs: the CUDA kernel, or
    the same gather operator on CPU (the C/OpenMP kernel, bit for bit
    `_bp_gather`; ``num_threads`` as in the Conventions). ``ops`` (CUDA)
    caches one projection operator per chunk height, so repeated calls
    (wSIRT) reuse textures."""
    if sinogram.is_cuda:
        n_angles, ny, N_det = sinogram.shape
        ops = {} if ops is None else ops
        op = ops.get(ny)
        if op is None:
            op = ops[ny] = _make_operator(
                Geometry("parallel", (ny, Nz, Nx), N_det, n_angles, angles_deg),
                sinogram.device, sinogram.dtype)
        return op.backproject(sinogram)
    return _back_project_gather_parallel_raw(sinogram, angles_deg, Nz, Nx, num_threads)


class _GatherBackProject(torch.autograd.Function):
    """CPU FBP back-projection: the C gather kernel forward (shift-invariant,
    the CUDA back-projection's operator with exact weights), and the CPU
    forward projector as its backward, the same pairing as the CUDA
    ``back_project`` (a gather whose backward is the ray-marching forward).
    ``run_fbp`` stays differentiable w.r.t. the sinogram, but its gradient is
    now approximate: the ray forward is not the exact adjoint of the gather
    (relative inner-product gap 2-5e-2 on positive uniform-random data at
    32^2 x 31 and 64^2 x 61 views, 0.3-1.3 on zero-mean random data: the
    gap is at high frequencies), where the old ray-driven CPU
    back-projection was the exact transpose of the CPU forward (gap 5e-9). CUDA has the same mismatched pairing. The
    MBIR splat forward is the gather's exact adjoint, if an exact gradient
    is ever needed. Returns fp32."""

    @staticmethod
    def forward(ctx, sinogram, angles_deg, Nz, Nx, num_threads):
        ctx.angles_deg, ctx.N_det, ctx.num_threads = angles_deg, sinogram.shape[2], num_threads
        return _back_project_gather_parallel_raw(sinogram, angles_deg, Nz, Nx, num_threads)

    @staticmethod
    def backward(ctx, grad):
        g = _forward_project_parallel_raw(grad, ctx.angles_deg, ctx.N_det, ctx.num_threads)
        return g, None, None, None, None


def _fbp_gather_backproject(sinogram: torch.Tensor, angles_deg: torch.Tensor, Nz: int,
                            Nx: int, num_threads: int | None) -> torch.Tensor:
    """`_GatherBackProject` in the sinogram's dtype. fp16 (computed in fp32
    like every CPU kernel) is checked like an fp16 projection: a result
    beyond the fp16 range raises OverflowError, which `run_fbp` catches to
    pre-scale."""
    out = _GatherBackProject.apply(sinogram, angles_deg, Nz, Nx, num_threads)
    if sinogram.dtype == torch.float16:
        out = out.to(torch.float16)
        _check_fp16_range(out, sinogram, "run_fbp")
    return out


def _compute_exact_wf(
    angles_deg: torch.Tensor,   # [n_angles] degrees
    Nz: int, Nx: int,
    N_det: int,
    device: torch.device,
    threshold_scale: float = 1.0,
) -> torch.Tensor:
    """
    Numerically compute the exact weighting filter for W-BP on the padded
    grid ``[Pz, Px] = [Nz, Nx] + 2 * margin`` (`_wbp_margin`).

    PSF: back-project the sinogram of a unit point at the rotation centre:
    a 1 at detector pixel ``(N_det - 1) // 2`` in every view. For even N_det
    that pixel is half a pixel off the centre; the filter uses |TF| only, and
    splitting the point 0.5/0.5 over the two central pixels instead adds a
    blur the data do not have (measured: worse error on every even size and
    divergence without the margin). WF = 1 / |TF(PSF)|, zero where |TF| is
    below ``threshold_scale * TF[0, 0] / n_angles`` (missing wedge).

    Returns real Fourier amplitudes ``[Pz, Px//2+1]`` to multiply against
    ``rfft2`` of a padded layergram slice.
    """
    n_angles = len(angles_deg)
    point_sino = torch.zeros(n_angles, 1, N_det, device=device)
    point_sino[:, 0, (N_det - 1) // 2] = 1.0
    Pz, Px = Nz + 2 * _wbp_margin(Nz), Nx + 2 * _wbp_margin(Nx)
    # The exact gather on both backends: the filter is a hard threshold on
    # |TF|, so a frequency sitting near it would be kept on one backend and
    # zeroed on the other if the PSF carried the CUDA texture's 8-bit
    # weights (up to 5e-2 CPU/CUDA difference measured). One Ny=1 slice,
    # built once per geometry.
    psf = _bp_gather(point_sino, angles_deg.to(device), Pz, Px)[0]   # [Pz, Px]

    TF_amp = torch.fft.rfft2(psf).abs()
    # threshold: avoid division by zero inside missing wedge
    thresh = max(threshold_scale * TF_amp[0, 0].item() / float(n_angles), _WBP_MIN_THRESHOLD)
    return torch.where(TF_amp > thresh, 1.0 / TF_amp, torch.zeros_like(TF_amp))


# The weighting filters of recent geometries (solver audit P3): building one
# is ~20 small device ops per view plus a host sync (5 ms, host-bound, at any
# volume size), 4-11x the cost of applying it, and a W-BP sweep or repeated
# wSIRT calls use the same geometry. The filter depends only on the key
# below (the angles enter `_bp_gather` as fp32 values), so a hit returns the
# identical tensor. At most _WF_CACHE_SIZE entries of [Pz, Px // 2 + 1] fp32
# on the filter's device (Pz, Px ~ 1.125 N: ~10.6 MiB per entry at 2048^2,
# 85 MiB for 8), kept until `release_cached_projectors()` clears them.
_WF_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_WF_CACHE_SIZE = 8
_WF_CACHE_LOCK = threading.Lock()


def _clear_wf_cache() -> int:
    """Drop every cached weighting filter; returns how many were kept."""
    with _WF_CACHE_LOCK:
        n = len(_WF_CACHE)
        _WF_CACHE.clear()
    return n


def _cached_exact_wf(
    angles_deg: torch.Tensor,
    Nz: int, Nx: int, N_det: int,
    device: torch.device,
    threshold_scale: float,
) -> torch.Tensor:
    """`_compute_exact_wf`, memoised per (fp32 angle values, sizes,
    threshold, device). The key reads the angles on the host (one transfer
    if they live on a GPU)."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    ang = tuple(angles_deg.detach().reshape(-1).to(device="cpu", dtype=torch.float32).tolist())
    key = (ang, int(Nz), int(Nx), int(N_det), float(threshold_scale), dev)
    with _WF_CACHE_LOCK:
        wf = _WF_CACHE.get(key)
        if wf is not None:
            _WF_CACHE.move_to_end(key)
            return wf
    wf = _compute_exact_wf(angles_deg, Nz, Nx, N_det, dev, threshold_scale)
    with _WF_CACHE_LOCK:
        _WF_CACHE[key] = wf
        while len(_WF_CACHE) > _WF_CACHE_SIZE:
            _WF_CACHE.popitem(last=False)
    return wf


def _make_wbp(
    angles_deg: torch.Tensor,
    Nz: int, Nx: int, N_det: int,
    device: torch.device,
    threshold_scale: float = 1.0,
    num_threads: int | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``wbp(sinogram) -> [Ny, Nz, Nx]``: the exact weighted
    back-projection shared by `run_weighted_backprojection` and `run_wsirt`
    (the filter is built once per geometry and kept for later calls,
    `_cached_exact_wf`). ``num_threads`` drives the CPU back-projection."""
    WF = _cached_exact_wf(angles_deg, Nz, Nx, N_det, device, threshold_scale)
    oz, ox = _wbp_margin(Nz), _wbp_margin(Nx)
    Pz, Px = Nz + 2 * oz, Nx + 2 * ox       # symmetric margin: the crop is centred for odd N too
    angles_deg = angles_deg.to(device)

    # Y-slices are independent; chunk them so the padded layergram and its
    # spectrum stay around _WBP_CHUNK_VOXELS elements.
    y_chunk = max(1, _WBP_CHUNK_VOXELS // (Pz * Px))
    ops: dict = {}      # CUDA back-projection operators by chunk height

    @torch.no_grad()
    def wbp(sinogram: torch.Tensor) -> torch.Tensor:
        Ny = sinogram.shape[1]
        out = torch.empty(Ny, Nz, Nx, dtype=torch.float32, device=sinogram.device)
        for y0 in range(0, Ny, y_chunk):
            sl = sinogram[:, y0:y0 + y_chunk]
            spec = torch.fft.rfft2(
                _wbp_backproject(sl, angles_deg, Pz, Px, ops, num_threads))   # [ny, Pz, Px//2+1]
            spec.mul_(WF)
            filtered = torch.fft.irfft2(spec, s=(Pz, Px))
            del spec
            out[y0:y0 + sl.shape[1]] = filtered[:, oz:oz + Nz, ox:ox + Nx]
            del filtered
        return out

    return wbp


# ── FBP apodisation windows (solver audit H4) ─────────────────────────────
#
# The Ram-Lak ramp passes the highest frequencies at full gain, which is
# what makes FBP noisy. A window W(f) multiplies the ramp's frequency
# response on the zero-padded grid of `_ramp_filter_coeffs` (bin k of P,
# f = k / P cycles per detector pixel, 0 <= f <= 1/2):
#   "hann":    W = 0.5 + 0.5 cos(2 pi f)     (1 at DC, 0 at Nyquist)
#   "hamming": W = 0.54 + 0.46 cos(2 pi f)   (1 at DC, 0.08 at Nyquist)
# Built in fp64 (ramp times window) and cast once, cached per
# (N_det, device, window) like the ramp itself. "none" is the bare ramp.
_FBP_WINDOWS = ("none", "hann", "hamming")
_WINDOWED_RAMP_CACHE: dict[tuple, tuple[int, torch.Tensor]] = {}


def _windowed_ramp_coeffs(N_det: int, device: torch.device, window: str
                          ) -> tuple[int, torch.Tensor]:
    """``(P, H * W)``: the ramp's padded length and fp32 frequency response
    times the window (see above)."""
    key = (int(N_det), device, window)
    hit = _WINDOWED_RAMP_CACHE.get(key)
    if hit is None:
        P, H = _ramp_filter_coeffs(N_det, torch.device("cpu"), torch.float64)
        f = torch.arange(P // 2 + 1, dtype=torch.float64) / P
        a = 0.5 if window == "hann" else 0.54
        W = a + (1.0 - a) * torch.cos(2.0 * math.pi * f)
        hit = (P, (H * W).to(device=device, dtype=torch.float32).contiguous())
        _WINDOWED_RAMP_CACHE[key] = hit
    return hit


def _fbp_filter_fp32(sinogram: torch.Tensor, window: str) -> torch.Tensor:
    """The ramp-filtered sinogram in fp32, apodised by ``window``."""
    if window == "none":
        return _ramp_filter_fp32(sinogram)
    N_det = sinogram.shape[-1]
    sino_f = sinogram.float() if sinogram.dtype != torch.float32 else sinogram
    P, HW = _windowed_ramp_coeffs(N_det, sino_f.device, window)
    f = torch.fft.rfft(sino_f, n=P, dim=-1) * HW
    return torch.fft.irfft(f, n=P, dim=-1)[..., :N_det].contiguous()


def run_fbp(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    *,
    Nz: int | None = None,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    projector: object | None = None,
    num_threads: int | None = None,
    window: str = "none",
) -> torch.Tensor:
    """Filtered back-projection.

    Ramp-filters every detector row, back-projects with the voxel-driven
    (gather, shift-invariant) back-projection on both backends and scales by
    ``pi / n_angles`` (fp16: if the unscaled back-projection would exceed
    the fp16 range, the filtered sinogram is scaled in fp32 before its cast
    to fp16 instead, so it stays in range whenever the result does).
    ``pi / n_angles`` is pi times the mean over the measured
    views: it keeps the mean level correct at limited tilt too, but
    anisotropic objects still show an orientation-dependent bias there
    (the missing wedge), which no angular scale removes. Shapes, dtypes (fp32/fp16), the support contract and
    the return value follow the Conventions of :mod:`fasttomo.algorithms`;
    the support is applied once, to the final volume
    (``run_fbp(s, a, support_mask=M) == run_fbp(s, a) * M`` bit for bit).

    Parameters
    ----------
    sinogram, angles_deg
        ``[n_angles, Ny, N_det]`` and ``[n_angles]`` degrees.
    Nz, positivity, mask, support_mask, num_threads
        See Conventions.
    projector : ParallelProjector, optional
        Reuse its texture cache across calls (a tilt-series sweep on CUDA).
        Its angles and volume shape replace ``angles_deg`` and ``Nz`` for the
        back-projection. Numerics are identical to the call without it.
    window : {"none", "hann", "hamming"}
        Apodisation of the ramp filter (default ``"none"``: the bare Ram-Lak
        ramp, as before). With f the frequency in cycles per detector pixel
        (0 to 1/2), ``"hann"`` multiplies the ramp by ``0.5 + 0.5 cos(2 pi
        f)`` and ``"hamming"`` by ``0.54 + 0.46 cos(2 pi f)``: less noise
        and ringing, a little blur. The mean level (DC) is unchanged.

    Returns
    -------
    volume : Tensor ``[Ny, Nz, Nx]`` in the sinogram's dtype.

    Notes
    -----
    fp16 runs the ramp filter in fp32 (cuFFT needs power-of-two sizes for
    fp16) and the back-projection in fp16: faster, visibly noisier on edges;
    use fp32 for quantitative work.
    """
    if support_mask is not None and mask:
        raise ValueError(
            "run_fbp: mask=True and support_mask are mutually exclusive — "
            "pass one support per reconstruction."
        )
    if window not in _FBP_WINDOWS:
        raise ValueError(
            f"run_fbp: window must be one of {_FBP_WINDOWS}, got {window!r}")
    # Validation only: the support is resolved below against the actual
    # output shape (a cached projector may reconstruct with Nz != N_det).
    setup = _prepare_recon(
        "run_fbp", sinogram, angles_deg, Nz=Nz,
        allowed_dtypes=(torch.float32, torch.float16), init_volume=False,
    )
    n_angles, Ny, N_det, Nz, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.device)
    # angular sampling normalisation: π / n_angles (π times the mean over
    # the measured views; keeps the mean level right at limited tilt too)
    scale = math.pi / n_angles
    filtered32 = _fbp_filter_fp32(sinogram, window)
    # one back-projection: the one-shot (scoped-texture) path on CUDA
    op = _make_operator(
        Geometry("parallel", (Ny, Nz, N_det), N_det, n_angles, angles_deg),
        device, sinogram.dtype, projector, cached=False, num_threads=num_threads)
    backproject = op.backproject
    # CPU (own operator or a CPU ParallelProjector): the shift-invariant
    # gather, like CUDA and the CPU W-BP, with that operator's geometry and
    # threads (the CPU operator's ray-driven transpose puts a 2-5 % voxel
    # pattern on a flat field: 0.5-1.1 % RMS error in a uniform cylinder)
    cpu_op = op if projector is None else getattr(projector, "_op", None)
    if (isinstance(cpu_op, UncachedOperator) and cpu_op.kind == "parallel"
            and cpu_op.device.type == "cpu"):
        def backproject(s):
            cpu_op._check_device(s, "sinogram")
            cpu_op._check_shape(s, "sinogram", cpu_op.sino_shape)
            cpu_op._lock_dtype(s)
            return _fbp_gather_backproject(s, cpu_op._angles(None), cpu_op.Nz, cpu_op.Nx,
                                           cpu_op.num_threads)
    if sinogram.dtype != torch.float16:
        volume = backproject(filtered32) * scale
    else:
        # The back-projection sums n_angles filtered views before the
        # scale, so its fp16 result is n / pi times the answer and can
        # overflow (OverflowError) although the answer fits (density 400
        # over 720 views). Then, and only then, scale the filtered sinogram
        # in fp32 before its cast to fp16 and back-project again. Always
        # pre-scaling would instead push small filtered values into the
        # fp16 subnormal range (a density of 1e-4 at 720 views: 8 %
        # error, 1e-5: all lost, against 0.08 % and 0.4 % this way).
        try:
            volume = backproject(filtered32.to(torch.float16)) * scale
        except OverflowError:
            volume = backproject((filtered32 * scale).to(torch.float16))
    if support_mask is not None:
        # Validate against the actual output shape — a cached projector may
        # reconstruct with Nz != N_det.
        _mask = _resolve_support_mask(
            support_mask,
            volume.shape[0], volume.shape[1], volume.shape[2],
            volume.device, volume.dtype, "run_fbp",
        )
    else:
        _mask = _maybe_cylindrical_mask(
            volume.shape[1],
            volume.shape[2],
            volume.device,
            mask,
            dtype=volume.dtype,
        )
    return ConstraintConfig(positivity=positivity, support_mask=_mask).apply(volume)


def run_weighted_backprojection(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    *,
    Nz: int | None = None,
    positivity: bool = False,
    threshold_scale: float = 1.0,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    num_threads: int | None = None,
) -> torch.Tensor:
    """Exact weighted back-projection (W-BP, Wolf et al. 2014). Also
    available as ``run_wbp`` (the same function).

    Back-projects, then deconvolves every XZ slice by the numerically
    computed exact weighting filter (the inverse transfer function of the
    back-projection of a central point, zeroed inside the missing wedge
    below ``threshold_scale * TF[0, 0] / n_angles``). The back-projection
    and the filter run on a grid with a margin of max(2, N // 16) voxels on
    each side of Z and X (the circular deconvolution otherwise destabilises
    the volume edge) and use the voxel-driven back-projection on both
    backends, so CPU and CUDA agree up to the CUDA texture interpolation;
    see the notes above `_bp_gather`. The weighting filter is built once per
    geometry (angles, sizes, ``threshold_scale``, device) and kept for later
    calls (``run_wsirt`` too): up to 8 filters of ``[Pz, Px // 2 + 1]`` fp32
    on their device (~10.6 MiB each at 2048^2), until
    :func:`release_cached_projectors`. Extra memory: one padded layergram and
    its spectrum per Y-chunk (at most ~64 Mi elements). fp32 only.

    Parameters
    ----------
    sinogram, angles_deg
        ``[n_angles, Ny, N_det]`` fp32 and ``[n_angles]`` degrees.
    Nz, positivity, mask, support_mask, num_threads
        See Conventions of :mod:`fasttomo.algorithms` (support applied once,
        to the final volume; ``num_threads`` drives the CPU back-projection).
    threshold_scale : float
        Missing-wedge threshold of the filter (default 1.0).

    Returns
    -------
    volume : Tensor ``[Ny, Nz, Nx]`` fp32.
    """
    setup = _prepare_recon(
        "run_weighted_backprojection", sinogram, angles_deg, Nz=Nz,
        init_volume=False, positivity=positivity, mask=mask, support_mask=support_mask,
    )
    N_det, Nz, Nx, device = setup.N_det, setup.Nz, setup.Nx, setup.device

    volume = _make_wbp(angles_deg, Nz, Nx, N_det, device, threshold_scale,
                       num_threads)(sinogram)

    return setup.constraints.apply(volume)


# Short name (same function object, not deprecated): ``run_wbp`` is what the
# W-BP algorithm is usually called.
run_wbp = run_weighted_backprojection
