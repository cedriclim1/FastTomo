"""
Model-Based Iterative Reconstruction (ICD, Huber prior, per-view offsets).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

import fasttomo._core as _core
from fasttomo._backend import (
    _assert_dtype_match, _check_shapes, _data_ptr, _resolve_num_threads,
)
from fasttomo.autograd import _launch
from fasttomo.configs import MbirConfig
from fasttomo.algorithms._common import (
    _REL_EPS,
    _MBIR_CONVERGENCE_STRIDE,
    _MBIR_WEIGHT_EPS,
    _MBIR_BLANK_STD,
    _MBIR_SIGMA_MIN_RATIO,
    _is_default,
    _deprecated_kwarg,
    _reject_unknown_kwargs,
    _as_int,
    _prepare_recon,
    _LossRecorder,
)


# ── MBIR ───────────────────────────────────────────────────────────────────
# Classical Model-Based Iterative Reconstruction for single-axis parallel-beam
# tomography. Reference: Venkatakrishnan et al. 2015 (IEEE TCI 1(1)). The
# kernels are in csrc/{cuda,cpu}/mbir_icd.{cu,c}; see
# fasttomo-wiki/wiki/algorithms/mbir.md for the surrogate-MM derivation.


@dataclass
class _MbirState:
    """One MBIR optimization level's running state.

    Maintained by `_run_mbir_single_scale` and passed across multires
    levels so the per-tilt offset survives upsampling. (σ is re-initialized
    per scale from that scale's sinogram std; carrying the coarse level's
    σ over converged no better, see ``multires_levels`` in `run_mbir`.)
    """
    volume: torch.Tensor          # [Ny, Nz, Nx] float32
    d_offsets: torch.Tensor       # [n_angles] float32
    sigma: float
    e_sino: torch.Tensor | None   # [n_angles, Ny, N_det] float32 — residual (None: not yet computed)
    n_done: int = 0               # outer iterations run at this scale


def _huber_surr_weight(r_abs: torch.Tensor, T: float, delta: float) -> torch.Tensor:
    """Per-element Huber surrogate weight matching `huber_surr_weight_dev` in
    the CUDA kernel and `huber_surr_weight` in the CPU reference. When
    T = +inf the weight collapses to ones (conventional MBIR, no anomaly
    suppression)."""
    if not math.isfinite(T):
        return torch.ones_like(r_abs)
    inside = r_abs < T
    outside_w = delta * T / r_abs.clamp(min=_MBIR_WEIGHT_EPS)
    return torch.where(inside, torch.ones_like(r_abs), outside_w)


def _mbir_update_offset_variance(
    e_sino: torch.Tensor,
    d_offsets: torch.Tensor,
    sigma: float,
    sigma_min: float,
    T: float,
    delta: float,
    lambdas: torch.Tensor | None,
    estimate_offset: bool,
    estimate_variance: bool,
) -> tuple[torch.Tensor, float]:
    """Closed-form updates for d_k and σ under the quadratic surrogate.

    The only implementation: the CPU and CUDA sweeps run the ICD voxel
    updates only, and `_run_mbir_single_scale` calls this afterwards on the
    live e_sino for both backends (the C copy was removed, ARCH P12).

        Δd_k = Σ_i b·Λ·e / Σ_i b·Λ   (per tilt k; skipped if Σ b·Λ < 1e-30)
        σ    = sqrt( Σ b·Λ·e² / (n_angles·Ny·N_det) ), floored at sigma_min

    with b the Huber surrogate weight of r = e·√Λ/σ, recomputed after the
    offset update. Element products are fp32; the sums accumulate in fp64
    on both backends (large M·K would otherwise lose digits).

    Returns (d_offsets_new, sigma_new) and mutates e_sino in place to
    maintain the invariant e = g - A·f - d.
    """
    if not (estimate_offset or estimate_variance):
        return d_offsets, sigma
    n_angles, Ny, N_det = e_sino.shape
    if lambdas is None:
        r = e_sino / sigma
    else:
        r = e_sino * lambdas.sqrt() / sigma
    r_abs = r.abs()
    bw = _huber_surr_weight(r_abs, T, delta)
    lam = lambdas if lambdas is not None else torch.ones_like(e_sino)

    if estimate_offset:
        num = (bw * lam * e_sino).sum(dim=(1, 2), dtype=torch.float64)
        den = (bw * lam).sum(dim=(1, 2), dtype=torch.float64)
        delta_d = torch.where(den >= _MBIR_WEIGHT_EPS, num / den.clamp(min=_MBIR_WEIGHT_EPS),
                              torch.zeros_like(num)).float()
        d_offsets = d_offsets + delta_d
        e_sino.sub_(delta_d.view(n_angles, 1, 1))
        # recompute weights for the variance update on the updated residual
        if estimate_variance:
            if lambdas is None:
                r = e_sino / sigma
            else:
                r = e_sino * lambdas.sqrt() / sigma
            r_abs = r.abs()
            bw = _huber_surr_weight(r_abs, T, delta)

    sigma_new = sigma
    if estimate_variance:
        total = n_angles * Ny * N_det
        s2 = (bw * lam * e_sino * e_sino).sum(dtype=torch.float64) / total
        sigma_new = max(sigma_min, float(s2.clamp(min=0.0).sqrt().float().item()))

    return d_offsets, sigma_new


_M64 = (1 << 64) - 1


def _mix64(z: int) -> int:
    """splitmix64 finalizer on a Python int (mod 2**64); ``mbir_mix64`` in
    csrc/cuda/mbir_icd.cu."""
    z = (z + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def _mbir_perm_key(seed: int, it: int) -> int:
    """64-bit key of the voxel order of outer iteration ``it`` for ``seed``."""
    return _mix64((_mix64(int(seed) & _M64) + int(it)) & _M64)


def _hash32(x: np.ndarray) -> np.ndarray:
    """lowbias32 on a uint32 array (wrapping); ``mbir_hash32`` on the GPU."""
    x = x ^ (x >> np.uint32(16))
    x = x * np.uint32(0x7FEB352D)
    x = x ^ (x >> np.uint32(15))
    x = x * np.uint32(0x846CA68B)
    return x ^ (x >> np.uint32(16))


def _mbir_permutation(Ny: int, Nvox: int, key: int) -> np.ndarray:
    """Per-Y-slice voxel order, int32 ``[Ny, Nvox]``: row ``iy`` is a
    pseudo-random permutation of ``range(Nvox)`` given by a keyed 4-round
    Feistel network on ``2h`` bits (``4**h >= Nvox``) with cycle-walking.

    The CUDA sweep generates the same order inside the kernel (``mbir_perm_*``
    in csrc/cuda/mbir_icd.cu) instead of receiving this array, so nothing is
    copied to the device per outer iteration; this NumPy mirror feeds the
    CPU sweep and is bit-identical to it (tested). Distinct keys give
    distinct orders across slices and iterations."""
    h = 0
    while (1 << (2 * h)) < Nvox:
        h += 1
    mask = np.uint32((1 << h) - 1)
    sh = np.uint32(h)
    rk = np.empty((Ny, 4), dtype=np.uint32)
    for iy in range(Ny):
        k1 = _mix64((key + iy) & _M64)
        k2 = _mix64(k1)
        rk[iy] = (k1 & 0xFFFFFFFF, k1 >> 32, k2 & 0xFFFFFFFF, k2 >> 32)

    def feistel(x: np.ndarray, rows: np.ndarray | slice) -> np.ndarray:
        L, R = x >> sh, x & mask
        for r in range(4):
            k = rk[rows, r] if isinstance(rows, np.ndarray) else rk[:, r:r + 1]
            L, R = R, L ^ (_hash32(R ^ k) & mask)
        return (L << sh) | R

    y = feistel(np.broadcast_to(np.arange(Nvox, dtype=np.uint32), (Ny, Nvox)), slice(None))
    rows, cols = np.nonzero(y >= np.uint32(Nvox))
    while rows.size:                        # cycle-walk back into [0, Nvox)
        y[rows, cols] = feistel(y[rows, cols], rows)
        still = y[rows, cols] >= np.uint32(Nvox)
        rows, cols = rows[still], cols[still]
    return y.astype(np.int32)


_SWEEP_KERNELS = ("auto", "serial", "parallel")


def _mbir_group_spacing(cos_a: torch.Tensor, Nz: int, Nx: int, N_det: int) -> int:
    """Row spacing ``S`` of the parallel sweep's conflict-free voxel groups
    for the views ``cos_a`` (the fp32 values the kernels use), or 0 when
    there is none (a view at 90 degrees, or ``S > Nx // 2`` so groups would
    have fewer than two members).

    Voxels of one row ``m * S`` apart project to detector coordinates
    ``m * S * cos`` apart; with ``S * min|cos| >= 2 + eps`` the floors differ
    by at least 2, so their two-pixel footprints are disjoint in every view.
    ``eps`` bounds the fp32 rounding of the two coordinates (three rounded
    operations each on values up to ``umax``): ``2**-20 * umax``, at least
    1e-3. One host synchronisation (the minimum)."""
    if cos_a.numel() == 0 or Nx < 2:
        return 0
    cmin = float(cos_a.detach().abs().min().double())
    if cmin <= 0.0:
        return 0
    umax = 0.5 * (Nx - 1) + 0.5 * (Nz - 1) + 0.5 * (N_det - 1) + 1.0
    eps = max(1e-3, umax * 2.0 ** -20)
    S = math.ceil((2.0 + eps) / cmin)
    return S if S <= Nx // 2 else 0


def _mbir_group_permutation(Ny: int, Nz: int, Nx: int, S: int, key: int) -> np.ndarray:
    """The parallel sweep's visiting order, flattened to one voxel order per
    Y-slice (int32 ``[Ny, Nz*Nx]``), for the serial sweeps: the ``Nz * S``
    groups ``g = iz * S + r`` of a slice in the keyed Feistel order of
    `_mbir_permutation` over ``range(Nz * S)``, each group's members
    ``ix = r, r + S, ...`` of row ``iz`` in increasing order. The members of
    a group are independent (``S`` from `_mbir_group_spacing`), so the
    serial CUDA sweep given this order reproduces the parallel sweep bit for
    bit, and the CPU sweep given it visits the voxels as CUDA does."""
    groups = _mbir_permutation(Ny, Nz * S, key)             # [Ny, Nz*S]
    iz, r = groups // S, groups % S
    m = np.arange(-(-Nx // S), dtype=np.int32)             # members 0 .. ceil(Nx/S)-1
    ix = r[:, :, None] + S * m[None, None, :]              # [Ny, Nz*S, G]
    lin = iz[:, :, None] * Nx + ix
    out = np.empty((Ny, Nz * Nx), dtype=np.int32)
    for iy in range(Ny):
        out[iy] = lin[iy][ix[iy] < Nx]
    return out


def _mbir_resolve_sweep(sweep_kernel: str, device: torch.device, cos_a: torch.Tensor,
                        Nz: int, Nx: int, N_det: int) -> int:
    """The group spacing to sweep with at one level: 0 for the serial sweep,
    ``S > 0`` for the parallel one (CUDA) or its visiting order (CPU).
    ``"auto"`` takes the parallel sweep on CUDA when it applies (a group
    spacing exists and the slice's error sinogram fits in shared memory)
    and the serial one on CPU; ``"parallel"`` raises when it does not
    apply."""
    if sweep_kernel == "serial":
        return 0
    if sweep_kernel == "auto" and device.type != "cuda":
        return 0
    n_angles = cos_a.numel()
    S = _mbir_group_spacing(cos_a, Nz, Nx, N_det)
    fits = (device.type != "cuda"
            or bool(_core.mbir_group_sweep_supported(n_angles=n_angles, N_det=N_det)))
    if S > 0 and fits:
        return S
    if sweep_kernel == "auto":
        return 0
    why = ("no view may be at 90 degrees and the volume must be at least 2 * S voxels "
           f"wide (Nx={Nx})" if S == 0 else
           f"n_angles <= 256 and n_angles * (N_det + 1) * 4 bytes of shared memory "
           f"(n_angles={n_angles}, N_det={N_det})")
    raise ValueError(f"run_mbir: sweep_kernel='parallel' does not apply here: {why}; "
                     "use 'auto' or 'serial'")


def _generate_permutation(Ny: int, Nvox: int, rng: np.random.Generator) -> np.ndarray:
    """Legacy NumPy voxel order (independent ``rng.permutation`` per Y-slice,
    int32 ``[Ny, Nvox]``). ``run_mbir`` no longer uses it (see
    `_mbir_permutation`); kept because ``fasttomo.algorithms`` exposed it."""
    perm = np.empty((Ny, Nvox), dtype=np.int32)
    for iy in range(Ny):
        perm[iy] = rng.permutation(Nvox).astype(np.int32, copy=False)
    return perm


def _mbir_forward_project(
    volume: torch.Tensor,
    N_det: int,
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
    num_threads: int = 1,
) -> torch.Tensor:
    """A_MBIR · f: the voxel-driven scatter FP that matches the column-A
    model in the ICD kernel. Returns ``[n_angles, Ny, N_det]`` fp32."""
    Ny, Nz, Nx = volume.shape
    n_angles = cos_a.shape[0]
    Af = torch.empty(n_angles, Ny, N_det, dtype=torch.float32, device=volume.device)
    vol = volume.detach().float().contiguous()
    _launch("mbir_forward_project", Af,
            vol, Af, cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det,
            shapes=[("volume", (Ny, Nz, Nx)), ("Af", (n_angles, Ny, N_det)),
                    ("cos", (n_angles,)), ("sin", (n_angles,))],
            num_threads=num_threads)
    return Af


def _mbir_initial_residual(
    volume: torch.Tensor,
    sinogram: torch.Tensor,
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
    num_threads: int = 1,
) -> torch.Tensor:
    """e = g - A_MBIR · f at the start of an MBIR level. Falls back to
    `e = g` when `volume` is exactly zero (a common warm-start case) to
    skip a kernel launch."""
    if volume.abs().sum().item() == 0.0:
        return sinogram.clone()
    return sinogram - _mbir_forward_project(volume, sinogram.shape[2], cos_a, sin_a,
                                            num_threads)


def _mbir_icd_sweep_cpu(
    volume: torch.Tensor,             # [Ny, Nz, Nx] fp32, updated in place
    e_sino: torch.Tensor,             # [n_angles, Ny, N_det] fp32, updated in place
    lambdas: torch.Tensor | None,     # [n_angles, Ny, N_det] fp32
    vol_snap: torch.Tensor,           # [Ny, Nz, Nx] fp32
    perm: torch.Tensor,               # [Ny, Nz*Nx] int32
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
    *,
    qggmrf_p: float, qggmrf_q: float, qggmrf_c: float, sigma_f: float,
    huber_T: float, huber_delta: float, sigma: float, num_threads: int,
) -> None:
    """One CPU ICD sweep (voxel updates only; the offset/variance updates
    are `_mbir_update_offset_variance`, shared with CUDA). The C code updates
    volume / e_sino through raw pointers, so every buffer must be a dense
    CPU tensor of the right dtype: a silent copy would drop the in-place
    writes, so strided buffers raise instead (#58)."""
    if volume.ndim != 3 or e_sino.ndim != 3:
        raise ValueError("_mbir_icd_sweep_cpu: volume and e_sino must be 3-D")
    n_angles, Ny, N_det = e_sino.shape
    _, Nz, Nx = volume.shape
    for name, t, dt in (("volume", volume, torch.float32),
                        ("e_sino", e_sino, torch.float32),
                        ("volume_snapshot", vol_snap, torch.float32),
                        ("perm", perm, torch.int32),
                        ("lambdas", lambdas, torch.float32)):
        if t is not None and (t.dtype != dt or t.device.type != "cpu"):
            raise TypeError(f"_mbir_icd_sweep_cpu: {name} must be a CPU {dt} tensor")
    # One place for every size the C sweep trusts (#58 checks included).
    _check_shapes("_mbir_icd_sweep_cpu", [
        ("volume", volume, (Ny, Nz, Nx)),            # e_sino Ny must match volume
        ("e_sino", e_sino, (n_angles, Ny, N_det)),
        ("volume_snapshot", vol_snap, (Ny, Nz, Nx)),
        ("perm", perm, (Ny, Nz * Nx)),
        ("lambdas", lambdas, (n_angles, Ny, N_det)),
        ("cos_a (angles)", cos_a, (n_angles,)),
        ("sin_a (angles)", sin_a, (n_angles,)),
    ])
    _core.mbir_icd_sweep_cpu(
        vol_ptr=_data_ptr(volume, "volume"),
        e_sino_ptr=_data_ptr(e_sino, "e_sino"),
        vol_snapshot_ptr=_data_ptr(vol_snap, "volume_snapshot"),
        lambdas_ptr=_data_ptr(lambdas, "lambdas") if lambdas is not None else 0,
        perm_ptr=_data_ptr(perm, "perm"),
        cos_ptr=_data_ptr(cos_a, "cos_a"), sin_ptr=_data_ptr(sin_a, "sin_a"),
        Ny=Ny, Nz=Nz, Nx=Nx, n_angles=n_angles, N_det=N_det,
        qggmrf_p=qggmrf_p, qggmrf_q=qggmrf_q, qggmrf_c=qggmrf_c, sigma_f=sigma_f,
        huber_T=huber_T, huber_delta=huber_delta, sigma=sigma,
        num_threads=num_threads,
    )


def _check_mbir_perm(op: str, perm: torch.Tensor, Ny: int, Nvox: int) -> None:
    """An explicit voxel order must be ``[Ny, Nz*Nx]`` int32 with every row
    a permutation of ``range(Nz*Nx)``: the sweep indexes the volume and its
    scratch with it unchecked, and a repeated voxel would be updated twice
    (the CUDA prepare kernel reads each voxel's pre-sweep value once)."""
    if perm.dtype != torch.int32:
        raise TypeError(f"{op}: perm must be int32, got {perm.dtype}")
    _check_shapes(op, [("perm", perm, (Ny, Nvox))])
    if perm.numel() == 0:
        return
    lo, hi = int(perm.min()), int(perm.max())
    if lo < 0 or hi >= Nvox:
        raise ValueError(f"{op}: perm values must lie in [0, {Nvox}), got [{lo}, {hi}]")
    ref = torch.arange(Nvox, dtype=torch.int32, device=perm.device)
    if not torch.equal(torch.sort(perm, dim=1).values, ref.expand(Ny, Nvox)):
        raise ValueError(f"{op}: every perm row must be a permutation of range({Nvox}) "
                         "(repeated voxel)")


def _mbir_icd_sweep_cuda(
    volume: torch.Tensor,             # [Ny, Nz, Nx] fp32, updated in place
    e_sino: torch.Tensor,             # [n_angles, Ny, N_det] fp32, updated in place
    lambdas: torch.Tensor | None,     # [n_angles, Ny, N_det] fp32
    vol_snap: torch.Tensor,           # [Ny, Nz, Nx] fp32
    work: torch.Tensor,               # [Ny * Nz * Nx * 2] fp32 scratch
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
    *,
    perm_key: int,
    perm: torch.Tensor | None = None,  # [Ny, Nz*Nx] int32: explicit order (tests)
    qggmrf_p: float, qggmrf_q: float, qggmrf_c: float, sigma_f: float,
    huber_T: float, huber_delta: float, sigma: float,
    sweep_kernel: str = "serial",
    group_spacing: int | None = None,
) -> None:
    """One CUDA ICD sweep (voxel updates only), on ``volume``'s device and
    current stream. The voxel order is built in the kernel from
    ``perm_key`` unless ``perm`` gives it explicitly (validated by
    `_check_mbir_perm`). Every buffer is checked like the CPU sweep's: the
    kernels get raw pointers plus dims.

    ``sweep_kernel`` is resolved by `_mbir_resolve_sweep` (default here:
    the serial sweep) unless ``group_spacing`` gives the resolved value
    (0 = serial, ``S > 0`` = parallel groups of spacing ``S``, which the
    caller must have taken from `_mbir_group_spacing`). The parallel sweep
    builds its own (group) order, so it takes no ``perm``."""
    op = "_mbir_icd_sweep_cuda"
    if volume.ndim != 3 or e_sino.ndim != 3:
        raise ValueError(f"{op}: volume and e_sino must be 3-D")
    n_angles, Ny, N_det = e_sino.shape
    _, Nz, Nx = volume.shape
    dev = volume.device
    for name, t, dt in (("volume", volume, torch.float32),
                        ("e_sino", e_sino, torch.float32),
                        ("volume_snapshot", vol_snap, torch.float32),
                        ("work", work, torch.float32),
                        ("cos_a", cos_a, torch.float32),
                        ("sin_a", sin_a, torch.float32),
                        ("lambdas", lambdas, torch.float32),
                        ("perm", perm, torch.int32)):
        if t is not None and (t.dtype != dt or t.device != dev):
            raise TypeError(f"{op}: {name} must be a {dt} tensor on {dev}")
    _check_shapes(op, [
        ("volume", volume, (Ny, Nz, Nx)),
        ("e_sino", e_sino, (n_angles, Ny, N_det)),
        ("volume_snapshot", vol_snap, (Ny, Nz, Nx)),
        ("work", work, (Ny * Nz * Nx * 2,)),
        ("lambdas", lambdas, (n_angles, Ny, N_det)),
        ("cos_a (angles)", cos_a, (n_angles,)),
        ("sin_a (angles)", sin_a, (n_angles,)),
    ])
    if perm is not None:
        _check_mbir_perm(op, perm, Ny, Nz * Nx)
    if group_spacing is None:
        if sweep_kernel not in _SWEEP_KERNELS:
            raise ValueError(f"{op}: sweep_kernel must be one of {_SWEEP_KERNELS}, "
                             f"got {sweep_kernel!r}")
        group_spacing = 0 if perm is not None and sweep_kernel == "auto" else \
            _mbir_resolve_sweep(sweep_kernel, dev, cos_a, Nz, Nx, N_det)
    if group_spacing and perm is not None:
        raise ValueError(f"{op}: the parallel sweep builds its own order; an explicit perm "
                         "needs sweep_kernel='serial'")
    with torch.cuda.device(dev):   # D7: launch on the data's device
        _core.mbir_icd_sweep_cuda(
            vol_ptr=_data_ptr(volume, "volume"),
            e_sino_ptr=_data_ptr(e_sino, "e_sino"),
            vol_snapshot_ptr=_data_ptr(vol_snap, "volume_snapshot"),
            lambdas_ptr=_data_ptr(lambdas, "lambdas") if lambdas is not None else 0,
            perm_ptr=_data_ptr(perm, "perm") if perm is not None else 0,
            perm_key=perm_key, work_ptr=_data_ptr(work, "work"),
            cos_ptr=_data_ptr(cos_a, "cos_a"), sin_ptr=_data_ptr(sin_a, "sin_a"),
            Ny=Ny, Nz=Nz, Nx=Nx, n_angles=n_angles, N_det=N_det,
            qggmrf_p=qggmrf_p, qggmrf_q=qggmrf_q, qggmrf_c=qggmrf_c, sigma_f=sigma_f,
            huber_T=huber_T, huber_delta=huber_delta, sigma=sigma,
            stream_ptr=torch.cuda.current_stream(dev).cuda_stream,
            group_spacing=int(group_spacing),
        )


def _run_mbir_single_scale(
    sinogram: torch.Tensor,       # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,     # [n_angles]
    *,
    sigma_f: float,
    qggmrf_p: float,
    qggmrf_q: float,
    qggmrf_c: float,
    huber_T: float,
    huber_delta: float,
    sigma: float,
    sigma_min: float,
    estimate_offset: bool,
    estimate_variance: bool,
    outer_iters: int,
    convergence_tol: float,
    seed: int,
    Nz: int,
    Nx: int,
    init_state: _MbirState | None,
    lambdas: torch.Tensor | None,
    return_history: bool,
    rec: _LossRecorder,
    it_offset: int = 0,
    num_threads: int = 1,
    support: torch.Tensor | None = None,
    sweep_kernel: str = "auto",
) -> tuple[_MbirState, list[float]]:
    """Inner loop for one resolution scale. The public `run_mbir` wraps
    this in a coarse-to-fine multiresolution loop.

    Convergence is measured by the relative L2 change in the volume
    between successive outer iterations. The check uses every-N-th voxel
    sampling to keep the comparison cheap relative to the kernel.

    ``rec`` records, after every outer iteration, the data MSE
    ``mean(e²)`` of the residual ``e = g - A·f - d`` at this scale, under
    the global iteration index ``it_offset + it``.

    ``support`` (``[1|Ny, Nz, Nx]`` on this scale's grid) multiplies the
    initial volume and the volume after every ICD sweep. The part it
    removes, ``r = f·(1 - support)``, is forward-projected and added back
    to the error sinogram (``e += A·r``), so the invariant
    ``e = g - A·f - d`` holds for the masked volume before the offset and
    variance updates and the next sweep read it."""
    device = sinogram.device
    n_angles, Ny, N_det = sinogram.shape

    angles_rad = (angles_deg.float().to(device) * (math.pi / 180.0)).contiguous()
    cos_a = torch.cos(angles_rad).contiguous()
    sin_a = torch.sin(angles_rad).contiguous()

    if init_state is None:
        volume = torch.zeros((Ny, Nz, Nx), device=device, dtype=torch.float32)
        d_offsets = torch.zeros(n_angles, device=device, dtype=torch.float32)
    else:
        volume = init_state.volume.clone().contiguous()
        d_offsets = init_state.d_offsets.clone().contiguous()
    if support is not None:
        volume.mul_(support)

    # Residual e = g - A·f - d. d is rolled into A·f via the offset
    # term (we maintain e ≡ g - A·f - d, so initial e adjusts for both).
    e_sino = _mbir_initial_residual(volume, sinogram, cos_a, sin_a, num_threads).contiguous()
    if d_offsets.abs().sum().item() > 0.0:
        e_sino.sub_(d_offsets.view(n_angles, 1, 1))

    Nvox = Nz * Nx
    # 0: serial sweep in the Feistel voxel order; S > 0: parallel sweep over
    # voxel groups of spacing S (CUDA), or its visiting order (CPU).
    group_S = _mbir_resolve_sweep(sweep_kernel, device, cos_a, Nz, Nx, N_det)
    # CUDA sweep scratch (per voxel: its two prior sums; 8 bytes),
    # allocated once per level and reused every iteration.
    work = (torch.empty(Ny * Nvox * 2, dtype=torch.float32, device=device)
            if device.type == "cuda" else None)
    history: list[float] = []
    sigma_cur = float(sigma)
    n_done = 0

    for it in range(outer_iters):
        n_done = it + 1
        vol_snap = volume.clone().contiguous()
        # Random voxel order per Y-slice, new every outer iteration. CUDA
        # builds it inside the sweep kernel from the key (no host array, no
        # copy); the CPU sweep gets the identical order as an array.
        perm_key = _mbir_perm_key(seed, it)

        if device.type == "cuda":
            _mbir_icd_sweep_cuda(
                volume, e_sino, lambdas.contiguous() if lambdas is not None else None,
                vol_snap, work, cos_a, sin_a, perm_key=perm_key,
                qggmrf_p=qggmrf_p, qggmrf_q=qggmrf_q, qggmrf_c=qggmrf_c,
                sigma_f=sigma_f, huber_T=huber_T, huber_delta=huber_delta,
                sigma=sigma_cur, group_spacing=group_S,
            )
        else:
            order = (_mbir_group_permutation(Ny, Nz, Nx, group_S, perm_key) if group_S
                     else _mbir_permutation(Ny, Nvox, perm_key))
            _mbir_icd_sweep_cpu(
                volume, e_sino,
                lambdas.contiguous() if lambdas is not None else None,
                vol_snap, torch.from_numpy(order),
                cos_a, sin_a,
                qggmrf_p=qggmrf_p, qggmrf_q=qggmrf_q, qggmrf_c=qggmrf_c,
                sigma_f=sigma_f, huber_T=huber_T, huber_delta=huber_delta,
                sigma=sigma_cur, num_threads=num_threads,
            )
        if support is not None:
            # Hard support after the sweep; keep e = g - A·f - d consistent.
            removed = volume * (1.0 - support)
            volume.sub_(removed)
            e_sino.add_(_mbir_forward_project(removed, N_det, cos_a, sin_a, num_threads))
        # Closed-form updates for d_k and σ: one implementation, both backends.
        d_offsets, sigma_cur = _mbir_update_offset_variance(
            e_sino, d_offsets, sigma_cur, sigma_min,
            huber_T, huber_delta, lambdas,
            estimate_offset, estimate_variance,
        )
        if rec.wants_loss:
            rec.record(it_offset + it, e_sino.pow(2).mean())
        rec.tick()

        # Convergence check: relative L2 change on a strided sample so it
        # scales as O(Nvox / 8). Cheap vs the per-iter compute.
        with torch.no_grad():
            stride = _MBIR_CONVERGENCE_STRIDE
            f_new = volume.flatten()[::stride]
            f_old = vol_snap.flatten()[::stride]
            denom = f_old.norm().clamp(min=_REL_EPS)
            rel_change = float((f_new - f_old).norm() / denom)
        if return_history:
            history.append(rel_change)
        if rel_change < convergence_tol:
            break

    state = _MbirState(volume=volume, d_offsets=d_offsets, sigma=sigma_cur,
                       e_sino=e_sino, n_done=n_done)
    return state, history


def _check_qggmrf(p: float, q: float, c: float, sigma_f: float,
                  sigma_f_per_scale: list[float] | None) -> None:
    """The qGGMRF potential ``|D/sigma_f|^q / (c + |D/sigma_f|^(q-p))`` is a
    valid (majorizable, edge-preserving) prior only for ``1 <= p <= q <= 2``
    and ``c > 0``; ``sigma_f`` divides. ``p`` and ``q`` used to have swapped
    roles in the kernels (see CHANGELOG), so the old workaround ``qggmrf_p=2, qggmrf_q=1.2``
    now raises instead of silently meaning something else."""
    vals = dict(qggmrf_p=p, qggmrf_q=q, qggmrf_c=c, sigma_f=sigma_f)
    for name, v in vals.items():
        if not math.isfinite(float(v)):
            raise ValueError(f"run_mbir: {name} must be finite, got {v}")
    if not 1.0 <= float(p) <= float(q) <= 2.0:
        raise ValueError(
            "run_mbir: the qGGMRF exponents need 1 <= qggmrf_p <= qggmrf_q <= 2 "
            f"(p: exponent for large differences, q: near zero), got p={p}, q={q}")
    if float(c) <= 0.0:
        raise ValueError(f"run_mbir: qggmrf_c must be > 0, got {c}")
    sfs = [sigma_f] if sigma_f_per_scale is None else [sigma_f, *sigma_f_per_scale]
    for v in sfs:
        if not (math.isfinite(float(v)) and float(v) > 0.0):
            raise ValueError(f"run_mbir: sigma_f must be finite and > 0, got {v}")


def _downsample_sino_det(sino: torch.Tensor, factor: int) -> torch.Tensor:
    """Avg-pool the sinogram in the detector axis only. Y (tilt-axis row)
    is preserved because each Y-slice is an independent MBIR sub-problem
    under single-axis parallel-beam geometry."""
    if factor == 1:
        return sino
    K, Y, M = sino.shape
    s = sino.reshape(K * Y, 1, M)
    s = F.avg_pool1d(s, kernel_size=factor, stride=factor)
    return s.reshape(K, Y, s.shape[-1])


def _resize_volume_zx(vol: torch.Tensor, target_nz: int, target_nx: int) -> torch.Tensor:
    """Trilinear-style resize in (Z, X) only. Y is preserved. Values are not
    rescaled; see `_rescale_between_levels`."""
    Ny, Nz, Nx = vol.shape
    if Nz == target_nz and Nx == target_nx:
        return vol
    v = vol.unsqueeze(0).unsqueeze(0)              # [1,1,Ny,Nz,Nx]
    v = F.interpolate(
        v,
        size=(Ny, target_nz, target_nx),
        mode="trilinear",
        align_corners=False,
    )
    return v.squeeze(0).squeeze(0).contiguous()


def _to_level(vol: torch.Tensor, from_factor: int, to_factor: int,
              target_nz: int, target_nx: int) -> torch.Tensor:
    """Carry a volume from the grid of one multires level to another.

    A level with factor ``f`` works on voxels (and averaged detector pixels)
    ``f`` fine voxels wide, with unit voxel size in its own units: a ray
    crosses ``f`` times fewer voxels, so matching the same (averaged)
    measurements takes ``f`` times the fine-grid values (``rho_f = f *
    rho``; the default ``sigma_f * f`` prior scale per level is in the same
    units). Resizing between levels therefore also multiplies the values by
    ``to_factor / from_factor`` (0.5 from one level to the next finer one,
    ``coarse_factor`` for a fine-grid warm start)."""
    out = _resize_volume_zx(vol, target_nz, target_nx)
    if from_factor != to_factor:
        out = out * (to_factor / from_factor)
    return out


def run_mbir(
    sinogram: torch.Tensor,           # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,         # [n_angles] degrees
    n_iter: int = 100,                # max outer iterations per scale
    *,
    Nz: int | None = None,
    sigma_f: float = 1.0,
    qggmrf_p: float = 1.2,
    qggmrf_q: float = 2.0,
    qggmrf_c: float = 1e-3,
    huber_T: float = float("inf"),
    huber_delta: float = 0.5,
    estimate_offset: bool = True,
    estimate_variance: bool = True,
    sigma_init: float | None = None,
    sigma_min: float | None = None,
    multires_levels: int = 1,
    convergence_tol: float = 1e-4,
    seed: int = 0,
    lambdas: torch.Tensor | None = None,
    sigma_f_per_scale: list[float] | None = None,
    sweep_kernel: str = "auto",
    config: MbirConfig | None = None,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    warm_start: torch.Tensor | None = None,
    num_threads: int | None = None,
    return_loss: bool = False,
    return_state: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
    **deprecated: Any,
) -> torch.Tensor | tuple:
    """Classical MBIR (MBIR-AM-PE, Venkatakrishnan et al. 2015). fp32, CPU and CUDA.

    Joint MAP estimate of the volume ``f``, a per-view offset ``d_k`` and
    the noise level ``sigma`` under a generalised-Huber likelihood and a
    qGGMRF prior, by Gauss-Seidel ICD sweeps (random voxel order per
    Y-slice) with closed-form offset and variance updates between sweeps,
    optionally coarse-to-fine. Recorded loss: the data MSE ``mean(e^2)`` of
    the residual ``e = g - A f - d`` after each outer iteration; indices run
    across multires levels. Stops a level early when the relative volume
    change falls below ``convergence_tol``. On CUDA each level allocates a
    sweep scratch buffer of 8 bytes per voxel of that level (128 MiB at
    256^3, 1 GiB at 512^3), released when the level returns.

    Parameters
    ----------
    sinogram, angles_deg
        ``[n_angles, Ny, N_det]`` fp32 and ``[n_angles]`` degrees. Other dtypes
        raise TypeError (they used to be cast silently).
    n_iter : int
        Maximum outer iterations per level (default 100; formerly
        ``outer_iters``, still accepted with a DeprecationWarning).
    sigma_f, qggmrf_p, qggmrf_q, qggmrf_c : float
        qGGMRF prior of the paper (its eq. after (9)): the potential
        ``rho(D) = |D/sigma_f|^q / (c + |D/sigma_f|^(q-p))`` on the
        differences ``D`` to the 26 neighbours (weights 1/distance, summing
        to 1). ``q=2`` is the exponent near ``D = 0`` (quadratic, smooth),
        ``p=1.2`` the one for large differences (edge preserving), and
        ``c=1e-3`` puts the transition at ``|D| ~ c^(1/(q-p)) sigma_f``
        (1.8e-4 ``sigma_f`` at the defaults); ``1 <= p <= q <= 2`` and
        ``c > 0`` are required (ValueError otherwise). ``sigma_f`` sets the
        strength: smaller is smoother. A good start is about a tenth of the
        expected contrast between neighbouring regions (0.1 for a
        0.5-2.2 phantom with 1 % noise: RMSE 0.046). Near ``D = 0`` the prior
        is ``1/c`` stiffer than ``(D/sigma_f)^2``, so ICD needs more
        iterations than with a quadratic prior, and a small ``sigma_f`` on a
        zero start with ``estimate_variance=True`` can barely move for tens of
        iterations (the prior's curvature dominates the data term until
        ``sigma`` falls). Until this release the kernels had ``p`` and ``q``
        in swapped roles, which made the prior quadratic up to ~1e4
        ``sigma_f``; results and good ``sigma_f`` values differ from earlier
        versions (see CHANGELOG).
    huber_T, huber_delta : float
        Generalised-Huber threshold (``inf``, the default, is conventional
        MBIR; 3 is the paper's choice for BF-TEM with Bragg scatter) and tail
        slope (0.5).
    estimate_offset, estimate_variance : bool
        Closed-form ``d_k`` and ``sigma`` updates (MBIR-AM -> MBIR-AM-PE).
    sigma_init, sigma_min : float, optional
        Initial noise std (default: the sinogram std, 1.0 for a blank
        sinogram) and its floor (default ``sigma_init * 1e-3``).
    multires_levels : int
        Coarse-to-fine levels (1 = single scale). Nz and N_det must be
        divisible by ``2**(levels - 1)``. Offsets carry over between levels
        unchanged (sinogram units; the coarse sinogram is detector-averaged,
        not summed), ``sigma`` restarts at each level's sinogram std, and the
        volume is resized and rescaled to the level's voxel size.
        With ``estimate_variance=False`` more levels reach the single-level
        solution quickly (5e-4 relative at 60 iterations per level). With
        ``estimate_variance=True`` they reach the same fixed point (same
        ``sigma``), but slowly, and so does a single level: on noise-free
        data ``sigma`` falls to the model-error floor (~0.1 against a
        sinogram std of 18.8), which leaves the limited-tilt problem almost
        unregularised, and ICD converges slowly on its missing-wedge
        components. Measured 2-level vs 1-level volume difference (64^2
        cylinder, +-70 degrees, 61 views, offsets off): 22 / 17 / 10 / 3.6 /
        0.3 % at 30 / 150 / 600 / 2400 / 9600 iterations per level, while
        the single-level result itself still moves 13 % from 600 to 2400
        iterations. With ``estimate_offset=True`` too, the two also differ
        by a common offset on every view (1.3-1.6 on this phantom, spread
        0.015), a near-null direction of the joint volume/offset problem
        that has not settled after 9600 iterations in either run. This is a
        property of the joint estimate, not of the level hand-over
        (restarting a level at the coarse ``sigma`` instead of the
        sinogram std measured no better: 18 % instead of 17 % at 150).
    convergence_tol : float
        Per-level stop on the relative L2 volume change (default 1e-4).
    seed : int
        Seed of the voxel order (a keyed Feistel permutation per Y-slice and
        iteration, of the voxels for the serial sweep and of the voxel
        groups for the parallel one). A fixed ``seed`` is bit-reproducible
        for a given backend and ``sweep_kernel``. Different sweeps visit the voxels in different
        orders, so their volumes differ slightly along the way and agree at
        convergence.
    lambdas : Tensor ``[n_angles, Ny, N_det]``, optional
        Per-pixel data weights (default 1), the sinogram's dtype.
    sigma_f_per_scale : list of float, optional
        ``sigma_f`` per level, coarse to fine (default ``sigma_f * 2**level``).
    sweep_kernel : {"auto", "serial", "parallel"}
        The CUDA ICD sweep. ``"serial"``: one warp per Y-slice updates the
        voxels one at a time. ``"parallel"``: the voxels of a row that are
        ``S`` apart, ``S = ceil((2 + eps) / min|cos theta|)`` (5 at +-60
        degrees, 6 at +-70, 12 at +-80), have disjoint detector footprints
        in every view, so each such group is updated at once, with the
        slice's error sinogram in shared memory. Still exact Gauss-Seidel
        ICD: the result equals the serial sweep run on the parallel sweep's
        visiting order, bit for bit; the group order converges a little more
        slowly than the voxel order. Measured (61 views, +-60 degrees, fp32,
        RTX PRO 6000 Blackwell): 9.7 -> 1.25 ms per iteration at 128^3 and
        40.2 -> 8.8 ms at 256^3, 1.2-1.3x more iterations to the same
        ``convergence_tol``, 6.1x / 3.5-3.9x faster end to end. It needs a
        spacing ``S <= Nx // 2`` (no view at or near 90 degrees),
        ``n_angles <= 256``, and ``n_angles * (N_det + 1) * 4`` bytes of
        shared memory per block (99 KB on sm_120: 256^3 at 61 views fits,
        512^3 does not). ``"auto"`` (default) takes ``"parallel"`` on CUDA
        where it applies (per multires level) and ``"serial"`` otherwise;
        ``"parallel"`` raises ValueError where it does not apply. On CPU,
        ``"auto"`` and ``"serial"`` run the C sweep in the voxel order and
        ``"parallel"`` runs it in the parallel sweep's visiting order, so it
        matches a CUDA ``"parallel"`` run up to rounding; with the defaults
        CPU and CUDA visit the voxels in different orders and agree at
        convergence (the same MAP estimate), not iteration by iteration.
    config : MbirConfig, optional
        The tuning keywords above as one object; mixing it with a non-default
        keyword raises TypeError.
    mask, support_mask : optional
        Support (Conventions), applied to each level's initial volume and
        after every outer ICD sweep, not inside the voxel update. The removed
        part is forward-projected into the error sinogram, so
        ``e = g - A f - d`` stays exact. Coarse levels use the support
        max-pooled to their grid.
    warm_start : Tensor, optional
        Initial volume on the full grid (resized to the coarsest level and
        scaled to its voxel size, see `_to_level`); offsets start at 0.
    num_threads : int, optional
        CPU threads (it was hardcoded to 4); see Conventions.
    return_loss, loss_callback, progress
        See Conventions of :mod:`fasttomo.algorithms`.
    return_state : bool
        Also return ``(d_offsets, sigma)``: the ``[n_angles]`` offsets and the
        final noise std.

    Returns
    -------
    ``volume``, plus ``loss_history`` with ``return_loss`` and
    ``d_offsets, sigma`` with ``return_state``, in that order. The
    deprecated ``return_history=True`` still returns the legacy
    ``(volume, d_offsets, sigma, history)`` with per-level lists of relative
    volume changes.
    """
    n_iter = _deprecated_kwarg(deprecated, "outer_iters", "n_iter", n_iter, "run_mbir",
                               default=100)
    return_history = deprecated.pop("return_history", False)
    _reject_unknown_kwargs(deprecated, "run_mbir")
    if return_history:
        if return_loss or return_state:
            raise TypeError(
                "run_mbir: return_history (deprecated) cannot be combined with "
                "return_loss or return_state."
            )
        warnings.warn(
            "run_mbir: return_history is deprecated; use return_loss=True for the "
            "loss history and return_state=True for (d_offsets, sigma). It will "
            "be removed in the next release.",
            DeprecationWarning, stacklevel=2,
        )
    if config is not None:
        if not isinstance(config, MbirConfig):
            raise TypeError(f"run_mbir: config must be an MbirConfig, "
                            f"got {type(config).__name__}")
        tuning = dict(
            sigma_f=sigma_f, qggmrf_p=qggmrf_p, qggmrf_q=qggmrf_q, qggmrf_c=qggmrf_c,
            huber_T=huber_T, huber_delta=huber_delta, estimate_offset=estimate_offset,
            estimate_variance=estimate_variance, sigma_init=sigma_init,
            sigma_min=sigma_min, multires_levels=multires_levels,
            convergence_tol=convergence_tol, seed=seed,
            sigma_f_per_scale=sigma_f_per_scale, sweep_kernel=sweep_kernel,
        )
        defaults = MbirConfig()
        clash = [k for k, v in tuning.items() if not _is_default(v, getattr(defaults, k))]
        if clash:
            raise TypeError(
                f"run_mbir: pass the tuning options either through config or as "
                f"keywords, not both (got config and {', '.join(clash)})."
            )
        (sigma_f, qggmrf_p, qggmrf_q, qggmrf_c, huber_T, huber_delta,
         estimate_offset, estimate_variance, sigma_init, sigma_min,
         multires_levels, convergence_tol, seed, sigma_f_per_scale, sweep_kernel) = (
            getattr(config, k) for k in tuning)
    if multires_levels < 1:
        raise ValueError("multires_levels must be >= 1")
    _check_qggmrf(qggmrf_p, qggmrf_q, qggmrf_c, sigma_f, sigma_f_per_scale)
    if sweep_kernel not in _SWEEP_KERNELS:
        raise ValueError(f"run_mbir: sweep_kernel must be one of {_SWEEP_KERNELS}, "
                         f"got {sweep_kernel!r}")
    # fp32 only, like every other fp32 algorithm: other dtypes raise instead
    # of being cast silently. The ICD kernels are fp32-only.
    setup = _prepare_recon(
        "run_mbir", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        init_volume=False, warm_start=warm_start, mask=mask, support_mask=support_mask,
    )
    support = setup.constraints.support_mask     # [1|Ny, Nz, Nx] fp32 or None
    n_angles, Ny, N_det, Nz, Nx = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx)
    n_iter = setup.n_iter
    nt = _resolve_num_threads(num_threads)
    multires_levels = _as_int(multires_levels, "multires_levels", "run_mbir", minimum=1)
    angles_deg = angles_deg.float()
    # A NaN/inf pixel would otherwise surface as "sigma_init must be finite"
    # (sinogram.std() is NaN) even though the caller never passed sigma_init,
    # or poison every voxel if sigma_init is given.
    if not torch.isfinite(sinogram).all():
        raise ValueError("run_mbir: sinogram contains NaN or inf values")

    if lambdas is not None:
        if lambdas.dim() != 3 or tuple(lambdas.shape) != (n_angles, Ny, N_det):
            raise ValueError(
                "lambdas must match sinogram shape "
                f"[n_angles, Ny, N_det]={sinogram.shape}, got {tuple(lambdas.shape)}"
            )
        _assert_dtype_match(sinogram, lambdas, "sinogram", "lambdas", "run_mbir")
        if lambdas.device != sinogram.device:
            lambdas = lambdas.to(sinogram.device)
        # The MBIR sweeps read lambdas through a raw linear pointer (CUDA) or
        # a numpy buffer (CPU), so a strided view such as one produced by
        # ``expand`` would be indexed as if it were dense. Materialise it.
        lambdas = lambdas.contiguous()
        # Negative weights make lambdas.sqrt() NaN inside the variance update;
        # non-finite weights poison every voxel they touch.
        if not torch.isfinite(lambdas).all():
            raise ValueError("run_mbir: lambdas must contain only finite values")
        if (lambdas < 0).any():
            raise ValueError("run_mbir: lambdas must be non-negative")

    if sigma_init is None:
        sigma_init = float(sinogram.std().item())
        if sigma_init < _MBIR_BLANK_STD:
            sigma_init = 1.0
    # sigma divides the data term, so zero or negative values blow up.
    if not math.isfinite(float(sigma_init)) or float(sigma_init) <= 0.0:
        raise ValueError(f"run_mbir: sigma_init must be finite and > 0, got {sigma_init}")
    if sigma_min is None:
        sigma_min = sigma_init * _MBIR_SIGMA_MIN_RATIO
    if not math.isfinite(float(sigma_min)) or float(sigma_min) <= 0.0:
        raise ValueError(f"run_mbir: sigma_min must be finite and > 0, got {sigma_min}")

    # Multires schedule: factors = [2**(L-1), ..., 2, 1] for L levels.
    factors = [2 ** (multires_levels - 1 - lvl) for lvl in range(multires_levels)]
    coarse_factor = factors[0]
    if coarse_factor > 1 and (
        N_det % coarse_factor != 0
        or Nx % coarse_factor != 0
        or Nz % coarse_factor != 0
    ):
        raise ValueError(
            "run_mbir multires_levels requires detector and reconstruction "
            f"dims divisible by {coarse_factor}; got N_det={N_det}, "
            f"Nx={Nx}, Nz={Nz}"
        )
    if sigma_f_per_scale is None:
        sigma_f_per_scale = [sigma_f * float(f) for f in factors]
    elif len(sigma_f_per_scale) != multires_levels:
        raise ValueError(
            "sigma_f_per_scale must have length multires_levels"
        )

    history_all: list[list[float]] = []
    state: _MbirState | None = None
    if setup.volume0 is not None:
        # warm start: resized to the coarsest grid; offsets start at zero
        state = _MbirState(
            volume=_to_level(setup.volume0, 1, coarse_factor,
                             max(Nz // coarse_factor, 1), max(Nx // coarse_factor, 1)),
            d_offsets=torch.zeros(n_angles, device=sinogram.device, dtype=torch.float32),
            sigma=float(sigma_init), e_sino=None,
        )
    rec = _LossRecorder(return_loss, loss_callback, progress,
                        n_iter * multires_levels, desc="run_mbir")
    it_offset = 0

    for lvl, f in enumerate(factors):
        sino_lvl = _downsample_sino_det(sinogram, f)
        # Scale lambdas the same way if provided.
        if lambdas is not None and f > 1:
            lam_lvl = _downsample_sino_det(lambdas, f)
        else:
            lam_lvl = lambdas

        Nz_lvl = max(Nz // f, 1)
        Nx_lvl = max(Nx // f, 1)
        # The support on this level's grid: a coarse voxel keeps the largest
        # weight of the f x f fine voxels it covers (max-pool; f divides Nz
        # and Nx, checked above), so a coarse level never cuts into the
        # support. Exact at f = 1.
        support_lvl = (support if support is None or f == 1
                       else F.max_pool2d(support, kernel_size=f, stride=f))

        # First iteration at the coarsest scale: force T=inf so the
        # anomaly modelling is disabled until the next finer scale. This
        # matches the paper's multires + AM-PE bootstrap.
        if multires_levels > 1 and lvl == 0:
            level_T = float("inf")
        else:
            level_T = huber_T

        # Carry d_offsets across scales unchanged; re-initialize σ from
        # the current sinogram's std (intensity statistics shift with
        # the detector binning).
        if lvl == 0:
            level_sigma = sigma_init
        else:
            level_sigma = max(sigma_min, float(sino_lvl.std().item()))
            state.sigma = level_sigma

        state, hist = _run_mbir_single_scale(
            sino_lvl, angles_deg,
            sigma_f=sigma_f_per_scale[lvl],
            qggmrf_p=qggmrf_p, qggmrf_q=qggmrf_q, qggmrf_c=qggmrf_c,
            huber_T=level_T, huber_delta=huber_delta,
            sigma=level_sigma, sigma_min=sigma_min,
            estimate_offset=estimate_offset,
            estimate_variance=estimate_variance,
            outer_iters=n_iter,
            convergence_tol=convergence_tol,
            seed=seed + lvl,
            Nz=Nz_lvl, Nx=Nx_lvl,
            init_state=state,
            lambdas=lam_lvl,
            return_history=return_history,
            rec=rec, it_offset=it_offset, num_threads=nt, support=support_lvl,
            sweep_kernel=sweep_kernel,
        )
        history_all.append(hist)
        it_offset += state.n_done

        # Carry the current solution to the next scale with an exact resize
        # and the voxel-size rescale (`_to_level`: values halve from one
        # level to the next finer one). Using a fixed ×2 upsample breaks odd
        # detector sizes (e.g. 65 → 32 → 64), causing shape mismatches at the
        # next level.
        if lvl + 1 < len(factors):
            next_f = factors[lvl + 1]
            next_nz = max(Nz // next_f, 1)
            next_nx = max(Nx // next_f, 1)
            state.volume = _to_level(state.volume, f, next_f, next_nz, next_nx)

    rec.close()
    if return_history:
        return state.volume, state.d_offsets, state.sigma, history_all
    result: list = [state.volume]
    if return_loss:
        result.append(rec.history())
    if return_state:
        result += [state.d_offsets, state.sigma]
    return result[0] if len(result) == 1 else tuple(result)
