"""
Utility functions for FastTomo: masks, filters, and shift helpers.

All public functions accept and return torch.Tensor (float32).
"""

from __future__ import annotations

import math

import torch

# Moved to fasttomo.constraints (ARCH P10); re-exported here, its historical home.
from fasttomo.constraints import cylindrical_mask

__all__ = [
    "cylindrical_mask",
    "ramp_filter",
    "tv_loss_3d",
    "apply_shifts",
    "apply_shifts_2d",
    "estimate_shifts_xcorr",
]


# ── ramp filter (PyTorch, runs on same device as input) ───────────────────

# Ramp-filter |ξ| is shape-only; cache per (N_det, device, dtype) so tight
# loops (FBP per slice, SIRT-style residual filtering) avoid the rfftfreq +
# abs allocation each call.
_RAMP_FILTER_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _ramp_filter_coeffs(N_det: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (int(N_det), device, dtype)
    coeffs = _RAMP_FILTER_CACHE.get(key)
    if coeffs is None:
        coeffs = torch.fft.rfftfreq(N_det, device=device, dtype=dtype).abs().contiguous()
        _RAMP_FILTER_CACHE[key] = coeffs
    return coeffs


def ramp_filter(sinogram: torch.Tensor) -> torch.Tensor:
    """
    Apply ramp filter |ξ| to each detector row of sinogram.

    sinogram: [n_angles, Ny, N_det]
    Returns same shape, same device, same dtype. fp32 / fp16 supported; bf16
    raises.

    Implementation note on fp16: cuFFT only supports power-of-two signal
    lengths in half precision. Rather than restrict the user to power-of-two
    detectors, we always run the FFT in fp32 and cast the result back to the
    input dtype. Mirrors TorchRadon's `padded_sinogram.float() ... .to(dtype)`
    pattern — minor cost on the FFT itself, no cost on the BP that dominates.
    """
    if sinogram.dtype == torch.bfloat16:
        raise TypeError(
            "ramp_filter does not support torch.bfloat16; cast to "
            "torch.float16 or torch.float32 first."
        )
    N_det = sinogram.shape[-1]
    sino_f = sinogram.float() if sinogram.dtype != torch.float32 else sinogram
    f      = torch.fft.rfft(sino_f, n=N_det, dim=-1)
    f      = f * _ramp_filter_coeffs(N_det, sino_f.device, torch.float32)
    out    = torch.fft.irfft(f, n=N_det, dim=-1)
    return out.to(sinogram.dtype) if sinogram.dtype != torch.float32 else out


# ── 3-D isotropic Total-Variation loss ────────────────────────────────────

def tv_loss_3d(volume: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Isotropic 3-D total-variation loss.

        loss = mean( sqrt( dy² + dz² + dx² + eps ) )

    where the differences are the standard half-pixel-shifted forward
    differences anchored at each (i, j, k) in the inner-corner grid of
    shape (Ny-1) × (Nz-1) × (Nx-1).

    Differentiable w.r.t. ``volume`` — drop into any autograd loop in
    place of the pure-PyTorch expression. On CUDA, dispatches to a single
    fused kernel (one read-pass over the volume, one atomic-reduction to
    a scalar) and a voxel-driven gather kernel for the backward. On CPU,
    falls back to the equivalent pure-PyTorch expression so the same
    public function works on either device.

    Args:
        volume: [Ny, Nz, Nx] float32 tensor.
        eps:    smoothing constant inside the sqrt (default 1e-8). Keeps
                the gradient well-defined near flat regions.

    Returns:
        Scalar tensor (shape ``[]``) on the same device as ``volume``.
    """
    if volume.dtype != torch.float32:
        raise TypeError(f"tv_loss_3d is fp32-only (got {volume.dtype}). "
                        "Cast your tensor to float32 first.")
    if volume.ndim != 3:
        raise ValueError(f"tv_loss_3d expects a 3-D [Ny, Nz, Nx] tensor, "
                         f"got shape {tuple(volume.shape)}")

    if volume.is_cuda:
        from fasttomo.autograd import _TVLossIso3D
        return _TVLossIso3D.apply(volume, eps)

    # CPU fallback — pure PyTorch, autograd works automatically.
    dy = volume[1:, :, :] - volume[:-1, :, :]
    dz = volume[:, 1:, :] - volume[:, :-1, :]
    dx = volume[:, :, 1:] - volume[:, :, :-1]
    return (dx[:-1, :-1, :] ** 2 + dy[:, :-1, :-1] ** 2
            + dz[:-1, :, :-1] ** 2 + eps).sqrt().mean()


def apply_shifts(
    sinogram: torch.Tensor,   # [n_angles, Ny, N_det]
    shifts: torch.Tensor,     # [n_angles] detector-axis shifts in pixels
) -> torch.Tensor:            # [n_angles, Ny, N_det], same device as sinogram
    """
    Sub-pixel shift each projection row along the detector axis via Fourier
    phase multiplication.  Positive shift moves the signal to the right.

    shifts may be on CPU even when sinogram is on CUDA.
    """
    N_det = sinogram.shape[-1]
    s = shifts.to(device=sinogram.device, dtype=torch.float32)
    freqs = torch.fft.rfftfreq(N_det, device=sinogram.device)          # [N_freq]
    phase_arg = -2.0 * math.pi * freqs.view(1, 1, -1) * s.view(-1, 1, 1)
    phase_mul = torch.polar(torch.ones_like(phase_arg), phase_arg)
    F = torch.fft.rfft(sinogram.float(), n=N_det, dim=-1)
    return torch.fft.irfft(F * phase_mul, n=N_det, dim=-1)


def apply_shifts_2d(
    sinogram: torch.Tensor,   # [n_angles, Ny, N_det]
    shifts: torch.Tensor,     # [n_angles, 2] — (det-axis, tilt-axis) pixel shifts
) -> torch.Tensor:            # [n_angles, Ny, N_det], same device as sinogram
    """
    Per-tilt sub-pixel translation of the sinogram in the detector plane via
    bilinear ``grid_sample``.  Differentiable w.r.t. both ``sinogram`` and
    ``shifts`` — autograd flows through ``grid_sample`` natively.

    ``shifts[:, 0]`` shifts along the detector axis (last dim, ``N_det``);
    ``shifts[:, 1]`` shifts along the tilt axis (``Ny``).  Sign convention
    matches ``apply_shifts``: positive shift moves the signal toward larger
    indices (right / down).

    Intended for joint optimization with the volume and pose: drop in after
    ``forward_project_3d`` (or pass ``shifts=`` to it directly) so per-tilt
    translational alignment becomes a learnable parameter.
    """
    if shifts.ndim != 2 or shifts.shape[-1] != 2:
        raise ValueError(f"shifts must have shape [n_angles, 2], got {tuple(shifts.shape)}")
    n_a, Ny, Nd = sinogram.shape
    if shifts.shape[0] != n_a:
        raise ValueError(
            f"shifts has {shifts.shape[0]} entries but sinogram has {n_a} angles"
        )

    s = shifts.to(device=sinogram.device, dtype=torch.float32)

    ys = torch.linspace(-1.0, 1.0, Ny, device=sinogram.device, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, Nd, device=sinogram.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")                  # [Ny, Nd]
    base = torch.stack((gx, gy), dim=-1)                            # [Ny, Nd, 2]
    base = base.unsqueeze(0).expand(n_a, -1, -1, -1)                # [n_a, Ny, Nd, 2]

    # Sample from (x - sx, y - sy) so positive shift moves the signal +x / +y.
    # Normalized coords with align_corners=True: 1 pixel = 2/(N-1).
    norm_x = 2.0 / max(Nd - 1, 1)
    norm_y = 2.0 / max(Ny - 1, 1)
    off = torch.stack((s[:, 0] * norm_x, s[:, 1] * norm_y), dim=-1) # [n_a, 2]
    grid = base - off.view(n_a, 1, 1, 2)                            # [n_a, Ny, Nd, 2]

    inp = sinogram.unsqueeze(1).float()                             # [n_a, 1, Ny, Nd]
    out = torch.nn.functional.grid_sample(
        inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return out.squeeze(1)


def estimate_shifts_xcorr(
    volume: torch.Tensor,                       # [Ny, Nz, Nx]
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    *,
    max_shift: float | None = None,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — CUDA only
    num_threads: int | None = None,             # CPU paths
    **removed: object,
) -> torch.Tensor:                              # [n_angles] float32, CPU
    """
    Estimate the translational shift of each projection (along the detector
    axis) by 1-D phase cross-correlation between the forward projection of
    *volume* and the corresponding row of *sinogram*.

    Returns the correction shift to apply so that::

        aligned_sino = ft.apply_shifts(sinogram, shifts)

    brings sinogram into alignment with the volume (no negation needed at the
    call site).

    Sub-pixel precision is achieved via 3-point parabolic interpolation around
    the cross-correlation peak. (The former ``upsample_factor`` argument never
    had an effect and was removed; passing it raises TypeError.)

    Typical outer-loop usage::

        for _ in range(n_outer):
            vol    = ft.run_resire(aligned_sino, angles, n_iter=100)
            shifts = ft.estimate_shifts_xcorr(vol, aligned_sino, angles)
            aligned_sino = ft.apply_shifts(aligned_sino, shifts)
    """
    # Import here to avoid circular imports: projection imports utilities,
    # so utilities must not import from projection at module level.
    from fasttomo.projection import forward_project, forward_project_3d

    if "upsample_factor" in removed:
        raise TypeError(
            "estimate_shifts_xcorr: upsample_factor was removed. It never had an "
            "effect: the sub-pixel shift comes from a 3-point parabolic fit of "
            "the correlation peak, and no upsampled-DFT refinement exists. "
            "Drop the argument; the result is unchanged."
        )
    if removed:
        raise TypeError(
            "estimate_shifts_xcorr() got an unexpected keyword argument "
            f"'{next(iter(removed))}'"
        )
    N_det = sinogram.shape[-1]

    if euler_zyz_deg is not None:
        pred = forward_project_3d(volume, euler_zyz_deg, num_threads=num_threads)
    else:
        pred = forward_project(volume, angles_deg, num_threads=num_threads)

    F_pred = torch.fft.rfft(pred.float(),      n=N_det, dim=-1)  # [n_a, Ny, N_freq]
    F_meas = torch.fft.rfft(sinogram.float(),  n=N_det, dim=-1)

    # Phase cross-power spectrum: peaks at +s when sinogram is shifted +s vs pred
    cross   = F_pred * F_meas.conj()
    cross_n = cross / cross.abs().clamp(min=1e-12)

    # Coherent sum over Ny before IRFFT (SNR ∝ sqrt(Ny))
    cross_sum = cross_n.sum(dim=1)                                  # [n_a, N_freq]
    ccf       = torch.fft.irfft(cross_sum, n=N_det, dim=-1)        # [n_a, N_det]

    # Integer peak → signed shift
    peak_idx  = ccf.argmax(dim=-1)                                  # [n_a]
    shifts_int = torch.where(
        peak_idx <= N_det // 2,
        peak_idx.float(),
        peak_idx.float() - N_det,
    )

    # Parabolic sub-pixel refinement with circular neighbour wrap
    idx_m = (peak_idx - 1) % N_det
    idx_p = (peak_idx + 1) % N_det
    y_m   = ccf.gather(1, idx_m.unsqueeze(1)).squeeze(1)
    y_0   = ccf.gather(1, peak_idx.unsqueeze(1)).squeeze(1)
    y_p   = ccf.gather(1, idx_p.unsqueeze(1)).squeeze(1)
    denom = y_m - 2.0 * y_0 + y_p
    delta = torch.where(
        denom.abs() > 1e-9,
        0.5 * (y_m - y_p) / denom,
        torch.zeros_like(denom),
    )

    # Negate: F_pred * conj(F_meas) peaks at +s when sino is +s vs pred;
    # returning -s so that apply_shifts(sino, shifts) corrects the alignment.
    shifts_out = -(shifts_int + delta)

    if max_shift is not None:
        shifts_out = shifts_out.clamp(-max_shift, max_shift)

    return shifts_out.cpu()
