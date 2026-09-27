"""
Utility functions for FastTomo: the ramp filter and shift estimation.

Re-exported here, their historical home: ``cylindrical_mask`` (lives in
:mod:`fasttomo.constraints`), ``tv_loss_3d`` (:mod:`fasttomo.regularizers`)
and ``apply_shifts`` / ``apply_shifts_2d`` (:mod:`fasttomo.projection`,
which applies them inside ``forward_project_3d``). Keeping the shift
operators next to the projectors is what lets this module import
:mod:`fasttomo.projection` at the top without an import cycle.
"""

from __future__ import annotations

import torch

# Moved to fasttomo.constraints (ARCH P10); re-exported here, its historical home.
from fasttomo.constraints import cylindrical_mask
# Moved to fasttomo.projection / fasttomo.regularizers (ARCH P8); re-exported here.
from fasttomo.projection import (
    apply_shifts,
    apply_shifts_2d,
    forward_project,
    forward_project_3d,
)
from fasttomo.regularizers import tv_loss_3d

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
