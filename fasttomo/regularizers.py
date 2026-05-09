"""
Differentiable regularizers for the AD reconstruction path.

Each function returns a scalar tensor; add ``weight * regularizer(vol, ...)``
to the data-fidelity loss and call ``.backward()`` as usual. Inputs are
``[Ny, Nz, Nx]`` float32 volumes on CPU or CUDA.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


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


# ── Log-Sum L0-proxy sparsity ─────────────────────────────────────────────

def logsum_sparsity(volume: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    Log-Sum L0-proxy sparsity penalty (Candès–Wakin–Boyd, 2008).

        loss = mean( log(1 + |volume| / eps) )
        ∂loss/∂ρ = sign(ρ) / (eps + |ρ|) / N

    Smoother than L1; the gradient *shrinks* for large |ρ|, so true atom
    peaks are barely shrunk while background voxels are driven aggressively
    toward zero. Recommended over L1 soft-thresholding for atomic electron
    tomography (see ``wiki/analyses/atomic-et-regularization-plan.md``,
    Priority 1).

    Args:
        volume: float tensor (any shape, CPU or CUDA). 3-D ``[Ny, Nz, Nx]``
                is the typical AD reconstruction case; the function does not
                impose a shape constraint.
        eps:    smoothing scale. Set near 0.1 × the expected atom-peak
                amplitude (~1e-2 for normalized HEA reconstructions).
                Smaller eps = more aggressive zeroing; eps→0 reduces toward
                ``log|ρ|`` (singular at zero).

    Returns:
        Scalar tensor on the same device as ``volume``.
    """
    return torch.log1p(volume.abs() / eps).mean()


# ── Per-tilt SSIM loss on predicted projections ───────────────────────────

def _gaussian_window(win_size: int, sigma: float,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Separable 2-D Gaussian kernel of shape [1, 1, win, win], summing to 1."""
    coords = torch.arange(win_size, device=device, dtype=dtype) - (win_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w2 = g[:, None] * g[None, :]
    return w2[None, None, :, :]


def tilt_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    win_size: int = 11,
    sigma: float = 1.5,
    data_range: float | None = None,
) -> torch.Tensor:
    """
    Per-tilt structural-similarity loss on a stack of 2-D projection images.

        loss = 1 - mean( SSIM_map(pred[b], target[b]) ) over batch b

    Wang-2004 single-scale SSIM with a Gaussian window (σ=1.5, 11×11 by
    default). Each tilt is treated as an independent single-channel image;
    SSIM maps are averaged within a tilt and across tilts. Reflect padding
    keeps the SSIM map covering the full image.

    Args:
        pred, target: ``[B, H, W]`` float32 tensors on the same device.
        win_size, sigma: Gaussian window parameters.
        data_range: dynamic range used for the SSIM constants. If ``None``,
            defaults to ``target.max() - target.min()`` (skimage convention).

    Returns:
        Scalar tensor on the same device as ``pred``. ``0`` for identical
        inputs, up to ``2`` in the worst case (since SSIM ∈ [-1, 1]).
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs "
                         f"target {tuple(target.shape)}")
    if pred.ndim != 3:
        raise ValueError(f"tilt_ssim_loss expects [B, H, W], got "
                         f"{tuple(pred.shape)}")
    if pred.dtype != torch.float32 or target.dtype != torch.float32:
        raise TypeError("tilt_ssim_loss is fp32-only "
                        f"(got pred={pred.dtype}, target={target.dtype}).")

    if data_range is None:
        data_range = float(target.max() - target.min())
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    pad = win_size // 2
    window = _gaussian_window(win_size, sigma, pred.device, pred.dtype)

    x = pred.unsqueeze(1)        # [B, 1, H, W]
    y = target.unsqueeze(1)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.pad(y, (pad, pad, pad, pad), mode="reflect")

    mu_x  = F.conv2d(x,     window)
    mu_y  = F.conv2d(y,     window)
    mu_xx = F.conv2d(x * x, window)
    mu_yy = F.conv2d(y * y, window)
    mu_xy = F.conv2d(x * y, window)

    sigma_xx = mu_xx - mu_x * mu_x
    sigma_yy = mu_yy - mu_y * mu_y
    sigma_xy = mu_xy - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_xx + sigma_yy + C2)
    ssim_map = num / den

    return 1.0 - ssim_map.mean()


# ── Atom-prototype helpers (private, torch-only) ──────────────────────────

def _gaussian_kernel_3d(sigma: float,
                        radius: int | None = None,
                        device: torch.device | None = None,
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Isotropic 3-D Gaussian kernel of shape ``[1, 1, K, K, K]``.

    Centred so the centre voxel has the kernel's maximum (``= 1`` before
    any normalisation). The kernel is **not** normalised to sum to 1 —
    we want a peak-amplitude render: ``conv3d(spike_at_p_with_value_A, K)``
    deposits ``A`` at ``p`` and Gaussian-decaying values elsewhere.

    Args:
        sigma:  Gaussian σ in voxels.
        radius: half-width of the kernel (kernel size = 2*radius+1).
                Defaults to ``ceil(3*sigma)`` (covers ~99.7% of the bell).
        device, dtype: Tensor placement.

    Returns:
        ``[1, 1, K, K, K]`` tensor on the requested device.
    """
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    # Outer-product additions instead of meshgrid allocation.
    g1 = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    k3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    return k3[None, None, :, :, :]


def _gaussian_blur_3d(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    """Sum-to-1 Gaussian blur of a single 3-D volume via one ``F.conv3d``.

    Used by the atom detector to suppress per-voxel noise before peak NMS.
    Returns the same ``[Ny, Nz, Nx]`` shape as the input; reflect padding
    keeps the boundary well-defined.
    """
    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel = _gaussian_kernel_3d(sigma, radius=radius,
                                 device=volume.device, dtype=volume.dtype)
    kernel = kernel / kernel.sum()              # sum-to-1 for blurring
    x = volume[None, None, :, :, :]
    pad = (radius,) * 6                         # (Wl, Wr, Hl, Hr, Dl, Dr)
    x = F.pad(x, pad, mode="reflect")
    y = F.conv3d(x, kernel)
    return y[0, 0]


def _detect_atom_peaks_3d(volume: torch.Tensor,
                          min_distance: int,
                          threshold: float) -> torch.Tensor:
    """Bool mask of strict local maxima in ``volume``.

    A voxel is a peak iff (a) it equals the max-pool over a
    ``(2·min_distance + 1)`` cube around it and (b) its value exceeds
    ``threshold``. This is local-max NMS — no lattice / periodicity
    assumption.

    Args:
        volume:       ``[Ny, Nz, Nx]`` float tensor (typically the
                      Gaussian-blurred volume).
        min_distance: NMS half-width in voxels (≥ 1).
        threshold:   Absolute lower bound on peak value.

    Returns:
        Bool tensor of the same shape; ``True`` at peak voxels.
    """
    if min_distance < 1:
        raise ValueError(f"min_distance must be ≥ 1, got {min_distance}")
    k = 2 * min_distance + 1
    x = volume[None, None, :, :, :]
    pooled = F.max_pool3d(x, kernel_size=k, stride=1, padding=min_distance)
    return (volume == pooled[0, 0]) & (volume > threshold)


def _dilate_3d(mask_bool: torch.Tensor, radius: int) -> torch.Tensor:
    """Spherical dilation of a bool mask via a single ``F.max_pool3d``.

    Returns a bool tensor of the same shape with ``True`` everywhere
    within ``radius`` voxels (Chebyshev metric — cube neighbourhood) of
    any ``True`` input voxel.
    """
    if radius < 0:
        raise ValueError(f"radius must be ≥ 0, got {radius}")
    if radius == 0:
        return mask_bool
    k = 2 * radius + 1
    x = mask_bool.to(torch.float32)[None, None, :, :, :]
    y = F.max_pool3d(x, kernel_size=k, stride=1, padding=radius)
    return y[0, 0] > 0.5


def _peak_nn_distances(peaks_bool: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (peak_coords [N,3] int64, nn_distances [N] float32)."""
    coords = torch.nonzero(peaks_bool)
    if coords.shape[0] <= 1:
        return coords, torch.empty(0, device=peaks_bool.device, dtype=torch.float32)
    pts = coords.to(torch.float32)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = (diff * diff).sum(dim=-1)
    d2.fill_diagonal_(float("inf"))
    nn_d = d2.min(dim=1).values.sqrt()
    return coords, nn_d


def _cull_isolated_peaks(peaks_bool: torch.Tensor,
                         max_neighbor_dist: float) -> torch.Tensor:
    """Drop peaks whose nearest neighbour is farther than a fixed
    ``max_neighbor_dist`` voxels (Euclidean).

    Use ``_cull_isolated_peaks_zscore`` instead when the absolute lattice
    spacing isn't known a priori.
    """
    coords, nn_d = _peak_nn_distances(peaks_bool)
    if nn_d.numel() == 0:
        return peaks_bool
    keep = nn_d <= max_neighbor_dist
    if bool(keep.all()):
        return peaks_bool
    out = torch.zeros_like(peaks_bool)
    kept = coords[keep]
    if kept.numel() > 0:
        out[kept[:, 0], kept[:, 1], kept[:, 2]] = True
    return out


def _cull_isolated_peaks_zscore(peaks_bool: torch.Tensor,
                                k_sigma: float) -> torch.Tensor:
    """Drop peaks whose nearest-neighbour distance exceeds the dataset's
    *own* mean NN distance + ``k_sigma`` standard deviations.

    Adaptive to whatever the local lattice spacing is — useful when the
    voxel pitch / bond length isn't known a priori but the cluster of
    true atoms forms a coherent NN-distance distribution. Typical
    ``k_sigma`` values are 2.0 (aggressive) to 3.0 (loose).
    """
    coords, nn_d = _peak_nn_distances(peaks_bool)
    if nn_d.numel() <= 2:
        return peaks_bool
    mu = nn_d.mean()
    sigma = nn_d.std()
    threshold = mu + k_sigma * sigma
    keep = nn_d <= threshold
    if bool(keep.all()):
        return peaks_bool
    out = torch.zeros_like(peaks_bool)
    kept = coords[keep]
    if kept.numel() > 0:
        out[kept[:, 0], kept[:, 1], kept[:, 2]] = True
    return out


# ── Atomic-Prototype RED loss ─────────────────────────────────────────────

def atom_red_loss(
    volume: torch.Tensor,
    *,
    sigma_atom: float = 1.0,
    min_distance: int = 2,
    dilate_radius: int = 3,
    threshold_rel: float = 0.05,
    cull_max_neighbor_dist: float | None = None,
    cull_nn_zscore: float | None = None,
) -> torch.Tensor:
    """RED-style atom-prior regulariser for atomic electron tomography.

    Implements the proximal-form RED penalty

        loss = 0.5 * mean( (volume - f_atom(volume).detach())² )

    where ``f_atom`` is the **atom-prototype denoiser**: it (1) blurs the
    volume with σ=``sigma_atom``, (2) detects local-maxima peaks via NMS
    (peaks must exceed ``threshold_rel * volume.max()``), (3) dilates the
    peak mask by ``dilate_radius`` voxels, and (4) returns
    ``volume * dilated_mask`` (zero everywhere outside the dilated atom
    neighbourhood).

    The detector branch runs under ``torch.no_grad()`` and the target is
    detached, so gradient flows only through ``volume``. The penalty
    pulls the volume toward the atom-supported version — soft sparsity
    that respects atom-blob structure rather than per-voxel L1.

    Args:
        volume:        ``[Ny, Nz, Nx]`` float32 tensor.
        sigma_atom:    σ of the blur kernel used by the peak detector
                       (in voxels). Set near the expected atom radius.
        min_distance:  NMS half-width (voxels). Voxels closer than this
                       to a stronger peak are suppressed.
        dilate_radius: Half-width of the cube dilation that defines each
                       peak's support. ``f_atom`` keeps voxels within
                       this radius of any peak.
        threshold_rel: Peak detection threshold, expressed as a fraction
                       of ``volume.max()``. Peaks below this are ignored.
        cull_max_neighbor_dist: If set, drop peaks whose nearest neighbour
                       is farther than this many voxels before building
                       the support mask (isolated-atom culling).

    Returns:
        Scalar tensor on the same device as ``volume``.
    """
    if volume.dtype != torch.float32:
        raise TypeError(f"atom_red_loss is fp32-only (got {volume.dtype}).")
    if volume.ndim != 3:
        raise ValueError(f"atom_red_loss expects a 3-D [Ny, Nz, Nx] tensor, "
                         f"got shape {tuple(volume.shape)}")

    v = volume.detach()
    vmax = float(v.max())
    if vmax <= 0.0:
        # Degenerate volume — penalty is trivially zero, but we still
        # need a tensor that backprop sees as zero.
        return 0.5 * (volume ** 2).mean() * 0.0

    with torch.no_grad():
        threshold = threshold_rel * vmax
        v_blur = _gaussian_blur_3d(v, sigma_atom)
        peaks = _detect_atom_peaks_3d(v_blur, min_distance, threshold)
        if cull_nn_zscore is not None and cull_nn_zscore > 0:
            peaks = _cull_isolated_peaks_zscore(peaks, cull_nn_zscore)
        if cull_max_neighbor_dist is not None and cull_max_neighbor_dist > 0:
            peaks = _cull_isolated_peaks(peaks, cull_max_neighbor_dist)
        mask = _dilate_3d(peaks, dilate_radius).to(v.dtype)
        target = v * mask

    return 0.5 * ((volume - target) ** 2).mean()


# ── In-line Gaussian-atom projection ──────────────────────────────────────

@torch.no_grad()
def gaussian_atom_project(
    volume: torch.Tensor,
    *,
    sigma_atom: float = 1.0,
    min_distance: int = 2,
    threshold_rel: float = 0.05,
    cull_max_neighbor_dist: float | None = None,
    cull_nn_zscore: float | None = None,
    mass_conserving: bool = False,
) -> torch.Tensor:
    """Snap-render projection onto the Gaussian-atom manifold.

    Used as a **non-differentiable projection step** inside the AD loop:
    every K iterations, replace ``vol.data`` with the output of this
    function so the volume snaps back to a clean sum-of-Gaussians and
    the next K AD iterations refine sub-pixel positions and amplitudes
    around the snapped atoms.

    Algorithm (one fused ``conv3d`` call after detection):
        1. Blur volume with σ = ``sigma_atom``; detect local-max peaks.
        2. Build a *spike volume*: zeros everywhere except at peak voxels,
           where the value is ``volume[peak]`` (integer-snapped position).
        3. Render = ``F.conv3d(spike, gaussian_kernel(sigma_atom))``.

    Sub-pixel localisation is sacrificed at the snap — the outer AD loop
    reclaims it before the next snap.

    Args:
        volume:        ``[Ny, Nz, Nx]`` float32 tensor.
        sigma_atom:    σ of the rendering Gaussian (voxels). Should
                       typically match the σ used for detection.
        min_distance:  NMS half-width for peak detection (voxels).
        threshold_rel: Peak threshold as a fraction of ``volume.max()``.
        cull_max_neighbor_dist: If set, drop peaks whose nearest neighbour
                       is farther than this many voxels (isolated-atom
                       culling — true atoms always sit within bond length
                       of at least one neighbour).
        mass_conserving: If True, normalise the rendering kernel to sum to
                       1, so the rendered volume's *integral* per peak
                       equals the source amplitude. Default False reuses
                       a centre-equals-1 kernel: the peak voxel's
                       amplitude is preserved at the cost of total mass.

    Returns:
        ``[Ny, Nz, Nx]`` float32 tensor on the same device, with
        ``requires_grad=False``. Returns a zero volume if no peaks are
        detected.
    """
    if volume.dtype != torch.float32:
        raise TypeError(f"gaussian_atom_project is fp32-only "
                        f"(got {volume.dtype}).")
    if volume.ndim != 3:
        raise ValueError(f"gaussian_atom_project expects a 3-D tensor, "
                         f"got shape {tuple(volume.shape)}")

    v = volume.detach()
    vmax = float(v.max())
    if vmax <= 0.0:
        return torch.zeros_like(v)

    threshold = threshold_rel * vmax
    v_blur = _gaussian_blur_3d(v, sigma_atom)
    peaks = _detect_atom_peaks_3d(v_blur, min_distance, threshold)
    if cull_nn_zscore is not None and cull_nn_zscore > 0:
        peaks = _cull_isolated_peaks_zscore(peaks, cull_nn_zscore)
    if cull_max_neighbor_dist is not None and cull_max_neighbor_dist > 0:
        peaks = _cull_isolated_peaks(peaks, cull_max_neighbor_dist)

    if not bool(peaks.any()):
        return torch.zeros_like(v)

    # Spike volume: amplitude = original (unblurred) value at each peak.
    spike = torch.where(peaks, v, torch.zeros_like(v))

    radius = max(1, int(math.ceil(3.0 * sigma_atom)))
    kernel = _gaussian_kernel_3d(sigma_atom, radius=radius,
                                 device=v.device, dtype=v.dtype)
    if mass_conserving:
        kernel = kernel / kernel.sum()
    x = spike[None, None, :, :, :]
    x = F.pad(x, (radius,) * 6, mode="constant", value=0.0)
    rendered = F.conv3d(x, kernel)
    return rendered[0, 0]
