"""Differentiable regularisers for the AD reconstruction path.

Each function returns a scalar tensor; add ``weight * regularizer(...)``
to the data-fidelity loss and call ``.backward()`` as usual.

* :func:`logsum_sparsity` — Log-Sum L0-proxy sparsity penalty (Candès–Wakin–Boyd
  2008). Smoother than L1; gradient *shrinks* for large |ρ| so true signal peaks
  are barely penalised while background voxels are driven aggressively toward zero.

* :func:`tilt_ssim_loss` — Wang-2004 single-scale SSIM applied per tilt on the
  actual 2-D projection image (local neighbourhoods). Deterministic; penalises
  local structural dissimilarity within each projection without the Y-slab banding
  risk of 3-D TV.

* :func:`s3im_loss` — Stochastic Structural SIMilarity (Xie et al. 2023,
  ``fasttomo-wiki/raw/papers/S3IM.pdf``). Each step, randomly samples B
  sinogram entries from anywhere in ``pred``/``target``, reshapes them into a
  fake √B × √B "patch", and applies SSIM with a K×K kernel at stride K.
  Captures *non-local* sinogram structure; stochastic noise at small lr can
  cause hot-pixel speckling in the volume.

All return scalar tensors that drop directly into any autograd loop.
* :func:`tv_loss_3d` — 3-D isotropic total-variation regularisation on the
  volume (fused CUDA kernel, PyTorch on CPU). Moved here from
  ``fasttomo.utilities``, which re-exports it.

Combine them additively in the loss expression as needed.
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
    place of the pure-PyTorch expression. On CUDA, dispatches to a fused
    kernel (one read-pass over the volume and a deterministic fixed-order
    two-pass reduction with fp64 partials, no atomics, so the value is
    bit-reproducible run to run) and a voxel-driven gather kernel for the
    backward. On CPU,
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


# ── XZ-plane 2-D isotropic TV (no Y-coupling) ────────────────────────────

def tv_loss_xz(volume: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Isotropic 2-D total-variation in the XZ plane of every Y-slice.

        loss = mean( sqrt( dz² + dx² + eps ) )

    The mean runs over all ``Ny * (Nz-1) * (Nx-1)`` forward-difference sites
    (forward differences at ``[:, :-1, :-1]``), so the value is an average,
    not a sum over slices.

    Only penalises X and Z gradients — Y-gradients are **never** touched.
    This is the physically appropriate TV for single-axis tilt tomography:
    each Y-slice reconstructs an independent XZ cross-section, so smoothing
    should operate within that plane. 3-D isotropic TV includes a Y term
    that drives adjacent Y-slices toward identical values, producing
    horizontal-band artefacts in axis-2 projections; this function avoids
    that by construction.

    Also available as ``tv_loss_zx`` (the same function; the volume axes
    are ``[Ny, Nz, Nx]``).

    Args:
        volume: [Ny, Nz, Nx] float32 tensor.
        eps:    smoothing constant inside the sqrt (default 1e-8).

    Returns:
        Scalar tensor (shape ``[]``) on the same device as ``volume``.
    """
    if volume.dtype != torch.float32:
        raise TypeError(f"tv_loss_xz is fp32-only (got {volume.dtype}).")
    if volume.ndim != 3:
        raise ValueError(f"tv_loss_xz expects [Ny, Nz, Nx], got {tuple(volume.shape)}")
    dz = volume[:, 1:, :]  - volume[:, :-1, :]   # [Ny, Nz-1, Nx]
    dx = volume[:, :, 1:]  - volume[:, :, :-1]    # [Ny, Nz, Nx-1]
    return (dz[:, :, :-1].pow(2) + dx[:, :-1, :].pow(2) + eps).sqrt().mean()


# Same function object under the other axis order (not deprecated): the
# volume axes are [Ny, Nz, Nx], so "zx" names the plane in storage order.
tv_loss_zx = tv_loss_xz


# ── Log-Sum L0-proxy sparsity ─────────────────────────────────────────────

def logsum_sparsity(volume: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """Log-Sum L0-proxy sparsity penalty (Candès–Wakin–Boyd, 2008).

        loss = mean( log(1 + |volume| / eps) )
        ∂loss/∂ρ = sign(ρ) / (eps + |ρ|) / N

    Smoother than L1; the gradient *shrinks* for large |ρ|, so true signal
    peaks are barely shrunk while background voxels are driven aggressively
    toward zero. Recommended over L1 soft-thresholding when the volume has
    a sparse signal on a near-zero background.

    Args:
        volume: float tensor (any shape, CPU or CUDA).
        eps:    smoothing scale. Set near 0.1 × the expected peak amplitude.
                Smaller eps = more aggressive zeroing.

    Returns:
        Scalar tensor on the same device as ``volume``.
    """
    return torch.log1p(volume.abs() / eps).mean()


# ── Shared helpers ────────────────────────────────────────────────────────

def _gaussian_window(win_size: int, sigma: float,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Separable 2-D Gaussian kernel of shape [1, 1, win, win], summing to 1."""
    coords = torch.arange(win_size, device=device, dtype=dtype) - (win_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w2 = g[:, None] * g[None, :]
    return w2[None, None, :, :]


def _ssim_constants(target: torch.Tensor, data_range: float | None):
    """SSIM's ``C1 = (0.01 L)^2`` and ``C2 = (0.03 L)^2``.

    ``L = data_range``, or ``target.max() - target.min()`` when it is None,
    computed on the device without a host sync: 0-d tensors with the values
    the former ``float(target.max() - target.min())`` gave (fp64 arithmetic,
    rounded to fp32 once, as the fp32 ops did with the Python float), so the
    loss is unchanged bit for bit."""
    if data_range is not None:
        return (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    L = (target.max() - target.min()).detach().double()
    return ((0.01 * L) ** 2).float(), ((0.03 * L) ** 2).float()


def _ssim_map(x: torch.Tensor, y: torch.Tensor,
              window: torch.Tensor, *,
              C1: float, C2: float,
              stride: int = 1, padding: int = 0) -> torch.Tensor:
    """Standard Wang-2004 SSIM map for ``[B, 1, H, W]`` inputs.

    ``window`` is a normalised ``[1, 1, K, K]`` kernel. ``stride`` and
    ``padding`` are forwarded to the conv2d calls verbatim, so the caller
    chooses whether windows overlap (stride 1, padding K//2 with reflect-pad
    upstream) or tile (stride=K, padding=0).
    """
    mu_x  = F.conv2d(x,     window, stride=stride, padding=padding)
    mu_y  = F.conv2d(y,     window, stride=stride, padding=padding)
    mu_xx = F.conv2d(x * x, window, stride=stride, padding=padding)
    mu_yy = F.conv2d(y * y, window, stride=stride, padding=padding)
    mu_xy = F.conv2d(x * y, window, stride=stride, padding=padding)

    sigma_xx = mu_xx - mu_x * mu_x
    sigma_yy = mu_yy - mu_y * mu_y
    sigma_xy = mu_xy - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_xx + sigma_yy + C2)
    return num / den


# ── Per-tilt SSIM loss on predicted projections ───────────────────────────

def tilt_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    win_size: int = 11,
    sigma: float = 1.5,
    data_range: float | None = None,
) -> torch.Tensor:
    """Per-tilt structural-similarity loss on a stack of 2-D projection images.

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

    C1, C2 = _ssim_constants(target, data_range)

    pad = win_size // 2
    window = _gaussian_window(win_size, sigma, pred.device, pred.dtype)

    x = pred.unsqueeze(1)        # [B, 1, H, W]
    y = target.unsqueeze(1)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.pad(y, (pad, pad, pad, pad), mode="reflect")

    ssim_map = _ssim_map(x, y, window, C1=C1, C2=C2, stride=1, padding=0)
    return 1.0 - ssim_map.mean()


# ── Stochastic Structural SIMilarity (S3IM) ───────────────────────────────

def s3im_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    patch_pixels: int = 4096,
    kernel_size: int = 4,
    n_patches: int = 1,
    data_range: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Stochastic Structural SIMilarity loss (Xie et al. 2023, S3IM).

        loss = 1 - (1/M) Σ_m  SSIM( P^(m)(pred), P^(m)(target) )

    Each step samples ``patch_pixels`` (B) random sinogram entries (without
    replacement) from anywhere in ``pred`` / ``target`` — the same indices for
    both — and reshapes them into a ``[√B × √B]`` "patch". A K×K SSIM is then
    computed at stride K (non-overlapping windows) on that patch. The
    procedure is repeated ``n_patches`` (M) times per call and averaged.

    Captures non-local structural information that ``tilt_ssim_loss`` cannot
    reach — pixels in the fake patch come from arbitrary tilts and detector
    positions, so the loss penalises mismatched joint distributions of
    sinogram values across the entire dataset rather than only within each
    projection image. The paper notes M=1 already matches M=10 in practice,
    so the per-step cost is essentially one ``conv2d`` over a small patch.

    Args:
        pred, target: ``[n_angles, Ny, N_det]`` float32 tensors on the same
            device. (Any same-shape pair works; the function flattens both.)
        patch_pixels: B, the number of sinogram entries gathered per patch.
            Must be a perfect square so it can be reshaped to √B × √B.
            Default ``4096`` → 64 × 64 patch.
        kernel_size: K, the SSIM kernel side length. Stride is also K (the
            paper sets stride = K so windows tile the patch without overlap).
            Default ``4``.
        n_patches: M, how many independent random patches to sample and
            average over per call. Default ``1`` (paper notes M=1 ≈ M=10).
        data_range: dynamic range for the SSIM C1/C2 constants. If ``None``,
            uses ``target.max() - target.min()`` (skimage convention; matches
            ``tilt_ssim_loss``).
        generator: optional ``torch.Generator`` for reproducible index
            sampling. If ``None``, uses the global RNG.

    Returns:
        Scalar tensor on the same device as ``pred``. ``0`` for identical
        inputs, up to ``2`` in the worst case.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs "
                         f"target {tuple(target.shape)}")
    if pred.dtype != torch.float32 or target.dtype != torch.float32:
        raise TypeError("s3im_loss is fp32-only "
                        f"(got pred={pred.dtype}, target={target.dtype}).")
    side = int(round(math.sqrt(patch_pixels)))
    if side * side != patch_pixels:
        raise ValueError(f"patch_pixels must be a perfect square (got "
                         f"{patch_pixels}); √{patch_pixels} = {math.sqrt(patch_pixels):.4f}")
    if side % kernel_size != 0:
        raise ValueError(f"√patch_pixels ({side}) must be divisible by "
                         f"kernel_size ({kernel_size})")
    if n_patches < 1:
        raise ValueError(f"n_patches must be ≥ 1, got {n_patches}")

    n_total = pred.numel()
    if patch_pixels > n_total:
        raise ValueError(f"patch_pixels ({patch_pixels}) exceeds total sinogram "
                         f"size ({n_total})")

    C1, C2 = _ssim_constants(target, data_range)

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    # Gaussian window with σ matched to kernel size — uniform-ish at K=4.
    window = _gaussian_window(kernel_size, kernel_size / 2.0,
                              pred.device, pred.dtype)

    accum = pred.new_zeros(())
    for _ in range(n_patches):
        # Random indices without replacement, shared between pred and target.
        if generator is None:
            idx = torch.randperm(n_total, device=pred.device)[:patch_pixels]
        else:
            idx = torch.randperm(n_total, device=pred.device,
                                 generator=generator)[:patch_pixels]

        x = pred_flat.gather(0, idx).reshape(1, 1, side, side)
        y = target_flat.gather(0, idx).reshape(1, 1, side, side)

        ssim_map = _ssim_map(x, y, window, C1=C1, C2=C2,
                             stride=kernel_size, padding=0)
        accum = accum + ssim_map.mean()

    return 1.0 - accum / n_patches
