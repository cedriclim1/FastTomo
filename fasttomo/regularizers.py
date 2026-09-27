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

    A dimension of size 1 leaves no corners: the loss and its gradient are
    0 on both backends. In particular a single-slice volume (``Ny = 1``)
    gets no 3-D TV at all, not a 2-D one; use :func:`tv_loss_xz` for a
    per-slice XZ TV.

    The CUDA path is first-order only: a backward with
    ``create_graph=True`` raises RuntimeError (the CPU expression is
    differentiable to any order).

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
    if min(volume.shape) < 2:
        # no corners: 0 like the CUDA kernel (the mean of an empty tensor
        # would be NaN); an empty sum stays attached to the graph (zero
        # gradient) and is 0 whatever the volume holds
        return volume.reshape(-1)[:0].sum()
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

    On CUDA, a fused kernel (the planar mode of the 3-D TV kernel: one pass
    over the volume, deterministic fixed-order two-pass reduction with fp64
    partials, voxel-driven gather backward); on CPU, the PyTorch
    expression. ``Nz = 1`` or ``Nx = 1`` leaves no corners: 0 on both. The
    CUDA path is first-order only (``create_graph=True`` raises
    RuntimeError); the CPU expression is differentiable to any order.

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
    if volume.is_cuda:
        from fasttomo.autograd import _TVLossXZ
        return _TVLossXZ.apply(volume, eps)

    # CPU fallback — pure PyTorch, autograd works automatically.
    if min(volume.shape[1:]) < 2:
        return volume.reshape(-1)[:0].sum()    # no corners: 0, like CUDA
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

def _gaussian_1d(win_size: int, sigma: float,
                 device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalised 1-D Gaussian ``g`` of length ``win_size`` (sums to 1). The
    2-D SSIM window is ``g[:, None] * g[None, :]``, so filtering with it is
    the two 1-D passes of `_blur`."""
    coords = torch.arange(win_size, device=device, dtype=dtype) - (win_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    return g / g.sum()


def _first_order_only(what: str) -> None:
    """Called at the top of the hand-written backwards (the SSIM helpers
    and the fused TV kernels): they compute the first derivative only.
    Autograd runs a backward with grad mode enabled exactly when a graph
    of the gradient is requested (``create_graph=True``, e.g. for a
    Hessian-vector product); the result would then treat this backward as
    a constant and give a wrong second derivative, so raise instead.
    (``once_differentiable`` catches only the case where the incoming
    gradient itself requires grad, not a second derivative with respect to
    the input.)"""
    if torch.is_grad_enabled():
        raise RuntimeError(
            f"{what} supports first-order gradients only: its backward is not "
            "differentiable, so create_graph=True (double backward, "
            "Hessian-vector products) is not supported")


class _Blur(torch.autograd.Function):
    """Separable Gaussian filter of every channel, as a Function that saves
    nothing but the kernel: the input gradient of a convolution needs only
    the weights, but autograd through ``F.conv2d`` would keep both passes'
    inputs alive until the backward."""

    @staticmethod
    def forward(ctx, maps, g, stride):
        C, K = maps.shape[1], g.numel()
        kw = g.view(1, 1, 1, K).expand(C, 1, 1, K)
        kh = g.view(1, 1, K, 1).expand(C, 1, K, 1)
        mid = F.conv2d(maps, kw, stride=(1, stride), groups=C)
        out = F.conv2d(mid, kh, stride=(stride, 1), groups=C)
        ctx.save_for_backward(g)
        ctx.shapes = (maps.shape, mid.shape)
        ctx.stride = stride
        return out

    @staticmethod
    def backward(ctx, grad):
        _first_order_only("the SSIM Gaussian filter (tilt_ssim_loss / s3im_loss)")
        (g,) = ctx.saved_tensors
        in_shape, mid_shape = ctx.shapes
        s = ctx.stride
        C, K = in_shape[1], g.numel()
        kw = g.view(1, 1, 1, K).expand(C, 1, 1, K)
        kh = g.view(1, 1, K, 1).expand(C, 1, K, 1)
        grad_mid = torch.nn.grad.conv2d_input(mid_shape, kh, grad.contiguous(),
                                              stride=(s, 1), groups=C)
        grad_in = torch.nn.grad.conv2d_input(in_shape, kw, grad_mid,
                                             stride=(1, s), groups=C)
        return grad_in, None, None


def _blur(maps: torch.Tensor, g: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """Separable Gaussian filter of every channel of ``maps`` ``[B, C, H, W]``
    ("valid" region, no padding): a depthwise 1 x K pass along W, then a
    K x 1 pass along H, each with ``stride`` along its own axis. Equal to
    one 2-D pass with the K x K window at stride ``stride``, in 2K
    instead of K^2 multiply-adds per output (the 2-D window is the outer
    product ``g[:, None] * g[None, :]``). Depthwise convolutions do not
    take cuDNN's TF32 tensor-core path (the dense single-channel 11 x 11
    convolution did, A6-1), and their input gradient avoids the slow dense
    one (A6-2)."""
    return _Blur.apply(maps, g, stride)


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


class _SSIMMean(torch.autograd.Function):
    """``mean(SSIM map)`` from stacked local moments ``E = [ex, exx, edd]``
    (``[B, 3, H, W]``) of the prediction and the target's ``ey``, ``eyy``
    (``[B, 1, H, W]``), with the closed-form gradient.
    Saves only ``E``; the backward works in place so that about six
    map-sized temporaries are alive at once (autograd through the formula
    keeps about ten until the backward and then makes as many more)."""

    @staticmethod
    def forward(ctx, E, ey, eyy, c, C1, C2):
        ex, exx, edd = E.unbind(1)
        ey, eyy, c = ey.squeeze(1), eyy.squeeze(1), c.squeeze(1)
        dm = ex - ey
        mx, my = ex + c, ey + c
        lum = 1.0 - dm * dm / (mx * mx + my * my + C1)
        cs = 1.0 - (edd - dm * dm) / ((exx - ex * ex) + (eyy - ey * ey) + C2)
        ctx.save_for_backward(E, ey, eyy, c)
        ctx.C = (C1, C2)
        return (lum * cs).mean()

    @staticmethod
    def backward(ctx, grad):
        _first_order_only("the SSIM map (tilt_ssim_loss / s3im_loss)")
        E, ey, eyy, c = ctx.saved_tensors
        C1, C2 = ctx.C
        ex, exx, edd = E.unbind(1)
        g = grad / ex.numel()
        # lum = 1 - dm^2 / d1, d1 = mu_x^2 + mu_y^2 + C1
        # cs  = 1 - var_d / d2, var_d = edd - dm^2, d2 = s_xx + s_yy + C2
        dm = ex - ey
        d1 = (ex + c).square_().add_((ey + c).square_()).add_(C1)
        r = dm / d1                                         # dm / d1
        lum = (dm * r).neg_().add_(1.0)
        d2 = (ex * ex).neg_().add_(exx).add_(eyy).sub_(ey * ey).add_(C2)
        var_d = (dm * dm).neg_().add_(edd)
        q = var_d.div_(d2)                                  # var_d / d2
        cs = (1.0 - q)
        out = torch.empty_like(E)
        g_ex, g_exx, g_edd = out.unbind(1)
        # d/d exx = lum var_d / d2^2 ;  d/d edd = -lum / d2
        torch.mul(lum, q, out=g_exx).div_(d2).mul_(g)
        torch.div(lum, d2, out=g_edd)                      # lum / d2 (reused)
        # d/d ex = cs dlum/dex + lum dcs/dex
        #        = 2 cs r^2 mu_x - 2 cs r + 2 (lum / d2) dm - 2 (lum var_d / d2^2) ex
        a = r.square().mul_(cs)                             # cs r^2
        b = g_edd * dm - cs.mul_(r)                          # lum/d2 dm - cs r
        torch.add(ex, c, out=g_ex).mul_(a).add_(b).mul_(2.0 * g)
        g_ex.sub_(g_exx * ex, alpha=2.0)
        g_ey = g_eyy = None
        if ctx.needs_input_grad[1]:       # a target that requires grad
            g_ey = (ey + c).mul_(a).sub_(b).mul_(2.0 * g)
            g_ey = g_ey.sub_(g_exx * ey, alpha=2.0).unsqueeze(1)
        if ctx.needs_input_grad[2]:
            g_eyy = g_exx.clone().unsqueeze(1)
        g_edd.mul_(-g)
        del d1, r, lum, d2, var_d, cs, a, b, dm
        return out, g_ey, g_eyy, None, None, None


def _ssim_from_moments(E, ey, eyy, c, C1, C2) -> torch.Tensor:
    """Mean of the Wang-2004 SSIM map from local moments of centred data.

    ``x' = x - c``, ``y' = y - c`` (``c`` a constant, e.g. the target's
    mean) and Gaussian-weighted local means ``E = [ex, exx, edd]`` with
    ``ex = E[x']``, ``exx = E[x'^2]``, ``edd = E[(x' - y')^2]``, and
    ``ey = E[y']``, ``eyy = E[y'^2]``. With ``mu = e + c``,
    ``sigma_xx = exx - ex^2``, ``sigma_yy = eyy - ey^2`` and
    ``2 sigma_xy = sigma_xx + sigma_yy - var(x - y)``, the textbook

        SSIM = (2 mu_x mu_y + C1) (2 sigma_xy + C2)
               / ((mu_x^2 + mu_y^2 + C1) (sigma_xx + sigma_yy + C2))

    is evaluated as ``(1 - (ex - ey)^2 / (mu_x^2 + mu_y^2 + C1)) *
    (1 - var(x - y) / (sigma_xx + sigma_yy + C2))``: the same number, but no
    step subtracts two values of the size of ``E[x^2]``. The textbook
    ``E[x^2] - mu^2`` loses about ``mean^2 / variance`` of relative precision
    (1e4 at sinogram scale), which in TF32 was a 1-6 % error (A6-1)."""
    return _SSIMMean.apply(E, ey, eyy, c, C1, C2)


class _TiltSSIMTarget:
    """The target-only part of `tilt_ssim_loss`, computed once.

    ``run_ad_recon`` compares every iteration's prediction against the
    same measured sinogram, so the target's centre, padding, local moments
    and SSIM constants are built once here; each call then filters only
    ``[x', x'^2, (x' - y')^2]``. ``tilt_ssim_loss`` builds one per call,
    so both give the same value."""

    def __init__(self, target: torch.Tensor, *, win_size: int = 11,
                 sigma: float = 1.5, data_range: float | None = None) -> None:
        self.pad = win_size // 2
        self.g = _gaussian_1d(win_size, sigma, target.device, target.dtype)
        self.C1, self.C2 = _ssim_constants(target, data_range)
        # per-tilt centre [B, 1, 1, 1]: the statistics below are then of
        # values of the size of the local contrast, not of the sinogram
        # (detached: the SSIM value does not depend on it, and a gradient
        # to a target that requires grad then flows only through y')
        self.c = target.detach().mean(dim=(-2, -1), keepdim=True).unsqueeze(1)
        self.target = target.unsqueeze(1)
        yp = self._pad(self.target - self.c)
        self.ey, self.eyy = _blur(torch.cat([yp, yp * yp], 1), self.g).split(1, dim=1)

    def _pad(self, v: torch.Tensor) -> torch.Tensor:
        p = self.pad
        return F.pad(v, (p, p, p, p), mode="reflect")

    def __call__(self, pred: torch.Tensor) -> torch.Tensor:
        x = pred.unsqueeze(1)
        xp = self._pad(x - self.c)
        d = self._pad(x - self.target)          # x' - y', the centre cancels
        E = _blur(torch.cat([xp, xp * xp, d * d], 1), self.g)
        return 1.0 - _ssim_from_moments(E, self.ey, self.eyy, self.c, self.C1, self.C2)


# ── Per-tilt SSIM loss on predicted projections ───────────────────────────

def _check_ssim_inputs(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs "
                         f"target {tuple(target.shape)}")
    if pred.ndim != 3:
        raise ValueError(f"tilt_ssim_loss expects [B, H, W], got "
                         f"{tuple(pred.shape)}")
    if pred.dtype != torch.float32 or target.dtype != torch.float32:
        raise TypeError("tilt_ssim_loss is fp32-only "
                        f"(got pred={pred.dtype}, target={target.dtype}).")


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
    default, normalised to 1), ``C1 = (0.01 L)^2`` and ``C2 = (0.03 L)^2``,
    population (not sample) covariances. Each tilt is treated as an
    independent single-channel image; SSIM maps are averaged within a tilt
    and across tilts. Reflect padding (torch ``reflect``: ``d c b | a b c d``)
    keeps the SSIM map covering the full image; the map on the valid
    interior equals scikit-image's ``structural_similarity(...,
    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)``.

    The window is applied as two 1-D passes and the local statistics are
    formed from data centred on each target tilt's mean (see
    ``_ssim_from_moments``), which keeps the value accurate to ~1e-5
    relative in fp32 at sinogram scale on CPU and CUDA, independently of
    cuDNN's TF32 setting.

    First-order only: the gradient is hand-written, and a backward with
    ``create_graph=True`` (double backward, Hessian-vector products)
    raises RuntimeError.

    Args:
        pred, target: ``[B, H, W]`` float32 tensors on the same device;
            ``H, W > win_size // 2`` (reflect padding).
        win_size, sigma: Gaussian window parameters.
        data_range: dynamic range used for the SSIM constants. If ``None``,
            defaults to ``target.max() - target.min()`` (skimage convention).

    Returns:
        Scalar tensor on the same device as ``pred``. ``0`` for identical
        inputs, up to ``2`` in the worst case (since SSIM ∈ [-1, 1]).
    """
    _check_ssim_inputs(pred, target)
    return _TiltSSIMTarget(target, win_size=win_size, sigma=sigma,
                           data_range=data_range)(pred)


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

    First-order only, like :func:`tilt_ssim_loss` (``create_graph=True``
    raises RuntimeError).

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
            sampling. If ``None``, uses the global RNG. The entries of a
            patch are the ``B`` largest of ``pred.numel()`` uniform random
            numbers, in decreasing order (``rand(n).topk(B)``, a uniformly
            random ordered subset, 3x cheaper than the ``randperm(n)[:B]``
            used before 2026-09, which drew a different subset for the same
            seed).

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
    return _S3IMTarget(target, patch_pixels=patch_pixels, kernel_size=kernel_size,
                       n_patches=n_patches, data_range=data_range)(pred, generator)


class _S3IMTarget:
    """`s3im_loss` against a fixed target: the constants ``C1``, ``C2`` (a
    reduction over the whole target) and the window are computed once;
    ``run_ad_recon`` keeps one per run, ``s3im_loss`` builds one per call.

    Each call draws ``n_patches`` index sets without replacement: the
    ``patch_pixels`` largest of ``n_total`` uniform random numbers, in
    decreasing order (``rand(n).topk(k)``: a uniformly random ordered
    subset, like ``randperm(n)[:k]`` but without sorting all ``n``). The
    patches go through the SSIM as one batch; their mean equals the average
    of the per-patch means. The statistics use data centred on each target
    patch's mean (see ``_ssim_from_moments``)."""

    def __init__(self, target: torch.Tensor, *, patch_pixels: int = 4096,
                 kernel_size: int = 4, n_patches: int = 1,
                 data_range: float | None = None) -> None:
        side = int(round(math.sqrt(patch_pixels)))
        if side * side != patch_pixels:
            raise ValueError(f"patch_pixels must be a perfect square (got "
                             f"{patch_pixels}); √{patch_pixels} = {math.sqrt(patch_pixels):.4f}")
        if side % kernel_size != 0:
            raise ValueError(f"√patch_pixels ({side}) must be divisible by "
                             f"kernel_size ({kernel_size})")
        if n_patches < 1:
            raise ValueError(f"n_patches must be ≥ 1, got {n_patches}")
        if patch_pixels > target.numel():
            raise ValueError(f"patch_pixels ({patch_pixels}) exceeds total sinogram "
                             f"size ({target.numel()})")
        self.shape = tuple(target.shape)
        self.target_flat = target.reshape(-1)
        self.side, self.k, self.m, self.kernel = side, patch_pixels, n_patches, kernel_size
        self.C1, self.C2 = _ssim_constants(target, data_range)
        # Gaussian window with σ matched to kernel size — uniform-ish at K=4.
        self.g = _gaussian_1d(kernel_size, kernel_size / 2.0, target.device, target.dtype)

    def _indices(self, generator: torch.Generator | None) -> torch.Tensor:
        n = self.target_flat.numel()
        dev = self.target_flat.device
        draws = [torch.rand(n, device=dev, generator=generator).topk(self.k).indices
                 for _ in range(self.m)]
        return draws[0].unsqueeze(0) if self.m == 1 else torch.stack(draws)

    def __call__(self, pred: torch.Tensor,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        idx = self._indices(generator)                       # [M, B]
        shape = (self.m, 1, self.side, self.side)
        flat = idx.view(-1)
        x = pred.reshape(-1).gather(0, flat).view(shape)
        y = self.target_flat.gather(0, flat).view(shape)
        c = y.detach().mean(dim=(-2, -1), keepdim=True)
        xc, yc, d = x - c, y - c, x - y
        mom = _blur(torch.cat([xc, xc * xc, d * d, yc, yc * yc], 1), self.g,
                    stride=self.kernel)
        return 1.0 - _ssim_from_moments(mom[:, :3], mom[:, 3:4], mom[:, 4:5], c,
                                        self.C1, self.C2)
