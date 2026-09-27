"""
Differentiable regularisers: tilt-SSIM, S3IM, the 3-D isotropic TV loss
(tv_loss_3d, fused CUDA kernel), the XZ-plane TV and the log-sum sparsity.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import (
    needs_cuda,
)

import fasttomo as ft


# ── SSIM constants: device-side, no host sync (scope report 2g) ──────────────


def _ssim_pair(device, seed=5):
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(6, 20, 24, generator=g).to(device)
    target = (3.0 * torch.rand(6, 20, 24, generator=g) - 0.5).to(device)
    return pred, target


@pytest.mark.parametrize("loss", ["tilt_ssim_loss", "s3im_loss"])
def test_ssim_losses_unchanged_by_device_side_data_range(backend, loss):
    """data_range=None now builds C1/C2 on the device instead of reading
    ``float(target.max() - target.min())`` back; the loss (and gradient) is
    the one the explicit float gives: identical on CPU, within 1e-6 on
    CUDA, for a fixed S3IM seed."""
    pred, target = _ssim_pair(backend)
    fn = getattr(ft, loss)
    kw = {}
    if loss == "s3im_loss":
        kw = lambda: dict(patch_pixels=256, n_patches=2,             # noqa: E731
                          generator=torch.Generator(device=backend).manual_seed(42))
    else:
        kw = lambda: {}                                              # noqa: E731
    p1 = pred.clone().requires_grad_(True)
    p2 = pred.clone().requires_grad_(True)
    new = fn(p1, target, **kw())
    old = fn(p2, target, data_range=float(target.max() - target.min()), **kw())
    new.backward()
    old.backward()
    if backend.type == "cpu":
        assert torch.equal(new, old) and torch.equal(p1.grad, p2.grad)
    else:
        assert abs(new.item() - old.item()) <= 1e-6
        torch.testing.assert_close(p1.grad, p2.grad, rtol=0, atol=1e-6)


@needs_cuda
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_ssim_losses_do_not_sync_the_host():
    """Neither regulariser synchronises the host (sync debug mode 'error'
    raises on any synchronising call): the S3IM index draw (randperm with
    a device generator) and the SSIM constants stay on the GPU."""
    pred, target = _ssim_pair("cuda")
    pred.requires_grad_(True)
    gen = torch.Generator(device="cuda").manual_seed(42)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        loss = ft.tilt_ssim_loss(pred, target) + ft.s3im_loss(
            pred, target, patch_pixels=256, generator=gen)
        loss.backward()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert torch.isfinite(loss).item()


def _ssim_map_fp64(x, y, *, win_size=11, sigma=1.5, data_range=None):
    """Reference SSIM map (Wang et al. 2004), written out independently of
    fasttomo: one dense 2-D Gaussian window (normalised to 1), torch
    ``reflect`` padding (scipy's ``mirror``), population covariances,
    ``C1 = (0.01 L)^2``, ``C2 = (0.03 L)^2`` with ``L = max(y) - min(y)``
    unless given, and the textbook formula

        SSIM = (2 mu_x mu_y + C1) (2 sigma_xy + C2)
               / ((mu_x^2 + mu_y^2 + C1) (sigma_xx + sigma_yy + C2)).

    ``x``, ``y``: ``[B, H, W]``; computed in float64 on the CPU."""
    import torch.nn.functional as F

    x, y = x.cpu().double(), y.cpu().double()
    L = float(y.max() - y.min()) if data_range is None else data_range
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    c = torch.arange(win_size, dtype=torch.float64) - (win_size - 1) / 2
    g = torch.exp(-c ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    p = win_size // 2

    def local_mean(v):
        return F.conv2d(F.pad(v.unsqueeze(1), (p, p, p, p), mode="reflect"), w).squeeze(1)

    mu_x, mu_y = local_mean(x), local_mean(y)
    s_xx = local_mean(x * x) - mu_x ** 2
    s_yy = local_mean(y * y) - mu_y ** 2
    s_xy = local_mean(x * y) - mu_x * mu_y
    return ((2 * mu_x * mu_y + C1) * (2 * s_xy + C2)
            / ((mu_x ** 2 + mu_y ** 2 + C1) * (s_xx + s_yy + C2)))


class TestTiltSSIM:
    def test_zero_for_identity(self):
        torch.manual_seed(0)
        x = torch.rand(4, 32, 32, dtype=torch.float32)
        loss = ft.tilt_ssim_loss(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_in_valid_range(self):
        torch.manual_seed(0)
        x = torch.rand(4, 32, 32, dtype=torch.float32)
        y = torch.rand(4, 32, 32, dtype=torch.float32)
        loss = ft.tilt_ssim_loss(x, y).item()
        assert 0.0 <= loss <= 2.0

    def test_increases_with_blur(self):
        torch.manual_seed(0)
        x = torch.rand(4, 32, 32, dtype=torch.float32)
        loss_clean = ft.tilt_ssim_loss(x, x.clone()).item()
        loss_noisy = ft.tilt_ssim_loss(x, x + 0.5 * torch.randn_like(x)).item()
        assert loss_noisy > loss_clean

    def test_grad_flows(self):
        torch.manual_seed(0)
        x = torch.rand(4, 32, 32, dtype=torch.float32, requires_grad=True)
        y = torch.rand(4, 32, 32, dtype=torch.float32)
        loss = ft.tilt_ssim_loss(x, y)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum().item() > 0.0

    def test_matches_skimage(self):
        """The fp64 reference below reproduces scikit-image's
        ``structural_similarity`` on its valid region (value computed with
        scikit-image 0.26 in fp64 on this input, hard-coded so the test does
        not need scikit-image), and tilt_ssim_loss's full reflect-padded map
        matches the same reference."""
        torch.manual_seed(42)
        x = torch.rand(32, 32)
        y = torch.rand(32, 32) * 0.8 + 0.1
        ssim_map = _ssim_map_fp64(x[None], y[None], data_range=1.0)
        pad = 5
        valid = ssim_map[:, pad:-pad, pad:-pad].mean().item()
        assert valid == pytest.approx(0.019964165908094127, abs=1e-12)
        loss = ft.tilt_ssim_loss(x[None], y[None], win_size=11, sigma=1.5,
                                 data_range=1.0).item()
        assert (1.0 - loss) == pytest.approx(ssim_map.mean().item(), abs=2e-7)

    @pytest.mark.parametrize("device,tf32", [
        ("cpu", False),
        pytest.param("cuda", True, marks=needs_cuda),
        pytest.param("cuda", False, marks=needs_cuda)])
    @pytest.mark.parametrize("B,H", [(1, 128), (4, 128), (61, 64), (4, 256)])
    def test_matches_fp64_reference_at_sinogram_scale(self, device, tf32, B, H):
        """A6-1: loss and gradient against an fp64 evaluation of the
        textbook formula, on sinogram-scale data (mean ~100, local std ~1-5)
        where the old ``E[x^2] - mu^2`` form lost 1-6 % under cuDNN's
        TF32 at small batches. cuDNN TF32 is set explicitly both ways."""
        g = torch.Generator().manual_seed(B * 1000 + H)
        base = 100.0 + 5.0 * torch.randn(B, H, H, generator=g)
        target = base + torch.randn(B, H, H, generator=g)
        pred = target * (1 + 0.02 * torch.randn(B, H, H, generator=g))

        x64 = pred.double().requires_grad_(True)
        ref = 1.0 - _ssim_map_fp64(x64, target.double()).mean()
        ref.backward()

        prev = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = tf32
        try:
            x = pred.to(device).requires_grad_(True)
            loss = ft.tilt_ssim_loss(x, target.to(device))
            loss.backward()
        finally:
            torch.backends.cudnn.allow_tf32 = prev
        assert abs(loss.item() - ref.item()) <= 1e-5 * abs(ref.item())
        gerr = (x.grad.cpu().double() - x64.grad).norm() / x64.grad.norm()
        assert gerr <= 1e-5

    def test_cached_target_equals_public_function(self):
        """run_ad_recon's cached target statistics give the public
        function's value and gradient."""
        from fasttomo.regularizers import _TiltSSIMTarget

        g = torch.Generator().manual_seed(3)
        y = 50 + torch.randn(5, 40, 36, generator=g)
        x = (y + 0.3 * torch.randn(5, 40, 36, generator=g)).requires_grad_(True)
        a = ft.tilt_ssim_loss(x, y)
        (ga,) = torch.autograd.grad(a, x)
        b = _TiltSSIMTarget(y)(x)
        (gb,) = torch.autograd.grad(b, x)
        assert torch.equal(a, b) and torch.equal(ga, gb)

    def test_gradient_reaches_a_target_that_requires_grad(self):
        g = torch.Generator().manual_seed(4)
        y64 = (torch.rand(2, 24, 24, generator=g)).double().requires_grad_(True)
        x = torch.rand(2, 24, 24, generator=g)
        (1.0 - _ssim_map_fp64(x.double(), y64, data_range=1.0).mean()).backward()
        y = y64.detach().float().requires_grad_(True)
        ft.tilt_ssim_loss(x, y, data_range=1.0).backward()
        torch.testing.assert_close(y.grad.double(), y64.grad, rtol=1e-4, atol=1e-7)

    def test_shape_mismatch_raises(self):
        x = torch.rand(4, 32, 32, dtype=torch.float32)
        y = torch.rand(4, 16, 16, dtype=torch.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            ft.tilt_ssim_loss(x, y)

    def test_ndim_mismatch_raises(self):
        x = torch.rand(4, 4, 32, 32, dtype=torch.float32)
        y = torch.rand(4, 4, 32, 32, dtype=torch.float32)
        with pytest.raises(ValueError, match=r"\[B, H, W\]"):
            ft.tilt_ssim_loss(x, y)

    def test_dtype_raises(self):
        x = torch.rand(4, 32, 32, dtype=torch.float64)
        y = torch.rand(4, 32, 32, dtype=torch.float64)
        with pytest.raises(TypeError, match="fp32"):
            ft.tilt_ssim_loss(x, y)


class TestS3IM:
    @pytest.mark.parametrize("device,tf32", [
        ("cpu", False),
        pytest.param("cuda", True, marks=needs_cuda),
        pytest.param("cuda", False, marks=needs_cuda)])
    def test_matches_fp64_reference_with_the_same_indices(self, device, tf32):
        """A6-1 / A6-9: value and gradient against an fp64 textbook SSIM
        (dense K x K Gaussian window, sigma K/2, stride K) of the same
        patches, at sinogram scale; each patch is drawn without replacement."""
        import torch.nn.functional as F
        from fasttomo.regularizers import _S3IMTarget

        g = torch.Generator().manual_seed(11)
        y = 100.0 + 5.0 * torch.randn(9, 40, 40, generator=g)
        x = y + torch.randn(9, 40, 40, generator=g)
        M, K, side = 3, 4, 32
        gen = torch.Generator(device=device).manual_seed(5)
        idx = _S3IMTarget(y.to(device), patch_pixels=side * side, kernel_size=K,
                          n_patches=M)._indices(gen).cpu()
        assert idx.shape == (M, side * side)
        assert all(row.unique().numel() == side * side for row in idx)

        x64 = x.double().requires_grad_(True)
        L = float(y.max() - y.min())
        C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
        c = torch.arange(K, dtype=torch.float64) - (K - 1) / 2
        w1 = torch.exp(-c ** 2 / (2 * (K / 2.0) ** 2))
        w1 = w1 / w1.sum()
        w = (w1[:, None] * w1[None, :])[None, None]
        px = x64.reshape(-1)[idx].view(M, 1, side, side)
        py = y.double().reshape(-1)[idx].view(M, 1, side, side)

        def lm(v):
            return F.conv2d(v, w, stride=K)

        mx, my = lm(px), lm(py)
        sxx, syy, sxy = lm(px * px) - mx ** 2, lm(py * py) - my ** 2, lm(px * py) - mx * my
        ssim = ((2 * mx * my + C1) * (2 * sxy + C2)
                / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2)))
        ref = 1.0 - ssim.mean()
        ref.backward()

        prev = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = tf32
        try:
            xd = x.to(device).requires_grad_(True)
            gen = torch.Generator(device=device).manual_seed(5)
            loss = ft.s3im_loss(xd, y.to(device), patch_pixels=side * side, kernel_size=K,
                                n_patches=M, generator=gen)
            loss.backward()
        finally:
            torch.backends.cudnn.allow_tf32 = prev
        assert abs(loss.item() - ref.item()) <= 1e-5 * abs(ref.item())
        gerr = (xd.grad.cpu().double() - x64.grad).norm() / x64.grad.norm()
        assert gerr <= 1e-5

    def test_zero_for_identity(self):
        torch.manual_seed(0)
        x = torch.rand(29, 200, 200, dtype=torch.float32)
        loss = ft.s3im_loss(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_in_valid_range(self):
        torch.manual_seed(0)
        x = torch.rand(29, 200, 200, dtype=torch.float32)
        y = torch.rand(29, 200, 200, dtype=torch.float32)
        loss = ft.s3im_loss(x, y).item()
        assert 0.0 <= loss <= 2.0

    def test_increases_with_blur(self):
        torch.manual_seed(0)
        x = torch.rand(29, 200, 200, dtype=torch.float32)
        gen = torch.Generator().manual_seed(123)
        loss_clean = ft.s3im_loss(x, x.clone(), generator=gen).item()
        gen = torch.Generator().manual_seed(123)
        loss_noisy = ft.s3im_loss(
            x, x + 0.5 * torch.randn_like(x), generator=gen,
        ).item()
        assert loss_noisy > loss_clean

    def test_grad_flows(self):
        torch.manual_seed(0)
        x = torch.rand(8, 64, 64, dtype=torch.float32, requires_grad=True)
        y = torch.rand(8, 64, 64, dtype=torch.float32)
        loss = ft.s3im_loss(x, y)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum().item() > 0.0

    def test_reproducible_with_generator(self):
        torch.manual_seed(0)
        x = torch.rand(29, 100, 100, dtype=torch.float32)
        y = torch.rand(29, 100, 100, dtype=torch.float32)
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        l1 = ft.s3im_loss(x, y, generator=g1).item()
        l2 = ft.s3im_loss(x, y, generator=g2).item()
        assert l1 == pytest.approx(l2, abs=1e-6)

    def test_m_averages_lowers_variance(self):
        """M=10 gives more deterministic value than M=1 across draws."""
        torch.manual_seed(0)
        x = torch.rand(29, 200, 200, dtype=torch.float32)
        y = torch.rand(29, 200, 200, dtype=torch.float32)

        def std_over_draws(M):
            vals = []
            for seed in range(8):
                g = torch.Generator().manual_seed(seed)
                vals.append(ft.s3im_loss(x, y, n_patches=M, generator=g).item())
            return torch.tensor(vals).std().item()

        std1 = std_over_draws(1)
        std10 = std_over_draws(10)
        assert std10 < std1

    def test_invalid_patch_pixels_raises(self):
        x = torch.rand(29, 200, 200, dtype=torch.float32)
        y = torch.rand(29, 200, 200, dtype=torch.float32)
        with pytest.raises(ValueError, match="perfect square"):
            ft.s3im_loss(x, y, patch_pixels=1000)

    def test_kernel_does_not_divide_side_raises(self):
        x = torch.rand(29, 200, 200, dtype=torch.float32)
        y = torch.rand(29, 200, 200, dtype=torch.float32)
        # √4096 = 64; 64 is not divisible by 5
        with pytest.raises(ValueError, match="divisible"):
            ft.s3im_loss(x, y, patch_pixels=4096, kernel_size=5)

    def test_shape_mismatch_raises(self):
        x = torch.rand(29, 200, 200, dtype=torch.float32)
        y = torch.rand(29, 200, 100, dtype=torch.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            ft.s3im_loss(x, y)

    def test_dtype_raises(self):
        x = torch.rand(29, 200, 200, dtype=torch.float64)
        y = torch.rand(29, 200, 200, dtype=torch.float64)
        with pytest.raises(TypeError, match="fp32"):
            ft.s3im_loss(x, y)

    def test_patch_pixels_exceeds_total_raises(self):
        x = torch.rand(2, 4, 4, dtype=torch.float32)  # 32 entries
        y = torch.rand(2, 4, 4, dtype=torch.float32)
        with pytest.raises(ValueError, match="exceeds"):
            ft.s3im_loss(x, y, patch_pixels=64)


# ── tv_loss_3d (fused CUDA kernel) ────────────────────────────────────────────


DEVICE = torch.device("cuda")


def _tv_reference(vol: torch.Tensor, eps: float) -> torch.Tensor:
    dy = vol[1:, :, :] - vol[:-1, :, :]
    dz = vol[:, 1:, :] - vol[:, :-1, :]
    dx = vol[:, :, 1:] - vol[:, :, :-1]
    return (dx[:-1, :-1, :] ** 2 + dy[:, :-1, :-1] ** 2
            + dz[:-1, :, :-1] ** 2 + eps).sqrt().mean()


@needs_cuda
def test_tv_loss_iso_3d_forward_matches_pytorch():
    """Fused CUDA TV loss matches the pure-PyTorch reference within fp32."""
    torch.manual_seed(0)
    vol = torch.randn(16, 24, 20, device=DEVICE) * 5.0
    eps = 1e-8

    loss_cuda = ft.tv_loss_3d(vol, eps=eps)
    loss_ref  = _tv_reference(vol, eps)

    rel = (loss_cuda - loss_ref).abs() / loss_ref.abs().clamp_min(1e-12)
    assert rel.item() < 1e-5, (loss_cuda.item(), loss_ref.item())


@needs_cuda
@pytest.mark.parametrize("shape", [(40, 70, 67), (2, 2, 2), (2, 3, 40), (9, 2, 33), (3, 17, 2),
                                   (5, 300, 7)])
def test_tv_loss_iso_3d_forward_deterministic_and_close_to_fp64(shape):
    """The forward is a fixed-order two-pass reduction (no atomics, fp64
    above the per-lane fp32 sums): bit-identical run to run and within 1e-6
    of an fp64 reference (the fp32 same-address atomicAdd it replaced drifted
    run to run and reached 1e-3 relative on a 512^3 phantom). Shapes cover
    several blocks, partial warps (Nx-1 not a multiple of 32), one corner
    per row and a single row."""
    g = torch.Generator().manual_seed(sum(shape))
    vol = (torch.rand(*shape, generator=g) * 3.0).to(DEVICE)
    vals = [ft.tv_loss_3d(vol, eps=1e-8) for _ in range(5)]
    assert all(torch.equal(v, vals[0]) for v in vals)
    ref = _tv_reference(vol.double(), 1e-8).item()
    assert abs(vals[0].item() - ref) / ref < 1e-6, (vals[0].item(), ref)


@needs_cuda
def test_tv_loss_iso_3d_backward_matches_pytorch():
    """Fused TV backward matches PyTorch autograd on the closed-form expression."""
    torch.manual_seed(1)
    vol_a = torch.randn(12, 14, 10, device=DEVICE, requires_grad=True)
    vol_b = vol_a.detach().clone().requires_grad_(True)
    eps = 1e-7  # avoid eps-domination so the gradient signal is meaningful

    loss_a = ft.tv_loss_3d(vol_a, eps=eps)
    loss_a.backward()

    loss_b = _tv_reference(vol_b, eps)
    loss_b.backward()

    assert vol_a.grad is not None and vol_b.grad is not None
    diff  = (vol_a.grad - vol_b.grad).abs().max()
    scale = vol_b.grad.abs().max().clamp_min(1e-12)
    assert (diff / scale).item() < 1e-4, (diff.item(), scale.item())


@needs_cuda
@pytest.mark.parametrize("shape", [(12, 14, 10), (2, 2, 2), (1, 6, 5), (6, 1, 5), (6, 5, 1)])
def test_tv_loss_iso_3d_backward_ignores_dirty_output_memory(shape):
    """The backward allocates its output without zeroing it (kernel audit
    F10): the kernel writes every voxel when each dimension is >= 2, and a
    degenerate volume still gets an exact zero gradient. Run on reused
    allocator blocks filled with NaN and with zeros: same bits, finite."""
    from fasttomo.autograd import _tv_loss_iso_3d_grad_raw

    torch.manual_seed(2)
    vol = torch.randn(*shape, device=DEVICE)
    g = torch.ones((), device=DEVICE)
    grads = []
    for fill in (float("nan"), 0.0):
        junk = torch.full(shape, fill, device=DEVICE)
        del junk                      # the next same-size allocation reuses it
        grads.append(_tv_loss_iso_3d_grad_raw(vol, g, 1e-7))
    assert torch.equal(grads[0], grads[1])
    assert torch.isfinite(grads[0]).all()
    if min(shape) < 2:
        assert torch.equal(grads[0], torch.zeros_like(vol))


@needs_cuda
def test_tv_loss_iso_3d_chained_backward_matches_pytorch():
    """Chained loss (data + λ·TV) should yield matching gradients."""
    torch.manual_seed(2)
    vol_a = torch.randn(10, 12, 14, device=DEVICE, requires_grad=True)
    vol_b = vol_a.detach().clone().requires_grad_(True)
    target = torch.randn(10, 12, 14, device=DEVICE)
    lam = 0.37

    loss_a = ((vol_a - target) ** 2).mean() + lam * ft.tv_loss_3d(vol_a, eps=1e-7)
    loss_a.backward()
    loss_b = ((vol_b - target) ** 2).mean() + lam * _tv_reference(vol_b, 1e-7)
    loss_b.backward()

    diff  = (vol_a.grad - vol_b.grad).abs().max()
    scale = vol_b.grad.abs().max().clamp_min(1e-12)
    assert (diff / scale).item() < 1e-4


@needs_cuda
def test_tv_loss_iso_3d_scalar_shape_and_device():
    vol = torch.randn(8, 8, 8, device=DEVICE)
    loss = ft.tv_loss_3d(vol)
    assert loss.shape == torch.Size([])
    assert loss.device.type == "cuda"
    assert loss.dtype == torch.float32


@needs_cuda
def test_tv_loss_iso_3d_rejects_fp16():
    vol = torch.randn(8, 8, 8, device=DEVICE, dtype=torch.float16)
    with pytest.raises(TypeError, match="fp32"):
        ft.tv_loss_3d(vol)


@needs_cuda
def test_tv_loss_iso_3d_rejects_wrong_ndim():
    vol = torch.randn(8, 8, device=DEVICE)
    with pytest.raises(ValueError, match="3-D"):
        ft.tv_loss_3d(vol)


# ── tv_loss_xz and logsum_sparsity ────────────────────────────────────────────


def _tv_xz_ref(v: np.ndarray, eps: float) -> float:
    dz = v[:, 1:, :-1] - v[:, :-1, :-1]
    dx = v[:, :-1, 1:] - v[:, :-1, :-1]
    return float(np.sqrt(dz ** 2 + dx ** 2 + eps).mean())


def _fd_grad(f, v: np.ndarray, h: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(v)
    it = np.nditer(v, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        vp = v.copy()
        vp[i] += h
        vm = v.copy()
        vm[i] -= h
        g[i] = (f(vp) - f(vm)) / (2 * h)
    return g


@pytest.mark.parametrize("shape", [(1, 5, 7), (3, 4, 6), (2, 2, 2)])
def test_tv_loss_xz_value_and_grad(backend, shape):
    """Value and gradient against an fp64 NumPy reference (gradient by
    central finite differences). The Y axis is never differenced: a volume
    that varies only along Y has zero TV-xz and zero gradient."""
    eps = 1e-3
    rng = np.random.default_rng(8)
    v = rng.standard_normal(shape)
    t = torch.tensor(v, dtype=torch.float32, device=backend, requires_grad=True)
    loss = ft.tv_loss_xz(t, eps=eps)
    assert loss.shape == () and loss.dtype == torch.float32
    assert loss.item() == pytest.approx(_tv_xz_ref(v, eps), rel=1e-5)
    loss.backward()
    ref = _fd_grad(lambda a: _tv_xz_ref(a, eps), v)
    torch.testing.assert_close(t.grad.cpu().double(), torch.from_numpy(ref), rtol=1e-3, atol=1e-5)

    y_only = torch.arange(shape[0], dtype=torch.float32, device=backend)[:, None, None]
    y_only = y_only.expand(shape).clone().requires_grad_(True)
    val = ft.tv_loss_xz(y_only, eps=0.0)
    val.backward()
    assert val.item() == 0.0


def _tv_xz_torch(vol: torch.Tensor, eps: float) -> torch.Tensor:
    dz = vol[:, 1:, :-1] - vol[:, :-1, :-1]
    dx = vol[:, :-1, 1:] - vol[:, :-1, :-1]
    return (dz ** 2 + dx ** 2 + eps).sqrt().mean()


_XZ_SHAPES = [(1, 5, 7), (3, 4, 6), (2, 2, 2), (5, 33, 65), (4, 2, 33), (7, 31, 2),
              (17, 31, 65), (3, 300, 7), (1, 64, 64), (40, 70, 67)]


@needs_cuda
@pytest.mark.parametrize("shape", _XZ_SHAPES)
def test_tv_loss_xz_cuda_kernel_matches_fp64(shape):
    """A6-5: the planar CUDA kernel against the fp64 formula (value and
    autograd gradient) on shapes with partial warps (Nx - 1 not a multiple
    of 32), one corner per row (Nx = 2), Nz = 2, Ny = 1 and several blocks;
    piecewise-constant data (flat regions, where eps matters) and a 1e3
    offset. The value is bit-identical run to run (fixed-order reduction)."""
    g = torch.Generator().manual_seed(sum(shape))
    for data in ("rand", "flat", "offset"):
        v = torch.rand(*shape, generator=g) * 3.0
        if data == "flat":
            v = (v > 1.5).float()
        elif data == "offset":
            v = v + 1e3
        for eps in (1e-8, 1e-3):
            v64 = v.double().requires_grad_(True)
            ref = _tv_xz_torch(v64, eps)
            ref.backward()
            vc = v.cuda().requires_grad_(True)
            vals = [ft.tv_loss_xz(vc, eps=eps) for _ in range(3)]
            assert all(torch.equal(x, vals[0]) for x in vals)
            vals[0].backward()
            assert abs(vals[0].item() - ref.item()) <= 1e-6 * abs(ref.item()) + 1e-7
            gerr = (vc.grad.cpu().double() - v64.grad).abs().max()
            # fp32 inputs quantise 1e3-offset differences to ~6e-5
            tol = 1e-4 if data == "offset" else 1e-5
            assert gerr <= tol * v64.grad.abs().max(), (data, eps, gerr.item())


@needs_cuda
@pytest.mark.parametrize("shape", [(12, 14, 10), (2, 2, 2), (1, 6, 5), (6, 1, 5), (6, 5, 1)])
def test_tv_loss_xz_backward_ignores_dirty_output_memory(shape):
    """Like the 3-D kernel: the planar backward writes every voxel when
    Nz, Nx >= 2 (its output is not zero-filled), and a degenerate volume
    gets an exact zero gradient."""
    from fasttomo.autograd import _tv_loss_grad_raw

    torch.manual_seed(2)
    vol = torch.randn(*shape, device=DEVICE)
    g = torch.ones((), device=DEVICE)
    grads = []
    for fill in (float("nan"), 0.0):
        junk = torch.full(shape, fill, device=DEVICE)
        del junk
        grads.append(_tv_loss_grad_raw(vol, g, 1e-7, "tv_loss_xz"))
    assert torch.equal(grads[0], grads[1])
    assert torch.isfinite(grads[0]).all()
    if min(shape[1:]) < 2:
        assert torch.equal(grads[0], torch.zeros_like(vol))


@pytest.mark.parametrize("fn,shape", [
    ("tv_loss_3d", (1, 8, 8)), ("tv_loss_3d", (8, 1, 8)), ("tv_loss_3d", (8, 8, 1)),
    ("tv_loss_xz", (8, 1, 8)), ("tv_loss_xz", (8, 8, 1)), ("tv_loss_xz", (1, 1, 1))])
def test_tv_losses_are_zero_without_corners(backend, fn, shape):
    """A size-1 dimension leaves no difference sites: the loss is exactly 0
    and the gradient zero on both backends (the CPU path returned NaN, the
    mean of an empty tensor), even for non-finite volumes."""
    v = torch.randn(*shape, device=backend)
    v.view(-1)[0] = float("inf")
    v.requires_grad_(True)
    loss = getattr(ft, fn)(v)
    assert loss.shape == () and loss.item() == 0.0
    loss.backward()
    assert torch.equal(v.grad, torch.zeros_like(v))


def test_tv_loss_xz_single_slice_is_the_2d_tv(backend):
    """Ny = 1: tv_loss_xz is the 2-D TV of that slice (tv_loss_3d is 0)."""
    v = torch.rand(1, 9, 11, generator=torch.Generator().manual_seed(1))
    got = ft.tv_loss_xz(v.to(backend), eps=1e-6).item()
    assert got == pytest.approx(_tv_xz_torch(v.double(), 1e-6).item(), rel=1e-6)
    assert ft.tv_loss_3d(v.to(backend)).item() == 0.0


def _second_order_cases(device):
    g = torch.Generator().manual_seed(9)
    y = (100 + 3 * torch.randn(3, 24, 24, generator=g)).to(device)
    v3 = torch.rand(4, 5, 6, generator=g).to(device)
    return {
        "tilt_ssim_loss": (y + torch.randn(y.shape, generator=g).to(device),
                           lambda x: ft.tilt_ssim_loss(x, y)),
        "s3im_loss": (y + torch.randn(y.shape, generator=g).to(device),
                      lambda x: ft.s3im_loss(x, y, patch_pixels=256,
                                             generator=torch.Generator(device=device).manual_seed(1))),
        "tv_loss_3d": (v3, ft.tv_loss_3d),
        "tv_loss_xz": (v3, ft.tv_loss_xz),
    }


_FIRST_ORDER = [("cpu", "tilt_ssim_loss"), ("cpu", "s3im_loss")] + [
    pytest.param("cuda", fn, marks=needs_cuda)
    for fn in ("tilt_ssim_loss", "s3im_loss", "tv_loss_3d", "tv_loss_xz")]


@pytest.mark.parametrize("device,fn", _FIRST_ORDER)
def test_hand_written_backwards_refuse_second_order(device, fn):
    """The hand-written backwards (SSIM helpers, fused CUDA TV) are first
    order only. A Hessian-vector product with respect to the input alone
    (create_graph=True, no weight leaf) must raise, not silently treat the
    backward as a constant (``once_differentiable`` did not catch this
    case: the HVP came out 17-22 % wrong for the SSIM losses and without
    the TV-xz term), and so must a second derivative through a weight. The
    ordinary first-order gradient is unaffected."""
    x0, f = _second_order_cases(device)[fn]
    x = x0.clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match="first-order gradients only"):
        (g,) = torch.autograd.grad(f(x), x, create_graph=True)
        (g * torch.ones_like(g)).sum().backward()          # H @ 1
    w = torch.tensor(2.0, device=device, requires_grad=True)
    with pytest.raises(RuntimeError, match="first-order gradients only"):
        torch.autograd.grad(w * f(x), x, create_graph=True)
    (g,) = torch.autograd.grad(f(x), x)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


@pytest.mark.parametrize("fn", ["tv_loss_3d", "tv_loss_xz"])
def test_cpu_tv_losses_stay_twice_differentiable(fn):
    """The CPU TV expressions are plain autograd: their Hessian-vector
    product matches central finite differences of the gradient."""
    g = torch.Generator().manual_seed(4)
    v = torch.rand(3, 5, 6, generator=g)
    u = torch.randn(3, 5, 6, generator=g)
    f = getattr(ft, fn)
    x = v.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(f(x, eps=1e-2), x, create_graph=True)
    (hvp,) = torch.autograd.grad((grad * u).sum(), x)
    h = 1e-3

    def grad_at(z):
        z = z.clone().requires_grad_(True)
        return torch.autograd.grad(f(z, eps=1e-2), z)[0]

    fd = (grad_at(v + h * u) - grad_at(v - h * u)) / (2 * h)
    torch.testing.assert_close(hvp, fd, rtol=2e-2, atol=2e-4)


def test_tv_loss_xz_rejects_bad_input():
    with pytest.raises(TypeError):
        ft.tv_loss_xz(torch.zeros(2, 3, 3, dtype=torch.float64))
    with pytest.raises(ValueError):
        ft.tv_loss_xz(torch.zeros(3, 3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_logsum_sparsity_value_and_grad(backend, dtype):
    """mean(log(1 + |v| / eps)) and its gradient sign(v) / (eps + |v|) / N,
    checked against NumPy fp64; the gradient at v == 0 is 0."""
    eps = 0.05
    v = np.array([[-2.0, -0.3, 0.0], [1e-4, 0.7, 5.0]])
    t = torch.tensor(v, dtype=dtype, device=backend, requires_grad=True)
    loss = ft.logsum_sparsity(t, eps=eps)
    assert loss.shape == () and loss.dtype == dtype
    ref = np.log1p(np.abs(v) / eps).mean()
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    assert loss.item() == pytest.approx(ref, rel=tol)
    loss.backward()
    gref = np.sign(v) / (eps + np.abs(v)) / v.size
    torch.testing.assert_close(t.grad.cpu().double(), torch.from_numpy(gref), rtol=tol * 10, atol=tol)
    assert t.grad[0, 2].item() == 0.0

    # random volume: gradient also matches finite differences of the reference
    rng = np.random.default_rng(9)
    v = rng.standard_normal((2, 3, 4))
    t = torch.tensor(v, dtype=torch.float64, device=backend, requires_grad=True)
    ft.logsum_sparsity(t, eps=eps).backward()
    fd = _fd_grad(lambda a: float(np.log1p(np.abs(a) / eps).mean()), v)
    torch.testing.assert_close(t.grad.cpu(), torch.from_numpy(fd), rtol=1e-5, atol=1e-8)
