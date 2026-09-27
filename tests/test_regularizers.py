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
        """Cross-check against skimage's structural_similarity.

        Tolerance is loose (~1e-2) because skimage crops the SSIM map to the
        valid region while we reflect-pad to full size — same SSIM math, just
        different boundary handling.
        """
        skimage = pytest.importorskip("skimage.metrics")
        torch.manual_seed(42)
        x_np = torch.rand(32, 32).numpy()
        y_np = (torch.rand(32, 32) * 0.8 + 0.1).numpy()
        sk = skimage.structural_similarity(
            x_np, y_np, data_range=1.0, win_size=11,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        )
        loss = ft.tilt_ssim_loss(
            torch.from_numpy(x_np)[None], torch.from_numpy(y_np)[None],
            win_size=11, sigma=1.5, data_range=1.0,
        ).item()
        assert (1.0 - loss) == pytest.approx(sk, abs=1e-2)

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
