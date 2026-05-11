"""Unit tests for fasttomo.regularizers — SSIM-based losses.

Coverage:
    * tilt_ssim_loss     — Wang-2004 per-tilt SSIM (validated against skimage).
    * s3im_loss          — Stochastic SSIM (Xie 2023).
"""

from __future__ import annotations

import math

import pytest
import torch

import fasttomo as ft


# ── tilt_ssim_loss ────────────────────────────────────────────────────────

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


# ── s3im_loss ─────────────────────────────────────────────────────────────

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
