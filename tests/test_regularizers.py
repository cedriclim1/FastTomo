"""
Tests for fasttomo.regularizers.

Covers Log-Sum L0-proxy sparsity (CPU + CUDA) and verifies the TV-loss
public symbol is reachable from the new module location.
"""

import math
import pytest
import torch

import fasttomo as ft
from fasttomo.regularizers import (
    _detect_atom_peaks_3d,
    _gaussian_blur_3d,
    _gaussian_kernel_3d,
    atom_red_loss,
    gaussian_atom_project,
    logsum_sparsity,
    tilt_ssim_loss,
    tv_loss_3d,
)


_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")


# ── Log-Sum L0 sparsity ───────────────────────────────────────────────────

@pytest.mark.parametrize("device", _DEVICES)
def test_logsum_value_matches_formula(device):
    """logsum_sparsity(x, eps) = mean(log1p(|x|/eps))."""
    torch.manual_seed(0)
    x = torch.randn(7, 5, 9, device=device) * 3.0
    eps = 0.05

    got = ft.logsum_sparsity(x, eps=eps)
    expected = torch.log1p(x.abs() / eps).mean()

    assert got.shape == torch.Size([])
    assert got.device.type == device
    assert torch.allclose(got, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("device", _DEVICES)
def test_logsum_gradient_matches_analytical(device):
    """∂/∂ρ mean(log1p(|ρ|/ε)) = sign(ρ) / (ε + |ρ|) / N."""
    torch.manual_seed(1)
    x = torch.randn(6, 4, 5, device=device, requires_grad=True)
    # Avoid exactly-zero entries where sign() is undefined.
    with torch.no_grad():
        x.add_(0.05 * x.sign())
    eps = 0.1

    loss = ft.logsum_sparsity(x, eps=eps)
    loss.backward()

    n = float(x.numel())
    expected_grad = x.detach().sign() / (eps + x.detach().abs()) / n

    assert x.grad is not None
    diff = (x.grad - expected_grad).abs().max()
    scale = expected_grad.abs().max().clamp_min(1e-12)
    assert (diff / scale).item() < 1e-5, (diff.item(), scale.item())


@pytest.mark.parametrize("device", _DEVICES)
def test_logsum_chained_with_data_loss(device):
    """data_loss + λ·logsum produces gradients on a leaf parameter."""
    torch.manual_seed(2)
    target = torch.randn(8, 8, 8, device=device)
    vol = torch.randn(8, 8, 8, device=device, requires_grad=True)
    lam = 0.3

    loss = ((vol - target) ** 2).mean() + lam * ft.logsum_sparsity(vol, eps=1e-2)
    loss.backward()

    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert torch.isfinite(vol.grad).all()


@pytest.mark.parametrize("device", _DEVICES)
def test_logsum_shrinks_more_for_small_values(device):
    """Per-voxel gradient magnitude is monotonically decreasing in |ρ| —
    this is the Log-Sum's defining property: small voxels feel a bigger
    push toward zero than large ones."""
    eps = 0.1
    # Two leaves with the same shape so .mean() reduction is identical.
    small = torch.full((4, 4, 4), 1e-3, device=device, requires_grad=True)
    large = torch.full((4, 4, 4), 1.0,  device=device, requires_grad=True)

    ft.logsum_sparsity(small, eps=eps).backward()
    ft.logsum_sparsity(large, eps=eps).backward()

    g_small = small.grad.abs().mean().item()
    g_large = large.grad.abs().mean().item()
    assert g_small > g_large, (g_small, g_large)


def test_logsum_zero_input_is_zero():
    """log1p(0) = 0; the loss must vanish for an all-zero volume."""
    x = torch.zeros(4, 4, 4)
    loss = ft.logsum_sparsity(x, eps=1e-2)
    assert loss.item() == 0.0


def test_logsum_eps_scaling():
    """Halving eps roughly doubles the penalty for |x| ≪ eps (Taylor: |x|/eps)."""
    x = torch.full((10, 10, 10), 1e-4)
    a = ft.logsum_sparsity(x, eps=1e-1).item()
    b = ft.logsum_sparsity(x, eps=5e-2).item()
    assert b > a
    # Within Taylor regime |x|/eps ≪ 1, log1p(t) ≈ t so b/a ≈ 2.
    assert math.isclose(b / a, 2.0, rel_tol=1e-3)


# ── TV-loss reachable from new module location ───────────────────────────

def test_tv_loss_3d_importable_from_regularizers_module():
    """The move from utilities.py to regularizers.py preserved the public symbol."""
    assert tv_loss_3d is ft.tv_loss_3d
    x = torch.randn(6, 6, 6)
    loss = tv_loss_3d(x)
    assert loss.shape == torch.Size([])


# ── Per-tilt SSIM ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("device", _DEVICES)
def test_tilt_ssim_zero_for_identity(device):
    """SSIM(x, x) = 1, so the loss (1 - SSIM) is essentially 0."""
    torch.manual_seed(0)
    x = torch.rand(4, 32, 48, device=device)
    loss = ft.tilt_ssim_loss(x, x, data_range=1.0)
    assert loss.shape == torch.Size([])
    assert loss.device.type == device
    assert loss.abs().item() < 1e-5, loss.item()


@pytest.mark.parametrize("device", _DEVICES)
def test_tilt_ssim_in_valid_range(device):
    """SSIM ∈ [-1, 1] ⇒ (1 - SSIM) ∈ [0, 2]."""
    torch.manual_seed(1)
    a = torch.rand(3, 24, 24, device=device)
    b = torch.rand(3, 24, 24, device=device)
    loss = ft.tilt_ssim_loss(a, b, data_range=1.0)
    assert 0.0 <= loss.item() <= 2.0


@pytest.mark.parametrize("device", _DEVICES)
def test_tilt_ssim_increases_with_blur(device):
    """A blurred copy is structurally different ⇒ loss > identity."""
    import torch.nn.functional as F
    torch.manual_seed(2)
    x = torch.rand(2, 32, 32, device=device)
    win = torch.ones(1, 1, 5, 5, device=device) / 25.0
    blurred = F.conv2d(x.unsqueeze(1), win, padding=2).squeeze(1)
    l_id = ft.tilt_ssim_loss(x, x,       data_range=1.0).item()
    l_bl = ft.tilt_ssim_loss(x, blurred, data_range=1.0).item()
    assert l_bl > l_id + 1e-3, (l_id, l_bl)


@pytest.mark.parametrize("device", _DEVICES)
def test_tilt_ssim_grad_flows(device):
    """Gradient must flow back into pred with finite values of the right shape."""
    torch.manual_seed(3)
    target = torch.rand(2, 16, 16, device=device)
    pred = torch.rand(2, 16, 16, device=device, requires_grad=True)
    loss = ft.tilt_ssim_loss(pred, target, data_range=1.0)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().max() > 0


def test_tilt_ssim_matches_skimage():
    """Hand-rolled SSIM agrees with skimage.metrics.structural_similarity."""
    skimage_metrics = pytest.importorskip("skimage.metrics")
    torch.manual_seed(4)
    np_rng = torch.Generator().manual_seed(4)
    a = torch.rand(3, 40, 40, generator=np_rng)
    b = a + 0.1 * torch.rand(3, 40, 40, generator=np_rng)

    ours = ft.tilt_ssim_loss(a, b, win_size=11, sigma=1.5,
                             data_range=1.0).item()

    a_np = a.numpy()
    b_np = b.numpy()
    ssim_vals = [
        skimage_metrics.structural_similarity(
            a_np[i], b_np[i],
            win_size=11, gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False, data_range=1.0,
        )
        for i in range(a_np.shape[0])
    ]
    skimage_loss = 1.0 - sum(ssim_vals) / len(ssim_vals)
    assert abs(ours - skimage_loss) < 1e-3, (ours, skimage_loss)


def test_tilt_ssim_importable_from_fasttomo():
    assert tilt_ssim_loss is ft.tilt_ssim_loss


# ── Atomic-Prototype RED loss + Gaussian-atom projection ─────────────────

def _planted_atom_volume(positions, amplitudes, sigma, vol_shape, device):
    """Render a clean sum-of-Gaussians on a fresh volume — used by tests
    that need a known atom configuration."""
    Ny, Nz, Nx = vol_shape
    coords_y = torch.arange(Ny, device=device, dtype=torch.float32)
    coords_z = torch.arange(Nz, device=device, dtype=torch.float32)
    coords_x = torch.arange(Nx, device=device, dtype=torch.float32)
    gy, gz, gx = torch.meshgrid(coords_y, coords_z, coords_x, indexing="ij")
    vol = torch.zeros(vol_shape, device=device, dtype=torch.float32)
    for (py, pz, px), A in zip(positions, amplitudes):
        d2 = (gy - py) ** 2 + (gz - pz) ** 2 + (gx - px) ** 2
        vol = vol + A * torch.exp(-d2 / (2.0 * sigma * sigma))
    return vol


@pytest.mark.parametrize("device", _DEVICES)
def test_gaussian_blur_3d_preserves_constant(device):
    """A sum-to-1 blur of a constant volume returns the same constant."""
    vol = torch.full((12, 14, 16), 0.7, device=device)
    blurred = _gaussian_blur_3d(vol, sigma=1.5)
    assert torch.allclose(blurred, vol, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_gaussian_kernel_3d_centre_is_max(device):
    """The kernel's max is at the centre voxel."""
    k = _gaussian_kernel_3d(sigma=1.0, device=torch.device(device))
    K = k.shape[-1]
    centre = K // 2
    flat_max = k.flatten().max().item()
    assert math.isclose(k[0, 0, centre, centre, centre].item(), flat_max,
                        rel_tol=0, abs_tol=1e-7)


@pytest.mark.parametrize("device", _DEVICES)
def test_detect_atom_peaks_isolates_local_maxima(device):
    """Plant 4 well-separated atoms; detector recovers exactly 4 peaks at
    the right voxel locations."""
    vol_shape = (24, 24, 24)
    positions = [(6, 6, 6), (6, 18, 18), (18, 6, 18), (18, 18, 6)]
    amplitudes = [1.0, 0.8, 0.6, 0.5]
    vol = _planted_atom_volume(positions, amplitudes, sigma=1.0,
                               vol_shape=vol_shape, device=device)
    blurred = _gaussian_blur_3d(vol, sigma=1.0)
    peaks = _detect_atom_peaks_3d(blurred, min_distance=2,
                                  threshold=0.05 * float(vol.max()))
    coords = torch.nonzero(peaks).cpu().tolist()
    assert len(coords) == len(positions), coords
    assert sorted(map(tuple, coords)) == sorted(positions)


@pytest.mark.parametrize("device", _DEVICES)
def test_atom_red_loss_low_for_clean_atom_volume(device):
    """A volume that already IS a sum-of-Gaussians gets a small RED penalty
    (not exactly zero — voxels near the atom but outside the dilation
    contribute residual). Penalty must be small relative to a noisy
    volume of comparable energy."""
    vol_shape = (24, 24, 24)
    positions = [(6, 6, 6), (6, 18, 18), (18, 6, 18), (18, 18, 6)]
    amplitudes = [1.0, 0.8, 0.6, 0.5]
    clean = _planted_atom_volume(positions, amplitudes, sigma=1.0,
                                 vol_shape=vol_shape, device=device)

    torch.manual_seed(0)
    noisy = clean + 0.1 * torch.rand(vol_shape, device=device)

    loss_clean = atom_red_loss(clean, sigma_atom=1.0, min_distance=2,
                               dilate_radius=4, threshold_rel=0.05)
    loss_noisy = atom_red_loss(noisy, sigma_atom=1.0, min_distance=2,
                               dilate_radius=4, threshold_rel=0.05)
    assert loss_clean.item() < loss_noisy.item(), (
        loss_clean.item(), loss_noisy.item())


@pytest.mark.parametrize("device", _DEVICES)
def test_atom_red_loss_grad_flows(device):
    torch.manual_seed(1)
    vol = torch.rand(16, 16, 16, device=device, requires_grad=True)
    loss = atom_red_loss(vol, sigma_atom=1.0, min_distance=2,
                         dilate_radius=3, threshold_rel=0.05)
    loss.backward()
    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert torch.isfinite(vol.grad).all()


@pytest.mark.parametrize("device", _DEVICES)
def test_atom_red_loss_zero_volume(device):
    """An all-zero volume produces zero loss and zero gradient (but the
    loss must remain a tensor that backward() can be called on)."""
    vol = torch.zeros(8, 8, 8, device=device, requires_grad=True)
    loss = atom_red_loss(vol)
    assert loss.item() == 0.0
    loss.backward()
    assert vol.grad is not None
    assert torch.allclose(vol.grad, torch.zeros_like(vol))


@pytest.mark.parametrize("device", _DEVICES)
def test_gaussian_atom_project_renders_at_peaks(device):
    """The projection of a planted-atom volume puts its mass at the
    same peak coordinates."""
    vol_shape = (24, 24, 24)
    positions = [(6, 6, 6), (18, 18, 18)]
    amplitudes = [1.0, 0.6]
    vol = _planted_atom_volume(positions, amplitudes, sigma=1.0,
                               vol_shape=vol_shape, device=device)
    snapped = gaussian_atom_project(vol, sigma_atom=1.0, min_distance=2,
                                    threshold_rel=0.05)
    # Re-detect on the snapped volume — should find the same peaks.
    blurred = _gaussian_blur_3d(snapped, sigma=1.0)
    peaks = _detect_atom_peaks_3d(blurred, min_distance=2,
                                  threshold=0.05 * float(snapped.max()))
    coords = sorted(map(tuple, torch.nonzero(peaks).cpu().tolist()))
    assert coords == sorted(positions), coords


@pytest.mark.parametrize("device", _DEVICES)
def test_gaussian_atom_project_idempotent(device):
    """Applying the projection twice gives ~the same volume as one pass
    (the second snap re-detects the same peaks at the same amplitudes)."""
    vol_shape = (20, 20, 20)
    positions = [(5, 5, 5), (5, 15, 15), (15, 5, 15), (15, 15, 5)]
    amplitudes = [1.0, 0.7, 0.6, 0.5]
    vol = _planted_atom_volume(positions, amplitudes, sigma=1.0,
                               vol_shape=vol_shape, device=device)
    once = gaussian_atom_project(vol, sigma_atom=1.0, min_distance=2,
                                 threshold_rel=0.05)
    twice = gaussian_atom_project(once, sigma_atom=1.0, min_distance=2,
                                  threshold_rel=0.05)
    diff = (twice - once).abs().max().item()
    assert diff < 1e-5, diff


@pytest.mark.parametrize("device", _DEVICES)
def test_gaussian_atom_project_no_grad_on_output(device):
    vol = torch.rand(12, 12, 12, device=device, requires_grad=True)
    out = gaussian_atom_project(vol, sigma_atom=1.0, min_distance=2,
                                threshold_rel=0.05)
    assert out.requires_grad is False


@pytest.mark.parametrize("device", _DEVICES)
def test_gaussian_atom_project_zero_volume(device):
    """No peaks → zero output (early return), same shape and device."""
    vol = torch.zeros(8, 8, 8, device=device)
    out = gaussian_atom_project(vol)
    assert out.shape == vol.shape
    assert out.device.type == device
    assert out.abs().max().item() == 0.0


def test_atom_regularizers_importable_from_fasttomo():
    assert atom_red_loss is ft.atom_red_loss
    assert gaussian_atom_project is ft.gaussian_atom_project
