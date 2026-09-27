"""
Direct reconstructions: FBP (run_fbp) and weighted back-projection
(run_weighted_backprojection). W-BP operator details are in test_wsirt.py.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    make_angles,
    needs_cuda,
    phantom_2d,
    rel_err,
)

import fasttomo as ft

# ── FBP ───────────────────────────────────────────────────────────────────────


def test_fbp_shape(backend):
    """Replaces test_cpu.py::test_fbp_shape and test_cuda.py::test_fbp_shape_cuda."""
    N      = 64
    angles = make_angles(60, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fbp(sino, angles, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == backend.type


def _cylinder_fbp_level(backend, radius, tilt, N=128):
    """Mean FBP value over the inner half-radius of a unit-density cylinder
    (a disk in XZ, constant along Y) of ``radius`` voxels. ``tilt=90``:
    180 equally spaced views over [-90, 90); else ``linspace(-tilt, tilt)``
    at the same 1-degree spacing."""
    z = torch.arange(N) - (N - 1) / 2
    r2 = z[:, None] ** 2 + z[None, :] ** 2
    vol = (r2 <= radius ** 2).float().expand(2, N, N).contiguous().to(backend)
    if tilt == 90:
        ang = torch.arange(180) * 1.0 - 90
    else:
        ang = torch.linspace(-tilt, tilt, 2 * tilt + 1)
    ang = ang.to(backend)
    rec = ft.run_fbp(ft.forward_project(vol, ang), ang)
    return rec[:, r2 <= (radius / 2) ** 2].mean().item()


@pytest.mark.parametrize("radius", [20, 40])
def test_fbp_angular_scale_keeps_mean_level_at_limited_tilt(backend, radius):
    """Scope report 2g [S]: FBP scales by pi / n_angles, which looks like
    it assumes 180 degrees of coverage, so +-70 degrees was suspected to be
    off by 180/140. pi / n_angles is pi times the mean over the measured
    views: it keeps the mean level correct, which is what this pins. For a
    rotationally symmetric object every view carries the same ray integral,
    so the level at +-70 equals the one at 180 degrees; scaling by the
    actual spacing (140 degrees / n) would make it 0.78 of that and fail
    here. This does not make +-70 unbiased: anisotropic objects still show
    an orientation-dependent bias at limited tilt (the missing wedge; a 4:1
    ellipse reads 1.20 or 0.50 depending on its orientation, #95 review),
    which no angular scale removes."""
    full = _cylinder_fbp_level(backend, radius, 90)
    limited = _cylinder_fbp_level(backend, radius, 70)
    assert abs(limited / full - 1.0) < 5e-3, (limited, full)


@pytest.mark.parametrize("tilt", [90, 70])
@pytest.mark.parametrize("radius", [20, 40, 60])
def test_fbp_cylinder_absolute_value(backend, radius, tilt):
    """A unit-density cylinder reconstructs at 1 (inner half-radius mean)
    at 180 degrees and at +-70 degrees, from a small one to one filling the
    field of view. The unpadded |xi| ramp gave 0.96 / 0.83 / 0.57."""
    level = _cylinder_fbp_level(backend, radius, tilt)
    assert abs(level - 1.0) < 5e-3, level


def test_ramp_filter_is_the_linear_ram_lak_convolution():
    """ramp_filter == the direct (linear, unpadded-wraparound-free)
    convolution of each row with h[0] = 1/4, h[odd n] = -1/(pi n)^2."""
    import math

    g = torch.Generator().manual_seed(3)
    for N_det in (7, 16, 33):
        sino = torch.rand(2, 3, N_det, generator=g, dtype=torch.float64)
        n = torch.arange(-(N_det - 1), N_det, dtype=torch.float64)
        h = torch.where(n.abs() % 2 == 1, -1.0 / (math.pi * n) ** 2, torch.zeros_like(n))
        h[N_det - 1] = 0.25
        ref = torch.stack([torch.stack([
            torch.tensor([(row * h.flip(0)[N_det - 1 - u: 2 * N_det - 1 - u]).sum().item()
                          for u in range(N_det)], dtype=torch.float64)
            for row in view]) for view in sino])
        out = ft.ramp_filter(sino.float())
        assert out.shape == sino.shape and out.dtype == torch.float32
        torch.testing.assert_close(out.double(), ref, rtol=0, atol=1e-6)


def test_fbp_better_than_plain_backproject(backend):
    """FBP reconstruction MSE should beat raw (unfiltered) back-projection.

    Replaces test_cpu.py::test_fbp_better_than_plain_backproject and
    test_cuda.py::test_fbp_better_than_plain_backproject_cuda."""
    N      = 64
    angles = make_angles(90, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    def norm_mse(a, b):
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        b = (b - b.min()) / (b.max() - b.min() + 1e-9)
        return ((a - b) ** 2).mean().item()

    bp_mse  = norm_mse(ft.back_project(sino, angles, Nz=N), vol)
    fbp_mse = norm_mse(ft.run_fbp(sino, angles, Nz=N), vol)
    assert fbp_mse < bp_mse, (
        f"FBP MSE ({fbp_mse:.4f}) should be < raw BP MSE ({bp_mse:.4f})"
    )


def test_fbp_mask_uses_projector_output_shape():
    """mask=True must follow the projector output geometry.

    Cached-projector FBP can reconstruct with Nz != N_det. The mask must be
    built from the reconstructed volume shape, not from the input sinogram.
    """
    n_angles, Ny, N_det = 7, 2, 16
    Nz_proj = 20
    sino = torch.randn(n_angles, Ny, N_det)
    angles = make_angles(n_angles)

    class DummyProjector:
        def back(self, _sino: torch.Tensor) -> torch.Tensor:
            return torch.ones(Ny, Nz_proj, N_det, dtype=_sino.dtype, device=_sino.device)

    rec = ft.run_fbp(sino, angles, Nz=-1, mask=True, projector=DummyProjector())
    assert rec.shape == (Ny, Nz_proj, N_det)


DEVICE = torch.device("cuda")


@needs_cuda
def test_run_fbp_fp16_matches_fp32():
    """fp16 run_fbp recon should match fp32 to within ~1% relative on
    a structured phantom (FFT in fp16 is accepted, see run_fbp docstring)."""
    N      = 64
    angles = make_angles(60, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec_fp32 = ft.run_fbp(sino, angles, Nz=N, mask=True)
    rec_fp16 = ft.run_fbp(sino.half(), angles, Nz=N, mask=True)
    assert rec_fp16.dtype == torch.float16
    # Looser tolerance because fp16 FFT compounds the precision loss.
    assert rel_err(rec_fp16, rec_fp32) < 5e-2


# ── W-BP ──────────────────────────────────────────────────────────────────────


def test_wbp_shape(backend):
    """Replaces test_cpu.py::test_wbp_shape and test_cuda.py::test_wbp_shape_cuda."""
    N      = 32
    angles = make_angles(20, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_weighted_backprojection(sino, angles, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == backend.type


# ── support_mask ──────────────────────────────────────────────────────────────


def _box_support(Ny: int, N: int) -> torch.Tensor:
    """Binary [Ny, N, N] support: 1 inside the central half-box, 0 outside."""
    M = torch.zeros(Ny, N, N)
    q = N // 4
    M[:, q:3*q, q:3*q] = 1.0
    return M


def test_fbp_support_mask_equals_post_multiply():
    """FBP applies the support once at the end: == unmasked recon * mask."""
    N      = 32
    angles = make_angles(20)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)

    M = _box_support(1, N)
    rec_masked = ft.run_fbp(sino, angles, Nz=N, support_mask=M)
    rec_plain  = ft.run_fbp(sino, angles, Nz=N)
    assert torch.equal(rec_masked, rec_plain * M)


def test_fbp_support_mask_reproduces_cylinder():
    """Passing the cylinder as a support tensor reproduces mask=True exactly."""
    N      = 32
    angles = make_angles(20)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)

    M = ft.cylindrical_mask(N, N).unsqueeze(0)
    rec_support  = ft.run_fbp(sino, angles, Nz=N, support_mask=M)
    rec_cylinder = ft.run_fbp(sino, angles, Nz=N, mask=True)
    assert torch.equal(rec_support, rec_cylinder)


def test_fbp_support_mask_with_mask_true_raises():
    N      = 32
    angles = make_angles(10)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ft.run_fbp(sino, angles, Nz=N, mask=True,
                   support_mask=_box_support(1, N))


def test_fbp_support_mask_bad_shape_raises():
    N      = 32
    angles = make_angles(10)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)
    with pytest.raises(ValueError, match="support_mask"):
        ft.run_fbp(sino, angles, Nz=N, support_mask=torch.ones(N + 1, N))
    with pytest.raises(ValueError, match="support_mask"):
        ft.run_fbp(sino, angles, Nz=N, support_mask=torch.ones(3, N, N))


def _half_support(N: int, left: bool) -> torch.Tensor:
    """Binary [1, N, N] support covering the left or right half in x."""
    M = torch.zeros(1, N, N, device=DEVICE)
    if left:
        M[:, :, : N // 2] = 1.0
    else:
        M[:, :, N // 2:] = 1.0
    return M


@needs_cuda
def test_fbp_support_mask_cuda():
    """support_mask on CUDA: equals post-multiply; bool tensors accepted."""
    N      = 32
    angles = make_angles(20, device=DEVICE)
    sino   = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles)

    M = _half_support(N, left=True)
    rec_masked = ft.run_fbp(sino, angles, Nz=N, support_mask=M)
    rec_plain  = ft.run_fbp(sino, angles, Nz=N)
    assert torch.equal(rec_masked, rec_plain * M)

    rec_bool = ft.run_fbp(sino, angles, Nz=N, support_mask=M.bool())
    assert torch.equal(rec_bool, rec_masked)


@needs_cuda
def test_fbp_support_mask_fp16_volume_fp32_mask_cuda():
    """fp16 sinogram + fp32 mask: mask is cast, dtype preserved end-to-end."""
    N      = 32
    angles = make_angles(20, device=DEVICE)
    sino   = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles).half()

    M   = _half_support(N, left=True)            # float32
    rec = ft.run_fbp(sino, angles, Nz=N, support_mask=M)
    assert rec.dtype == torch.float16
    assert rec[M.expand_as(rec) == 0].abs().max().item() == 0.0
