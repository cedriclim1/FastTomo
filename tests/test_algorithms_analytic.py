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
