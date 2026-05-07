"""Verify ParallelProjector / RotmatProjector match the functional API."""

from __future__ import annotations

import math

import pytest
import torch

import fasttomo as ft
from fasttomo.autograd import _euler_zyz_to_rotmat


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cached projector is CUDA-only"
)


@pytest.fixture
def workload():
    torch.manual_seed(0)
    Ny, Nz, Nx = 32, 40, 36
    n_angles = 11
    device = torch.device("cuda")
    vol = torch.randn(Ny, Nz, Nx, device=device, dtype=torch.float32)
    sino = torch.randn(n_angles, Ny, Nx, device=device, dtype=torch.float32)
    angles_deg = torch.linspace(-60.0, 60.0, n_angles, device=device, dtype=torch.float32)
    return vol, sino, angles_deg


def test_parallel_projector_forward_matches_functional(workload):
    vol, _, ang = workload
    proj = ft.ParallelProjector(vol.shape, ang, device=vol.device)
    sino_func = ft.forward_project(vol, ang)
    for _ in range(5):
        sino_cached = proj.forward(vol)
        torch.testing.assert_close(sino_cached, sino_func, rtol=0, atol=1e-5)


def test_parallel_projector_back_matches_functional(workload):
    _, sino, ang = workload
    Ny, n_angles, N_det = sino.shape[1], sino.shape[0], sino.shape[2]
    proj = ft.ParallelProjector((Ny, N_det, N_det), ang, device=sino.device)
    vol_func = ft.back_project(sino, ang)
    for _ in range(5):
        vol_cached = proj.back(sino)
        torch.testing.assert_close(vol_cached, vol_func, rtol=0, atol=1e-5)


def test_parallel_projector_grad_through_volume(workload):
    vol, sino, ang = workload
    target = sino.clone()
    proj = ft.ParallelProjector(vol.shape, ang, device=vol.device)

    v_func = vol.clone().detach().requires_grad_(True)
    pred_f = ft.forward_project(v_func, ang)
    loss_f = ((pred_f - target) ** 2).mean()
    loss_f.backward()

    v_cached = vol.clone().detach().requires_grad_(True)
    pred_c = proj.forward(v_cached)
    loss_c = ((pred_c - target) ** 2).mean()
    loss_c.backward()

    torch.testing.assert_close(v_cached.grad, v_func.grad, rtol=1e-4, atol=1e-5)


def test_rotmat_projector_forward_matches_functional(workload):
    vol, _, ang = workload
    n_angles = ang.numel()
    euler = torch.zeros(n_angles, 3, device=vol.device, dtype=torch.float32)
    euler[:, 1] = ang
    mats = _euler_zyz_to_rotmat(euler)

    sino_func = ft.forward_project_3d(vol, euler, N_det=vol.shape[-1])

    proj = ft.RotmatProjector(vol.shape, n_angles, device=vol.device)
    for _ in range(5):
        sino_cached = proj.forward(vol, mats)
        torch.testing.assert_close(sino_cached, sino_func, rtol=0, atol=1e-5)


def test_run_fbp_projector_matches_functional(workload):
    _, sino, ang = workload
    n_angles, Ny, N_det = sino.shape
    proj = ft.ParallelProjector((Ny, N_det, N_det), ang, device=sino.device)

    rec_func = ft.run_fbp(sino, ang, positivity=True, mask=True)
    for _ in range(5):
        rec_cached = ft.run_fbp(sino, ang, positivity=True, mask=True, projector=proj)
        torch.testing.assert_close(rec_cached, rec_func, rtol=0, atol=1e-5)


def test_rotmat_projector_back_matches_functional(workload):
    _, sino, ang = workload
    n_angles, Ny, N_det = sino.shape
    euler = torch.zeros(n_angles, 3, device=sino.device, dtype=torch.float32)
    euler[:, 1] = ang
    mats = _euler_zyz_to_rotmat(euler)

    vol_func = ft.back_project_3d(sino, euler)

    proj = ft.RotmatProjector((Ny, N_det, N_det), n_angles, device=sino.device)
    for _ in range(5):
        vol_cached = proj.back(sino, mats)
        torch.testing.assert_close(vol_cached, vol_func, rtol=0, atol=1e-5)


# ── FP16 path ────────────────────────────────────────────────────────────
#
# fp16 textures are 16-bit half-precision; hardware bilinear filtering
# operates on the half values directly, accumulators stay fp32. We expect
# ~1% relative agreement with fp32 — the same precision floor as the
# CPU-vs-CUDA fp32 test, just bumped slightly because both axes lose
# fractional bits.

_FP16_REL = 1e-2  # 1% relative tolerance (TorchRadon precedent)


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).abs().max().item()
    scale = expected.abs().max().item() + 1e-12
    return diff / scale


def test_parallel_projector_forward_fp16_matches_fp32(workload):
    vol, _, ang = workload
    sino_fp32 = ft.forward_project(vol, ang)
    proj = ft.ParallelProjector(vol.shape, ang, device=vol.device)
    sino_fp16 = proj.forward(vol.half())
    assert sino_fp16.dtype == torch.float16
    assert proj.dtype == torch.float16
    assert _rel_err(sino_fp16, sino_fp32) < _FP16_REL


def test_parallel_projector_back_fp16_matches_fp32(workload):
    _, sino, ang = workload
    Ny, N_det = sino.shape[1], sino.shape[2]
    vol_fp32 = ft.back_project(sino, ang)
    proj = ft.ParallelProjector((Ny, N_det, N_det), ang, device=sino.device)
    vol_fp16 = proj.back(sino.half())
    assert vol_fp16.dtype == torch.float16
    assert _rel_err(vol_fp16, vol_fp32) < _FP16_REL


def test_rotmat_projector_forward_fp16_matches_fp32(workload):
    vol, _, ang = workload
    n_angles = ang.numel()
    euler = torch.zeros(n_angles, 3, device=vol.device, dtype=torch.float32)
    euler[:, 1] = ang
    mats = _euler_zyz_to_rotmat(euler)
    sino_fp32 = ft.forward_project_3d(vol, euler, N_det=vol.shape[-1])
    proj = ft.RotmatProjector(vol.shape, n_angles, device=vol.device)
    sino_fp16 = proj.forward(vol.half(), mats)
    assert sino_fp16.dtype == torch.float16
    assert _rel_err(sino_fp16, sino_fp32) < _FP16_REL


def test_rotmat_projector_back_fp16_matches_fp32(workload):
    _, sino, ang = workload
    n_angles, Ny, N_det = sino.shape
    euler = torch.zeros(n_angles, 3, device=sino.device, dtype=torch.float32)
    euler[:, 1] = ang
    mats = _euler_zyz_to_rotmat(euler)
    vol_fp32 = ft.back_project_3d(sino, euler)
    proj = ft.RotmatProjector((Ny, N_det, N_det), n_angles, device=sino.device)
    vol_fp16 = proj.back(sino.half(), mats)
    assert vol_fp16.dtype == torch.float16
    assert _rel_err(vol_fp16, vol_fp32) < _FP16_REL


def test_projector_dtype_lock_in_raises(workload):
    """A projector locks its dtype on first FP/BP call. Calling with a
    mismatched dtype afterwards raises a clear TypeError."""
    vol, _, ang = workload
    proj = ft.ParallelProjector(vol.shape, ang, device=vol.device)
    _ = proj.forward(vol)              # locks fp32
    with pytest.raises(TypeError, match="dtype lock-in"):
        proj.forward(vol.half())
