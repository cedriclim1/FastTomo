"""
Regression tests for kernel and binding correctness fixes in ``csrc/``.

- D2: the rotation-matrix gradient (CPU and CUDA, cached and uncached) is the
  gradient of the forward projection, including the voxels within one voxel
  of the box faces. Checked against central finite differences of an
  independent fp64 NumPy port of the forward projector.

CUDA tests skip automatically when no GPU is visible; the CPU tests always run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import fasttomo as ft
from fasttomo.autograd import _euler_zyz_to_rotmat, _grad_rotmat_raw

HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA GPU available")


def _rel(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    a = torch.as_tensor(a, dtype=torch.float64).cpu()
    b = torch.as_tensor(b, dtype=torch.float64).cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# ── D2: rotation-matrix gradient vs finite differences ─────────────────────

def _fp_rotmat_ref(vol: np.ndarray, mats: np.ndarray, N_det: int) -> np.ndarray:
    """fp64 reference of the rotmat forward projector.

    Detector-driven ray march with unit steps, trilinear interpolation and
    border-zero semantics (a sample contributes while any of its 8 corners is
    inside the volume). Written from the geometry in ``proj_geom.h`` without
    sharing code with the kernels: corners outside the volume read the zero
    padding of ``pad``.
    """
    Ny, Nz, Nx = vol.shape
    ray_len = int(math.ceil(0.5 * math.sqrt(Nx * Nx + Ny * Ny + Nz * Nz))) + 1
    hd, hx, hy, hz = 0.5 * (N_det - 1), 0.5 * (Nx - 1), 0.5 * (Ny - 1), 0.5 * (Nz - 1)
    pad = np.zeros((Ny + 2, Nz + 2, Nx + 2))
    pad[1:-1, 1:-1, 1:-1] = vol
    V, U, S = np.meshgrid(np.arange(Ny) - hy, np.arange(N_det) - hd,
                          np.arange(-ray_len, ray_len + 1, dtype=np.float64),
                          indexing="ij")
    out = np.zeros((mats.shape[0], Ny, N_det))
    for a, M in enumerate(mats):
        fx = U * M[0] + V * M[3] + S * M[6] + hx
        fy = U * M[1] + V * M[4] + S * M[7] + hy
        fz = U * M[2] + V * M[5] + S * M[8] + hz
        # Clamp far-away samples onto the zero padding; the weights below
        # then multiply zeros only.
        fx = np.clip(fx, -1.0, Nx)
        fy = np.clip(fy, -1.0, Ny)
        fz = np.clip(fz, -1.0, Nz)
        x0, y0, z0 = np.floor(fx), np.floor(fy), np.floor(fz)
        wx, wy, wz = fx - x0, fy - y0, fz - z0
        xi = x0.astype(int) + 1
        yi = y0.astype(int) + 1
        zi = z0.astype(int) + 1
        xj = np.minimum(xi + 1, Nx + 1)
        yj = np.minimum(yi + 1, Ny + 1)
        zj = np.minimum(zi + 1, Nz + 1)
        val = ((1 - wy) * (1 - wz) * ((1 - wx) * pad[yi, zi, xi] + wx * pad[yi, zi, xj])
               + (1 - wy) * wz * ((1 - wx) * pad[yi, zj, xi] + wx * pad[yi, zj, xj])
               + wy * (1 - wz) * ((1 - wx) * pad[yj, zi, xi] + wx * pad[yj, zi, xj])
               + wy * wz * ((1 - wx) * pad[yj, zj, xi] + wx * pad[yj, zj, xj]))
        out[a] = val.sum(-1)
    return out


def _fd_grad_mats(vol: np.ndarray, mats: np.ndarray, w: np.ndarray,
                  eps: float = 1e-6) -> np.ndarray:
    """Central finite difference of L(M) = <w, FP(vol, M)> w.r.t. every M entry."""
    N_det = w.shape[-1]
    fd = np.zeros_like(mats)
    for a in range(mats.shape[0]):
        for k in range(9):
            mp = mats.copy()
            mm = mats.copy()
            mp[a, k] += eps
            mm[a, k] -= eps
            fd[a, k] = ((w * _fp_rotmat_ref(vol, mp, N_det)).sum()
                        - (w * _fp_rotmat_ref(vol, mm, N_det)).sum()) / (2 * eps)
    return fd


# Generic (non-special) Euler angles: with angles like 0 or 90 deg every
# sample lands exactly on a trilinear kink and the derivative is one-sided,
# so finite differences and the analytic gradient legitimately disagree.
_EULER = torch.tensor([[12.3, 27.1, -7.9], [41.7, -33.2, 16.4], [3.3, 61.9, -2.2]])


def _face_touching_case(shape, kind):
    g = torch.Generator().manual_seed(1234)
    Ny, Nz, Nx = shape
    if kind == "ones":
        vol = torch.ones(Ny, Nz, Nx)
    else:
        vol = torch.rand(Ny, Nz, Nx, generator=g) + 0.5   # non-zero on every face
    mats = _euler_zyz_to_rotmat(_EULER).reshape(-1, 9).float()
    w = torch.randn(mats.shape[0], Ny, Nx, generator=g)
    fd = _fd_grad_mats(vol.double().numpy(), mats.double().numpy(), w.double().numpy())
    return vol, mats, w, fd


_D2_SHAPES = [(8, 8, 8), (7, 9, 6), (1, 5, 5)]


@pytest.mark.parametrize("shape", _D2_SHAPES)
@pytest.mark.parametrize("kind", ["ones", "rand"])
def test_grad_rotmat_cpu_matches_finite_difference(shape, kind):
    vol, mats, w, fd = _face_touching_case(shape, kind)
    g = _grad_rotmat_raw(vol, w, mats)
    assert g.shape == (3, 9) and g.dtype == torch.float32
    assert _rel(g, fd) < 1e-4


@needs_cuda
@pytest.mark.parametrize("shape", _D2_SHAPES)
@pytest.mark.parametrize("kind", ["ones", "rand"])
def test_grad_rotmat_cuda_matches_finite_difference_and_cpu(shape, kind):
    vol, mats, w, fd = _face_touching_case(shape, kind)
    g_cpu = _grad_rotmat_raw(vol, w, mats)
    g_cuda = _grad_rotmat_raw(vol.cuda(), w.cuda(), mats.cuda())
    assert g_cuda.shape == (3, 9) and g_cuda.dtype == torch.float32
    assert _rel(g_cuda, fd) < 1e-4
    assert _rel(g_cuda, g_cpu) < 1e-4


@needs_cuda
@pytest.mark.parametrize("kind", ["ones", "rand"])
def test_grad_rotmat_cached_matches_finite_difference(kind):
    """The RotmatProjector path (grad_rotmat_cuda_cached) gets the same fix."""
    vol, mats, w, fd = _face_touching_case((7, 9, 6), kind)
    Ny, Nz, Nx = vol.shape
    proj = ft.RotmatProjector((Ny, Nz, Nx), n_angles=3, device="cuda")
    m = mats.cuda().requires_grad_(True)
    sino = proj.forward(vol.cuda(), m)
    (sino * w.cuda()).sum().backward()
    assert _rel(m.grad, fd) < 1e-4


@needs_cuda
def test_grad_rotmat_euler_cpu_cuda_agree_on_box_filling_volume():
    """End-to-end: d loss / d euler through forward_project_3d, CPU vs CUDA,
    on a volume that fills the whole box (the D2 failure scenario). The loss
    is linear in the sinogram so both backends see the same grad_sino (the
    CUDA forward's 8-bit texture filter weights would otherwise leak in)."""
    torch.manual_seed(0)
    vol = torch.rand(10, 10, 10) + 0.5
    target = torch.randn(3, 10, 10)
    grads = []
    for dev in ("cpu", "cuda"):
        e = _EULER.clone().to(dev).requires_grad_(True)
        sino = ft.forward_project_3d(vol.to(dev), e)
        (sino * target.to(dev)).sum().backward()
        grads.append(e.grad.cpu())
    assert _rel(grads[1], grads[0]) < 1e-4
