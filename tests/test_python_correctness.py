"""
Python-side correctness tests (PHASE1 section 4).

Each block names the item it covers. Tests are parametrised over the
``backend`` fixture: the CPU case always runs, the CUDA case skips when no
GPU is visible. Random inputs use fixed seeds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import fasttomo as ft

HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA GPU available")

BACKENDS = ["cpu", pytest.param("cuda", marks=needs_cuda)]


@pytest.fixture(params=BACKENDS)
def backend(request) -> torch.device:
    return torch.device(request.param)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double().cpu()
    b = b.detach().double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _blob(Ny: int, Nz: int, Nx: int, device, c=(0.1, -0.2), s=4.0) -> torch.Tensor:
    """Smooth Gaussian blob, constant along Y. Smooth inputs keep the CUDA
    texture back-projection close to the exact adjoint (~1 % instead of
    15-180 % for white noise), so the inner-product test is meaningful."""
    z = torch.arange(Nz, device=device) - (Nz - 1) / 2
    x = torch.arange(Nx, device=device) - (Nx - 1) / 2
    g = torch.exp(-((z[:, None] - c[0] * Nz) ** 2 + (x[None, :] - c[1] * Nx) ** 2) / (2 * s * s))
    return g.expand(Ny, Nz, Nx).contiguous()


# ── back_project(Nx=...) and autograd with N_det != Nx ─────────────────────

NX_NDET = [(24, 32), (32, 24), (24, 24)]


@pytest.mark.parametrize("Nx,N_det", NX_NDET)
def test_back_project_nx_shape_and_default(backend, Nx, N_det):
    """back_project(Nx=...) returns [Ny, Nz, Nx]; the default is still N_det."""
    ang = torch.linspace(-60, 60, 17, device=backend)
    sino = torch.randn(17, 3, N_det, generator=torch.Generator().manual_seed(1)).to(backend)
    vol = ft.back_project(sino, ang, Nz=20, Nx=Nx)
    assert vol.shape == (3, 20, Nx) and vol.dtype == torch.float32
    assert ft.back_project(sino, ang, Nz=20).shape == (3, 20, N_det)
    assert torch.equal(ft.back_project(sino, ang, Nz=20),
                       ft.back_project(sino, ang, Nz=20, Nx=N_det))


@pytest.mark.parametrize("Nx,N_det", NX_NDET)
def test_forward_back_adjoint_nx_ne_ndet(backend, Nx, N_det):
    """<A x, y> == <x, A^T y> with A = forward_project(., N_det) on a
    [Ny, Nz, Nx] volume and A^T = back_project(., Nx=Nx).

    CPU: the scatter BP is the exact transpose, so random inputs agree to
    fp32 round-off. CUDA: the texture-gather BP is an approximate adjoint;
    with smooth inputs it is within ~1.1 % for every geometry, including the
    square one, so a 3 % bound catches any mis-centring of Nx vs N_det.
    """
    Ny, Nz, n = 2, 20, 17
    ang = torch.linspace(-60, 60, n, device=backend)
    if backend.type == "cpu":
        g = torch.Generator().manual_seed(0)
        x = torch.randn(Ny, Nz, Nx, generator=g)
        y = torch.randn(n, Ny, N_det, generator=g)
        tol = 1e-5
    else:
        x = _blob(Ny, Nz, Nx, backend)
        y = ft.forward_project(_blob(Ny, Nz, Nx, backend, c=(-0.15, 0.1), s=3.0), ang, N_det=N_det)
        tol = 3e-2
    lhs = (ft.forward_project(x, ang, N_det=N_det) * y).double().sum()
    rhs = (x * ft.back_project(y, ang, Nz=Nz, Nx=Nx)).double().sum()
    assert float(abs(lhs - rhs) / abs(lhs)) < tol


@pytest.mark.parametrize("Nx,N_det", [(24, 32), (32, 24)])
def test_forward_project_autograd_nx_ne_ndet(backend, Nx, N_det):
    """Backward of forward_project with N_det != Nx used to raise
    ('_FPParallelBackward returned an invalid gradient'). Now the volume
    gradient is back_project(grad, Nx=Nx) and the angle gradient exists."""
    Ny, Nz, n = 3, 20, 17
    g = torch.Generator().manual_seed(2)
    x = torch.randn(Ny, Nz, Nx, generator=g).to(backend).requires_grad_(True)
    w = torch.randn(n, Ny, N_det, generator=g).to(backend)
    ang = torch.linspace(-60, 60, n, device=backend).requires_grad_(True)
    (ft.forward_project(x, ang, N_det=N_det) * w).sum().backward()
    assert x.grad.shape == x.shape and ang.grad.shape == ang.shape
    assert torch.isfinite(ang.grad).all()
    ref = ft.back_project(w, ang.detach(), Nz=Nz, Nx=Nx)
    assert torch.equal(x.grad, ref)

    # and back_project's own backward is forward_project(., N_det)
    s = w.clone().requires_grad_(True)
    (ft.back_project(s, ang.detach(), Nz=Nz, Nx=Nx) * x.detach()).sum().backward()
    assert torch.equal(s.grad, ft.forward_project(x.detach(), ang.detach(), N_det=N_det))


def test_forward_project_angle_grad_fd_nx_ne_ndet():
    """CPU angle gradient with N_det > Nx matches central finite differences
    of the loss (the forward kernel and grad_angle share one operator)."""
    Ny, Nz, Nx, N_det, n = 1, 24, 24, 34, 5
    x = _blob(Ny, Nz, Nx, "cpu", s=3.0) + 0.5 * _blob(Ny, Nz, Nx, "cpu", c=(-0.2, 0.2), s=2.0)
    w = torch.linspace(0.5, 1.5, N_det).expand(n, Ny, N_det).contiguous()
    ang = torch.tensor([-50.0, -20.0, 5.0, 30.0, 55.0], requires_grad=True)
    (ft.forward_project(x, ang, N_det=N_det) * w).sum().backward()
    h = 1e-2
    fd = []
    for i in range(n):
        a_p = ang.detach().clone(); a_p[i] += h
        a_m = ang.detach().clone(); a_m[i] -= h
        lp = (ft.forward_project(x, a_p, N_det=N_det) * w).double().sum()
        lm = (ft.forward_project(x, a_m, N_det=N_det) * w).double().sum()
        fd.append(float((lp - lm) / (2 * h)))
    fd = torch.tensor(fd)
    assert _rel(ang.grad, fd) < 2e-2, (ang.grad, fd)


@pytest.mark.parametrize("Nx,N_det", [(24, 32), (32, 24)])
def test_back_project_nx_matches_centre_crop(backend, Nx, N_det):
    """Geometry check independent of the adjoint: the gather BP evaluates
    each voxel at its physical position, so BP onto a narrower volume is the
    centre crop of BP onto a wider one (even width difference). Exact on
    CUDA; on CPU (ray-driven scatter) the outermost voxel ring differs
    because the wider volume's bilinear stencils straddle the crop edge."""
    Ny, Nz, n = 2, 20, 17
    ang = torch.linspace(-60, 60, n, device=backend)
    y = torch.randn(n, Ny, N_det, generator=torch.Generator().manual_seed(3)).to(backend)
    narrow, wide = min(Nx, N_det), max(Nx, N_det)
    off = (wide - narrow) // 2
    small = ft.back_project(y, ang, Nz=Nz, Nx=narrow)
    big = ft.back_project(y, ang, Nz=Nz, Nx=wide)[:, :, off:off + narrow]
    if backend.type == "cuda":
        assert torch.equal(small, big)
    else:
        torch.testing.assert_close(small[:, 1:-1, 1:-1], big[:, 1:-1, 1:-1], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("Nx,N_det", [(24, 32), (32, 24)])
def test_forward_project_3d_autograd_nx_ne_ndet(backend, Nx, N_det):
    """Same fix for the rotation-matrix pair: forward_project_3d with
    N_det != Nx has a working backward, equal to back_project_3d(Nx=Nx), and
    back_project_3d(Nx=...) is the centre crop of the wider BP."""
    Ny, Nz, n = 3, 20, 9
    g = torch.Generator().manual_seed(4)
    x = torch.randn(Ny, Nz, Nx, generator=g).to(backend).requires_grad_(True)
    w = torch.randn(n, Ny, N_det, generator=g).to(backend)
    eul = torch.zeros(n, 3, device=backend)
    eul[:, 1] = torch.linspace(-60, 60, n, device=backend)
    eul.requires_grad_(True)
    (ft.forward_project_3d(x, eul, N_det=N_det) * w).sum().backward()
    assert x.grad.shape == x.shape and eul.grad.shape == eul.shape
    ref = ft.back_project_3d(w, eul.detach(), Nz=Nz, Nx=Nx)
    assert torch.equal(x.grad, ref)
    s = w.clone().requires_grad_(True)
    (ft.back_project_3d(s, eul.detach(), Nz=Nz, Nx=Nx) * x.detach()).sum().backward()
    assert torch.equal(s.grad, ft.forward_project_3d(x.detach(), eul.detach(), N_det=N_det))
    narrow, wide = min(Nx, N_det), max(Nx, N_det)
    off = (wide - narrow) // 2
    small = ft.back_project_3d(w, eul.detach(), Nz=Nz, Nx=narrow)
    big = ft.back_project_3d(w, eul.detach(), Nz=Nz, Nx=wide)[:, :, off:off + narrow]
    torch.testing.assert_close(small, big, rtol=0, atol=0)
