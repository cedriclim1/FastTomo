"""
Weighted SIRT / exact weighted back-projection: CPU/CUDA parity and
convergence.

Root causes fixed (see run_wsirt):
- the exact-filter deconvolution was a circular FFT on the unpadded
  [Nz, Nx] grid, so the back-projected lines of the layergram wrapped
  around; W-BP was then not close to an inverse of the projector and
  wSIRT's data residual rose again after ~4 iterations at omega=1 (CUDA);
- the CPU back-projection is ray-driven (the exact transpose of the CPU
  ray-marching forward projector), which is not shift-invariant: a constant
  sinogram back-projects to a +-34 % voxel pattern at 45 degrees. The
  exact filter, computed from the back-projection of one point, amplifies
  that pattern up to ~40x, so CPU W-BP had 83 % error vs 30 % on CUDA. The
  W-BP step now uses the voxel-driven (gather) back-projection on both
  backends, the model the CUDA kernel already implements.
"""

from __future__ import annotations

import pytest
import torch

import fasttomo as ft

HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA GPU available")
BACKENDS = ["cpu", pytest.param("cuda", marks=needs_cuda)]


def _rel(a, b) -> float:
    a = a.detach().double().cpu()
    b = b.detach().double().cpu()
    return float((a - b).norm() / b.norm())


def _phantom(N: int, device, Ny: int = 2) -> torch.Tensor:
    vol = torch.zeros(Ny, N, N, device=device)
    vol[:, N // 4:3 * N // 4, 3 * N // 10:5 * N // 8] = 1.0
    vol[Ny - 1, 5 * N // 12:7 * N // 12, 5 * N // 12:N // 2 + 2] = 2.0
    return vol


GEOMS = [(48, 41, 70.0), (49, 41, 70.0), (40, 61, 60.0)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("N,n,tilt", GEOMS)
def test_wsirt_residual_monotone_at_default_omega(backend, N, n, tilt):
    """At the default omega=1 the data residual never rises over 20
    iterations (before: it rose from iteration 4 on CUDA and grew ~15x by
    iteration 40), and the volume error after 20 iterations is at the
    limited-angle floor."""
    ang = torch.linspace(-tilt, tilt, n, device=backend)
    vol = _phantom(N, backend)
    sino = ft.forward_project(vol, ang)
    rec, loss = ft.run_wsirt(sino, ang, n_iter=20, Nz=N, return_loss=True)
    assert (loss[1:] <= loss[:-1]).all(), loss
    assert loss[-1] < loss[0] / 100
    assert _rel(rec, vol) < 0.35


@needs_cuda
@pytest.mark.parametrize("N,n,tilt", GEOMS)
def test_wsirt_cpu_cuda_parity(N, n, tilt):
    """Same input, 20 iterations: CPU and CUDA agree to 1e-4 (before: 34 %
    apart; CPU 83 % error after one iteration vs CUDA 30 %)."""
    ang = torch.linspace(-tilt, tilt, n)
    sino = ft.forward_project(_phantom(N, "cpu"), ang)
    cpu = ft.run_wsirt(sino, ang, n_iter=20, Nz=N)
    gpu = ft.run_wsirt(sino.cuda(), ang.cuda(), n_iter=20, Nz=N)
    assert _rel(gpu, cpu) < 1e-4


@needs_cuda
@pytest.mark.parametrize("N,n,tilt", GEOMS)
def test_weighted_backprojection_cpu_cuda_parity(N, n, tilt):
    """run_weighted_backprojection shares the W-BP operator: CPU and CUDA
    agree to 1e-4 and both recover the phantom to the one-step level."""
    ang = torch.linspace(-tilt, tilt, n)
    vol = _phantom(N, "cpu")
    sino = ft.forward_project(vol, ang)
    cpu = ft.run_weighted_backprojection(sino, ang, Nz=N)
    gpu = ft.run_weighted_backprojection(sino.cuda(), ang.cuda(), Nz=N)
    assert _rel(gpu, cpu) < 1e-4
    assert _rel(cpu, vol) < 0.45


@needs_cuda
def test_gather_back_projection_matches_cuda_kernel():
    """The CPU W-BP back-projection is the voxel-driven gather of the CUDA
    back_project kernel, including its texture weights rounded to 1/256
    (with exact weights the two differ by 2e-4), for Nz != Nx and a detector
    wider or narrower than the volume."""
    from fasttomo.algorithms import _bp_gather
    g = torch.Generator().manual_seed(0)
    for Ny, Nz, Nx, Nd in [(2, 30, 30, 30), (3, 20, 34, 26), (1, 64, 64, 48)]:
        ang = torch.linspace(-70, 70, 23)
        sino = torch.rand(23, Ny, Nd, generator=g)
        ref = ft.back_project(sino.cuda(), ang.cuda(), Nz=Nz, Nx=Nx).cpu()
        out = _bp_gather(sino, ang, Nz, Nx)
        assert out.shape == ref.shape
        assert _rel(out, ref) < 2e-5
