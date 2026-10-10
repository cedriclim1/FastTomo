"""
Weighted SIRT / exact weighted back-projection: CPU/CUDA parity and
convergence.

Root causes fixed (see the "exact weighting filter" notes in
fasttomo/algorithms/analytic.py):
- the exact-filter deconvolution was a circular FFT on the bare [Nz, Nx]
  grid, with the circular boundary on the volume edge; wSIRT diverged at
  omega=1 on 32 of 45 geometries tested (CUDA), through a mode in the
  outermost voxels and, with many views, one near the missing-wedge edge.
  W-BP now works on a grid with a max(2, N // 16) voxel margin per side,
  symmetric so the crop is exact for odd sizes too;
- the CPU back-projection is ray-driven (the exact transpose of the CPU
  ray-marching forward projector), which is not shift-invariant: a constant
  sinogram back-projects to a +-34 % voxel pattern at 45 degrees. The
  exact filter, computed from the back-projection of one point, amplifies
  that pattern up to ~40x, so CPU W-BP had 83 % error vs 30 % on CUDA. The
  W-BP step now uses the voxel-driven (gather) back-projection on both
  backends, the model the CUDA kernel already implements.

CPU/CUDA parity bound: 6e-4 for W-BP and wSIRT. The source is the CUDA
texture unit's 8-bit interpolation weights (~2e-4 in the raw
back-projection, ~1.3e-4 in the forward projector), amplified by the exact
filter's high-frequency gain: measured up to 4.6e-4 over 18 geometries
(3 tilt schemes x N in 40..65). The filter itself is built from the exact
gather PSF on both backends, so its hard threshold cannot flip between
them. The CPU path uses exact weights and is the reference, pinned
separately against an fp64 NumPy back-projection.
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
    """Same input, 20 iterations: CPU and CUDA agree to 6e-4, the CUDA
    projectors' 8-bit texture interpolation floor as amplified by the
    exact filter (see the module docstring). Before: 34 % apart; CPU 83 %
    error after one iteration vs CUDA 30 %."""
    ang = torch.linspace(-tilt, tilt, n)
    sino = ft.forward_project(_phantom(N, "cpu"), ang)
    cpu = ft.run_wsirt(sino, ang, n_iter=20, Nz=N)
    gpu = ft.run_wsirt(sino.cuda(), ang.cuda(), n_iter=20, Nz=N)
    assert _rel(gpu, cpu) < 6e-4


@needs_cuda
@pytest.mark.parametrize("N,n,tilt", GEOMS)
def test_weighted_backprojection_cpu_cuda_parity(N, n, tilt):
    """run_weighted_backprojection shares the W-BP operator: CPU and CUDA
    agree to 6e-4 (texture floor, see the module docstring) and both
    recover the phantom to the one-step level."""
    ang = torch.linspace(-tilt, tilt, n)
    vol = _phantom(N, "cpu")
    sino = ft.forward_project(vol, ang)
    cpu = ft.run_weighted_backprojection(sino, ang, Nz=N)
    gpu = ft.run_weighted_backprojection(sino.cuda(), ang.cuda(), Nz=N)
    assert _rel(gpu, cpu) < 6e-4
    assert _rel(cpu, vol) < 0.45


# ── geometry of the padded W-BP ────────────────────────────────────────────

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("N", [40, 41, 48, 49])
def test_wbp_margin_crop_equals_direct_back_projection(backend, N):
    """Back-projecting onto the padded grid and cropping the margin gives
    exactly the direct back-projection, for odd and even N (a 2x pad of an
    odd size shifted the crop by half a voxel)."""
    from fasttomo.algorithms import _wbp_backproject, _wbp_margin
    m = _wbp_margin(N)
    ang = torch.linspace(-70, 70, 23, device=backend)
    s = torch.rand(23, 2, N, generator=torch.Generator().manual_seed(N)).to(backend)
    direct = _wbp_backproject(s, ang, N, N)
    crop = _wbp_backproject(s, ang, N + 2 * m, N + 2 * m)[:, m:m + N, m:m + N]
    assert _rel(crop, direct) < 1e-5


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("N", [40, 41, 48, 49])
def test_wbp_keeps_a_centred_disk_centred(backend, N):
    """One W-BP step of a disk at the rotation centre (symmetric tilts):
    the centroid stays within 0.02 voxel of (N-1)/2 on both axes."""
    c = (N - 1) / 2
    zz, xx = torch.meshgrid(torch.arange(N) - c, torch.arange(N) - c, indexing="ij")
    vol = ((zz ** 2 + xx ** 2) <= (N / 5) ** 2).float().unsqueeze(0)
    ang = torch.linspace(-70, 70, 71)
    sino = ft.forward_project(vol, ang).to(backend)
    rec = ft.run_weighted_backprojection(sino, ang.to(backend), Nz=N)[0].cpu().clamp(min=0)
    total = rec.sum()
    cz = float((rec.sum(1) * torch.arange(N)).sum() / total)
    cx = float((rec.sum(0) * torch.arange(N)).sum() / total)
    assert abs(cz - c) < 0.02 and abs(cx - c) < 0.02, (cz, cx, c)


# ── golden quality (41 views, +-70 deg) ────────────────────────────────────
#
# Pinned so a later change cannot lose accuracy silently: one-step W-BP and
# 50-iteration wSIRT errors must stay within 2 % of these values (CPU and
# CUDA agree to 1e-4 here). For reference, main before this fix gave
# N=48: 0.295 / diverging (x2.4 residual) and N=49: 0.299 / 0.275 on CUDA,
# 0.832 / 0.287 and 0.492 / 0.275 on CPU.

GOLDEN = {48: (0.28923, 0.27236), 49: (0.28737, 0.26883)}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("N", sorted(GOLDEN))
def test_wsirt_golden_quality(backend, N):
    e1_ref, e50_ref = GOLDEN[N]
    ang = torch.linspace(-70, 70, 41, device=backend)
    vol = _phantom(N, backend)
    sino = ft.forward_project(vol, ang)
    e1 = _rel(ft.run_weighted_backprojection(sino, ang, Nz=N), vol)
    e50 = _rel(ft.run_wsirt(sino, ang, n_iter=50, Nz=N), vol)
    assert abs(e1 - e1_ref) < 0.02 * e1_ref, e1
    assert abs(e50 - e50_ref) < 0.02 * e50_ref, e50


def _bp_voxel_ref(sino: np.ndarray, ang_deg: np.ndarray, Nz: int, Nx: int) -> np.ndarray:
    """Independent fp64 voxel-driven back-projection: every voxel sums, over
    views, the sinogram row linearly interpolated at its detector position
    u = x cos + z sin + (N_det-1)/2 (centred coordinates), zero outside the
    detector."""
    n, Ny, Nd = sino.shape
    out = np.zeros((Ny, Nz, Nx))
    for k in range(n):
        c, s = math.cos(math.radians(ang_deg[k])), math.sin(math.radians(ang_deg[k]))
        for iz in range(Nz):
            z = iz - 0.5 * (Nz - 1)
            for ix in range(Nx):
                u = (ix - 0.5 * (Nx - 1)) * c + z * s + 0.5 * (Nd - 1)
                i0 = math.floor(u)
                w = u - i0
                for i, wt in ((i0, 1.0 - w), (i0 + 1, w)):
                    if 0 <= i < Nd:
                        out[:, iz, ix] += wt * sino[k, :, i]
    return out


@pytest.mark.parametrize("Ny,Nz,Nx,Nd", [(2, 12, 12, 12), (3, 9, 14, 11), (1, 16, 10, 20)])
def test_gather_back_projection_exact(Ny, Nz, Nx, Nd):
    """The CPU W-BP back-projection is pinned exactly: fp64 input against
    the fp64 NumPy reference at 1e-6 (the operator computes in fp32)."""
    from fasttomo.algorithms import _bp_gather
    g = torch.Generator().manual_seed(Ny * 100 + Nd)
    ang = torch.tensor([-70.0, -41.5, -12.0, 0.0, 7.25, 33.0, 64.0])
    sino = torch.rand(len(ang), Ny, Nd, generator=g)
    out = _bp_gather(sino, ang, Nz, Nx)
    ref = _bp_voxel_ref(sino.double().numpy(), ang.double().numpy(), Nz, Nx)
    assert out.shape == (Ny, Nz, Nx) and out.dtype == torch.float32
    assert _rel(out, torch.from_numpy(ref)) < 1e-6


@needs_cuda
def test_gather_back_projection_matches_cuda_kernel():
    """The CPU W-BP back-projection is the operator of the CUDA
    back_project kernel, for Nz != Nx and a detector wider or narrower than
    the volume; they differ only by the texture unit's 8-bit interpolation
    weights (~2e-4 on random data), hence the 3e-4 bound."""
    from fasttomo.algorithms import _bp_gather
    g = torch.Generator().manual_seed(0)
    for Ny, Nz, Nx, Nd in [(2, 30, 30, 30), (3, 20, 34, 26), (1, 64, 64, 48)]:
        ang = torch.linspace(-70, 70, 23)
        sino = torch.rand(23, Ny, Nd, generator=g)
        ref = ft.back_project(sino.cuda(), ang.cuda(), Nz=Nz, Nx=Nx).cpu()
        out = _bp_gather(sino, ang, Nz, Nx)
        assert out.shape == ref.shape
        assert _rel(out, ref) < 3e-4


def _sphere_slice(N: int, Ny: int = 2) -> torch.Tensor:
    """The phantom of the solver audit's long-run tables: the central slice
    of a sphere with a denser inclusion, repeated over Ny."""
    y = torch.linspace(-1, 1, N)[:, None, None]
    z = torch.linspace(-1, 1, N)[None, :, None]
    x = torch.linspace(-1, 1, N)[None, None, :]
    v = ((x ** 2 + z ** 2 + (y * 0.9) ** 2) < 0.36).float()
    v += 0.5 * (((x - 0.2) ** 2 + (z + 0.1) ** 2 + y ** 2) < 0.04).float()
    return v[N // 2:N // 2 + 1].repeat(Ny, 1, 1).contiguous()


@needs_cuda
def test_wsirt_long_run_behaviour():
    """Solver audit A3, pinned as documented in run_wsirt: without
    positivity wSIRT is accurate for ~200 iterations and then diverges
    slowly (spectral radius of I - W A slightly above 1); with positivity it
    keeps converging. If this starts failing because the long run became
    stable, update the run_wsirt docstring."""
    N = 64
    ang = torch.linspace(-60, 60, 61)
    xt = _sphere_slice(N)
    b = ft.forward_project(xt, ang).cuda()
    a = ang.cuda()
    e200 = _rel(ft.run_wsirt(b, a, 200), xt)
    e3000 = _rel(ft.run_wsirt(b, a, 3000), xt)
    assert e200 < 0.25
    assert e3000 > 4 * e200, (e200, e3000)               # the slow divergence
    p200 = _rel(ft.run_wsirt(b, a, 200, positivity=True), xt)
    p3000 = _rel(ft.run_wsirt(b, a, 3000, positivity=True), xt)
    assert p3000 < p200 < 0.16, (p200, p3000)            # stable with positivity
