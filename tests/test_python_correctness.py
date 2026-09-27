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


# ── ConstraintConfig.sparsity honoured by run_ad_recon ─────────────────────

class _MeanZProjector:
    """CPU stand-in for ParallelProjector: every view is the Z-mean of the
    volume. Linear, so the fidelity gradient is non-zero for every voxel."""

    def __init__(self, n_angles: int):
        self.n_angles = n_angles

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return volume.mean(dim=1).unsqueeze(0).expand(self.n_angles, -1, -1)


def _cpu_ad_recon(monkeypatch, *, n_angles=4, Ny=2, N=6, projector=None):
    """Patch run_ad_recon's CUDA-only input check and projector class so the
    loop runs on CPU with ``projector`` (default `_MeanZProjector`)."""
    import fasttomo.algorithms as algo

    proj = projector if projector is not None else _MeanZProjector(n_angles)

    def fake_check_inputs(sinogram, angles_deg, Nz, **kwargs):
        n, ny, nd = sinogram.shape
        return n, ny, nd, (nd if Nz < 0 else Nz), sinogram.device

    monkeypatch.setattr(algo, "ParallelProjector", type(proj))
    monkeypatch.setattr(algo, "_check_inputs", fake_check_inputs)
    return proj


@pytest.mark.parametrize("lam,expected", [(0.0, 0.2), (0.05, 0.15), (0.3, 0.0)])
def test_ad_recon_constraint_sparsity_soft_thresholds(monkeypatch, lam, expected):
    """One Adam step from a zero volume moves every voxel by exactly
    lr_volume (Adam's first step is lr * g / (|g| + eps)); the post-step
    soft-threshold then leaves |v| = max(lr - lam, 0). Before the fix
    ConstraintConfig.sparsity was ignored and |v| stayed 0.2."""
    proj = _cpu_ad_recon(monkeypatch)
    sino = torch.full((4, 2, 6), 3.0)           # target above the start: v grows
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=1, lr_volume=0.2, cosine_schedule=False),
        constraints=ft.ConstraintConfig(sparsity=lam),
    )
    rec = ft.run_ad_recon(sino, torch.linspace(-30, 30, 4), Nz=6,
                          projector=proj, config=cfg)
    torch.testing.assert_close(rec, torch.full_like(rec, expected), rtol=0, atol=1e-6)


def test_ad_recon_constraint_sparsity_matches_config_apply(monkeypatch):
    """Several iterations: run_ad_recon's result equals a hand-written loop
    (Adam step, then ConstraintConfig.apply) with the same settings."""
    proj = _cpu_ad_recon(monkeypatch)
    g = torch.Generator().manual_seed(5)
    sino = torch.rand(4, 2, 6, generator=g) * 2 - 0.5
    cons = ft.ConstraintConfig(positivity=False, sparsity=0.02)
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=6, lr_volume=0.1, cosine_schedule=False),
        constraints=cons,
    )
    rec = ft.run_ad_recon(sino, torch.linspace(-30, 30, 4), Nz=6,
                          projector=proj, config=cfg)

    v = torch.zeros(2, 6, 6, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.1)
    for _ in range(6):
        opt.zero_grad()
        torch.nn.functional.mse_loss(proj.forward(v), sino).backward()
        opt.step()
        with torch.no_grad():
            v.copy_(cons.apply(v))
    assert torch.equal(rec, v.detach())
    cfg.constraints = ft.ConstraintConfig(positivity=False)
    unthresholded = ft.run_ad_recon(sino, torch.linspace(-30, 30, 4), Nz=6,
                                    projector=proj, config=cfg)
    assert not torch.equal(rec, unthresholded)


@needs_cuda
def test_ad_recon_constraint_sparsity_cuda():
    """Real projector: the soft-threshold zeroes the small background voxels
    that plain AD leaves as noise, and sparsity=0 keeps the default result
    bit-identical."""
    torch.manual_seed(0)
    N, n = 32, 25
    ang = torch.linspace(-60, 60, n, device="cuda")
    vol = torch.zeros(2, N, N, device="cuda")
    vol[:, 10:20, 12:22] = 1.0
    sino = ft.forward_project(vol, ang)
    sino = sino + 0.05 * torch.randn_like(sino)
    base = ft.AdReconConfig(optim=ft.OptimConfig(n_iter=40, lr_volume=0.05))
    rec0 = ft.run_ad_recon(sino, ang, Nz=N, config=base)
    rec0b = ft.run_ad_recon(sino, ang, Nz=N, config=ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=40, lr_volume=0.05),
        constraints=ft.ConstraintConfig(sparsity=0.0)))
    assert torch.equal(rec0, rec0b)
    rec = ft.run_ad_recon(sino, ang, Nz=N, config=ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=40, lr_volume=0.05),
        constraints=ft.ConstraintConfig(sparsity=0.01)))
    bg = vol == 0
    assert (rec[bg] == 0).float().mean() > 0.5 > (rec0[bg] == 0).float().mean()
    assert _rel(rec, vol) < 1.0
