"""
FISTA and FISTA-TV (run_fista, run_fista_tv): convergence, constraints,
warm start, the TV proximal solvers and projector reuse.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    make_angles,
    needs_cuda,
    phantom_2d,
)

import fasttomo as ft

# ── FISTA ─────────────────────────────────────────────────────────────────────


def test_fista_shape(backend):
    """Replaces test_cpu.py::test_fista_shape and test_cuda.py::test_fista_shape_cuda."""
    N      = 32
    angles = make_angles(30, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista(sino, angles, n_iter=5, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == backend.type


def test_fista_converges(backend):
    """FISTA data-fit loss must drop by 10x over the run.

    Replaces test_cpu.py::test_fista_converges (30 iterations) and
    test_cuda.py::test_fista_converges_cuda (40 iterations)."""
    n_iter = {"cpu": 30, "cuda": 40}[backend.type]
    N      = 64
    angles = make_angles(60, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    rec, losses = ft.run_fista(sino, angles, n_iter=n_iter, Nz=N, return_loss=True)
    assert losses.shape == (n_iter,)
    # FISTA can oscillate slightly (Nesterov momentum); 10% slack over window
    assert losses[-1].item() < losses[0].item() * 0.1, (
        f"FISTA loss did not decrease: first={losses[0].item():.4e}, "
        f"last={losses[-1].item():.4e}"
    )


def test_fista_positivity(backend):
    """Replaces test_cpu.py::test_fista_positivity and
    test_cuda.py::test_fista_positivity_cuda."""
    N      = 32
    angles = make_angles(30, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista(sino, angles, n_iter=10, Nz=N, positivity=True)
    assert rec.min().item() >= -1e-6


def test_fista_mask_zeros_corners():
    """mask=True must zero the inscribed-cylinder corners."""
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista(sino, angles, n_iter=5, Nz=N, mask=True)
    corners = torch.stack([
        rec[0,  0,  0], rec[0,  0, -1],
        rec[0, -1,  0], rec[0, -1, -1],
    ])
    assert corners.abs().max().item() < 1e-6


def test_fista_warm_start_shape_mismatch_raises():
    n_angles, Ny, N = 12, 4, 16
    sino = torch.zeros(n_angles, Ny, N, dtype=torch.float32)
    angles = make_angles(n_angles)
    warm = torch.zeros(1, N, N, dtype=torch.float32)  # Ny mismatch
    with pytest.raises(ValueError, match="volume must have shape"):
        ft.run_fista(sino, angles, n_iter=1, Nz=N, warm_start=warm)


def test_fista_user_L_skips_power_iteration():
    """L= kwarg must produce the same shape and a sane reconstruction."""
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista(sino, angles, n_iter=10, Nz=N, L=20.0)
    assert rec.shape == vol.shape
    assert torch.isfinite(rec).all()


DEVICE = torch.device("cuda")


def make_euler_pure_y(angles_deg: torch.Tensor) -> torch.Tensor:
    """Wrap scalar tilt angles as ZYZ triples (φ=0, θ=angle, ψ=0)."""
    n = len(angles_deg)
    euler = torch.zeros(n, 3)
    euler[:, 1] = angles_deg.cpu()
    return euler


@needs_cuda
def test_fista_3d_zyz_cuda():
    """run_fista with euler_zyz_deg = single-axis ZYZ should match scalar."""
    N      = 32
    angles = make_angles(30, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project_3d(vol, euler)

    torch.cuda.manual_seed(0)
    rec_zyz = ft.run_fista(sino, angles, n_iter=30, Nz=N, euler_zyz_deg=euler)
    torch.cuda.manual_seed(0)
    rec_sa  = ft.run_fista(sino, angles, n_iter=30, Nz=N)
    rel = (rec_zyz - rec_sa).pow(2).mean().sqrt().item() / (rec_sa.abs().mean().item() + 1e-12)
    assert rel < 0.1, f"ZYZ path diverges from scalar (rel-RMSE={rel:.3f})"


@needs_cuda
def test_fista_with_parallel_projector_cuda():
    """Passing a pre-built ParallelProjector should produce the same output
    as letting run_fista build one internally."""
    N      = 32
    angles = make_angles(30, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    # Seed CUDA RNG before each call so _estimate_lipschitz draws the same
    # random starting vector → identical L_eff → identical FISTA trajectories.
    torch.cuda.manual_seed(0)
    rec_internal = ft.run_fista(sino, angles, n_iter=20, Nz=N)

    proj = ft.ParallelProjector(
        (1, N, N), angles, device=DEVICE, N_det=N,
    )
    torch.cuda.manual_seed(0)
    rec_supplied = ft.run_fista(sino, angles, n_iter=20, Nz=N, projector=proj)

    diff = (rec_internal - rec_supplied).abs().max().item()
    assert diff < 1e-4, f"Supplied projector diverges (max abs diff={diff:.2e})"


@needs_cuda
def test_fista_warm_start_continues_cuda():
    """Warm-starting from a converged volume with n_iter=1 should change
    the result only slightly (already near the optimum)."""
    N      = 32
    angles = make_angles(30, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    converged = ft.run_fista(sino, angles, n_iter=50, Nz=N)
    one_more  = ft.run_fista(sino, angles, n_iter=1, Nz=N, warm_start=converged)

    # Should still be a near-optimum reconstruction
    final_err = (one_more - vol).pow(2).mean().item()
    converged_err = (converged - vol).pow(2).mean().item()
    assert final_err < converged_err * 2.0, (
        f"Warm-start blew up: {final_err:.4e} vs {converged_err:.4e}"
    )


# ── FISTA-TV ──────────────────────────────────────────────────────────────────


def test_fista_tv_shape(backend):
    """Replaces test_cpu.py::test_fista_tv_shape and
    test_cuda.py::test_fista_tv_shape_cuda."""
    N      = 32
    angles = make_angles(30, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista_tv(sino, angles, tv_weight=0.01, n_iter=5, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == backend.type


def test_fista_tv_positivity_inside_prox(backend):
    """positivity=True is baked into the TV prox — output must be ≥ 0
    without any after-step clamp.

    Replaces test_cpu.py::test_fista_tv_positivity_inside_prox and
    test_cuda.py::test_fista_tv_positivity_inside_prox_cuda."""
    N      = 32
    angles = make_angles(30, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista_tv(
        sino, angles, tv_weight=0.05, n_iter=20, Nz=N, positivity=True,
    )
    assert rec.min().item() >= -1e-6


def test_fista_tv_reduces_tv_vs_fista():
    """At matched n_iter, FISTA-TV must produce a less-TV reconstruction."""
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    rec        = ft.run_fista(sino, angles, n_iter=40, Nz=N)
    rec_tv     = ft.run_fista_tv(sino, angles, tv_weight=0.05, n_iter=40, Nz=N)

    # CPU pure-PyTorch TV (tv_loss_3d's CUDA path is fp32-only and Ny=1 hits an
    # empty Y-diff; recompute here for portability)
    def tv(v):
        dz = v[:, 1:, :] - v[:, :-1, :]
        dx = v[:, :, 1:] - v[:, :, :-1]
        return (dz[:, :, :-1].pow(2) + dx[:, :-1, :].pow(2) + 1e-8).sqrt().mean().item()

    tv_fista    = tv(rec)
    tv_fista_tv = tv(rec_tv)
    assert tv_fista_tv < tv_fista, (
        f"FISTA-TV did not reduce TV (FISTA={tv_fista:.4f}, FISTA-TV={tv_fista_tv:.4f})"
    )


@needs_cuda
def test_fista_tv_reduces_tv_vs_fista_cuda():
    """3-D phantom — verifies the prox actually smooths along all three axes."""
    N      = 32
    angles = make_angles(30, device=DEVICE)
    vol    = torch.zeros(N, N, N, device=DEVICE)
    vol[N//4:3*N//4, N//4:3*N//4, N//4:3*N//4] = 1.0
    sino   = ft.forward_project(vol, angles)

    rec    = ft.run_fista(sino, angles, n_iter=40, Nz=N)
    rec_tv = ft.run_fista_tv(sino, angles, tv_weight=0.1, n_iter=40, Nz=N)

    tv_fista    = ft.tv_loss_3d(rec).item()
    tv_fista_tv = ft.tv_loss_3d(rec_tv).item()
    assert tv_fista_tv < tv_fista, (
        f"FISTA-TV did not reduce TV (FISTA={tv_fista:.4f}, "
        f"FISTA-TV={tv_fista_tv:.4f})"
    )


def test_fista_tv_chambolle_mode():
    """tv_inner='chambolle' must run and produce a finite reconstruction."""
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista_tv(
        sino, angles, tv_weight=0.05, n_iter=20, Nz=N, tv_inner="chambolle",
    )
    assert rec.shape == vol.shape
    assert torch.isfinite(rec).all()


@needs_cuda
def test_fista_tv_chambolle_matches_fgp_at_convergence_cuda():
    """FGP and Chambolle solve the same prox; at enough inner iters the
    reconstructions should agree closely."""
    N      = 32
    angles = make_angles(30, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    torch.cuda.manual_seed(0)
    rec_fgp = ft.run_fista_tv(
        sino, angles, tv_weight=0.05, n_iter=30, Nz=N,
        tv_inner="fgp", tv_inner_iter=40,
    )
    torch.cuda.manual_seed(0)
    rec_cha = ft.run_fista_tv(
        sino, angles, tv_weight=0.05, n_iter=30, Nz=N,
        tv_inner="chambolle", tv_inner_iter=100,
    )
    diff = (rec_fgp - rec_cha).pow(2).mean().sqrt().item()
    scale = rec_fgp.abs().mean().item() + 1e-12
    assert diff / scale < 0.05, (
        f"FGP and Chambolle disagree more than 5% rel-RMSE: "
        f"diff={diff:.4e}, scale={scale:.4e}"
    )


def test_fista_tv_zero_weight_raises():
    N      = 16
    angles = make_angles(10)
    sino   = torch.zeros(len(angles), 1, N)
    with pytest.raises(ValueError, match="tv_weight > 0"):
        ft.run_fista_tv(sino, angles, tv_weight=0.0, n_iter=1, Nz=N)


def test_fista_tv_unknown_inner_mode_raises():
    N      = 16
    angles = make_angles(10)
    sino   = torch.zeros(len(angles), 1, N)
    with pytest.raises(ValueError, match="unknown mode"):
        ft.run_fista_tv(sino, angles, tv_weight=0.01, n_iter=1, Nz=N, tv_inner="bogus")


# ── Lipschitz power iteration in fp16 (solver audit A4) ──────────────────────


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=needs_cuda)])
def test_lipschitz_fp16_above_fp16_range(dev):
    """The power-iteration norms are fp32: an operator with ||A^T A|| above
    65504 (the fp16 maximum) gives its true L in fp16 too. The fp16 norm
    used to overflow to inf, zero the iterate and make 1 / L raise
    ZeroDivisionError."""
    from fasttomo.algorithms.fista import _estimate_lipschitz

    s = 300.0                               # A = s * I: L = s**2 = 90000

    def fp(v):
        return v * s

    def bp(r):
        return r * s

    torch.manual_seed(0)
    L = _estimate_lipschitz(fp, bp, (2, 8, 8), torch.device(dev), torch.float16, 5)
    assert L == pytest.approx(s * s * 1.01, rel=2e-3)


@needs_cuda
def test_fista_fp16_large_lipschitz_runs():
    """512-pixel detector, 180 views: L ~ 1.3e5 > 65504. fp16 run_fista used
    to raise ZeroDivisionError (audit A4); it now runs and agrees with fp32."""
    N, Ny, n = 512, 4, 180
    ang = torch.linspace(0, 179, n, device="cuda")
    vol = torch.zeros(Ny, N, N, device="cuda")
    vol[:, N // 2 - 20:N // 2 + 20, N // 2 - 20:N // 2 + 20] = 1.0
    sino = ft.forward_project(vol, ang)
    torch.manual_seed(0)
    x32 = ft.run_fista(sino, ang, 5)
    torch.manual_seed(0)
    x16 = ft.run_fista(sino.half(), ang, 5)
    assert x16.dtype == torch.float16 and torch.isfinite(x16).all()
    err = (x16.float() - x32).norm() / x32.norm()
    assert err < 1e-2, err


# ── fused FISTA update (solver audit P4) ─────────────────────────────────────


def _edge(t: torch.Tensor) -> torch.Tensor:
    """``t`` with NaN, +-inf, +-0, subnormals and fp16 extremes up front."""
    vals = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-40, -1e-40,
            6e-8, -6e-8, 65504.0, -65504.0, 1e-8, -1e-8]
    t.view(-1)[:len(vals)] = torch.tensor(vals, device=t.device).to(t.dtype)
    return t


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.float16 else torch.int32)


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(3, 5, 7), (2, 33, 65), (4, 16, 24)])
def test_fista_update_kernel_bit_identical(dtype, shape):
    """The three modes of fista_update_cuda give the bits of the PyTorch
    chains of _fista_loop, for every constraint combination, broadcast and
    full support masks, several step sizes and momenta, and NaN / inf /
    signed zero / subnormal inputs."""
    from fasttomo.algorithms.fista import _fista_update
    from fasttomo.configs import ConstraintConfig

    g = torch.Generator().manual_seed(0)

    def rand(*sz, scale=1.0, positive=False):
        x = torch.rand(*sz, generator=g) if positive else torch.randn(*sz, generator=g)
        return (x * scale).to("cuda", dtype)

    Ny, Nz, Nx = shape
    y = _edge(rand(*shape, scale=3.0))
    grad = _edge(rand(*shape, scale=300.0))
    xp = rand(*shape, scale=2.0)
    xn = _edge(rand(*shape, scale=2.0))
    for inv_L in (1.0 / 3.0, 0.0123, 1.7e-5):
        u = torch.empty_like(y)
        assert _fista_update(1, y, grad, None, u, None, inv_L, 0.0, None, None)
        assert torch.equal(_bits(u), _bits(y - inv_L * grad))
        for beta in (0.0, 0.3747, 0.97):
            yo = torch.empty_like(y)
            assert _fista_update(2, xn, None, xp, None, yo, inv_L, beta, None, None)
            assert torch.equal(_bits(yo), _bits(xn + beta * (xn - xp)))
            for mask_shape in (None, (1, Nz, Nx), (Ny, Nz, Nx)):
                mask = None if mask_shape is None else rand(*mask_shape, positive=True)
                for positivity in (False, True):
                    cons = ConstraintConfig(positivity=positivity, support_mask=mask)
                    x_want = cons.apply(y - inv_L * grad)
                    y_want = x_want + beta * (x_want - xp)
                    x_got, y_got = xp.clone(), y.clone()          # in place, as the loop
                    assert _fista_update(0, y_got, grad, x_got, x_got, y_got, inv_L, beta,
                                         cons, None)
                    assert torch.equal(_bits(x_got), _bits(x_want)), (mask_shape, positivity)
                    assert torch.equal(_bits(y_got), _bits(y_want)), (mask_shape, positivity)


@needs_cuda
def test_fista_update_declines_unfusable_inputs():
    """CPU tensors, mixed dtypes, a soft threshold and odd mask shapes fall
    back to the PyTorch chain (the helper returns False)."""
    from fasttomo.algorithms.fista import _fista_update
    from fasttomo.configs import ConstraintConfig

    v = torch.ones(2, 4, 4, device="cuda")
    assert not _fista_update(1, v.cpu(), v.cpu(), None, v.cpu(), None, 0.1, 0.0, None, None)
    assert not _fista_update(1, v, v.half(), None, v.clone(), None, 0.1, 0.0, None, None)
    cons = ConstraintConfig(soft_threshold=0.1)
    assert not _fista_update(0, v, v, v.clone(), v.clone(), v.clone(), 0.1, 0.5, cons, None)
    odd = ConstraintConfig(support_mask=torch.ones(4, 4, device="cuda"))
    assert not _fista_update(0, v, v, v.clone(), v.clone(), v.clone(), 0.1, 0.5, odd, None)


def _fista_cases(dtype, Ny):
    N, n = 24, 10
    ang = make_angles(n, device="cuda")
    g = torch.Generator().manual_seed(1)
    vol = torch.rand(Ny, N, N, generator=g).cuda()
    sino = ft.forward_project(vol, ang).to(dtype)
    eul = torch.stack([torch.zeros(n), ang.cpu(), torch.zeros(n)], 1).cuda()
    sino3d = ft.forward_project_3d(vol, eul).to(dtype)
    sm = torch.zeros(1, N, N, device="cuda", dtype=dtype)
    sm[:, 3:-3, 2:-5] = 1
    return [
        (ft.run_fista, dict(sinogram=sino, angles_deg=ang, n_iter=5, L=300.0)),
        (ft.run_fista, dict(sinogram=sino, angles_deg=ang, n_iter=5, L=300.0,
                            positivity=True, support_mask=sm, return_loss=True)),
        (ft.run_fista, dict(sinogram=sino, angles_deg=ang, n_iter=4, L=300.0, mask=True,
                            projector=ft.ParallelProjector((Ny, N, N), ang, device="cuda"))),
        (ft.run_fista, dict(sinogram=sino3d, angles_deg=ang, n_iter=4, L=300.0,
                            euler_zyz_deg=eul, positivity=True)),
        (ft.run_fista_tv, dict(sinogram=sino, angles_deg=ang, n_iter=4, L=300.0,
                               tv_weight=0.05, positivity=True)),
    ]


@needs_cuda
@pytest.mark.parametrize("dtype, Ny", [(torch.float32, 3), (torch.float16, 3),
                                       (torch.float16, 4)])
def test_fista_fused_path_bit_identical(monkeypatch, dtype, Ny):
    """run_fista / run_fista_tv with the fused kernels and direct volume-
    texture writes give exactly the PyTorch chain's bits (fp16 with
    Ny % 4 == 0: packed textures; rotmat: 3-D texture), and the forward
    projection of the new iterate skips its upload."""
    from fasttomo.algorithms import fista

    fused = []
    for fn, kw in _fista_cases(dtype, Ny):
        r = fn(**kw)
        fused.append(r)
    proj = _fista_cases(dtype, Ny)[2][1]["projector"]
    ft.run_fista(**dict(_fista_cases(dtype, Ny)[2][1], projector=proj))
    assert proj._op.upload_counts["vol_skipped"] >= 3
    monkeypatch.setattr(fista, "_fista_update", lambda *a, **k: False)
    for (fn, kw), a in zip(_fista_cases(dtype, Ny), fused):
        b = fn(**kw)
        for x, y in zip(a if isinstance(a, tuple) else (a,), b if isinstance(b, tuple) else (b,)):
            assert torch.equal(x, y), (fn.__name__, kw.keys())
