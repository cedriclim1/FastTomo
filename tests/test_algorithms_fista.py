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
