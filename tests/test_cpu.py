"""
CPU correctness tests for FastTomo projection primitives and reconstructions.

Run with:  pytest tests/test_cpu.py -v
"""

import pytest
import torch
import numpy as np

import fasttomo as ft


# ── helpers ────────────────────────────────────────────────────────────────

def make_angles(n: int, max_deg: float = 70.0) -> torch.Tensor:
    """n evenly spaced angles in [-max_deg, +max_deg) in degrees."""
    return torch.linspace(-max_deg, max_deg, n + 1)[:-1]


def phantom_2d(N: int) -> torch.Tensor:
    """Simple square phantom centred in an N×N volume slice."""
    vol = torch.zeros(N, N)
    q   = N // 4
    vol[q:3*q, q:3*q] = 1.0
    return vol


# ── adjoint test ───────────────────────────────────────────────────────────

def test_adjoint_cpu():
    """
    Verify <Ax, y> ≈ <x, A^T y> to reasonable tolerance.

    A = forward_project,  A^T = back_project.
    This is the fundamental correctness check for the kernel pair.
    """
    N      = 64
    Ny     = 1
    angles = make_angles(30)

    x = torch.randn(Ny, N, N)
    y = torch.randn(len(angles), Ny, N)

    Ax  = ft.forward_project(x, angles)
    ATy = ft.back_project(y, angles, Nz=N)

    lhs = (Ax  * y).sum().item()
    rhs = (x * ATy).sum().item()

    rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-12)
    assert rel_err < 5e-3, (
        f"Adjoint test failed: <Ax,y>={lhs:.6f} vs <x,A^Ty>={rhs:.6f} "
        f"(rel_err={rel_err:.2e})"
    )


# ── shape smoke tests ──────────────────────────────────────────────────────

def test_forward_back_shapes():
    N      = 32
    angles = make_angles(20)
    vol    = torch.ones(1, N, N)

    sino = ft.forward_project(vol, angles)
    assert sino.shape == (len(angles), 1, N)

    rec = ft.back_project(sino, angles, Nz=N)
    assert rec.shape == (1, N, N)


def test_forward_project_zero_volume():
    N      = 32
    angles = make_angles(10)
    vol    = torch.zeros(1, N, N)
    sino   = ft.forward_project(vol, angles)
    assert sino.abs().max().item() == 0.0


# ── FBP ───────────────────────────────────────────────────────────────────

def test_fbp_shape():
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fbp(sino, angles, Nz=N)
    assert rec.shape == vol.shape


def test_fbp_better_than_plain_backproject():
    """FBP reconstruction MSE should beat raw (unfiltered) back-projection."""
    N      = 64
    angles = make_angles(90)
    vol    = phantom_2d(N).unsqueeze(0)
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


# ── SIRT ──────────────────────────────────────────────────────────────────

def test_sirt_shape():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_sirt(sino, angles, n_iter=5, Nz=N)
    assert rec.shape == vol.shape


def test_sirt_converges():
    """Reconstruction MSE should decrease as iterations increase."""
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    errors = []
    for n in (1, 10, 30, 80):
        rec = ft.run_sirt(sino, angles, n_iter=n, Nz=N)
        errors.append(((rec - vol) ** 2).mean().item())

    # allow 5% slack to account for oscillation near convergence
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] * 1.05, (
            f"SIRT did not converge: MSE history = {errors}"
        )


def test_sirt_positivity():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_sirt(sino, angles, n_iter=10, Nz=N, positivity=True)
    assert rec.min().item() >= -1e-6


# ── W-BP ──────────────────────────────────────────────────────────────────

def test_wbp_shape():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_weighted_backprojection(sino, angles, Nz=N)
    assert rec.shape == vol.shape


# ── GENFIRE ───────────────────────────────────────────────────────────────

def test_genfire_shape():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)        # [1, N, N]
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=10)
    assert rec.shape == vol.shape


def test_genfire_positivity():
    """positivity=True must eliminate all negative voxels."""
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=10, positivity=True)
    assert rec.min().item() >= -1e-6, f"negative voxels found: min={rec.min():.4e}"


def test_genfire_mask():
    """With mask=True the corner voxels (outside inscribed cylinder) should be ~0."""
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=5, mask=True)
    # corners are always outside the inscribed cylinder
    corners = torch.stack([
        rec[0,  0,  0], rec[0,  0, -1],
        rec[0, -1,  0], rec[0, -1, -1],
    ])
    assert corners.abs().max().item() < 1e-6, (
        f"corner voxels should be masked to 0, got {corners.abs().max():.2e}"
    )


def test_genfire_converges():
    """Reconstruction error should decrease (or stay flat) as iterations increase."""
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    errors = []
    for n in (5, 20, 50):
        rec = ft.run_genfire(sino, angles, Nz=N, n_iter=n,
                             positivity=True, mask=False,
                             resolution_extension=False)
        errors.append(((rec - vol) ** 2).mean().item())

    # allow 10 % slack to accommodate best-iterate tracking non-monotonicity
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] * 1.10, (
            f"GENFIRE did not converge: MSE history = {errors}"
        )


def test_genfire_reconstruction_quality():
    """On noiseless data GENFIRE NMSE should be below 0.15."""
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=50,
                            positivity=True, mask=True)

    # normalised MSE (both normalised to [0,1])
    def nmse(a, b):
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        b = (b - b.min()) / (b.max() - b.min() + 1e-9)
        return ((a - b) ** 2).mean().item()

    err = nmse(rec, vol)
    assert err < 0.15, f"GENFIRE NMSE too high: {err:.4f}"


def test_genfire_no_resolution_extension():
    """Algorithm must run without resolution extension and return correct shape."""
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=10,
                            resolution_extension=False)
    assert rec.shape == vol.shape


# ── RESIRE ────────────────────────────────────────────────────────────────

def test_resire_shape():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_resire(sino, angles, n_iter=10, Nz=N)
    assert rec.shape == vol.shape


def test_resire_positivity():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_resire(sino, angles, n_iter=10, Nz=N, positivity=True)
    assert rec.min().item() >= -1e-6


def test_resire_mask():
    """With mask=True the corner voxels (outside inscribed cylinder) should be ~0."""
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_resire(sino, angles, n_iter=10, Nz=N, mask=True)
    corners = torch.stack([
        rec[0,  0,  0], rec[0,  0, -1],
        rec[0, -1,  0], rec[0, -1, -1],
    ])
    assert corners.abs().max().item() < 1e-6


def test_resire_converges():
    """Reconstruction MSE should decrease as iterations increase."""
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    errors = []
    for n in (1, 10, 30, 80):
        rec = ft.run_resire(sino, angles, n_iter=n, Nz=N, positivity=False)
        errors.append(((rec - vol) ** 2).mean().item())

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] * 1.05, (
            f"RESIRE did not converge: MSE history = {errors}"
        )


def test_resire_custom_support():
    """Explicit support tensor should zero voxels outside it."""
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    support = ft.cylindrical_mask(N, N).unsqueeze(0)  # [1, N, N]
    rec    = ft.run_resire(sino, angles, n_iter=5, Nz=N, support=support)
    corners = torch.stack([rec[0, 0, 0], rec[0, 0, -1], rec[0, -1, 0], rec[0, -1, -1]])
    assert corners.abs().max().item() < 1e-6


# ── refine_angles ─────────────────────────────────────────────────────────

def test_refine_angles_shape():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    refined = ft.refine_angles(vol, sino, angles)
    assert refined.shape == angles.shape


def test_refine_angles_returns_cpu_tensor():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    refined = ft.refine_angles(vol, sino, angles)
    assert refined.device.type == "cpu"


def test_refine_angles_recovers_perturbed_angles():
    """
    Perturb true angles by up to 2°; refinement should reduce angular error.
    Uses noiseless forward projections so ground truth is exact.
    """
    N          = 64
    true_angles = make_angles(20, max_deg=60.0)
    vol         = phantom_2d(N).unsqueeze(0)
    sino        = ft.forward_project(vol, true_angles)

    torch.manual_seed(0)
    perturbed = true_angles + torch.randn(len(true_angles)) * 1.5
    error_before = (perturbed - true_angles).abs().mean().item()

    # one round of refinement with a 3° search window
    refined = ft.refine_angles(vol, sino, perturbed,
                               search_range_deg=3.0, n_steps=61)
    error_after = (refined - true_angles).abs().mean().item()

    assert error_after < error_before, (
        f"refine_angles did not reduce angular error: "
        f"{error_before:.3f}° → {error_after:.3f}°"
    )


def test_refine_angles_exact_angles_unchanged():
    """
    If angles are already correct the refined angles should stay within
    the grid resolution of the search (search_range / (n_steps - 1)).
    """
    N      = 32
    angles = make_angles(15, max_deg=60.0)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    n_steps = 31
    search  = 3.0
    refined = ft.refine_angles(vol, sino, angles,
                               search_range_deg=search, n_steps=n_steps)
    grid_step = 2 * search / (n_steps - 1)
    max_shift = (refined - angles).abs().max().item()
    assert max_shift <= grid_step + 1e-5, (
        f"correct angles shifted more than one grid step: {max_shift:.4f}°"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
