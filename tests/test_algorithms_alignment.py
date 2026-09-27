"""
Alignment: tilt-angle refinement (refine_angles) and detector-shift
estimation (estimate_shifts_xcorr).
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

# ── refine_angles ─────────────────────────────────────────────────────────────


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


def test_refine_angles_recovers_perturbed_angles(backend):
    """
    Perturb true angles by up to 2°; refinement should reduce angular error.
    Uses noiseless forward projections so ground truth is exact.

    Replaces test_cpu.py::test_refine_angles_recovers_perturbed_angles and
    test_cuda.py::test_refine_angles_cuda_recovers_perturbed_angles.
    """
    N           = 64
    true_angles = make_angles(20, max_deg=60.0, device=backend)
    vol         = phantom_2d(N, device=backend).unsqueeze(0)
    sino        = ft.forward_project(vol, true_angles)

    torch.manual_seed(0)
    perturbed = true_angles + torch.randn(len(true_angles)).to(backend) * 1.5
    error_before = (perturbed.cpu() - true_angles.cpu()).abs().mean().item()

    # one round of refinement with a 3° search window
    refined = ft.refine_angles(vol, sino, perturbed,
                               search_range_deg=3.0, n_steps=61)
    error_after = (refined - true_angles.cpu()).abs().mean().item()

    assert error_after < error_before, (
        f"refine_angles did not reduce angular error ({backend}): "
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


DEVICE = torch.device("cuda")


def make_euler_pure_y(angles_deg: torch.Tensor) -> torch.Tensor:
    """Wrap scalar tilt angles as ZYZ triples (φ=0, θ=angle, ψ=0)."""
    n = len(angles_deg)
    euler = torch.zeros(n, 3)
    euler[:, 1] = angles_deg.cpu()
    return euler


@needs_cuda
def test_refine_angles_3d_cuda():
    """ZYZ coordinate-descent refinement reduces mean angular error."""
    N           = 64
    true_angles = make_angles(20, max_deg=60.0, device=DEVICE)
    true_euler  = make_euler_pure_y(true_angles).to(DEVICE)
    vol         = phantom_2d(N, device=DEVICE).unsqueeze(0).to(DEVICE)
    sino        = ft.forward_project_3d(vol, true_euler)

    torch.manual_seed(0)
    perturbed = true_euler.clone()
    perturbed[:, 1] = perturbed[:, 1] + torch.randn(len(true_angles)).to(DEVICE) * 1.5

    error_before = (perturbed[:, 1] - true_euler[:, 1]).abs().mean().item()
    refined = ft.refine_angles(vol, sino, euler_zyz_deg=perturbed,
                               search_range_deg=3.0, n_steps=61)
    error_after = (refined[:, 1].to(DEVICE) - true_euler[:, 1]).abs().mean().item()

    assert error_after < error_before, (
        f"3D refine_angles did not reduce θ error: "
        f"{error_before:.3f}° → {error_after:.3f}°"
    )


@needs_cuda
def test_refine_angles_dtype_mismatch_raises():
    """refine_angles takes both volume and sinogram; their dtypes must match."""
    N        = 32
    n_angles = 6
    angles   = make_angles(n_angles, device=DEVICE)
    vol      = phantom_2d(N, device=DEVICE).unsqueeze(0)             # fp32
    sino     = ft.forward_project(vol, angles).half() # fp16
    with pytest.raises(TypeError, match="must have the same dtype"):
        ft.refine_angles(vol, sino, angles_deg=angles)


# ── estimate_shifts_xcorr ─────────────────────────────────────────────────────


def test_estimate_shifts_xcorr_recovers_known_shifts(backend):
    """Replaces test_cpu.py::test_estimate_shifts_xcorr_recovers_known_shifts
    and test_cuda.py::test_estimate_shifts_xcorr_cuda."""
    N      = 64
    angles = make_angles(10, max_deg=60.0, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)   # [1, 64, 64]
    sino   = ft.forward_project(vol, angles)               # [10, 1, 64]

    true_shifts = torch.tensor([5.0, -3.0, 7.5, -2.5, 0.0,
                                 4.0, -6.0,  1.5, -4.5, 3.0])
    sino_shifted = ft.apply_shifts(sino, true_shifts.to(backend))

    est = ft.estimate_shifts_xcorr(vol, sino_shifted, angles)
    assert est.device.type == "cpu"
    err = (est - true_shifts).abs().max().item()
    assert err < 0.15, f"max shift error {err:.3f} px > 0.15 px tolerance"


def test_estimate_shifts_xcorr_returns_cpu():
    angles = make_angles(8)
    vol    = phantom_2d(32).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    est    = ft.estimate_shifts_xcorr(vol, sino, angles)
    assert est.device.type == "cpu"


def test_estimate_shifts_xcorr_max_shift():
    N      = 64
    angles = make_angles(5, max_deg=60.0)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    large  = torch.full((5,), 10.0)
    sino_shifted = ft.apply_shifts(sino, large)
    est = ft.estimate_shifts_xcorr(vol, sino_shifted, angles, max_shift=5.0)
    assert est.abs().max().item() <= 5.0 + 1e-6
