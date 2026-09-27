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


# ── R4: all views' candidates in one batch, bit-identical to the per-view loop ──


def _refine_per_view(volume, sinogram, angles_deg=None, euler_zyz_deg=None,
                     search_range_deg=3.0, n_steps=31):
    """The former refine_angles loop (one projection batch and host syncs per
    view, per axis in ZYZ mode), kept verbatim as the reference."""
    from fasttomo._operator import Geometry, _make_operator
    from fasttomo.algorithms._common import _REL_EPS
    from fasttomo.autograd import euler_zyz_to_rotmat

    def op(kind):
        return _make_operator(Geometry(kind, tuple(volume.shape), sinogram.shape[-1], n_steps),
                              volume.device, volume.dtype)
    if angles_deg is not None:
        refined = angles_deg.detach().cpu().clone()
        p = op("parallel")
        for j in range(sinogram.shape[0]):
            t = refined[j].item()
            c = torch.linspace(t - search_range_deg, t + search_range_deg, n_steps)
            projs = p.project(volume, c.to(sinogram.device))
            m = sinogram[j]
            res = (projs - m.unsqueeze(0)).abs().sum(dim=(-2, -1))
            refined[j] = c[(res / (m.abs().sum().clamp(min=_REL_EPS))).argmin()]
        return refined
    refined = euler_zyz_deg.detach().cpu().clone()
    p = op("rotmat")
    for j in range(sinogram.shape[0]):
        m = sinogram[j]
        norm = m.abs().sum().clamp(min=_REL_EPS)
        for axis in range(3):
            v = refined[j, axis].item()
            c = torch.linspace(v - search_range_deg, v + search_range_deg, n_steps)
            eb = refined[j].unsqueeze(0).expand(n_steps, -1).clone()
            eb[:, axis] = c
            projs = p.project(volume, euler_zyz_to_rotmat(eb.to(sinogram.device)))
            res = (projs - m.unsqueeze(0)).abs().sum(dim=(-2, -1))
            refined[j, axis] = c[res.div(norm).argmin()]
    return refined


def _refine_case(device, shape=(3, 21, 19), N_det=23, n=11, dtype=torch.float32):
    g = torch.Generator().manual_seed(n)
    Ny, Nz, Nx = shape
    vol = torch.zeros(shape)
    vol[:, Nz // 4:3 * Nz // 4, Nx // 3:2 * Nx // 3] = 1.0
    vol += 0.2 * torch.rand(shape, generator=g)
    true = torch.linspace(-62, 62, n) + 0.1 * torch.randn(n, generator=g)
    e_true = torch.stack([3 * torch.rand(n, generator=g) - 1.5, true,
                          3 * torch.rand(n, generator=g) - 1.5], 1)
    vol = vol.to(device)
    sino = ft.forward_project(vol, true.to(device), N_det=N_det).to(dtype)
    s3 = ft.forward_project_3d(vol, e_true.to(device), N_det=N_det).to(dtype)
    start = true + 0.8 * torch.randn(n, generator=g)
    e0 = e_true + 0.6 * torch.randn(n, 3, generator=g)
    return vol.to(dtype), sino, s3, start, e0


@pytest.mark.parametrize("chunk_bytes", [None, 1])
@pytest.mark.parametrize("mode", ["single", "zyz"])
def test_refine_angles_batched_is_bit_identical_to_per_view(backend, mode, chunk_bytes,
                                                            monkeypatch):
    """R4: one projection per call (per axis in ZYZ mode) gives exactly the
    per-view loop's result, also when the batch is split into chunks
    (chunk_bytes=1: one view per chunk) and for fp16 on CUDA."""
    import fasttomo.algorithms.alignment as al
    if chunk_bytes is not None:
        monkeypatch.setattr(al, "_REFINE_BATCH_BYTES", chunk_bytes)
    dtypes = [torch.float32] + ([torch.float16] if backend.type == "cuda" else [])
    for dtype in dtypes:
        for shape, N_det, n, steps in [((3, 21, 19), 23, 11, 9), ((1, 17, 13), 15, 7, 1)]:
            vol, sino, s3, start, e0 = _refine_case(backend, shape, N_det, n, dtype)
            for rng in (2.0, 0.37):
                if mode == "single":
                    got = ft.refine_angles(vol, sino, start.clone(),
                                           search_range_deg=rng, n_steps=steps)
                    want = _refine_per_view(vol, sino, angles_deg=start.clone(),
                                            search_range_deg=rng, n_steps=steps)
                else:
                    got = ft.refine_angles(vol, s3, euler_zyz_deg=e0.clone(),
                                           search_range_deg=rng, n_steps=steps)
                    want = _refine_per_view(vol, s3, euler_zyz_deg=e0.clone(),
                                            search_range_deg=rng, n_steps=steps)
                assert got.device.type == "cpu" and got.dtype == want.dtype
                assert torch.equal(got, want), (dtype, shape, rng)


@pytest.mark.parametrize("mode", ["single", "zyz"])
def test_refine_angles_projects_once_per_axis_without_autograd(backend, mode, monkeypatch):
    """One projection batch per call (single-axis) or per axis (ZYZ), made
    with autograd disabled; the volume's requires_grad does not matter."""
    import fasttomo.algorithms.alignment as al
    calls = []
    real = al._make_operator

    def spy(*a, **k):
        op = real(*a, **k)
        project = op.project

        def counted(volume, geom):
            calls.append((geom.shape[0], torch.is_grad_enabled()))
            return project(volume, geom)
        op.project = counted
        return op

    monkeypatch.setattr(al, "_make_operator", spy)
    vol, sino, s3, start, e0 = _refine_case(backend, n=13)
    vol = vol.clone().requires_grad_(True)
    if mode == "single":
        out = ft.refine_angles(vol, sino, start, n_steps=5)
        assert calls == [(13 * 5, False)]
    else:
        out = ft.refine_angles(vol, s3, euler_zyz_deg=e0, n_steps=5)
        assert calls == [(13 * 5, False)] * 3
    assert not out.requires_grad


@needs_cuda
@pytest.mark.parametrize("mode", ["single", "zyz"])
def test_refine_angles_host_syncs_do_not_scale_with_views(mode):
    """The former loop synchronised with the host 2 times per view (6 per
    view in ZYZ mode); the batched search does it a fixed number of times."""
    import warnings

    def count(n):
        vol, sino, s3, start, e0 = _refine_case(torch.device("cuda"), n=n)
        run = ((lambda: ft.refine_angles(vol, sino, start, n_steps=5)) if mode == "single"
               else (lambda: ft.refine_angles(vol, s3, euler_zyz_deg=e0, n_steps=5)))
        run()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                run()
        finally:
            torch.cuda.set_sync_debug_mode(0)
        return sum("synchroniz" in str(x.message) for x in w)

    few, many = count(5), count(29)
    assert few == many
    assert many <= (2 if mode == "single" else 6)


def test_refine_angles_zero_views_returns_empty(backend):
    """No views: an empty [0] / [0, 3] CPU tensor of the input's dtype, as
    the per-view loop returned (the batched search must not stack nothing)."""
    vol = torch.rand(2, 8, 8, device=backend)
    sino = torch.zeros(0, 2, 8, device=backend)
    out = ft.refine_angles(vol, sino, torch.zeros(0))
    assert out.shape == (0,) and out.dtype == torch.float32 and out.device.type == "cpu"
    out = ft.refine_angles(vol, sino, euler_zyz_deg=torch.zeros(0, 3, dtype=torch.float64))
    assert out.shape == (0, 3) and out.dtype == torch.float64 and out.device.type == "cpu"


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
