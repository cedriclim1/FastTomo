"""
GENFIRE and RESIRE (run_genfire, run_resire). Both algorithms are frozen:
their numerical output is pinned by tests/test_frozen_identity.py and the
RESIRE golden below; these tests cover behaviour and input handling.
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

# ── GENFIRE ───────────────────────────────────────────────────────────────────


def test_genfire_shape(backend):
    """Replaces test_cpu.py::test_genfire_shape and
    test_cuda.py::test_genfire_shape_cuda."""
    N      = 32
    angles = make_angles(20, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)        # [1, N, N]
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=10)
    assert rec.shape == vol.shape
    assert rec.device.type == backend.type


def test_genfire_positivity(backend):
    """positivity=True must eliminate all negative voxels.

    Replaces test_cpu.py::test_genfire_positivity and
    test_cuda.py::test_genfire_positivity_cuda."""
    N      = 32
    angles = make_angles(30, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=10, positivity=True)
    assert rec.min().item() >= -1e-6, f"negative voxels found: min={rec.min():.4e}"


def test_genfire_mask(backend):
    """With mask=True the corner voxels (outside inscribed cylinder) should be ~0.

    Replaces test_cpu.py::test_genfire_mask and test_cuda.py::test_genfire_mask_cuda."""
    N      = 32
    angles = make_angles(20, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
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


def test_genfire_reconstruction_quality(backend):
    """On noiseless data GENFIRE NMSE should be below 0.15.

    Replaces test_cpu.py::test_genfire_reconstruction_quality and
    test_cuda.py::test_genfire_reconstruction_quality_cuda."""
    N      = 64
    angles = make_angles(60, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=50,
                            positivity=True, mask=True)

    # normalised MSE (both normalised to [0,1])
    def nmse(a, b):
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        b = (b - b.min()) / (b.max() - b.min() + 1e-9)
        return ((a - b) ** 2).mean().item()

    err = nmse(rec, vol)
    assert err < 0.15, f"GENFIRE NMSE too high ({backend}): {err:.4f}"


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


def test_genfire_no_resolution_extension():
    """Algorithm must run without resolution extension and return correct shape."""
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=10,
                            resolution_extension=False)
    assert rec.shape == vol.shape


DEVICE = torch.device("cuda")


@needs_cuda
def test_run_genfire_rejects_fp16():
    """GENFIRE gridding kernel is fp32-only — calling with fp16 sinogram
    must raise an explicit TypeError instead of crashing inside the
    gridding kernel or producing a silent fp32 cast."""
    N        = 32
    n_angles = 20
    angles   = make_angles(n_angles, device=DEVICE)
    sino     = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles).half()
    with pytest.raises(TypeError, match=r"torch\.float32 sinogram"):
        ft.run_genfire(sino, angles, Nz=N, n_iter=2)


# ── RESIRE (CUDA-only) ────────────────────────────────────────────────────────


def test_resire_requires_cuda():
    """run_resire is CUDA-only — must raise RuntimeError on CPU tensors.

    Replaces test_cpu.py::test_resire_requires_cuda and the identical CPU
    check test_cuda.py::test_resire_requires_cuda (it only ran with a GPU
    present)."""
    N      = 32
    angles = make_angles(20)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)
    with pytest.raises(RuntimeError, match="CUDA"):
        ft.run_resire(sino, angles, n_iter=1, Nz=N)


@needs_cuda
def test_resire_shape_cuda():
    N      = 32
    angles = make_angles(20, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_resire(sino, angles, n_iter=10, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == "cuda"


@needs_cuda
def test_resire_positivity_cuda():
    N      = 32
    angles = make_angles(30, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_resire(sino, angles, n_iter=10, Nz=N, positivity=True)
    assert rec.min().item() >= -1e-6


@needs_cuda
def test_resire_mask_cuda():
    """With mask=True the corner voxels (outside inscribed cylinder) should be ~0."""
    N      = 32
    angles = make_angles(20, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_resire(sino, angles, n_iter=10, Nz=N, mask=True)
    corners = torch.stack([
        rec[0,  0,  0], rec[0,  0, -1],
        rec[0, -1,  0], rec[0, -1, -1],
    ])
    assert corners.abs().max().item() < 1e-6


@needs_cuda
def test_resire_converges_cuda():
    """Reconstruction MSE should decrease as iterations increase (GPU path)."""
    N      = 64
    angles = make_angles(60, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    errors = []
    for n in (1, 10, 30, 80):
        rec = ft.run_resire(sino, angles, n_iter=n, Nz=N, positivity=False)
        errors.append(((rec - vol) ** 2).mean().item())

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] * 1.05, (
            f"RESIRE did not converge (GPU): MSE history = {errors}"
        )


@needs_cuda
def test_resire_custom_support_cuda():
    """Explicit support tensor should zero voxels outside it."""
    N       = 32
    angles  = make_angles(20, device=DEVICE)
    vol     = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino    = ft.forward_project(vol, angles)
    support = ft.cylindrical_mask(N, N, device=DEVICE).unsqueeze(0)
    rec     = ft.run_resire(sino, angles, n_iter=5, Nz=N, support_mask=support)
    corners = torch.stack([rec[0, 0, 0], rec[0, 0, -1], rec[0, -1, 0], rec[0, -1, -1]])
    assert corners.abs().max().item() < 1e-6


@needs_cuda
def test_run_resire_rejects_fp16():
    """RESIRE FP/BP kernels are fp32-only — explicit TypeError on fp16."""
    N        = 32
    n_angles = 20
    angles   = make_angles(n_angles, device=DEVICE)
    sino     = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles).half()
    with pytest.raises(TypeError, match=r"torch\.float32 sinogram"):
        ft.run_resire(sino, angles, Nz=N, n_iter=2)


# RESIRE is frozen: the guard only adds checks before the unchanged kernel
# calls. The golden file was generated by `_resire_cases()` running against
# origin/main's fasttomo/ (98fa473). The RESIRE forward kernel scatters with
# fp32 atomicAdd, so for general angles it is not bit-reproducible even
# main-vs-main; those outputs are compared with torch.equal where the
# arithmetic is exact (identity rotation, dyadic volume: every partial sum
# is exactly representable, so accumulation order cannot matter) and with a
# tolerance at the atomics' noise level otherwise. Every back-projection
# (a gather, no atomics) is compared with torch.equal.
RESIRE_GOLDEN = __import__("pathlib").Path(__file__).parent / "data" / "resire_golden_main.pt"


RESIRE_EXACT = ("bp", "bp_nz", "bp_default", "fp_grad", "fp_exact", "fp_exact_ndet")


RESIRE_NOISY = ("fp", "fp_ndet", "bp_grad_sq", "run", "run_loss", "run_euler")


# The golden file was made on an RTX PRO 6000 Blackwell (sm_120) with torch
# 2.12.1+cu130 and nvcc 13.0. Another GPU architecture, compiler or torch
# version may round the rotation matrices or contract FMAs differently, so
# the torch.equal comparison of the RESIRE_EXACT keys only holds in that
# environment; elsewhere they are compared at the RESIRE_NOISY tolerance.
# To regenerate: check out origin/main's fasttomo/ at 98fa473 and save
# `_resire_cases()` with torch.save.
RESIRE_GOLDEN_ENV = {"capability": (12, 0), "torch": "2.12."}


def _resire_golden_env_matches() -> bool:
    return (torch.cuda.get_device_capability() == RESIRE_GOLDEN_ENV["capability"]
            and torch.__version__.startswith(RESIRE_GOLDEN_ENV["torch"]))


def _resire_cases() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    Ny, Nz, Nx, n = 5, 18, 22, 9
    vol = torch.rand(Ny, Nz, Nx, generator=g).cuda()
    eul = torch.stack([torch.linspace(-6, 6, n), torch.linspace(-60, 60, n),
                       torch.linspace(3, -3, n)], 1).cuda()
    sino = torch.rand(n, Ny, 26, generator=g).cuda()
    sino_sq = torch.rand(n, Ny, Nx, generator=g).cuda()
    ang = torch.linspace(-60, 60, n).cuda()
    out = {}
    out["fp"] = ft.forward_project_resire(vol, eul)
    out["fp_ndet"] = ft.forward_project_resire(vol, eul, N_det=26)
    out["bp"] = ft.back_project_resire(sino, eul, Nz=Nz)
    out["bp_nz"] = ft.back_project_resire(sino_sq, eul, Nz=11)
    out["bp_default"] = ft.back_project_resire(sino, eul)
    v = vol.clone().requires_grad_(True)
    (ft.forward_project_resire(v, eul) * sino_sq).sum().backward()
    out["fp_grad"] = v.grad
    s = sino_sq.clone().requires_grad_(True)
    (ft.back_project_resire(s, eul, Nz=Nz) * vol).sum().backward()
    out["bp_grad_sq"] = s.grad            # the RESIRE forward: atomics noise
    dyadic = torch.randint(0, 8, (Ny, Nz, Nx), generator=g).float().cuda() / 4
    ident = torch.zeros(4, 3, device="cuda")
    out["fp_exact"] = ft.forward_project_resire(dyadic, ident)
    out["fp_exact_ndet"] = ft.forward_project_resire(dyadic, ident, N_det=Nx + 4)
    sq = ft.forward_project_resire(torch.rand(Ny, Nx, Nx, generator=g).cuda(), eul)
    out["run"], out["run_loss"] = ft.run_resire(sq, ang, n_iter=4, Nz=Nx, return_loss=True)
    out["run_euler"] = ft.run_resire(sq, ang, n_iter=3, Nz=Nx, euler_zyz_deg=eul.cpu(),
                                     mask=True, sparsity=0.001)
    return {k: t.detach().cpu() for k, t in out.items()}


@needs_cuda
def test_resire_outputs_identical_to_main():
    golden = torch.load(RESIRE_GOLDEN)
    now = _resire_cases()
    assert set(golden) == set(RESIRE_EXACT) | set(RESIRE_NOISY) == set(now)
    exact_env = _resire_golden_env_matches()
    for k in RESIRE_EXACT:
        if exact_env:
            assert torch.equal(now[k], golden[k]), k
        else:
            torch.testing.assert_close(now[k], golden[k], rtol=1e-5, atol=1e-5, msg=k)
    for k in RESIRE_NOISY:
        assert now[k].shape == golden[k].shape
        torch.testing.assert_close(now[k], golden[k], rtol=1e-5, atol=1e-5, msg=k)


@needs_cuda
def test_run_resire_rejects_mismatched_euler():
    """A 1-row Euler table used to broadcast the forward projection over the
    whole sinogram in the residual; now it raises before iterating."""
    sino = torch.rand(9, 4, 12, device="cuda")
    ang = torch.linspace(-60, 60, 9, device="cuda")
    with pytest.raises(ValueError, match=r"euler_zyz_deg must have shape \(9, 3\)"):
        ft.run_resire(sino, ang, n_iter=1, euler_zyz_deg=torch.zeros(1, 3))
