"""
Projection primitives: forward_project / back_project (single-axis),
forward_project_3d / back_project_3d (rotation matrices), the RESIRE pair,
their autograd behaviour, dtypes, N_det != Nx geometry, zero-size inputs
and the detector-plane shift operators.

CPU<->CUDA and cached<->uncached agreement lives in test_parity.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from conftest import (
    make_angles,
    needs_cuda,
    phantom_2d,
    phantom_3d,
    rel_err,
    rel_l2,
)

import fasttomo as ft

# ── shapes and basic properties ───────────────────────────────────────────────


def test_forward_back_shapes(backend):
    """Replaces test_cpu.py::test_forward_back_shapes and
    test_cuda.py::test_forward_back_shapes_cuda."""
    N      = 32
    angles = make_angles(20, device=backend)
    vol    = torch.ones(1, N, N, device=backend)

    sino = ft.forward_project(vol, angles)
    assert sino.device.type == backend.type
    assert sino.shape == (len(angles), 1, N)

    rec = ft.back_project(sino, angles, Nz=N)
    assert rec.device.type == backend.type
    assert rec.shape == (1, N, N)


def test_forward_project_zero_volume(backend):
    """Replaces test_cpu.py::test_forward_project_zero_volume and
    test_cuda.py::test_forward_project_zero_volume_cuda."""
    N      = 32
    angles = make_angles(10, device=backend)
    vol    = torch.zeros(1, N, N, device=backend)
    sino   = ft.forward_project(vol, angles)
    assert sino.abs().max().item() == 0.0


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


DEVICE = torch.device("cuda")


@needs_cuda
def test_back_project_self_consistent_cuda():
    """
    CUDA back-projection is self-consistent: running it twice on the same input
    gives the same result (determinism check).

    Note: CPU uses scatter-based (strict adjoint) and CUDA uses gather-based
    (voxel-driven). They are intentionally different algorithms and are not
    expected to match numerically.
    """
    N      = 64
    angles = make_angles(30, device=DEVICE)
    sino   = torch.randn(len(angles), 1, N, device=DEVICE)

    vol1 = ft.back_project(sino, angles, Nz=N)
    vol2 = ft.back_project(sino, angles, Nz=N)

    diff = (vol1 - vol2).abs().max().item()
    assert diff == 0.0, f"CUDA back-projection is not deterministic: max_diff={diff:.2e}"


# ── rotation-matrix (3D) projectors ───────────────────────────────────────────


def make_euler_pure_y(angles_deg: torch.Tensor) -> torch.Tensor:
    """Wrap scalar tilt angles as ZYZ triples (φ=0, θ=angle, ψ=0)."""
    n = len(angles_deg)
    euler = torch.zeros(n, 3)
    euler[:, 1] = angles_deg.cpu()
    return euler


@needs_cuda
def test_fp3d_shape_cuda():
    N      = 32
    angles = make_angles(20, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0).to(DEVICE)
    sino   = ft.forward_project_3d(vol, euler)
    assert sino.shape == (len(angles), 1, N)
    assert sino.device.type == "cuda"


@needs_cuda
def test_bp3d_shape_cuda():
    N      = 32
    angles = make_angles(20, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0).to(DEVICE)
    sino   = ft.forward_project_3d(vol, euler)
    rec    = ft.back_project_3d(sino, euler, Nz=N)
    assert rec.shape == (1, N, N)
    assert rec.device.type == "cuda"


def test_fp3d_matches_scalar_fp(backend):
    """With φ=ψ=0, forward_project_3d must match the scalar-axis FP to 1e-2
    absolute. The small residual comes from the 3D diagonal ray-length
    being slightly longer than the 2D one, so the very last ray step can
    fall just past the volume face.

    Replaces test_cpu.py::test_fp3d_matches_scalar_fp_cpu and
    test_cuda.py::test_fp3d_matches_scalar_fp (num_threads is ignored on
    CUDA)."""
    N      = 64
    angles = make_angles(30, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)

    sino_2d = ft.forward_project(vol, angles)

    euler = torch.zeros(len(angles), 3, dtype=torch.float32)
    euler[:, 1] = angles.cpu()
    euler = euler.to(backend)
    sino_3d = ft.forward_project_3d(vol, euler, num_threads=4)

    diff = (sino_2d - sino_3d).abs()
    assert diff.max().item() < 1e-2, (
        f"3D FP (pure-Y) disagrees with scalar FP: max_diff={diff.max():.2e}"
    )


def test_fp3d_adjoint(backend):
    """<Ax, y> ≈ <x, Aᵀ y> for the 3D pair (approximate adjoint), random
    inputs with pure-Y Euler angles. The texture-interpolation kernels are
    only approximately adjoint (~65 % relative error with random inputs,
    consistent with the 2D CUDA pair); SIRT convergence
    (test_sirt_3d_converges_cuda) is the practical check.

    Replaces test_cpu.py::test_fp3d_adjoint_cpu and
    test_cuda.py::test_fp3d_bp3d_adjoint (now seeded on both backends)."""
    N      = 32
    angles = make_angles(20)
    euler  = torch.zeros(len(angles), 3, dtype=torch.float32)
    euler[:, 1] = angles
    euler = euler.to(backend)

    torch.manual_seed(0)
    x = torch.randn(1, N, N, device=backend)
    y = torch.randn(len(angles), 1, N, device=backend)

    Ax  = ft.forward_project_3d(x, euler, num_threads=4)
    ATy = ft.back_project_3d(y, euler, Nz=N, num_threads=4)

    lhs = (Ax * y).sum().item()
    rhs = (x * ATy).sum().item()
    rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-12)
    assert rel_err < 0.75, (
        f"3D adjoint test failed: <Ax,y>={lhs:.4f} <x,Aᵀy>={rhs:.4f} "
        f"rel_err={rel_err:.2e}"
    )


def test_fp3d_threading_consistency_cpu():
    """FP output must be byte-identical across thread counts (no atomics)."""
    N      = 24
    Ny     = 3
    torch.manual_seed(0)
    angles = make_angles(20, max_deg=60.0)
    euler  = torch.zeros(len(angles), 3, dtype=torch.float32)
    euler[:, 1] = angles
    vol    = torch.randn(Ny, N, N)

    sino_serial = ft.forward_project_3d(vol, euler, num_threads=1)
    sino_par    = ft.forward_project_3d(vol, euler, num_threads=8)

    assert torch.equal(sino_serial, sino_par), (
        "FP3D output differs between num_threads=1 and num_threads=8 — "
        f"max-diff={(sino_serial - sino_par).abs().max().item():.3e}"
    )


def test_fp3d_default_is_serial(monkeypatch):
    """With no num_threads kwarg and no OMP_NUM_THREADS, kernel runs serial.

    We can't observe thread count directly from Python, but we can verify
    the resolution helper: it must return 1 in this configuration."""
    from fasttomo.autograd import _resolve_num_threads
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert _resolve_num_threads(None) == 1
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert _resolve_num_threads(None) == 8
    assert _resolve_num_threads(2) == 2   # explicit kwarg wins over env


# ── N_det != Nx and zero-size inputs ──────────────────────────────────────────


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


def _blob(Ny: int, Nz: int, Nx: int, device, c=(0.1, -0.2), s=4.0) -> torch.Tensor:
    """Smooth Gaussian blob, constant along Y. Smooth inputs keep the CUDA
    texture back-projection close to the exact adjoint (~1 % instead of
    15-180 % for white noise), so the inner-product test is meaningful."""
    z = torch.arange(Nz, device=device) - (Nz - 1) / 2
    x = torch.arange(Nx, device=device) - (Nx - 1) / 2
    g = torch.exp(-((z[:, None] - c[0] * Nz) ** 2 + (x[None, :] - c[1] * Nx) ** 2) / (2 * s * s))
    return g.expand(Ny, Nz, Nx).contiguous()


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
        a_p = ang.detach().clone()
        a_p[i] += h
        a_m = ang.detach().clone()
        a_m[i] -= h
        lp = (ft.forward_project(x, a_p, N_det=N_det) * w).double().sum()
        lm = (ft.forward_project(x, a_m, N_det=N_det) * w).double().sum()
        fd.append(float((lp - lm) / (2 * h)))
    fd = torch.tensor(fd)
    assert rel_l2(ang.grad, fd) < 2e-2, (ang.grad, fd)


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


def _zero_size_cases(dev):
    from fasttomo.autograd import _grad_angle_raw, _grad_rotmat_raw
    a = torch.linspace(-30, 30, 5, device=dev)
    e = torch.zeros(5, 3, device=dev)
    mats = torch.eye(3, device=dev).expand(5, 3, 3)
    v = torch.rand(2, 6, 6, device=dev)
    s = torch.rand(5, 2, 6, device=dev)
    return {
        "fp N_det=0": lambda: ft.forward_project(v, a, N_det=0),
        "fp no angles": lambda: ft.forward_project(v, a[:0]),
        "fp Ny=0": lambda: ft.forward_project(v[:0], a),
        "bp Nz=0": lambda: ft.back_project(s, a, Nz=0),
        "bp Nx=0": lambda: ft.back_project(s, a, Nx=0),
        "bp N_det=0": lambda: ft.back_project(s[:, :, :0], a),
        "fp3d N_det=0": lambda: ft.forward_project_3d(v, e, N_det=0),
        "fp3d no angles": lambda: ft.forward_project_3d(v, e[:0]),
        "bp3d Nz=0": lambda: ft.back_project_3d(s, e, Nz=0),
        "bp3d Ny=0": lambda: ft.back_project_3d(s[:, :0], e),
        "grad_rotmat": lambda: _grad_rotmat_raw(v, s[:, :, :0], mats),
        "grad_angle": lambda: _grad_angle_raw(v, s[:, :, :0], a),
    }


@pytest.mark.parametrize("case", list(_zero_size_cases("cpu")))
def test_zero_size_raises_value_error(backend, case):
    """Before: CPU returned empty tensors while CUDA raised a launch
    RuntimeError or a tc_create ValueError. Now both raise the same
    ValueError from the shared launch helper."""
    with pytest.raises(ValueError, match="zero-size dimension"):
        _zero_size_cases(backend)[case]()


# ── autograd ──────────────────────────────────────────────────────────────────


def test_autograd_grad_flow_fp(backend):
    """forward_project propagates gradients into the volume tensor.

    Replaces test_cpu.py::test_autograd_grad_flow_fp_cpu and
    test_cuda.py::test_autograd_grad_flow_fp_cuda."""
    N      = 32
    angles = make_angles(20, device=backend)
    vol    = torch.randn(1, N, N, device=backend, requires_grad=True)

    sino = ft.forward_project(vol, angles)
    loss = (sino ** 2).mean()
    loss.backward()

    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert vol.grad.device.type == backend.type
    assert torch.isfinite(vol.grad).all().item()
    assert vol.grad.abs().sum().item() > 0.0


def test_autograd_grad_flow_bp(backend):
    """back_project propagates gradients into the sinogram tensor.

    Replaces test_cpu.py::test_autograd_grad_flow_bp_cpu and
    test_cuda.py::test_autograd_grad_flow_bp_cuda (the CUDA case gains the
    non-zero gradient check)."""
    N      = 32
    angles = make_angles(20, device=backend)
    sino   = torch.randn(len(angles), 1, N, device=backend, requires_grad=True)

    rec  = ft.back_project(sino, angles, Nz=N)
    loss = (rec ** 2).mean()
    loss.backward()

    assert sino.grad is not None
    assert sino.grad.shape == sino.shape
    assert sino.grad.device.type == backend.type
    assert torch.isfinite(sino.grad).all().item()
    assert sino.grad.abs().sum().item() > 0.0


def test_autograd_no_grad_regression(backend):
    """
    With requires_grad=False, forward_project must produce identical output
    to a direct kernel call — Function.apply short-circuits, no behaviour
    change vs the pre-autograd implementation.

    Replaces test_cpu.py::test_autograd_no_grad_regression_cpu and
    test_cuda.py::test_autograd_no_grad_regression_cuda.
    """
    N      = 32
    angles = make_angles(20, device=backend)
    vol    = torch.randn(1, N, N, device=backend)  # no requires_grad

    sino = ft.forward_project(vol, angles)
    assert not sino.requires_grad

    # Determinism: two calls must give identical bytes
    sino2 = ft.forward_project(vol, angles)
    assert torch.equal(sino, sino2)


def test_autograd_grad_through_squeeze_cpu():
    """
    A 2-D volume input (no Ny axis) should still receive a gradient of the
    same 2-D shape. Verifies the unsqueeze(0) hop is differentiable.
    """
    N      = 32
    angles = make_angles(20)
    vol_2d = torch.randn(N, N, requires_grad=True)

    sino = ft.forward_project(vol_2d, angles)
    sino.sum().backward()

    assert vol_2d.grad is not None
    assert vol_2d.grad.shape == (N, N)


@needs_cuda
def test_autograd_grad_flow_fp3d_cuda():
    N      = 32
    angles = make_angles(20, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = torch.randn(1, N, N, device=DEVICE, requires_grad=True)

    sino = ft.forward_project_3d(vol, euler)
    sino.pow(2).mean().backward()

    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert torch.isfinite(vol.grad).all().item()
    assert vol.grad.abs().sum().item() > 0.0


@needs_cuda
def test_autograd_grad_flow_bp3d_cuda():
    N      = 32
    angles = make_angles(20, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    sino   = torch.randn(len(angles), 1, N, device=DEVICE, requires_grad=True)

    rec = ft.back_project_3d(sino, euler, Nz=N)
    rec.pow(2).mean().backward()

    assert sino.grad is not None
    assert sino.grad.shape == sino.shape
    assert torch.isfinite(sino.grad).all().item()


@needs_cuda
def test_autograd_grad_flow_resire_cuda():
    N      = 32
    angles = make_angles(20, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = torch.randn(1, N, N, device=DEVICE, requires_grad=True)

    sino = ft.forward_project_resire(vol, euler)
    sino.pow(2).mean().backward()

    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert torch.isfinite(vol.grad).all().item()
    assert vol.grad.abs().sum().item() > 0.0


@needs_cuda
def test_autograd_double_apply_cuda():
    """
    Calling forward_project twice on the same requires_grad volume, summing
    the losses, and backward()ing must work — confirms Function instances
    do not share ctx state.
    """
    N      = 32
    angles = make_angles(20, device=DEVICE)
    vol    = torch.randn(1, N, N, device=DEVICE, requires_grad=True)

    sino_a = ft.forward_project(vol, angles)
    sino_b = ft.forward_project(vol, angles)
    loss = sino_a.pow(2).mean() + sino_b.pow(2).mean()
    loss.backward()

    assert vol.grad is not None
    assert torch.isfinite(vol.grad).all().item()


# ── dtypes (fp16 in -> fp16 out, bf16 raises) ─────────────────────────────────


_PROJ_OPS = ["forward_project", "back_project", "forward_project_3d", "back_project_3d"]


@pytest.mark.parametrize("op", _PROJ_OPS)
def test_projection_preserves_dtype(op, backend, dtype):
    """Each projection op returns its input dtype on both backends (fp16 in,
    fp16 out; the CPU kernels upcast internally) and the fp16 result matches
    the fp32 one to the fp16 texture floor (1 %). Skips per conftest.DTYPE_CAPS."""
    N, n = 16, 8
    angles = make_angles(n, device=backend)
    euler = torch.zeros(n, 3, device=backend)
    euler[:, 1] = angles
    geom = angles if op in ("forward_project", "back_project") else euler
    fn = getattr(ft, op)
    g = torch.Generator().manual_seed(0)
    if op.startswith("forward"):
        x, kw = phantom_3d(N, device=backend)[6:8] + 0.1 * torch.rand(2, N, N, generator=g).to(backend), {}
    else:
        x, kw = torch.rand(n, 2, N, generator=g).to(backend), {"Nz": N}
    out = fn(x.to(dtype), geom, **kw)
    ref = fn(x, geom, **kw)
    assert out.dtype == dtype
    assert out.device.type == backend.type
    assert rel_err(out, ref) < 1e-2


@needs_cuda
def test_forward_project_fp16_matches_fp32():
    """fp16 forward projection must match fp32 to within 1% relative."""
    N      = 64
    angles = make_angles(30, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino_fp32 = ft.forward_project(vol, angles)
    sino_fp16 = ft.forward_project(vol.half(), angles)
    assert sino_fp16.dtype == torch.float16
    assert rel_err(sino_fp16, sino_fp32) < 1e-2


@needs_cuda
def test_back_project_fp16_matches_fp32():
    """fp16 back-projection must match fp32 to within 1% relative."""
    N      = 64
    angles = make_angles(30, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec_fp32 = ft.back_project(sino, angles, Nz=N)
    rec_fp16 = ft.back_project(sino.half(), angles, Nz=N)
    assert rec_fp16.dtype == torch.float16
    assert rel_err(rec_fp16, rec_fp32) < 1e-2


@needs_cuda
def test_fp3d_fp16_matches_fp32():
    N      = 32
    angles = make_angles(20, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino_fp32 = ft.forward_project_3d(vol, euler)
    sino_fp16 = ft.forward_project_3d(vol.half(), euler)
    assert sino_fp16.dtype == torch.float16
    assert rel_err(sino_fp16, sino_fp32) < 1e-2


@needs_cuda
def test_bp3d_fp16_matches_fp32():
    N      = 32
    angles = make_angles(20, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project_3d(vol, euler)
    rec_fp32 = ft.back_project_3d(sino, euler, Nz=N)
    rec_fp16 = ft.back_project_3d(sino.half(), euler, Nz=N)
    assert rec_fp16.dtype == torch.float16
    assert rel_err(rec_fp16, rec_fp32) < 1e-2


@needs_cuda
def test_bf16_raises_on_forward_project():
    N      = 16
    angles = make_angles(5, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0).bfloat16()
    with pytest.raises(TypeError, match="bfloat16"):
        ft.forward_project(vol, angles)


@needs_cuda
def test_bf16_raises_on_back_project():
    N      = 16
    angles = make_angles(5, device=DEVICE)
    sino   = torch.zeros(angles.numel(), 1, N, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="bfloat16"):
        ft.back_project(sino, angles, Nz=N)


@needs_cuda
def test_resire_rejects_fp16():
    """RESIRE projector ops are fp32-only (out of scope for fp16 path)."""
    N      = 16
    angles = make_angles(5, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0).half()
    with pytest.raises(TypeError, match="fp32-only"):
        ft.forward_project_resire(vol, euler)


# ── RESIRE projector pair (CUDA-only, frozen kernels) ─────────────────────────


@needs_cuda
def test_resire_projector_shapes():
    N      = 32
    angles = make_angles(20, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0).to(DEVICE)

    sino = ft.forward_project_resire(vol, euler)
    assert sino.shape == (len(angles), 1, N), f"unexpected FP shape {sino.shape}"
    assert sino.device.type == "cuda"

    rec = ft.back_project_resire(sino, euler, Nz=N)
    assert rec.shape == (1, N, N), f"unexpected BP shape {rec.shape}"
    assert rec.device.type == "cuda"


@needs_cuda
def test_resire_projector_nonzero():
    """FP of a point source must produce a non-zero sinogram; BP of all-ones must produce a non-zero volume."""
    N      = 32
    angles = make_angles(20, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)

    vol = torch.zeros(1, N, N, device=DEVICE)
    vol[0, N//2, N//2] = 1.0
    sino = ft.forward_project_resire(vol, euler)
    assert sino.max().item() > 0.0, "FP of point source gave all-zeros"

    ones_sino = torch.ones(len(angles), 1, N, device=DEVICE)
    rec = ft.back_project_resire(ones_sino, euler, Nz=N)
    assert rec.max().item() > 0.0, "BP of all-ones gave all-zeros"


@needs_cuda
def test_resire_projector_adjoint():
    """<FP(x), y> ≈ <x, BP(y)> — approximate adjoint for the RESIRE projector pair."""
    N      = 32
    angles = make_angles(20, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)

    x = torch.randn(1, N, N, device=DEVICE)
    y = torch.randn(len(angles), 1, N, device=DEVICE)

    Ax  = ft.forward_project_resire(x, euler)
    ATy = ft.back_project_resire(y, euler, Nz=N)

    lhs = (Ax * y).sum().item()
    rhs = (x * ATy).sum().item()
    rel_err = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12)
    assert rel_err < 0.15, (
        f"RESIRE projector adjoint error too large: rel_err={rel_err:.3f} "
        f"(lhs={lhs:.4f}, rhs={rhs:.4f})"
    )


@needs_cuda
@pytest.mark.parametrize("which,kwargs,match", [
    ("fp", dict(euler=(9, 2)), r"\[n_angles, 3\]"),
    ("fp", dict(euler=(9, 9)), r"\[n_angles, 3\]"),
    ("fp", dict(euler=(9,)), r"\[n_angles, 3\]"),
    ("fp", dict(euler=(0, 3)), "no views"),
    ("fp", dict(data=(0, 6, 6)), "zero-size"),
    ("fp", dict(N_det=0), "N_det must be positive"),
    ("bp", dict(euler=(8, 3)), "9 views but euler_zyz_deg has 8"),
    ("bp", dict(euler=(10, 3)), "9 views but euler_zyz_deg has 10"),
    ("bp", dict(euler=(9, 4)), r"\[n_angles, 3\]"),
    ("bp", dict(data=(9, 0, 6)), "zero-size"),
    ("bp", dict(Nz=0), "Nz and Nx must be positive"),
])
def test_resire_raw_wrappers_reject_bad_shapes(which, kwargs, match):
    """Mismatched sizes raise ValueError before any kernel launch."""
    eul = torch.zeros(kwargs.get("euler", (9, 3)), device="cuda")
    if which == "fp":
        vol = torch.rand(kwargs.get("data", (4, 6, 6)), device="cuda")
        with pytest.raises(ValueError, match=match):
            ft.forward_project_resire(vol, eul, N_det=kwargs.get("N_det", -1))
    else:
        sino = torch.rand(kwargs.get("data", (9, 4, 6)), device="cuda")
        with pytest.raises(ValueError, match=match):
            ft.back_project_resire(sino, eul, Nz=kwargs.get("Nz", -1))


@needs_cuda
@pytest.mark.parametrize("Nx,N_det", [(22, 26), (26, 22)])
def test_resire_autograd_nx_ne_ndet(Nx, N_det):
    """Same fix as the texture pair: forward_project_resire with N_det != Nx
    has a working backward (it raised 'invalid gradient' before), equal to
    back_project_resire(Nx=Nx); back_project_resire(Nx=...) of the wider
    volume centre-cropped equals the narrow one (gather, exact)."""
    g = torch.Generator().manual_seed(12)
    Ny, Nz, n = 4, 14, 7
    eul = torch.stack([torch.zeros(n), torch.linspace(-60, 60, n), torch.zeros(n)], 1).cuda()
    x = torch.rand(Ny, Nz, Nx, generator=g).cuda().requires_grad_(True)
    w = torch.rand(n, Ny, N_det, generator=g).cuda()
    (ft.forward_project_resire(x, eul, N_det=N_det) * w).sum().backward()
    assert x.grad.shape == x.shape
    assert torch.equal(x.grad, ft.back_project_resire(w, eul, Nz=Nz, Nx=Nx))
    s = w.clone().requires_grad_(True)
    (ft.back_project_resire(s, eul, Nz=Nz, Nx=Nx) * x.detach()).sum().backward()
    assert s.grad.shape == w.shape
    narrow, wide = min(Nx, N_det), max(Nx, N_det)
    off = (wide - narrow) // 2
    small = ft.back_project_resire(w, eul, Nz=Nz, Nx=narrow)
    big = ft.back_project_resire(w, eul, Nz=Nz, Nx=wide)[:, :, off:off + narrow]
    assert torch.equal(small, big)


# ── detector-plane shifts ─────────────────────────────────────────────────────


def test_apply_shifts_shape(backend):
    """Replaces test_cpu.py::test_apply_shifts_shape and
    test_cuda.py::test_apply_shifts_cuda."""
    angles = make_angles(10, device=backend)
    vol    = phantom_2d(32, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    out    = ft.apply_shifts(sino, torch.zeros(10))
    assert out.shape == sino.shape
    assert out.device.type == backend.type


def test_apply_shifts_zero_shift(backend):
    """Was CPU-only (test_cpu.py::test_apply_shifts_zero_shift)."""
    angles = make_angles(10, device=backend)
    vol    = phantom_2d(32, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    out    = ft.apply_shifts(sino, torch.zeros(10))
    assert (out - sino).abs().max().item() < 1e-5


def test_apply_shifts_integer_shift(backend):
    """Was CPU-only (test_cpu.py::test_apply_shifts_integer_shift)."""
    angles = make_angles(10, device=backend)
    vol    = phantom_2d(64, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    s      = 8
    shifts = torch.full((10,), float(s))
    out    = ft.apply_shifts(sino, shifts)
    rolled = torch.roll(sino, s, dims=-1)
    # Fourier phase shift and torch.roll agree up to boundary ringing (1e-4)
    assert (out - rolled).abs().max().item() < 1e-4


def _shift_ref(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """fp64 reference: out[y, x] = bilinear sample of img at (y - sy, x - sx)
    with zeros outside [0, N-1] (grid_sample align_corners=True, padding
    zeros). Positive shifts move content toward larger indices."""
    Ny, Nx = img.shape
    out = np.zeros_like(img, dtype=np.float64)
    for y in range(Ny):
        for x in range(Nx):
            fy, fx = y - sy, x - sx
            y0, x0 = math.floor(fy), math.floor(fx)
            wy, wx = fy - y0, fx - x0
            acc = 0.0
            for dy, wyy in ((0, 1 - wy), (1, wy)):
                for dx, wxx in ((0, 1 - wx), (1, wx)):
                    yy, xx = y0 + dy, x0 + dx
                    if 0 <= yy < Ny and 0 <= xx < Nx:
                        acc += wyy * wxx * img[yy, xx]
            out[y, x] = acc
    return out


@pytest.mark.parametrize("shifts", [
    [(3.0, 0.0), (-2.0, 0.0)],        # integer, detector axis
    [(0.0, 2.0), (0.0, -1.0)],        # integer, tilt axis
    [(0.25, 0.0), (-1.75, 0.0)],      # sub-pixel, detector axis
    [(0.0, 0.5), (0.0, -0.3)],        # sub-pixel, tilt axis
    [(1.4, -0.6), (-0.2, 2.7)],       # both axes
])
def test_apply_shifts_2d_matches_reference(backend, shifts):
    g = torch.Generator().manual_seed(6)
    sino = torch.rand(2, 7, 11, generator=g)
    sh = torch.tensor(shifts, dtype=torch.float32)
    out = ft.apply_shifts_2d(sino.to(backend), sh.to(backend))
    assert out.shape == sino.shape and out.dtype == torch.float32
    ref = np.stack([_shift_ref(sino[i].double().numpy(), *shifts[i]) for i in range(2)])
    torch.testing.assert_close(out.cpu().double(), torch.from_numpy(ref), rtol=1e-5, atol=1e-5)


def test_apply_shifts_2d_integer_is_roll_with_zero_fill(backend):
    """Integer shifts are an exact translation with zero fill."""
    sino = torch.arange(2 * 5 * 9, dtype=torch.float32).reshape(2, 5, 9).to(backend)
    out = ft.apply_shifts_2d(sino, torch.tensor([[2.0, 1.0], [-3.0, 0.0]], device=backend))
    exp = torch.zeros_like(sino)
    exp[0, 1:, 2:] = sino[0, :-1, :-2]
    exp[1, :, :-3] = sino[1, :, 3:]
    torch.testing.assert_close(out, exp, rtol=0, atol=1e-4)


def test_apply_shifts_2d_zero_is_identity_and_grad(backend):
    """Zero shift is the identity; the shifts gradient matches central
    finite differences at a sub-pixel point (grid_sample autograd)."""
    g = torch.Generator().manual_seed(7)
    sino = torch.rand(3, 6, 10, generator=g, dtype=torch.float64).float().to(backend)
    torch.testing.assert_close(ft.apply_shifts_2d(sino, torch.zeros(3, 2, device=backend)),
                               sino, rtol=0, atol=1e-6)
    w = torch.rand(3, 6, 10, generator=g).to(backend)
    sh = torch.tensor([[0.3, -0.4], [1.2, 0.6], [-0.7, 0.2]], device=backend, requires_grad=True)
    (ft.apply_shifts_2d(sino, sh) * w).sum().backward()
    h = 1e-2
    fd = torch.zeros(3, 2)
    for i in range(3):
        for j in range(2):
            p = sh.detach().clone()
            p[i, j] += h
            m = sh.detach().clone()
            m[i, j] -= h
            fd[i, j] = ((ft.apply_shifts_2d(sino, p) - ft.apply_shifts_2d(sino, m)) * w).double().sum() / (2 * h)
    assert rel_l2(sh.grad, fd) < 1e-3


def test_apply_shifts_2d_rejects_bad_shapes():
    sino = torch.zeros(3, 4, 5)
    with pytest.raises(ValueError, match=r"\[n_angles, 2\]"):
        ft.apply_shifts_2d(sino, torch.zeros(3))
    with pytest.raises(ValueError, match="3 angles"):
        ft.apply_shifts_2d(sino, torch.zeros(2, 2))
