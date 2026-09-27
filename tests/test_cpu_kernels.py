"""
CPU (C/OpenMP) projection kernels: robustness, bit identity of the
Y-batched rewrites, thread invariance (phase-1b PR A).

- C1: a NaN or Inf tilt angle (or rotation matrix) raises ValueError at the
  Python entry points; below them the parallel kernels skip NaN samples
  instead of indexing memory with ``(int)NaN`` (they used to segfault).
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

import fasttomo as ft

HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA GPU available")


def _devices():
    return ["cpu"] + (["cuda"] if HAS_CUDA else [])


# ── C1: non-finite angles ──────────────────────────────────────────────────

def _bad_angle_calls(device):
    """name -> call(angles) for every entry point that takes tilt angles or
    Euler angles; the volume / sinogram live on ``device``, the angles are
    the host tensor passed in."""
    g = torch.Generator().manual_seed(11)
    vol = torch.rand(3, 10, 10, generator=g).to(device)
    sino = torch.rand(3, 3, 10, generator=g).to(device)

    def grad_angle(a):
        a = a.clone().requires_grad_(True)
        ft.forward_project(vol, a).sum().backward()
        return a.grad

    def euler(a):
        return torch.stack([torch.zeros_like(a), a, torch.zeros_like(a)], 1)

    return {
        "forward_project": lambda a: ft.forward_project(vol, a),
        "back_project": lambda a: ft.back_project(sino, a),
        "grad_angle": grad_angle,
        "forward_project_3d": lambda a: ft.forward_project_3d(vol, euler(a)),
        "back_project_3d": lambda a: ft.back_project_3d(sino, euler(a)),
        "run_fbp": lambda a: ft.run_fbp(sino, a),
        "run_sirt": lambda a: ft.run_sirt(sino, a, 1),
        "run_weighted_backprojection": lambda a: ft.run_weighted_backprojection(sino, a),
        "projector": lambda a: ft.ParallelProjector((3, 10, 10), a, device=device).forward(vol),
    }


_BAD_CALLS = ["forward_project", "back_project", "grad_angle", "forward_project_3d",
              "back_project_3d", "run_fbp", "run_sirt", "run_weighted_backprojection",
              "projector"]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("call", _BAD_CALLS)
def test_non_finite_host_angles_raise(device, bad, call):
    """A NaN/Inf in host-side angles raises ValueError on both backends
    (the CPU kernels used to crash the process on it)."""
    angles = torch.tensor([-30.0, bad, 30.0])
    with pytest.raises(ValueError, match="finite"):
        _bad_angle_calls(device)[call](angles)


@pytest.mark.parametrize("device", _devices())
def test_non_finite_rotation_matrices_raise_on_host(device):
    from fasttomo.autograd import _forward_project_rotmat_raw

    vol = torch.rand(3, 8, 8).to(device)
    mats = torch.eye(3).repeat(2, 1, 1)
    mats[1, 0, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _forward_project_rotmat_raw(vol, mats)


def test_finite_angles_still_accepted_at_the_extremes():
    vol = torch.rand(2, 8, 8)
    a = torch.tensor([-90.0, 0.0, 90.0, 1e6], dtype=torch.float64)   # finite, any range
    assert torch.isfinite(ft.forward_project(vol, a)).all()


# The kernel-level guard, below the Python check: call the C kernels
# directly with a NaN / Inf trig table in a subprocess (a regression would
# segfault, which must not take the test runner down).
_KERNEL_NAN_SCRIPT = textwrap.dedent("""
    import math, sys, torch
    import fasttomo._core as c
    Ny, Nz, Nx, n, Nd = 3, 12, 11, 4, 13
    torch.manual_seed(0)
    vol = torch.rand(Ny, Nz, Nx) + 0.5
    sino = torch.rand(n, Ny, Nd)
    bad = float(sys.argv[1])
    ang = torch.tensor([-40.0, bad, 25.0, 70.0]) * (math.pi / 180)
    cs, sn = torch.cos(ang).contiguous(), torch.sin(ang).contiguous()
    P = lambda t: t.data_ptr()
    for threads in (1, 3):
        out = torch.full((n, Ny, Nd), 7.0)
        c.forward_project_parallel_cpu(P(vol), P(out), P(cs), P(sn), Ny, Nz, Nx, n, Nd, threads)
        assert torch.isfinite(out).all(), "fp"
        assert (out[1] == 0).all(), "fp: every sample of the NaN view is skipped"
        bp = torch.full((Ny, Nz, Nx), 7.0)
        c.back_project_parallel_cpu(P(sino), P(bp), P(cs), P(sn), Ny, Nz, Nx, n, Nd, threads)
        assert torch.isfinite(bp).all(), "bp"
        ga = torch.zeros(n)
        c.grad_angle_cpu(P(vol), P(sino), P(ga), P(cs), P(sn), Ny, Nz, Nx, n, Nd, threads)
        assert torch.isfinite(ga).all() and ga[1] == 0, "grad_angle"
        gb = torch.empty(Ny, Nz, Nx)
        c.back_project_gather_parallel_cpu(P(sino), P(gb), P(cs), P(sn), Ny, Nz, Nx, n, Nd, threads)
        assert not torch.isfinite(gb).all(), "gather: NaN weights give NaN (clamped index)"
        mats = torch.eye(3).repeat(n, 1, 1).reshape(n, 9).contiguous()
        mats[1] = bad
        fr = torch.empty(n, Ny, Nd)
        c.forward_project_rotmat_cpu(P(vol), P(fr), P(mats), Ny, Nz, Nx, n, Nd, threads)
        assert torch.isfinite(fr[0]).all()
        gm = torch.zeros(n, 9)
        c.grad_rotmat_cpu(P(vol), P(sino), P(mats), P(gm), Ny, Nz, Nx, n, Nd, threads)
        br = torch.empty(Ny, Nz, Nx)
        c.back_project_rotmat_cpu(P(sino), P(br), P(mats), Ny, Nz, Nx, n, Nd, threads)
    print("ok")
""")


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_cpu_kernels_survive_non_finite_trig_tables(bad):
    r = subprocess.run([sys.executable, "-c", _KERNEL_NAN_SCRIPT, bad],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip() == "ok", (r.returncode, r.stdout, r.stderr)


# ── P1-P3: Y-batched parallel kernels vs fp32 ports of the scalar loops ────
#
# The Y-batched kernels must reproduce the per-slice scalar loops of the
# original kernels bit for bit (same samples, same fp32 operations in the
# same order). These NumPy ports do each fp32 operation of the scalar loop
# elementwise in the same order (NumPy float32 ufuncs are IEEE single
# operations, no contraction), so ``torch.equal`` against them pins every
# lane of every Y block, for Ny across the block size (FT_YB = 16) and for
# several thread counts.

_F = np.float32

# (Ny, Nz, Nx, N_det, n_angles): Ny = 1, a partial block, a full block plus a
# partial one, odd sizes, N_det != Nx, Nz != Nx.
_YB_SHAPES = [(1, 9, 11, 13, 5), (5, 12, 10, 14, 7), (17, 10, 13, 9, 6), (33, 8, 9, 12, 4)]


def _yb_case(shape, seed=0):
    Ny, Nz, Nx, Nd, n = shape
    g = torch.Generator().manual_seed(seed)
    vol = torch.rand(Ny, Nz, Nx, generator=g) - 0.3
    sino = torch.rand(n, Ny, Nd, generator=g) - 0.5
    # includes 0 and +-90 degrees (exact trig values) and generic angles
    ang = torch.tensor([-90.0, -47.3, 0.0, 33.7, 90.0, 12.1, -71.9][:n])
    return vol, sino, ang


def _trig32(ang):
    a = ang.to(torch.float32) * (math.pi / 180.0)
    return torch.cos(a).numpy(), torch.sin(a).numpy()


def _geom32(Nz, Nx, Nd):
    ray_len = int(math.ceil(0.5 * math.sqrt(float(np.float32(Nx * Nx + Nz * Nz))))) + 1
    return ray_len, _F(0.5) * _F(Nd - 1), _F(0.5) * _F(Nx - 1), _F(0.5) * _F(Nz - 1)


def _samples32(t, ca, sa, step, hx, hz, Nz, Nx):
    """fp32 sample geometry of the scalar loop, exactly as the C code."""
    s = _F(step)
    x = t * ca - s * sa
    z = t * sa + s * ca
    fx = x + hx
    fz = z + hz
    ok = (fx >= 0) & (fx < _F(Nx - 1)) & (fz >= 0) & (fz < _F(Nz - 1))
    fxs, fzs = np.where(ok, fx, _F(0)), np.where(ok, fz, _F(0))
    ix0, iz0 = fxs.astype(np.int64), fzs.astype(np.int64)
    wx, wz = fxs - ix0.astype(_F), fzs - iz0.astype(_F)
    return ok, ix0, iz0, wx, wz, x, z


def _fp_scalar32(vol, ang, Nd):
    """forward_project_parallel_cpu's per-slice loop in fp32."""
    v = vol.numpy()
    Ny, Nz, Nx = v.shape
    R, hd, hx, hz = _geom32(Nz, Nx, Nd)
    cs, sn = _trig32(ang)
    t = np.arange(Nd).astype(_F) - hd
    out = np.zeros((len(cs), Ny, Nd), _F)
    for a in range(len(cs)):
        acc = np.zeros((Ny, Nd), _F)
        for step in range(-R, R + 1):
            ok, ix0, iz0, wx, wz, _, _ = _samples32(t, cs[a], sn[a], step, hx, hz, Nz, Nx)
            one = _F(1)
            term = ((one - wz) * (one - wx) * v[:, iz0, ix0]
                    + (one - wz) * wx * v[:, iz0, ix0 + 1]
                    + wz * (one - wx) * v[:, iz0 + 1, ix0]
                    + wz * wx * v[:, iz0 + 1, ix0 + 1])
            acc = np.where(ok[None], acc + term, acc)
        out[a] = acc
    return torch.from_numpy(out)


@pytest.mark.parametrize("shape", _YB_SHAPES)
@pytest.mark.parametrize("threads", [1, 3, 8])
def test_cpu_fp_is_the_scalar_loop_bit_for_bit(shape, threads):
    from fasttomo.autograd import _forward_project_parallel_raw

    vol, _, ang = _yb_case(shape)
    out = _forward_project_parallel_raw(vol, ang, shape[3], num_threads=threads)
    assert torch.equal(out, _fp_scalar32(vol, ang, shape[3]))
