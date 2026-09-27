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


def _bp_scalar32(sino, ang, Nz, Nx):
    """back_project_parallel_cpu's per-slice scatter in fp32: every voxel
    receives its additions in (angle, detector, step, corner) order."""
    sg = sino.numpy()
    n, Ny, Nd = sg.shape
    R, hd, hx, hz = _geom32(Nz, Nx, Nd)
    cs, sn = _trig32(ang)
    out = np.zeros((Ny, Nz * Nx), _F)
    one = _F(1)
    steps = np.arange(-R, R + 1)
    for a in range(n):
        for d in range(Nd):
            t = _F(d) - hd
            ok, ix0, iz0, wx, wz, _, _ = _samples32(np.full(steps.shape, t, _F), cs[a], sn[a],
                                                    steps, hx, hz, Nz, Nx)
            ix0, iz0, wx, wz = ix0[ok], iz0[ok], wx[ok], wz[ok]
            idx = np.stack([iz0 * Nx + ix0, iz0 * Nx + ix0 + 1,
                            (iz0 + 1) * Nx + ix0, (iz0 + 1) * Nx + ix0 + 1], 1).reshape(-1)
            w = np.stack([(one - wz) * (one - wx), (one - wz) * wx,
                          wz * (one - wx), wz * wx], 1).reshape(-1)
            np.add.at(out, (slice(None), idx), w[None, :] * sg[a, :, d][:, None])
    return torch.from_numpy(out.reshape(Ny, Nz, Nx))


@pytest.mark.parametrize("shape", _YB_SHAPES)
@pytest.mark.parametrize("threads", [1, 3, 8])
def test_cpu_bp_is_the_scalar_scatter_bit_for_bit(shape, threads):
    from fasttomo.autograd import _back_project_parallel_raw

    _, sino, ang = _yb_case(shape)
    Ny, Nz, Nx, Nd, n = shape
    out = _back_project_parallel_raw(sino, ang, Nz, num_threads=threads, Nx=Nx)
    assert torch.equal(out, _bp_scalar32(sino, ang, Nz, Nx))


@pytest.mark.parametrize("op", ["fp", "bp"])
def test_cpu_parallel_kernels_are_thread_invariant(op):
    """Bigger than the ports above: several Y blocks, more threads than Z
    rows per band; every thread count gives the same bits."""
    from fasttomo.autograd import _back_project_parallel_raw, _forward_project_parallel_raw

    g = torch.Generator().manual_seed(5)
    Ny, Nz, Nx, Nd = 37, 21, 26, 30
    ang = torch.linspace(-70, 70, 23)
    if op == "fp":
        x = torch.rand(Ny, Nz, Nx, generator=g)
        run = lambda k: _forward_project_parallel_raw(x, ang, Nd, num_threads=k)  # noqa: E731
    else:
        x = torch.rand(23, Ny, Nd, generator=g)
        run = lambda k: _back_project_parallel_raw(x, ang, Nz, num_threads=k, Nx=Nx)  # noqa: E731
    ref = run(1)
    for k in (2, 5, 16, 40):
        assert torch.equal(run(k), ref), k


# ── P5: rotmat forward (clipping + interior fast path) vs its fp32 port ────

def _fp_rotmat_scalar32(vol, mats, Nd):
    """forward_project_rotmat_cpu's unclipped loop with border-zero SAMP3
    corners, in fp32 (the corners read a zero-padded copy)."""
    v = vol.numpy()
    Ny, Nz, Nx = v.shape
    vp = np.pad(v, 1)
    M = mats.reshape(-1, 9).numpy()
    R = int(math.ceil(0.5 * math.sqrt(float(np.float32(Nx * Nx + Ny * Ny + Nz * Nz))))) + 1
    hd, hx = _F(0.5) * _F(Nd - 1), _F(0.5) * _F(Nx - 1)
    hy, hz = _F(0.5) * _F(Ny - 1), _F(0.5) * _F(Nz - 1)
    V, U = np.meshgrid(np.arange(Ny).astype(_F) - hy, np.arange(Nd).astype(_F) - hd,
                       indexing="ij")
    out = np.zeros((len(M), Ny, Nd), _F)
    one = _F(1)
    for a, m in enumerate(M):
        ox, oy, oz = U * m[0] + V * m[3], U * m[1] + V * m[4], U * m[2] + V * m[5]
        acc = np.zeros((Ny, Nd), _F)
        for step in range(-R, R + 1):
            s = _F(step)
            fx, fy, fz = ox + s * m[6] + hx, oy + s * m[7] + hy, oz + s * m[8] + hz
            ok = ((fx > -1) & (fx < _F(Nx)) & (fy > -1) & (fy < _F(Ny))
                  & (fz > -1) & (fz < _F(Nz)))
            fx, fy, fz = np.where(ok, fx, _F(0)), np.where(ok, fy, _F(0)), np.where(ok, fz, _F(0))
            x0, y0, z0 = np.floor(fx), np.floor(fy), np.floor(fz)
            wx, wy, wz = fx - x0, fy - y0, fz - z0
            ux, uy, uz = one - wx, one - wy, one - wz
            x0, y0, z0 = x0.astype(np.int64) + 1, y0.astype(np.int64) + 1, z0.astype(np.int64) + 1

            def c(dy, dz, dx):
                return vp[y0 + dy, z0 + dz, x0 + dx]

            term = (uy * uz * (ux * c(0, 0, 0) + wx * c(0, 0, 1))
                    + uy * wz * (ux * c(0, 1, 0) + wx * c(0, 1, 1))
                    + wy * uz * (ux * c(1, 0, 0) + wx * c(1, 0, 1))
                    + wy * wz * (ux * c(1, 1, 0) + wx * c(1, 1, 1)))
            acc = np.where(ok, acc + term, acc)
        out[a] = acc
    return torch.from_numpy(out)


@pytest.mark.parametrize("shape", [(3, 9, 11, 13), (6, 10, 8, 7), (1, 12, 12, 12)])
@pytest.mark.parametrize("threads", [1, 4])
def test_cpu_rotmat_fp_is_the_unclipped_loop_bit_for_bit(shape, threads):
    from fasttomo.autograd import _forward_project_rotmat_raw, euler_zyz_to_rotmat

    Ny, Nz, Nx, Nd = shape
    g = torch.Generator().manual_seed(9)
    vol = torch.rand(Ny, Nz, Nx, generator=g) + 0.5          # non-zero on every face
    ang = torch.tensor([-90.0, -41.3, 0.0, 27.9, 90.0])
    # pure tilts (exact 0/1 matrix entries: samples on grid planes) and
    # generic ZYZ rotations
    eul = torch.cat([torch.stack([torch.zeros(5), ang, torch.zeros(5)], 1),
                     torch.stack([torch.linspace(-30, 40, 5), ang, torch.linspace(25, -10, 5)], 1)])
    mats = euler_zyz_to_rotmat(eul).reshape(-1, 9).contiguous()
    out = _forward_project_rotmat_raw(vol, mats, Nd, num_threads=threads)
    assert torch.equal(out, _fp_rotmat_scalar32(vol, mats, Nd))


def test_cpu_rotmat_kernels_are_thread_invariant():
    from fasttomo.autograd import _forward_project_rotmat_raw, _grad_rotmat_raw, euler_zyz_to_rotmat

    g = torch.Generator().manual_seed(4)
    vol = torch.rand(7, 15, 13, generator=g)
    eul = torch.stack([torch.linspace(-20, 20, 11), torch.linspace(-65, 65, 11),
                       torch.linspace(10, -5, 11)], 1)
    mats = euler_zyz_to_rotmat(eul).reshape(-1, 9).contiguous()
    gs = torch.randn(11, 7, 17, generator=g)
    fp1 = _forward_project_rotmat_raw(vol, mats, 17, num_threads=1)
    gm1 = _grad_rotmat_raw(vol, gs, mats, num_threads=1)
    for k in (3, 16):
        assert torch.equal(_forward_project_rotmat_raw(vol, mats, 17, num_threads=k), fp1)
        assert torch.equal(_grad_rotmat_raw(vol, gs, mats, num_threads=k), gm1)


# ── P4: grad_angle, fp64 accumulation, thread-invariant ───────────────────

def _grad_angle_fp64(vol, gs, ang):
    """grad_angle_cpu's formula with every sum in fp64, on the same fp32
    sample geometry (so the same samples and stencils as the kernel)."""
    v = vol.double().numpy()
    g = gs.double().numpy()
    Ny, Nz, Nx = v.shape
    n, _, Nd = g.shape
    R, hd, hx, hz = _geom32(Nz, Nx, Nd)
    cs, sn = _trig32(ang)
    t = np.arange(Nd).astype(_F) - hd
    out = np.zeros(n)
    for a in range(n):
        for step in range(-R, R + 1):
            ok, ix0, iz0, wx, wz, x, z = _samples32(t, cs[a], sn[a], step, hx, hz, Nz, Nx)
            wx, wz = wx.astype(np.float64), wz.astype(np.float64)
            v00, v01 = v[:, iz0, ix0], v[:, iz0, ix0 + 1]
            v10, v11 = v[:, iz0 + 1, ix0], v[:, iz0 + 1, ix0 + 1]
            dx = (1 - wz) * (v01 - v00) + wz * (v11 - v10)
            dz = (1 - wx) * (v10 - v00) + wx * (v11 - v01)
            term = g[a] * (dx * -z.astype(np.float64) + dz * x.astype(np.float64))
            out[a] += np.where(ok[None], term, 0.0).sum()
    return torch.from_numpy(out) * (math.pi / 180.0)


def _grad_angle_case(Ny, N, n, seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.arange(N, dtype=torch.float64) - 0.5 * (N - 1)
    blob = torch.exp(-((c[None, :] / (0.25 * N)) ** 2 + (c[:, None] / (0.2 * N)) ** 2))
    vol = (blob.float().expand(Ny, N, N) + 0.05 * torch.rand(Ny, N, N, generator=g)).contiguous()
    gs = torch.randn(n, Ny, N, generator=g)
    return vol, gs, torch.linspace(-61.3, 58.7, n)


@pytest.mark.parametrize("Ny,N,n", [(17, 48, 13), (1, 40, 9), (6, 33, 11)])
def test_cpu_grad_angle_is_fp64_accurate_and_thread_invariant(Ny, N, n):
    """The per-angle sums are fp64 in an order fixed by the geometry: the
    result matches the all-fp64 evaluation of the same formula to fp32
    rounding of the ray sums (the old single fp32 accumulator per thread
    was 1e-6 to 8e-6 off and changed with the thread count), and it is the
    same bits for every thread count."""
    from fasttomo.autograd import _grad_angle_raw

    vol, gs, ang = _grad_angle_case(Ny, N, n, seed=N)
    ref = _grad_angle_fp64(vol, gs, ang)
    out = _grad_angle_raw(vol, gs, ang, num_threads=1)
    assert float((out.double() - ref).norm() / ref.norm()) < 1e-6
    for k in (3, 8, 16):
        assert torch.equal(_grad_angle_raw(vol, gs, ang, num_threads=k), out), k
