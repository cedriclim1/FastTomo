"""
CPU (C/OpenMP) projection kernels: robustness, bit identity of the
Y-batched rewrites, thread invariance (phase-1b PR A).

- C1: a NaN or Inf tilt angle (or rotation matrix) raises ValueError at the
  Python entry points; below them the parallel kernels skip NaN samples
  instead of indexing memory with ``(int)NaN`` (they used to segfault).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

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
