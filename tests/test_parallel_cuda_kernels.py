"""
Parallel-beam (single tilt axis) CUDA forward/back-projection kernels and
their texture caches: NaN/Inf tilt angles, Y-slice packing (float2 texels
for fp32, 4-channel half texels for fp16, padded when Ny is not a
multiple), the fp16 overflow flag and the functional API's texture pool.

Every packing test compares against per-slice results with ``torch.equal``:
the packed and the single-channel textures filter each channel with the
same weights, so the outputs are the same bits.
"""

from __future__ import annotations

import pytest
import torch

import fasttomo as ft
from conftest import needs_cuda

DEV = "cuda"


def _angles(n: int) -> torch.Tensor:
    return torch.linspace(-70.0, 70.0, n, device=DEV)


# ── NaN / Inf tilt angles on the device ─────────────────────────────────────

@needs_cuda
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("Ny", [1, 3, 4, 6])
@pytest.mark.parametrize("cached", [False, True])
def test_non_finite_device_angle_view_is_skipped(bad, dtype, Ny, cached):
    """A NaN/Inf angle in a CUDA-resident angle tensor (not checked on the
    host: that would cost a sync per call) makes every sample of that view
    fail the kernels' bounds test, as in the CPU kernels: its forward
    projection is 0 and it adds nothing to the back-projection. The other
    views are the bits they are without it. (It used to give finite wrong
    values: the texture unit returns garbage for a NaN coordinate.)"""
    Nz, Nx, Nd, n = 17, 19, 21, 7
    g = torch.Generator().manual_seed(3)
    vol = torch.rand(Ny, Nz, Nx, generator=g).to(DEV, dtype)
    sino = torch.rand(n, Ny, Nd, generator=g).to(DEV, dtype)
    ang = _angles(n)
    bad_ang = ang.clone()
    bad_ang[2] = bad
    keep = torch.tensor([i for i in range(n) if i != 2], device=DEV)

    if cached:
        def proj(a):
            return ft.ParallelProjector((Ny, Nz, Nx), a, N_det=Nd, device=DEV)
        fp = lambda a: proj(a).forward(vol)                    # noqa: E731
        bp = lambda s, a: proj(a).back(s)                      # noqa: E731
    else:
        fp = lambda a: ft.forward_project(vol, a, N_det=Nd)    # noqa: E731
        bp = lambda s, a: ft.back_project(s, a, Nz=Nz, Nx=Nx)  # noqa: E731

    out = fp(bad_ang)
    assert (out[2] == 0).all()
    assert torch.equal(out[keep], fp(ang[keep]))
    assert torch.equal(bp(sino, bad_ang), bp(sino[keep], ang[keep]))
