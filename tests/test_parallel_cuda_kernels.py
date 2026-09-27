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


# ── Y-slice packing (float2 for fp32, 4-channel half for fp16) ──────────────

@needs_cuda
def test_parallel_channel_policy():
    """fp32 packs 2 Y-slices per float2 texel from Ny = 2 on; fp16 packs 4
    per texel when Ny % 4 == 0; one slice stays 1-channel. The cached
    operator uses the same layout as the one-shot wrappers."""
    import fasttomo._core as core

    assert [core.parallel_channels(0, ny) for ny in (1, 2, 3, 4, 5)] == [1, 2, 2, 2, 2]
    assert [core.parallel_channels(1, ny) for ny in (1, 3, 4, 8)] == [1, 1, 4, 4]
    for dtype, code in ((torch.float32, 0), (torch.float16, 1)):
        for Ny in (1, 3, 4):
            p = ft.ParallelProjector((Ny, 8, 8), _angles(3), device=DEV)
            p.forward(torch.zeros(Ny, 8, 8, device=DEV, dtype=dtype))
            assert p._op._channels() == core.parallel_channels(code, Ny)


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("Ny", [2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("cached", [False, True])
def test_packed_projection_equals_per_slice(dtype, Ny, cached):
    """FP and BP of a [Ny, Nz, Nx] volume (packed textures, the last texel
    zero-padded when Ny is not a multiple of the channel count) are, bit for
    bit, the per-slice results of [1, Nz, Nx] calls (1-channel textures)."""
    Nz, Nx, Nd, n = 13, 17, 19, 11
    g = torch.Generator().manual_seed(Ny)
    vol = torch.rand(Ny, Nz, Nx, generator=g).to(DEV, dtype)
    sino = torch.rand(n, Ny, Nd, generator=g).to(DEV, dtype)
    ang = _angles(n)

    def fp(v):
        if cached:
            return ft.ParallelProjector(tuple(v.shape), ang, N_det=Nd, device=DEV).forward(v)
        return ft.forward_project(v, ang, N_det=Nd)

    def bp(s):
        if cached:
            return ft.ParallelProjector((s.shape[1], Nz, Nx), ang, N_det=Nd, device=DEV).back(s)
        return ft.back_project(s, ang, Nz=Nz, Nx=Nx)

    assert torch.equal(fp(vol), torch.cat([fp(vol[y:y + 1]) for y in range(Ny)], dim=1))
    assert torch.equal(bp(sino), torch.cat([bp(sino[:, y:y + 1]) for y in range(Ny)], dim=0))


@needs_cuda
@pytest.mark.parametrize("dtype, channels, Ny", [
    (torch.float32, 2, 8), (torch.float32, 2, 7), (torch.float32, 2, 1),
    (torch.float16, 4, 8), (torch.float16, 4, 6), (torch.float16, 4, 5)])
def test_packed_tc_download_round_trip(dtype, channels, Ny):
    """tc_download inverts tc_upload for packed volume caches, including a
    zero-padded last texel."""
    import fasttomo._core as core
    from fasttomo._backend import _stream_ptr

    if dtype == torch.float16 and Ny % 4 and core.parallel_channels(1, Ny) != 4:
        pytest.skip("fp16 caches with Ny % 4 != 0 are not packed in this build")
    src = torch.randn(Ny, 5, 7, device=DEV).to(dtype)
    dst = torch.empty_like(src)
    h = core.tc_create(0, 0 if dtype == torch.float32 else 1, channels, *src.shape)
    try:
        assert core.tc_has_surface(h)
        core.tc_upload(h, src.data_ptr(), _stream_ptr(src.device))
        core.tc_download(h, dst.data_ptr(), _stream_ptr(src.device))
        torch.cuda.synchronize()
    finally:
        core.tc_destroy(h)
    assert torch.equal(dst, src)


@needs_cuda
@pytest.mark.parametrize("kind, dtype, channels", [
    (2, 0, 2), (2, 1, 4),      # TC_VOL_3D is 1-channel only
    (0, 1, 2), (1, 0, 4)])     # float2 is fp32-only, 4 channels fp16-only
def test_tc_create_rejects_unsupported_packing(kind, dtype, channels):
    import fasttomo._core as core

    with pytest.raises((ValueError, RuntimeError)):
        core.tc_create(kind, dtype, channels, 8, 8, 8)
