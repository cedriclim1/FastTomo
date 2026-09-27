"""
Regression tests for kernel and binding correctness fixes in ``csrc/``.

- D2: the rotation-matrix gradient (CPU and CUDA, cached and uncached) is the
  gradient of the forward projection, including the voxels within one voxel
  of the box faces. Checked against central finite differences of an
  independent fp64 NumPy port of the forward projector.
- D5: no GPU memory leaks when a one-shot (non-cached) CUDA wrapper or
  ``tc_create`` itself throws after the cudaArray exists.
- P1: textures are not destroyed under in-flight kernels (one-shot wrappers
  and a projector dropped right after launching work).
- P4: texture extents beyond the device limits raise a clear ValueError.
- P3 / scope 2a: every op (CPU and CUDA) gives the same result for strided
  (transposed, sliced, expanded) inputs as for their contiguous copies; the
  CPU bindings take data_ptr ints, so the Python layer must densify first.
- D1 (same bug class): the tilt-angle gradient accepts fp16 and strided
  volumes on CUDA.
- GENFIRE CPU gridding is bit-identical to the pre-change numpy binding.

CUDA tests skip automatically when no GPU is visible; the CPU tests always run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import fasttomo as ft
from fasttomo.autograd import (
    _euler_zyz_to_rotmat,
    _forward_project_rotmat_raw,
    _grad_rotmat_raw,
)

HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA GPU available")


def _rel(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    a = torch.as_tensor(a, dtype=torch.float64).cpu()
    b = torch.as_tensor(b, dtype=torch.float64).cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# ── D2: rotation-matrix gradient vs finite differences ─────────────────────

def _fp_rotmat_ref(vol: np.ndarray, mats: np.ndarray, N_det: int) -> np.ndarray:
    """fp64 reference of the rotmat forward projector.

    Detector-driven ray march with unit steps, trilinear interpolation and
    border-zero semantics (a sample contributes while any of its 8 corners is
    inside the volume). Written from the geometry in ``proj_geom.h`` without
    sharing code with the kernels: corners outside the volume read the zero
    padding of ``pad``.
    """
    Ny, Nz, Nx = vol.shape
    ray_len = int(math.ceil(0.5 * math.sqrt(Nx * Nx + Ny * Ny + Nz * Nz))) + 1
    hd, hx, hy, hz = 0.5 * (N_det - 1), 0.5 * (Nx - 1), 0.5 * (Ny - 1), 0.5 * (Nz - 1)
    pad = np.zeros((Ny + 2, Nz + 2, Nx + 2))
    pad[1:-1, 1:-1, 1:-1] = vol
    V, U, S = np.meshgrid(np.arange(Ny) - hy, np.arange(N_det) - hd,
                          np.arange(-ray_len, ray_len + 1, dtype=np.float64),
                          indexing="ij")
    out = np.zeros((mats.shape[0], Ny, N_det))
    for a, M in enumerate(mats):
        fx = U * M[0] + V * M[3] + S * M[6] + hx
        fy = U * M[1] + V * M[4] + S * M[7] + hy
        fz = U * M[2] + V * M[5] + S * M[8] + hz
        # Clamp far-away samples onto the zero padding; the weights below
        # then multiply zeros only.
        fx = np.clip(fx, -1.0, Nx)
        fy = np.clip(fy, -1.0, Ny)
        fz = np.clip(fz, -1.0, Nz)
        x0, y0, z0 = np.floor(fx), np.floor(fy), np.floor(fz)
        wx, wy, wz = fx - x0, fy - y0, fz - z0
        xi = x0.astype(int) + 1
        yi = y0.astype(int) + 1
        zi = z0.astype(int) + 1
        xj = np.minimum(xi + 1, Nx + 1)
        yj = np.minimum(yi + 1, Ny + 1)
        zj = np.minimum(zi + 1, Nz + 1)
        val = ((1 - wy) * (1 - wz) * ((1 - wx) * pad[yi, zi, xi] + wx * pad[yi, zi, xj])
               + (1 - wy) * wz * ((1 - wx) * pad[yi, zj, xi] + wx * pad[yi, zj, xj])
               + wy * (1 - wz) * ((1 - wx) * pad[yj, zi, xi] + wx * pad[yj, zi, xj])
               + wy * wz * ((1 - wx) * pad[yj, zj, xi] + wx * pad[yj, zj, xj]))
        out[a] = val.sum(-1)
    return out


def _fd_grad_mats(vol: np.ndarray, mats: np.ndarray, w: np.ndarray,
                  eps: float = 1e-6) -> np.ndarray:
    """Central finite difference of L(M) = <w, FP(vol, M)> w.r.t. every M entry."""
    N_det = w.shape[-1]
    fd = np.zeros_like(mats)
    for a in range(mats.shape[0]):
        for k in range(9):
            mp = mats.copy()
            mm = mats.copy()
            mp[a, k] += eps
            mm[a, k] -= eps
            fd[a, k] = ((w * _fp_rotmat_ref(vol, mp, N_det)).sum()
                        - (w * _fp_rotmat_ref(vol, mm, N_det)).sum()) / (2 * eps)
    return fd


# Generic (non-special) Euler angles: with angles like 0 or 90 deg every
# sample lands exactly on a trilinear kink and the derivative is one-sided,
# so finite differences and the analytic gradient legitimately disagree.
_EULER = torch.tensor([[12.3, 27.1, -7.9], [41.7, -33.2, 16.4], [3.3, 61.9, -2.2]])


def _face_touching_case(shape, kind):
    g = torch.Generator().manual_seed(1234)
    Ny, Nz, Nx = shape
    if kind == "ones":
        vol = torch.ones(Ny, Nz, Nx)
    else:
        vol = torch.rand(Ny, Nz, Nx, generator=g) + 0.5   # non-zero on every face
    mats = _euler_zyz_to_rotmat(_EULER).reshape(-1, 9).float()
    w = torch.randn(mats.shape[0], Ny, Nx, generator=g)
    fd = _fd_grad_mats(vol.double().numpy(), mats.double().numpy(), w.double().numpy())
    return vol, mats, w, fd


_D2_SHAPES = [(8, 8, 8), (7, 9, 6), (1, 5, 5)]


@pytest.mark.parametrize("shape", _D2_SHAPES)
@pytest.mark.parametrize("kind", ["ones", "rand"])
def test_grad_rotmat_cpu_matches_finite_difference(shape, kind):
    vol, mats, w, fd = _face_touching_case(shape, kind)
    g = _grad_rotmat_raw(vol, w, mats)
    assert g.shape == (3, 9) and g.dtype == torch.float32
    assert _rel(g, fd) < 1e-4


@needs_cuda
@pytest.mark.parametrize("shape", _D2_SHAPES)
@pytest.mark.parametrize("kind", ["ones", "rand"])
def test_grad_rotmat_cuda_matches_finite_difference_and_cpu(shape, kind):
    vol, mats, w, fd = _face_touching_case(shape, kind)
    g_cpu = _grad_rotmat_raw(vol, w, mats)
    g_cuda = _grad_rotmat_raw(vol.cuda(), w.cuda(), mats.cuda())
    assert g_cuda.shape == (3, 9) and g_cuda.dtype == torch.float32
    assert _rel(g_cuda, fd) < 1e-4
    assert _rel(g_cuda, g_cpu) < 1e-4


@needs_cuda
@pytest.mark.parametrize("kind", ["ones", "rand"])
def test_grad_rotmat_cached_matches_finite_difference(kind):
    """The RotmatProjector path (grad_rotmat_cuda_cached) gets the same fix."""
    vol, mats, w, fd = _face_touching_case((7, 9, 6), kind)
    Ny, Nz, Nx = vol.shape
    proj = ft.RotmatProjector((Ny, Nz, Nx), n_angles=3, device="cuda")
    m = mats.cuda().requires_grad_(True)
    sino = proj.forward(vol.cuda(), m)
    (sino * w.cuda()).sum().backward()
    assert _rel(m.grad, fd) < 1e-4


@needs_cuda
def test_grad_rotmat_euler_cpu_cuda_agree_on_box_filling_volume():
    """End-to-end: d loss / d euler through forward_project_3d, CPU vs CUDA,
    on a volume that fills the whole box (the D2 failure scenario). The loss
    is linear in the sinogram so both backends see the same grad_sino (the
    CUDA forward's 8-bit texture filter weights would otherwise leak in)."""
    torch.manual_seed(0)
    vol = torch.rand(10, 10, 10) + 0.5
    target = torch.randn(3, 10, 10)
    grads = []
    for dev in ("cpu", "cuda"):
        e = _EULER.clone().to(dev).requires_grad_(True)
        sino = ft.forward_project_3d(vol.to(dev), e)
        (sino * target.to(dev)).sum().backward()
        grads.append(e.grad.cpu())
    assert _rel(grads[1], grads[0]) < 1e-4


# ── F2: ray clipping in the rotmat forward-projection kernel ───────────────
#
# forward_project_rotmat_kernel marches only the steps whose trilinear
# stencil can touch the volume (slab test, widened by a voxel and a step on
# each side, united over the thread block). The dropped steps read border
# zeros, so the sinogram must be bit-identical to the full [-ray_len,
# ray_len] loop, which `_debug_set_rotmat_ray_clip(0)` restores. The
# gradient w.r.t. the matrices reuses the forward's sinogram and grad_rotmat
# (not clipped), so it is checked too.

def _clip_cases():
    g = torch.Generator().manual_seed(77)
    rnd = torch.rand(4, 3, generator=g) * torch.tensor([360.0, 180.0, 360.0]) - 180.0
    return {
        "tilt": torch.stack([torch.zeros(9), torch.linspace(-70, 70, 9), torch.zeros(9)], 1),
        "zyz": torch.cat([_EULER, rnd]),
        # axis-aligned rays: direction components exactly 0 (slab-parallel case)
        "axes": torch.tensor([[0.0, 0.0, 0.0], [0.0, 90.0, 0.0], [90.0, 90.0, 0.0],
                              [0.0, 180.0, 0.0], [90.0, 0.0, 0.0]]),
    }


def _clip_volumes(shape):
    g = torch.Generator().manual_seed(sum(shape))
    Ny, Nz, Nx = shape
    face = torch.rand(Ny, Nz, Nx, generator=g) + 0.5         # content on every face
    corners = torch.zeros(Ny, Nz, Nx)
    for y in (0, Ny - 1):
        for z in (0, Nz - 1):
            for x in (0, Nx - 1):
                corners[y, z, x] = 1.0 + x + 10 * z + 100 * y  # content only at the 8 corners
    return {"face": face, "corners": corners}


def _with_clip(on, fn):
    from fasttomo import _core
    prev = _core._debug_set_rotmat_ray_clip(on)
    try:
        return fn()
    finally:
        _core._debug_set_rotmat_ray_clip(prev)


# (Ny, Nz, Nx), N_det: a single 16x16 tile, odd/anisotropic, a detector wider
# than the volume diagonal (rays that miss it), one narrower than the volume,
# and a multi-block one (several tiles per view, partial tiles in u and v).
# The small ones have at most 2 grad_rotmat blocks per view, so its
# cross-block fp32 atomicAdds are order-independent and the gradient can be
# compared with torch.equal; the multi-block shape checks the sinogram only.
_CLIP_SHAPES = [((16, 16, 16), 16), ((7, 13, 10), 10), ((9, 5, 12), 29), ((12, 11, 9), 5),
                ((37, 30, 41), 45)]


@needs_cuda
@pytest.mark.parametrize("shape,N_det", _CLIP_SHAPES)
@pytest.mark.parametrize("case", ["tilt", "zyz", "axes"])
def test_rotmat_ray_clipping_is_bit_identical(shape, N_det, case):
    euler = _clip_cases()[case]
    mats = _euler_zyz_to_rotmat(euler).reshape(-1, 9).float().cuda()
    # also a scaled (non-orthonormal) matrix: the clipping must not assume |d| = 1
    mats = torch.cat([mats, mats[-1:] * 0.37, mats[:1] * 1.9])
    n = mats.shape[0]
    Ny, Nz, Nx = shape
    small = N_det <= 32 and Ny <= 16
    gs = torch.randn(n, Ny, N_det, generator=torch.Generator().manual_seed(5)).cuda()
    for name, vol in _clip_volumes(shape).items():
        vol = vol.cuda()
        for dtype in (torch.float32, torch.float16):
            v = vol.to(dtype)

            def fp(v=v):
                P = ft.RotmatProjector(shape, n, "cuda", N_det=N_det)
                return (P.forward(v, mats),
                        _forward_project_rotmat_raw(v, mats, N_det))

            (a_c, b_c), (a_u, b_u) = _with_clip(1, fp), _with_clip(0, fp)
            assert torch.equal(a_c, a_u) and torch.equal(b_c, b_u), (name, dtype)
            assert a_c.abs().sum() > 0
            if not small:
                continue

            def grad(v=v):
                P = ft.RotmatProjector(shape, n, "cuda", N_det=N_det)
                m = mats.clone().requires_grad_(True)
                (P.forward(v, m).float() * gs).sum().backward()
                return m.grad

            assert torch.equal(_with_clip(1, grad), _with_clip(0, grad)), (name, dtype)


# ── CPU voxel-driven (gather) back-projection for W-BP / wSIRT ────────────

@pytest.mark.parametrize("Ny,Nz,Nx,Nd,n_ang", [
    (2, 12, 12, 12, 7), (3, 9, 14, 11, 7), (1, 16, 10, 20, 7), (4, 40, 40, 32, 61),
    (5, 33, 40, 29, 13), (1, 1, 1, 1, 1), (2, 7, 5, 3, 4)])
def test_cpu_gather_back_projection_is_bp_gather_bit_for_bit(Ny, Nz, Nx, Nd, n_ang):
    """The C/OpenMP kernel the CPU W-BP uses is `_bp_gather` bit for bit
    (same fp32 arithmetic and order, no FMA), for grids wider and narrower
    than the detector and any thread count."""
    from fasttomo.algorithms.analytic import _bp_gather
    from fasttomo.autograd import _back_project_gather_parallel_raw
    g = torch.Generator().manual_seed(Ny * 1000 + Nz * 10 + Nd)
    ang = torch.linspace(-70.0, 70.0, n_ang) + 0.123
    sino = torch.rand(n_ang, Ny, Nd, generator=g) - 0.25
    ref = _bp_gather(sino, ang, Nz, Nx)
    for nt in (1, 3, 8):
        out = _back_project_gather_parallel_raw(sino, ang, Nz, Nx, nt)
        assert out.shape == (Ny, Nz, Nx) and out.dtype == torch.float32
        assert torch.equal(out, ref), nt


def test_cpu_wbp_uses_the_c_gather_kernel(monkeypatch):
    """run_weighted_backprojection / run_wsirt on CPU go through the C
    kernel with the caller's num_threads; plain back_project stays the
    ray-driven transpose."""
    from fasttomo import _core
    calls = []
    real = _core.back_project_gather_parallel_cpu
    monkeypatch.setattr(_core, "back_project_gather_parallel_cpu",
                        lambda *a: (calls.append(a[-1]), real(*a))[1])
    vol = torch.zeros(2, 16, 16)
    vol[:, 4:12, 5:11] = 1.0
    ang = torch.linspace(-60, 60, 9)
    sino = ft.forward_project(vol, ang)
    ft.run_weighted_backprojection(sino, ang, num_threads=3)
    ft.run_wsirt(sino, ang, 2, num_threads=2)
    assert calls == [3, 2, 2]
    calls.clear()
    ft.back_project(sino, ang)
    assert calls == []


# ── D5 / P1 / P4: texture-cache lifetime on the CUDA one-shot paths ─────────

_MiB = 1 << 20


def _driver_free_bytes() -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0]


def _assert_no_leak(fail_once, array_bytes, n_calls=200, attempts=3):
    """Call ``fail_once`` (which must raise) ``n_calls`` times and check the
    driver-level free memory does not drop by a leak-sized amount.

    The bound is a quarter of ``n_calls * array_bytes`` (at least 32 MiB).
    On the pre-fix code the measured leak was half of ``n_calls *
    array_bytes`` per attempt, so the bound catches it with a 2x margin
    (not 4x). The GPU may be shared, so
    another process allocating during the loop could fake a drop: the loop
    is retried up to ``attempts`` times and passes if any attempt stays under
    the bound (a real leak shows up on every attempt)."""
    with pytest.raises((RuntimeError, ValueError)):
        fail_once()                      # warm-up: allocator pools, module init
    bound = max(n_calls * array_bytes // 4, 32 * _MiB)
    drops = []
    for _ in range(attempts):
        before = _driver_free_bytes()
        for _ in range(n_calls):
            with pytest.raises((RuntimeError, ValueError)):
                fail_once()
        drops.append(before - _driver_free_bytes())
        if drops[-1] < bound:
            return
    pytest.fail(f"leaked on every attempt: {[d // _MiB for d in drops]} MiB "
                f"(bound {bound // _MiB} MiB)")


# grid.z = n_angles (rotmat FP / grad) and grid.y = ceil(n_angles / 16)
# (parallel FP) exceed the 65535 launch limit: the launch fails *after* the
# one-shot cudaArray was created and filled, which used to leak it (D5).
_TOO_MANY_ROTMAT = 65536
_TOO_MANY_PARALLEL = 65535 * 16 + 1


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_forward_project_3d_launch_failure_does_not_leak(dtype):
    vol = torch.zeros(1, 2048, 2048, device="cuda", dtype=dtype)   # 8-16 MiB array
    euler = torch.zeros(_TOO_MANY_ROTMAT, 3, device="cuda")
    _assert_no_leak(lambda: ft.forward_project_3d(vol, euler, N_det=2),
                    vol.numel() * vol.element_size())


@needs_cuda
@pytest.mark.parametrize("dtype,Ny", [(torch.float32, 1),
                                      (torch.float16, 4),    # 4-channel packed texture
                                      (torch.float16, 1)])   # 1-channel fp16 texture
def test_forward_project_launch_failure_does_not_leak(dtype, Ny):
    vol = torch.zeros(Ny, 2048, 1024, device="cuda", dtype=dtype)
    angles = torch.zeros(_TOO_MANY_PARALLEL, device="cuda")
    _assert_no_leak(lambda: ft.forward_project(vol, angles, N_det=1),
                    vol.numel() * vol.element_size())


@needs_cuda
def test_grad_rotmat_uncached_launch_failure_does_not_leak():
    vol = torch.zeros(1, 2048, 2048, device="cuda")
    gs = torch.zeros(_TOO_MANY_ROTMAT, 1, 2, device="cuda")
    mats = torch.zeros(_TOO_MANY_ROTMAT, 9, device="cuda")
    _assert_no_leak(lambda: _grad_rotmat_raw(vol, gs, mats), vol.numel() * 4)


@needs_cuda
@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("kind,dtype,channels,dims", [
    (2, 0, 1, (64, 256, 256)),     # TC_VOL_3D fp32, also builds the point texture
    (0, 1, 4, (64, 256, 512)),     # TC_VOL_LAYERED fp16 4-channel, builds a surface
    (1, 0, 1, (512, 64, 128)),     # TC_SINO_LAYERED fp32
])
def test_tc_create_failure_does_not_leak(stage, kind, dtype, channels, dims):
    """Inject a failure at each throw site inside tc_create (after the array,
    after the texture, after every handle) and check nothing is left behind."""
    import fasttomo._core as _core

    def fail_once():
        _core._tc_debug_fail_next_create(stage)
        try:
            h = _core.tc_create(kind, dtype, channels, *dims)
        finally:
            _core._tc_debug_fail_next_create(0)
        _core.tc_destroy(h)   # only reached if the hook did not fire

    _assert_no_leak(fail_once, dims[0] * dims[1] * dims[2] * (4 if dtype == 0 else 2))


@needs_cuda
def test_texture_limits_raise_clear_error_without_leak():
    """P4: more than 2048 layers is a clear ValueError, not a generic CUDA
    'invalid argument', and it leaks nothing."""
    vol = torch.zeros(2049, 8, 8, device="cuda")
    with pytest.raises(ValueError, match="limit"):
        ft.forward_project(vol, torch.zeros(4, device="cuda"))
    sino = torch.zeros(2049, 1, 8, device="cuda")
    with pytest.raises(ValueError, match="limit"):
        ft.back_project(sino, torch.zeros(2049, device="cuda"))
    _assert_no_leak(lambda: ft.forward_project(vol, torch.zeros(4, device="cuda")),
                    vol.numel() * 4)
    # audit regression 6: BP with n_angles = 13000 (also past the 96 KB smem cap)
    big = torch.zeros(13000, 1, 8, device="cuda")
    angles = torch.zeros(13000, device="cuda")
    _assert_no_leak(lambda: ft.back_project(big, angles), big.numel() * 4, n_calls=10)
    _assert_no_leak(lambda: ft.back_project_3d(big, torch.zeros(13000, 3, device="cuda")),
                    big.numel() * 4, n_calls=10)


def _read_on_side_stream(t: torch.Tensor) -> torch.Tensor:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out = t.clone()
    torch.cuda.synchronize()
    return out


@needs_cuda
def test_dropping_projector_with_work_in_flight_keeps_results():
    """P1: launch FP/BP, drop the projector (its textures) at once, then read
    the outputs on a side stream. They must equal an ordinary synchronous run."""
    import gc

    torch.manual_seed(0)
    N, n_ang = 192, 180
    vol = torch.rand(N, N, N, device="cuda")
    angles = torch.linspace(-70, 70, n_ang, device="cuda")
    euler = torch.stack([torch.zeros(n_ang, device="cuda"), angles,
                         torch.zeros(n_ang, device="cuda")], dim=1)
    mats = _euler_zyz_to_rotmat(euler).reshape(n_ang, 9).contiguous()

    ref_proj = ft.RotmatProjector((N, N, N), n_angles=n_ang, device="cuda")
    ref_sino = ref_proj.forward(vol, mats)
    ref_back = ref_proj.back(ref_sino, mats)
    ref_psino = ft.ParallelProjector((N, N, N), angles).forward(vol)
    torch.cuda.synchronize()

    proj = ft.RotmatProjector((N, N, N), n_angles=n_ang, device="cuda")
    sino = proj.forward(vol, mats)
    back = proj.back(sino, mats)
    pproj = ft.ParallelProjector((N, N, N), angles)
    psino = pproj.forward(vol)
    del proj, pproj
    gc.collect()
    assert torch.equal(_read_on_side_stream(sino), ref_sino)
    assert torch.equal(_read_on_side_stream(back), ref_back)
    assert torch.equal(_read_on_side_stream(psino), ref_psino)


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_one_shot_wrappers_results_readable_on_side_stream(dtype):
    """P1 on the functional (one-shot texture) path: outputs read on another
    stream right after the call equal the cached projector's outputs."""
    torch.manual_seed(0)
    N, n_ang = 128, 90
    vol = torch.rand(N, N, N, device="cuda").to(dtype)
    angles = torch.linspace(-70, 70, n_ang, device="cuda")
    proj = ft.ParallelProjector((N, N, N), angles)
    ref = proj.forward(vol)
    ref_bp = proj.back(ref)
    torch.cuda.synchronize()
    sino = ft.forward_project(vol, angles)
    assert torch.equal(_read_on_side_stream(sino), ref)
    bp = ft.back_project(sino, angles, Nz=N)
    assert torch.equal(_read_on_side_stream(bp), ref_bp)


@needs_cuda
def test_run_sirt_internal_projector_result_is_complete():
    """P1: run_sirt drops its internal projector on return while the last
    FP/BP may still be queued. The result read on a side stream must equal a
    second, fully synchronized run (the kernels are deterministic)."""
    torch.manual_seed(0)
    N = 96
    angles = torch.linspace(-70, 70, 60, device="cuda")
    sino = ft.forward_project(torch.rand(N, N, N, device="cuda"), angles)
    a = ft.run_sirt(sino, angles, n_iter=5)
    a = _read_on_side_stream(a)
    b = ft.run_sirt(sino, angles, n_iter=5)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


def _capture_release_attempt() -> tuple[list[int], int]:
    """One run of the deferred-release scenario. Returns the three memory
    readings (held before capture, still held after capture, freed by the
    drain) and the texture bytes they should each exceed 75 % of. The
    structural checks (two deferred handles, empty after drain, graph
    replays) are asserted directly: they do not depend on other GPU users."""
    import gc

    from fasttomo import projector as projector_mod

    N = 128
    angles = torch.linspace(-60, 60, 256, device="cuda")
    vol = torch.rand(N, N, N, device="cuda")
    free0 = _driver_free_bytes()
    proj = ft.ParallelProjector((N, N, N), angles)
    sino = proj.forward(vol)
    proj.back(sino)                        # both caches exist
    arrays = (N * N * N + 256 * N * N) * 4  # vol + sino cudaArrays: 24 MiB
    del sino
    held = free0 - _driver_free_bytes()

    x = torch.zeros(16, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x.add_(1)
        del proj
        gc.collect()
        assert len(projector_mod._deferred_handles) == 2
    still_held = free0 - _driver_free_bytes()
    projector_mod._drain_deferred_handles()
    assert not projector_mod._deferred_handles
    freed = still_held - (free0 - _driver_free_bytes())
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(x, torch.ones_like(x))
    return [held, still_held, freed], arrays


@needs_cuda
def test_projector_collected_during_graph_capture_is_freed_after_capture():
    """A projector garbage-collected while a CUDA graph is being captured
    cannot free its textures then (the device sync / cudaFreeArray would break
    the capture). The release is deferred, and _drain_deferred_handles (called
    by run_sirt right after its capture) frees them.

    Memory is read at driver level and the card may be shared, so each
    reading must exceed 75 % of the 24 MiB of textures (held before the
    capture, still held after it, freed by the drain), and like the leak
    tests the scenario is retried: it passes if any attempt meets all three
    bounds; a real regression misses them on every attempt."""
    seen = []
    for _ in range(3):
        readings, arrays = _capture_release_attempt()
        seen.append([r // _MiB for r in readings])
        if all(r > 0.75 * arrays for r in readings):
            return
    pytest.fail(f"deferred release bounds missed on every attempt: {seen} MiB "
                f"(held, still held, freed; each must exceed {0.75 * arrays / _MiB:.0f} MiB)")


# ── P3: strided inputs give the same result as contiguous ones ─────────────

def _variants(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Non-contiguous tensors with the same shape as ``x``."""
    t = x.transpose(1, 2).contiguous().transpose(1, 2)       # permuted strides
    big = torch.zeros(*x.shape[:-1], 2 * x.shape[-1], dtype=x.dtype, device=x.device)
    big[..., ::2] = x
    sl = big[..., ::2]                                         # sliced, stride 2
    ex = x[:1].expand_as(x)                                    # stride-0 broadcast
    for v in (t, sl, ex):
        assert not v.is_contiguous()
    return {"transposed": t, "sliced": sl, "expanded": ex}


def _devices():
    return ["cpu"] + (["cuda"] if HAS_CUDA else [])


def _ops(device):
    """name -> (make_inputs, run) for every projection op and its gradients."""
    g = torch.Generator().manual_seed(7)
    Ny, Nz, Nx, n = 5, 12, 10, 9
    vol = torch.rand(Ny, Nz, Nx, generator=g).to(device)
    sino = torch.rand(n, Ny, Nx, generator=g).to(device)
    angles = torch.linspace(-60, 55, n).to(device)
    euler = torch.stack([torch.linspace(-10, 10, n), torch.linspace(-60, 55, n),
                         torch.linspace(5, -5, n)], dim=1).to(device)

    def grad_angle(v):
        a = angles.clone().requires_grad_(True)
        ft.forward_project(v, a).mul(sino).sum().backward()
        return a.grad

    def grad_euler(v):
        e = euler.clone().requires_grad_(True)
        ft.forward_project_3d(v, e).mul(sino).sum().backward()
        return e.grad

    return {
        "forward_project": (vol, lambda v: ft.forward_project(v, angles)),
        "back_project": (sino, lambda s: ft.back_project(s, angles, Nz=Nz)),
        "forward_project_3d": (vol, lambda v: ft.forward_project_3d(v, euler)),
        "back_project_3d": (sino, lambda s: ft.back_project_3d(s, euler, Nz=Nz)),
        "grad_angle": (vol, grad_angle),
        "grad_rotmat": (vol, grad_euler),
    }


_OP_NAMES = ["forward_project", "back_project", "forward_project_3d",
             "back_project_3d", "grad_angle", "grad_rotmat"]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("op", _OP_NAMES)
@pytest.mark.parametrize("layout", ["transposed", "sliced", "expanded"])
def test_strided_input_matches_contiguous(device, op, layout):
    x, run = _ops(device)[op]
    v = _variants(x)[layout]
    out = run(v)
    ref = run(v.contiguous())
    assert out.shape == ref.shape and out.dtype == ref.dtype
    if device == "cpu":
        assert torch.equal(out, ref)
    else:   # CUDA angle/matrix gradients use atomics: order may differ
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _devices())
def test_strided_angles_and_fp16_on_cpu_and_cuda(device):
    """Strided angle tables and fp16 volumes (upcast on CPU) are densified too."""
    g = torch.Generator().manual_seed(3)
    vol = torch.rand(4, 9, 9, generator=g).to(device)
    angles_big = torch.linspace(-60, 60, 16).to(device)
    a = angles_big[::2]
    assert not a.is_contiguous()
    assert torch.equal(ft.forward_project(vol, a), ft.forward_project(vol, a.contiguous()))
    h = _variants(vol.half())["transposed"]
    out = ft.forward_project(h, a)
    assert out.dtype == torch.float16
    assert torch.equal(out, ft.forward_project(h.contiguous(), a))


def test_mbir_cpu_strided_sinogram_matches_contiguous():
    torch.manual_seed(0)
    angles = torch.linspace(-60, 60, 12)
    vol = torch.zeros(4, 12, 12)
    vol[:, 3:9, 3:9] = 1.0
    sino = ft.forward_project(vol, angles)
    strided = _variants(sino)["transposed"]
    kw = dict(n_iter=2, multires_levels=1, seed=3)
    assert torch.equal(ft.run_mbir(strided, angles, **kw),
                       ft.run_mbir(strided.contiguous(), angles, **kw))


@needs_cuda
@pytest.mark.parametrize("layout", ["fp16", "transposed"])
def test_grad_angle_cuda_fp16_and_strided_match_cpu(layout):
    """D1: the CUDA tilt-angle kernel is fp32/dense-only; fp16 and strided
    volumes must be densified before their pointer is taken."""
    torch.manual_seed(0)
    vol = torch.rand(6, 16, 16)
    angles = torch.linspace(-50, 50, 8)
    w = torch.randn(8, 6, 16)

    def grad(v, dev):
        a = angles.clone().to(dev).requires_grad_(True)
        ft.forward_project(v, a).float().mul(w.to(dev)).sum().backward()
        return a.grad.cpu()

    ref = grad(vol, "cpu")
    v = vol.cuda()
    v = v.half() if layout == "fp16" else _variants(v)["transposed"]
    ref_same_input = grad(v.float().contiguous().cpu(), "cpu")
    out = grad(v, "cuda")
    assert torch.isfinite(out).all()
    assert _rel(out, ref_same_input) < 1e-3
    if layout == "transposed":
        assert _rel(out, ref) < 1e-3


# ── GENFIRE CPU gridding: bit-identical to the pre-change binding ──────────

def test_genfire_gridding_cpu_bit_identical_to_numpy_binding():
    """The CPU gridding binding moved from py::array_t to data_ptr ints. The
    golden file holds inputs and outputs produced by the previous (numpy)
    binding on origin/main; the new call path must reproduce them exactly
    (GENFIRE is frozen)."""
    import pathlib

    import fasttomo._core as _core

    d = np.load(pathlib.Path(__file__).parent / "data" / "genfire_gridding_cpu_golden.npz")
    fk_r = torch.from_numpy(d["fk_real"])
    fk_i = torch.from_numpy(d["fk_imag"])
    cos_a = torch.from_numpy(d["cos_a"])
    sin_a = torch.from_numpy(d["sin_a"])
    n_angles, Ny, bigN = fk_r.shape
    grids = [torch.zeros(Ny, bigN, bigN, dtype=torch.float32) for _ in range(3)]
    _core.genfire_gridding_2d_cpu(
        fk_r.data_ptr(), fk_i.data_ptr(), cos_a.data_ptr(), sin_a.data_ptr(),
        *(t.data_ptr() for t in grids), bigN, Ny, n_angles, float(d["cutoff_dist"]),
    )
    for t, key in zip(grids, ("grid_real", "grid_imag", "weight_grid")):
        assert torch.equal(t, torch.from_numpy(d[key])), key


# ── B1: size mismatches raise ValueError before any pointer is taken ───────
#
# The kernels get bare pointers plus dims, so an input smaller than the dims
# claim is read out of bounds (NaN, garbage or a segfault). Every raw op
# checks every tensor's shape in `_launch`.

def _bad_calls(device):
    from fasttomo import autograd as ag

    d = device
    vol = torch.rand(3, 8, 8, device=d)
    sino = torch.rand(5, 3, 8, device=d)
    ang5 = torch.linspace(-40, 40, 5, device=d)
    mats5 = _euler_zyz_to_rotmat(torch.stack([ang5 * 0, ang5, ang5 * 0], 1)).reshape(5, 9)
    return {
        "bp_parallel: too few angles":
            lambda: ag._back_project_parallel_raw(sino, ang5[:1]),
        "bp_parallel: too many angles":
            lambda: ag._back_project_parallel_raw(sino, torch.zeros(9, device=d)),
        "bp_parallel: 2-D sinogram":
            lambda: ag._back_project_parallel_raw(sino[0], ang5),
        "fp_parallel: 4-D volume":
            lambda: ag._forward_project_parallel_raw(vol[None], ang5),
        "grad_angle: angles vs grad_sino":
            lambda: ag._grad_angle_raw(vol, sino, ang5[:2]),
        "grad_angle: grad_sino Ny":
            lambda: ag._grad_angle_raw(vol, sino[:, :2], ang5),
        "bp_rotmat: fewer matrices than views":
            lambda: ag._back_project_rotmat_raw(sino, mats5[:1]),
        "bp_rotmat: more matrices than views":
            lambda: ag._back_project_rotmat_raw(sino[:2], mats5),
        "fp_rotmat: matrices not [n, 9]":
            lambda: ag._forward_project_rotmat_raw(vol, mats5.reshape(-1)[:40]),
        "grad_rotmat: grad_sino views":
            lambda: ag._grad_rotmat_raw(vol, sino[:2], mats5),
        "grad_rotmat: grad_sino Ny":
            lambda: ag._grad_rotmat_raw(vol, sino[:, :2], mats5),
        "public back_project_3d (review repro)":
            lambda: ft.back_project_3d(torch.ones(4000, 2, 8, device=d),
                                       torch.zeros(1, 3, device=d)),
        "public back_project (review repro)":
            lambda: ft.back_project(torch.ones(4000, 2, 8, device=d),
                                    torch.zeros(1, device=d)),
    }


_BAD_NAMES = list(_bad_calls("cpu"))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("case", _BAD_NAMES)
def test_raw_ops_reject_mismatched_sizes(device, case):
    with pytest.raises(ValueError):
        _bad_calls(device)[case]()


def _sweep_args(Ny=2, Nz=4, Nx=4, n_angles=3):
    from fasttomo.autograd import _trig_on_device
    cos_a, sin_a = _trig_on_device(torch.linspace(-30, 30, n_angles), torch.device("cpu"))
    return dict(
        volume=torch.zeros(Ny, Nz, Nx), e_sino=torch.ones(n_angles, Ny, Nx),
        lambdas=None,
        vol_snap=torch.zeros(Ny, Nz, Nx),
        perm=torch.arange(Nz * Nx, dtype=torch.int32).repeat(Ny, 1),
        cos_a=cos_a, sin_a=sin_a)


@pytest.mark.parametrize("bad", ["e_sino_Ny", "angles", "lambdas", "perm", "snapshot"])
def test_mbir_cpu_sweep_rejects_mismatched_sizes(bad):
    """The #58 size checks (e_sino Ny, angles) and the others. (d_offsets is
    no longer passed to the C sweep: the offset update moved to Python, P12.)"""
    from fasttomo.algorithms import _mbir_icd_sweep_cpu

    kw = _sweep_args()
    if bad == "e_sino_Ny":
        kw["e_sino"] = torch.ones(3, 1, 4)
    elif bad == "angles":
        kw["cos_a"], kw["sin_a"] = kw["cos_a"][:2], kw["sin_a"][:2]
    elif bad == "lambdas":
        kw["lambdas"] = torch.ones(3, 2, 3)
    elif bad == "perm":
        kw["perm"] = kw["perm"][:, :5].contiguous()
    else:
        kw["vol_snap"] = torch.zeros(2, 4, 3)
    hyper = dict(qggmrf_p=1.2, qggmrf_q=2.0, qggmrf_c=0.01, sigma_f=1.0,
                 huber_T=float("inf"), huber_delta=1.0, sigma=1.0, num_threads=1)
    with pytest.raises(ValueError):
        _mbir_icd_sweep_cpu(kw["volume"], kw["e_sino"], kw["lambdas"],
                            kw["vol_snap"], kw["perm"], kw["cos_a"], kw["sin_a"], **hyper)
    _mbir_icd_sweep_cpu(*(_sweep_args()[k] for k in
                          ("volume", "e_sino", "lambdas", "vol_snap",
                           "perm", "cos_a", "sin_a")), **hyper)   # valid call runs


# ── Parallel-beam path: tight bounds are consistent end to end ─────────────
#
# Unlike the rotmat path (D2), the single-axis operators use *tight* bounds
# on every backend: a sample contributes only if its whole bilinear stencil
# lies inside the slice, fx in [0, Nx-1), fz in [0, Nz-1). The CUDA forward
# skips out-of-stencil samples explicitly before its (border-mode) texture
# fetch, the CPU forward does the same, and grad_angle (CPU and CUDA) is the
# derivative of exactly that forward. These tests pin that consistency on
# volumes that touch every face, with the D2 methodology: an independent fp64
# reference, central finite differences, generic angles.

def _fp_parallel_ref(vol: np.ndarray, angles_deg: np.ndarray, N_det: int) -> np.ndarray:
    """fp64 single-axis forward: unit steps, bilinear, tight stencil bounds."""
    Ny, Nz, Nx = vol.shape
    ray_len = int(math.ceil(0.5 * math.sqrt(Nx * Nx + Nz * Nz))) + 1
    hd, hx, hz = 0.5 * (N_det - 1), 0.5 * (Nx - 1), 0.5 * (Nz - 1)
    T, S = np.meshgrid(np.arange(N_det) - hd,
                       np.arange(-ray_len, ray_len + 1, dtype=np.float64), indexing="ij")
    out = np.zeros((len(angles_deg), Ny, N_det))
    for a, th in enumerate(np.deg2rad(angles_deg)):
        fx = T * math.cos(th) - S * math.sin(th) + hx
        fz = T * math.sin(th) + S * math.cos(th) + hz
        inside = (fx >= 0) & (fx < Nx - 1) & (fz >= 0) & (fz < Nz - 1)
        fx = np.where(inside, fx, 0.0)
        fz = np.where(inside, fz, 0.0)
        x0 = np.floor(fx).astype(int)
        z0 = np.floor(fz).astype(int)
        wx, wz = fx - x0, fz - z0
        x1 = np.minimum(x0 + 1, Nx - 1)
        z1 = np.minimum(z0 + 1, Nz - 1)
        val = ((1 - wz) * ((1 - wx) * vol[:, z0, x0] + wx * vol[:, z0, x1])
               + wz * ((1 - wx) * vol[:, z1, x0] + wx * vol[:, z1, x1]))
        out[a] = np.where(inside[None], val, 0.0).sum(-1)
    return out


_PAR_ANGLES = np.array([-47.3, -12.9, 8.1, 33.7, 61.2])


def _parallel_case(seed=1, shape=(3, 11, 12)):
    g = torch.Generator().manual_seed(seed)
    vol = torch.rand(*shape, generator=g) + 0.5             # non-zero on every face
    w = torch.randn(len(_PAR_ANGLES), shape[0], shape[2], generator=g)
    eps = 1e-5
    fd = np.zeros(len(_PAR_ANGLES))
    for i in range(len(_PAR_ANGLES)):
        e = eps * (np.arange(len(_PAR_ANGLES)) == i)
        lp = (w.double().numpy() * _fp_parallel_ref(vol.double().numpy(), _PAR_ANGLES + e, shape[2])).sum()
        lm = (w.double().numpy() * _fp_parallel_ref(vol.double().numpy(), _PAR_ANGLES - e, shape[2])).sum()
        fd[i] = (lp - lm) / (2 * eps)
    return vol, w, fd


@pytest.mark.parametrize("device", _devices())
def test_grad_angle_matches_finite_difference(device):
    from fasttomo.autograd import _grad_angle_raw

    vol, w, fd = _parallel_case()
    angles = torch.tensor(_PAR_ANGLES, dtype=torch.float32, device=device)
    g = _grad_angle_raw(vol.to(device), w.to(device), angles)
    assert g.shape == (len(_PAR_ANGLES),)
    assert _rel(g, fd) < 1e-4


@pytest.mark.parametrize("kind", ["ones", "rand"])
def test_forward_project_parallel_cpu_matches_reference_on_box_filling_volume(kind):
    vol = torch.ones(3, 11, 12) if kind == "ones" else _parallel_case()[0]
    angles = torch.tensor(_PAR_ANGLES, dtype=torch.float32)
    ref = _fp_parallel_ref(vol.double().numpy(), angles.double().numpy(), 12)
    assert _rel(ft.forward_project(vol, angles), ref) < 1e-5


@needs_cuda
@pytest.mark.parametrize("kind", ["ones", "rand"])
def test_forward_project_parallel_cpu_cuda_parity_on_box_filling_volume(kind):
    """CPU and CUDA parallel forwards agree on a box-filling volume up to the
    texture unit's 8-bit filter weights (~3e-4 relative)."""
    vol = torch.ones(3, 11, 12) if kind == "ones" else _parallel_case()[0]
    angles = torch.tensor(_PAR_ANGLES, dtype=torch.float32)
    cpu = ft.forward_project(vol, angles)
    gpu = ft.forward_project(vol.cuda(), angles.cuda()).cpu()
    assert _rel(gpu, cpu) < 2e-3
