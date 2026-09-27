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

CUDA tests skip automatically when no GPU is visible; the CPU tests always run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import fasttomo as ft
from fasttomo.autograd import _euler_zyz_to_rotmat, _grad_rotmat_raw

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


# ── D5 / P1 / P4: texture-cache lifetime on the CUDA one-shot paths ─────────

_MiB = 1 << 20


def _driver_free_bytes() -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0]


def _assert_no_leak(fail_once, n_calls=200, tol=64 * _MiB):
    """Call ``fail_once`` (which must raise) n_calls times and check the
    driver-level free memory does not drop. Each call allocates a >= 8 MiB
    cudaArray before throwing, so a leak shows up as >= 1.6 GB."""
    with pytest.raises((RuntimeError, ValueError)):
        fail_once()                      # warm-up: allocator pools, module init
    before = _driver_free_bytes()
    for _ in range(n_calls):
        with pytest.raises((RuntimeError, ValueError)):
            fail_once()
    after = _driver_free_bytes()
    assert before - after < tol, f"leaked {(before - after) / _MiB:.0f} MiB"


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
    _assert_no_leak(lambda: ft.forward_project_3d(vol, euler, N_det=2))


@needs_cuda
@pytest.mark.parametrize("dtype,Ny", [(torch.float32, 1),
                                      (torch.float16, 4),    # 4-channel packed texture
                                      (torch.float16, 1)])   # 1-channel fp16 texture
def test_forward_project_launch_failure_does_not_leak(dtype, Ny):
    vol = torch.zeros(Ny, 2048, 1024, device="cuda", dtype=dtype)
    angles = torch.zeros(_TOO_MANY_PARALLEL, device="cuda")
    _assert_no_leak(lambda: ft.forward_project(vol, angles, N_det=1))


@needs_cuda
def test_grad_rotmat_uncached_launch_failure_does_not_leak():
    vol = torch.zeros(1, 2048, 2048, device="cuda")
    gs = torch.zeros(_TOO_MANY_ROTMAT, 1, 2, device="cuda")
    mats = torch.zeros(_TOO_MANY_ROTMAT, 9, device="cuda")
    _assert_no_leak(lambda: _grad_rotmat_raw(vol, gs, mats))


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

    _assert_no_leak(fail_once)


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
    _assert_no_leak(lambda: ft.forward_project(vol, torch.zeros(4, device="cuda")))


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


@needs_cuda
def test_projector_collected_during_graph_capture_is_freed_after_capture():
    """A projector garbage-collected while a CUDA graph is being captured
    cannot free its textures then (the device sync / cudaFreeArray would break
    the capture). The release is deferred, and _drain_deferred_handles (called
    by run_sirt right after its capture) frees them."""
    import gc

    from fasttomo import projector as projector_mod

    N = 128
    angles = torch.linspace(-60, 60, 256, device="cuda")
    vol = torch.rand(N, N, N, device="cuda")
    free0 = _driver_free_bytes()
    proj = ft.ParallelProjector((N, N, N), angles)
    sino = proj.forward(vol)
    proj.back(sino)                        # both caches exist: ~8 + 16 MiB of arrays
    del sino
    held = free0 - _driver_free_bytes()
    assert held > 16 * _MiB

    x = torch.zeros(16, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x.add_(1)
        del proj
        gc.collect()
        assert len(projector_mod._deferred_handles) == 2
    assert free0 - _driver_free_bytes() > 16 * _MiB   # still held after capture
    projector_mod._drain_deferred_handles()
    assert not projector_mod._deferred_handles
    assert free0 - _driver_free_bytes() < 8 * _MiB
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(x, torch.ones_like(x))
