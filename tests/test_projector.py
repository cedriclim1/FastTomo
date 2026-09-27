"""
The cached projectors (ParallelProjector, RotmatProjector): input
normalisation, geometry and dtype checks, fp16 textures, gradients, and
device handling on multi-GPU machines.

Cached-vs-functional agreement lives in test_parity.py.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    make_angles,
    needs_cuda,
    rel_err,
)

import fasttomo as ft
from fasttomo.autograd import _euler_zyz_to_rotmat

# ── RotmatProjector mats normalisation (CPU-testable) ─────────────────────────


def _bare_rotmat_projector(n_angles: int):
    """RotmatProjector without __init__ (which needs CUDA): enough to test
    the pure-PyTorch mats normalisation on CPU."""
    proj = ft.RotmatProjector.__new__(ft.RotmatProjector)
    proj.n_angles = n_angles
    return proj


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.bfloat16])
def test_rotmat_projector_normalize_casts_mats_to_fp32(dtype):
    """The rotmat kernels read mats as float32; other float dtypes must be
    cast, not reinterpreted."""
    n_angles = 5
    proj = _bare_rotmat_projector(n_angles)
    mats = torch.eye(3, dtype=dtype).unsqueeze(0).expand(n_angles, -1, -1).contiguous()
    mats_flat = proj._normalize_and_validate_mats(mats)
    assert mats_flat.shape == (n_angles, 9)
    assert mats_flat.dtype == torch.float32
    assert mats_flat.is_contiguous()


def test_rotmat_projector_normalize_makes_mats_contiguous():
    """Strided mats (expand, transpose views) must come out dense with the
    same logical values."""
    n_angles = 4
    proj = _bare_rotmat_projector(n_angles)
    ref = torch.randn(n_angles, 9)

    strided_2d = ref.t().contiguous().t()          # [n, 9], strides (1, n)
    assert not strided_2d.is_contiguous()
    out = proj._normalize_and_validate_mats(strided_2d)
    assert out.is_contiguous() and torch.equal(out, ref)

    expanded = torch.eye(3).unsqueeze(0).expand(n_angles, -1, -1)
    assert not expanded.is_contiguous()
    out = proj._normalize_and_validate_mats(expanded)
    assert out.is_contiguous()
    assert torch.equal(out, torch.eye(3).reshape(1, 9).expand(n_angles, 9))


def test_rotmat_projector_normalize_keeps_grad_path():
    """Casting and densifying must not cut the autograd graph to the pose."""
    n_angles = 3
    proj = _bare_rotmat_projector(n_angles)
    mats = torch.randn(n_angles, 3, 3, dtype=torch.float64, requires_grad=True)
    proj._normalize_and_validate_mats(mats).sum().backward()
    assert mats.grad is not None and mats.grad.dtype == torch.float64


def test_rotmat_projector_normalize_rejects_non_floating_mats():
    n_angles = 3
    proj = _bare_rotmat_projector(n_angles)
    mats_int = torch.zeros(n_angles, 3, 3, dtype=torch.int32)
    with pytest.raises(TypeError, match="floating tensor"):
        proj._normalize_and_validate_mats(mats_int)


DEVICE = torch.device("cuda")


def _rotmat_setup(N: int = 16, n_angles: int = 8, seed: int = 7):
    from fasttomo.autograd import _euler_zyz_to_rotmat

    torch.manual_seed(seed)
    volume = torch.randn(N, N, N, device=DEVICE, dtype=torch.float32)
    grad_sino = torch.randn(n_angles, N, N, device=DEVICE, dtype=torch.float32)
    euler = torch.zeros(n_angles, 3, device=DEVICE)
    euler[:, 0] = torch.linspace(-10.0, 10.0, n_angles, device=DEVICE)
    euler[:, 1] = torch.linspace(-50.0, 50.0, n_angles, device=DEVICE)
    mats_flat = _euler_zyz_to_rotmat(euler).reshape(n_angles, 9).contiguous().float()
    return volume, grad_sino, mats_flat


@needs_cuda
def test_rotmat_projector_fp64_mats_match_fp32():
    """fp64 mats must be cast, not reinterpreted as a float32 buffer."""
    N, n_angles = 16, 8
    volume, _, mats32 = _rotmat_setup(N, n_angles)
    proj = ft.RotmatProjector((N, N, N), n_angles=n_angles, device=DEVICE)
    sino32 = proj.forward(volume, mats32)
    sino64 = proj.forward(volume, mats32.double())
    torch.testing.assert_close(sino64, sino32, rtol=1e-5, atol=1e-5)

    sino = torch.randn_like(sino32)
    torch.testing.assert_close(
        proj.back(sino, mats32.double()), proj.back(sino, mats32), rtol=1e-5, atol=1e-5
    )


@needs_cuda
def test_rotmat_projector_strided_mats_forward_and_pose_grad():
    """A non-contiguous mats view must give the same projection and the same
    pose gradient as its dense copy, through RotmatProjector.forward."""
    N, n_angles = 16, 8
    volume, grad_sino, mats32 = _rotmat_setup(N, n_angles)
    proj = ft.RotmatProjector((N, N, N), n_angles=n_angles, device=DEVICE)

    dense = mats32.clone().requires_grad_(True)
    base_t = mats32.t().contiguous().requires_grad_(True)   # [9, n_angles]
    strided = base_t.t()                                     # [n_angles, 9] view
    assert not strided.is_contiguous()

    sino_d = proj.forward(volume, dense)
    (sino_d * grad_sino).sum().backward()
    sino_s = proj.forward(volume, strided)
    (sino_s * grad_sino).sum().backward()

    torch.testing.assert_close(sino_s, sino_d, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(base_t.grad.t(), dense.grad, rtol=1e-4, atol=1e-4)


@needs_cuda
def test_rotmat_grad_paths_accept_strided_mats():
    """Both backward gradient kernels (the cached operator's
    _grad_mats_cached and the uncached _grad_rotmat_raw) must densify mats
    themselves: the projection Function's backward passes the saved mats
    straight through."""
    from fasttomo.autograd import _grad_rotmat_raw

    N, n_angles = 16, 8
    volume, grad_sino, mats32 = _rotmat_setup(N, n_angles)
    strided = mats32.t().contiguous().t()
    assert not strided.is_contiguous()

    ref = _grad_rotmat_raw(volume, grad_sino, mats32)
    torch.testing.assert_close(
        _grad_rotmat_raw(volume, grad_sino, strided), ref, rtol=1e-5, atol=1e-5
    )

    proj = ft.RotmatProjector((N, N, N), n_angles=n_angles, device=DEVICE)
    _ = proj.forward(volume, mats32)  # populates the cached volume texture
    torch.testing.assert_close(
        proj._op._grad_mats_cached(volume, grad_sino, strided),
        proj._op._grad_mats_cached(volume, grad_sino, mats32),
        rtol=1e-5, atol=1e-5,
    )


# ── geometry checks ───────────────────────────────────────────────────────────


def test_parallel_projector_volume_shape_mismatch_raises(backend):
    """ParallelProjector.forward must raise if volume shape != projector geometry."""
    N = 16
    angles = make_angles(10, device=backend)
    proj = ft.ParallelProjector((1, N, N), angles, device=backend)
    wrong_vol = torch.zeros(2, N, N, device=backend)  # Ny mismatch
    with pytest.raises(ValueError, match="volume must have shape"):
        proj.forward(wrong_vol)


def test_parallel_projector_sinogram_shape_mismatch_raises(backend):
    """ParallelProjector.back must raise if sinogram shape != projector geometry."""
    N = 16
    n_angles = 10
    angles = make_angles(n_angles, device=backend)
    proj = ft.ParallelProjector((1, N, N), angles, device=backend)
    wrong_sino = torch.zeros(n_angles, 2, N, device=backend)  # Ny mismatch
    with pytest.raises(ValueError, match="sinogram must have shape"):
        proj.back(wrong_sino)


def test_rotmat_projector_volume_shape_mismatch_raises(backend):
    """RotmatProjector.forward must raise if volume shape != projector geometry."""
    N = 16
    n_angles = 10
    proj = ft.RotmatProjector((1, N, N), n_angles=n_angles, device=backend)
    mats = torch.eye(3, device=backend).unsqueeze(0).expand(n_angles, -1, -1).contiguous()
    wrong_vol = torch.zeros(2, N, N, device=backend)  # Ny mismatch
    with pytest.raises(ValueError, match="volume must have shape"):
        proj.forward(wrong_vol, mats)


def test_rotmat_projector_mats_shape_mismatch_raises(backend):
    """RotmatProjector must raise on bad mats shape."""
    N = 16
    n_angles = 10
    proj = ft.RotmatProjector((1, N, N), n_angles=n_angles, device=backend)
    vol = torch.zeros(1, N, N, device=backend)
    wrong_mats = torch.eye(3, device=backend).unsqueeze(0).expand(5, -1, -1).contiguous()
    with pytest.raises(ValueError, match="mats must have shape"):
        proj.forward(vol, wrong_mats)


@pytest.mark.parametrize("make", [
    lambda: ft.ParallelProjector((2, 6, -1), torch.zeros(5), device="cpu"),
    lambda: ft.RotmatProjector((2, 6, 6), n_angles=5, device="cpu", N_det=-4),
    lambda: ft.ParallelProjector((2, 0, 6), torch.zeros(5), device="cpu"),
])
def test_cpu_projectors_reject_non_positive_geometry(make):
    with pytest.raises(ValueError, match="at least 1"):
        make()


def test_functional_negative_sizes_mean_default():
    """N_det / Nz / Nx < 0 select the defaults (any negative value, as
    before the operator seam)."""
    v = torch.rand(2, 8, 8)
    a = torch.linspace(-30, 30, 5)
    assert torch.equal(ft.forward_project(v, a, N_det=-7), ft.forward_project(v, a))
    s = ft.forward_project(v, a)
    assert torch.equal(ft.back_project(s, a, Nz=-3, Nx=-9), ft.back_project(s, a))


@needs_cuda
@pytest.mark.parametrize("make", [
    lambda: ft.ParallelProjector((2, 0, 6), torch.zeros(5, device="cuda")),
    lambda: ft.ParallelProjector((2, 6, 6), torch.zeros(0, device="cuda")),
    lambda: ft.ParallelProjector((2, 6, 6), torch.zeros(5, device="cuda"), N_det=0),
    lambda: ft.RotmatProjector((0, 6, 6), n_angles=5, device="cuda"),
    lambda: ft.RotmatProjector((2, 6, 6), n_angles=5, device="cuda", N_det=0),
])
def test_projectors_reject_zero_size_geometry(make):
    with pytest.raises(ValueError, match="at least 1"):
        make()


# ── CPU projectors (additive since the operator seam) ─────────────────────────


def _cpu_workload():
    g = torch.Generator().manual_seed(3)
    Ny, Nz, Nx, n = 3, 10, 13, 7
    vol = torch.rand(Ny, Nz, Nx, generator=g)
    sino = torch.rand(n, Ny, 15, generator=g)
    ang = torch.linspace(-50, 50, n)
    eul = torch.stack([torch.linspace(-5, 5, n), ang, torch.linspace(4, -4, n)], 1)
    return vol, sino, ang, eul


def test_cpu_parallel_projector_matches_functional():
    """ParallelProjector(device="cpu") runs the CPU kernels: forward, back
    and both gradients equal the functional API bit for bit (N_det != Nx,
    Nz != Nx)."""
    vol, sino, ang, _ = _cpu_workload()
    proj = ft.ParallelProjector(tuple(vol.shape), ang, device="cpu", N_det=15, num_threads=2)
    assert proj.device == torch.device("cpu") and proj.dtype is None
    fp = proj.forward(vol)
    assert fp.shape == (7, 3, 15) and fp.dtype == torch.float32 and proj.dtype == torch.float32
    assert torch.equal(fp, ft.forward_project(vol, ang, N_det=15))
    bp = proj.back(sino)
    assert torch.equal(bp, ft.back_project(sino, ang, Nz=10, Nx=13))
    v = vol.clone().requires_grad_(True)
    (proj.forward(v) * sino).sum().backward()
    assert torch.equal(v.grad, bp)
    s = sino.clone().requires_grad_(True)
    (proj.back(s) * vol).sum().backward()
    assert torch.equal(s.grad, fp)
    assert proj._sirt_graph_cache is None          # run_sirt never captures on CPU


def test_cpu_rotmat_projector_matches_functional():
    vol, sino, _, eul = _cpu_workload()
    mats = ft.euler_zyz_to_rotmat(eul)
    proj = ft.RotmatProjector(tuple(vol.shape), n_angles=7, device="cpu", N_det=15)
    fp = proj.forward(vol, mats)
    assert torch.equal(fp, ft.forward_project_3d(vol, eul, N_det=15))
    assert torch.equal(proj.back(sino, mats), ft.back_project_3d(sino, eul, Nz=10, Nx=13))
    v = vol.clone().requires_grad_(True)
    m = mats.clone().requires_grad_(True)
    (proj.forward(v, m) * sino).sum().backward()
    v2 = vol.clone().requires_grad_(True)
    (ft.forward_project_3d(v2, eul, N_det=15) * sino).sum().backward()
    assert torch.equal(v.grad, v2.grad)
    from fasttomo.autograd import _grad_rotmat_raw
    assert m.grad.shape == (7, 3, 3)
    assert torch.equal(m.grad.reshape(7, 9), _grad_rotmat_raw(vol, sino, mats))


def test_cpu_projector_dtype_lock_and_fp16():
    vol, sino, ang, _ = _cpu_workload()
    proj = ft.ParallelProjector(tuple(vol.shape), ang, device="cpu", N_det=15)
    out = proj.forward(vol.half())
    assert out.dtype == torch.float16
    assert torch.equal(out, ft.forward_project(vol.half(), ang, N_det=15))
    with pytest.raises(TypeError, match="dtype lock-in"):
        proj.forward(vol)
    with pytest.raises(TypeError, match="bfloat16"):
        ft.ParallelProjector(tuple(vol.shape), ang, device="cpu").forward(vol.bfloat16())


@needs_cuda
def test_cpu_projector_rejects_cuda_tensor():
    vol, _, ang, _ = _cpu_workload()
    proj = ft.ParallelProjector(tuple(vol.shape), ang, device="cpu")
    with pytest.raises(ValueError, match="projector was built on cpu"):
        proj.forward(vol.cuda())


def test_algorithms_accept_cpu_projector():
    """A CPU projector passed as ``projector=`` gives the same result as the
    algorithm's own operator."""
    N = 16
    ang = make_angles(12)
    vol = torch.zeros(2, N, N)
    vol[:, 4:12, 5:11] = 1.0
    sino = ft.forward_project(vol, ang)
    proj = ft.ParallelProjector((2, N, N), ang, device="cpu")
    assert torch.equal(ft.run_sirt(sino, ang, 4, projector=proj), ft.run_sirt(sino, ang, 4))
    assert torch.equal(ft.run_fbp(sino, ang, projector=proj), ft.run_fbp(sino, ang))
    torch.manual_seed(0)
    a = ft.run_fista(sino, ang, 3, projector=proj)
    torch.manual_seed(0)
    assert torch.equal(a, ft.run_fista(sino, ang, 3))


# ── gradients and dtypes ──────────────────────────────────────────────────────


@pytest.fixture
def workload():
    torch.manual_seed(0)
    Ny, Nz, Nx = 32, 40, 36
    n_angles = 11
    device = torch.device("cuda")
    vol = torch.randn(Ny, Nz, Nx, device=device, dtype=torch.float32)
    sino = torch.randn(n_angles, Ny, Nx, device=device, dtype=torch.float32)
    angles_deg = torch.linspace(-60.0, 60.0, n_angles, device=device, dtype=torch.float32)
    return vol, sino, angles_deg


@needs_cuda
def test_parallel_projector_grad_through_volume(workload):
    vol, sino, ang = workload
    target = sino.clone()
    proj = ft.ParallelProjector(vol.shape, ang, device=vol.device)

    v_func = vol.clone().detach().requires_grad_(True)
    pred_f = ft.forward_project(v_func, ang)
    loss_f = ((pred_f - target) ** 2).mean()
    loss_f.backward()

    v_cached = vol.clone().detach().requires_grad_(True)
    pred_c = proj.forward(v_cached)
    loss_c = ((pred_c - target) ** 2).mean()
    loss_c.backward()

    torch.testing.assert_close(v_cached.grad, v_func.grad, rtol=1e-4, atol=1e-5)


# fp16 textures are 16-bit half-precision; hardware bilinear filtering
# operates on the half values directly, accumulators stay fp32. We expect
# ~1% relative agreement with fp32 — the same precision floor as the
# CPU-vs-CUDA fp32 test, just bumped slightly because both axes lose
# fractional bits.
_FP16_REL = 1e-2  # 1% relative tolerance (TorchRadon precedent)


@needs_cuda
def test_parallel_projector_forward_fp16_matches_fp32(workload):
    vol, _, ang = workload
    sino_fp32 = ft.forward_project(vol, ang)
    proj = ft.ParallelProjector(vol.shape, ang, device=vol.device)
    sino_fp16 = proj.forward(vol.half())
    assert sino_fp16.dtype == torch.float16
    assert proj.dtype == torch.float16
    assert rel_err(sino_fp16, sino_fp32) < _FP16_REL


@needs_cuda
def test_parallel_projector_back_fp16_matches_fp32(workload):
    _, sino, ang = workload
    Ny, N_det = sino.shape[1], sino.shape[2]
    vol_fp32 = ft.back_project(sino, ang)
    proj = ft.ParallelProjector((Ny, N_det, N_det), ang, device=sino.device)
    vol_fp16 = proj.back(sino.half())
    assert vol_fp16.dtype == torch.float16
    assert rel_err(vol_fp16, vol_fp32) < _FP16_REL


@needs_cuda
def test_rotmat_projector_forward_fp16_matches_fp32(workload):
    vol, _, ang = workload
    n_angles = ang.numel()
    euler = torch.zeros(n_angles, 3, device=vol.device, dtype=torch.float32)
    euler[:, 1] = ang
    mats = _euler_zyz_to_rotmat(euler)
    sino_fp32 = ft.forward_project_3d(vol, euler, N_det=vol.shape[-1])
    proj = ft.RotmatProjector(vol.shape, n_angles, device=vol.device)
    sino_fp16 = proj.forward(vol.half(), mats)
    assert sino_fp16.dtype == torch.float16
    assert rel_err(sino_fp16, sino_fp32) < _FP16_REL


@needs_cuda
def test_rotmat_projector_back_fp16_matches_fp32(workload):
    _, sino, ang = workload
    n_angles, Ny, N_det = sino.shape
    euler = torch.zeros(n_angles, 3, device=sino.device, dtype=torch.float32)
    euler[:, 1] = ang
    mats = _euler_zyz_to_rotmat(euler)
    vol_fp32 = ft.back_project_3d(sino, euler)
    proj = ft.RotmatProjector((Ny, N_det, N_det), n_angles, device=sino.device)
    vol_fp16 = proj.back(sino.half(), mats)
    assert vol_fp16.dtype == torch.float16
    assert rel_err(vol_fp16, vol_fp32) < _FP16_REL


@needs_cuda
def test_projector_dtype_lock_in_raises(workload):
    """A projector locks its dtype on first FP/BP call. Calling with a
    mismatched dtype afterwards raises a clear TypeError."""
    vol, _, ang = workload
    proj = ft.ParallelProjector(vol.shape, ang, device=vol.device)
    _ = proj.forward(vol)              # locks fp32
    with pytest.raises(TypeError, match="dtype lock-in"):
        proj.forward(vol.half())


# ── non-default device (needs 2 GPUs) ─────────────────────────────────────────


@needs_cuda
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="cross-device tests need >=2 GPUs",
)
class TestCrossDevice:
    """Regression tests for running CUDA ops on a non-default device.

    Default device stays cuda:0 throughout; tensors live on cuda:1. Without
    the ``with torch.cuda.device(t.device)`` guards in the autograd helpers,
    projector raw paths, and projector ``__del__``, every one of these would
    crash with ``cudaErrorIllegalAddress``.
    """

    DEV = torch.device("cuda:1")

    def _angles(self, n: int = 20) -> torch.Tensor:
        return torch.linspace(-60, 60, n).to(self.DEV)

    def _phantom(self, N: int = 32) -> torch.Tensor:
        v = torch.zeros(1, N, N, device=self.DEV)
        v[:, N // 4:3 * N // 4, N // 4:3 * N // 4] = 1.0
        return v

    def _euler(self, n: int = 20) -> torch.Tensor:
        e = torch.zeros(n, 3, dtype=torch.float32, device=self.DEV)
        e[:, 1] = torch.linspace(-60, 60, n)
        return e

    # ── functional projection ops ────────────────────────────────────────

    def test_forward_project(self):
        sino = ft.forward_project(self._phantom(), self._angles())
        torch.cuda.synchronize(self.DEV)
        assert sino.device == self.DEV and torch.isfinite(sino).all()

    def test_back_project(self):
        sino = ft.forward_project(self._phantom(), self._angles())
        rec = ft.back_project(sino, self._angles())
        torch.cuda.synchronize(self.DEV)
        assert rec.device == self.DEV and torch.isfinite(rec).all()

    def test_forward_project_3d(self):
        sino = ft.forward_project_3d(self._phantom(), self._euler())
        torch.cuda.synchronize(self.DEV)
        assert sino.device == self.DEV and torch.isfinite(sino).all()

    def test_back_project_3d(self):
        sino = ft.forward_project_3d(self._phantom(), self._euler())
        rec = ft.back_project_3d(sino, self._euler())
        torch.cuda.synchronize(self.DEV)
        assert rec.device == self.DEV and torch.isfinite(rec).all()

    def test_forward_project_resire(self):
        sino = ft.forward_project_resire(self._phantom(), self._euler())
        torch.cuda.synchronize(self.DEV)
        assert sino.device == self.DEV and torch.isfinite(sino).all()

    def test_back_project_resire(self):
        sino = ft.forward_project_resire(self._phantom(), self._euler())
        rec = ft.back_project_resire(sino, self._euler())
        torch.cuda.synchronize(self.DEV)
        assert rec.device == self.DEV and torch.isfinite(rec).all()

    # ── algorithm wrappers ───────────────────────────────────────────────

    def test_run_fbp(self):
        sino = ft.forward_project(self._phantom(), self._angles())
        rec = ft.run_fbp(sino, self._angles())
        torch.cuda.synchronize(self.DEV)
        assert rec.device == self.DEV and torch.isfinite(rec).all()

    def test_run_sirt(self):
        sino = ft.forward_project(self._phantom(), self._angles())
        rec = ft.run_sirt(sino, self._angles(), n_iter=5)
        torch.cuda.synchronize(self.DEV)
        assert rec.device == self.DEV and torch.isfinite(rec).all()

    def test_run_sart(self):
        sino = ft.forward_project(self._phantom(), self._angles())
        rec = ft.run_sart(sino, self._angles(), n_iter=2, subset_size=5)
        torch.cuda.synchronize(self.DEV)
        assert rec.device == self.DEV and torch.isfinite(rec).all()

    def test_run_resire(self):
        sino = ft.forward_project(self._phantom(), self._angles())
        rec = ft.run_resire(sino, self._angles(), n_iter=5)
        torch.cuda.synchronize(self.DEV)
        assert rec.device == self.DEV and torch.isfinite(rec).all()

    def test_run_genfire(self):
        sino = ft.forward_project(self._phantom(), self._angles())
        rec = ft.run_genfire(sino, self._angles(), n_iter=4,
                             resolution_extension=False)
        torch.cuda.synchronize(self.DEV)
        assert rec.device == self.DEV and torch.isfinite(rec).all()

    def test_run_grad_descent(self):
        sino = ft.forward_project(self._phantom(), self._angles())
        rec = ft.run_grad_descent(sino, self._angles(), n_iter=5, lr=1e-2)
        torch.cuda.synchronize(self.DEV)
        assert rec.device == self.DEV and torch.isfinite(rec).all()

    # ── projector lifecycle (GC must run tc_destroy on the right device) ─

    def test_projector_gc_after_device_switch(self):
        import gc
        proj = ft.ParallelProjector((1, 32, 32), self._angles(), device=self.DEV)
        sino = proj.forward(self._phantom(N=32))
        rec = proj.back(sino)
        torch.cuda.synchronize(self.DEV)
        # Drop the only reference, then run any work on cuda:0 and force GC.
        # Without the device guard in __del__, the trailing tc_destroy on the
        # cuda:1-owned cudaArray runs while cuda:0 is current and crashes.
        del proj, sino, rec
        _ = torch.randn(8, device="cuda:0") * 2.0
        gc.collect()
        torch.cuda.synchronize(torch.device("cuda:0"))

    # ── input-device validation (clear ValueError, not illegal access) ───

    def test_projector_rejects_wrong_device_volume(self):
        angles_d0 = torch.linspace(-60, 60, 20).to("cuda:0")
        proj = ft.ParallelProjector((1, 32, 32), angles_d0, device="cuda:0")
        vol_d1 = torch.zeros(1, 32, 32, device=self.DEV)
        with pytest.raises(ValueError, match="cuda:0"):
            proj.forward(vol_d1)

    def test_projector_rejects_wrong_device_sinogram(self):
        angles_d0 = torch.linspace(-60, 60, 20).to("cuda:0")
        proj = ft.ParallelProjector((1, 32, 32), angles_d0, device="cuda:0")
        sino_d1 = torch.zeros(20, 1, 32, device=self.DEV)
        with pytest.raises(ValueError, match="cuda:0"):
            proj.back(sino_d1)

    # ── sequential cross-device usage in the same process ─────────────────

    def test_sequential_devices(self):
        for dev in [torch.device("cuda:0"), self.DEV, torch.device("cuda:0")]:
            v = torch.zeros(1, 32, 32, device=dev)
            v[:, 8:24, 8:24] = 1.0
            a = torch.linspace(-60, 60, 20).to(dev)
            sino = ft.forward_project(v, a)
            _ = ft.run_sirt(sino, a, n_iter=3)
            torch.cuda.synchronize(dev)
