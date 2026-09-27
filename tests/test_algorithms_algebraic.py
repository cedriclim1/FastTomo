"""
Algebraic reconstructions: SIRT (eager and CUDA-graph paths), SART and
weighted SIRT (wSIRT). The W-BP operator behind wSIRT is covered in
test_wsirt.py.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    make_angles,
    needs_cuda,
    phantom_2d,
    rel_err,
    rel_l2,
)

import fasttomo as ft

# ── SIRT ──────────────────────────────────────────────────────────────────────


def test_sirt_shape(backend):
    """Replaces test_cpu.py::test_sirt_shape and test_cuda.py::test_sirt_shape_cuda."""
    N      = 32
    angles = make_angles(30, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_sirt(sino, angles, n_iter=5, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == backend.type


def test_sirt_converges(backend):
    """Reconstruction MSE should decrease as iterations increase.

    Replaces test_cpu.py::test_sirt_converges and
    test_cuda.py::test_sirt_converges_cuda."""
    N      = 64
    angles = make_angles(60, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    errors = []
    for n in (1, 10, 30, 80):
        rec = ft.run_sirt(sino, angles, n_iter=n, Nz=N)
        errors.append(((rec - vol) ** 2).mean().item())

    # allow 5% slack to account for oscillation near convergence
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] * 1.05, (
            f"SIRT did not converge ({backend}): MSE history = {errors}"
        )


def test_sirt_positivity(backend):
    """Replaces test_cpu.py::test_sirt_positivity and
    test_cuda.py::test_sirt_positivity_cuda."""
    N      = 32
    angles = make_angles(30, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_sirt(sino, angles, n_iter=10, Nz=N, positivity=True)
    assert rec.min().item() >= -1e-6


def test_sirt_warm_start_shape_mismatch_raises():
    """Warm-start volume shape must exactly match [Ny, Nz, Nx] — without
    this guard a [1, N, N] warm-start with Ny>1 silently broadcasts inside
    the SIRT update, corrupting the reconstruction with no warning."""
    n_angles, Ny, N = 12, 4, 16
    sino = torch.zeros(n_angles, Ny, N, dtype=torch.float32)
    angles = make_angles(n_angles)
    warm = torch.zeros(1, N, N, dtype=torch.float32)  # Ny mismatch
    with pytest.raises(ValueError, match="volume must have shape"):
        ft.run_sirt(sino, angles, n_iter=1, Nz=N, warm_start=warm)


def test_sirt_projector_shape_mismatch_raises():
    """Projector outputs must match the active reconstruction geometry.

    Regression guard for silent broadcasting: if a user reuses a projector
    built for Ny=1 with a Ny>1 sinogram, SIRT used to broadcast normalisers
    and updates across Y without raising."""

    class MismatchedProjector:
        def __init__(self, n_angles: int, N: int):
            self.n_angles = n_angles
            self.N = N

        def forward(self, volume: torch.Tensor) -> torch.Tensor:
            return torch.zeros(self.n_angles, 1, self.N, dtype=volume.dtype)

        def back(self, sinogram: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, self.N, self.N, dtype=sinogram.dtype)

    n_angles, Ny, N = 12, 4, 16
    sino = torch.zeros(n_angles, Ny, N, dtype=torch.float32)
    angles = make_angles(n_angles)

    with pytest.raises(ValueError, match="projector returned"):
        ft.run_sirt(
            sino,
            angles,
            n_iter=1,
            Nz=N,
            projector=MismatchedProjector(n_angles=n_angles, N=N),
        )


DEVICE = torch.device("cuda")


def make_euler_pure_y(angles_deg: torch.Tensor) -> torch.Tensor:
    """Wrap scalar tilt angles as ZYZ triples (φ=0, θ=angle, ψ=0)."""
    n = len(angles_deg)
    euler = torch.zeros(n, 3)
    euler[:, 1] = angles_deg.cpu()
    return euler


@needs_cuda
def test_sirt_3d_converges_cuda():
    """run_sirt with euler_zyz_deg should converge (MSE decreases)."""
    N      = 64
    angles = make_angles(30, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0).to(DEVICE)
    sino   = ft.forward_project_3d(vol, euler)

    errors = []
    for n in (1, 10, 30):
        rec = ft.run_sirt(sino, angles, n_iter=n, Nz=N, euler_zyz_deg=euler)
        errors.append(((rec - vol) ** 2).mean().item())

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] * 1.05, (
            f"SIRT-3D did not converge: MSE history = {errors}"
        )


# Each public reconstruction algorithm honours the dtype of its sinogram
# (and warm-start volume, when supplied). Algorithms whose underlying
# kernels do not have an fp16 path (run_genfire, run_resire, run_wsirt)
# raise an explicit TypeError on non-fp32 sinograms instead of crashing
# inside the projector with a dtype-lock-mismatch.
@needs_cuda
def test_run_sirt_fp16_matches_fp32():
    """run_sirt with fp16 sinogram returns an fp16 volume that matches
    the fp32 reference within ~5% relative — same tolerance band as
    test_run_fbp_fp16_matches_fp32. Regression for PR #25 / #26: this
    used to crash with `ParallelProjector dtype lock-in mismatch` because
    the volume was hardcoded fp32 while ones_sino inherited fp16 from the
    sinogram."""
    N        = 64
    n_angles = 30
    angles   = make_angles(n_angles, device=DEVICE)
    vol      = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino     = ft.forward_project(vol, angles)

    rec_fp32 = ft.run_sirt(sino,        angles, n_iter=20, Nz=N, positivity=True)
    rec_fp16 = ft.run_sirt(sino.half(), angles, n_iter=20, Nz=N, positivity=True)
    assert rec_fp16.dtype == torch.float16
    assert torch.isfinite(rec_fp16).all()
    assert rel_err(rec_fp16, rec_fp32) < 5e-2


@needs_cuda
def test_run_sirt_fp16_with_cached_projector():
    """fp16 sinogram + a freshly-built ParallelProjector must lock the
    projector to fp16 and reconstruct without dtype mismatch. Regression
    guard for the original crash site (`_bp(ones_sino)` followed by
    `_fp(ones_vol)`)."""
    N        = 32
    n_angles = 20
    angles   = make_angles(n_angles, device=DEVICE)
    sino     = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles).half()
    proj     = ft.ParallelProjector((1, N, N), angles, device=DEVICE)
    rec      = ft.run_sirt(sino, angles, n_iter=4, Nz=N, projector=proj)
    assert rec.dtype == torch.float16
    assert torch.isfinite(rec).all()


@needs_cuda
def test_run_sirt_warm_start_volume_controls_dtype():
    """When a warm-start volume is passed in, its dtype controls the
    compute dtype. Sinogram must match — a mismatch raises TypeError."""
    N        = 32
    n_angles = 20
    angles   = make_angles(n_angles, device=DEVICE)
    vol      = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino     = ft.forward_project(vol, angles)
    warm     = torch.zeros(1, N, N, device=DEVICE)               # fp32

    rec = ft.run_sirt(sino, angles, n_iter=4, Nz=N, warm_start=warm)
    assert rec.dtype == torch.float32

    warm_half = torch.zeros(1, N, N, device=DEVICE, dtype=torch.float16)
    rec_half  = ft.run_sirt(sino.half(), angles, n_iter=4, Nz=N, warm_start=warm_half)
    assert rec_half.dtype == torch.float16

    with pytest.raises(TypeError, match="must have the same dtype"):
        ft.run_sirt(sino, angles, n_iter=4, Nz=N, warm_start=warm_half)


# ── SIRT support_mask ─────────────────────────────────────────────────────────


def _box_support(Ny: int, N: int) -> torch.Tensor:
    """Binary [Ny, N, N] support: 1 inside the central half-box, 0 outside."""
    M = torch.zeros(Ny, N, N)
    q = N // 4
    M[:, q:3*q, q:3*q] = 1.0
    return M


def test_sirt_support_mask_zeros_outside_support():
    """Support is enforced every iteration: out-of-support voxels stay 0."""
    N      = 32
    angles = make_angles(20)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)

    M   = _box_support(1, N)
    rec = ft.run_sirt(sino, angles, n_iter=5, Nz=N, support_mask=M)
    outside = rec[M == 0]
    assert outside.abs().max().item() == 0.0, (
        f"out-of-support voxels must be exactly 0, got "
        f"{outside.abs().max():.2e}"
    )
    assert rec[M == 1].abs().max().item() > 0.0, (
        "in-support voxels should be reconstructed (non-zero)"
    )


def test_sirt_support_mask_2d_broadcasts():
    """A [Nz, Nx] support broadcasts along the tilt axis."""
    N, Ny  = 32, 3
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0).expand(Ny, -1, -1).contiguous()
    sino   = ft.forward_project(vol, angles)

    M   = _box_support(1, N).squeeze(0)          # [Nz, Nx]
    rec = ft.run_sirt(sino, angles, n_iter=3, Nz=N, support_mask=M)
    assert rec.shape == (Ny, N, N)
    outside = rec[:, M == 0]
    assert outside.abs().max().item() == 0.0


def test_sirt_support_mask_with_mask_true_raises():
    N      = 32
    angles = make_angles(10)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ft.run_sirt(sino, angles, n_iter=1, Nz=N, mask=True,
                    support_mask=_box_support(1, N))


def test_sirt_support_mask_bool_tensor_accepted():
    """A bool support tensor is cast to the compute dtype internally."""
    N      = 32
    angles = make_angles(20)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)

    M_bool  = _box_support(1, N).bool()
    M_float = _box_support(1, N)
    rec_b = ft.run_sirt(sino, angles, n_iter=3, Nz=N, support_mask=M_bool)
    rec_f = ft.run_sirt(sino, angles, n_iter=3, Nz=N, support_mask=M_float)
    assert torch.equal(rec_b, rec_f)


def _half_support(N: int, left: bool) -> torch.Tensor:
    """Binary [1, N, N] support covering the left or right half in x."""
    M = torch.zeros(1, N, N, device=DEVICE)
    if left:
        M[:, :, : N // 2] = 1.0
    else:
        M[:, :, N // 2:] = 1.0
    return M


@needs_cuda
def test_sirt_support_mask_cuda_zeros_outside_support():
    """Support enforced each iteration on the CUDA (graph) path."""
    N      = 32
    angles = make_angles(20, device=DEVICE)
    sino   = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles)

    M   = _half_support(N, left=True)
    rec = ft.run_sirt(sino, angles, n_iter=5, Nz=N, support_mask=M)
    assert rec[M == 0].abs().max().item() == 0.0
    assert rec[M == 1].abs().max().item() > 0.0


@needs_cuda
def test_sirt_support_mask_cuda_graph_not_stale():
    """Two same-geometry calls with different masks must not share a cached
    graph's baked-in mask: the second call's output must follow the second
    mask, not the first."""
    N      = 32
    angles = make_angles(20, device=DEVICE)
    sino   = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles)

    M_left  = _half_support(N, left=True)
    M_right = _half_support(N, left=False)

    rec_left  = ft.run_sirt(sino, angles, n_iter=5, Nz=N, support_mask=M_left)
    rec_right = ft.run_sirt(sino, angles, n_iter=5, Nz=N, support_mask=M_right)

    assert rec_left[M_left == 0].abs().max().item() == 0.0
    assert rec_right[M_right == 0].abs().max().item() == 0.0, (
        "second call replayed the first call's mask (stale CUDA Graph buffer)"
    )
    assert rec_right[M_right == 1].abs().max().item() > 0.0


@needs_cuda
def test_sirt_support_mask_full_3d_cuda():
    """A full [Ny, Nz, Nx] support that varies along the tilt axis is honored."""
    N, Ny  = 32, 4
    angles = make_angles(20, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0).expand(Ny, -1, -1).contiguous()
    sino   = ft.forward_project(vol, angles)

    M = torch.zeros(Ny, N, N, device=DEVICE)
    M[: Ny // 2, : N // 2, :] = 1.0
    M[Ny // 2:, N // 2:, :] = 1.0
    rec = ft.run_sirt(sino, angles, n_iter=5, Nz=N, support_mask=M)
    assert rec.shape == (Ny, N, N)
    assert rec[M == 0].abs().max().item() == 0.0


# ── SIRT CUDA graph ───────────────────────────────────────────────────────────


@needs_cuda
class TestRunSirtCudaGraph:
    """Verify the CUDA-Graph-accelerated path in run_sirt.

    Gate: auto-skip if no GPU is available (the @needs_cuda decorator).
    """

    def _make_inputs(self, N: int = 32, n_angles: int = 20):
        """Return (sinogram, angles_deg) on CUDA for a simple phantom."""
        angles = make_angles(n_angles, device=DEVICE)
        vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
        sino   = ft.forward_project(vol, angles)
        return sino, angles

    def test_graph_matches_eager(self):
        """Graph path must produce volumes within 1e-5 RMSE of the eager path."""
        sino, angles = self._make_inputs(N=32, n_angles=20)
        n_iter = 20

        # Eager path: return_loss=True forces the eager loop regardless.
        rec_eager, _ = ft.run_sirt(sino, angles, n_iter=n_iter, Nz=32,
                                   return_loss=True)

        # Graph path: default CUDA path with return_loss=False.
        rec_graph = ft.run_sirt(sino, angles, n_iter=n_iter, Nz=32,
                                return_loss=False)

        rmse = ((rec_eager - rec_graph) ** 2).mean().sqrt().item()
        assert rmse < 1e-5, (
            f"Graph path differs from eager path: RMSE={rmse:.2e} (threshold 1e-5)"
        )

    def test_graph_cache_reuse(self):
        """Second call with matching shape on the same projector must not recapture."""
        N        = 32
        n_angles = 20
        sino, angles = self._make_inputs(N=N, n_angles=n_angles)
        proj = ft.ParallelProjector((1, N, N), angles, device=DEVICE)

        ft.run_sirt(sino, angles, n_iter=5, Nz=N, projector=proj)
        ft.run_sirt(sino, angles, n_iter=5, Nz=N, projector=proj)

        # Should have exactly one cached graph regardless of how many calls.
        # (The former hasattr(proj, "_sirt_graph_cache") check was always
        # true: the attribute is declared on every projector since #85.)
        assert len(proj._sirt_graph_cache) == 1, (
            f"expected 1 cached graph, got {len(proj._sirt_graph_cache)}"
        )

    def test_graph_cache_uses_latest_inputs(self):
        """
        Cached graph replay must respect call-time sinogram and omega values.

        Regression guard: reusing a cached graph on the same projector must not
        silently keep stale tensors/scalars from the first capture.
        """
        N = 32
        n_angles = 20
        angles = make_angles(n_angles, device=DEVICE)
        proj = ft.ParallelProjector((1, N, N), angles, device=DEVICE)

        vol_a = phantom_2d(N, device=DEVICE).unsqueeze(0)
        vol_b = torch.flip(vol_a, dims=[-1])  # intentionally different target
        sino_a = ft.forward_project(vol_a, angles)
        sino_b = ft.forward_project(vol_b, angles)

        # Build cache entry.
        _ = ft.run_sirt(sino_a, angles, n_iter=6, Nz=N, omega=1.0, projector=proj)

        # Same omega, different sinogram should still match eager output.
        rec_graph_same_omega = ft.run_sirt(
            sino_b, angles, n_iter=6, Nz=N, omega=1.0, projector=proj
        )
        rec_eager_same_omega, _ = ft.run_sirt(
            sino_b, angles, n_iter=6, Nz=N, omega=1.0, projector=proj,
            return_loss=True,
        )
        rmse_same_omega = ((rec_graph_same_omega - rec_eager_same_omega) ** 2).mean().sqrt().item()
        assert rmse_same_omega < 1e-5, (
            f"cached graph ignored updated sinogram: RMSE={rmse_same_omega:.2e}"
        )

        # Different omega should also match eager output.
        rec_graph_new_omega = ft.run_sirt(
            sino_b, angles, n_iter=6, Nz=N, omega=0.5, projector=proj
        )
        rec_eager_new_omega, _ = ft.run_sirt(
            sino_b, angles, n_iter=6, Nz=N, omega=0.5, projector=proj,
            return_loss=True,
        )
        rmse_new_omega = ((rec_graph_new_omega - rec_eager_new_omega) ** 2).mean().sqrt().item()
        assert rmse_new_omega < 1e-5, (
            f"cached graph ignored updated omega: RMSE={rmse_new_omega:.2e}"
        )

    def test_graph_handles_positivity_sparsity_mask(self):
        """Smoke-test representative flag combinations via the graph path."""
        sino, angles = self._make_inputs(N=32, n_angles=20)
        n_iter = 5

        combos = [
            dict(positivity=False, sparsity=0.0, mask=False),
            dict(positivity=True,  sparsity=0.0, mask=False),
            dict(positivity=False, sparsity=0.01, mask=False),
            dict(positivity=True,  sparsity=0.01, mask=True),
        ]
        for kwargs in combos:
            rec = ft.run_sirt(sino, angles, n_iter=n_iter, Nz=32,
                              return_loss=False, **kwargs)
            assert rec.shape == (1, 32, 32), (
                f"wrong shape {rec.shape} for kwargs={kwargs}"
            )
            assert torch.isfinite(rec).all(), (
                f"non-finite values in reconstruction for kwargs={kwargs}"
            )
            if kwargs.get("positivity"):
                assert rec.min().item() >= -1e-6, (
                    f"positivity violated for kwargs={kwargs}"
                )

    def test_internal_graph_reused_across_calls(self):
        """Without ``projector=``, a repeated call replays the previous call's
        graph (kernel audit F8) and gives exactly what fresh calls give, with
        new sinogram data, new angles and a new support mask of the same
        shapes each time; the replaying call is faster than the capturing one."""
        import time

        from fasttomo.algorithms import algebraic

        N, n_angles = 32, 20
        angles_a = make_angles(n_angles, device=DEVICE)
        angles_b = angles_a * 0.9 + 1.0
        vol_a = phantom_2d(N, device=DEVICE).unsqueeze(0)
        vol_b = torch.flip(vol_a, dims=[-1])
        sino_a = ft.forward_project(vol_a, angles_a)
        sino_b = ft.forward_project(vol_b, angles_b)
        m_a = _half_support(N, left=True)
        m_b = _half_support(N, left=False)
        calls = [
            dict(sinogram=sino_a, angles_deg=angles_a, support_mask=m_a),
            dict(sinogram=sino_b, angles_deg=angles_b, support_mask=m_b),
            dict(sinogram=sino_a.clone(), angles_deg=angles_a.cpu(), support_mask=m_a),
        ]

        fresh = []
        for kw in calls:
            algebraic._SIRT_OPERATORS.clear()
            fresh.append(ft.run_sirt(n_iter=7, Nz=N, positivity=True, **kw))

        algebraic._SIRT_OPERATORS.clear()
        captures0 = algebraic._sirt_graph_captures
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reused = [ft.run_sirt(n_iter=7, Nz=N, positivity=True, **calls[0])]
        torch.cuda.synchronize()
        t_capture = time.perf_counter() - t0
        reused += [ft.run_sirt(n_iter=7, Nz=N, positivity=True, **kw) for kw in calls[1:]]
        assert algebraic._sirt_graph_captures == captures0 + 1   # one capture, then replays
        assert len(algebraic._SIRT_OPERATORS) == 1
        for a, b in zip(fresh, reused):
            assert torch.equal(a, b)

        t_replay = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ft.run_sirt(n_iter=7, Nz=N, positivity=True, **calls[0])
            torch.cuda.synchronize()
            t_replay.append(time.perf_counter() - t0)
        assert min(t_replay) < t_capture

        # another configuration evicts the kept one (one entry kept)
        ft.run_sirt(sino_a, angles_a, n_iter=7, Nz=N, omega=0.5)
        assert len(algebraic._SIRT_OPERATORS) == 1
        assert algebraic._sirt_graph_captures == captures0 + 2
        algebraic._SIRT_OPERATORS.clear()

    def test_release_cached_projectors_returns_memory(self):
        """What run_sirt keeps between calls (projector textures, graph
        buffers and pool) is freed by the public release call: the allocated
        tensors go back to baseline at once, and after
        torch.cuda.empty_cache() the device's free memory does too."""
        import gc

        from fasttomo.algorithms import algebraic

        N, n_angles, Ny = 96, 31, 64
        angles = make_angles(n_angles, device=DEVICE)
        sino = torch.rand(n_angles, Ny, N, device=DEVICE)
        ft.release_cached_projectors()
        # load the kernels and graph machinery once (lazy module loading
        # takes device memory for good), keeping nothing
        ft.run_sirt(sino, angles, n_iter=3, Nz=N, reuse_graph=False)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        alloc0 = torch.cuda.memory_allocated()
        free0 = torch.cuda.mem_get_info()[0]

        rec = ft.run_sirt(sino, angles, n_iter=3, Nz=N)
        del rec
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        vol_bytes = Ny * N * N * 4
        # kept: live tensors (static volume and C, >= 2 volumes), plus the
        # graph pool's reserved blocks (BP output) and the volume texture
        # array outside torch's allocator: >= 4 volumes of device memory
        assert torch.cuda.memory_allocated() - alloc0 >= 2 * vol_bytes
        assert free0 - torch.cuda.mem_get_info()[0] >= 4 * vol_bytes
        assert len(algebraic._SIRT_OPERATORS) == 1

        assert ft.release_cached_projectors() == 1
        assert ft.release_cached_projectors() == 0
        assert len(algebraic._SIRT_OPERATORS) == 0
        assert torch.cuda.memory_allocated() == alloc0
        torch.cuda.empty_cache()
        # within 2 MiB (driver-side bookkeeping granularity)
        assert abs(torch.cuda.mem_get_info()[0] - free0) <= 2 << 20

    def test_reuse_graph_false_keeps_nothing(self):
        """reuse_graph=False: same result, nothing kept, and a configuration
        kept by an earlier call is left alone."""
        from fasttomo.algorithms import algebraic

        sino, angles = self._make_inputs(N=32, n_angles=20)
        ft.release_cached_projectors()
        a = ft.run_sirt(sino, angles, n_iter=5, Nz=32, reuse_graph=False)
        assert len(algebraic._SIRT_OPERATORS) == 0
        b = ft.run_sirt(sino, angles, n_iter=5, Nz=32)
        assert len(algebraic._SIRT_OPERATORS) == 1
        kept = next(iter(algebraic._SIRT_OPERATORS._ops.values()))
        c = ft.run_sirt(sino, angles, n_iter=5, Nz=32, reuse_graph=False)
        assert next(iter(algebraic._SIRT_OPERATORS._ops.values())) is kept
        assert torch.equal(a, b) and torch.equal(a, c)
        ft.release_cached_projectors()

    def test_other_configuration_drops_kept_one_first(self):
        """A call with another shape / dtype drops the kept projector before
        building its own, so two configurations never coexist."""
        from fasttomo.algorithms import algebraic

        sino, angles = self._make_inputs(N=32, n_angles=20)
        ft.release_cached_projectors()
        ft.run_sirt(sino, angles, n_iter=3, Nz=32)
        (key_a,) = algebraic._SIRT_OPERATORS._ops

        seen = []
        orig = algebraic._build_fp_bp

        def spy(*args, **kw):
            seen.append(len(algebraic._SIRT_OPERATORS))
            return orig(*args, **kw)

        algebraic._build_fp_bp = spy
        try:
            ft.run_sirt(sino.half(), angles, n_iter=3, Nz=32)           # dtype
            ft.run_sirt(sino[:, :, :30].contiguous(), angles, n_iter=3)  # shape
        finally:
            algebraic._build_fp_bp = orig
        assert seen == [0, 0]            # nothing kept while building the new one
        (key_c,) = algebraic._SIRT_OPERATORS._ops
        assert key_c != key_a
        ft.release_cached_projectors()

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="cross-device graph capture needs >=2 GPUs",
    )
    def test_graph_capture_on_non_default_device(self):
        """Regression: torch.cuda.graph(g) must capture on the tensor's
        device, not the process-default device. Reproduces the case where
        run_sirt on cuda:1 with default cuda:0 raised
        'illegal memory access at texture_cache.cu:208'.
        """
        device = torch.device("cuda:1")
        N = 32
        n_angles = 20
        angles = make_angles(n_angles, device=DEVICE).to(device)
        vol = phantom_2d(N, device=DEVICE).unsqueeze(0).to(device)
        sino = ft.forward_project(vol, angles)

        rec = ft.run_sirt(sino, angles, n_iter=6, Nz=N)
        torch.cuda.synchronize(device)
        assert rec.device == device
        assert torch.isfinite(rec).all()


# ── fused SIRT elementwise kernels (kernel audit F7) ─────────────────────────


def _with_edge_values(t: torch.Tensor) -> torch.Tensor:
    """``t`` with NaN, +-inf, +-0, subnormals and fp16 extremes up front."""
    vals = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-40, -1e-40,
            6e-8, -6e-8, 65504.0, -65504.0, 1e-8, -1e-8]
    t.view(-1)[:len(vals)] = torch.tensor(vals, device=t.device).to(t.dtype)
    return t


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.float16 else torch.int32)


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(3, 5, 7), (2, 33, 65), (5, 16, 24)])
def test_fused_sirt_kernels_bit_identical_to_torch_chain(dtype, shape):
    """The fused residual and update kernels give the bits of the PyTorch
    chains they replace in both rounding orders (run_sirt's eager loop and
    `_sirt_iter_inplace`), for every constraint combination, broadcast and
    full support masks, and NaN / inf / signed zero / subnormal inputs.
    The shapes cover the scalar and the 16-byte vectorised kernels."""
    from fasttomo.algorithms.algebraic import _fused_residual, _fused_update
    from fasttomo.configs import ConstraintConfig
    g = torch.Generator(device="cpu").manual_seed(0)

    def rand(*sz, scale=1.0, positive=False):
        x = torch.rand(*sz, generator=g) if positive else torch.randn(*sz, generator=g)
        return (x * scale).to(DEVICE, dtype)

    Ny, Nz, Nx = shape
    vol = _with_edge_values(rand(*shape, scale=3.0))
    upd = _with_edge_values(rand(*shape, scale=10.0))
    C = rand(*shape, scale=0.1, positive=True)
    sino, fp = _with_edge_values(rand(*shape, scale=3.0)), rand(*shape)
    R = rand(*shape, positive=True)

    raw, res = _fused_residual(sino, fp, R, want_raw=True)
    assert torch.equal(_bits(raw), _bits(sino - fp))
    assert torch.equal(_bits(res), _bits((sino - fp) * R))

    for mask_shape in (None, (1, Nz, Nx), (Ny, Nz, Nx)):
        mask = None if mask_shape is None else rand(*mask_shape, scale=2.0, positive=True)
        for omega in (1.0, 0.8, 0.37):
            for positivity in (False, True):
                for soft in (0.0, 1e-3, 0.5):
                    cons = ConstraintConfig(positivity=positivity, support_mask=mask,
                                            soft_threshold=soft)
                    # run_sirt's eager loop
                    want = cons.apply(vol + omega * C * upd)
                    got = _fused_update(vol, upd, C, omega, cons, eager_order=True)
                    assert torch.equal(_bits(got), _bits(want)), (mask_shape, omega,
                                                                   positivity, soft)
                    # _sirt_iter_inplace (in place, as in the CUDA graph)
                    want = vol.clone()
                    u = upd.clone()
                    u.mul_(C)
                    want.add_(u, alpha=omega)
                    cons.apply_inplace_(want)
                    got = vol.clone()
                    assert _fused_update(got, upd, C, omega, cons, eager_order=False,
                                         out=got) is got
                    assert torch.equal(_bits(got), _bits(want)), (mask_shape, omega,
                                                                   positivity, soft)


@needs_cuda
def test_fused_sirt_kernels_decline_unfusable_inputs():
    """Mixed dtypes, CPU tensors and odd mask shapes fall back to the
    PyTorch chain (the helpers return None)."""
    from fasttomo.algorithms.algebraic import _fused_residual, _fused_update
    from fasttomo.configs import ConstraintConfig

    v = torch.ones(2, 4, 4, device=DEVICE)
    cons = ConstraintConfig()
    assert _fused_residual(v, v.half(), v, want_raw=False) is None
    assert _fused_residual(v.cpu(), v.cpu(), v.cpu(), want_raw=False) is None
    assert _fused_update(v, v, v.double(), 1.0, cons, eager_order=True) is None
    odd = ConstraintConfig(support_mask=torch.ones(4, 4, device=DEVICE))   # 2-D
    assert _fused_update(v, v, v, 1.0, odd, eager_order=True) is None


# ── direct texture writes (kernel audit F6) ─────────────────────────────────


def _sirt_cases(dtype, Ny):
    """run_sirt keyword sets covering the graph and eager paths, both
    geometries, constraints and ``projector=``."""
    N, n = 24, 10
    ang = make_angles(n, device=DEVICE)
    g = torch.Generator(device="cpu").manual_seed(1)
    vol = torch.rand(Ny, N, N, generator=g).to(DEVICE)
    sino = ft.forward_project(vol, ang).to(dtype)
    eul = torch.stack([torch.zeros(n), ang.cpu(), torch.zeros(n)], 1).to(DEVICE)
    sino3d = ft.forward_project_3d(vol, eul).to(dtype)
    sm = torch.zeros(Ny, N, N, device=DEVICE, dtype=dtype)
    sm[:, 3:-3, 2:-5] = 1
    return [
        dict(sinogram=sino, angles_deg=ang, n_iter=4),
        dict(sinogram=sino, angles_deg=ang, n_iter=4, positivity=True, mask=True,
             sparsity=1e-3, omega=0.8),
        dict(sinogram=sino, angles_deg=ang, n_iter=4, support_mask=sm, return_loss=True),
        dict(sinogram=sino, angles_deg=ang, n_iter=3, omega=0.6, positivity=True,
             projector=ft.ParallelProjector((Ny, N, N), ang, device=DEVICE)),
        dict(sinogram=sino3d, angles_deg=ang, n_iter=3, euler_zyz_deg=eul,
             positivity=True),
        dict(sinogram=sino3d, angles_deg=ang, n_iter=3, euler_zyz_deg=eul,
             projector=ft.RotmatProjector((Ny, N, N), n_angles=n, device=DEVICE)),
    ]


@needs_cuda
@pytest.mark.parametrize("dtype, Ny", [(torch.float32, 3), (torch.float16, 3),
                                       (torch.float16, 4)])
def test_sirt_texture_writes_bit_identical(monkeypatch, caplog, dtype, Ny):
    """Writing the residual and the updated volume straight into the
    projector's textures (and, on the graph path, keeping the volume there)
    gives exactly the bits of the upload path. fp16 with Ny % 4 == 0 covers
    the 4-channel packed textures, Ny = 3 the 1-channel ones."""
    from fasttomo.algorithms import algebraic

    def run_all():
        algebraic._SIRT_OPERATORS.clear()
        out = []
        for kw in _sirt_cases(dtype, Ny):
            r = ft.run_sirt(**kw)
            out.append(r[0] if isinstance(r, tuple) else r)
            if isinstance(r, tuple):
                out.append(r[1])
        return out

    direct = run_all()
    monkeypatch.setattr(algebraic, "_tex_target", lambda *a: None)
    uploads = run_all()
    algebraic._SIRT_OPERATORS.clear()
    assert "capture failed" not in caplog.text     # both runs took the graph path
    for a, b in zip(direct, uploads):
        assert torch.equal(a, b)


@needs_cuda
def test_sirt_texture_writes_skip_uploads():
    """The debug counters: the eager loop uploads the volume once (plus the
    normalisation projections) and never the residual; the graph path
    uploads the volume once per call; afterwards the projector uploads
    again as usual (the direct-write records are dropped)."""
    N, n, n_iter = 24, 10, 6
    ang = make_angles(n, device=DEVICE)
    vol = phantom_2d(N, device=DEVICE).unsqueeze(0).expand(2, -1, -1).contiguous()
    sino = ft.forward_project(vol, ang)
    proj = ft.ParallelProjector((2, N, N), ang, device=DEVICE)
    counts = proj._op.upload_counts

    ft.run_sirt(sino, ang, n_iter, projector=proj, return_loss=True)      # eager
    # normalisations: one sinogram and one volume upload; then iteration 1
    # uploads the starting volume, every later volume and every residual
    # come from the fused kernels
    assert counts == {"vol": 2, "vol_skipped": n_iter - 1,
                      "sino": 1, "sino_skipped": n_iter}

    for k in counts:
        counts[k] = 0
    ft.run_sirt(sino, ang, n_iter, projector=proj)                        # graph
    ft.run_sirt(sino, ang, n_iter, projector=proj)                        # replay
    # per call: normalisations (1 + 1) and the start volume (1), plus the
    # capture's own start-volume load; the replays are not counted
    assert counts["vol"] == 2 + 2 + 1 and counts["sino"] == 2
    assert proj._op._vol_tex_token is None and not proj._op._sino_tex_written

    x = torch.rand(2, N, N, device=DEVICE)
    before = counts["vol"]
    torch.testing.assert_close(proj.forward(x), ft.forward_project(x, ang), rtol=0, atol=0)
    assert counts["vol"] == before + 1


@needs_cuda
def test_rotmat_grad_after_sirt_uses_fresh_texture():
    """run_sirt's direct writes into a RotmatProjector's volume texture move
    the texture epoch on, so a pose gradient taken after it still matches
    the uncached kernel (the cached branch only reads a texture its own
    forward uploaded)."""
    N, n = 16, 6
    ang = make_angles(n, device=DEVICE)
    eul = torch.stack([torch.zeros(n), ang.cpu(), torch.zeros(n)], 1).to(DEVICE)
    mats = ft.euler_zyz_to_rotmat(eul)
    rp = ft.RotmatProjector((2, N, N), n_angles=n, device=DEVICE)
    vol = torch.rand(2, N, N, device=DEVICE)
    sino = ft.forward_project_3d(vol, eul)

    m = mats.clone().requires_grad_(True)
    pred = rp.forward(vol, m)                    # uploads vol, token = epoch
    ft.run_sirt(sino, ang, 3, euler_zyz_deg=eul, projector=rp)   # rewrites the texture
    pred.sum().backward()
    m_ref = mats.clone().requires_grad_(True)
    ft.RotmatProjector((2, N, N), n_angles=n, device=DEVICE).forward(vol, m_ref).sum().backward()
    torch.testing.assert_close(m.grad, m_ref.grad, rtol=0, atol=0)


@needs_cuda
@pytest.mark.parametrize("dtype, channels, kind", [
    (torch.float32, 1, 0), (torch.float16, 1, 0), (torch.float16, 4, 0),
    (torch.float32, 1, 2), (torch.float16, 1, 1)])
def test_tc_download_round_trip(dtype, channels, kind):
    """tc_download is the exact inverse of tc_upload (the texture-resident
    SIRT volume comes back bit for bit)."""
    import fasttomo._core as core
    from fasttomo._backend import _stream_ptr

    shape = (8, 5, 7)
    src = torch.randn(*shape, device=DEVICE).to(dtype)
    dst = torch.empty_like(src)
    h = core.tc_create(kind, 0 if dtype == torch.float32 else 1, channels, *shape,
                       surface=True)
    try:
        assert core.tc_has_surface(h)
        core.tc_upload(h, src.data_ptr(), _stream_ptr(src.device))
        core.tc_download(h, dst.data_ptr(), _stream_ptr(src.device))
        torch.cuda.synchronize()
    finally:
        core.tc_destroy(h)
    assert torch.equal(dst, src)


# ── SART ──────────────────────────────────────────────────────────────────────


class TestRunSart:
    """Tests for run_sart (SART / OS-SART baseline)."""

    def _make_fixture(self, N: int = 16):
        """Return a (sinogram, angles) pair produced by FP of a small phantom."""
        angles = make_angles(20, max_deg=60.0)
        vol = phantom_2d(N).unsqueeze(0)
        sino = ft.forward_project(vol, angles)
        return sino, angles, vol

    def test_shape(self):
        """Output volume has shape [Ny, Nz, Nx] and dtype float32."""
        N = 16
        sino, angles, vol = self._make_fixture(N)
        rec = ft.run_sart(sino, angles, n_iter=2, Nz=N)
        assert rec.shape == vol.shape, f"expected {vol.shape}, got {rec.shape}"
        assert rec.dtype == torch.float32

    def test_convergence(self):
        """SART residual MSE decreases over 5 iterations (subset_size=1)."""
        N = 16
        sino, angles, _ = self._make_fixture(N)
        _, losses = ft.run_sart(sino, angles, n_iter=5, subset_size=1, Nz=N,
                                return_loss=True)
        assert losses[0].item() > losses[-1].item(), (
            f"SART did not reduce loss: {losses.tolist()}"
        )

    def test_sirt_equivalence(self):
        """
        With subset_size=n_angles and omega=1.0, run_sart should produce
        RMSE within 1e-3 of run_sirt on the same fixture and iteration count.

        Both algorithms reduce to the same update equation when all angles form
        a single subset — the only numerical difference is floating-point
        ordering inside the update accumulation.
        """
        N = 16
        k = 3
        sino, angles, vol = self._make_fixture(N)
        n_angles = len(angles)

        rec_sirt = ft.run_sirt(sino, angles, n_iter=k, Nz=N)
        rec_sart = ft.run_sart(sino, angles, n_iter=k, subset_size=n_angles, Nz=N)

        rmse = ((rec_sirt - rec_sart).pow(2).mean()).sqrt().item()
        assert rmse < 1e-3, (
            f"SART (subset_size=n_angles) vs SIRT RMSE = {rmse:.2e} (threshold 1e-3)"
        )

    def test_subset_size_smoke(self):
        """subset_size=4, n_iter=2 produces finite, shape-correct output."""
        N = 16
        sino, angles, vol = self._make_fixture(N)
        rec = ft.run_sart(sino, angles, n_iter=2, subset_size=4, Nz=N)
        assert rec.shape == vol.shape
        assert torch.isfinite(rec).all().item(), "output contains NaN or Inf"

    def test_return_loss(self):
        """return_loss=True returns (volume, loss_history) with len == n_iter."""
        N = 16
        sino, angles, vol = self._make_fixture(N)
        n_iter = 4
        result = ft.run_sart(sino, angles, n_iter=n_iter, Nz=N, return_loss=True)
        assert isinstance(result, tuple) and len(result) == 2
        rec, losses = result
        assert rec.shape == vol.shape
        assert losses.shape == torch.Size([n_iter]), (
            f"expected loss_history length {n_iter}, got {losses.shape}"
        )
        assert torch.isfinite(losses).all().item()

    def test_warm_start_shape_mismatch_raises(self):
        """Warm-start volume shape must match [Ny, Nz, Nx] exactly — without
        this guard a [1, N, N] warm-start with Ny>1 silently broadcasts and
        corrupts the SART subset update with no warning."""
        n_angles, Ny, N = 12, 4, 16
        sino = torch.zeros(n_angles, Ny, N, dtype=torch.float32)
        angles = make_angles(n_angles)
        warm = torch.zeros(1, N, N, dtype=torch.float32)  # Ny mismatch
        with pytest.raises(ValueError, match="volume must have shape"):
            ft.run_sart(sino, angles, n_iter=1, Nz=N, warm_start=warm)


@needs_cuda
def test_run_sart_fp16_matches_fp32():
    """run_sart with fp16 sinogram returns an fp16 volume matching fp32
    within ~5% relative."""
    N        = 64
    n_angles = 30
    angles   = make_angles(n_angles, device=DEVICE)
    vol      = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino     = ft.forward_project(vol, angles)

    rec_fp32 = ft.run_sart(sino,        angles, n_iter=5, subset_size=5,
                            Nz=N, positivity=True)
    rec_fp16 = ft.run_sart(sino.half(), angles, n_iter=5, subset_size=5,
                            Nz=N, positivity=True)
    assert rec_fp16.dtype == torch.float16
    assert torch.isfinite(rec_fp16).all()
    assert rel_err(rec_fp16, rec_fp32) < 5e-2


# ── wSIRT ─────────────────────────────────────────────────────────────────────


def _wsirt_phantom(device, N=48):
    vol = torch.zeros(2, N, N, device=device)
    vol[:, 12:36, 14:30] = 1.0
    vol[1, 20:28, 20:26] = 2.0
    ang = torch.linspace(-70, 70, 41, device=device)
    return vol, ang, ft.forward_project(vol, ang)


def test_wsirt_recovers_phantom(backend):
    """wSIRT on a +-70 deg, 41-view phantom: the data residual falls by more
    than 100x within 5 iterations, and the volume error after 20 iterations
    is below 35 % (limited-angle floor: SIRT reaches ~29 %). Output has the
    [Ny, Nz, Nx] shape, fp32 dtype and the input device."""
    vol, ang, sino = _wsirt_phantom(backend)
    rec, loss = ft.run_wsirt(sino, ang, n_iter=20, Nz=48, return_loss=True)
    assert rec.shape == vol.shape and rec.dtype == torch.float32
    assert rec.device.type == backend.type
    assert loss.shape == (20,) and torch.isfinite(loss).all()
    assert loss[5] < loss[0] / 100
    assert rel_l2(rec, vol) < 0.35


@needs_cuda
def test_wsirt_converges_faster_than_sirt_cuda():
    """The point of wSIRT (Wolf et al. 2014): the exact-filter W-BP step gets
    close in a handful of iterations where plain SIRT is still far away."""
    vol, ang, sino = _wsirt_phantom(torch.device("cuda"))
    w = ft.run_wsirt(sino, ang, n_iter=5, Nz=48)
    s = ft.run_sirt(sino, ang, n_iter=5, Nz=48)
    assert rel_l2(w, vol) < 0.8 * rel_l2(s, vol)


def test_wsirt_constraints(backend):
    """positivity, mask (cylinder) and the sparsity soft-threshold are
    honoured on every backend."""
    vol, ang, sino = _wsirt_phantom(backend)
    rec = ft.run_wsirt(sino, ang, n_iter=5, Nz=48, positivity=True, mask=True)
    assert rec.min() >= 0
    outside = ft.cylindrical_mask(48, 48, device=backend) == 0
    assert (rec[:, outside] == 0).all()
    plain = ft.run_wsirt(sino, ang, n_iter=5, Nz=48)
    sparse = ft.run_wsirt(sino, ang, n_iter=5, Nz=48, sparsity=0.05)
    assert (sparse == 0).sum() > (plain == 0).sum()


@needs_cuda
def test_run_wsirt_rejects_fp16():
    """run_wsirt's exact-WF construction runs in fp32 (PSF + rfft2).
    Explicit TypeError on fp16 sinogram."""
    N        = 32
    n_angles = 20
    angles   = make_angles(n_angles, device=DEVICE)
    sino     = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles).half()
    with pytest.raises(TypeError, match=r"torch\.float32 sinogram"):
        ft.run_wsirt(sino, angles, Nz=N, n_iter=2)
