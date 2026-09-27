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
