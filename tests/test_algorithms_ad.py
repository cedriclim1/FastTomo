"""
Automatic-differentiation reconstructions: run_ad_recon (volume, pose and
shift refinement, regularisers, constraints, staged schedules, support
mask) and run_grad_descent.

run_ad_recon is CUDA-only in production; several tests patch its input
check (fasttomo.algorithms._common._check_inputs) and projector class
(fasttomo.algorithms.ad.ParallelProjector) with CPU stand-ins.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    make_angles,
    needs_cuda,
    phantom_2d,
    rel_l2,
)

import fasttomo as ft
import fasttomo.algorithms as algo
from fasttomo.algorithms import (
    _ad_recon_active_steps,
    _ad_recon_step_schedule,
    _prepare_ad_recon_pose_init,
)
from fasttomo.configs import PoseRefinementConfig

# ── sparsity term and constraints ─────────────────────────────────────────────


def test_ad_sparsity_term_is_l1_with_signed_gradients():
    """AD sparsity penalty must be λ·mean(|f|), not a signed mean bias."""
    from fasttomo.algorithms import _ad_sparsity_term

    vol = torch.tensor(
        [[[-2.0, -0.5], [0.0, 1.5]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    lam = 0.3

    loss = _ad_sparsity_term(vol, lam)
    assert loss.item() == pytest.approx((lam * vol.detach().abs().mean()).item())

    loss.backward()
    expected_grad = lam * torch.sign(vol.detach()) / vol.numel()
    assert torch.allclose(vol.grad, expected_grad)


def test_ad_recon_sparsity_does_not_drift_without_positivity(monkeypatch):
    """End-to-end (#73): with zero data and positivity=False, the L1 sparsity
    term must leave a zero volume at zero.

    run_ad_recon is CUDA-only in production, so the input check and the
    projector are replaced by CPU stand-ins; the dummy projector always
    returns zeros, so only the sparsity gradient can move the volume. The
    old signed mean gave every voxel the same +λ/N gradient and Adam drove
    the volume to about -n_iter * lr (≈ -1 here).
    """
    import fasttomo.algorithms as algo

    class DummyParallelProjector:
        def __init__(self, n_angles: int, Ny: int, N_det: int):
            self.n_angles = n_angles
            self.Ny = Ny
            self.N_det = N_det

        def forward(self, volume: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                self.n_angles, self.Ny, self.N_det,
                dtype=volume.dtype, device=volume.device,
            )

    def fake_check_inputs(sinogram, angles_deg, Nz, **kwargs):
        n_angles, Ny, N_det = sinogram.shape
        return n_angles, Ny, N_det, (N_det if Nz < 0 else Nz), sinogram.device

    monkeypatch.setattr(algo.ad, "ParallelProjector", DummyParallelProjector)
    monkeypatch.setattr(algo._common, "_check_inputs", fake_check_inputs)

    n_angles, Ny, N = 5, 1, 8
    sino = torch.zeros(n_angles, Ny, N, dtype=torch.float32)
    angles = torch.linspace(-20.0, 20.0, n_angles)
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=5, lr_volume=0.2, cosine_schedule=False),
        reg=ft.RegularizerConfig(sparsity=1.0),
        constraints=ft.ConstraintConfig(positivity=False),
    )

    rec = ft.run_ad_recon(
        sino, angles, Nz=N,
        projector=DummyParallelProjector(n_angles, Ny, N),
        config=cfg,
    )
    assert rec.abs().max().item() == 0.0, (
        "L1 sparsity should leave a zero volume unchanged when fidelity is zero"
    )


def test_ad_recon_rejects_projector_with_wrong_output_shape(monkeypatch):
    """Regression (#53): run_ad_recon must apply the same projector output
    shape guard as run_sirt instead of letting the loss broadcast a
    [1, Ny, N] prediction over the [n_angles, Ny, N] sinogram."""
    import fasttomo.algorithms as algo

    class WrongGeometryProjector:
        def forward(self, volume: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, volume.shape[0], volume.shape[2], dtype=volume.dtype)

    def fake_check_inputs(sinogram, angles_deg, Nz, **kwargs):
        n_angles, Ny, N_det = sinogram.shape
        return n_angles, Ny, N_det, (N_det if Nz < 0 else Nz), sinogram.device

    monkeypatch.setattr(algo.ad, "ParallelProjector", WrongGeometryProjector)
    monkeypatch.setattr(algo._common, "_check_inputs", fake_check_inputs)

    sino = torch.zeros(5, 1, 8)
    angles = torch.linspace(-20.0, 20.0, 5)
    cfg = ft.AdReconConfig(optim=ft.OptimConfig(n_iter=1))
    with pytest.raises(ValueError, match="projector returned forward-projection"):
        ft.run_ad_recon(sino, angles, projector=WrongGeometryProjector(), config=cfg)


class _MeanZProjector:
    """CPU stand-in for ParallelProjector: every view is the Z-mean of the
    volume. Linear, so the fidelity gradient is non-zero for every voxel."""

    def __init__(self, n_angles: int):
        self.n_angles = n_angles

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return volume.mean(dim=1).unsqueeze(0).expand(self.n_angles, -1, -1)


def _cpu_ad_recon(monkeypatch, *, n_angles=4, Ny=2, N=6, projector=None):
    """Patch run_ad_recon's CUDA-only input check and projector class so the
    loop runs on CPU with ``projector`` (default `_MeanZProjector`)."""
    import fasttomo.algorithms as algo

    proj = projector if projector is not None else _MeanZProjector(n_angles)

    def fake_check_inputs(sinogram, angles_deg, Nz, **kwargs):
        n, ny, nd = sinogram.shape
        return n, ny, nd, (nd if Nz < 0 else Nz), sinogram.device

    monkeypatch.setattr(algo.ad, "ParallelProjector", type(proj))
    monkeypatch.setattr(algo._common, "_check_inputs", fake_check_inputs)
    return proj


@pytest.mark.parametrize("lam,expected", [(0.0, 0.2), (0.05, 0.15), (0.3, 0.0)])
def test_ad_recon_constraint_sparsity_soft_thresholds(monkeypatch, lam, expected):
    """One Adam step from a zero volume moves every voxel by exactly
    lr_volume (Adam's first step is lr * g / (|g| + eps)); the post-step
    soft-threshold then leaves |v| = max(lr - lam, 0). Before the fix
    ConstraintConfig.sparsity was ignored and |v| stayed 0.2."""
    proj = _cpu_ad_recon(monkeypatch)
    sino = torch.full((4, 2, 6), 3.0)           # target above the start: v grows
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=1, lr_volume=0.2, cosine_schedule=False),
        constraints=ft.ConstraintConfig(soft_threshold=lam),
    )
    rec = ft.run_ad_recon(sino, torch.linspace(-30, 30, 4), Nz=6,
                          projector=proj, config=cfg)
    torch.testing.assert_close(rec, torch.full_like(rec, expected), rtol=0, atol=1e-6)


def test_ad_recon_constraint_order_mask_then_threshold(monkeypatch):
    """The soft-threshold runs after the support mask. A soft mask tells
    the orders apart: every voxel is +0.2 after the first Adam step;
    mask-then-threshold gives max(m * 0.2 - lam, 0), threshold-then-mask
    gives m * max(0.2 - lam, 0). With m = 0.5, lam = 0.05 these are 0.05 and
    0.075. (With a binary mask both orders agree, so a binary mask could not
    pin the order.) A voxel outside the mask (m = 0) ends at 0 either way,
    and positivity commutes with a non-negative mask, so its position
    relative to the mask is not observable."""
    proj = _cpu_ad_recon(monkeypatch)
    mask = torch.ones(6, 6)
    mask[:, :2] = 0.5                       # soft region
    mask[:, 4:] = 0.0                       # outside the support
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=1, lr_volume=0.2, cosine_schedule=False),
        constraints=ft.ConstraintConfig(positivity=True, soft_threshold=0.05),
    )
    rec = ft.run_ad_recon(torch.full((4, 2, 6), 3.0), torch.linspace(-30, 30, 4),
                          Nz=6, projector=proj, config=cfg, support_mask=mask)
    torch.testing.assert_close(rec[:, :, :2], torch.full((2, 6, 2), 0.05), rtol=0, atol=1e-6)
    torch.testing.assert_close(rec[:, :, 2:4], torch.full((2, 6, 2), 0.15), rtol=0, atol=1e-6)
    assert torch.equal(rec[:, :, 4:], torch.zeros(2, 6, 2))
    # and ConstraintConfig.apply uses the same order on the same input
    applied = ft.ConstraintConfig(positivity=True, support_mask=mask.unsqueeze(0), soft_threshold=0.05).apply(
        torch.full((2, 6, 6), 0.2))
    torch.testing.assert_close(rec, applied, rtol=0, atol=1e-6)


def test_ad_recon_constraint_sparsity_matches_config_apply(monkeypatch):
    """Several iterations: run_ad_recon's result equals a hand-written loop
    (Adam step, then ConstraintConfig.apply) with the same settings."""
    proj = _cpu_ad_recon(monkeypatch)
    g = torch.Generator().manual_seed(5)
    sino = torch.rand(4, 2, 6, generator=g) * 2 - 0.5
    cons = ft.ConstraintConfig(positivity=False, soft_threshold=0.02)
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=6, lr_volume=0.1, cosine_schedule=False),
        constraints=cons,
    )
    rec = ft.run_ad_recon(sino, torch.linspace(-30, 30, 4), Nz=6,
                          projector=proj, config=cfg)

    v = torch.zeros(2, 6, 6, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.1)
    for _ in range(6):
        opt.zero_grad()
        torch.nn.functional.mse_loss(proj.forward(v), sino).backward()
        opt.step()
        with torch.no_grad():
            v.copy_(cons.apply(v))
    assert torch.equal(rec, v.detach())
    cfg.constraints = ft.ConstraintConfig(positivity=False)
    unthresholded = ft.run_ad_recon(sino, torch.linspace(-30, 30, 4), Nz=6,
                                    projector=proj, config=cfg)
    assert not torch.equal(rec, unthresholded)


@needs_cuda
def test_ad_recon_constraint_sparsity_cuda():
    """Real projector: the soft-threshold zeroes the small background voxels
    that plain AD leaves as noise, and sparsity=0 keeps the default result
    bit-identical."""
    torch.manual_seed(0)
    N, n = 32, 25
    ang = torch.linspace(-60, 60, n, device="cuda")
    vol = torch.zeros(2, N, N, device="cuda")
    vol[:, 10:20, 12:22] = 1.0
    sino = ft.forward_project(vol, ang)
    sino = sino + 0.05 * torch.randn_like(sino)
    base = ft.AdReconConfig(optim=ft.OptimConfig(n_iter=40, lr_volume=0.05))
    rec0 = ft.run_ad_recon(sino, ang, Nz=N, config=base)
    rec0b = ft.run_ad_recon(sino, ang, Nz=N, config=ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=40, lr_volume=0.05),
        constraints=ft.ConstraintConfig(soft_threshold=0.0)))
    assert torch.equal(rec0, rec0b)
    rec = ft.run_ad_recon(sino, ang, Nz=N, config=ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=40, lr_volume=0.05),
        constraints=ft.ConstraintConfig(soft_threshold=0.01)))
    bg = vol == 0
    assert (rec[bg] == 0).float().mean() > 0.5 > (rec0[bg] == 0).float().mean()
    assert rel_l2(rec, vol) < 1.0


def test_ad_recon_shape_guard_runs_before_shift_correction(monkeypatch):
    """A projector built for the wrong number of views must give the
    projector-shape error even with refine_shifts=True; the shift correction
    used to run first and raise 'shifts has 5 entries but sinogram has 1
    angles', blaming the shifts."""

    class OneViewProjector:
        def forward(self, volume):
            return volume.mean(dim=1).unsqueeze(0)          # [1, Ny, Nx]

    proj = _cpu_ad_recon(monkeypatch, projector=OneViewProjector())
    cfg = ft.AdReconConfig(optim=ft.OptimConfig(n_iter=1),
                           pose=ft.PoseRefinementConfig(refine_shifts=True))
    with pytest.raises(ValueError, match="projector returned forward-projection"):
        ft.run_ad_recon(torch.zeros(5, 2, 6), torch.linspace(-20, 20, 5), Nz=6,
                        projector=proj, config=cfg)


# ── run_ad_recon end to end (CUDA) ────────────────────────────────────────────


DEVICE = torch.device("cuda")


@needs_cuda
class TestRunAdRecon:
    """Tests for run_ad_recon — the AD-with-pose library wrapper."""

    def test_no_pose_converges(self):
        """No-pose AD on a 16³ phantom: final loss < initial loss."""
        N = 16
        angles = make_angles(20, device=DEVICE)
        vol = phantom_2d(N, device=DEVICE).unsqueeze(0)
        sino = ft.forward_project(vol, angles)

        cfg = ft.AdReconConfig(optim=ft.OptimConfig(n_iter=200, lr_volume=0.05))
        _, losses = ft.run_ad_recon(sino, angles, config=cfg, return_loss=True)
        assert losses[-1].item() < losses[0].item(), (
            f"run_ad_recon (no pose) did not reduce loss: "
            f"{losses[0].item():.4f} -> {losses[-1].item():.4f}"
        )

    def test_with_pose_recovers_perturbed_euler(self):
        """With-pose AD: angles are updated from a perturbed starting point.

        The test verifies that run_ad_recon(refine_pose=True) actually
        optimises the angle parameters (they move from their initial values)
        and that the refined angles are closer to ground truth than the
        perturbation.

        To give the angle gradient a meaningful signal, the test uses a proper
        3-D [N, N, N] phantom (not a single-slice [1, N, N] volume).  The 3-D
        rotmat gradient kernel produces zero gradient on Ny=1 volumes because
        there is no Y-axis variation to differentiate with respect to the
        rotation matrix.
        """
        torch.manual_seed(42)
        N = 16
        n_angles = 10
        true_angles = torch.linspace(-30.0, 30.0, n_angles, device=DEVICE)
        true_euler = torch.zeros(n_angles, 3, device=DEVICE)
        true_euler[:, 1] = true_angles

        # 3-D phantom — required so the grad_rotmat kernel is non-zero
        vol = torch.zeros(N, N, N, device=DEVICE, dtype=torch.float32)
        q = N // 4
        vol[q:3*q, q:3*q, q:3*q] = 1.0
        sino = ft.forward_project_3d(vol, true_euler)

        # Perturb tilt angles by up to 2°
        perturbation = (torch.rand(n_angles, device=DEVICE) * 2.0 - 1.0) * 2.0
        perturbed_angles = true_angles + perturbation
        perturbed_angles_cpu = perturbed_angles.cpu()
        error_before = perturbation.abs().max().item()

        cfg = ft.AdReconConfig(
            optim=ft.OptimConfig(n_iter=300, lr_volume=0.05),
            pose=ft.PoseRefinementConfig(refine_pose=True, lr_angles=1e-2),
            constraints=ft.ConstraintConfig(positivity=True),
        )
        result = ft.run_ad_recon(
            sino, perturbed_angles_cpu,
            config=cfg,
            return_pose=True,
        )
        # result = (volume, euler_or_None, shifts_or_None)
        _, refined_euler, _ = result
        assert refined_euler is not None

        # The angles must have changed from their initial (perturbed) values
        angle_delta = (refined_euler[:, 1] - perturbed_angles_cpu).abs().max().item()
        assert angle_delta > 0.0, (
            "run_ad_recon(refine_pose=True) did not update angle parameters"
        )

        # The refined angles must be closer to ground truth than the perturbation
        error_after = (refined_euler[:, 1] - true_angles.cpu()).abs().max().item()
        assert error_after < error_before, (
            f"run_ad_recon pose refinement worsened angle accuracy: "
            f"error_before={error_before:.3f}°, error_after={error_after:.3f}°"
        )

    def test_projector_geometry_mismatch_raises(self):
        """Regression (#53): run_ad_recon must reject projector outputs that
        would broadcast against the sinogram."""
        N = 16
        angles = make_angles(20, device=DEVICE)
        vol = phantom_2d(N, device=DEVICE).unsqueeze(0)
        sino = ft.forward_project(vol, angles)

        # Same volume geometry but n_angles=1: without the shape check the
        # MSE would broadcast [1, 1, N] over [20, 1, N].
        bad_proj = ft.ParallelProjector((1, N, N), make_angles(1, device=DEVICE), device=DEVICE, N_det=N)
        cfg = ft.AdReconConfig(optim=ft.OptimConfig(n_iter=1))
        with pytest.raises(ValueError, match="projector returned"):
            ft.run_ad_recon(sino, angles, projector=bad_proj, config=cfg)

    def test_sparsity_does_not_drift_without_positivity(self):
        """Regression (#71/#73): the sparsity term is λ·mean(|f|). With zero
        data and positivity=False a zero volume must stay exactly zero; the
        old signed mean drove every voxel to about -n_iter * lr."""
        N = 16
        angles = make_angles(10, device=DEVICE)
        sino = torch.zeros(10, 1, N, device=DEVICE)
        cfg = ft.AdReconConfig(
            optim=ft.OptimConfig(n_iter=5, lr_volume=0.2, cosine_schedule=False),
            reg=ft.RegularizerConfig(sparsity=1.0),
            constraints=ft.ConstraintConfig(positivity=False),
        )
        rec = ft.run_ad_recon(sino, angles, config=cfg)
        assert rec.abs().max().item() == 0.0

    def test_projector_mismatch_raises(self):
        """Passing a RotmatProjector with refine_pose=False raises ValueError."""
        N = 16
        n_angles = 10
        angles = make_angles(n_angles, device=DEVICE)
        vol = phantom_2d(N, device=DEVICE).unsqueeze(0)
        sino = ft.forward_project(vol, angles)

        wrong_proj = ft.RotmatProjector(
            (1, N, N), n_angles=n_angles, device=DEVICE, N_det=N,
        )
        cfg_no_pose = ft.AdReconConfig(optim=ft.OptimConfig(n_iter=1))
        with pytest.raises(ValueError, match="ParallelProjector"):
            ft.run_ad_recon(sino, angles, projector=wrong_proj, config=cfg_no_pose)

        parallel_proj = ft.ParallelProjector(
            (1, N, N), angles, device=DEVICE, N_det=N,
        )
        cfg_pose = ft.AdReconConfig(
            optim=ft.OptimConfig(n_iter=1),
            pose=ft.PoseRefinementConfig(refine_pose=True),
        )
        with pytest.raises(ValueError, match="RotmatProjector"):
            ft.run_ad_recon(sino, angles, projector=parallel_proj, config=cfg_pose)

    def test_return_loss_shape(self):
        """return_loss=True: loss history length equals n_iter."""
        N = 16
        angles = make_angles(10, device=DEVICE)
        vol = phantom_2d(N, device=DEVICE).unsqueeze(0)
        sino = ft.forward_project(vol, angles)

        n_iter = 50
        cfg = ft.AdReconConfig(optim=ft.OptimConfig(n_iter=n_iter))
        _, losses = ft.run_ad_recon(sino, angles, config=cfg, return_loss=True)
        assert losses.shape == (n_iter,), (
            f"Expected loss history of length {n_iter}, got {losses.shape}"
        )
        assert losses.dtype == torch.float32

    def test_3d_mask_1_nz_nx_accepted(self):
        """ConstraintConfig.mask with shape [1, Nz, Nx] should not crash."""
        N = 16
        angles = make_angles(20, device=DEVICE)
        sino = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles)
        mask_3d = torch.ones(1, N, N, device=DEVICE)
        cfg = ft.AdReconConfig(
            optim=ft.OptimConfig(n_iter=5),
            constraints=ft.ConstraintConfig(support_mask=mask_3d),
        )
        vol = ft.run_ad_recon(sino, angles, config=cfg)
        assert vol.shape == (1, N, N)

    def test_3d_mask_ny_nz_nx_accepted(self):
        """ConstraintConfig.mask with shape [Ny, Nz, Nx] should not crash."""
        N, Ny = 16, 4
        angles = make_angles(20, device=DEVICE)
        vol_gt = phantom_2d(N, device=DEVICE).unsqueeze(0).expand(Ny, -1, -1).contiguous()
        sino = ft.forward_project(vol_gt, angles)
        mask_3d = torch.ones(Ny, N, N, device=DEVICE)
        cfg = ft.AdReconConfig(
            optim=ft.OptimConfig(n_iter=5),
            constraints=ft.ConstraintConfig(support_mask=mask_3d),
        )
        vol = ft.run_ad_recon(sino, angles, config=cfg)
        assert vol.shape == (Ny, N, N)

    def test_bad_3d_mask_shape_raises(self):
        """Wrong 3-D mask shape should raise a ValueError with a clear message."""
        N = 16
        angles = make_angles(20, device=DEVICE)
        sino = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles)
        bad_mask = torch.ones(3, N, N, device=DEVICE)  # Ny=3 but sino has Ny=1
        cfg = ft.AdReconConfig(
            optim=ft.OptimConfig(n_iter=1),
            constraints=ft.ConstraintConfig(support_mask=bad_mask),
        )
        with pytest.raises(ValueError, match="3-D"):
            ft.run_ad_recon(sino, angles, config=cfg)


@needs_cuda
class TestAdReconSupportMask:
    """support_mask kwarg on run_ad_recon (hard per-iteration constraint)."""

    def test_zeros_outside_support(self):
        N = 16
        angles = make_angles(20, device=DEVICE)
        sino = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles)

        M = torch.zeros(1, N, N, device=DEVICE)
        q = N // 4
        M[:, q:3*q, q:3*q] = 1.0

        cfg = ft.AdReconConfig(optim=ft.OptimConfig(n_iter=5))
        vol = ft.run_ad_recon(sino, angles, config=cfg, support_mask=M)
        assert vol[M == 0].abs().max().item() == 0.0
        assert vol[M == 1].abs().max().item() > 0.0

    def test_conflicts_with_config_mask(self):
        N = 16
        angles = make_angles(10, device=DEVICE)
        sino = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles)
        M = torch.ones(1, N, N, device=DEVICE)

        cfg_true = ft.AdReconConfig(
            optim=ft.OptimConfig(n_iter=1),
            constraints=ft.ConstraintConfig(cylinder=True),
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            ft.run_ad_recon(sino, angles, config=cfg_true, support_mask=M)

        cfg_tensor = ft.AdReconConfig(
            optim=ft.OptimConfig(n_iter=1),
            constraints=ft.ConstraintConfig(support_mask=M),
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            ft.run_ad_recon(sino, angles, config=cfg_tensor, support_mask=M)

    def test_bad_shape_raises(self):
        N = 16
        angles = make_angles(10, device=DEVICE)
        sino = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles)
        bad = torch.ones(3, N, N, device=DEVICE)   # Ny=3 but sino has Ny=1
        with pytest.raises(ValueError, match="support_mask"):
            ft.run_ad_recon(sino, angles, support_mask=bad)


# PR #39 added anchor_reference, tilt_bound_deg, antisymmetric_z and
# in_plane_weight with these tests; the merge c19ddee dropped the tests.
# Restored here, adapted from the old flat kwargs to AdReconConfig.
@needs_cuda
class TestAdReconPoseConstraints:
    # Ny=4 gives the rotmat projector real depth along the tilt axis so that
    # Y-Euler gradients are non-zero (needed for the drift tests).
    N        = 8
    NZ       = 8
    NY       = 4
    N_ANGLES = 5

    def _angles(self) -> torch.Tensor:
        return torch.tensor([-30., -15., 0., 15., 30.], device=DEVICE)

    def _sino(self) -> torch.Tensor:
        vol = torch.zeros(self.NY, self.NZ, self.N, device=DEVICE)
        # Asymmetric fill: only the first half of the Y slices, so Y-Euler
        # gradients are non-zero.
        vol[:self.NY // 2, 2:6, 2:6] = 1.0
        return ft.forward_project(vol, self._angles())  # [n_angles, NY, N_det]

    @staticmethod
    def _cfg(n_iter: int, **pose) -> ft.AdReconConfig:
        return ft.AdReconConfig(
            optim=ft.OptimConfig(n_iter=n_iter, cosine_schedule=False),
            pose=ft.PoseRefinementConfig(**pose) if pose else None,
        )

    def test_smoke_default_flags(self):
        """Default call (no pose config) returns the right shape/dtype/device."""
        vol = ft.run_ad_recon(self._sino(), self._angles(), config=self._cfg(5))
        assert vol.shape == (self.NY, self.NZ, self.N)
        assert vol.dtype == torch.float32
        assert vol.device.type == "cuda"

    def test_anchor_reference_row_frozen(self):
        """Reference row of euler and shifts is byte-identical to its initial value."""
        angles = self._angles()
        _, euler, shifts = ft.run_ad_recon(
            self._sino(), angles, return_pose=True,
            config=self._cfg(50, refine_pose=True, refine_shifts=True,
                             anchor_reference=True, lr_angles=1.0, lr_shifts=1.0),
        )
        ref_idx = int(angles.cpu().abs().argmin())
        expected_euler = torch.tensor([0.0, angles[ref_idx].item(), 0.0])
        assert torch.equal(euler[ref_idx], expected_euler), (
            f"Reference euler row drifted: {euler[ref_idx]} != {expected_euler}"
        )
        assert torch.equal(shifts[ref_idx], torch.zeros(2)), (
            f"Reference shift row drifted: {shifts[ref_idx]}"
        )
        # the other rows did move (the anchor is not a global freeze)
        others = [i for i in range(self.N_ANGLES) if i != ref_idx]
        assert (shifts[others].abs().amax() > 0).item()

    def test_tilt_bound_respected(self):
        """Y-Euler stays within nominal ± tilt_bound_deg under very high lr_angles."""
        angles = self._angles()
        bound = 1.0
        _, euler, _ = ft.run_ad_recon(
            self._sino(), angles, return_pose=True,
            config=self._cfg(50, refine_pose=True, tilt_bound_deg=bound, lr_angles=5.0),
        )
        max_drift = (euler[:, 1] - angles.cpu()).abs().max().item()
        assert max_drift <= bound + 1e-5, (
            f"Y-Euler drifted {max_drift:.4f}° beyond bound {bound}°"
        )

    def test_tilt_bound_none_allows_drift(self):
        """Without the bound, high lr_angles drives Y-Euler beyond ±1° of nominal."""
        angles = self._angles()
        _, euler, _ = ft.run_ad_recon(
            self._sino(), angles, return_pose=True,
            config=self._cfg(100, refine_pose=True, tilt_bound_deg=None, lr_angles=5.0),
        )
        max_drift = (euler[:, 1] - angles.cpu()).abs().max().item()
        assert max_drift > 1.0, (
            f"Expected Y-Euler drift > 1° with tilt_bound_deg=None and lr_angles=5.0, "
            f"but max drift was {max_drift:.4f}°"
        )

    def test_anchor_raises_without_refine(self):
        """anchor_reference=True without any refinement raises ValueError."""
        with pytest.raises(ValueError, match="anchor_reference"):
            ft.PoseRefinementConfig(anchor_reference=True)

    def test_antisymmetric_z_enforced(self):
        """Default antisymmetric_z=True keeps col2 == -col0 exactly after each step."""
        _, euler, _ = ft.run_ad_recon(
            self._sino(), self._angles(), return_pose=True,
            # tilt_bound_deg=None lets Y drift freely so gradients are non-trivial
            config=self._cfg(50, refine_pose=True, tilt_bound_deg=None, lr_angles=5.0),
        )
        residual = (euler[:, 2] + euler[:, 0]).abs().max().item()
        assert residual < 1e-5, (
            f"ZY(-Z) constraint violated: max |col2 + col0| = {residual:.2e}"
        )
        assert euler[:, 0].abs().max().item() > 1e-3   # col0 actually moved

    def test_antisymmetric_z_false_allows_drift(self):
        """antisymmetric_z=False lets col2 diverge from -col0 under high lr_angles."""
        _, euler, _ = ft.run_ad_recon(
            self._sino(), self._angles(), return_pose=True,
            config=self._cfg(100, refine_pose=True, tilt_bound_deg=None,
                             antisymmetric_z=False, lr_angles=5.0),
        )
        residual = (euler[:, 2] + euler[:, 0]).abs().max().item()
        assert residual > 1e-3, (
            f"Expected col2 != -col0 with antisymmetric_z=False, "
            f"but max |col2 + col0| = {residual:.2e}"
        )

    def test_in_plane_weight_drives_alpha_to_zero(self):
        """in_plane_weight suppresses α/γ drift that would otherwise occur at high lr_angles."""
        common = dict(refine_pose=True, antisymmetric_z=False, tilt_bound_deg=None,
                      lr_angles=5.0)
        _, euler_free, _ = ft.run_ad_recon(
            self._sino(), self._angles(), return_pose=True,
            config=self._cfg(100, in_plane_weight=0.0, **common),
        )
        baseline_drift = max(euler_free[:, 0].abs().max().item(),
                             euler_free[:, 2].abs().max().item())
        assert baseline_drift > 1.0, (
            f"Baseline (no penalty) should drift >1°, but max α/γ = "
            f"{baseline_drift:.4f}°"
        )
        _, euler_reg, _ = ft.run_ad_recon(
            self._sino(), self._angles(), return_pose=True,
            config=self._cfg(100, in_plane_weight=10.0, **common),
        )
        reg_drift = max(euler_reg[:, 0].abs().max().item(),
                        euler_reg[:, 2].abs().max().item())
        assert reg_drift < 0.5, (
            f"With in_plane_weight=10.0, max |α|/|γ| = {reg_drift:.4f}° (expected < 0.5°)"
        )

    def test_in_plane_weight_skipped_without_pose(self):
        """in_plane_weight has no effect when refine_pose=False (no Euler
        parameters exist): the result equals the no-pose run."""
        cfg = self._cfg(5, refine_shifts=True, in_plane_weight=1e-2)
        vol = ft.run_ad_recon(self._sino(), self._angles(), config=cfg)
        assert vol.shape == (self.NY, self.NZ, self.N)
        assert torch.isfinite(vol).all()
        cfg0 = self._cfg(5, refine_shifts=True)
        losses = ft.run_ad_recon(self._sino(), self._angles(), config=cfg, return_loss=True)[1]
        losses0 = ft.run_ad_recon(self._sino(), self._angles(), config=cfg0, return_loss=True)[1]
        torch.testing.assert_close(losses, losses0, rtol=1e-6, atol=1e-7)


# ── staged pose schedules (former test_ad_recon_staging.py) ───────────────────


def test_default_schedule_is_historical_joint_update() -> None:
    """All-default new fields preserve the old step schedule structurally."""
    cfg = PoseRefinementConfig(refine_pose=True, refine_shifts=True)
    assert cfg.pose_warmup_iters == 0
    assert cfg.alternating_period == 0
    assert cfg.pose_init_euler_deg is None
    assert cfg.pose_init_shifts is None
    assert [
        _ad_recon_step_schedule(
            iteration,
            pose_warmup_iters=cfg.pose_warmup_iters,
            alternating_period=cfg.alternating_period,
        )
        for iteration in range(12)
    ] == [(True, True)] * 12


def test_warmup_then_alternating_schedule() -> None:
    observed = [
        _ad_recon_step_schedule(
            iteration, pose_warmup_iters=3, alternating_period=2
        )
        for iteration in range(11)
    ]
    assert observed == [
        (True, False),
        (True, False),
        (True, False),
        (True, False),
        (True, False),
        (False, True),
        (False, True),
        (True, False),
        (True, False),
        (False, True),
        (False, True),
    ]


def test_warmup_without_alternation_becomes_joint() -> None:
    assert [
        _ad_recon_step_schedule(i, pose_warmup_iters=2)
        for i in range(5)
    ] == [
        (True, False),
        (True, False),
        (True, True),
        (True, True),
        (True, True),
    ]


@pytest.mark.parametrize("field", ["pose_warmup_iters", "alternating_period"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_stage_lengths_must_be_non_negative_integers(
    field: str, value: object
) -> None:
    kwargs = {field: value, "refine_pose": True}
    with pytest.raises(ValueError, match=field):
        PoseRefinementConfig(**kwargs)


def test_alternation_requires_pose_or_shift_refinement() -> None:
    with pytest.raises(ValueError, match="there is no pose-only block"):
        PoseRefinementConfig(alternating_period=2)


def test_initialisers_require_the_corresponding_refinement() -> None:
    with pytest.raises(ValueError, match="requires refine_pose"):
        PoseRefinementConfig(pose_init_euler_deg=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="requires refine_shifts"):
        PoseRefinementConfig(pose_init_shifts=[[0.0, 0.0]])


def test_pose_initialiser_preserves_values_as_independent_float32_tensor() -> None:
    source = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
    )
    initial = _prepare_ad_recon_pose_init(
        source,
        expected_shape=(2, 3),
        device=torch.device("cpu"),
        name="pose_init_euler_deg",
    )
    assert initial.dtype == torch.float32
    assert torch.equal(initial, source.float())
    initial[0, 0] = 99.0
    assert source[0, 0].item() == 1.0


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([[0.0, 1.0]], "must have shape"),
        ([[0.0, float("nan"), 1.0]], "only finite"),
        ([[0.0, float("inf"), 1.0]], "only finite"),
        ([["bad", 0.0, 1.0]], "finite numeric array"),
    ],
)
def test_pose_initialiser_rejects_invalid_values(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare_ad_recon_pose_init(
            value,
            expected_shape=(1, 3),
            device=torch.device("cpu"),
            name="pose_init_euler_deg",
        )


@pytest.mark.parametrize(("n_iter", "warmup", "period", "expected"), [
    (20, 0, 0, (20, 20)),          # default joint schedule: T_max = n_iter
    (20, 5, 0, (20, 15)),          # warm-up then joint
    (20, 4, 4, (12, 8)),           # the review's example
    (14, 2, 3, (8, 6)),
    (5, 10, 0, (5, 0)),            # warm-up longer than the run
])
def test_active_steps_counts_each_optimiser(n_iter, warmup, period, expected) -> None:
    assert _ad_recon_active_steps(
        n_iter, pose_warmup_iters=warmup, alternating_period=period) == expected
    flags = [_ad_recon_step_schedule(i, pose_warmup_iters=warmup, alternating_period=period)
             for i in range(n_iter)]
    assert expected == (sum(v for v, _ in flags), sum(p for _, p in flags))


class _ZMeanProjector:
    """CPU stand-in for ParallelProjector (Z-mean per view)."""

    def __init__(self, n_angles: int = 5):
        self.n_angles = n_angles

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return volume.mean(dim=1).unsqueeze(0).expand(self.n_angles, -1, -1)


def _run_cpu(monkeypatch, cfg, **kwargs):
    """run_ad_recon on CPU: the CUDA-only input check and the projector
    class are replaced (shift refinement works with a parallel projector)."""
    def fake_check_inputs(sinogram, angles_deg, Nz, **kw):
        n, ny, nd = sinogram.shape
        return n, ny, nd, (nd if Nz < 0 else Nz), sinogram.device

    monkeypatch.setattr(algo.ad, "ParallelProjector", _ZMeanProjector)
    monkeypatch.setattr(algo._common, "_check_inputs", fake_check_inputs)
    sino = torch.rand(5, 2, 6, generator=torch.Generator().manual_seed(0))
    return ft.run_ad_recon(sino, torch.linspace(-30, 30, 5), Nz=6,
                           projector=_ZMeanProjector(), config=cfg, **kwargs)


@pytest.mark.parametrize(("warmup", "period"), [(5, 0), (4, 4), (0, 0)])
def test_cosine_schedule_reaches_floor_per_optimiser(monkeypatch, warmup, period) -> None:
    """Each cosine schedule's T_max is its optimiser's number of active
    iterations, so every LR is ~0 after the run. With T_max = n_iter the
    staged pose LR ended at 21 % (warm-up 5) and the volume LR at 42 % /
    pose at 73 % (warm-up 4 + period 4) of the initial value."""
    made = []
    real = torch.optim.lr_scheduler.CosineAnnealingLR

    def recording(opt, T_max, **kw):
        sched = real(opt, T_max=T_max, **kw)
        made.append((opt, T_max, sched))
        return sched

    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", recording)
    n_iter = 20
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=n_iter, lr_volume=0.05, cosine_schedule=True),
        pose=ft.PoseRefinementConfig(refine_shifts=True, lr_shifts=0.1,
                                     pose_warmup_iters=warmup, alternating_period=period),
    )
    _run_cpu(monkeypatch, cfg)
    vol_steps, pose_steps = _ad_recon_active_steps(
        n_iter, pose_warmup_iters=warmup, alternating_period=period)
    assert [t for _, t, _ in made] == [vol_steps, pose_steps]
    for opt, _, sched in made:
        assert sched.last_epoch == sched.T_max          # stepped exactly T_max times
        assert opt.param_groups[0]["lr"] < 1e-12


@pytest.mark.parametrize(("mutate", "match"), [
    (lambda c: setattr(c.pose, "refine_shifts", False), "alternating_period > 0 requires"),
    (lambda c: setattr(c.pose, "alternating_period", -2), "non-negative integer"),
    (lambda c: setattr(c.pose, "pose_init_euler_deg", [[0.0, 0.0, 0.0]]), "requires refine_pose"),
    (lambda c: setattr(c.pose, "lr_shifts", float("nan")), "lr_shifts"),
    (lambda c: setattr(c.optim, "loss_fn", "huber"), "loss_fn"),
    (lambda c: setattr(c.optim, "n_iter", -1), "n_iter"),
    (lambda c: setattr(c.optim, "lr_volume", -0.1), "lr_volume"),
    (lambda c: setattr(c.reg, "tv_weight", -1.0), "tv_weight"),
    (lambda c: setattr(c.constraints, "soft_threshold", float("inf")), "soft_threshold"),
    (lambda c: setattr(c.constraints, "support_mask", "cylinder"), "support_mask"),
    (lambda c: setattr(c.constraints, "cylinder", "yes"), "cylinder"),
    (lambda c: setattr(c, "pose", object()), "PoseRefinementConfig"),
])
def test_run_ad_recon_revalidates_mutated_config(mutate, match) -> None:
    """Construction-time validation is bypassed by editing a field later;
    run_ad_recon validates again on entry (before any CUDA check, so this
    runs on CPU). Before, alternating_period without a refine flag silently
    froze the volume half the time and a negative period gave a garbage
    schedule."""
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=2),
        pose=ft.PoseRefinementConfig(refine_shifts=True, alternating_period=2),
    )
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        ft.run_ad_recon(torch.zeros(5, 2, 6), torch.linspace(-30, 30, 5), config=cfg)


def test_pose_init_shifts_without_refine_is_caught_after_mutation() -> None:
    cfg = ft.AdReconConfig(pose=ft.PoseRefinementConfig(refine_pose=True))
    cfg.pose.pose_init_shifts = [[0.0, 0.0]] * 5
    with pytest.raises(ValueError, match="pose_init_shifts requires refine_shifts"):
        ft.run_ad_recon(torch.zeros(5, 2, 6), torch.linspace(-30, 30, 5), config=cfg)


def test_warmup_requires_refinement() -> None:
    """Consistent with alternating_period: a warm-up with nothing to warm up
    for used to be accepted as a silent no-op."""
    with pytest.raises(ValueError, match="pose_warmup_iters > 0 requires"):
        PoseRefinementConfig(pose_warmup_iters=3)


def test_warmup_longer_than_run_warns(monkeypatch) -> None:
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=3),
        pose=ft.PoseRefinementConfig(refine_shifts=True, pose_warmup_iters=3),
    )
    with pytest.warns(UserWarning, match="never step"):
        _, _, shifts = _run_cpu(monkeypatch, cfg, return_pose=True)
    assert torch.equal(shifts, torch.zeros(5, 2))


class _RecordingRotmatProjector(ft.RotmatProjector):
    """Records the volume and rotation matrices each iteration's forward
    projection sees, i.e. the state *before* that iteration's step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vols, self.mats = [], []

    def forward(self, volume, mats):
        self.vols.append(volume.detach().clone())
        self.mats.append(mats.detach().clone())
        return super().forward(volume, mats)


def _staged_run(monkeypatch, n_iter, warmup, period):
    torch.manual_seed(0)
    Ny, N, n = 4, 24, 15
    dev = torch.device("cuda")
    angles = torch.linspace(-60, 60, n, device=dev)
    vol = torch.zeros(Ny, N, N, device=dev)
    vol[:, 6:18, 8:16] = 1.0
    vol[:2, 10:14, 10:12] = 2.0
    euler_true = torch.zeros(n, 3, device=dev)
    euler_true[:, 1] = angles + 0.5 * torch.sin(torch.arange(n, device=dev))
    shifts_true = 0.5 * torch.cos(torch.arange(n, device=dev)).unsqueeze(1).repeat(1, 2)
    sino = ft.forward_project_3d(vol, euler_true, shifts=shifts_true)

    shift_log = []
    real_shift = algo.ad.apply_shifts_2d

    def recording_shift(pred, shifts):
        shift_log.append(shifts.detach().clone())
        return real_shift(pred, shifts)

    monkeypatch.setattr(algo.ad, "apply_shifts_2d", recording_shift)
    proj = _RecordingRotmatProjector((Ny, N, N), n_angles=n, device=dev)
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=n_iter, lr_volume=0.05),
        pose=ft.PoseRefinementConfig(refine_pose=True, refine_shifts=True,
                                     lr_angles=0.05, lr_shifts=0.05,
                                     pose_warmup_iters=warmup, alternating_period=period),
    )
    calls = []
    rec, loss, euler, shifts = ft.run_ad_recon(
        sino, angles, projector=proj, config=cfg, return_loss=True, return_pose=True,
        loss_callback=lambda i, v: calls.append(i))
    assert calls == list(range(n_iter))
    # state after the last step, appended so step k is (k -> k+1)
    vols = proj.vols + [rec]
    mats = proj.mats + [ft.euler_zyz_to_rotmat(euler.to(dev))]
    shift_log.append(shifts.to(dev))
    return loss, vols, mats, shift_log


def _moved(seq, k):
    return not torch.equal(seq[k], seq[k + 1])


@needs_cuda
def test_run_ad_recon_warmup_end_to_end(monkeypatch) -> None:
    """Warm-up 6 of 16 iterations: the pose (Euler angles and shifts) does
    not move during warm-up and moves on every later iteration; the volume
    moves on every iteration; the loss falls."""
    n_iter, warmup = 16, 6
    loss, vols, mats, shifts = _staged_run(monkeypatch, n_iter, warmup, 0)
    for k in range(n_iter):
        assert _moved(vols, k), k
    for k in range(n_iter):
        pose_active = k >= warmup
        assert _moved(mats, k) == pose_active, k
        assert _moved(shifts, k) == pose_active, k
    assert (loss[1:] <= loss[:-1]).all()
    assert loss[-1] < 0.25 * loss[0]


@needs_cuda
def test_run_ad_recon_alternating_end_to_end(monkeypatch) -> None:
    """Warm-up 2 then period 3: volume-only (0-1 warm-up, 2-4, 8-10) and
    pose-only (5-7, 11-13) blocks. Each parameter moves exactly in its own
    blocks; the loss falls."""
    n_iter, warmup, period = 14, 2, 3
    loss, vols, mats, shifts = _staged_run(monkeypatch, n_iter, warmup, period)
    for k in range(n_iter):
        step_volume, step_pose = _ad_recon_step_schedule(
            k, pose_warmup_iters=warmup, alternating_period=period)
        assert _moved(vols, k) == step_volume, k
        assert _moved(mats, k) == step_pose, k
        assert _moved(shifts, k) == step_pose, k
    assert [k for k in range(n_iter) if _ad_recon_step_schedule(
        k, pose_warmup_iters=warmup, alternating_period=period)[1]] == [5, 6, 7, 11, 12, 13]
    assert (loss[1:] <= loss[:-1]).all()
    assert loss[8] < loss[5]            # the pose-only block alone lowers the loss
    assert loss[-1] < 0.5 * loss[0]


def test_soft_threshold_skipped_in_pose_only_iterations(monkeypatch) -> None:
    """#82 review N2: ConstraintConfig.soft_threshold must not run in
    pose-only iterations. It is not idempotent, so applying it there would
    shrink the frozen volume on every pose-only step. Alternating period 2:
    volume steps 0-1 and 4-5, pose-only steps 2-3."""
    seen = []

    class _Recording(_ZMeanProjector):
        def forward(self, volume):
            seen.append(volume.detach().clone())
            return super().forward(volume)

    def fake_check_inputs(sinogram, angles_deg, Nz, **kw):
        n, ny, nd = sinogram.shape
        return n, ny, nd, (nd if Nz < 0 else Nz), sinogram.device

    monkeypatch.setattr(algo.ad, "ParallelProjector", _Recording)
    monkeypatch.setattr(algo._common, "_check_inputs", fake_check_inputs)
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=6, lr_volume=0.2, cosine_schedule=False),
        constraints=ft.ConstraintConfig(soft_threshold=0.01),
        pose=ft.PoseRefinementConfig(refine_shifts=True, lr_shifts=0.1, alternating_period=2),
    )
    sino = torch.rand(5, 2, 6, generator=torch.Generator().manual_seed(0)) + 1.0
    rec = ft.run_ad_recon(sino, torch.linspace(-30, 30, 5), Nz=6,
                          projector=_Recording(), config=cfg)
    vols = seen + [rec]                       # step k maps vols[k] -> vols[k + 1]
    assert rec.abs().max() > 0.1              # far above the threshold: it would bite
    for k in range(6):
        step_volume, _ = _ad_recon_step_schedule(k, alternating_period=2)
        assert (not torch.equal(vols[k], vols[k + 1])) == step_volume, k


# ── run_grad_descent ──────────────────────────────────────────────────────────


def test_run_grad_descent_cpu_converges():
    """Adam-driven reconstruction loss drops by >50× from initial."""
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    rec, losses = ft.run_grad_descent(
        sino, angles, n_iter=200, lr=5e-2, return_loss=True
    )

    assert rec.shape == vol.shape
    assert not rec.requires_grad
    assert losses[-1].item() < losses[0].item() / 50.0, (
        f"grad_descent did not converge: loss[0]={losses[0]:.4e}, "
        f"loss[-1]={losses[-1]:.4e}"
    )


def test_run_grad_descent_legacy_positional_call_raises_cpu():
    """Everything after (sinogram, angles_deg, n_iter) is keyword-only, so a
    call written for the old positional order raises TypeError instead of
    binding to the wrong parameters; the keyword spelling runs."""
    N      = 24
    angles = make_angles(12)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    with pytest.raises(TypeError):
        ft.run_grad_descent(sino, angles, 5, 5e-2, -1, torch.optim.Adam)

    rec, losses = ft.run_grad_descent(
        sino, angles, 5, lr=5e-2, Nz=-1, optimizer_cls=torch.optim.Adam,
        positivity=False, mask=False, tv_weight=0.0, euler_zyz_deg=None,
        use_resire_projector=False, return_loss=True,
    )

    assert rec.shape == vol.shape
    assert losses.shape == torch.Size([5])
    assert torch.isfinite(losses).all().item()


@needs_cuda
def test_run_grad_descent_cuda_matches_sirt():
    """
    Adam-driven autograd reconstruction reaches comparable NMSE to SIRT on
    the same phantom (within 2× of SIRT@80 NMSE). Mirrors the SIRT
    convergence test geometry.
    """
    N      = 64
    angles = make_angles(60, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    rec_sirt = ft.run_sirt(sino, angles, n_iter=80, Nz=N, positivity=True)
    rec_grad = ft.run_grad_descent(
        sino, angles, n_iter=300, lr=5e-2, positivity=True
    )

    def nmse(a, b):
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        b = (b - b.min()) / (b.max() - b.min() + 1e-9)
        return ((a - b) ** 2).mean().item()

    err_sirt = nmse(rec_sirt, vol)
    err_grad = nmse(rec_grad, vol)
    assert err_grad < 2.0 * err_sirt + 1e-3, (
        f"run_grad_descent NMSE too high: grad={err_grad:.4f}, sirt={err_sirt:.4f}"
    )


def make_euler_pure_y(angles_deg: torch.Tensor) -> torch.Tensor:
    """Wrap scalar tilt angles as ZYZ triples (φ=0, θ=angle, ψ=0)."""
    n = len(angles_deg)
    euler = torch.zeros(n, 3)
    euler[:, 1] = angles_deg.cpu()
    return euler


@needs_cuda
def test_run_grad_descent_3d_cuda_converges():
    """run_grad_descent with the 3D rotmat projector (CUDA) converges."""
    N      = 32
    angles = make_angles(30, device=DEVICE)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0).to(DEVICE)
    sino   = ft.forward_project_3d(vol, euler)

    rec, losses = ft.run_grad_descent(
        sino, euler_zyz_deg=euler, n_iter=150, lr=5e-2, return_loss=True
    )

    assert rec.shape == vol.shape
    assert losses[-1].item() < losses[0].item() / 20.0
