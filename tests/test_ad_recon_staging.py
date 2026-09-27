"""CPU-safe contract tests for staged AD pose optimisation."""

import pytest
import torch

from fasttomo.algorithms import (
    _ad_recon_step_schedule,
    _prepare_ad_recon_pose_init,
)
from fasttomo.configs import PoseRefinementConfig


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


# ── #79 review follow-ups ──────────────────────────────────────────────────

import fasttomo as ft  # noqa: E402
import fasttomo.algorithms as algo  # noqa: E402
from fasttomo.algorithms import _ad_recon_active_steps  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU available")


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


# ── end-to-end staged runs on CUDA ─────────────────────────────────────────

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
