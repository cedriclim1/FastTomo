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
