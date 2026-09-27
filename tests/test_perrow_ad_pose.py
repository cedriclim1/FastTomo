"""CPU correctness checks for per-line AD pose refinement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from benchmarks.scripts.utils.perrow_ad import (
    ADConfig,
    POSE_AXIS_BOUND_DEG,
    POSE_CONVENTION,
    POSE_SHIFT_BOUND_PX,
    PerLineGeometry,
    _backward_fidelity_fixed_pose_chunks,
    _bounded_pose,
    _expand_learned_pose,
    generator_to_fasttomo_volume,
    interpolate_pose_nodes,
    reconstruct,
)
from tests.test_perrow_ad import assemble_reference, reference_generator_projections


def _asymmetric_volume(size: int = 17) -> torch.Tensor:
    volume_zyx = torch.zeros(size, size, size)
    volume_zyx[3:7, 4:8, 10:14] = 1.0
    volume_zyx[9:13, 11:15, 5:8] = 0.6
    volume_zyx[6, 9, 13] = 2.0
    return volume_zyx


def _generator_pose_projections(
    volume_zyx: torch.Tensor,
    angles_deg: torch.Tensor,
    axis_deg: torch.Tensor,
    shifts_dy_dx: torch.Tensor,
) -> torch.Tensor:
    """Independent generator ``rot_ZXZ(..., z3=phi); sum-Z; shift`` reference."""
    projections = reference_generator_projections(volume_zyx, angles_deg)
    radians = torch.deg2rad(axis_deg)
    cosines, sines = radians.cos(), radians.sin()
    zeros = torch.zeros_like(radians)
    rotations = torch.stack(
        (cosines, -sines, zeros, sines, cosines, zeros), dim=-1
    ).reshape(-1, 2, 3)
    projection_batch = projections[:, None]
    rotated = F.grid_sample(
        projection_batch,
        F.affine_grid(rotations, projection_batch.shape, align_corners=False),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0]

    n_projections, height, width = rotated.shape
    translations = torch.zeros(n_projections, 2, 3)
    translations[:, 0, 0] = 1.0
    translations[:, 1, 1] = 1.0
    translations[:, 0, 2] = -2.0 * shifts_dy_dx[:, 1] / width
    translations[:, 1, 2] = -2.0 * shifts_dy_dx[:, 0] / height
    rotated_batch = rotated[:, None]
    return F.grid_sample(
        rotated_batch,
        F.affine_grid(translations, rotated_batch.shape, align_corners=False),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0]


def _assemble_generator_pose_reference(
    volume_zyx: torch.Tensor,
    angles_per_line: torch.Tensor,
    *,
    arm: str,
    axis_deg_per_line: torch.Tensor,
    shifts_per_line: torch.Tensor,
) -> torch.Tensor:
    n_frames, n_lines = angles_per_line.shape
    projections = _generator_pose_projections(
        volume_zyx,
        angles_per_line.reshape(-1),
        axis_deg_per_line.reshape(-1),
        shifts_per_line.reshape(-1, 2),
    )
    slots = torch.arange(n_frames * n_lines)
    line_indices = torch.arange(n_lines).repeat(n_frames)
    if arm == "fast-par":
        return projections[slots, line_indices].reshape(
            n_frames, n_lines, volume_zyx.shape[2]
        )
    columns = projections[slots, :, line_indices]
    return columns.reshape(n_frames, n_lines, volume_zyx.shape[1]).permute(0, 2, 1)


def _assert_absolute_pose_arm(arm: str) -> None:
    size = 17
    n_frames = 2
    volume_zyx = _asymmetric_volume(size)
    angles = (-37.0 + torch.arange(n_frames * size) * (55.0 / size)).reshape(
        n_frames, size
    )
    shift_dy_dx = torch.tensor([1.5, -1.0])
    shifts = shift_dy_dx.expand(n_frames, size, 2)
    axis = torch.full((n_frames, size), 1.0)
    reference = _assemble_generator_pose_reference(
        volume_zyx,
        angles,
        arm=arm,
        axis_deg_per_line=axis,
        shifts_per_line=shifts,
    )
    geometry = PerLineGeometry.from_angles(
        angles,
        arm=arm,  # type: ignore[arg-type]
        frame_shape=tuple(reference.shape),
    )
    actual = geometry.predict(
        generator_to_fasttomo_volume(volume_zyx),
        chunk_size=9,
        shifts_per_line=shifts,
        axis_deg_per_line=axis,
    )

    relative_l2_error = (actual - reference).norm() / reference.norm()
    relative_max_error = (actual - reference).abs().max() / reference.abs().max()
    print(
        f"{arm} absolute pose residual: relative_l2={relative_l2_error.item():.8f}, "
        f"relative_max={relative_max_error.item():.8f}"
    )
    assert relative_l2_error.item() < 1e-5
    assert relative_max_error.item() < 1e-5


def _tiny_case() -> tuple[torch.Tensor, torch.Tensor, PerLineGeometry]:
    size = 11
    volume_zyx = torch.zeros(size, size, size)
    volume_zyx[3:8, 3:7, 4:8] = 1.0
    volume_zyx[5:7, 6:8, 2:4] = 0.4
    angles = (-38.0 + torch.arange(2 * size) * (19.0 / size)).reshape(2, size)
    target = assemble_reference(volume_zyx, angles, "fast-par")
    geometry = PerLineGeometry.from_angles(
        angles, arm="fast-par", frame_shape=tuple(target.shape)
    )
    return volume_zyx, target, geometry


def test_interpolate_pose_nodes_frame_centres_and_clamped_ends() -> None:
    nodes = torch.tensor([0.0, 10.0, 20.0])
    actual = interpolate_pose_nodes(nodes, n_lines=5)

    assert actual.shape == (3, 5)
    assert actual[0, 0].item() == pytest.approx(0.0)
    assert actual[0, 2].item() == pytest.approx(0.0)
    assert actual[1, 2].item() == pytest.approx(10.0)
    assert actual[2, 2].item() == pytest.approx(20.0)
    assert actual[2, 4].item() == pytest.approx(20.0)
    assert actual[0, 4].item() == pytest.approx(4.0)
    assert actual[1, 0].item() == pytest.approx(6.0)


@pytest.mark.parametrize("arm", ["fast-par", "fast-perp"])
def test_appended_zero_reference_dedup_and_pose_paths(arm: str) -> None:
    """Repeated zero angles dedup safely and the reference uses its own pose."""
    size = 11
    n_ramp_frames = 3
    volume_zyx, _, _ = _tiny_case()
    ramp_angles = torch.linspace(
        -25.0, 25.0, n_ramp_frames * size
    ).reshape(n_ramp_frames, size)
    angles = torch.cat((ramp_angles, torch.zeros(1, size)), dim=0)
    frame_shape = (n_ramp_frames + 1, size, size)
    geometry = PerLineGeometry.from_angles(
        angles,
        arm=arm,  # type: ignore[arg-type]
        frame_shape=frame_shape,
    )
    volume = generator_to_fasttomo_volume(volume_zyx)

    nominal_reference = assemble_reference(volume_zyx, angles, arm)
    nominal_actual = geometry.predict(volume, chunk_size=7)
    torch.testing.assert_close(nominal_actual, nominal_reference, atol=1e-5, rtol=0.0)

    zero_slots = geometry.line_to_angle[angles.reshape(-1) == 0.0]
    assert zero_slots.numel() == size + 1
    assert torch.unique(zero_slots).numel() == 1
    assert geometry.n_pseudo_angles == n_ramp_frames * size

    shifts = torch.zeros(n_ramp_frames + 1, size, 2)
    shifts[:-1] = torch.tensor([0.7, -0.5])
    axis = torch.zeros(n_ramp_frames + 1, size)
    axis[:-1] = 0.9
    pose_reference = _assemble_generator_pose_reference(
        volume_zyx,
        angles,
        arm=arm,
        axis_deg_per_line=axis,
        shifts_per_line=shifts,
    )
    pose_actual = geometry.predict(
        volume,
        chunk_size=7,
        shifts_per_line=shifts,
        axis_deg_per_line=axis,
    )
    torch.testing.assert_close(pose_actual, pose_reference, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(
        pose_actual[-1], nominal_actual[-1], atol=1e-6, rtol=0.0
    )


def test_pose_prediction_gradients_reach_line_pose() -> None:
    volume_zyx, _, geometry = _tiny_case()
    volume = generator_to_fasttomo_volume(volume_zyx).requires_grad_(True)
    n_frames, n_lines = geometry.angles_deg.new_tensor(
        geometry.frame_shape[:2], dtype=torch.long
    ).tolist()
    shifts = torch.zeros(n_frames, n_lines, 2, requires_grad=True)
    axis = torch.zeros(n_frames, n_lines, requires_grad=True)

    prediction = geometry.predict(
        volume,
        chunk_size=6,
        shifts_per_line=shifts,
        axis_deg_per_line=axis,
    )
    weights = torch.linspace(0.1, 1.0, prediction.numel()).reshape_as(prediction)
    (prediction * weights).sum().backward()

    assert shifts.grad is not None and torch.isfinite(shifts.grad).all()
    assert axis.grad is not None and torch.isfinite(axis.grad).all()
    assert shifts.grad[..., 0].abs().sum().item() > 0.0
    assert shifts.grad[..., 1].abs().sum().item() > 0.0
    assert axis.grad.abs().sum().item() > 0.0
    print(
        "exact pose gradient L1:",
        f"dy={shifts.grad[..., 0].abs().sum().item():.6f}",
        f"dx={shifts.grad[..., 1].abs().sum().item():.6f}",
        f"phi={axis.grad.abs().sum().item():.6f}",
    )


def test_fast_par_pose_matches_generator_absolute_frame() -> None:
    """Rows use the correct line angle, shift order/sign, and phi sign."""
    _assert_absolute_pose_arm("fast-par")


def test_fast_perp_pose_matches_generator_absolute_frame() -> None:
    """Columns use the correct line angle, shift order/sign, and phi sign."""
    _assert_absolute_pose_arm("fast-perp")


def test_exact_pose_theta_sweep_stays_at_projector_floor() -> None:
    """Exact 2-D phi adds no theta-dependent error above the even-size floor."""
    size = 20
    volume_zyx = _asymmetric_volume(size)
    theta = torch.tensor([-89.0, -75.0, -60.0, -30.0, 0.0, 30.0, 60.0, 75.0, 89.0])
    angles = theta[:, None].expand(-1, size).clone()
    geometry = PerLineGeometry.from_angles(
        angles, arm="fast-par", frame_shape=(len(theta), size, size)
    )
    expected_residuals = {
        0.0: torch.tensor(
            [0.00006, 0.00738, 0.02195, 0.02199, 0.0, 0.02044, 0.02276, 0.01242, 0.00013]
        ),
        1.0: torch.tensor(
            [0.00005, 0.00694, 0.02080, 0.02065, 0.0, 0.01946, 0.02141, 0.01144, 0.00012]
        ),
        8.0: torch.tensor(
            [0.00002, 0.00483, 0.01379, 0.01432, 0.0, 0.01541, 0.01570, 0.00764, 0.00005]
        ),
    }
    for phi, expected in expected_residuals.items():
        axis = torch.full((len(theta), size), phi)
        shifts = torch.zeros(len(theta), size, 2)
        reference = _assemble_generator_pose_reference(
            volume_zyx,
            angles,
            arm="fast-par",
            axis_deg_per_line=axis,
            shifts_per_line=shifts,
        )
        actual = geometry.predict(
            generator_to_fasttomo_volume(volume_zyx),
            chunk_size=32,
            shifts_per_line=shifts,
            axis_deg_per_line=axis,
        )
        residuals = (actual - reference).flatten(1).norm(dim=1) / reference.flatten(
            1
        ).norm(dim=1)
        max_residuals = (actual - reference).abs().flatten(1).max(
            dim=1
        ).values / reference.abs().flatten(1).max(dim=1).values
        print(
            f"phi={phi:g} theta-sweep relative_l2={residuals.tolist()} "
            f"relative_max={max_residuals.tolist()}"
        )
        torch.testing.assert_close(residuals, expected, rtol=0.0, atol=5e-4)
        assert residuals.max().item() < 0.025
        assert max_residuals.max().item() < 0.04

    assert "applied exactly" in POSE_CONVENTION
    assert "No ZYZ Euler approximation is used" in POSE_CONVENTION


def test_oracle_fixed_pose_path_reproduces_absolute_reference() -> None:
    """The reconstruction oracle consumes exact per-line GT without interpolation."""
    size = 17
    n_frames = 2
    volume_zyx = _asymmetric_volume(size)
    angles = (-31.0 + torch.arange(n_frames * size) * (46.0 / size)).reshape(
        n_frames, size
    )
    line_time = torch.linspace(0.0, 1.0, n_frames * size).reshape(n_frames, size)
    shifts = torch.stack(
        (0.4 + 0.6 * line_time, -0.8 + 0.6 * line_time), dim=-1
    )
    axis = 0.8 + 0.4 * line_time
    reference = _assemble_generator_pose_reference(
        volume_zyx,
        angles,
        arm="fast-par",
        axis_deg_per_line=axis,
        shifts_per_line=shifts,
    )
    geometry = PerLineGeometry.from_angles(
        angles, arm="fast-par", frame_shape=tuple(reference.shape)
    )
    volume = generator_to_fasttomo_volume(volume_zyx).requires_grad_(True)
    direct_prediction = geometry.predict(
        volume,
        chunk_size=9,
        shifts_per_line=shifts,
        axis_deg_per_line=axis,
    )
    expected_fidelity = (direct_prediction.detach() - reference).square().mean().item()
    oracle_fidelity = _backward_fidelity_fixed_pose_chunks(
        volume,
        reference,
        geometry,
        ADConfig(iters=1, chunk=9),
        shifts,
        axis,
    )

    assert oracle_fidelity == pytest.approx(expected_fidelity, rel=0.0, abs=1e-7)
    assert volume.grad is not None and volume.grad.abs().sum().item() > 0.0
    relative_l2 = (direct_prediction.detach() - reference).norm() / reference.norm()
    assert relative_l2.item() < 1e-5

    result = reconstruct(
        reference,
        geometry,
        config=ADConfig(iters=1, lr=0.01, chunk=9),
        device="cpu",
        fixed_shifts_per_line=shifts,
        fixed_axis_deg_per_line=axis,
    )
    assert result.poses is not None
    torch.testing.assert_close(result.poses.shifts_per_line, shifts)
    torch.testing.assert_close(result.poses.axis_deg_per_line, axis)
    assert "fixed dataset GT oracle" in result.poses.convention
    print(
        "oracle absolute reference:",
        f"relative_l2={relative_l2.item():.8f}",
        f"fidelity={oracle_fidelity:.8e}",
    )


def test_gt_initialization_and_frozen_control_preserve_nodes() -> None:
    """Inverse bounds initialize physical nodes; pose_start freezes them exactly."""
    size = 7
    n_frames = 3
    angles = torch.linspace(-20.0, 20.0, n_frames * size).reshape(n_frames, size)
    geometry = PerLineGeometry.from_angles(
        angles, arm="fast-par", frame_shape=(n_frames, size, size)
    )
    target = torch.zeros(n_frames, size, size)
    shifts_init = torch.tensor([[0.4, -0.3], [1.2, -0.8], [2.0, -1.1]])
    axis_init = torch.tensor([0.8, 1.0, 1.2])
    frozen = reconstruct(
        target,
        geometry,
        config=ADConfig(iters=2, chunk=8, refine_pose=True, pose_start=2),
        device="cpu",
        initial_shifts_frames=shifts_init,
        initial_axis_deg_frames=axis_init,
    )
    assert frozen.poses is not None
    torch.testing.assert_close(
        frozen.poses.shifts_frames, shifts_init, atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        frozen.poses.axis_deg_frames, axis_init, atol=1e-6, rtol=0.0
    )

    zero_frozen = reconstruct(
        target,
        geometry,
        config=ADConfig(iters=2, chunk=8, refine_pose=True, pose_start=2),
        device="cpu",
    )
    assert zero_frozen.poses is not None
    assert torch.count_nonzero(zero_frozen.poses.shifts_frames).item() == 0
    assert torch.count_nonzero(zero_frozen.poses.axis_deg_frames).item() == 0


def test_oracle_fixed_pose_bypasses_learned_bounds() -> None:
    size = 5
    geometry = PerLineGeometry.from_angles(
        torch.zeros(1, size), arm="fast-par", frame_shape=(1, size, size)
    )
    shifts = torch.full((1, size, 2), POSE_SHIFT_BOUND_PX + 2.0)
    axis = torch.full((1, size), POSE_AXIS_BOUND_DEG + 0.5)
    result = reconstruct(
        torch.zeros(1, size, size),
        geometry,
        config=ADConfig(iters=1, chunk=5),
        device="cpu",
        fixed_shifts_per_line=shifts,
        fixed_axis_deg_per_line=axis,
    )
    assert result.poses is not None
    torch.testing.assert_close(result.poses.shifts_per_line, shifts)
    torch.testing.assert_close(result.poses.axis_deg_per_line, axis)


def test_pose_off_reconstruction_numeric_regression() -> None:
    _, target, geometry = _tiny_case()
    result = reconstruct(
        target,
        geometry,
        config=ADConfig(iters=6, lr=0.03, chunk=4, loss="mse", seed=7),
        device="cpu",
    )

    expected_losses = np.array(
        [
            2.9346958994865417,
            2.5204447055002674,
            2.153826452617068,
            1.8301988600287586,
            1.545607248088345,
            1.2957146591506898,
        ]
    )
    actual_losses = np.array([record.fidelity for record in result.loss_log])
    np.testing.assert_allclose(actual_losses, expected_losses, rtol=0.0, atol=1e-7)
    assert result.volume.sum().item() == pytest.approx(56.42479705810547, abs=1e-6)
    assert result.poses is None


def test_pose_recovery_beats_pose_off_fit() -> None:
    """With the known volume fixed, constant pose nodes recover absolutely."""
    size = 9
    n_frames = 4
    volume_zyx = torch.zeros(size, size, size)
    volume_zyx[2:6, 2:5, 5:7] = 1.0
    volume_zyx[5:7, 5:7, 2:4] = 0.45
    angles = (-42.0 + torch.arange(n_frames * size) * (84.0 / (n_frames * size - 1))).reshape(
        n_frames, size
    )
    geometry = PerLineGeometry.from_angles(
        angles, arm="fast-par", frame_shape=(n_frames, size, size)
    )
    truth_shift = torch.tensor([0.8, -0.6])
    truth_axis = 1.2
    shifts = truth_shift.expand(n_frames, size, 2).clone()
    axis = torch.full((n_frames, size), truth_axis)
    target = geometry.predict(
        generator_to_fasttomo_volume(volume_zyx),
        chunk_size=12,
        shifts_per_line=shifts,
        axis_deg_per_line=axis,
    )

    fixed_volume = generator_to_fasttomo_volume(volume_zyx)
    raw_shifts = torch.zeros(n_frames, 2, requires_grad=True)
    raw_axis = torch.zeros(n_frames, requires_grad=True)
    optimizer = torch.optim.Adam(
        (
            {"params": [raw_shifts], "lr": 0.08},
            {"params": [raw_axis], "lr": 0.08},
        )
    )
    zero_prediction = geometry.predict(
        fixed_volume,
        chunk_size=12,
        shifts_per_line=torch.zeros_like(shifts),
        axis_deg_per_line=torch.zeros_like(axis),
    )
    pose_off_loss = (zero_prediction - target).square().mean().item()
    final_loss = float("nan")
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        shifts_frames, axis_frames = _bounded_pose(raw_shifts, raw_axis)
        prediction = geometry.predict(
            fixed_volume,
            chunk_size=12,
            shifts_per_line=interpolate_pose_nodes(shifts_frames, size),
            axis_deg_per_line=interpolate_pose_nodes(axis_frames, size),
        )
        loss = (prediction - target).square().mean()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    recovered_shifts, recovered_axes = _bounded_pose(raw_shifts, raw_axis)
    recovered_shift = recovered_shifts.mean(dim=0)
    recovered_axis = recovered_axes.mean()
    print(
        "recovery",
        recovered_shift.tolist(),
        recovered_axis.item(),
        final_loss,
        pose_off_loss,
    )
    torch.testing.assert_close(
        recovered_shifts,
        truth_shift.expand_as(recovered_shifts),
        atol=0.3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        recovered_axes,
        torch.full_like(recovered_axes, truth_axis),
        atol=0.35,
        rtol=0.0,
    )
    assert final_loss < pose_off_loss * 0.1


def test_anchored_reference_recovery_excludes_reference_from_timeline() -> None:
    """Ramp nodes recover a constant pose while the final reference stays zero."""
    size = 9
    n_ramp_frames = 4
    n_frames = n_ramp_frames + 1
    volume_zyx = torch.zeros(size, size, size)
    volume_zyx[2:6, 2:5, 5:7] = 1.0
    volume_zyx[5:7, 5:7, 2:4] = 0.45
    ramp_angles = torch.linspace(
        -42.0, 42.0, n_ramp_frames * size
    ).reshape(n_ramp_frames, size)
    angles = torch.cat((ramp_angles, torch.zeros(1, size)), dim=0)
    geometry = PerLineGeometry.from_angles(
        angles, arm="fast-par", frame_shape=(n_frames, size, size)
    )
    truth_shift = torch.tensor([0.8, -0.6])
    truth_axis = 1.2
    truth_shifts = torch.zeros(n_frames, size, 2)
    truth_shifts[:-1] = truth_shift
    truth_axes = torch.zeros(n_frames, size)
    truth_axes[:-1] = truth_axis
    fixed_volume = generator_to_fasttomo_volume(volume_zyx)
    target = geometry.predict(
        fixed_volume,
        chunk_size=12,
        shifts_per_line=truth_shifts,
        axis_deg_per_line=truth_axes,
    )

    raw_shifts = torch.zeros(n_ramp_frames, 2, requires_grad=True)
    raw_axis = torch.zeros(n_ramp_frames, requires_grad=True)
    optimizer = torch.optim.Adam(
        (
            {"params": [raw_shifts], "lr": 0.08},
            {"params": [raw_axis], "lr": 0.08},
        )
    )
    zero_shifts = torch.zeros_like(truth_shifts)
    zero_axes = torch.zeros_like(truth_axes)
    pose_off_loss = (
        geometry.predict(
            fixed_volume,
            chunk_size=12,
            shifts_per_line=zero_shifts,
            axis_deg_per_line=zero_axes,
        )
        - target
    ).square().mean().item()
    final_loss = float("nan")
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        _, _, shifts_per_line, axis_per_line = _expand_learned_pose(
            raw_shifts,
            raw_axis,
            n_scan_lines=size,
            anchor_reference=True,
        )
        prediction = geometry.predict(
            fixed_volume,
            chunk_size=12,
            shifts_per_line=shifts_per_line,
            axis_deg_per_line=axis_per_line,
        )
        loss = (prediction - target).square().mean()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    shifts_frames, axis_frames, shifts_per_line, axis_per_line = (
        _expand_learned_pose(
            raw_shifts,
            raw_axis,
            n_scan_lines=size,
            anchor_reference=True,
        )
    )
    torch.testing.assert_close(
        shifts_frames[:-1],
        truth_shift.expand_as(shifts_frames[:-1]),
        atol=0.3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        axis_frames[:-1],
        torch.full_like(axis_frames[:-1], truth_axis),
        atol=0.35,
        rtol=0.0,
    )
    assert torch.count_nonzero(shifts_frames[-1]).item() == 0
    assert axis_frames[-1].item() == 0.0
    assert torch.count_nonzero(shifts_per_line[-1]).item() == 0
    assert torch.count_nonzero(axis_per_line[-1]).item() == 0
    assert final_loss < pose_off_loss * 0.1
    print(
        "anchored recovery:",
        f"ramp_shift_max_error={(shifts_frames[:-1] - truth_shift).abs().max().item():.6f}",
        f"ramp_phi_max_error={(axis_frames[:-1] - truth_axis).abs().max().item():.6f}",
        f"final_loss={final_loss:.8e}",
        f"pose_off_loss={pose_off_loss:.8e}",
        "reference_pose=(0,0,0)",
    )

    initial_shifts = torch.full((n_frames, 2), 9.0)
    initial_shifts[:-1] = truth_shift
    initial_axis = torch.full((n_frames,), 2.0)
    initial_axis[:-1] = truth_axis
    frozen = reconstruct(
        target,
        geometry,
        config=ADConfig(
            iters=1,
            chunk=12,
            refine_pose=True,
            pose_start=1,
            anchor_reference=True,
        ),
        device="cpu",
        initial_shifts_frames=initial_shifts,
        initial_axis_deg_frames=initial_axis,
    )
    assert frozen.poses is not None
    assert frozen.poses.anchor_reference
    assert frozen.poses.anchor_frame_index == n_frames - 1
    assert torch.count_nonzero(frozen.poses.shifts_frames[-1]).item() == 0
    assert frozen.poses.axis_deg_frames[-1].item() == 0.0
    assert "excluded from the continuous scan interpolation timeline" in (
        frozen.poses.convention
    )


def test_anchor_cli_records_npz_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner artifacts make the anchored-frame contract machine-readable."""
    import sys

    import yaml

    from benchmarks.scripts.recon_continuous_adrecon import main

    size = 5
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    dataset.mkdir()
    tilt_series = np.zeros((3, size, size), dtype=np.float32)
    ramp_angles = np.linspace(-10.0, 10.0, 2 * size, dtype=np.float32).reshape(
        2, size
    )
    ramp_angles[0, 2] = 0.0
    angles_per_row = np.concatenate(
        (ramp_angles, np.zeros((1, size), dtype=np.float32)), axis=0
    )
    valid_lines = np.ones((3, size), dtype=np.bool_)
    valid_lines[0, 2] = False
    np.save(dataset / "tilt_series.npy", tilt_series)
    np.save(dataset / "angles.npy", np.asarray([-5.0, 5.0, 0.0], dtype=np.float32))
    np.save(dataset / "angles_per_row.npy", angles_per_row)
    np.save(dataset / "valid_lines.npy", valid_lines)
    with (dataset / "gen_manifest.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump({"scan_orientation": "fast-par"}, stream)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recon_continuous_adrecon.py",
            "--dataset",
            str(dataset),
            "--out",
            str(output),
            "--iters",
            "1",
            "--chunk",
            "6",
            "--device",
            "cpu",
            "--refine-pose",
            "--anchor-reference",
        ],
    )
    main()

    with np.load(output / "poses_learned.npz", allow_pickle=False) as poses:
        assert bool(poses["anchor_reference"])
        assert int(poses["anchor_frame_index"]) == 2
        np.testing.assert_array_equal(poses["shifts_frames"][-1], 0.0)
        np.testing.assert_array_equal(poses["axis_deg_frames"][-1], 0.0)
        assert "excluded from the continuous scan interpolation timeline" in str(
            poses["convention"]
        )
    with (output / "run_manifest.yaml").open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    assert manifest["config"]["anchor_reference"] is True
    assert manifest["config"]["pose_mode"].endswith(
        "anchored-final-reference"
    )
    assert "valid_lines.npy" in manifest["input_sha256"]
    assert manifest["line_selection"] == {
        "valid_lines_file": "valid_lines.npy",
        "total_lines": 15,
        "valid_lines": 14,
        "excluded_lines": 1,
        "normalization": "mean over valid measured pixels only",
    }
    pose_output = manifest["outputs"]["poses_learned.npz"]
    assert pose_output["anchor_reference"] is True
    assert pose_output["anchor_frame_index"] == 2
