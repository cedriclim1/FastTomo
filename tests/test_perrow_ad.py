"""CPU correctness proof for per-line-exact continuous-rotation geometry."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from benchmarks.scripts.utils.perrow_ad import (
    ADConfig,
    PerLineGeometry,
    generator_to_fasttomo_volume,
    reconstruct,
)


def reference_generator_projections(
    volume_zyx: torch.Tensor, angles_deg: torch.Tensor
) -> torch.Tensor:
    """Replicate generator ``rot_ZXZ(0, theta, 0); sum(Z)`` independently.

    This is the generator's ``grid_sample`` convention, expressed directly;
    it does not call FastTomo or any helper from the harness under test.
    """
    angles = torch.as_tensor(angles_deg, dtype=torch.float32).reshape(-1)
    radians = torch.deg2rad(angles)
    cosines, sines = radians.cos(), radians.sin()
    zeros = torch.zeros_like(angles)
    matrices = torch.stack(
        (cosines, -sines, zeros, sines, cosines, zeros), dim=-1
    ).reshape(-1, 2, 3)

    # quantem's rotx treats generator X as channels and rotates each Z/Y plane.
    batch = volume_zyx[None].permute(0, 3, 1, 2).expand(len(angles), -1, -1, -1)
    grid = F.affine_grid(matrices, batch.shape, align_corners=False)
    rotated = F.grid_sample(
        batch, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return rotated.permute(0, 2, 3, 1).sum(dim=1)


def assemble_reference(
    volume_zyx: torch.Tensor, angles_per_line: torch.Tensor, arm: str
) -> torch.Tensor:
    n_frames, n_lines = angles_per_line.shape
    all_projections = reference_generator_projections(
        volume_zyx, angles_per_line.reshape(-1)
    )
    line_indices = torch.arange(n_lines).repeat(n_frames)
    if arm == "fast-par":
        return all_projections[torch.arange(n_frames * n_lines), line_indices].reshape(
            n_frames, n_lines, volume_zyx.shape[2]
        )
    columns = all_projections[
        torch.arange(n_frames * n_lines), :, line_indices
    ]
    return columns.reshape(n_frames, n_lines, volume_zyx.shape[1]).permute(0, 2, 1)


@pytest.mark.parametrize("arm", ["fast-par", "fast-perp"])
@pytest.mark.parametrize("dtheta", [7.5, 21.0])
@pytest.mark.parametrize("size", [21, 20])
def test_forward_matches_absolute_generator_frame(
    arm: str, dtheta: float, size: int
) -> None:
    """A wrong permutation, flip, or angle sign fails against this reference."""
    n_frames = 3
    generator = torch.Generator().manual_seed(1234)
    volume_zyx = torch.zeros(size, size, size)
    volume_zyx[4:-4, 4:-4, 4:-4] = torch.rand(
        size - 8, size - 8, size - 8, generator=generator
    )
    # Deliberately asymmetric markers make flips and transposes conspicuous.
    volume_zyx[6:9, 5:7, 12:15] += 2.0
    volume_zyx[11:14, 13:16, 7:9] += 0.7

    flat_angles = -52.0 + torch.arange(n_frames * size) * (dtheta / size)
    angles_per_line = flat_angles.reshape(n_frames, size)
    reference = assemble_reference(volume_zyx, angles_per_line, arm)
    geometry = PerLineGeometry.from_angles(
        angles_per_line,
        arm=arm,  # type: ignore[arg-type]
        frame_shape=tuple(reference.shape),
    )
    actual = geometry.predict(
        generator_to_fasttomo_volume(volume_zyx), chunk_size=5
    )

    if size % 2 == 0:
        relative_max_error = (actual - reference).abs().max() / reference.abs().max()
        assert 0.005 < relative_max_error < 0.06
    else:
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("arm", ["fast-par", "fast-perp"])
def test_forward_matches_non_square_generator_frame(arm: str) -> None:
    """Non-square axes make an image-Y/image-X transposition observable."""
    shape = (15, 15, 11)
    n_frames = 3
    generator = torch.Generator().manual_seed(5678)
    volume_zyx = torch.zeros(shape)
    volume_zyx[3:-3, 3:-3, 3:-3] = torch.rand(
        shape[0] - 6, shape[1] - 6, shape[2] - 6, generator=generator
    )
    volume_zyx[4:7, 5:8, 6:8] += 1.5
    volume_zyx[7:10, 9:12, 3:5] += 0.6

    n_lines = shape[1] if arm == "fast-par" else shape[2]
    angles_per_line = (
        -47.0 + torch.arange(n_frames * n_lines) * (13.0 / n_lines)
    ).reshape(n_frames, n_lines)
    reference = assemble_reference(volume_zyx, angles_per_line, arm)
    geometry = PerLineGeometry.from_angles(
        angles_per_line,
        arm=arm,  # type: ignore[arg-type]
        frame_shape=tuple(reference.shape),
    )
    actual = geometry.predict(
        generator_to_fasttomo_volume(volume_zyx), chunk_size=5
    )

    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-3)


def test_tiny_cpu_reconstruction_decreases_loss() -> None:
    size = 11
    volume_zyx = torch.zeros(size, size, size)
    volume_zyx[3:8, 3:7, 4:8] = 1.0
    volume_zyx[5:7, 6:8, 2:4] = 0.4
    angles = (-38.0 + torch.arange(2 * size) * (19.0 / size)).reshape(2, size)
    target = assemble_reference(volume_zyx, angles, "fast-par")
    geometry = PerLineGeometry.from_angles(
        angles, arm="fast-par", frame_shape=tuple(target.shape)
    )

    result = reconstruct(
        target,
        geometry,
        config=ADConfig(iters=6, lr=0.03, chunk=4, loss="mse", seed=7),
        device="cpu",
    )

    losses = np.array([record.fidelity for record in result.loss_log])
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0] * 0.8
    assert torch.all(result.volume >= 0)


@pytest.mark.parametrize("arm", ["fast-par", "fast-perp"])
@pytest.mark.parametrize("pose_mode", ["off", "learned", "fixed"])
@pytest.mark.parametrize("loss_name", ["mse", "l1"])
def test_masked_line_residual_has_zero_loss_and_gradient_effect(
    arm: str, pose_mode: str, loss_name: str
) -> None:
    """Corrupting an invalid line changes neither loss nor optimized tensors."""
    size = 7
    n_frames = 2
    volume_zyx = torch.zeros(size, size, size)
    volume_zyx[2:5, 2:5, 3:6] = 1.0
    volume_zyx[4:6, 4:6, 1:3] = 0.4
    angles = torch.linspace(-28.0, 28.0, n_frames * size).reshape(n_frames, size)
    clean = assemble_reference(volume_zyx, angles, arm)
    geometry = PerLineGeometry.from_angles(
        angles,
        arm=arm,  # type: ignore[arg-type]
        frame_shape=tuple(clean.shape),
    )
    valid_lines = torch.ones(n_frames, size, dtype=torch.bool)
    invalid_frame, invalid_line = 0, 3
    valid_lines[invalid_frame, invalid_line] = False
    corrupted = clean.clone()
    if arm == "fast-par":
        corrupted[invalid_frame, invalid_line, :] += 1000.0
    else:
        corrupted[invalid_frame, :, invalid_line] += 1000.0

    config = ADConfig(
        iters=2,
        lr=0.02,
        chunk=5,
        seed=11,
        loss=loss_name,  # type: ignore[arg-type]
        refine_pose=pose_mode == "learned",
    )
    kwargs: dict[str, torch.Tensor] = {}
    if pose_mode == "fixed":
        kwargs = {
            "fixed_shifts_per_line": torch.zeros(n_frames, size, 2),
            "fixed_axis_deg_per_line": torch.zeros(n_frames, size),
        }
    baseline = reconstruct(
        clean,
        geometry,
        config=config,
        device="cpu",
        valid_lines=valid_lines,
        **kwargs,
    )
    perturbed = reconstruct(
        corrupted,
        geometry,
        config=config,
        device="cpu",
        valid_lines=valid_lines,
        **kwargs,
    )

    np.testing.assert_array_equal(
        [record.fidelity for record in baseline.loss_log],
        [record.fidelity for record in perturbed.loss_log],
    )
    valid_target = geometry.measurement_lines(clean)[valid_lines.reshape(-1)]
    expected_initial = (
        valid_target.square().mean()
        if loss_name == "mse"
        else valid_target.abs().mean()
    )
    assert baseline.loss_log[0].fidelity == pytest.approx(
        expected_initial.item(), rel=0.0, abs=1e-7
    )
    torch.testing.assert_close(baseline.volume, perturbed.volume, atol=0.0, rtol=0.0)
    if baseline.poses is not None:
        assert perturbed.poses is not None
        torch.testing.assert_close(
            baseline.poses.shifts_frames,
            perturbed.poses.shifts_frames,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            baseline.poses.axis_deg_frames,
            perturbed.poses.axis_deg_frames,
            atol=0.0,
            rtol=0.0,
        )
