"""CPU tests for continuous-rotation SIRT geometry and baselines."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import fasttomo as ft
from benchmarks.scripts.utils.perrow_ad import (
    ContinuousDataset,
    fasttomo_to_generator_volume,
    generator_to_fasttomo_volume,
)
from benchmarks.scripts.utils.perrow_sirt import sirt_continuous
from tests.test_perrow_ad import assemble_reference


def synthetic_dataset(
    tilt_series: torch.Tensor,
    frame_angles: torch.Tensor,
    per_line_angles: torch.Tensor,
    arm: str,
) -> ContinuousDataset:
    """Build an in-memory dataset after projections are independently generated."""
    line_name = "angles_per_col.npy" if arm == "fast-perp" else "angles_per_row.npy"
    return ContinuousDataset(
        path=Path("/synthetic"),
        tilt_series=tilt_series.numpy(),
        angles=frame_angles.numpy(),
        angles_per_line=per_line_angles.numpy(),
        valid_lines=torch.ones_like(per_line_angles, dtype=torch.bool).numpy(),
        arm=arm,  # type: ignore[arg-type]
        line_angle_path=Path("/synthetic") / line_name,
        valid_lines_path=None,
        generator_manifest={"scan_orientation": arm},
        input_paths={},
    )


def test_perp_exact_reconstruction_correlates_and_residual_decreases() -> None:
    """Exact per-column generator data is reconstructable slice by slice."""
    size = 11
    volume_zyx = torch.zeros(size, size, size)
    volume_zyx[2:8, 3:7, 2:6] = 1.0
    volume_zyx[5:9, 6:9, 7:9] = 0.55
    volume_zyx[3:5, 2:4, 8:9] = 1.4
    volume_zyx[1:2, 1:2, :] = 2.0
    volume_zyx[9:10, 2:3, :] = 1.6
    volume_zyx[3:4, 9:10, :] = 2.4

    frame_angles = torch.linspace(-72.0, 72.0, 15)
    offsets = torch.linspace(-7.5, 7.5, size)
    per_col_angles = frame_angles[:, None] + offsets[None, :]
    target = assemble_reference(volume_zyx, per_col_angles, "fast-perp")
    dataset = synthetic_dataset(
        target, frame_angles, per_col_angles, "fast-perp"
    )

    exact = sirt_continuous(
        dataset,
        "perp-exact",
        n_iter=30,
        omega=1.0,
        device="cpu",
        seed=3,
    )
    mid = sirt_continuous(
        dataset,
        "midangle",
        n_iter=30,
        omega=1.0,
        device="cpu",
        seed=3,
    )

    exact_reconstruction = fasttomo_to_generator_volume(exact.volume)
    corr_exact = torch.corrcoef(
        torch.stack((exact_reconstruction.flatten(), volume_zyx.flatten()))
    )[0, 1]
    mid_reconstruction = fasttomo_to_generator_volume(mid.volume)
    corr_mid = torch.corrcoef(
        torch.stack((mid_reconstruction.flatten(), volume_zyx.flatten()))
    )[0, 1]
    assert float(corr_exact) > 0.7
    assert float(corr_exact) > float(corr_mid) + 0.05
    assert exact.loss_history[-1] < exact.loss_history[0]
    assert torch.isfinite(exact.loss_history).all()


def test_perp_slice_sirt_equals_full_volume_for_constant_angles() -> None:
    """Slab extraction and assembly preserve ordinary SIRT exactly."""
    size = 9
    volume_zyx = torch.zeros(size, size, size)
    volume_zyx[2:7, 2:6, 3:7] = 1.0
    volume_zyx[4:6, 6:8, 1:3] = 0.4
    frame_angles = torch.linspace(-68.0, 68.0, 11)
    fasttomo_volume = generator_to_fasttomo_volume(volume_zyx)
    fasttomo_sinogram = ft.forward_project(fasttomo_volume, frame_angles)
    target = fasttomo_sinogram.permute(0, 2, 1).contiguous()
    per_col_angles = frame_angles[:, None].expand(-1, size).contiguous()
    dataset = synthetic_dataset(
        target, frame_angles, per_col_angles, "fast-perp"
    )

    exact = sirt_continuous(
        dataset, "perp-exact", n_iter=8, omega=0.85, device="cpu"
    )
    full = sirt_continuous(
        dataset, "midangle", n_iter=8, omega=0.85, device="cpu"
    )

    torch.testing.assert_close(exact.volume, full.volume, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        exact.loss_history, full.loss_history, rtol=1e-6, atol=1e-7
    )


def test_midangle_runs_on_tiny_fast_par_dataset() -> None:
    """The intentionally approximate fast-par baseline remains well-formed."""
    size = 7
    volume_zyx = torch.zeros(size, size, size)
    volume_zyx[2:6, 1:5, 2:5] = 1.0
    volume_zyx[1:3, 4:6, 4:6] = 0.35
    frame_angles = torch.tensor([-55.0, -17.0, 21.0, 59.0])
    row_offsets = torch.linspace(-8.0, 8.0, size)
    per_row_angles = frame_angles[:, None] + row_offsets[None, :]
    target = assemble_reference(volume_zyx, per_row_angles, "fast-par")
    dataset = synthetic_dataset(target, frame_angles, per_row_angles, "fast-par")

    result = sirt_continuous(
        dataset, "midangle", n_iter=4, omega=0.8, device="cpu", seed=5
    )

    assert result.volume.shape == (size, size, size)
    assert result.loss_history.shape == (4,)
    assert torch.isfinite(result.volume).all()
    assert torch.isfinite(result.loss_history).all()


def test_perp_exact_rejects_fast_par_geometry() -> None:
    target = torch.zeros(2, 5, 5)
    angles = torch.tensor([-10.0, 10.0])
    per_row = angles[:, None].expand(-1, 5).contiguous()
    dataset = synthetic_dataset(target, angles, per_row, "fast-par")

    with pytest.raises(ValueError, match="cannot be represented"):
        sirt_continuous(dataset, "perp-exact", n_iter=1, device="cpu")
