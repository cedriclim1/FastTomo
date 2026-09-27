"""CPU tests for the static conventional-reconstruction sampling harness."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.scripts import recon_sampling_conventional as conventional


@pytest.mark.parametrize("scheme", ["golden", "random", "symmetric"])
def test_angle_sequences_match_auto6_planners_byte_for_byte(scheme: str) -> None:
    actual = conventional.plan_angle_additions(
        scheme,  # type: ignore[arg-type]
        seed=42,
        acquired_additions=20,
    )
    expected = conventional.auto6.planned_angles(
        conventional.auto6.AcquisitionConfig(
            scheme=scheme,
            initial_tilts=conventional.INITIAL_TILTS.tolist(),
            additions=21,
            seed=42,
        )
    )

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert actual.tobytes() == np.asarray(expected, dtype=np.float32).tobytes()


def test_symmetric_acquired_prefix_matches_real_auto6_total() -> None:
    actual = conventional.plan_angle_additions(
        "symmetric",
        seed=42,
        acquired_additions=20,
    )
    expected = conventional.auto6.planned_angles(
        conventional.auto6.AcquisitionConfig(
            scheme="symmetric",
            initial_tilts=conventional.INITIAL_TILTS.tolist(),
            additions=20,
            seed=42,
        )
    )

    np.testing.assert_array_equal(actual[:20], expected)
    assert actual[:20].tobytes() == expected.tobytes()


def test_cv_projection_max_normalized_mse_is_hand_checkable() -> None:
    reconstruction = np.zeros((2, 2, 2), dtype=np.float32)
    reconstruction[:, 0, 0] = 1.0
    measured = np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)

    # At zero degrees the prediction is [[2,0],[0,0]]. Max normalization
    # leaves one of four pixels different by one, hence MSE = 1/4.
    mse = conventional.cv_next_mse(
        reconstruction,
        measured,
        0.0,
        device="cpu",
        apply_shift=False,
    )

    assert mse == pytest.approx(0.25)


def test_phantom_cv_prediction_and_measurement_share_legacy_shift() -> None:
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[1:6, 2:5, 3:7] = 1.0
    volume[5:8, 6:8, 1:3] = 0.35
    dataset = conventional.StudyDataset(
        name="phantom",
        volume=volume,
        apply_legacy_shift=True,
    )
    angle = 17.5
    measured = conventional.synthesize_projection(
        dataset.volume,
        angle,
        device="cpu",
        apply_shift=dataset.apply_legacy_shift,
    )

    matched = conventional.cv_next_mse(
        dataset.volume,
        measured,
        angle,
        device="cpu",
        apply_shift=dataset.apply_legacy_shift,
    )
    mismatched = conventional.cv_next_mse(
        dataset.volume,
        measured,
        angle,
        device="cpu",
        apply_shift=False,
    )

    assert matched == pytest.approx(0.0, abs=0.0)
    assert mismatched > 0.0


def test_auto12_rot_zxz_matches_default_checkout_for_scalar_angle() -> None:
    default_utils_path = Path(
        "/home/cedlim/quantem/quantem/src/quantem/tomography/utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "auto18_default_quantem_tomography_utils",
        default_utils_path,
    )
    assert spec is not None and spec.loader is not None
    default_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(default_utils)

    volume = torch.zeros((1, 7, 8, 9), dtype=torch.float32)
    volume[0, 1:5, 2:7, 3:8] = (
        torch.arange(4 * 5 * 5, dtype=torch.float32).reshape(4, 5, 5) / 100.0
    )
    volume[0, 5, 1, 7] = 2.3
    auto12 = conventional.auto6.rot_ZXZ(
        volume.clone(), z1=0, x=23.75, z3=0, device="cpu"
    )
    default = default_utils.rot_ZXZ(volume.clone(), z1=0, x=23.75, z3=0, device="cpu")

    torch.testing.assert_close(auto12, default, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("method", ["sirt", "ad"])
def test_reconstructors_are_fresh_across_growing_tilt_counts(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = np.zeros((7, 7, 7), dtype=np.float32)
    volume[1:5, 2:6, 2:5] = 1.0
    volume[4:6, 1:3, 5:7] = 0.4
    angles = np.asarray([-35.0, 0.0, 35.0], dtype=np.float32)
    series = conventional.synthesize_tilt_series(
        volume,
        angles,
        device="cpu",
        apply_shift=False,
    )
    if method == "sirt":
        monkeypatch.setattr(conventional, "SIRT_ITERS", 1)
        reconstructor = conventional.reconstruct_sirt
    else:
        monkeypatch.setattr(
            conventional,
            "AD_CONFIG",
            replace(conventional.AD_CONFIG, iters=2, chunk=4),
        )
        reconstructor = conventional.reconstruct_ad

    smaller = reconstructor(series[:2], angles[:2], device="cpu")
    smaller_snapshot = smaller.copy()
    grown_after_smaller = reconstructor(series, angles, device="cpu")
    grown_fresh_reference = reconstructor(series, angles, device="cpu")

    np.testing.assert_array_equal(smaller, smaller_snapshot)
    torch.testing.assert_close(
        torch.from_numpy(grown_after_smaller),
        torch.from_numpy(grown_fresh_reference),
        rtol=0.0,
        atol=0.0,
    )


def _complete_row(tilt_count: int) -> dict[str, object]:
    return {
        "tilt_count": tilt_count,
        "status": "complete",
        "angles_deg": [-70.0, -35.0, 0.0, 35.0, 70.0],
        "cv_next_angle_deg": 16.5,
        "cv_next_mse": 0.125,
        "pearson_global": 0.8,
        "pearson_fg": 0.9,
        "ssim": 0.7,
        "f1": None,
        "precision": None,
        "recall": None,
        "n_traced": None,
        "runtime_seconds": 1.5,
    }


def test_metrics_json_schema_marks_missing_tilt_frames(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    conventional.write_metrics_json(
        path,
        config={"dataset": "phantom"},
        completed_rows=[_complete_row(5), _complete_row(7)],
        min_tilts=5,
        max_tilts=7,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fasttomo conventional sampling metrics v1"
    assert payload["config"] == {"dataset": "phantom"}
    assert [row["tilt_count"] for row in payload["rows"]] == [5, 6, 7]
    assert [row["status"] for row in payload["rows"]] == [
        "complete",
        "missing",
        "complete",
    ]
    missing = payload["rows"][1]
    assert set(missing) == set(conventional.METRIC_FIELDS)
    assert all(
        value is None
        for key, value in missing.items()
        if key not in {"tilt_count", "status"}
    )
