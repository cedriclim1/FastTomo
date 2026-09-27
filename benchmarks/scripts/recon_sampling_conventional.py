#!/usr/bin/env python3
"""Fresh-from-scratch SIRT/AD sampling study on static single-angle tilts."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

import fasttomo as ft

try:
    from .utils.perrow_ad import (
        ADConfig,
        PerLineGeometry,
        fasttomo_to_generator_volume,
        generator_to_fasttomo_sinogram,
        reconstruct,
    )
except ImportError:  # Direct execution.
    from utils.perrow_ad import (  # type: ignore[no-redef]
        ADConfig,
        PerLineGeometry,
        fasttomo_to_generator_volume,
        generator_to_fasttomo_sinogram,
        reconstruct,
    )


AUTOMATED_TOMO_ROOT = Path("/home/cedlim/quantem/mm_2026/automated_tomography")
AUTO6_SCRIPTS = AUTOMATED_TOMO_ROOT / "scripts"
QUANTEM_AUTO12_SRC = Path("/home/cedlim/quantem/worktrees/auto-et-continuous/src")
PHANTOM_PATH = Path("/home/cedlim/quantem/quantem-tutorials/data/phantom.npy")
ATOMIC_PATH = Path("/home/cedlim/3dgstem_top/data/sim_np_gt/au_np_potential_gt.npy")
ATOMIC_COORDS_PATH = Path("/home/cedlim/3dgstem_top/data/sim_np_gt/gt_coords_vox.npy")
TRACER_CONFIG_PATH = AUTOMATED_TOMO_ROOT / "data/continuous/atomtrace_frozen_v3.json"
INITIAL_TILTS = np.asarray([-70.0, -35.0, 0.0, 35.0, 70.0], dtype=np.float32)
MAX_STUDY_ADDITIONS = 20
SELF_CHECK_ANGLES = np.linspace(-70.0, 70.0, 25, dtype=np.float32)
SELF_CHECK_THRESHOLDS = {"sirt": 0.7, "ad": 0.9}
FG_THRESHOLD = 0.0
METRIC_FIELDS = (
    "tilt_count",
    "status",
    "angles_deg",
    "cv_next_angle_deg",
    "cv_next_mse",
    "pearson_global",
    "pearson_fg",
    "ssim",
    "f1",
    "precision",
    "recall",
    "n_traced",
    "runtime_seconds",
)
AD_CONFIG = ADConfig(
    iters=2000,
    lr=0.05,
    tv=3e-2,
    sparsity=1e-3,
    chunk=256,
    loss="mse",
    seed=0,
)
SIRT_ITERS = 150
SIRT_OMEGA = 1.0

DatasetName = Literal["phantom", "atomic"]
SchemeName = Literal["golden", "random", "symmetric"]
MethodName = Literal["sirt", "ad"]


def _import_auto6_module(name: str) -> Any:
    # acquisition_loop imports the AUTO-12 dataset API, including
    # DeviceBatchSampler, which is not present in the older default checkout.
    for source in (AUTO6_SCRIPTS, QUANTEM_AUTO12_SRC):
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-auto18")
    return importlib.import_module(name)


auto6 = _import_auto6_module("acquisition_loop")


@dataclass(frozen=True)
class StudyDataset:
    name: DatasetName
    volume: np.ndarray
    apply_legacy_shift: bool
    gt_coords: np.ndarray | None = None
    tracer_config: dict[str, Any] | None = None
    tracer_document: dict[str, Any] | None = None


def load_dataset(name: DatasetName) -> StudyDataset:
    """Load and validate one study GT plus atomic tracing inputs when needed."""
    path = PHANTOM_PATH if name == "phantom" else ATOMIC_PATH
    volume = np.load(path, allow_pickle=False)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"{path} must be a cubic 3-D volume, got {volume.shape}")
    if not np.issubdtype(volume.dtype, np.number) or not np.isfinite(volume).all():
        raise ValueError(f"{path} must be finite and numeric")
    volume = np.asarray(volume, dtype=np.float32)
    if name == "phantom":
        return StudyDataset(name=name, volume=volume, apply_legacy_shift=True)

    atomic_eval = _import_auto6_module("eval_auto6_atomic")
    coords = atomic_eval.load_coords(ATOMIC_COORDS_PATH, volume.shape)
    tracer_config, tracer_document = atomic_eval.load_tracer_config(TRACER_CONFIG_PATH)
    return StudyDataset(
        name=name,
        volume=volume,
        apply_legacy_shift=False,
        gt_coords=coords,
        tracer_config=tracer_config,
        tracer_document=tracer_document,
    )


def plan_angle_additions(
    scheme: SchemeName,
    *,
    seed: int,
    acquired_additions: int,
) -> np.ndarray:
    """Return AUTO-6 additions plus the next held-out CV angle.

    The planner is called for one more angle than is acquired so the final
    25-tilt reconstruction still has a genuinely unacquired next-angle target.
    """
    if acquired_additions < 0 or acquired_additions > MAX_STUDY_ADDITIONS:
        raise ValueError(f"acquired additions must be in [0, {MAX_STUDY_ADDITIONS}]")
    config = auto6.AcquisitionConfig(
        scheme=scheme,
        initial_tilts=INITIAL_TILTS.tolist(),
        additions=acquired_additions + 1,
        seed=seed,
    )
    planned = auto6.planned_angles(config)
    return np.asarray(planned, dtype=np.float32)


def full_angle_schedule(
    scheme: SchemeName,
    *,
    seed: int,
    max_tilts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return acquired schedule and additions including the next CV angle."""
    additions = plan_angle_additions(
        scheme,
        seed=seed,
        acquired_additions=max_tilts - len(INITIAL_TILTS),
    )
    schedule = np.concatenate((INITIAL_TILTS, additions))
    return schedule[:max_tilts], additions


def synthesize_projection(
    volume_zyx: np.ndarray,
    angle_deg: float,
    *,
    device: str,
    apply_shift: bool,
) -> np.ndarray:
    """Use the exact AUTO-6 ``rot_ZXZ(0, theta, 0); sum(Z)`` path."""
    return auto6.synthesize_projection(
        volume_zyx,
        float(angle_deg),
        device,
        apply_shift=apply_shift,
    )


def synthesize_tilt_series(
    volume_zyx: np.ndarray,
    angles_deg: np.ndarray,
    *,
    device: str,
    apply_shift: bool,
) -> np.ndarray:
    return auto6.synthesize_tilt_series(
        volume_zyx,
        angles_deg,
        device,
        apply_shift=apply_shift,
    )


def max_normalize_projection(projection: np.ndarray) -> np.ndarray:
    projection = np.asarray(projection, dtype=np.float32)
    scale = float(np.max(np.abs(projection)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("cannot max-normalize a nonfinite or all-zero projection")
    return projection / scale


def cv_next_mse(
    reconstruction_zyx: np.ndarray,
    measured_projection_yx: np.ndarray,
    next_angle_deg: float,
    *,
    device: str,
    apply_shift: bool,
) -> float:
    """Project at an unacquired angle using the measurement's shift convention."""
    predicted = synthesize_projection(
        reconstruction_zyx,
        next_angle_deg,
        device=device,
        apply_shift=apply_shift,
    )
    measured = np.asarray(measured_projection_yx, dtype=np.float32)
    if predicted.shape != measured.shape:
        raise ValueError(
            f"predicted and measured projection shapes differ: "
            f"{predicted.shape} != {measured.shape}"
        )
    difference = max_normalize_projection(predicted) - max_normalize_projection(
        measured
    )
    return float(np.mean(np.square(difference, dtype=np.float32), dtype=np.float64))


def reconstruct_sirt(
    tilt_series_kyx: np.ndarray,
    angles_deg: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Run ordinary scalar-angle SIRT from its implicit zero initialization."""
    resolved_device = torch.device(device)
    frames = torch.as_tensor(
        tilt_series_kyx, dtype=torch.float32, device=resolved_device
    )
    sinogram = generator_to_fasttomo_sinogram(frames)
    angles = torch.as_tensor(angles_deg, dtype=torch.float32, device=resolved_device)
    volume = ft.run_sirt(
        sinogram,
        angles,
        n_iter=SIRT_ITERS,
        omega=SIRT_OMEGA,
        Nz=tilt_series_kyx.shape[1],
        positivity=True,
    )
    return (
        fasttomo_to_generator_volume(volume)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def reconstruct_ad(
    tilt_series_kyx: np.ndarray,
    angles_deg: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Run frozen AD from zero with a static degenerate per-line geometry."""
    n_frames, image_y, _ = tilt_series_kyx.shape
    per_line_angles = np.broadcast_to(
        np.asarray(angles_deg, dtype=np.float32)[:, None],
        (n_frames, image_y),
    ).copy()
    geometry = PerLineGeometry.from_angles(
        per_line_angles,
        arm="fast-par",
        frame_shape=tuple(int(size) for size in tilt_series_kyx.shape),
    )
    result = reconstruct(
        tilt_series_kyx,
        geometry,
        config=AD_CONFIG,
        device=device,
    )
    return (
        fasttomo_to_generator_volume(result.volume)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def reconstruct_volume(
    method: MethodName,
    tilt_series_kyx: np.ndarray,
    angles_deg: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    if method == "sirt":
        return reconstruct_sirt(tilt_series_kyx, angles_deg, device=device)
    if method == "ad":
        return reconstruct_ad(tilt_series_kyx, angles_deg, device=device)
    raise ValueError(f"unknown method {method!r}")


def pearson_metrics(
    reconstruction: np.ndarray, gt_volume: np.ndarray
) -> tuple[float, float]:
    atomic_eval = _import_auto6_module("eval_auto6_atomic")
    full = np.ones(gt_volume.shape, dtype=bool)
    foreground = gt_volume > FG_THRESHOLD
    return (
        atomic_eval.pearson(gt_volume, reconstruction, full),
        atomic_eval.pearson(gt_volume, reconstruction, foreground),
    )


def quality_metrics(
    reconstruction: np.ndarray,
    dataset: StudyDataset,
) -> dict[str, float | int | None]:
    if reconstruction.shape != dataset.volume.shape:
        raise ValueError(
            "reconstruction and GT shapes differ: "
            f"{reconstruction.shape} != {dataset.volume.shape}"
        )
    pearson_global, pearson_fg = pearson_metrics(reconstruction, dataset.volume)
    if dataset.name == "phantom":
        return {
            "pearson_global": pearson_global,
            "pearson_fg": pearson_fg,
            "ssim": float(auto6.compute_3d_ssim(reconstruction, dataset.volume)),
            "f1": None,
            "precision": None,
            "recall": None,
            "n_traced": None,
        }

    assert dataset.gt_coords is not None
    assert dataset.tracer_config is not None
    assert dataset.tracer_document is not None
    atomic_eval = _import_auto6_module("eval_auto6_atomic")
    traced = atomic_eval.evaluate_volume(
        reconstruction,
        dataset.volume,
        dataset.gt_coords,
        dataset.tracer_config,
        dataset.tracer_document,
    )
    return {
        "pearson_global": float(traced["pearson_global"]),
        "pearson_fg": float(traced["pearson_fg"]),
        "ssim": None,
        "f1": float(traced["f1"]),
        "precision": float(traced["precision"]),
        "recall": float(traced["recall"]),
        "n_traced": int(traced["n_traced"]),
    }


def _missing_metric_row(tilt_count: int) -> dict[str, Any]:
    return {
        "tilt_count": tilt_count,
        "status": "missing",
        **{
            field: None
            for field in METRIC_FIELDS
            if field not in {"tilt_count", "status"}
        },
    }


def build_metrics_payload(
    *,
    config: dict[str, Any],
    completed_rows: list[dict[str, Any]],
    min_tilts: int,
    max_tilts: int,
) -> dict[str, Any]:
    """Build a stable schema and retain explicit placeholders for missing frames."""
    completed: dict[int, dict[str, Any]] = {}
    for row in completed_rows:
        count = row.get("tilt_count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError("each completed metric row needs an integer tilt_count")
        if count in completed:
            raise ValueError(f"duplicate metric row for tilt_count={count}")
        missing = set(METRIC_FIELDS) - set(row)
        if missing:
            raise ValueError(f"metric row {count} is missing fields {sorted(missing)}")
        if row["status"] != "complete":
            raise ValueError("completed metric rows must have status='complete'")
        completed[count] = row

    rows = [
        completed.get(count, _missing_metric_row(count))
        for count in range(min_tilts, max_tilts + 1)
    ]
    return {
        "schema": "fasttomo conventional sampling metrics v1",
        "config": config,
        "rows": rows,
    }


def write_metrics_json(
    path: Path,
    *,
    config: dict[str, Any],
    completed_rows: list[dict[str, Any]],
    min_tilts: int,
    max_tilts: int,
) -> None:
    payload = build_metrics_payload(
        config=config,
        completed_rows=completed_rows,
        min_tilts=min_tilts,
        max_tilts=max_tilts,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def axis_sum_argmax_comparison(
    reconstruction: np.ndarray, gt_volume: np.ndarray
) -> dict[str, list[int]]:
    def positions(volume: np.ndarray) -> list[int]:
        return [
            int(
                np.argmax(
                    volume.sum(axis=tuple(other for other in range(3) if other != axis))
                )
            )
            for axis in range(3)
        ]

    return {
        "axis_order": ["z", "y", "x"],
        "gt": positions(gt_volume),
        "reconstruction": positions(reconstruction),
    }


def run_self_check(method: MethodName, *, device: str) -> dict[str, Any]:
    dataset = load_dataset("phantom")
    clean = synthesize_tilt_series(
        dataset.volume,
        SELF_CHECK_ANGLES,
        device=device,
        apply_shift=False,
    )
    started = time.perf_counter()
    reconstruction = reconstruct_volume(
        method,
        clean,
        SELF_CHECK_ANGLES,
        device=device,
    )
    runtime = time.perf_counter() - started
    pearson_global, pearson_fg = pearson_metrics(reconstruction, dataset.volume)
    threshold = SELF_CHECK_THRESHOLDS[method]
    result = {
        "method": method,
        "angles_deg": SELF_CHECK_ANGLES.tolist(),
        "pearson_global": pearson_global,
        "pearson_fg": pearson_fg,
        "threshold": threshold,
        "runtime_seconds": runtime,
        "passed": pearson_global >= threshold,
    }
    if not result["passed"]:
        comparison = axis_sum_argmax_comparison(reconstruction, dataset.volume)
        raise RuntimeError(
            f"absolute-frame self-check failed for {method}: "
            f"Pearson={pearson_global:.12g} < {threshold}; "
            f"axis-sum-argmax comparison={comparison}"
        )
    return result


def validate_study_range(min_tilts: int, max_tilts: int) -> None:
    initial_count = len(INITIAL_TILTS)
    maximum = initial_count + MAX_STUDY_ADDITIONS
    if min_tilts < initial_count or min_tilts > max_tilts or max_tilts > maximum:
        raise ValueError(
            f"require {initial_count} <= min_tilts <= max_tilts <= {maximum}"
        )


def run_study(args: argparse.Namespace) -> None:
    validate_study_range(args.min_tilts, args.max_tilts)
    dataset = load_dataset(args.dataset)
    acquired_schedule, additions = full_angle_schedule(
        args.scheme,
        seed=args.seed,
        max_tilts=args.max_tilts,
    )
    # Include the held-out angle after the final acquired tilt.
    synthesis_angles = np.concatenate(
        (INITIAL_TILTS, additions[: args.max_tilts - len(INITIAL_TILTS) + 1])
    )
    clean_series = synthesize_tilt_series(
        dataset.volume,
        synthesis_angles,
        device=args.device,
        apply_shift=dataset.apply_legacy_shift,
    )
    noisy_series = auto6.add_projection_noise(
        clean_series,
        args.noise_std,
        args.seed,
        entry_offset=0,
    )
    noisy_series = np.asarray(noisy_series, dtype=np.float32)

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": args.dataset,
        "scheme": args.scheme,
        "method": args.method,
        "noise_std": args.noise_std,
        "seed": args.seed,
        "min_tilts": args.min_tilts,
        "max_tilts": args.max_tilts,
        "device": args.device,
        "initial_tilts": INITIAL_TILTS.tolist(),
        "planned_additions_including_next_cv": additions.tolist(),
        "acquired_schedule": acquired_schedule.tolist(),
        "fresh_reconstruction_per_tilt_count": True,
        "foreground_threshold": FG_THRESHOLD,
        "sirt": {"iters": SIRT_ITERS, "omega": SIRT_OMEGA},
        "ad": asdict(AD_CONFIG),
        "projection_synthesis": (
            "AUTO-6 rot_ZXZ(z1=0,x=theta,z3=0), sum Z; "
            f"legacy_shift={dataset.apply_legacy_shift}"
        ),
    }
    completed_rows: list[dict[str, Any]] = []
    metrics_path = out / "metrics.json"
    write_metrics_json(
        metrics_path,
        config=config,
        completed_rows=completed_rows,
        min_tilts=args.min_tilts,
        max_tilts=args.max_tilts,
    )

    for tilt_count in range(args.min_tilts, args.max_tilts + 1):
        angles = synthesis_angles[:tilt_count]
        measurements = noisy_series[:tilt_count]
        next_angle = float(synthesis_angles[tilt_count])
        started = time.perf_counter()
        # Both called reconstructors allocate a new all-zero volume here.
        reconstruction = reconstruct_volume(
            args.method,
            measurements,
            angles,
            device=args.device,
        )
        np.save(
            out / f"vol_{tilt_count}.npy",
            reconstruction.astype(np.float32, copy=False),
        )
        row: dict[str, Any] = {
            "tilt_count": tilt_count,
            "status": "complete",
            "angles_deg": angles.tolist(),
            "cv_next_angle_deg": next_angle,
            "cv_next_mse": cv_next_mse(
                reconstruction,
                noisy_series[tilt_count],
                next_angle,
                device=args.device,
                apply_shift=dataset.apply_legacy_shift,
            ),
            **quality_metrics(reconstruction, dataset),
            "runtime_seconds": time.perf_counter() - started,
        }
        completed_rows.append(row)
        write_metrics_json(
            metrics_path,
            config=config,
            completed_rows=completed_rows,
            min_tilts=args.min_tilts,
            max_tilts=args.max_tilts,
        )
        print(
            f"tilts={tilt_count} cv={row['cv_next_mse']:.8g} "
            f"pearson={row['pearson_global']:.8g} "
            f"runtime={row['runtime_seconds']:.2f}s",
            flush=True,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("phantom", "atomic"), required=True)
    parser.add_argument(
        "--scheme",
        choices=("golden", "random", "symmetric"),
        default="golden",
    )
    parser.add_argument("--method", choices=("sirt", "ad"), required=True)
    parser.add_argument("--noise-std", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-tilts", type=int, default=5)
    parser.add_argument("--max-tilts", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="reconstruct 25 clean phantom tilts and enforce the g22 frame gate",
    )
    args = parser.parse_args(argv)
    if args.noise_std is not None and (
        not math.isfinite(args.noise_std) or args.noise_std < 0.0
    ):
        parser.error("--noise-std must be nonnegative and finite")
    if not args.self_check:
        if args.out is None:
            parser.error("--out is required unless --self-check is used")
        if args.noise_std is None:
            parser.error("--noise-std is required unless --self-check is used")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_check:
        print(json.dumps(run_self_check(args.method, device=args.device), indent=2))
        return
    run_study(args)


if __name__ == "__main__":
    main()
