"""Run per-line-exact AD reconstruction on a continuous-rotation dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

try:
    from .utils.perrow_ad import (
        ADConfig,
        PerLineGeometry,
        fasttomo_to_generator_volume,
        load_continuous_dataset,
        reconstruct,
    )
except ImportError:  # Direct execution: python benchmarks/scripts/recon_....py
    from utils.perrow_ad import (  # type: ignore[no-redef]
        ADConfig,
        PerLineGeometry,
        fasttomo_to_generator_volume,
        load_continuous_dataset,
        reconstruct,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tv", type=float, default=0.0)
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--loss", choices=("mse", "l1"), default="mse")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refine-pose", action="store_true")
    parser.add_argument("--pose-lr-shifts", type=float, default=1e-2)
    parser.add_argument("--pose-lr-angles", type=float, default=1e-2)
    parser.add_argument("--pose-start", type=int, default=0)
    parser.add_argument(
        "--anchor-reference",
        action="store_true",
        help=(
            "fix the appended final theta=0 reference frame at zero pose and "
            "exclude it from the learnable ramp interpolation timeline"
        ),
    )
    pose_source = parser.add_mutually_exclusive_group()
    pose_source.add_argument(
        "--gt-poses",
        action="store_true",
        help="use dataset GT per-line poses as a fixed volume-only oracle",
    )
    pose_source.add_argument(
        "--init-poses-from-gt",
        action="store_true",
        help="initialize learnable frame nodes from per-frame GT means",
    )
    return parser.parse_args()


def load_gt_poses(
    dataset: Path,
    *,
    n_frames: int,
    n_scan_lines: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Path]]:
    """Load and validate generator-convention per-line GT pose arrays."""
    paths = {
        "gt_shifts_per_line.npy": dataset / "gt_shifts_per_line.npy",
        "gt_tilt_axis_per_line.npy": dataset / "gt_tilt_axis_per_line.npy",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required GT pose inputs: {', '.join(missing)}")
    shifts = np.load(paths["gt_shifts_per_line.npy"], allow_pickle=False)
    axis = np.load(paths["gt_tilt_axis_per_line.npy"], allow_pickle=False)
    if shifts.shape != (n_frames, n_scan_lines, 2):
        raise ValueError(
            "gt_shifts_per_line.npy must have shape "
            f"{(n_frames, n_scan_lines, 2)}, got {shifts.shape}"
        )
    if axis.shape != (n_frames, n_scan_lines):
        raise ValueError(
            "gt_tilt_axis_per_line.npy must have shape "
            f"{(n_frames, n_scan_lines)}, got {axis.shape}"
        )
    if not np.isfinite(shifts).all() or not np.isfinite(axis).all():
        raise ValueError("GT pose arrays contain a non-finite value")
    return shifts, axis, paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out == dataset or dataset in out.parents:
        raise ValueError("--out must not be the read-only dataset or a directory inside it")
    if args.gt_poses and args.refine_pose:
        raise ValueError("--gt-poses is fixed and cannot be combined with --refine-pose")

    loaded = load_continuous_dataset(dataset)
    arm = loaded.arm
    tilt_series = loaded.tilt_series
    if args.anchor_reference and (
        loaded.angles[-1] != 0.0 or np.any(loaded.angles_per_line[-1] != 0.0)
    ):
        raise ValueError(
            "--anchor-reference requires the final frame nominal angle and all "
            "of its per-line angles to be exactly zero"
        )

    geometry = PerLineGeometry.from_angles(
        loaded.angles_per_line,
        arm=arm,
        frame_shape=loaded.frame_shape,
    )
    gt_shifts = None
    gt_axis = None
    gt_input_paths: dict[str, Path] = {}
    if args.gt_poses or args.init_poses_from_gt:
        gt_shifts, gt_axis, gt_input_paths = load_gt_poses(
            dataset,
            n_frames=loaded.frame_shape[0],
            n_scan_lines=geometry.n_scan_lines,
        )
        if args.anchor_reference and (
            np.any(gt_shifts[-1] != 0.0) or np.any(gt_axis[-1] != 0.0)
        ):
            raise ValueError(
                "--anchor-reference with GT initialization requires an exactly "
                "zero final-frame GT pose"
            )
    refine_pose = args.refine_pose or args.init_poses_from_gt
    if args.anchor_reference and not refine_pose:
        raise ValueError(
            "--anchor-reference requires --refine-pose or --init-poses-from-gt"
        )
    if args.gt_poses:
        pose_mode = "oracle-fixed-per-line-gt"
    elif args.init_poses_from_gt:
        pose_mode = "learned-frame-nodes-gt-mean-init"
    elif args.refine_pose:
        pose_mode = "learned-frame-nodes-zero-init"
    else:
        pose_mode = "off"
    if args.anchor_reference:
        pose_mode = f"{pose_mode}-anchored-final-reference"
    config = ADConfig(
        iters=args.iters,
        lr=args.lr,
        tv=args.tv,
        sparsity=args.sparsity,
        chunk=args.chunk,
        loss=args.loss,
        seed=args.seed,
        refine_pose=refine_pose,
        pose_lr_shifts=args.pose_lr_shifts,
        pose_lr_angles=args.pose_lr_angles,
        pose_start=args.pose_start,
        anchor_reference=args.anchor_reference,
    )

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    result = reconstruct(
        tilt_series,
        geometry,
        config=config,
        device=args.device,
        fixed_shifts_per_line=gt_shifts if args.gt_poses else None,
        fixed_axis_deg_per_line=gt_axis if args.gt_poses else None,
        initial_shifts_frames=(
            gt_shifts.mean(axis=1) if args.init_poses_from_gt else None
        ),
        initial_axis_deg_frames=(
            gt_axis.mean(axis=1) if args.init_poses_from_gt else None
        ),
        valid_lines=loaded.valid_lines,
    )
    runtime_seconds = time.perf_counter() - start
    finished_at = datetime.now(timezone.utc)

    out.mkdir(parents=True, exist_ok=True)
    output_volume = fasttomo_to_generator_volume(result.volume).cpu().numpy()
    np.save(out / "vol_final.npy", output_volume.astype(np.float32, copy=False))
    if result.poses is not None:
        np.savez(
            out / "poses_learned.npz",
            shifts_frames=result.poses.shifts_frames.cpu().numpy(),
            axis_deg_frames=result.poses.axis_deg_frames.cpu().numpy(),
            shifts_per_line=result.poses.shifts_per_line.cpu().numpy(),
            axis_deg_per_line=result.poses.axis_deg_per_line.cpu().numpy(),
            convention=np.asarray(result.poses.convention),
            anchor_reference=np.asarray(result.poses.anchor_reference),
            anchor_frame_index=np.asarray(
                -1
                if result.poses.anchor_frame_index is None
                else result.poses.anchor_frame_index,
                dtype=np.int64,
            ),
        )
    with (out / "loss_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("iter", "total", "fidelity", "tv", "sparsity")
        )
        writer.writeheader()
        for record in result.loss_log:
            row = asdict(record)
            row["iter"] = row.pop("iteration")
            writer.writerow(row)

    resolved_device = str(result.volume.device)
    run_manifest = {
        "dataset": str(dataset),
        "input_sha256": {
            name: sha256_file(path)
            for name, path in {**loaded.input_paths, **gt_input_paths}.items()
        },
        "config": {
            **asdict(config),
            "device_requested": args.device,
            "device_resolved": resolved_device,
            "arm": arm,
            "frame_shape_yx": list(tilt_series.shape),
            "fasttomo_volume_shape_xzy": list(geometry.fasttomo_volume_shape),
            "n_pseudo_angles": geometry.n_pseudo_angles,
            "pose_mode": pose_mode,
        },
        "git_revision": git_revision(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "runtime_seconds": runtime_seconds,
        "line_selection": {
            "valid_lines_file": (
                loaded.valid_lines_path.name
                if loaded.valid_lines_path is not None
                else None
            ),
            "total_lines": int(loaded.valid_lines.size),
            "valid_lines": int(loaded.valid_lines.sum()),
            "excluded_lines": int((~loaded.valid_lines).sum()),
            "normalization": "mean over valid measured pixels only",
        },
        "outputs": {
            "vol_final.npy": {
                "shape": list(output_volume.shape),
                "dtype": "float32",
                "array_order": "[z, y, x]",
            },
            "loss_log.csv": {"rows": len(result.loss_log)},
        },
    }
    if result.poses is not None:
        run_manifest["outputs"]["poses_learned.npz"] = {
            "shifts_frames_shape": list(result.poses.shifts_frames.shape),
            "axis_deg_frames_shape": list(result.poses.axis_deg_frames.shape),
            "shifts_per_line_shape": list(result.poses.shifts_per_line.shape),
            "axis_deg_per_line_shape": list(result.poses.axis_deg_per_line.shape),
            "convention": result.poses.convention,
            "anchor_reference": result.poses.anchor_reference,
            "anchor_frame_index": result.poses.anchor_frame_index,
        }
    with (out / "run_manifest.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(run_manifest, stream, sort_keys=False)

    print(
        f"Reconstructed {arm} dataset with {geometry.n_pseudo_angles} pseudo-angles "
        f"in {runtime_seconds:.3f}s; outputs written to {out}"
    )


if __name__ == "__main__":
    main()
