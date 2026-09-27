"""Run SIRT reconstruction on a continuous-rotation dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

try:
    from .utils.perrow_ad import (
        fasttomo_to_generator_volume,
        load_continuous_dataset,
    )
    from .utils.perrow_sirt import SIRTMode, sirt_continuous
except ImportError:  # Direct execution: python benchmarks/scripts/recon_....py
    from utils.perrow_ad import (  # type: ignore[no-redef]
        fasttomo_to_generator_volume,
        load_continuous_dataset,
    )
    from utils.perrow_sirt import SIRTMode, sirt_continuous  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("auto", "perp-exact", "midangle"),
        default="auto",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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


def resolve_mode(requested_mode: str, arm: str) -> SIRTMode:
    """Resolve auto mode without inventing a detector-varying SIRT operator."""
    if requested_mode == "auto":
        return "perp-exact" if arm == "fast-perp" else "midangle"
    if requested_mode not in ("perp-exact", "midangle"):
        raise ValueError(f"unsupported mode {requested_mode!r}")
    return requested_mode


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out == dataset_path or dataset_path in out.parents:
        raise ValueError("--out must not be the read-only dataset or a directory inside it")

    dataset = load_continuous_dataset(dataset_path)
    mode = resolve_mode(args.mode, dataset.arm)

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    result = sirt_continuous(
        dataset,
        mode,
        n_iter=args.iters,
        omega=args.omega,
        device=args.device,
        seed=args.seed,
    )
    runtime_seconds = time.perf_counter() - start
    finished_at = datetime.now(timezone.utc)

    out.mkdir(parents=True, exist_ok=True)
    output_volume = fasttomo_to_generator_volume(result.volume).cpu().numpy()
    np.save(out / "vol_final.npy", output_volume.astype(np.float32, copy=False))
    with (out / "loss_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("iter", "sinogram_mse"))
        writer.writeheader()
        for iteration, loss in enumerate(result.loss_history.tolist(), start=1):
            writer.writerow({"iter": iteration, "sinogram_mse": loss})

    run_manifest = {
        "dataset": str(dataset.path),
        "input_sha256": {
            name: sha256_file(path) for name, path in dataset.input_paths.items()
        },
        "config": {
            "mode_requested": args.mode,
            "mode": mode,
            "iters": args.iters,
            "omega": args.omega,
            "device_requested": args.device,
            "device_resolved": str(result.volume.device),
            "seed": args.seed,
            "arm": dataset.arm,
            "frame_shape_kyx": list(dataset.frame_shape),
            "fasttomo_volume_shape_xzy": list(result.volume.shape),
        },
        "git_revision": git_revision(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "runtime_seconds": runtime_seconds,
        "outputs": {
            "vol_final.npy": {
                "shape": list(output_volume.shape),
                "dtype": "float32",
                "array_order": "[z, y, x]",
            },
            "loss_log.csv": {"rows": len(result.loss_history)},
        },
    }
    with (out / "run_manifest.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(run_manifest, stream, sort_keys=False)

    print(
        f"Reconstructed {dataset.arm} dataset in {mode} mode in "
        f"{runtime_seconds:.3f}s; outputs written to {out}"
    )


if __name__ == "__main__":
    main()
