"""Generate the Bragg-anomaly-corrupted BF-TEM phantom sinogram artifact.

Loads `data/quantem_phantom_perfect_sino.npy` (the clean phantom forward
projection) and injects deterministic Bragg-style anomalies: at a small set
of (tilt, detector-strip) coordinates, multiply the sinogram values by a
high factor to simulate Bragg-scatter contrast reversal. Writes the result
to `data/quantem_phantom_perfect_sino_bftem.npy`.

The output is the input to the MBIR benchmark notebook. Deterministic
(fixed seed) so the artifact is reproducible across machines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def inject_bragg_streaks(
    sino: np.ndarray,
    n_streaks: int = 8,
    strip_width: int = 3,
    magnitude: float = 5.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject `n_streaks` Bragg-like anomalies into `sino`.

    Each anomaly is a contiguous strip of `strip_width` detector pixels
    in one tilt, multiplied by `magnitude`. Affects all Y rows in that
    tilt's strip (which simulates a constant-Y crystalline grain catching
    Bragg condition at one tilt). Returns the corrupted sinogram and a
    [n_angles, Ny, N_det] boolean mask marking the corrupted pixels.
    """
    rng = np.random.default_rng(seed)
    n_angles, _, N_det = sino.shape
    out = sino.copy()
    mask = np.zeros_like(sino, dtype=bool)
    for _ in range(n_streaks):
        angle_idx = int(rng.integers(0, n_angles))
        det_start = int(rng.integers(0, N_det - strip_width))
        out[angle_idx, :, det_start:det_start + strip_width] *= magnitude
        mask[angle_idx, :, det_start:det_start + strip_width] = True
    return out, mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("data/quantem_phantom_perfect_sino.npy"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/quantem_phantom_perfect_sino_bftem.npy"))
    parser.add_argument("--mask-output", type=Path,
                        default=Path("data/quantem_phantom_perfect_sino_bftem_mask.npy"))
    parser.add_argument("--n-streaks", type=int, default=8)
    parser.add_argument("--strip-width", type=int, default=3)
    parser.add_argument("--magnitude", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sino = np.load(args.input).astype(np.float32)
    print(f"loaded {args.input} shape={sino.shape} max={sino.max():.3f}")

    corrupt, mask = inject_bragg_streaks(
        sino,
        n_streaks=args.n_streaks,
        strip_width=args.strip_width,
        magnitude=args.magnitude,
        seed=args.seed,
    )
    np.save(args.output, corrupt)
    np.save(args.mask_output, mask)
    print(f"wrote {args.output}  ({corrupt.dtype}, max={corrupt.max():.3f})")
    print(f"wrote {args.mask_output}  ({mask.sum()} corrupted pixels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
