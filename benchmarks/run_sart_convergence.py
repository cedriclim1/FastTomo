"""SART vs SIRT convergence sweep.

For each (subset_size, n_iter) pair, run SART and record (time, SSIM, NMSE).
Mirrors run_pr23_24_bench but with multiple iteration counts so we can plot
SSIM-vs-time convergence for the wiki.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "32")

import torch

import fasttomo as ft
from benchmarks.datasets import load_phantom_tilt_series
from benchmarks.metrics import nmse, ssim
from benchmarks.timing import cuda_timer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/sart_convergence_results.json")
    args = parser.parse_args()

    device = torch.device("cuda")
    phantom, sino, ang_deg = load_phantom_tilt_series(device=device)
    n_angles, Ny, N_det = sino.shape
    print(f"sinogram {(n_angles, Ny, N_det)}, phantom {tuple(phantom.shape)}")

    rows: list[dict] = []

    def record(name: str, t: float, rec, n_iter: int, subset_size: int) -> None:
        s  = ssim(rec, phantom)
        nm = nmse(rec, phantom)
        rows.append({"method": name, "n_iter": n_iter, "subset_size": subset_size,
                     "time_s": t, "it_per_s": n_iter / t if t > 0 else float("nan"),
                     "ssim": s, "nmse": nm})
        print(f"  {name:<24} subset={subset_size:>3} iter={n_iter:>5}  "
              f"{t:>6.3f}s  SSIM={s:.4f}  NMSE={nm:.3e}")

    # SIRT sweep (graph path)
    proj = ft.ParallelProjector((Ny, N_det, N_det), ang_deg,
                                device=device, N_det=N_det)
    _ = ft.run_sirt(sino, ang_deg, n_iter=2, positivity=True, projector=proj)
    for n_iter in [50, 100, 200, 500, 1000, 2000]:
        with cuda_timer(device) as t:
            rec = ft.run_sirt(sino, ang_deg, n_iter=n_iter,
                              positivity=True, projector=proj)
        record("ft-sirt-graph", t[0], rec, n_iter, subset_size=n_angles)

    # SART sweep
    for subset in [1, 5, n_angles]:
        _ = ft.run_sart(sino, ang_deg, n_iter=2, subset_size=subset,
                        positivity=True)
        for n_iter in [10, 25, 50, 100, 200, 500]:
            with cuda_timer(device) as t:
                rec = ft.run_sart(sino, ang_deg, n_iter=n_iter,
                                  subset_size=subset, positivity=True)
            record(f"ft-sart-s{subset}", t[0], rec, n_iter, subset_size=subset)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": "cuda", "n_angles": n_angles, "Ny": Ny, "N_det": N_det,
        "rows": rows,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
