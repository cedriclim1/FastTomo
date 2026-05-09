"""Benchmark for PR#23 (SART baseline + CUDA-Graph SIRT) and PR#24
(graph-cache stale-input fix).

Measures speed + SSIM/NMSE of the iterative reconstructions on the
HEA-like phantom (29 angles, 200x200x200, 1° tilt-axis error baked in).

Reports:
  - SIRT eager (return_loss=True forces eager loop)
  - SIRT CUDA-Graph (default path, post-PR#23)
  - SART with subset_size in {1, 5, 29}

Wall-clock is timed with cuda events; first call per method is a warmup.

Usage:
    conda run -n FastTomo-bench python -m benchmarks.run_pr23_24_bench
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sirt-iter", type=int, default=1000)
    parser.add_argument("--sart-iter", type=int, default=50)
    parser.add_argument("--out", default="benchmarks/pr23_24_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    phantom, sino, ang_deg = load_phantom_tilt_series(device=device)
    n_angles, Ny, N_det = sino.shape
    print(f"sinogram {(n_angles, Ny, N_det)}, phantom {tuple(phantom.shape)}")

    rows: list[dict] = []

    def record(name: str, t: float, rec, n_iter: int, extra: dict | None = None) -> None:
        rec_t = rec if isinstance(rec, torch.Tensor) else torch.from_numpy(rec)
        s = ssim(rec_t, phantom)
        nm = nmse(rec_t, phantom)
        row = {
            "method": name,
            "n_iter": n_iter,
            "time_s": t,
            "it_per_s": n_iter / t if t > 0 else float("nan"),
            "ssim": s,
            "nmse": nm,
        }
        if extra:
            row.update(extra)
        rows.append(row)
        print(f"  {name:<32} {n_iter:>5} iter  {t:>7.3f}s  it/s={n_iter/t:>9.1f}  "
              f"SSIM={s:.4f}  NMSE={nm:.3e}")

    # --- SIRT eager (return_loss=True forces the eager loop) ---
    print(f"\n[FastTomo] run_sirt EAGER (return_loss=True), {args.sirt_iter} iter")
    proj_eager = ft.ParallelProjector((Ny, N_det, N_det), ang_deg,
                                      device=device, N_det=N_det)
    _ = ft.run_sirt(sino, ang_deg, n_iter=2, positivity=True,
                    projector=proj_eager, return_loss=True)
    with cuda_timer(device) as t:
        rec_eager, _loss = ft.run_sirt(sino, ang_deg, n_iter=args.sirt_iter,
                                       positivity=True, projector=proj_eager,
                                       return_loss=True)
    record("ft-sirt-eager", t[0], rec_eager, args.sirt_iter)

    # --- SIRT CUDA-Graph (default cached path) ---
    print(f"\n[FastTomo] run_sirt GRAPH (cached, default), {args.sirt_iter} iter")
    proj_graph = ft.ParallelProjector((Ny, N_det, N_det), ang_deg,
                                      device=device, N_det=N_det)
    # Warmup (also primes the graph cache)
    _ = ft.run_sirt(sino, ang_deg, n_iter=args.sirt_iter,
                    positivity=True, projector=proj_graph)
    with cuda_timer(device) as t:
        rec_graph = ft.run_sirt(sino, ang_deg, n_iter=args.sirt_iter,
                                positivity=True, projector=proj_graph)
    record("ft-sirt-graph", t[0], rec_graph, args.sirt_iter)

    # Sanity: graph and eager should reconstruct the same volume.
    diff = (rec_eager - rec_graph).abs().max().item()
    print(f"  max|eager - graph| = {diff:.3e} (sanity)")

    # --- SART (frozen baseline; no projector caching, no graphs) ---
    for subset in [1, 5, n_angles]:
        n_iter = args.sart_iter
        print(f"\n[FastTomo] run_sart subset_size={subset}, {n_iter} iter")
        # Warmup
        _ = ft.run_sart(sino, ang_deg, n_iter=2, subset_size=subset,
                        positivity=True)
        with cuda_timer(device) as t:
            rec_sart = ft.run_sart(sino, ang_deg, n_iter=n_iter,
                                   subset_size=subset, positivity=True)
        record(f"ft-sart-subset{subset}", t[0], rec_sart, n_iter,
               extra={"subset_size": subset})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "n_angles": n_angles,
        "Ny": Ny,
        "N_det": N_det,
        "sirt_iter": args.sirt_iter,
        "sart_iter": args.sart_iter,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
