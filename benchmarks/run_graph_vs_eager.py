"""SIRT: eager vs CUDA-Graph speedup on two real datasets.

Datasets:
  - quantem phantom: 29 angles, 200^3, synthetic (1 deg tilt-axis error baked in)
  - HEA (Miao group): 56 angles, 210^3, real experimental tilt series

For each dataset, runs SIRT (positivity, 1000 iter) twice with the same
projector instance — once with return_loss=True (eager), once with the default
graph path. SSIM is reported against:
  - the quantem ground-truth phantom for the phantom dataset
  - the Miao group's RESIRE reconstruction (recon_resire.npy) for HEA
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "32")

import numpy as np
import torch

import fasttomo as ft
from benchmarks.datasets import load_phantom_tilt_series
from benchmarks.metrics import nmse, ssim
from benchmarks.timing import cuda_timer


HEA_DIR = Path("/home/cedlim/FastTomo/fasttomo-wiki/raw/code/hea")


def _load_hea(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tilt_series = np.load(HEA_DIR / "tilt_series.npy")
    tilt_angles = np.load(HEA_DIR / "phis.npy")
    recon_ref = np.load(HEA_DIR / "recon_resire.npy")
    tilt_series = np.array([np.rot90(img) for img in tilt_series])
    tilt_series = (tilt_series / tilt_series.max()).astype(np.float32)
    sino = torch.from_numpy(tilt_series).float().to(device)
    ang = torch.from_numpy(tilt_angles).float().to(device)
    ref = torch.from_numpy(recon_ref).float().to(device)
    return ref, sino, ang


def _run_pair(name: str, ref: torch.Tensor, sino: torch.Tensor,
              ang: torch.Tensor, n_iter: int, device: torch.device) -> list[dict]:
    n_angles, Ny, N_det = sino.shape
    proj = ft.ParallelProjector((Ny, N_det, N_det), ang,
                                device=device, N_det=N_det)

    print(f"\n=== {name}  |  sino {tuple(sino.shape)}  |  {n_iter} iter ===")

    # Warmup eager (return_loss=True forces eager loop)
    _ = ft.run_sirt(sino, ang, n_iter=2, positivity=True,
                    projector=proj, return_loss=True)
    with cuda_timer(device) as t_e:
        rec_eager, _ = ft.run_sirt(sino, ang, n_iter=n_iter,
                                   positivity=True, projector=proj,
                                   return_loss=True)

    # Warmup graph (also primes the cache)
    _ = ft.run_sirt(sino, ang, n_iter=n_iter,
                    positivity=True, projector=proj)
    with cuda_timer(device) as t_g:
        rec_graph = ft.run_sirt(sino, ang, n_iter=n_iter,
                                positivity=True, projector=proj)

    s_e, s_g = ssim(rec_eager, ref), ssim(rec_graph, ref)
    n_e, n_g = nmse(rec_eager, ref), nmse(rec_graph, ref)
    diff = (rec_eager - rec_graph).abs().max().item()
    speedup = t_e[0] / t_g[0] if t_g[0] > 0 else float("nan")

    print(f"  eager : {t_e[0]*1000:8.2f} ms   it/s={n_iter/t_e[0]:8.1f}   "
          f"SSIM={s_e:.4f}   NMSE={n_e:.3e}")
    print(f"  graph : {t_g[0]*1000:8.2f} ms   it/s={n_iter/t_g[0]:8.1f}   "
          f"SSIM={s_g:.4f}   NMSE={n_g:.3e}")
    print(f"  speedup: {speedup:.2f}x   max|eager - graph| = {diff:.3e}")

    rows = [
        {"dataset": name, "path": "eager",
         "n_iter": n_iter, "time_s": t_e[0],
         "it_per_s": n_iter / t_e[0],
         "ssim": s_e, "nmse": n_e},
        {"dataset": name, "path": "graph",
         "n_iter": n_iter, "time_s": t_g[0],
         "it_per_s": n_iter / t_g[0],
         "ssim": s_g, "nmse": n_g},
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--out", default="benchmarks/graph_vs_eager_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows: list[dict] = []

    # quantem phantom
    phantom, sino_p, ang_p = load_phantom_tilt_series(device=device)
    rows += _run_pair("quantem-phantom", phantom, sino_p, ang_p,
                      args.n_iter, device)

    # HEA
    ref_h, sino_h, ang_h = _load_hea(device)
    rows += _run_pair("hea", ref_h, sino_h, ang_h, args.n_iter, device)

    out = Path(args.out)
    out.write_text(json.dumps({"n_iter": args.n_iter, "rows": rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
