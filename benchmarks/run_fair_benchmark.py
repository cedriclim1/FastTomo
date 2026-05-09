"""Fair benchmark — perfect tilt series, no pose optimization.

Generates timing and SSIM metrics for every algorithm on a zero-error sinogram
derived from the FePt phantom via forward projection.  Pose optimization is
disabled for all methods so this reflects pure reconstruction quality.

Prerequisites:
    python scripts/make_perfect_sino.py   # creates data/fept_phantom_200_perfect_sino.npy

Usage:
    conda run -n FastTomo-bench python -m benchmarks.run_fair_benchmark
    conda run -n FastTomo-bench python -m benchmarks.run_fair_benchmark --device cuda:1 --n-iter 1000
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "32")

import numpy as np
import torch

import fasttomo as ft
from benchmarks.ad_reconstruct import ad_reconstruct_cached
from benchmarks.datasets import load_fair_tilt_series
from benchmarks.metrics import nmse, ssim
from benchmarks.timing import cuda_timer


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        print(f"  [skipped: {name} unavailable: {type(exc).__name__}: {exc}]")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-iter", type=int, default=1000,
                        help="iterations for all iterative methods")
    parser.add_argument("--tr-iter", type=int, default=10,
                        help="TorchRadon iters (semi-converges early)")
    parser.add_argument("--tr-step", type=float, default=0.05)
    parser.add_argument("--tr-smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--tv-weight", type=float, default=1.0,
                        help="TV weight for AD reconstruction")
    parser.add_argument("--ssim-weight", type=float, default=100.0,
                        help="SSIM loss weight for ft-ad-tv-ssim")
    parser.add_argument("--out", default="benchmarks/fair_benchmark_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    phantom, sino, ang_deg = load_fair_tilt_series(device=device)
    n_angles, Ny, N_det = sino.shape
    print(f"sinogram {(n_angles, Ny, N_det)}, phantom {tuple(phantom.shape)}")

    ang_rad = (ang_deg.cpu().numpy() * np.pi / 180.0).astype(np.float32)
    sino_np = sino.cpu().numpy()
    n_iter = args.n_iter

    rows: list[dict] = []

    def record(name: str, kind: str, t: float, rec, n_it: int) -> None:
        rec_t = torch.from_numpy(rec) if isinstance(rec, np.ndarray) else rec
        s = ssim(rec_t, phantom)
        nm = nmse(rec_t, phantom)
        rows.append({
            "method": name,
            "kind": kind,
            "n_iter": n_it,
            "time_s": t,
            "it_per_s": n_it / t if t > 0 else float("nan"),
            "ssim": s,
            "nmse": nm,
        })
        print(f"  {name:<28} {n_it:>5} iter  {t:>7.3f}s  it/s={n_it/t:>9.1f}  "
              f"SSIM={s:.4f}  NMSE={nm:.3e}")

    # ------- FastTomo FBP (cached) -------
    print("\n[FastTomo] run_fbp (cached)")
    proj_fbp = ft.ParallelProjector((Ny, N_det, N_det), ang_deg,
                                    device=device, N_det=N_det)
    _ = ft.run_fbp(sino, ang_deg, positivity=True, projector=proj_fbp)
    with cuda_timer(device) as t:
        rec_fbp = ft.run_fbp(sino, ang_deg, positivity=True, projector=proj_fbp)
    record("ft-fbp", "ft", t[0], rec_fbp, 1)

    # ------- FastTomo SIRT (cached) -------
    print(f"\n[FastTomo] run_sirt (cached, {n_iter} iter, positivity)")
    proj_sirt = ft.ParallelProjector((Ny, N_det, N_det), ang_deg,
                                     device=device, N_det=N_det)
    _ = ft.run_sirt(sino, ang_deg, n_iter=2, positivity=True, projector=proj_sirt)
    with cuda_timer(device) as t:
        rec_sirt = ft.run_sirt(sino, ang_deg, n_iter=n_iter,
                               positivity=True, projector=proj_sirt)
    record("ft-sirt", "ft", t[0], rec_sirt, n_iter)

    # ------- FastTomo SART -------
    print(f"\n[FastTomo] run_sart (subset_size=1, {n_iter} iter, positivity)")
    _ = ft.run_sart(sino[:, :1, :], ang_deg, n_iter=2, positivity=True)
    with cuda_timer(device) as t:
        rec_sart = ft.run_sart(sino, ang_deg, n_iter=n_iter,
                               subset_size=1, positivity=True)
    record("ft-sart", "ft", t[0], rec_sart, n_iter)

    # ------- FastTomo W-SIRT -------
    print(f"\n[FastTomo] run_wsirt ({n_iter} iter, positivity)")
    _ = ft.run_wsirt(sino[:, :1, :], ang_deg, n_iter=2, positivity=True)
    with cuda_timer(device) as t:
        rec_wsirt = ft.run_wsirt(sino, ang_deg, n_iter=n_iter, positivity=True)
    record("ft-wsirt", "ft", t[0], rec_wsirt, n_iter)

    # ------- FastTomo RESIRE -------
    print(f"\n[FastTomo] run_resire ({n_iter} iter, positivity)")
    _ = ft.run_resire(sino[:, :1, :], ang_deg, n_iter=2, positivity=True)
    with cuda_timer(device) as t:
        rec_resire = ft.run_resire(sino, ang_deg, n_iter=n_iter, positivity=True)
    record("ft-resire", "ft", t[0], rec_resire, n_iter)

    # ------- FastTomo GENFIRE -------
    print(f"\n[FastTomo] run_genfire ({n_iter} iter, positivity, mask, res-ext)")
    _ = ft.run_genfire(sino[:, :1, :], ang_deg, n_iter=2, positivity=True, mask=True)
    with cuda_timer(device) as t:
        rec_genfire = ft.run_genfire(sino, ang_deg, n_iter=n_iter,
                                     positivity=True, mask=True,
                                     resolution_extension=True)
    record("ft-genfire", "ft", t[0], rec_genfire, n_iter)

    # ------- FastTomo AD no regularization no-pose (cached) -------
    print(f"\n[FastTomo] AD no-reg no-pose CACHED ({n_iter} iter)")
    with cuda_timer(device) as t:
        res_ad_noreg = ad_reconstruct_cached(sino, ang_deg, n_iter=n_iter,
                                             tv_weight=0.0)
    record("ft-ad", "ft-ad", t[0], res_ad_noreg.volume, n_iter)

    # ------- FastTomo AD + TV no-pose (cached) -------
    print(f"\n[FastTomo] AD + TV no-pose CACHED ({n_iter} iter, tv={args.tv_weight})")
    with cuda_timer(device) as t:
        res_ad = ad_reconstruct_cached(sino, ang_deg, n_iter=n_iter,
                                       tv_weight=args.tv_weight)
    record("ft-ad-tv", "ft-ad", t[0], res_ad.volume, n_iter)

    # ------- FastTomo AD + TV + SSIM no-pose (cached) -------
    print(f"\n[FastTomo] AD + TV + SSIM no-pose CACHED "
          f"({n_iter} iter, tv={args.tv_weight}, ssim={args.ssim_weight})")
    with cuda_timer(device) as t:
        res_ad_ssim = ad_reconstruct_cached(sino, ang_deg, n_iter=n_iter,
                                            tv_weight=args.tv_weight,
                                            ssim_weight=args.ssim_weight)
    record("ft-ad-tv-ssim", "ft-ad", t[0], res_ad_ssim.volume, n_iter)

    # ------- ASTRA -------
    if _try_import("astra") is not None:
        from benchmarks.astra_wrapper import AstraParallelBeam
        astra_w = AstraParallelBeam(N_det, ang_rad)

        print(f"\n[ASTRA] SIRT_CUDA ({n_iter} iter, MinConstraint=0)")
        _ = astra_w.sirt(sino_np[:, :1, :], num_iter=2)
        with cuda_timer(device) as t:
            rec_astra_sirt = astra_w.sirt(sino_np, num_iter=n_iter, positivity=True)
        record("astra-sirt", "astra", t[0], rec_astra_sirt, n_iter)

        print("\n[ASTRA] FBP_CUDA")
        _ = astra_w.fbp(sino_np[:, :1, :], positivity=True)
        with cuda_timer(device) as t:
            rec_astra_fbp = astra_w.fbp(sino_np, positivity=True)
        record("astra-fbp", "astra", t[0], rec_astra_fbp, 1)

    # ------- TorchRadon -------
    sigma = args.tr_smoothing_sigma if args.tr_smoothing_sigma > 0 else None
    if _try_import("torch_radon") is not None:
        from benchmarks.torch_radon_wrapper import TorchRadonParallelBeam
        tr_w = TorchRadonParallelBeam(N_det, ang_rad)

        print(f"\n[TorchRadon] iterative-FBP ({args.tr_iter} iter, step={args.tr_step})")
        _ = tr_w.sirt(sino[:, :1, :], num_iter=2, step_size=args.tr_step,
                      smoothing_sigma=sigma)
        with cuda_timer(device) as t:
            rec_tr = tr_w.sirt(sino, num_iter=args.tr_iter,
                               step_size=args.tr_step, positivity=True,
                               smoothing_sigma=sigma)
        record("torch_radon-sirt", "torch_radon", t[0], rec_tr, args.tr_iter)

        print("\n[TorchRadon] FBP")
        _ = tr_w.fbp(sino[:, :1, :], positivity=True)
        with cuda_timer(device) as t:
            rec_tr_fbp = tr_w.fbp(sino, positivity=True)
        record("torch_radon-fbp", "torch_radon", t[0], rec_tr_fbp, 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "n_angles": n_angles,
        "Ny": Ny,
        "N_det": N_det,
        "n_iter": n_iter,
        "tr_iter": args.tr_iter,
        "tv_weight": args.tv_weight,
        "ssim_weight": args.ssim_weight,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
