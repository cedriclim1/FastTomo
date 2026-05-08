"""Presentation benchmark — collect timing + SSIM for the cached / fastest path
of every iterative method we want to show in slides.

Writes ``benchmarks/presentation_results.json``. Pair with
``benchmarks/plot_presentation.py`` to render figures.

Usage:
    conda run -n FastTomo-bench python -m benchmarks.run_presentation
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

# Set OpenMP thread count BEFORE importing anything that loads libgomp
# (FastTomo C kernels, torch, numpy, ASTRA). Honors a user-provided env override.
os.environ.setdefault("OMP_NUM_THREADS", "32")

import numpy as np
import torch

import fasttomo as ft
from benchmarks.ad_reconstruct import (
    ad_reconstruct,
    ad_reconstruct_cached,
    ad_reconstruct_with_pose,
    ad_reconstruct_with_pose_cached,
)
from benchmarks.datasets import load_phantom_tilt_series
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
    parser.add_argument("--ad-iter", type=int, default=1000)
    parser.add_argument("--sirt-iter", type=int, default=1000)
    parser.add_argument("--tv-weight", type=float, default=1.0)
    parser.add_argument("--tr-iter", type=int, default=10,
                        help="TorchRadon iterative-FBP iters (semi-converges; see "
                             "wiki/analyses/torch-radon-sirt-semi-convergence.md)")
    parser.add_argument("--tr-step", type=float, default=0.05)
    parser.add_argument("--tr-smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--out", default="benchmarks/presentation_results.json")
    parser.add_argument("--no-cpu", action="store_true",
                        help="skip the CPU benchmark section")
    parser.add_argument("--cpu-iter", type=int, default=20,
                        help="iters for CPU SIRT (FastTomo + ASTRA)")
    parser.add_argument("--cpu-ad-iter", type=int, default=20,
                        help="iters for CPU AD (FastTomo)")
    parser.add_argument("--cpu-Ny", type=int, default=0,
                        help="if >0, use only this many Y-slices for CPU "
                             "benchmarks to keep wall-time tractable")
    args = parser.parse_args()

    device = torch.device(args.device)
    phantom, sino, ang_deg = load_phantom_tilt_series(device=device)
    n_angles, Ny, N_det = sino.shape
    print(f"sinogram {(n_angles, Ny, N_det)}, phantom {tuple(phantom.shape)}")

    ang_rad = (ang_deg.cpu().numpy() * np.pi / 180.0).astype(np.float32)
    sino_np = sino.cpu().numpy()

    rows: list[dict] = []

    def record(name: str, kind: str, t: float, rec, n_iter: int) -> None:
        if isinstance(rec, np.ndarray):
            rec_t = torch.from_numpy(rec)
        else:
            rec_t = rec
        s = ssim(rec_t, phantom)
        nm = nmse(rec_t, phantom)
        rows.append({
            "method": name,
            "kind": kind,
            "n_iter": n_iter,
            "time_s": t,
            "it_per_s": n_iter / t if t > 0 else float("nan"),
            "ssim": s,
            "nmse": nm,
        })
        print(f"  {name:<28} {n_iter:>5} iter  {t:>7.3f}s  it/s={n_iter/t:>9.1f}  "
              f"SSIM={s:.4f}  NMSE={nm:.3e}")

    # ------- FBP (one-shot, not iterative) — for SSIM-only context -------
    print("\n[FastTomo] run_fbp (cached)")
    proj_fbp = ft.ParallelProjector((Ny, N_det, N_det), ang_deg,
                                    device=device, N_det=N_det)
    _ = ft.run_fbp(sino, ang_deg, positivity=True, projector=proj_fbp)
    with cuda_timer(device) as t:
        rec_fbp = ft.run_fbp(sino, ang_deg, positivity=True, projector=proj_fbp)
    record("ft-fbp-cached", "ft", t[0], rec_fbp, 1)

    # ------- FastTomo SIRT (cached) -------
    print("\n[FastTomo] run_sirt (cached, positivity)")
    proj_sirt = ft.ParallelProjector((Ny, N_det, N_det), ang_deg,
                                     device=device, N_det=N_det)
    _ = ft.run_sirt(sino, ang_deg, n_iter=2, positivity=True, projector=proj_sirt)
    with cuda_timer(device) as t:
        rec_sirt = ft.run_sirt(sino, ang_deg, n_iter=args.sirt_iter,
                               positivity=True, projector=proj_sirt)
    record("ft-sirt-cached", "ft", t[0], rec_sirt, args.sirt_iter)

    # ------- FastTomo RESIRE -------
    print("\n[FastTomo] run_resire (positivity)")
    _ = ft.run_resire(sino[:, :1, :], ang_deg, n_iter=2, positivity=True)
    with cuda_timer(device) as t:
        rec_resire = ft.run_resire(sino, ang_deg, n_iter=args.sirt_iter,
                                   positivity=True)
    record("ft-resire", "ft", t[0], rec_resire, args.sirt_iter)

    # ------- FastTomo GENFIRE -------
    print("\n[FastTomo] run_genfire (positivity, mask, resolution-extension)")
    _ = ft.run_genfire(sino[:, :1, :], ang_deg, n_iter=2,
                       positivity=True, mask=True)
    with cuda_timer(device) as t:
        rec_gen = ft.run_genfire(sino, ang_deg, n_iter=args.sirt_iter,
                                 positivity=True, mask=True,
                                 resolution_extension=True)
    record("ft-genfire", "ft", t[0], rec_gen, args.sirt_iter)

    # ------- FastTomo AD no-pose (cached) -------
    print(f"\n[FastTomo] AD no-pose CACHED ({args.ad_iter} iter)")
    with cuda_timer(device) as t:
        res_ad = ad_reconstruct_cached(sino, ang_deg, n_iter=args.ad_iter,
                                       tv_weight=args.tv_weight)
    record("ft-ad-cached", "ft-ad", t[0], res_ad.volume, args.ad_iter)

    # ------- FastTomo AD with pose (cached) -------
    print(f"\n[FastTomo] AD with-pose CACHED ({args.ad_iter} iter)")
    with cuda_timer(device) as t:
        res_pose = ad_reconstruct_with_pose_cached(
            sino, ang_deg, n_iter=args.ad_iter, tv_weight=args.tv_weight,
        )
    record("ft-ad-with-pose-cached", "ft-ad", t[0], res_pose.volume, args.ad_iter)
    refined = res_pose.euler_zyz_deg.cpu()
    theta_err = (refined[:, 1] - ang_deg.cpu()).abs().max().item()
    print(f"  pose: max|Δθ|={theta_err:.3f}°  max|φ|={refined[:,0].abs().max():.3f}°  "
          f"max|ψ|={refined[:,2].abs().max():.3f}°  "
          f"max|shift|={res_pose.shifts.abs().max():.3f}px")

    # ------- ASTRA SIRT_CUDA -------
    print(f"\n[ASTRA] SIRT_CUDA ({args.sirt_iter} iter, MinConstraint=0)")
    if _try_import("astra") is not None:
        from benchmarks.astra_wrapper import AstraParallelBeam
        astra_w = AstraParallelBeam(N_det, ang_rad)
        _ = astra_w.sirt(sino_np[:, :1, :], num_iter=2)
        with cuda_timer(device) as t:
            rec_astra = astra_w.sirt(sino_np, num_iter=args.sirt_iter,
                                     positivity=True)
        record("astra-sirt", "astra", t[0], rec_astra, args.sirt_iter)

        # ASTRA FBP for SSIM context
        print("\n[ASTRA] FBP_CUDA")
        _ = astra_w.fbp(sino_np[:, :1, :], positivity=True)
        with cuda_timer(device) as t:
            rec_astra_fbp = astra_w.fbp(sino_np, positivity=True)
        record("astra-fbp", "astra", t[0], rec_astra_fbp, 1)

    # ------- TorchRadon iterative-FBP -------
    sigma = args.tr_smoothing_sigma if args.tr_smoothing_sigma > 0 else None
    print(f"\n[TorchRadon] iterative-FBP ({args.tr_iter} iter, step={args.tr_step})")
    if _try_import("torch_radon") is not None:
        from benchmarks.torch_radon_wrapper import TorchRadonParallelBeam
        tr_w = TorchRadonParallelBeam(N_det, ang_rad)
        _ = tr_w.sirt(sino[:, :1, :], num_iter=2, step_size=args.tr_step,
                      smoothing_sigma=sigma)
        with cuda_timer(device) as t:
            rec_tr = tr_w.sirt(sino, num_iter=args.tr_iter,
                               step_size=args.tr_step, positivity=True,
                               smoothing_sigma=sigma)
        record("torch_radon-sirt", "torch_radon", t[0], rec_tr, args.tr_iter)

        # TorchRadon FBP for SSIM context
        print("\n[TorchRadon] FBP")
        _ = tr_w.fbp(sino[:, :1, :], positivity=True)
        with cuda_timer(device) as t:
            rec_tr_fbp = tr_w.fbp(sino, positivity=True)
        record("torch_radon-fbp", "torch_radon", t[0], rec_tr_fbp, 1)

    # ------- CPU section: FastTomo CPU + ASTRA CPU --------------------------
    cpu_rows: list[dict] = []
    if not args.no_cpu:
        import time as _time

        cpu_iter = args.cpu_iter
        cpu_ad_iter = args.cpu_ad_iter
        if args.cpu_Ny > 0 and args.cpu_Ny < Ny:
            sino_cpu_t = sino[:, :args.cpu_Ny, :].cpu().contiguous()
            phantom_cpu = phantom[:args.cpu_Ny].cpu().contiguous()
            cpu_Ny = args.cpu_Ny
        else:
            sino_cpu_t = sino.cpu().contiguous()
            phantom_cpu = phantom.cpu().contiguous()
            cpu_Ny = Ny
        sino_cpu_np = sino_cpu_t.numpy()
        ang_deg_cpu = ang_deg.cpu()

        n_threads = int(os.environ.get("OMP_NUM_THREADS", "32"))
        torch.set_num_threads(n_threads)
        print(f"\n=== CPU section  (Ny={cpu_Ny}, OMP_NUM_THREADS={n_threads}, "
              f"torch.get_num_threads={torch.get_num_threads()}) ===")

        def record_cpu(name: str, kind: str, t: float, rec, n_iter: int) -> None:
            rec_t = torch.from_numpy(rec) if isinstance(rec, np.ndarray) else rec
            s  = ssim(rec_t, phantom_cpu)
            nm = nmse(rec_t, phantom_cpu)
            cpu_rows.append({
                "method": name, "kind": kind, "n_iter": n_iter,
                "time_s": t, "it_per_s": n_iter / t if t > 0 else float("nan"),
                "ssim": s, "nmse": nm,
            })
            print(f"  {name:<28} {n_iter:>5} iter  {t:>7.3f}s  "
                  f"it/s={n_iter/t:>9.2f}  SSIM={s:.4f}  NMSE={nm:.3e}")

        # FastTomo CPU FBP
        print("\n[FastTomo CPU] run_fbp")
        _ = ft.run_fbp(sino_cpu_t[:, :1, :], ang_deg_cpu, positivity=True)
        s = _time.time(); rec_cpu_fbp = ft.run_fbp(sino_cpu_t, ang_deg_cpu, positivity=True); t_e = _time.time() - s
        record_cpu("ft-fbp-cpu", "ft-cpu", t_e, rec_cpu_fbp, 1)

        # FastTomo CPU SIRT
        print(f"\n[FastTomo CPU] run_sirt  ({cpu_iter} iter, positivity)")
        _ = ft.run_sirt(sino_cpu_t[:, :1, :], ang_deg_cpu, n_iter=2, positivity=True)
        s = _time.time(); rec_cpu_sirt = ft.run_sirt(sino_cpu_t, ang_deg_cpu, n_iter=cpu_iter, positivity=True); t_e = _time.time() - s
        record_cpu("ft-sirt-cpu", "ft-cpu", t_e, rec_cpu_sirt, cpu_iter)

        # ASTRA CPU FBP / SIRT
        if _try_import("astra") is not None:
            from benchmarks.astra_wrapper import AstraParallelBeam
            astra_w_cpu = AstraParallelBeam(N_det, ang_rad)

            print("\n[ASTRA CPU] FBP (strip)")
            _ = astra_w_cpu.fbp_cpu(sino_cpu_np[:, :1, :], positivity=True)
            s = _time.time(); rec_a_fbp = astra_w_cpu.fbp_cpu(sino_cpu_np, positivity=True); t_e = _time.time() - s
            record_cpu("astra-fbp-cpu", "astra-cpu", t_e, rec_a_fbp, 1)

            print(f"\n[ASTRA CPU] SIRT (strip, {cpu_iter} iter, MinConstraint=0)")
            _ = astra_w_cpu.sirt_cpu(sino_cpu_np[:, :1, :], num_iter=2)
            s = _time.time(); rec_a_sirt = astra_w_cpu.sirt_cpu(sino_cpu_np, num_iter=cpu_iter, positivity=True); t_e = _time.time() - s
            record_cpu("astra-sirt-cpu", "astra-cpu", t_e, rec_a_sirt, cpu_iter)

        # FastTomo CPU AD no-pose
        print(f"\n[FastTomo CPU] AD no-pose ({cpu_ad_iter} iter)")
        _ = ad_reconstruct(sino_cpu_t[:, :2, :], ang_deg_cpu, n_iter=2,
                           tv_weight=args.tv_weight)
        s = _time.time()
        res_ad_cpu = ad_reconstruct(sino_cpu_t, ang_deg_cpu,
                                    n_iter=cpu_ad_iter, tv_weight=args.tv_weight)
        t_e = _time.time() - s
        record_cpu("ft-ad-cpu", "ft-ad-cpu", t_e, res_ad_cpu.volume, cpu_ad_iter)

        # FastTomo CPU AD with pose
        print(f"\n[FastTomo CPU] AD with-pose ({cpu_ad_iter} iter)")
        _ = ad_reconstruct_with_pose(sino_cpu_t[:, :2, :], ang_deg_cpu, n_iter=2,
                                     tv_weight=args.tv_weight)
        s = _time.time()
        res_ad_pose_cpu = ad_reconstruct_with_pose(
            sino_cpu_t, ang_deg_cpu,
            n_iter=cpu_ad_iter, tv_weight=args.tv_weight,
        )
        t_e = _time.time() - s
        record_cpu("ft-ad-with-pose-cpu", "ft-ad-cpu", t_e,
                   res_ad_pose_cpu.volume, cpu_ad_iter)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "n_angles": n_angles,
        "Ny": Ny,
        "N_det": N_det,
        "ad_iter": args.ad_iter,
        "sirt_iter": args.sirt_iter,
        "tr_iter": args.tr_iter,
        "tv_weight": args.tv_weight,
        "cpu_iter": args.cpu_iter,
        "cpu_ad_iter": args.cpu_ad_iter,
        "cpu_Ny": args.cpu_Ny if args.cpu_Ny > 0 else Ny,
        "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "32")),
        "rows": rows,
        "cpu_rows": cpu_rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
