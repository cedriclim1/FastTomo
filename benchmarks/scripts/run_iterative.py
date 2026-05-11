"""Stage 4 cross-backend iterative comparison.

Methods, all reconstructing the same 1°-tilt-axis-error tilt series with
positivity enforced. SSIM/NMSE vs the ground-truth phantom.

* FastTomo AD without pose optimization (Adam + cosine LR + TV)
* FastTomo AD without pose, ParallelProjector cached
* FastTomo AD with pose optimization (Euler + shifts learnable)
* FastTomo AD with pose, RotmatProjector cached
* FastTomo run_sirt (no Adam/optimizer — pure SIRT)
* FastTomo run_sirt with ParallelProjector (cached SIRT)
* FastTomo run_resire (projected gradient, RESIRE 4-pt scatter projector)
* FastTomo run_genfire (Fourier gridding + real-space constraint iteration)
* ASTRA SIRT_CUDA with MinConstraint=0
* TorchRadon iterative-FBP (per ``sirt_recon_ex.py``) with positivity

AD methods are 1000-iter; SIRT-style methods default to 100 iterations
(override with --sirt-iter).

Usage:
    conda run -n FastTomo-bench python -m benchmarks.run_iterative
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch

import fasttomo as ft
from benchmarks.scripts.utils.ad_reconstruct import (
    ad_reconstruct,
    ad_reconstruct_cached,
    ad_reconstruct_with_pose,
    ad_reconstruct_with_pose_cached,
)
from benchmarks.scripts.utils.datasets import load_phantom_tilt_series
from benchmarks.scripts.utils.metrics import nmse, ssim
from benchmarks.scripts.utils.timing import cuda_timer


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        print(f"  [skipped: {name} unavailable: {type(e).__name__}: {e}]")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ad-iter", type=int, default=1000)
    parser.add_argument("--sirt-iter", type=int, default=1000)
    parser.add_argument("--resire-iter", type=int, default=None,
                        help="iterations for run_resire (default: --sirt-iter)")
    parser.add_argument("--genfire-iter", type=int, default=None,
                        help="iterations for run_genfire (default: --sirt-iter)")
    parser.add_argument("--tv-weight", type=float, default=1.0)
    parser.add_argument("--skip-astra", action="store_true")
    # TorchRadon's iterative-FBP loop semi-converges on missing-wedge data: per
    # wiki/analyses/torch-radon-sirt-semi-convergence.md the SSIM peak is iter 1
    # with sigma=2.0, step=0.05 (SSIM 0.857). 1000 iter at the same hyperparams
    # would decay to ~0.74 — not a fair comparison.
    parser.add_argument("--tr-iter", type=int, default=10,
                        help="iterations for TorchRadon iterative-FBP (default: 10; "
                             "the SSIM peak is iter 1 at 0.857, iter 10 is 0.831 — "
                             "10 is used so it/s isn't dominated by warm-up overhead)")
    parser.add_argument(
        "--tr-smoothing-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma for TorchRadon SIRT (per sirt_recon_ex.py). "
        "Pass 0 to disable.",
    )
    parser.add_argument("--tr-step", type=float, default=0.05,
                        help="step size for TorchRadon iterative-FBP")
    args = parser.parse_args()

    device = torch.device(args.device)
    phantom, sino, ang_deg = load_phantom_tilt_series(device=device)
    n_angles, Ny, N_det = sino.shape
    print(f"sinogram {(n_angles, Ny, N_det)}, phantom {tuple(phantom.shape)}")

    ang_rad = (ang_deg.cpu().numpy() * np.pi / 180.0).astype(np.float32)
    sino_np = sino.cpu().numpy()

    rows: list[dict] = []

    def record(name: str, t: float, rec: torch.Tensor | np.ndarray, n_iter: int) -> None:
        if isinstance(rec, np.ndarray):
            rec = torch.from_numpy(rec)
        rows.append({
            "method": name,
            "time_s": t,
            "n_iter": n_iter,
            "ssim": ssim(rec, phantom),
            "nmse": nmse(rec, phantom),
            "min": float(rec.min()),
            "max": float(rec.max()),
        })
        print(
            f"  done in {t:.2f}s, range [{float(rec.min()):.3f}, {float(rec.max()):.3f}], "
            f"SSIM={rows[-1]['ssim']:.4f}"
        )

    # --- FastTomo AD without pose (uncached + cached) ---
    print(f"\n[FastTomo] AD no-pose (n_iter={args.ad_iter})")
    with cuda_timer(device) as t:
        res_no = ad_reconstruct(sino, ang_deg, n_iter=args.ad_iter, tv_weight=args.tv_weight)
    record("ft-ad-no-pose", t[0], res_no.volume, args.ad_iter)

    print(f"\n[FastTomo] AD no-pose CACHED (n_iter={args.ad_iter})")
    with cuda_timer(device) as t:
        res_no_c = ad_reconstruct_cached(
            sino, ang_deg, n_iter=args.ad_iter, tv_weight=args.tv_weight
        )
    record("ft-ad-no-pose-cached", t[0], res_no_c.volume, args.ad_iter)

    # --- FastTomo AD with pose (uncached + cached) ---
    print(f"\n[FastTomo] AD with-pose (n_iter={args.ad_iter})")
    with cuda_timer(device) as t:
        res_yes = ad_reconstruct_with_pose(
            sino, ang_deg, n_iter=args.ad_iter, tv_weight=args.tv_weight
        )
    record("ft-ad-with-pose", t[0], res_yes.volume, args.ad_iter)
    refined = res_yes.euler_zyz_deg.cpu()
    theta_err = (refined[:, 1] - ang_deg.cpu()).abs().max().item()
    print(
        f"  pose recovery: max |Δθ|={theta_err:.3f}°, "
        f"max |φ|={refined[:, 0].abs().max():.3f}°, "
        f"max |ψ|={refined[:, 2].abs().max():.3f}°, "
        f"max |shift|={res_yes.shifts.abs().max():.3f}px"
    )

    print(f"\n[FastTomo] AD with-pose CACHED (n_iter={args.ad_iter})")
    with cuda_timer(device) as t:
        res_yes_c = ad_reconstruct_with_pose_cached(
            sino, ang_deg, n_iter=args.ad_iter, tv_weight=args.tv_weight
        )
    record("ft-ad-with-pose-cached", t[0], res_yes_c.volume, args.ad_iter)
    refined_c = res_yes_c.euler_zyz_deg.cpu()
    theta_err_c = (refined_c[:, 1] - ang_deg.cpu()).abs().max().item()
    print(
        f"  pose recovery: max |Δθ|={theta_err_c:.3f}°, "
        f"max |φ|={refined_c[:, 0].abs().max():.3f}°, "
        f"max |ψ|={refined_c[:, 2].abs().max():.3f}°, "
        f"max |shift|={res_yes_c.shifts.abs().max():.3f}px"
    )

    # --- FastTomo run_sirt (uncached) — no torch optimizer, pure SIRT ---
    print(f"\n[FastTomo] run_sirt (n_iter={args.sirt_iter}, positivity)")
    _ = ft.run_sirt(sino[:, :1, :], ang_deg, n_iter=2, positivity=True)  # warm-up
    with cuda_timer(device) as t:
        rec_sirt = ft.run_sirt(
            sino, ang_deg, n_iter=args.sirt_iter, positivity=True
        )
    record("ft-sirt", t[0], rec_sirt, args.sirt_iter)

    # --- FastTomo run_sirt with ParallelProjector (cached) ---
    print(f"\n[FastTomo] run_sirt CACHED (n_iter={args.sirt_iter}, positivity)")
    proj_sirt = ft.ParallelProjector(
        (Ny, N_det, N_det), ang_deg, device=device, N_det=N_det
    )
    _ = ft.run_sirt(
        sino, ang_deg, n_iter=2, positivity=True, projector=proj_sirt
    )  # warm-up
    with cuda_timer(device) as t:
        rec_sirt_c = ft.run_sirt(
            sino, ang_deg, n_iter=args.sirt_iter, positivity=True,
            projector=proj_sirt,
        )
    record("ft-sirt-cached", t[0], rec_sirt_c, args.sirt_iter)

    # --- FastTomo run_resire (RESIRE 4-pt scatter projector, projected GD) ---
    resire_iter = args.resire_iter if args.resire_iter is not None else args.sirt_iter
    print(f"\n[FastTomo] run_resire (n_iter={resire_iter}, positivity)")
    _ = ft.run_resire(sino[:, :1, :], ang_deg, n_iter=2, positivity=True)  # warm-up
    with cuda_timer(device) as t:
        rec_resire = ft.run_resire(
            sino, ang_deg, n_iter=resire_iter, positivity=True
        )
    record("ft-resire", t[0], rec_resire, resire_iter)

    # --- FastTomo run_genfire (Fourier gridding + IFFT/FFT iteration) ---
    genfire_iter = args.genfire_iter if args.genfire_iter is not None else args.sirt_iter
    print(f"\n[FastTomo] run_genfire (n_iter={genfire_iter}, positivity, mask)")
    _ = ft.run_genfire(sino[:, :1, :], ang_deg, n_iter=2,
                       positivity=True, mask=True)  # warm-up
    with cuda_timer(device) as t:
        rec_genfire = ft.run_genfire(
            sino, ang_deg, n_iter=genfire_iter,
            positivity=True, mask=True, resolution_extension=True,
        )
    record("ft-genfire", t[0], rec_genfire, genfire_iter)

    # --- ASTRA SIRT ---
    print(f"\n[ASTRA] SIRT_CUDA (num_iter={args.sirt_iter}, MinConstraint=0)")
    if not args.skip_astra and _try_import("astra") is not None:
        from benchmarks.scripts.utils.astra_wrapper import AstraParallelBeam
        astra_w = AstraParallelBeam(N_det, ang_rad)
        _ = astra_w.sirt(sino_np[:, :1, :], num_iter=2)  # warm-up (2 iters, single slice)
        with cuda_timer(device) as t:
            rec_a = astra_w.sirt(sino_np, num_iter=args.sirt_iter, positivity=True)
        record("astra-sirt", t[0], rec_a, args.sirt_iter)

    # --- TorchRadon iterative-FBP with positivity ---
    sigma = args.tr_smoothing_sigma if args.tr_smoothing_sigma > 0 else None
    sigma_tag = f"sigma={sigma}" if sigma is not None else "no-smoothing"
    print(
        f"\n[TorchRadon] iterative-FBP (num_iter={args.tr_iter}, step={args.tr_step}, "
        f"positivity, {sigma_tag})"
    )
    if _try_import("torch_radon") is not None:
        from benchmarks.scripts.utils.torch_radon_wrapper import TorchRadonParallelBeam
        tr_w = TorchRadonParallelBeam(N_det, ang_rad)
        _ = tr_w.sirt(
            sino[:, :1, :], num_iter=2, step_size=args.tr_step, smoothing_sigma=sigma,
        )  # warm-up
        with cuda_timer(device) as t:
            rec_tr = tr_w.sirt(
                sino,
                num_iter=args.tr_iter,
                step_size=args.tr_step,
                positivity=True,
                smoothing_sigma=sigma,
            )
        record("torch_radon-sirt", t[0], rec_tr, args.tr_iter)

    # --- table ---
    print("\nResults:")
    print(
        f"  {'method':<24} {'n_iter':>7} {'time_s':>9} {'it/s':>7} "
        f"{'ssim':>8} {'nmse':>10} {'min':>7} {'max':>7}"
    )
    for r in rows:
        ips = r["n_iter"] / r["time_s"]
        print(
            f"  {r['method']:<24} {r['n_iter']:>7} {r['time_s']:>9.2f} {ips:>7.1f} "
            f"{r['ssim']:>8.4f} {r['nmse']:>10.4e} {r['min']:>7.3f} {r['max']:>7.3f}"
        )


if __name__ == "__main__":
    main()
