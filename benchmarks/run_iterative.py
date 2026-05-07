"""Stage 4 cross-backend iterative comparison.

Four methods, all reconstructing the same 1°-tilt-axis-error tilt series with
positivity enforced. SSIM/NMSE vs the ground-truth phantom.

* FastTomo AD without pose optimization
* FastTomo AD with pose optimization (Euler + shifts learnable)
* ASTRA SIRT_CUDA with MinConstraint=0
* TorchRadon iterative-FBP (per ``sirt_recon_ex.py``) with positivity

The two AD methods are 1000-iter; the two SIRT-style methods default to 100
iterations each (override with --sirt-iter).

Usage:
    conda run -n FastTomo-bench python -m benchmarks.run_iterative
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch

from benchmarks.ad_reconstruct import ad_reconstruct, ad_reconstruct_with_pose
from benchmarks.datasets import load_phantom_tilt_series
from benchmarks.metrics import nmse, ssim
from benchmarks.timing import cuda_timer


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
    parser.add_argument("--sirt-iter", type=int, default=100)
    parser.add_argument("--tv-weight", type=float, default=1.0)
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

    # --- FastTomo AD without pose ---
    print(f"\n[FastTomo] AD no-pose (n_iter={args.ad_iter})")
    with cuda_timer(device) as t:
        res_no = ad_reconstruct(sino, ang_deg, n_iter=args.ad_iter, tv_weight=args.tv_weight)
    record("ft-ad-no-pose", t[0], res_no.volume, args.ad_iter)

    # --- FastTomo AD with pose ---
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

    # --- ASTRA SIRT ---
    print(f"\n[ASTRA] SIRT_CUDA (num_iter={args.sirt_iter}, MinConstraint=0)")
    if _try_import("astra") is not None:
        from benchmarks.astra_wrapper import AstraParallelBeam
        astra_w = AstraParallelBeam(N_det, ang_rad)
        _ = astra_w.sirt(sino_np[:, :1, :], num_iter=2)  # warm-up (2 iters, single slice)
        with cuda_timer(device) as t:
            rec_a = astra_w.sirt(sino_np, num_iter=args.sirt_iter, positivity=True)
        record("astra-sirt", t[0], rec_a, args.sirt_iter)

    # --- TorchRadon iterative-FBP with positivity ---
    print(f"\n[TorchRadon] iterative-FBP (num_iter={args.sirt_iter}, positivity)")
    if _try_import("torch_radon") is not None:
        from benchmarks.torch_radon_wrapper import TorchRadonParallelBeam
        tr_w = TorchRadonParallelBeam(N_det, ang_rad)
        _ = tr_w.sirt(sino[:, :1, :], num_iter=2)  # warm-up
        with cuda_timer(device) as t:
            rec_tr = tr_w.sirt(sino, num_iter=args.sirt_iter, positivity=True)
        record("torch_radon-sirt", t[0], rec_tr, args.sirt_iter)

    # --- table ---
    print("\nResults:")
    print(
        f"  {'method':<18} {'n_iter':>7} {'time_s':>9} {'it/s':>7} "
        f"{'ssim':>8} {'nmse':>10} {'min':>7} {'max':>7}"
    )
    for r in rows:
        ips = r["n_iter"] / r["time_s"]
        print(
            f"  {r['method']:<18} {r['n_iter']:>7} {r['time_s']:>9.2f} {ips:>7.1f} "
            f"{r['ssim']:>8.4f} {r['nmse']:>10.4e} {r['min']:>7.3f} {r['max']:>7.3f}"
        )


if __name__ == "__main__":
    main()
