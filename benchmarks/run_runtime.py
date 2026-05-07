"""Stage 1 runtime benchmark: AD reconstruction with vs without pose optimization.

Run on the phantom dataset (200³ phantom, 29-angle tilt series with a 1° tilt-axis
error baked in). Reports total wall time, time-per-iteration, final loss, and
SSIM/NMSE vs the ground-truth phantom for each configuration.

Usage:
    conda run -n FastTomo-bench python -m benchmarks.run_runtime
    conda run -n FastTomo-bench python -m benchmarks.run_runtime --device cuda --n-iter 1000
"""

from __future__ import annotations

import argparse

import torch

from benchmarks.ad_reconstruct import ad_reconstruct, ad_reconstruct_with_pose
from benchmarks.datasets import load_phantom_tilt_series
from benchmarks.metrics import nmse, ssim
from benchmarks.timing import cuda_timer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--tv-weight", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    phantom, sinogram, angles_deg = load_phantom_tilt_series(device=device)
    print(
        f"Loaded phantom {tuple(phantom.shape)}, sinogram {tuple(sinogram.shape)}, "
        f"angles {tuple(angles_deg.shape)}"
    )

    rows: list[dict] = []

    # --- without pose optimization ---
    print(f"\n[1/2] AD reconstruction (no pose, n_iter={args.n_iter})")
    with cuda_timer(device) as t:
        res_no = ad_reconstruct(
            sinogram, angles_deg, n_iter=args.n_iter, tv_weight=args.tv_weight
        )
    rows.append({
        "config": "no-pose",
        "total_s": t[0],
        "iter_per_s": args.n_iter / t[0],
        "final_loss": res_no.losses[-1],
        "ssim": ssim(res_no.volume, phantom),
        "nmse": nmse(res_no.volume, phantom),
    })
    print(f"  done in {t[0]:.2f}s ({args.n_iter / t[0]:.1f} it/s)")

    # --- with pose optimization ---
    print(f"\n[2/2] AD reconstruction (with pose, n_iter={args.n_iter})")
    with cuda_timer(device) as t:
        res_yes = ad_reconstruct_with_pose(
            sinogram, angles_deg, n_iter=args.n_iter, tv_weight=args.tv_weight
        )
    rows.append({
        "config": "with-pose",
        "total_s": t[0],
        "iter_per_s": args.n_iter / t[0],
        "final_loss": res_yes.losses[-1],
        "ssim": ssim(res_yes.volume, phantom),
        "nmse": nmse(res_yes.volume, phantom),
    })
    print(f"  done in {t[0]:.2f}s ({args.n_iter / t[0]:.1f} it/s)")

    # --- pose recovery sanity check ---
    refined = res_yes.euler_zyz_deg.cpu()
    theta_err = (refined[:, 1] - angles_deg.cpu()).abs().max().item()
    phi_max = refined[:, 0].abs().max().item()
    psi_max = refined[:, 2].abs().max().item()
    shift_max = res_yes.shifts.abs().max().item()
    print(
        "\nPose recovery: max |Δθ|={:.3f}°, max |φ|={:.3f}°, max |ψ|={:.3f}°, "
        "max |shift|={:.3f}px".format(theta_err, phi_max, psi_max, shift_max)
    )

    # --- table ---
    print("\nResults:")
    print(f"  {'config':<12} {'total_s':>9} {'it/s':>9} {'final_loss':>12} {'ssim':>8} {'nmse':>10}")
    for r in rows:
        print(
            f"  {r['config']:<12} {r['total_s']:>9.2f} {r['iter_per_s']:>9.1f} "
            f"{r['final_loss']:>12.5f} {r['ssim']:>8.4f} {r['nmse']:>10.4e}"
        )


if __name__ == "__main__":
    main()
