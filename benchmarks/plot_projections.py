"""Sum-projection grid: rows = axis 0/1/2, columns = each reconstruction method.

Re-runs all fair-benchmark reconstructions on the quantem phantom (perfect
sinogram, no pose optimization) and plots orthogonal sum projections so
reconstruction artefacts are immediately visible.

Usage:
    conda run -n FastTomo-bench python benchmarks/plot_projections.py --device cuda:0
    conda run -n FastTomo-bench python benchmarks/plot_projections.py --device cuda:0 --n-iter 500
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch

import fasttomo as ft
from benchmarks.ad_reconstruct import ad_reconstruct_cached
from benchmarks.datasets import load_fair_tilt_series

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def sum_projections(vol: torch.Tensor) -> list[np.ndarray]:
    """Return [sum_axis0, sum_axis1, sum_axis2] as float32 numpy arrays."""
    v = vol.detach().cpu().float()
    return [v.sum(0).numpy(), v.sum(1).numpy(), v.sum(2).numpy()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--tv-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=100.0)
    parser.add_argument("--tr-iter", type=int, default=10)
    parser.add_argument("--tr-step", type=float, default=0.05)
    parser.add_argument("--tr-smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--out", default=str(FIG_DIR / "fig_projections.png"))
    args = parser.parse_args()

    device = torch.device(args.device)
    phantom, sino, ang_deg = load_fair_tilt_series(device=device)
    n_angles, Ny, N_det = sino.shape
    ang_rad = (ang_deg.cpu().numpy() * np.pi / 180.0).astype(np.float32)
    sino_np = sino.cpu().numpy()
    n_iter = args.n_iter

    # columns: (key, display_label, projections_list)
    columns: list[tuple[str, str, list[np.ndarray]]] = []

    # Ground truth
    columns.append(("gt", "Ground truth", sum_projections(phantom)))

    # FBP
    print("[FastTomo] FBP")
    proj_fbp = ft.ParallelProjector((Ny, N_det, N_det), ang_deg, device=device, N_det=N_det)
    rec = ft.run_fbp(sino, ang_deg, positivity=True, projector=proj_fbp)
    columns.append(("ft-fbp", "FT FBP", sum_projections(rec)))
    del rec

    # SIRT
    print(f"[FastTomo] SIRT ({n_iter} iter)")
    proj_sirt = ft.ParallelProjector((Ny, N_det, N_det), ang_deg, device=device, N_det=N_det)
    _ = ft.run_sirt(sino, ang_deg, n_iter=2, positivity=True, projector=proj_sirt)
    rec = ft.run_sirt(sino, ang_deg, n_iter=n_iter, positivity=True, projector=proj_sirt)
    columns.append(("ft-sirt", f"FT SIRT\n{n_iter}it", sum_projections(rec)))
    del rec

    # SART
    print(f"[FastTomo] SART ({n_iter} iter)")
    _ = ft.run_sart(sino[:, :1, :], ang_deg, n_iter=2, positivity=True)
    rec = ft.run_sart(sino, ang_deg, n_iter=n_iter, subset_size=1, positivity=True)
    columns.append(("ft-sart", f"FT SART\n{n_iter}it", sum_projections(rec)))
    del rec

    # W-SIRT
    print(f"[FastTomo] W-SIRT ({n_iter} iter)")
    _ = ft.run_wsirt(sino[:, :1, :], ang_deg, n_iter=2, positivity=True)
    rec = ft.run_wsirt(sino, ang_deg, n_iter=n_iter, positivity=True)
    columns.append(("ft-wsirt", f"FT W-SIRT\n{n_iter}it", sum_projections(rec)))
    del rec

    # RESIRE
    print(f"[FastTomo] RESIRE ({n_iter} iter)")
    _ = ft.run_resire(sino[:, :1, :], ang_deg, n_iter=2, positivity=True)
    rec = ft.run_resire(sino, ang_deg, n_iter=n_iter, positivity=True)
    columns.append(("ft-resire", f"FT RESIRE\n{n_iter}it", sum_projections(rec)))
    del rec

    # GENFIRE
    print(f"[FastTomo] GENFIRE ({n_iter} iter)")
    _ = ft.run_genfire(sino[:, :1, :], ang_deg, n_iter=2, positivity=True, mask=True)
    rec = ft.run_genfire(sino, ang_deg, n_iter=n_iter, positivity=True, mask=True,
                         resolution_extension=True)
    columns.append(("ft-genfire", f"FT GENFIRE\n{n_iter}it", sum_projections(rec)))
    del rec

    # AD (no regularization)
    print(f"[FastTomo] AD no-reg ({n_iter} iter)")
    res = ad_reconstruct_cached(sino, ang_deg, n_iter=n_iter, tv_weight=0.0)
    columns.append(("ft-ad", f"FT AD\n{n_iter}it", sum_projections(res.volume)))
    del res

    # AD + TV
    print(f"[FastTomo] AD + TV ({n_iter} iter)")
    res = ad_reconstruct_cached(sino, ang_deg, n_iter=n_iter, tv_weight=args.tv_weight)
    columns.append(("ft-ad-tv", f"FT AD+TV\n{n_iter}it", sum_projections(res.volume)))
    del res

    # AD + TV + SSIM
    print(f"[FastTomo] AD + TV + SSIM ({n_iter} iter)")
    res = ad_reconstruct_cached(sino, ang_deg, n_iter=n_iter,
                                tv_weight=args.tv_weight, ssim_weight=args.ssim_weight)
    columns.append(("ft-ad-tv-ssim", f"FT AD+TV\n+SSIM {n_iter}it",
                    sum_projections(res.volume)))
    del res

    # ASTRA
    if _try_import("astra") is not None:
        from benchmarks.astra_wrapper import AstraParallelBeam
        astra_w = AstraParallelBeam(N_det, ang_rad)

        print("[ASTRA] FBP")
        rec = astra_w.fbp(sino_np, positivity=True)
        columns.append(("astra-fbp", "ASTRA FBP",
                         sum_projections(torch.from_numpy(rec))))

        print(f"[ASTRA] SIRT ({n_iter} iter)")
        _ = astra_w.sirt(sino_np[:, :1, :], num_iter=2)
        rec = astra_w.sirt(sino_np, num_iter=n_iter, positivity=True)
        columns.append(("astra-sirt", f"ASTRA SIRT\n{n_iter}it",
                         sum_projections(torch.from_numpy(rec))))

    # TorchRadon
    sigma = args.tr_smoothing_sigma if args.tr_smoothing_sigma > 0 else None
    if _try_import("torch_radon") is not None:
        from benchmarks.torch_radon_wrapper import TorchRadonParallelBeam
        tr_w = TorchRadonParallelBeam(N_det, ang_rad)

        print("[TorchRadon] FBP")
        rec = tr_w.fbp(sino, positivity=True)
        columns.append(("torch_radon-fbp", "TR FBP",
                         sum_projections(rec if isinstance(rec, torch.Tensor)
                                         else torch.from_numpy(rec))))

        print(f"[TorchRadon] iter-FBP ({args.tr_iter} iter)")
        _ = tr_w.sirt(sino[:, :1, :], num_iter=2, step_size=args.tr_step,
                      smoothing_sigma=sigma)
        rec = tr_w.sirt(sino, num_iter=args.tr_iter, step_size=args.tr_step,
                        positivity=True, smoothing_sigma=sigma)
        columns.append(("torch_radon-sirt", f"TR iter-FBP\n{args.tr_iter}it",
                         sum_projections(rec if isinstance(rec, torch.Tensor)
                                         else torch.from_numpy(rec))))

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    import matplotlib.pyplot as plt

    n_cols = len(columns)
    n_rows = 3
    row_labels = ["sum axis 0\n(along tilt axis)", "sum axis 1\n(along depth)",
                  "sum axis 2\n(along lateral)"]

    # Normalise using the ground-truth phantom's projection range so brightness
    # is comparable across all methods and all three views.
    gt_projs = columns[0][2]
    vmin = [p.min() for p in gt_projs]
    vmax = [p.max() for p in gt_projs]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.4 * n_rows),
        squeeze=False,
    )

    for col_idx, (key, label, projs) in enumerate(columns):
        for row_idx in range(n_rows):
            ax = axes[row_idx][col_idx]
            ax.imshow(projs[row_idx], cmap="turbo",
                      vmin=0, vmax=vmax[row_idx],
                      aspect="equal", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(label, fontsize=7, pad=3)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=7)

    fig.suptitle(
        "Sum projections — perfect tilt series, no pose optimization",
        fontsize=9, y=1.005,
    )
    fig.tight_layout(pad=0.4, h_pad=0.3, w_pad=0.3)

    out = Path(args.out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
