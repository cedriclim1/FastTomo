"""Extended-iteration probe for AD-TV-S3IM.

Question: does AD-TV-S3IM keep improving past the 1500-iter cap used in the
main Poisson sweep, or does the val MSE bottom and climb (= noise overfit)?

We re-run the best AD-TV-S3IM recipe (tv=100, λ_S3IM=3) at n_iter=3000 on:
  - clean quantem sinogram
  - Poisson peak=50 quantem sinogram

…with held-out projection CV (val on 3 of 29 angles). Logs train/val MSE +
SSIM-3D every 20 iters, reports early-stop pick vs final, and emits a slice
comparison at slice_idx=125.

Outputs (benchmarks/results/ad_s3im_long/):
  - curves_clean.png        — train/val MSE + SSIM vs iter (clean sino)
  - curves_poisson.png      — same (Poisson peak=50)
  - slices.png              — slice_idx=125 grid: clean ES, clean final,
                              poisson ES, poisson final, GT
  - results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim

import fasttomo as ft

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SINO_PATH = REPO_ROOT / "data" / "quantem_phantom_perfect_sino.npy"
QUANTEM   = Path("/home/cedlim/quantem/quantem-tutorials/data") / "phantom.npy"
OUT_DIR   = REPO_ROOT / "benchmarks" / "results" / "ad_s3im_long"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = torch.device("cuda")
SLICE_IDX   = 125
PEAK_COUNTS = 50
N_ITER      = 3000
LR          = 0.05
EVAL_EVERY  = 20
SEED        = 0
VAL_IDX     = (5, 14, 23)

# Best recipe from the prior CV sweep (tune_ad_recipes_poisson.py)
TV_WEIGHT     = 100.0
S3IM_LAMBDA   = 3.0


def load_phantom() -> torch.Tensor:
    arr = np.load(QUANTEM).astype(np.float32)
    arr = arr.transpose(2, 0, 1)[::-1].copy()
    return torch.from_numpy(arr).to(DEVICE)


def load_perfect_sino() -> torch.Tensor:
    return torch.from_numpy(np.load(SINO_PATH).astype(np.float32)).to(DEVICE)


def poisson(sino: torch.Tensor, peak: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device=sino.device).manual_seed(seed)
    scale = peak / float(sino.max())
    return torch.poisson((sino * scale).clamp(min=0.0), generator=g) / scale


def split_train_val(sino, angles, val_idx):
    n = sino.shape[0]
    vm = torch.zeros(n, dtype=torch.bool)
    for i in val_idx:
        vm[i] = True
    tm = ~vm
    return sino[tm], angles[tm], sino[vm], angles[vm]


def ssim3d(vol, gt):
    v = vol.detach().cpu().numpy()
    g = gt.detach().cpu().numpy()
    dr = float(g.max() - g.min())
    return float(sk_ssim(
        v, g, data_range=dr, win_size=7,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    ))


def train_ad_s3im(sino_tr, ang_tr, sino_va, ang_va, phantom, tag: str):
    n_ang, Ny, N_det = sino_tr.shape
    Nx = N_det
    proj_tr = ft.ParallelProjector((Ny, N_det, Nx), ang_tr, device=DEVICE, N_det=N_det)
    proj_va = ft.ParallelProjector((Ny, N_det, Nx), ang_va, device=DEVICE, N_det=N_det)
    torch.manual_seed(SEED)
    vol = torch.zeros(Ny, N_det, Nx, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([vol], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_ITER)
    g_s3im = torch.Generator(device=DEVICE).manual_seed(SEED + 1)

    log_it, val_mses, ssim_curve, train_losses = [], [], [], []
    best_val = float("inf")
    best_vol_es = None
    best_iter_es = 0
    best_ssim_es = -1.0

    t0 = time.perf_counter()
    for it in range(N_ITER):
        opt.zero_grad(set_to_none=True)
        pred = proj_tr.forward(vol)
        loss = F.mse_loss(pred, sino_tr) + TV_WEIGHT * ft.tv_loss_3d(vol)
        if S3IM_LAMBDA > 0:
            loss = loss + S3IM_LAMBDA * ft.s3im_loss(
                pred, sino_tr,
                patch_pixels=4096, kernel_size=4, n_patches=1, generator=g_s3im,
            )
        train_losses.append(float(loss.item()))
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            vol.clamp_(min=0)
            if (it + 1) % EVAL_EVERY == 0 or it == N_ITER - 1:
                vp = proj_va.forward(vol)
                vm = float(F.mse_loss(vp, sino_va).item())
                ss = ssim3d(vol, phantom)
                log_it.append(it + 1); val_mses.append(vm); ssim_curve.append(ss)
                if vm < best_val:
                    best_val = vm
                    best_vol_es = vol.detach().clone()
                    best_iter_es = it + 1
                    best_ssim_es = ss
    dt = time.perf_counter() - t0
    print(f"[{tag}] {N_ITER} iters in {dt:.1f}s  "
          f"early-stop @ {best_iter_es:4d} (SSIM={best_ssim_es:.4f})  "
          f"final SSIM={ssim_curve[-1]:.4f}  "
          f"val_MSE: es={best_val:.3e}  final={val_mses[-1]:.3e}")
    return {
        "vol_final": vol.detach(),
        "vol_es": best_vol_es,
        "best_iter_es": best_iter_es,
        "ssim_es": best_ssim_es,
        "ssim_final": ssim_curve[-1],
        "val_mse_es": best_val,
        "val_mse_final": val_mses[-1],
        "train_losses": train_losses,
        "log_iters": log_it,
        "val_mses": val_mses,
        "ssim_curve": ssim_curve,
    }


def emit_curves(r, out_path, title):
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.5), sharex=True)
    axes[0].plot(r["train_losses"], color="C0", linewidth=0.7)
    axes[0].set_yscale("log"); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("train loss\n(data + TV + S3IM)")
    axes[0].set_title(title, fontsize=11)

    axes[1].plot(r["log_iters"], r["val_mses"], color="C3", marker="o", markersize=3)
    axes[1].axvline(r["best_iter_es"], color="red", ls="--", lw=1.0,
                    label=f"early-stop iter {r['best_iter_es']}")
    axes[1].set_yscale("log"); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel("held-out val MSE\n(GT-free)")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(r["log_iters"], r["ssim_curve"], color="C2", marker="o", markersize=3)
    axes[2].axvline(r["best_iter_es"], color="red", ls="--", lw=1.0)
    axes[2].scatter([r["best_iter_es"]], [r["ssim_es"]], s=80,
                    facecolors="none", edgecolors="red", linewidths=2.0, zorder=5)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylabel("SSIM-3D vs GT\n(diagnostic)")
    axes[2].set_xlabel("iter")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def emit_slices(panels, gt_slice, out_path, title):
    n = 1 + len(panels)
    ncols = n
    fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 3.0), squeeze=False)
    vmin, vmax = float(gt_slice.min()), float(gt_slice.max())
    items = [("Ground truth", gt_slice, None)] + panels
    for k, (lbl, img, ss) in enumerate(items):
        ax = axes[0, k]
        ax.imshow(img, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if ss is None:
            ax.set_title(lbl, fontsize=9)
        else:
            ax.set_title(f"{lbl}\nssim={ss:.4f}", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading data …")
    phantom = load_phantom()
    perfect = load_perfect_sino()
    noisy = poisson(perfect, peak=PEAK_COUNTS, seed=SEED)
    n_ang = perfect.shape[0]
    angles = torch.linspace(-70.0, 70.0, n_ang, device=DEVICE)

    sino_tr_c, ang_tr, sino_va_c, ang_va = split_train_val(perfect, angles, VAL_IDX)
    sino_tr_n, _,      sino_va_n, _      = split_train_val(noisy,   angles, VAL_IDX)
    print(f"Train: {sino_tr_c.shape[0]} ang  Val: {sino_va_c.shape[0]} ang  "
          f"(held-out {VAL_IDX})")
    print(f"AD-TV-S3IM   tv={TV_WEIGHT}  λ_S3IM={S3IM_LAMBDA}  n_iter={N_ITER}")

    print("\n=== Clean sinogram ===")
    r_clean = train_ad_s3im(sino_tr_c, ang_tr, sino_va_c, ang_va, phantom, "clean")
    print("\n=== Poisson peak=50 ===")
    r_pois  = train_ad_s3im(sino_tr_n, ang_tr, sino_va_n, ang_va, phantom, "pois50")

    emit_curves(r_clean, OUT_DIR / "curves_clean.png",
                f"AD-TV-S3IM long run (clean sino, tv={TV_WEIGHT}, λ_S3IM={S3IM_LAMBDA}, "
                f"n_iter={N_ITER})")
    emit_curves(r_pois,  OUT_DIR / "curves_poisson.png",
                f"AD-TV-S3IM long run (Poisson peak={PEAK_COUNTS}, tv={TV_WEIGHT}, "
                f"λ_S3IM={S3IM_LAMBDA}, n_iter={N_ITER})")

    gt_slice = phantom.detach().cpu().numpy()[SLICE_IDX]
    panels = [
        (f"Clean — early-stop\niter {r_clean['best_iter_es']}",
         r_clean["vol_es"][SLICE_IDX].cpu().numpy(),  r_clean["ssim_es"]),
        (f"Clean — final\niter {N_ITER}",
         r_clean["vol_final"][SLICE_IDX].cpu().numpy(), r_clean["ssim_final"]),
        (f"Poisson — early-stop\niter {r_pois['best_iter_es']}",
         r_pois["vol_es"][SLICE_IDX].cpu().numpy(),    r_pois["ssim_es"]),
        (f"Poisson — final\niter {N_ITER}",
         r_pois["vol_final"][SLICE_IDX].cpu().numpy(), r_pois["ssim_final"]),
    ]
    emit_slices(panels, gt_slice, OUT_DIR / "slices.png",
                f"AD-TV-S3IM long run — slice_idx={SLICE_IDX}  "
                f"(tv={TV_WEIGHT}, λ_S3IM={S3IM_LAMBDA}, n_iter={N_ITER})")

    def _s(r):
        return {k: r[k] for k in ("best_iter_es", "ssim_es", "ssim_final",
                                  "val_mse_es", "val_mse_final")}
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({
            "tv_weight": TV_WEIGHT,
            "s3im_lambda": S3IM_LAMBDA,
            "n_iter": N_ITER,
            "lr": LR,
            "val_idx": list(VAL_IDX),
            "clean":   _s(r_clean),
            "poisson": _s(r_pois),
        }, f, indent=2)

    print(f"\nSaved → {OUT_DIR}")
    print("  curves_clean.png  /  curves_poisson.png  /  slices.png  /  results.json")


if __name__ == "__main__":
    main()
