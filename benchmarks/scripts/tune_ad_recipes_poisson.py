"""Hyperparameter search for AD-TV / AD-TV-S3IM on the Poisson-noised quantem
sinogram (peak=50). Uses held-out projection cross-validation to detect when
the reconstruction starts fitting noise — gives a principled early-stopping
signal independent of the ground-truth phantom.

The script does three things per recipe:

  1. Splits the 29 projections into train (26) + val (3, spread across tilt).
  2. Sweeps tv_weight (wider than the main comparison did — picks up the right
     edge that the main sweep was still rising at).
  3. For each setting, runs AD for 1500 iters and tracks:
       - train data-fit MSE (the loss being minimised)
       - val data-fit MSE (held-out projection MSE — GT-free overfit detector)
       - SSIM-3D vs GT every 20 iters (diagnostic only)
     Reports both the **early-stopped** SSIM (at iter where val MSE bottoms
     out) and the **final** SSIM at iter 1500 (what the main script would have
     gotten).

Outputs (benchmarks/results/ad_tune_poisson_peak50/):
  - ad_tv_curves.png         — train/val MSE + SSIM vs iter for each tv
  - ad_tv_s3im_curves.png    — same for the AD-TV-S3IM family
  - slice_comparison.png     — slice_idx=125 for early-stopped vs final
                                across all best settings
  - results.json             — every (tv, λ) point's early/final metrics
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
OUT_DIR   = REPO_ROOT / "benchmarks" / "results" / "ad_tune_poisson_peak50"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = torch.device("cuda")
SLICE_IDX    = 125
PEAK_COUNTS  = 50
N_ITER       = 1500
LR           = 0.05
EVAL_EVERY   = 20            # how often to log val MSE + SSIM
SEED         = 0

# Held-out projection indices (evenly spread across the ±70° tilt range)
VAL_IDX = (5, 14, 23)

TV_SWEEP_AD       = [30.0, 100.0, 300.0, 1000.0, 3000.0]
S3IM_LAMBDA_SWEEP = [3.0, 10.0, 30.0, 100.0, 300.0]


# ── data prep ────────────────────────────────────────────────────────────

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


def split_train_val(
    sino: torch.Tensor, angles: torch.Tensor, val_idx: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_angles = sino.shape[0]
    val_mask = torch.zeros(n_angles, dtype=torch.bool)
    for i in val_idx:
        val_mask[i] = True
    train_mask = ~val_mask
    return (
        sino[train_mask], angles[train_mask],
        sino[val_mask],   angles[val_mask],
    )


def nmse(vol: torch.Tensor, gt: torch.Tensor) -> float:
    return float(((vol - gt) ** 2).sum() / (gt ** 2).sum())


def ssim3d(vol: torch.Tensor, gt: torch.Tensor) -> float:
    v = vol.detach().cpu().numpy()
    g = gt.detach().cpu().numpy()
    drange = float(g.max() - g.min())
    return float(sk_ssim(
        v, g, data_range=drange, win_size=7,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    ))


# ── AD training loop with CV tracking ────────────────────────────────────

def train_ad_with_cv(
    sino_train: torch.Tensor, angles_train: torch.Tensor,
    sino_val: torch.Tensor,   angles_val: torch.Tensor,
    phantom: torch.Tensor,
    *,
    tv_weight: float,
    s3im_weight: float = 0.0,
    n_iter: int = N_ITER,
    lr: float = LR,
    eval_every: int = EVAL_EVERY,
    seed: int = SEED,
):
    """Run AD with per-step train loss + periodic (val MSE, SSIM) tracking.

    Saves the volume snapshot at the iter where val MSE is minimal — that's
    the early-stopping pick. Also returns the final volume at the end."""
    n_angles, Ny, N_det = sino_train.shape
    Nx = N_det
    proj_train = ft.ParallelProjector(
        (Ny, N_det, Nx), angles_train, device=sino_train.device, N_det=N_det,
    )
    proj_val = ft.ParallelProjector(
        (Ny, N_det, Nx), angles_val, device=sino_train.device, N_det=N_det,
    )
    torch.manual_seed(seed)
    vol = torch.zeros(Ny, N_det, Nx, device=sino_train.device, requires_grad=True)
    opt = torch.optim.Adam([vol], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter)
    s3im_gen = torch.Generator(device=sino_train.device).manual_seed(seed + 1)

    train_losses: list[float] = []
    log_iters: list[int]    = []
    val_mses:   list[float] = []
    ssim_curve: list[float] = []

    best_val = float("inf")
    best_vol_es: torch.Tensor | None = None
    best_iter_es = 0
    best_ssim_at_es = -1.0

    for it in range(n_iter):
        opt.zero_grad(set_to_none=True)
        pred = proj_train.forward(vol)
        loss = F.mse_loss(pred, sino_train)
        if tv_weight > 0:
            loss = loss + tv_weight * ft.tv_loss_3d(vol)
        if s3im_weight > 0:
            loss = loss + s3im_weight * ft.s3im_loss(
                pred, sino_train,
                patch_pixels=4096, kernel_size=4, n_patches=1,
                generator=s3im_gen,
            )
        train_losses.append(float(loss.item()))
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            vol.clamp_(min=0)

            if (it + 1) % eval_every == 0 or it == n_iter - 1:
                val_pred = proj_val.forward(vol)
                vm = float(F.mse_loss(val_pred, sino_val).item())
                ss = ssim3d(vol, phantom)
                log_iters.append(it + 1)
                val_mses.append(vm)
                ssim_curve.append(ss)
                if vm < best_val:
                    best_val = vm
                    best_vol_es = vol.detach().clone()
                    best_iter_es = it + 1
                    best_ssim_at_es = ss

    return {
        "vol_final": vol.detach(),
        "vol_es":    best_vol_es,
        "best_iter_es": best_iter_es,
        "ssim_es":   best_ssim_at_es,
        "ssim_final": ssim_curve[-1],
        "val_mse_es":   best_val,
        "val_mse_final": val_mses[-1],
        "train_losses": train_losses,
        "log_iters":  log_iters,
        "val_mses":   val_mses,
        "ssim_curve": ssim_curve,
    }


# ── plotting ─────────────────────────────────────────────────────────────

def emit_curves(
    runs: dict[str, dict], out_path: Path, title: str,
) -> None:
    """Per-tv: train MSE / val MSE / SSIM-3D vs iter, with early-stop marker."""
    n = len(runs)
    fig, axes = plt.subplots(3, n, figsize=(3.6 * n, 8.5), sharex=True, squeeze=False)
    for j, (label, r) in enumerate(runs.items()):
        # train loss
        ax = axes[0, j]
        ax.plot(r["train_losses"], color="C0", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("train loss\n(data-fit + reg)")

        # val MSE (held-out projection MSE)
        ax = axes[1, j]
        ax.plot(r["log_iters"], r["val_mses"], color="C3", marker="o", markersize=3)
        ax.axvline(r["best_iter_es"], color="red", linestyle="--", linewidth=1.0,
                   label=f"early-stop iter {r['best_iter_es']}")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("held-out val MSE\n(GT-free)")

        # SSIM-3D (diagnostic; uses GT)
        ax = axes[2, j]
        ax.plot(r["log_iters"], r["ssim_curve"], color="C2", marker="o", markersize=3)
        ax.axvline(r["best_iter_es"], color="red", linestyle="--", linewidth=1.0)
        ax.scatter([r["best_iter_es"]], [r["ssim_es"]], s=80,
                   facecolors="none", edgecolors="red", linewidths=2.0, zorder=5)
        ax.set_xlabel("iter")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("SSIM-3D vs GT\n(diagnostic only)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def emit_slice_comparison(
    panels: list[tuple[str, np.ndarray, float]],
    gt_slice: np.ndarray, out_path: Path, title: str,
) -> None:
    n = 1 + len(panels)
    n_cols = min(6, n)
    n_rows = (n + n_cols - 1) // n_cols
    vmin, vmax = float(gt_slice.min()), float(gt_slice.max())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows), squeeze=False)
    items = [("Ground truth", gt_slice, None)] + panels
    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]
        if k >= len(items):
            ax.axis("off")
            continue
        label, img, ssim_val = items[k]
        ax.imshow(img, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if ssim_val is None:
            ax.set_title(label, fontsize=9)
        else:
            ax.set_title(f"{label}\nssim={ssim_val:.4f}", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    phantom = load_phantom()
    perfect = load_perfect_sino()
    sino = poisson(perfect, peak=PEAK_COUNTS, seed=SEED)
    n_angles = sino.shape[0]
    angles = torch.linspace(-70.0, 70.0, n_angles, device=DEVICE)

    sino_tr, ang_tr, sino_va, ang_va = split_train_val(sino, angles, VAL_IDX)
    print(f"Train: {sino_tr.shape[0]} angles  Val: {sino_va.shape[0]} angles "
          f"(held-out indices {VAL_IDX})")

    # ── AD-TV sweep ──────────────────────────────────────────────────────
    print(f"\n=== AD-TV sweep (tv ∈ {TV_SWEEP_AD}, n_iter={N_ITER}) ===")
    ad_tv_runs: dict[str, dict] = {}
    for tv in TV_SWEEP_AD:
        print(f"\n→ tv={tv:g}")
        t0 = time.perf_counter()
        r = train_ad_with_cv(
            sino_tr, ang_tr, sino_va, ang_va, phantom,
            tv_weight=tv, s3im_weight=0.0,
        )
        dt = time.perf_counter() - t0
        print(f"  early-stop @ iter {r['best_iter_es']:4d}  "
              f"SSIM_es={r['ssim_es']:.4f}  SSIM_final={r['ssim_final']:.4f}  "
              f"val_MSE_es={r['val_mse_es']:.3e}  ({dt:.1f}s)")
        ad_tv_runs[f"tv={tv:g}"] = r

    # ── AD-TV-S3IM sweep at AD-TV best ───────────────────────────────────
    best_tv = TV_SWEEP_AD[int(np.argmax([r["ssim_es"] for r in ad_tv_runs.values()]))]
    print(f"\n=== AD-TV-S3IM sweep at tv={best_tv:g} (λ_S3IM ∈ {S3IM_LAMBDA_SWEEP}) ===")
    ad_s3im_runs: dict[str, dict] = {}
    for lam in S3IM_LAMBDA_SWEEP:
        print(f"\n→ tv={best_tv:g}  λ_S3IM={lam:g}")
        t0 = time.perf_counter()
        r = train_ad_with_cv(
            sino_tr, ang_tr, sino_va, ang_va, phantom,
            tv_weight=best_tv, s3im_weight=lam,
        )
        dt = time.perf_counter() - t0
        print(f"  early-stop @ iter {r['best_iter_es']:4d}  "
              f"SSIM_es={r['ssim_es']:.4f}  SSIM_final={r['ssim_final']:.4f}  "
              f"val_MSE_es={r['val_mse_es']:.3e}  ({dt:.1f}s)")
        ad_s3im_runs[f"λ={lam:g}"] = r

    # ── reference: FISTA-TV at the same tv=10 from main script ───────────
    print("\n=== Reference: FISTA-TV at tv=10 (full sino, for context) ===")
    fista_tv_vol = ft.run_fista_tv(
        sino, angles, tv_weight=10.0, n_iter=100,
        tv_inner="fgp", tv_inner_iter=20,
        positivity=True, mask=True,
    )
    fista_ssim = ssim3d(fista_tv_vol, phantom)
    print(f"  FISTA-TV SSIM_3D={fista_ssim:.4f}")

    # ── save curves & slice comparison ───────────────────────────────────
    emit_curves(ad_tv_runs, OUT_DIR / "ad_tv_curves.png",
                title=f"AD-TV — train/val MSE + SSIM-3D vs iter "
                      f"(Poisson peak={PEAK_COUNTS})")
    emit_curves(ad_s3im_runs, OUT_DIR / "ad_tv_s3im_curves.png",
                title=f"AD-TV-S3IM at tv={best_tv:g} — train/val MSE + SSIM-3D vs iter "
                      f"(Poisson peak={PEAK_COUNTS})")

    gt_slice = phantom.detach().cpu().numpy()[SLICE_IDX]
    panels: list[tuple[str, np.ndarray, float]] = []
    # Best AD-TV at early stop
    best_ad_tv = max(ad_tv_runs.items(), key=lambda kv: kv[1]["ssim_es"])
    panels.append((f"AD-TV {best_ad_tv[0]}\nearly-stop iter {best_ad_tv[1]['best_iter_es']}",
                   best_ad_tv[1]["vol_es"][SLICE_IDX].cpu().numpy(),
                   best_ad_tv[1]["ssim_es"]))
    # Same tv, no early stop (final iter)
    panels.append((f"AD-TV {best_ad_tv[0]}\nfinal iter {N_ITER}",
                   best_ad_tv[1]["vol_final"][SLICE_IDX].cpu().numpy(),
                   best_ad_tv[1]["ssim_final"]))
    # Best AD-TV-S3IM at early stop
    best_s3im = max(ad_s3im_runs.items(), key=lambda kv: kv[1]["ssim_es"])
    panels.append((f"AD-TV-S3IM tv={best_tv:g}, {best_s3im[0]}\n"
                   f"early-stop iter {best_s3im[1]['best_iter_es']}",
                   best_s3im[1]["vol_es"][SLICE_IDX].cpu().numpy(),
                   best_s3im[1]["ssim_es"]))
    panels.append((f"AD-TV-S3IM tv={best_tv:g}, {best_s3im[0]}\nfinal iter {N_ITER}",
                   best_s3im[1]["vol_final"][SLICE_IDX].cpu().numpy(),
                   best_s3im[1]["ssim_final"]))
    panels.append(("FISTA-TV tv=10\n(reference, no CV needed)",
                   fista_tv_vol[SLICE_IDX].cpu().numpy(), fista_ssim))

    emit_slice_comparison(
        panels, gt_slice, OUT_DIR / "slice_comparison.png",
        title=f"Best AD-TV / AD-TV-S3IM after tuning + CV early stopping "
              f"(Poisson peak={PEAK_COUNTS}, slice_idx={SLICE_IDX})",
    )

    # ── persist results ──────────────────────────────────────────────────
    def _scrape(r: dict) -> dict:
        return {k: v for k, v in r.items()
                if k in ("best_iter_es", "ssim_es", "ssim_final",
                         "val_mse_es", "val_mse_final")}

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({
            "ad_tv":      {k: _scrape(r) for k, r in ad_tv_runs.items()},
            "ad_tv_s3im": {k: _scrape(r) for k, r in ad_s3im_runs.items()},
            "fista_tv_reference_ssim": fista_ssim,
            "best_tv": best_tv,
            "best_s3im_lambda": float(best_s3im[0].split("=")[1]),
        }, f, indent=2)

    print(f"\nSaved → {OUT_DIR}")
    print("  ad_tv_curves.png  /  ad_tv_s3im_curves.png  /  slice_comparison.png  /  results.json")


if __name__ == "__main__":
    main()
