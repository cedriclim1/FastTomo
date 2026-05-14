"""Compare every implemented FastTomo reconstruction algorithm on the perfect
quantem sinogram (29 angles, ±70° step 5°).

Tunes ``tv_weight`` for the TV-dependent algorithms (FISTA-TV, AD-TV) with a
log-grid sweep, picks the per-algorithm best by 3-D SSIM, then runs the full
comparison at the tuned weights. The non-TV algorithms use library defaults
already validated by the surrounding benchmark suite.

Outputs (benchmarks/results/algorithm_compare/):
  - tv_sweep.png        — SSIM-3D vs tv_weight curves
  - tv_sweep.json       — full sweep results
  - axes_grid.png       — 3 rows (Σ-along-Y, Σ-along-Z, Σ-along-X) × N cols (algo)
  - slice125_and_ssim.png — slice_idx=125 row + 3-D SSIM bar plot
  - results.json        — {algo: {nmse, ssim_3d, time_s, tv_weight}}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim

import fasttomo as ft

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
SINO_PATH  = REPO_ROOT / "data" / "quantem_phantom_perfect_sino.npy"
QUANTEM    = Path("/home/cedlim/quantem/quantem-tutorials/data") / "phantom.npy"
OUT_ROOT   = REPO_ROOT / "benchmarks" / "results"

DEVICE = torch.device("cuda")
SLICE_IDX = 125

# Will be set in main() based on CLI args. Module-level so existing helpers
# can keep their current signatures.
OUT_DIR: Path = OUT_ROOT / "algorithm_compare"
NOISE_TAG: str = "perfect"


def add_poisson_noise(sino: torch.Tensor, peak_counts: float, seed: int = 0) -> torch.Tensor:
    """Apply Poisson shot noise scaled so ``sino.max() → peak_counts`` counts.

    Lower ``peak_counts`` → noisier (relative std ≈ 1/√peak_counts).
    Negative or zero values clamp to 0 to keep the Poisson rate non-negative.
    """
    g = torch.Generator(device=sino.device).manual_seed(seed)
    scale = peak_counts / float(sino.max())
    rate = (sino * scale).clamp(min=0.0)
    noisy = torch.poisson(rate, generator=g) / scale
    return noisy


def load_phantom() -> torch.Tensor:
    arr = np.load(QUANTEM).astype(np.float32)
    arr = arr.transpose(2, 0, 1)[::-1].copy()
    return torch.from_numpy(arr).to(DEVICE)


def load_sino() -> torch.Tensor:
    return torch.from_numpy(np.load(SINO_PATH).astype(np.float32)).to(DEVICE)


def nmse(vol: torch.Tensor, gt: torch.Tensor) -> float:
    return float(((vol - gt) ** 2).sum() / (gt ** 2).sum())


def ssim3d(vol: torch.Tensor, gt: torch.Tensor) -> float:
    v = vol.detach().cpu().numpy()
    g = gt.detach().cpu().numpy()
    drange = float(g.max() - g.min())
    if drange == 0.0:
        return 0.0
    return float(sk_ssim(
        v, g, data_range=drange, win_size=7,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    ))


# ── Reconstruction recipes ────────────────────────────────────────────────

def recon_fbp(sino, angles):
    return ft.run_fbp(sino, angles, positivity=True, mask=True)

def recon_wbp(sino, angles):
    return ft.run_weighted_backprojection(sino, angles, positivity=True, mask=True)

def recon_sirt(sino, angles):
    return ft.run_sirt(sino, angles, n_iter=50, positivity=True, mask=True)

def recon_wsirt(sino, angles):
    return ft.run_wsirt(sino, angles, n_iter=50, positivity=True, mask=True)

def recon_sart(sino, angles):
    return ft.run_sart(sino, angles, n_iter=5, positivity=True, mask=True)

def recon_genfire(sino, angles):
    return ft.run_genfire(sino, angles, n_iter=100, positivity=True, mask=True)

def recon_resire(sino, angles):
    return ft.run_resire(sino, angles, n_iter=100, positivity=True, mask=True)

def recon_fista(sino, angles):
    return ft.run_fista(sino, angles, n_iter=100, positivity=True, mask=True)

def recon_fista_tv(sino, angles, tv_weight: float = 1.0):
    return ft.run_fista_tv(
        sino, angles, tv_weight=tv_weight, n_iter=100,
        tv_inner="fgp", tv_inner_iter=20,
        positivity=True, mask=True,
    )

def recon_ad(
    sino, angles, *,
    tv_weight: float = 0.0,
    s3im_weight: float = 0.0,
    s3im_patch_pixels: int = 4096,
    s3im_kernel_size: int = 4,
    s3im_n_patches: int = 1,
    n_iter: int = 1000,
    lr: float = 0.05,
    seed: int = 0,
):
    n_angles, Ny, N_det = sino.shape
    proj = ft.ParallelProjector((Ny, N_det, N_det), angles, device=sino.device, N_det=N_det)
    torch.manual_seed(seed)
    vol = torch.zeros(Ny, N_det, N_det, device=sino.device, requires_grad=True)
    opt = torch.optim.Adam([vol], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter)
    s3im_gen = torch.Generator(device=sino.device).manual_seed(seed + 1)
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        pred = proj.forward(vol)
        loss = F.mse_loss(pred, sino)
        if tv_weight > 0:
            loss = loss + tv_weight * ft.tv_loss_3d(vol)
        if s3im_weight > 0:
            loss = loss + s3im_weight * ft.s3im_loss(
                pred, sino,
                patch_pixels=s3im_patch_pixels,
                kernel_size=s3im_kernel_size,
                n_patches=s3im_n_patches,
                generator=s3im_gen,
            )
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            vol.clamp_(min=0)
    return vol.detach()


TV_SWEEP_FISTA      = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
TV_SWEEP_AD         = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0]
TV_SWEEP_AD_S3IM    = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
S3IM_LAMBDA_INITIAL = 1.0   # baseline used in the TV-weight sweep
S3IM_LAMBDA_SWEEP   = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
# Internal s3im_loss knobs — swept at fixed (tv, λ_S3IM) = (best, 1.0)
S3IM_PATCH_PIXELS_SWEEP = [64, 256, 1024, 4096, 16384]      # √B ∈ {8,16,32,64,128}
S3IM_KERNEL_SIZE_SWEEP  = [2, 4, 8, 16, 32, 64]              # divisors of √4096=64
S3IM_N_PATCHES_SWEEP    = [1, 2, 4, 8, 16]


def build_algorithms(
    best_tv_fista: float,
    best_tv_ad: float,
    best_tv_ad_s3im: float,
    best_lambda_s3im: float,
):
    return [
        ("FBP",         recon_fbp,                                          None),
        ("W-BP",        recon_wbp,                                          None),
        ("SIRT",        recon_sirt,                                         None),
        ("W-SIRT",      recon_wsirt,                                        None),
        ("SART",        recon_sart,                                         None),
        ("GENFIRE",     recon_genfire,                                      None),
        ("RESIRE",      recon_resire,                                       None),
        ("FISTA",       recon_fista,                                        None),
        ("FISTA-TV",    lambda s, a: recon_fista_tv(s, a, tv_weight=best_tv_fista),       best_tv_fista),
        ("AD",          lambda s, a: recon_ad(s, a, tv_weight=0.0),         None),
        ("AD-TV",       lambda s, a: recon_ad(s, a, tv_weight=best_tv_ad),  best_tv_ad),
        ("AD-TV-S3IM",  lambda s, a: recon_ad(
            s, a, tv_weight=best_tv_ad_s3im, s3im_weight=best_lambda_s3im,
        ), best_tv_ad_s3im),
    ]


# ── Figure helpers ────────────────────────────────────────────────────────

def axis_projections(v: np.ndarray):
    return v.sum(axis=0), v.sum(axis=1), v.sum(axis=2)


def emit_axes_grid(
    recons: dict[str, np.ndarray], phantom: np.ndarray, out_path: Path,
    *, algos_per_bank: int = 6,
) -> None:
    """Two stacked banks. Each bank: 3 rows (Σ along each axis) × (1 + K) cols
    (GT then K algos). With ``algos_per_bank=6`` and 12 algos, the figure is
    6 rows × 7 cols — much more square than a single-row 3×13 layout."""
    row_labels = (
        "Σ along Y\n(view ZX)",
        "Σ along Z\n(view YX)",
        "Σ along X\n(view YZ)",
    )
    gt_rows = list(axis_projections(phantom))
    rec_rows = {k: list(axis_projections(v)) for k, v in recons.items()}

    algo_names = list(recons.keys())
    banks = [algo_names[i:i + algos_per_bank] for i in range(0, len(algo_names), algos_per_bank)]
    n_banks = len(banks)
    n_cols = 1 + algos_per_bank

    fig, axes = plt.subplots(
        3 * n_banks, n_cols,
        figsize=(2.0 * n_cols, 2.0 * 3 * n_banks),
        squeeze=False,
    )

    for bank_idx, bank in enumerate(banks):
        for i in range(3):  # axis row within bank
            row_global = bank_idx * 3 + i
            gt = gt_rows[i]
            vmin, vmax = float(gt.min()), float(gt.max())
            for j in range(n_cols):
                ax = axes[row_global, j]
                if j == 0:
                    arr = gt
                    if i == 0:
                        ax.set_title("Ground truth", fontsize=8)
                elif j - 1 < len(bank):
                    name = bank[j - 1]
                    arr = rec_rows[name][i]
                    if i == 0:
                        ax.set_title(name, fontsize=8)
                else:
                    ax.axis("off")
                    continue
                ax.imshow(arr, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(row_labels[i], fontsize=9)
    fig.suptitle(
        f"FastTomo algorithm comparison — quantem sinogram [{NOISE_TAG}] (29 angles, ±70°/5°)\n"
        "Sum-along-axis projections (each row shares the GT colour scale).",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def emit_slice_and_ssim(
    recons: dict[str, np.ndarray],
    phantom: np.ndarray,
    ssim_scores: dict[str, float],
    out_path: Path,
    *, algos_per_bank: int = 6,
) -> None:
    """Two banks of slice_idx=125 images (each bank: GT + K algos) stacked
    above a 3-D SSIM bar plot covering every algorithm."""
    gt_slice = phantom[SLICE_IDX]
    vmin, vmax = float(gt_slice.min()), float(gt_slice.max())

    algo_names = list(recons.keys())
    banks = [algo_names[i:i + algos_per_bank] for i in range(0, len(algo_names), algos_per_bank)]
    n_banks = len(banks)
    n_cols = 1 + algos_per_bank

    fig = plt.figure(figsize=(2.2 * n_cols, 2.4 * n_banks + 4.0))
    gs = fig.add_gridspec(
        n_banks + 1, n_cols,
        height_ratios=[1.4] * n_banks + [1.6],
        hspace=0.35,
    )

    for bank_idx, bank in enumerate(banks):
        # GT on the left of every bank
        ax_gt = fig.add_subplot(gs[bank_idx, 0])
        ax_gt.imshow(gt_slice, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        if bank_idx == 0:
            ax_gt.set_title("Ground truth", fontsize=9)
        ax_gt.set_ylabel(f"slice_idx = {SLICE_IDX}\n(view ZX)", fontsize=9)

        for j, name in enumerate(bank):
            ax = fig.add_subplot(gs[bank_idx, j + 1])
            ax.imshow(recons[name][SLICE_IDX], cmap="turbo",
                      vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(name, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    # SSIM bar plot spans every column on the last row
    ax_bar = fig.add_subplot(gs[-1, :])
    names = list(ssim_scores.keys())
    vals  = [ssim_scores[k] for k in names]
    bars = ax_bar.bar(names, vals, color="steelblue", edgecolor="k", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, v + 0.002,
            f"{v:.4f}", ha="center", va="bottom", fontsize=8,
        )
    ax_bar.set_ylabel("3-D SSIM vs ground truth")
    ax_bar.set_ylim(min(vals) * 0.985, 1.0)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.tick_params(axis="x", rotation=20)

    fig.suptitle(
        f"slice_idx={SLICE_IDX} reconstructions + 3-D SSIM "
        f"(quantem sinogram [{NOISE_TAG}], 29 angles ±70°/5°)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── TV-weight tuning ──────────────────────────────────────────────────────

def tune_tv(
    name: str, recon_fn, sino: torch.Tensor, angles: torch.Tensor,
    phantom: torch.Tensor, weights: list[float],
) -> list[dict]:
    """Sweep tv_weight; return list of {tv_weight, ssim_3d, nmse, time_s}."""
    sweep = []
    for w in weights:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        vol = recon_fn(sino, angles, tv_weight=w)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        nm = nmse(vol, phantom)
        ss = ssim3d(vol, phantom)
        sweep.append({"tv_weight": w, "ssim_3d": ss, "nmse": nm, "time_s": dt})
        print(f"   [{name}] tv={w:>8.4g}  ssim_3d={ss:.4f}  nmse={nm:.4e}  ({dt:.1f}s)")
        del vol
        torch.cuda.empty_cache()
    return sweep


def emit_tv_sweep_plot(sweeps: dict[str, list[dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    markers = {"FISTA-TV": "o", "AD-TV": "s", "AD-TV-S3IM": "^"}
    for name, sweep in sweeps.items():
        xs = [r["tv_weight"] for r in sweep]
        ys = [r["ssim_3d"]   for r in sweep]
        ax.plot(xs, ys, marker=markers.get(name, "x"), label=name)
        i_best = int(np.argmax(ys))
        ax.scatter([xs[i_best]], [ys[i_best]], s=120, facecolors="none",
                   edgecolors="red", linewidths=2.0, zorder=5)
        ax.annotate(
            f"  {name} best tv={xs[i_best]:g}  ssim={ys[i_best]:.4f}",
            (xs[i_best], ys[i_best]), fontsize=8, va="center",
        )
    ax.set_xscale("log")
    ax.set_xlabel("tv_weight")
    ax.set_ylabel("3-D SSIM vs ground truth")
    ax.set_title(f"TV-weight sweep — quantem sinogram [{NOISE_TAG}] (29 angles, ±70°/5°)")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def emit_s3im_sweep_plot(sweep: list[dict], tv_fixed: float, ad_tv_ssim: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    xs = [r["s3im_weight"] for r in sweep]
    ys = [r["ssim_3d"]     for r in sweep]
    ax.plot(xs, ys, marker="^", color="C2", label="AD-TV-S3IM")

    # AD-TV reference at λ=0
    ax.axhline(
        ad_tv_ssim, color="C1", linestyle="--", linewidth=1.0,
        label=f"AD-TV (λ_S3IM=0)  ssim={ad_tv_ssim:.4f}",
    )

    i_best = int(np.argmax(ys))
    ax.scatter([xs[i_best]], [ys[i_best]], s=120, facecolors="none",
               edgecolors="red", linewidths=2.0, zorder=5)
    ax.annotate(
        f"  best λ_S3IM={xs[i_best]:g}\n  ssim={ys[i_best]:.4f}",
        (xs[i_best], ys[i_best]), fontsize=8, va="center",
    )
    ax.set_xscale("log")
    ax.set_xlabel("λ_S3IM (S3IM loss weight)")
    ax.set_ylabel("3-D SSIM vs ground truth")
    ax.set_title(
        f"S3IM-weight sweep at fixed tv_weight={tv_fixed:g}\n"
        f"quantem sinogram [{NOISE_TAG}] (29 angles, ±70°/5°)"
    )
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def tune_s3im_internals(
    sino: torch.Tensor, angles: torch.Tensor, phantom: torch.Tensor,
    tv_weight: float, lam: float,
) -> dict[str, list[dict]]:
    """Sweep ``patch_pixels``, ``kernel_size``, ``n_patches`` separately at
    fixed (tv_weight, λ_S3IM). Each inner sweep changes one knob and holds
    the others at the library defaults."""
    results: dict[str, list[dict]] = {"patch_pixels": [], "kernel_size": [], "n_patches": []}

    print(f"\n   patch_pixels sweep (K=4, M=1)")
    for B in S3IM_PATCH_PIXELS_SWEEP:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        vol = recon_ad(
            sino, angles, tv_weight=tv_weight, s3im_weight=lam,
            s3im_patch_pixels=B, s3im_kernel_size=4, s3im_n_patches=1,
        )
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        nm, ss = nmse(vol, phantom), ssim3d(vol, phantom)
        side = int(round(B ** 0.5))
        results["patch_pixels"].append({"value": B, "side": side, "ssim_3d": ss, "nmse": nm, "time_s": dt})
        print(f"     B={B:>6d} ({side}×{side})  ssim_3d={ss:.4f}  nmse={nm:.4e}  ({dt:.1f}s)")
        del vol; torch.cuda.empty_cache()

    print(f"\n   kernel_size sweep (B=4096 → 64×64 patch, M=1)")
    for K in S3IM_KERNEL_SIZE_SWEEP:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        vol = recon_ad(
            sino, angles, tv_weight=tv_weight, s3im_weight=lam,
            s3im_patch_pixels=4096, s3im_kernel_size=K, s3im_n_patches=1,
        )
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        nm, ss = nmse(vol, phantom), ssim3d(vol, phantom)
        results["kernel_size"].append({"value": K, "ssim_3d": ss, "nmse": nm, "time_s": dt})
        print(f"     K={K:>3d}  ssim_3d={ss:.4f}  nmse={nm:.4e}  ({dt:.1f}s)")
        del vol; torch.cuda.empty_cache()

    print(f"\n   n_patches sweep (B=4096, K=4)")
    for M in S3IM_N_PATCHES_SWEEP:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        vol = recon_ad(
            sino, angles, tv_weight=tv_weight, s3im_weight=lam,
            s3im_patch_pixels=4096, s3im_kernel_size=4, s3im_n_patches=M,
        )
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        nm, ss = nmse(vol, phantom), ssim3d(vol, phantom)
        results["n_patches"].append({"value": M, "ssim_3d": ss, "nmse": nm, "time_s": dt})
        print(f"     M={M:>2d}  ssim_3d={ss:.4f}  nmse={nm:.4e}  ({dt:.1f}s)")
        del vol; torch.cuda.empty_cache()

    return results


def emit_s3im_internals_plot(
    sweeps: dict[str, list[dict]], baseline_ssim: float,
    tv_weight: float, lam: float, out_path: Path,
) -> None:
    """3-panel figure: SSIM vs patch_pixels, kernel_size, n_patches.

    The horizontal dashed line is the AD-TV-S3IM baseline at the library
    defaults (B=4096, K=4, M=1)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    panels = [
        ("patch_pixels", "patch_pixels  (√B × √B random sinogram entries)", "o", "C2"),
        ("kernel_size",  "kernel_size  (SSIM window side, stride=K)",       "s", "C0"),
        ("n_patches",    "n_patches  (independent patches averaged)",       "^", "C3"),
    ]
    for ax, (key, xlabel, marker, color) in zip(axes, panels):
        xs = [r["value"]   for r in sweeps[key]]
        ys = [r["ssim_3d"] for r in sweeps[key]]
        ax.plot(xs, ys, marker=marker, color=color)
        ax.axhline(baseline_ssim, color="gray", linestyle="--", linewidth=1.0,
                   label=f"defaults baseline = {baseline_ssim:.4f}")
        i_best = int(np.argmax(ys))
        ax.scatter([xs[i_best]], [ys[i_best]], s=140, facecolors="none",
                   edgecolors="red", linewidths=2.0, zorder=5)
        ax.annotate(
            f"  best {xs[i_best]}\n  ssim={ys[i_best]:.4f}",
            (xs[i_best], ys[i_best]), fontsize=8, va="center",
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("3-D SSIM vs ground truth")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"S3IM internal-parameter sweeps at tv_weight={tv_weight:g}, λ_S3IM={lam:g}\n"
        f"quantem sinogram [{NOISE_TAG}] (29 angles, ±70°/5°)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def tune_s3im(
    sino: torch.Tensor, angles: torch.Tensor, phantom: torch.Tensor,
    tv_weight: float, lambdas: list[float],
) -> tuple[list[dict], list[np.ndarray]]:
    """Sweep S3IM weight at fixed tv_weight. Also returns the slice_idx=125
    image for each λ so a downstream plot can show them side by side."""
    sweep = []
    slices: list[np.ndarray] = []
    for lam in lambdas:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        vol = recon_ad(sino, angles, tv_weight=tv_weight, s3im_weight=lam)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        nm = nmse(vol, phantom)
        ss = ssim3d(vol, phantom)
        sweep.append({"s3im_weight": lam, "ssim_3d": ss, "nmse": nm, "time_s": dt})
        slices.append(vol[SLICE_IDX].detach().cpu().numpy())
        print(f"   [AD-TV-S3IM] tv={tv_weight:g}  λ_S3IM={lam:>8.4g}  "
              f"ssim_3d={ss:.4f}  nmse={nm:.4e}  ({dt:.1f}s)")
        del vol
        torch.cuda.empty_cache()
    return sweep, slices


def emit_s3im_slice_grid(
    sweep: list[dict], slices: list[np.ndarray],
    gt_slice: np.ndarray, ad_tv_slice: np.ndarray, ad_tv_ssim: float,
    tv_fixed: float, out_path: Path,
) -> None:
    """Slice_idx=125 panel for every λ_S3IM value. First two cells are GT and
    the AD-TV (λ_S3IM=0) reference. Each panel is titled with its SSIM-3D."""
    n = 2 + len(sweep)
    n_cols = min(5, n)
    n_rows = (n + n_cols - 1) // n_cols

    vmin, vmax = float(gt_slice.min()), float(gt_slice.max())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols, 2.6 * n_rows),
        squeeze=False,
    )

    titles = (
        ["Ground truth", f"AD-TV (λ=0)\nssim={ad_tv_ssim:.4f}"]
        + [f"λ_S3IM={r['s3im_weight']:g}\nssim={r['ssim_3d']:.4f}"
           for r in sweep]
    )
    imgs = [gt_slice, ad_tv_slice] + list(slices)
    best_idx = 2 + int(np.argmax([r["ssim_3d"] for r in sweep]))

    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]
        if k >= n:
            ax.axis("off")
            continue
        ax.imshow(imgs[k], cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[k], fontsize=9)
        if k == best_idx:
            for spine in ax.spines.values():
                spine.set_edgecolor("red"); spine.set_linewidth(2.0)

    fig.suptitle(
        f"S3IM-λ sweep — slice_idx={SLICE_IDX} (view ZX) at fixed tv_weight={tv_fixed:g}\n"
        f"quantem sinogram [{NOISE_TAG}] (29 angles, ±70°/5°)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poisson-peak", type=float, default=None,
        help="Apply Poisson shot noise scaled so sino.max() = this many photon "
             "counts. Lower = noisier (relative std ≈ 1/√peak). Omit for the "
             "perfect (noiseless) sinogram.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the Poisson generator (default 0).",
    )
    args = parser.parse_args()

    global OUT_DIR, NOISE_TAG
    if args.poisson_peak is not None:
        NOISE_TAG = f"poisson_peak{int(args.poisson_peak)}"
        OUT_DIR = OUT_ROOT / f"algorithm_compare_{NOISE_TAG}"
    else:
        NOISE_TAG = "perfect"
        OUT_DIR = OUT_ROOT / "algorithm_compare"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sino = load_sino()
    phantom = load_phantom()
    n_angles = sino.shape[0]
    angles = torch.linspace(-70.0, 70.0, n_angles, device=DEVICE)
    print(f"Sinogram: {tuple(sino.shape)}  Phantom: {tuple(phantom.shape)}  "
          f"Angles: {angles[0].item():.1f}…{angles[-1].item():.1f} ({n_angles})")

    if args.poisson_peak is not None:
        clean_sino = sino
        sino = add_poisson_noise(clean_sino, args.poisson_peak, seed=args.seed)
        rel_std = ((sino - clean_sino).std() / clean_sino.std()).item()
        peak_count = float(clean_sino.max() * (args.poisson_peak / clean_sino.max()))  # = peak_counts
        snr_db = 20.0 * np.log10(clean_sino.std().item() / (sino - clean_sino).std().item())
        print(f"Poisson noise applied: peak_counts={args.poisson_peak:g}  "
              f"rel_noise={rel_std*100:.1f}%  SNR={snr_db:.1f} dB")
        np.save(OUT_DIR / "noisy_sino.npy", sino.detach().cpu().numpy())
    else:
        print("Using perfect (noiseless) sinogram.")

    # ── Phase 1: tune tv_weight per TV-dependent algorithm ────────────────
    print("\n=== Phase 1: TV-weight sweep ===")
    print(f"\n→ FISTA-TV sweep ({len(TV_SWEEP_FISTA)} weights)")
    sweep_fista = tune_tv("FISTA-TV", recon_fista_tv, sino, angles, phantom, TV_SWEEP_FISTA)
    best_fista = max(sweep_fista, key=lambda r: r["ssim_3d"])
    print(f"   → best FISTA-TV tv_weight = {best_fista['tv_weight']:g}  "
          f"(ssim_3d={best_fista['ssim_3d']:.4f})")

    print(f"\n→ AD-TV sweep ({len(TV_SWEEP_AD)} weights)")
    def _ad_tv(s, a, *, tv_weight):
        return recon_ad(s, a, tv_weight=tv_weight)
    sweep_ad = tune_tv("AD-TV", _ad_tv, sino, angles, phantom, TV_SWEEP_AD)
    best_ad = max(sweep_ad, key=lambda r: r["ssim_3d"])
    print(f"   → best AD-TV tv_weight = {best_ad['tv_weight']:g}  "
          f"(ssim_3d={best_ad['ssim_3d']:.4f})")

    print(f"\n→ AD-TV-S3IM tv-sweep ({len(TV_SWEEP_AD_S3IM)} weights, λ_S3IM={S3IM_LAMBDA_INITIAL})")
    def _ad_tv_s3im(s, a, *, tv_weight):
        return recon_ad(s, a, tv_weight=tv_weight, s3im_weight=S3IM_LAMBDA_INITIAL)
    sweep_ad_s3im = tune_tv("AD-TV-S3IM", _ad_tv_s3im, sino, angles, phantom, TV_SWEEP_AD_S3IM)
    best_ad_s3im = max(sweep_ad_s3im, key=lambda r: r["ssim_3d"])
    print(f"   → best AD-TV-S3IM tv_weight = {best_ad_s3im['tv_weight']:g}  "
          f"(ssim_3d={best_ad_s3im['ssim_3d']:.4f})")

    # ── Phase 1b: sweep λ_S3IM at the best TV weight ──────────────────────
    print(f"\n→ AD-TV-S3IM λ_S3IM sweep at tv={best_ad_s3im['tv_weight']:g} "
          f"({len(S3IM_LAMBDA_SWEEP)} λ values)")
    sweep_lambda, lambda_slices = tune_s3im(
        sino, angles, phantom,
        tv_weight=best_ad_s3im["tv_weight"],
        lambdas=S3IM_LAMBDA_SWEEP,
    )
    best_lambda = max(sweep_lambda, key=lambda r: r["ssim_3d"])
    print(f"   → best λ_S3IM = {best_lambda['s3im_weight']:g}  "
          f"(ssim_3d={best_lambda['ssim_3d']:.4f})")

    with open(OUT_DIR / "tv_sweep.json", "w") as f:
        json.dump({
            "fista_tv":   sweep_fista,
            "ad_tv":      sweep_ad,
            "ad_tv_s3im": sweep_ad_s3im,
            "s3im_lambda_sweep_at_best_tv": sweep_lambda,
        }, f, indent=2)
    emit_tv_sweep_plot(
        {"FISTA-TV": sweep_fista, "AD-TV": sweep_ad, "AD-TV-S3IM": sweep_ad_s3im},
        OUT_DIR / "tv_sweep.png",
    )
    emit_s3im_sweep_plot(
        sweep_lambda,
        tv_fixed=best_ad_s3im["tv_weight"],
        ad_tv_ssim=best_ad["ssim_3d"],
        out_path=OUT_DIR / "s3im_sweep.png",
    )

    # ── Phase 1c: slice_idx=125 panel for every λ_S3IM ────────────────────
    # AD-TV (λ=0) baseline at the SAME tv as the sweep, so the panel is a
    # clean ablation along λ_S3IM only.
    print(f"\n→ AD-TV reference at tv={best_ad_s3im['tv_weight']:g} for slice panel")
    ad_tv_ref_vol = recon_ad(sino, angles, tv_weight=best_ad_s3im["tv_weight"], s3im_weight=0.0)
    ad_tv_ref_ssim = ssim3d(ad_tv_ref_vol, phantom)
    ad_tv_ref_slice = ad_tv_ref_vol[SLICE_IDX].detach().cpu().numpy()
    del ad_tv_ref_vol
    torch.cuda.empty_cache()

    phantom_np_full = phantom.detach().cpu().numpy()
    emit_s3im_slice_grid(
        sweep=sweep_lambda,
        slices=lambda_slices,
        gt_slice=phantom_np_full[SLICE_IDX],
        ad_tv_slice=ad_tv_ref_slice,
        ad_tv_ssim=ad_tv_ref_ssim,
        tv_fixed=best_ad_s3im["tv_weight"],
        out_path=OUT_DIR / "s3im_sweep_slices.png",
    )

    # ── Phase 1d: sweep internal s3im_loss knobs at the tuned (tv, λ) ─────
    print(f"\n=== Phase 1d: S3IM internal-parameter sweeps "
          f"(tv={best_ad_s3im['tv_weight']:g}, λ={best_lambda['s3im_weight']:g}) ===")
    sweeps_internal = tune_s3im_internals(
        sino, angles, phantom,
        tv_weight=best_ad_s3im["tv_weight"],
        lam=best_lambda["s3im_weight"],
    )
    # The "defaults baseline" matches best_lambda's reconstruction (B=4096, K=4, M=1)
    baseline_ssim = best_lambda["ssim_3d"]
    emit_s3im_internals_plot(
        sweeps_internal,
        baseline_ssim=baseline_ssim,
        tv_weight=best_ad_s3im["tv_weight"],
        lam=best_lambda["s3im_weight"],
        out_path=OUT_DIR / "s3im_internals_sweep.png",
    )
    with open(OUT_DIR / "s3im_internals_sweep.json", "w") as f:
        json.dump(sweeps_internal, f, indent=2)

    # ── Phase 2: full comparison at tuned weights ─────────────────────────
    print("\n=== Phase 2: full algorithm comparison ===")
    algorithms = build_algorithms(
        best_fista["tv_weight"],
        best_ad["tv_weight"],
        best_ad_s3im["tv_weight"],
        best_lambda["s3im_weight"],
    )
    recons: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, fn, tv in algorithms:
        print(f"\n→ {name}" + (f" (tv_weight={tv:g})" if tv is not None else ""))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        vol = fn(sino, angles)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        nm = nmse(vol, phantom)
        ss = ssim3d(vol, phantom)
        metrics[name] = {"nmse": nm, "ssim_3d": ss, "time_s": dt, "tv_weight": tv}
        recons[name] = vol.detach().cpu().numpy()
        del vol
        torch.cuda.empty_cache()
        print(f"   nmse={nm:.4e}  ssim_3d={ss:.4f}  time={dt:.1f}s")

    phantom_np = phantom.detach().cpu().numpy()
    ssim_scores = {k: v["ssim_3d"] for k, v in metrics.items()}

    emit_axes_grid(recons, phantom_np, OUT_DIR / "axes_grid.png")
    emit_slice_and_ssim(recons, phantom_np, ssim_scores, OUT_DIR / "slice125_and_ssim.png")

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved figures to {OUT_DIR}")
    print(f"  tv_sweep.png  ({len(TV_SWEEP_FISTA)} FISTA-TV + {len(TV_SWEEP_AD)} AD-TV + "
          f"{len(TV_SWEEP_AD_S3IM)} AD-TV-S3IM weights)")
    print(f"  s3im_sweep.png  ({len(S3IM_LAMBDA_SWEEP)} λ_S3IM values at fixed tv)")
    print(f"  s3im_sweep_slices.png  (slice {SLICE_IDX} for GT + AD-TV(λ=0) + each λ)")
    print(f"  s3im_internals_sweep.png  (B / K / M sweeps)")
    print(f"  axes_grid.png")
    print(f"  slice125_and_ssim.png")
    print(f"  results.json  /  tv_sweep.json")


if __name__ == "__main__":
    main()
