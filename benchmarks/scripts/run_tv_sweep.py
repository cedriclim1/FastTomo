"""TV-weight sweep on the two SSIM-bearing AD-recon recipes.

Holds the structural weight constant (``ssim_weight=0.5`` for the local SSIM
recipe; ``λ_S3IM=1.0`` for the multiplex recipe) and varies *only* the TV
weight. Reports NMSE and 3-D SSIM on three representative cells across the
missing-wedge spectrum so the user can pick the right TV value per regime.

Outputs:
    benchmarks/ssim-ablation-04-tv-sweep/results.json
    benchmarks/ssim-ablation-04-tv-sweep/tv_sweep.png
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim

import fasttomo as ft

REPO_ROOT = Path(__file__).resolve().parent.parent
QUANTEM_DATA_DIR = Path("/home/cedlim/quantem/quantem-tutorials/data")
OUT_DIR = REPO_ROOT / "benchmarks" / "results" / "ssim-ablation-04-tv-sweep"
OUT_JSON = OUT_DIR / "results.json"
OUT_PNG = OUT_DIR / "tv_sweep.png"

# Sweep grid
CELLS = ((70.0, 5.0), (40.0, 8.0), (25.0, 12.0))   # (range_deg, step_deg)
TV_WEIGHTS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
RECIPES = ("mse_tv_ssim", "mse_tv_s3im")
SSIM_WEIGHT = 0.5            # weight on tilt_ssim_loss in mse_tv_ssim recipe
LAMBDA_S3IM = 1.0            # weight on s3im_loss in mse_tv_s3im recipe
S3IM_PATCH_PIXELS = 4096
S3IM_KERNEL_SIZE = 4
S3IM_N_PATCHES = 1
N_ITER = 1000
LR = 0.05
SEED = 0


@dataclass
class Result:
    range_deg: float
    step_deg: float
    recipe: str
    tv_weight: float
    nmse: float
    ssim_3d: float
    final_loss: float
    time_s: float
    n_angles: int
    losses: list[float] = field(default_factory=list)


def load_phantom(device: torch.device) -> torch.Tensor:
    arr = np.load(QUANTEM_DATA_DIR / "phantom.npy").astype(np.float32)
    arr = arr.transpose(2, 0, 1)[::-1].copy()
    return torch.from_numpy(arr).to(device)


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


def reconstruct(
    sino: torch.Tensor, angles: torch.Tensor,
    *, recipe: str, tv_weight: float, seed: int,
) -> tuple[torch.Tensor, list[float]]:
    n_angles, Ny, N_det = sino.shape
    proj = ft.ParallelProjector((Ny, N_det, N_det), angles, device=sino.device,
                                N_det=N_det)
    torch.manual_seed(seed)
    vol = torch.zeros(Ny, N_det, N_det, device=sino.device, requires_grad=True)
    opt = torch.optim.Adam([vol], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_ITER)
    s3im_gen = torch.Generator(device=sino.device).manual_seed(seed + 1)

    losses: list[float] = []
    for _ in range(N_ITER):
        opt.zero_grad(set_to_none=True)
        pred = proj.forward(vol)
        loss = F.mse_loss(pred, sino) + tv_weight * ft.tv_loss_3d(vol)
        if recipe == "mse_tv_ssim":
            loss = loss + SSIM_WEIGHT * ft.tilt_ssim_loss(pred, sino)
        elif recipe == "mse_tv_s3im":
            loss = loss + LAMBDA_S3IM * ft.s3im_loss(
                pred, sino,
                patch_pixels=S3IM_PATCH_PIXELS, kernel_size=S3IM_KERNEL_SIZE,
                n_patches=S3IM_N_PATCHES, generator=s3im_gen,
            )
        else:
            raise ValueError(f"unknown recipe: {recipe}")
        losses.append(float(loss.item()))
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            vol.clamp_(min=0)
    return vol.detach(), losses


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    phantom = load_phantom(device)
    phantom_np = phantom.detach().cpu().numpy()

    results: list[Result] = []
    # Cache all 42 volumes (in CPU memory) so we can build comparison figures
    # after the sweep finishes. 42 × 200³ × 4B ≈ 1.3 GB — fine on CPU.
    vol_cache: dict[tuple[float, float, str, float], np.ndarray] = {}
    total = len(CELLS) * len(RECIPES) * len(TV_WEIGHTS)
    idx = 0
    for r, s in CELLS:
        n_angles = int(round(2 * r / s)) + 1
        angles = torch.linspace(-r, r, n_angles, device=device)
        with torch.no_grad():
            sino = ft.forward_project(phantom, angles)
        print(f"\n=== ±{r:.0f}°, step {s:.0f}° → {n_angles} angles ===")
        for recipe in RECIPES:
            for tv in TV_WEIGHTS:
                idx += 1
                t0 = time.perf_counter()
                vol, losses = reconstruct(
                    sino, angles, recipe=recipe, tv_weight=tv, seed=SEED,
                )
                dt = time.perf_counter() - t0
                nm = nmse(vol, phantom)
                ss = ssim3d(vol, phantom)
                print(f"[{idx:>3}/{total}] {recipe:<14s} tv={tv:>6.2f}  "
                      f"NMSE={nm:.4f}  SSIM3D={ss:.4f}  ({dt:.1f}s)")
                results.append(Result(
                    range_deg=r, step_deg=s, recipe=recipe, tv_weight=tv,
                    nmse=nm, ssim_3d=ss, final_loss=losses[-1], time_s=dt,
                    n_angles=n_angles, losses=losses,
                ))
                vol_cache[(r, s, recipe, tv)] = vol.detach().cpu().numpy().astype(np.float32)
                del vol; torch.cuda.empty_cache()

    with OUT_JSON.open("w") as f:
        json.dump([asdict(rec) for rec in results], f, indent=2)
    print(f"\nSaved {len(results)} results → {OUT_JSON.relative_to(REPO_ROOT)}")

    # ── Plot: rows = cells, cols = (NMSE, SSIM3D) ────────────────────
    n_cells = len(CELLS)
    fig, axes = plt.subplots(n_cells, 2, figsize=(12, 3.5 * n_cells),
                             squeeze=False, sharex=True)
    recipe_colors = {"mse_tv_ssim": "#2ca02c", "mse_tv_s3im": "#9467bd"}
    recipe_labels = {
        "mse_tv_ssim": f"L2+TV+SSIM(local) (w={SSIM_WEIGHT})",
        "mse_tv_s3im": f"L2+TV+λ·S3IM (λ={LAMBDA_S3IM})",
    }

    for row, (r, s) in enumerate(CELLS):
        n_ang = next(rec.n_angles for rec in results
                     if rec.range_deg == r and rec.step_deg == s)
        for col, (metric, label, lower_better) in enumerate((
            ("nmse", "NMSE", True),
            ("ssim_3d", "SSIM (3D)", False),
        )):
            ax = axes[row, col]
            for recipe in RECIPES:
                rows = sorted(
                    [rec for rec in results
                     if rec.range_deg == r and rec.step_deg == s
                     and rec.recipe == recipe],
                    key=lambda x: x.tv_weight,
                )
                xs = [rec.tv_weight for rec in rows]
                ys = [getattr(rec, metric) for rec in rows]
                ax.plot(xs, ys, marker="o", color=recipe_colors[recipe],
                        lw=1.5, label=recipe_labels[recipe])
                # Mark the optimum
                if lower_better:
                    opt_idx = int(np.argmin(ys))
                else:
                    opt_idx = int(np.argmax(ys))
                ax.scatter([xs[opt_idx]], [ys[opt_idx]],
                           color=recipe_colors[recipe], s=130, zorder=5,
                           edgecolor="black", linewidth=1.2)
                ax.annotate(f"  TV={xs[opt_idx]:g}\n  {label}={ys[opt_idx]:.4f}",
                            (xs[opt_idx], ys[opt_idx]),
                            fontsize=7, color=recipe_colors[recipe])
            ax.set_xscale("log")
            ax.set_xlabel("TV weight")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
            ax.set_title(f"{label} vs TV — ±{r:.0f}°, step {s:.0f}°  ({n_ang} angles)",
                         fontsize=10)

    fig.suptitle("TV-weight sensitivity for L2+TV+SSIM and L2+TV+λ·S3IM",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {OUT_PNG.relative_to(REPO_ROOT)}")

    # ── Per-(cell, recipe) volume comparison figures ─────────────────
    proj_dir = OUT_DIR / "cells"
    proj_dir.mkdir(parents=True, exist_ok=True)

    proj_row_labels = (
        "Σ along Y\n(view ZX)",
        "Σ along Z\n(view YX)",
        "Σ along X\n(view YZ)",
    )
    slice_row_labels = (
        "mid-Y slice\n(view ZX)",
        "mid-Z slice\n(view YX)",
        "mid-X slice\n(view YZ)",
    )

    def axis_projections(v: np.ndarray):
        return v.sum(axis=0), v.sum(axis=1), v.sum(axis=2)

    def middle_slices(v: np.ndarray):
        Ny, Nz, Nx = v.shape
        return v[Ny // 2], v[:, Nz // 2], v[:, :, Nx // 2]

    def emit_grid(rows: list[list[np.ndarray]], col_titles: list[str],
                  row_labels: tuple[str, ...], out_path: Path, suptitle: str,
                  cmap: str = "turbo", vmin: float | None = 0.0) -> None:
        n_rows, n_cols = len(rows), len(rows[0])
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.2 * n_cols, 2.6 * n_rows),
            squeeze=False,
        )
        for i, row in enumerate(rows):
            row_vmin = min(a.min() for a in row) if vmin is None else vmin
            vmax = max(a.max() for a in row)
            for j, arr in enumerate(row):
                ax = axes[i, j]
                ax.imshow(arr, cmap=cmap, vmin=row_vmin, vmax=vmax, aspect="auto")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(col_titles[j], fontsize=9)
                if j == 0:
                    ax.set_ylabel(row_labels[i], fontsize=9)
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_path.relative_to(REPO_ROOT)}")

    print("\nGenerating per-(cell, recipe) projection + slice figures:")
    for r, s in CELLS:
        for recipe in RECIPES:
            cols = [phantom_np]
            titles = ["Ground truth"]
            for tv in TV_WEIGHTS:
                cols.append(vol_cache[(r, s, recipe, tv)])
                rec = next(rr for rr in results
                           if rr.range_deg == r and rr.step_deg == s
                           and rr.recipe == recipe and rr.tv_weight == tv)
                titles.append(f"TV={tv:g}\nNMSE={rec.nmse:.3f}\nSSIM={rec.ssim_3d:.3f}")
            tag = f"r{int(r)}_s{int(s)}_{recipe}"

            projs = [axis_projections(v) for v in cols]
            proj_rows = [[projs[c][axis] for c in range(len(cols))] for axis in range(3)]
            emit_grid(
                proj_rows, titles, proj_row_labels,
                proj_dir / f"{tag}_projections.png",
                suptitle=(f"Axis-summed projections — TV sweep on {recipe} "
                          f"— ±{r:.0f}°, step {s:.0f}°"),
            )

            sl = [middle_slices(v) for v in cols]
            slice_rows = [[sl[c][axis] for c in range(len(cols))] for axis in range(3)]
            emit_grid(
                slice_rows, titles, slice_row_labels,
                proj_dir / f"{tag}_slices.png",
                suptitle=(f"Central slices — TV sweep on {recipe} "
                          f"— ±{r:.0f}°, step {s:.0f}°"),
            )

    # Save all 42 reconstructions + phantom for later inspection.
    npz_payload = {"phantom": phantom_np.astype(np.float32)}
    for (r, s, recipe, tv), arr in vol_cache.items():
        npz_payload[f"r{int(r)}_s{int(s)}_{recipe}_tv{tv:g}"] = arr
    out_npz = OUT_DIR / "recons.npz"
    np.savez_compressed(out_npz, **npz_payload)
    print(f"\nSaved phantom + {len(vol_cache)} recons → {out_npz.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
