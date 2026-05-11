"""Plot SSIM-loss ablation results.

Inputs:
    benchmarks/ssim-ablation/results.json
    data/ssim_ablation_recons.npz       (phantom + best recon per cell/kind)

Outputs (all under benchmarks/ssim-ablation/):
    nmse_heatmap.png             — 5 NMSE heatmaps over (range × step)
    ssim_heatmap.png             — 5 2D-slice-mean SSIM heatmaps
    ssim3d_heatmap.png           — 5 full-volume 3D SSIM heatmaps
    convergence.png              — 1 subplot per (range, step), 5 lines
    lambda_sensitivity.png       — NMSE vs λ for multiplex cells
    tv_sensitivity.png           — NMSE vs TV-weight for L2+TV+λ·S3IM cell
    cells/r{R}_s{S}_projections.png  — axis-summed projections per cell
    cells/r{R}_s{S}_slices.png       — central slices per cell

By default produces one projection+slice figure for every (range, step) cell.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "benchmarks" / "results" / "ssim-ablation-03-tv3-tiltsweep"
RESULTS_JSON = OUT_DIR / "results.json"
RECONS_NPZ = OUT_DIR / "recons.npz"
CELL_DIR = OUT_DIR / "cells"

LOSS_KINDS = ("mse", "mse_tv", "mse_tv_ssim", "mse_s3im", "mse_tv_s3im")
PRETTY = {
    "mse": "L2",
    "mse_tv": "L2 + TV",
    "mse_tv_ssim": "L2 + TV + SSIM (local)",
    "mse_s3im": "Multiplex (L2 + λ·S3IM)",
    "mse_tv_s3im": "L2 + TV + λ·S3IM",
}
COLORS = {
    "mse": "#888888",
    "mse_tv": "#1f77b4",
    "mse_tv_ssim": "#2ca02c",
    "mse_s3im": "#d62728",
    "mse_tv_s3im": "#9467bd",
}


# ── IO ────────────────────────────────────────────────────────────────────

def load_results() -> list[dict]:
    with RESULTS_JSON.open() as f:
        return json.load(f)


def best_per_cell(results: list[dict]) -> dict[tuple[float, float, str], dict]:
    """Best (lowest-NMSE) sub-run per (range, step, loss_kind) cell.

    For ``mse_tv_s3im`` this picks the best (λ, tv_weight) combination; for
    ``mse_s3im`` the best λ; for the rest there's only one record.
    """
    grouped: dict[tuple[float, float, str], list[dict]] = defaultdict(list)
    for r in results:
        grouped[(r["range_deg"], r["step_deg"], r["loss_kind"])].append(r)
    return {key: min(items, key=lambda x: x["nmse"]) for key, items in grouped.items()}


def _hp_tag(rec: dict) -> str:
    """Compact hyperparam tag for plot labels."""
    bits = []
    if rec["loss_kind"] in ("mse_s3im", "mse_tv_s3im"):
        bits.append(f"λ={rec['lam']:g}")
    if rec["loss_kind"] == "mse_tv_s3im":
        bits.append(f"tv={rec['tv_weight']:g}")
    return ", ".join(bits)


# ── Heatmaps ──────────────────────────────────────────────────────────────

def heatmap_grid(results: list[dict], metric: str, lower_is_better: bool,
                 out_path: Path) -> None:
    """5 heatmaps over (range × step); each cell shows the best sub-run."""
    best = best_per_cell(results)
    ranges = sorted({r["range_deg"] for r in results}, reverse=True)
    steps = sorted({r["step_deg"] for r in results})

    fig, axes = plt.subplots(1, len(LOSS_KINDS), figsize=(4 * len(LOSS_KINDS), 4),
                             sharey=True)
    vals_all = [best[(r, s, k)][metric] for r in ranges for s in steps for k in LOSS_KINDS
                if (r, s, k) in best]
    vmin, vmax = min(vals_all), max(vals_all)

    cmap = "viridis_r" if lower_is_better else "viridis"
    for ax, kind in zip(axes, LOSS_KINDS):
        grid = np.full((len(ranges), len(steps)), np.nan)
        for i, r in enumerate(ranges):
            for j, s in enumerate(steps):
                rec = best.get((r, s, kind))
                if rec is not None:
                    grid[i, j] = rec[metric]
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([f"{s:.0f}°" for s in steps])
        ax.set_yticks(range(len(ranges)))
        ax.set_yticklabels([f"±{r:.0f}°" for r in ranges])
        ax.set_xlabel("tilt step")
        ax.set_title(PRETTY[kind], fontsize=10)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white", fontsize=8)
    axes[0].set_ylabel("tilt range")
    fig.suptitle(f"{metric.upper()} per (tilt range × tilt step) — best (λ, TV) per cell",
                 fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


# ── Convergence ───────────────────────────────────────────────────────────

def convergence_grid(results: list[dict], out_path: Path) -> None:
    """One subplot per (range, step), 5 lines (best sub-run per loss kind)."""
    best = best_per_cell(results)
    ranges = sorted({r["range_deg"] for r in results}, reverse=True)
    steps = sorted({r["step_deg"] for r in results})
    nrows, ncols = len(ranges), len(steps)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=False, squeeze=False)
    last_rec = None
    for i, r in enumerate(ranges):
        for j, s in enumerate(steps):
            ax = axes[i, j]
            n_ang = None
            for kind in LOSS_KINDS:
                rec = best.get((r, s, kind))
                if rec is None:
                    continue
                last_rec = rec
                n_ang = rec.get("n_angles")
                lbl = PRETTY[kind]
                tag = _hp_tag(rec)
                if tag:
                    lbl = f"{lbl} ({tag})"
                ax.plot(rec["losses"], color=COLORS[kind], label=lbl, lw=1.0)
            ax.set_yscale("log")
            title = f"±{r:.0f}°  step {s:.0f}°"
            if n_ang is not None:
                title += f"  ({n_ang} angles)"
            ax.set_title(title, fontsize=10)
            if i == nrows - 1:
                ax.set_xlabel("iteration")
            if j == 0:
                ax.set_ylabel("loss (log)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("AD-recon convergence per loss recipe (best sub-run shown)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


# ── λ and TV sensitivity ──────────────────────────────────────────────────

def lambda_sensitivity(results: list[dict], out_path: Path) -> None:
    """Multiplex cells: NMSE vs λ. For mse_tv_s3im, fix tv at the best value
    per cell so the curve isolates λ.
    """
    lams_seen = {rec["lam"] for rec in results
                 if rec["loss_kind"] in ("mse_s3im", "mse_tv_s3im")}
    if len(lams_seen) < 2:
        print(f"  skipping {out_path.name}: only {len(lams_seen)} λ value(s)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, kind in zip(axes, ("mse_s3im", "mse_tv_s3im")):
        for r in sorted({rec["range_deg"] for rec in results}, reverse=True):
            for s in sorted({rec["step_deg"] for rec in results}):
                rows = [
                    rec for rec in results
                    if rec["loss_kind"] == kind
                    and rec["range_deg"] == r and rec["step_deg"] == s
                ]
                if not rows:
                    continue
                # For tv-aware kind, pick the tv_weight that yields the best
                # NMSE at this cell, then plot NMSE vs λ at that tv.
                if kind == "mse_tv_s3im":
                    best_tv = min(rows, key=lambda x: x["nmse"])["tv_weight"]
                    rows = [r for r in rows if r["tv_weight"] == best_tv]
                rows = sorted(rows, key=lambda x: x["lam"])
                lams = [rec["lam"] for rec in rows]
                nmses = [rec["nmse"] for rec in rows]
                lbl = f"±{r:.0f}°  step {s:.0f}°"
                if kind == "mse_tv_s3im":
                    lbl += f"  [tv={best_tv:g}]"
                ax.plot(lams, nmses, marker="o", label=lbl, lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("λ")
        ax.set_ylabel("NMSE")
        ax.set_title(PRETTY[kind], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc="best")
    fig.suptitle("λ sensitivity for multiplex losses (best TV held fixed per cell)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def tv_sensitivity(results: list[dict], out_path: Path) -> None:
    """For mse_tv_s3im: NMSE vs TV-weight, one line per (range, step), best
    λ held fixed per cell.
    """
    rows_all = [r for r in results if r["loss_kind"] == "mse_tv_s3im"]
    if not rows_all:
        print(f"  skipping {out_path.name}: no mse_tv_s3im records")
        return
    tvs_seen = {rec["tv_weight"] for rec in rows_all}
    if len(tvs_seen) < 2:
        print(f"  skipping {out_path.name}: only {len(tvs_seen)} TV value(s)")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in sorted({rec["range_deg"] for rec in rows_all}, reverse=True):
        for s in sorted({rec["step_deg"] for rec in rows_all}):
            rows = [rec for rec in rows_all
                    if rec["range_deg"] == r and rec["step_deg"] == s]
            if not rows:
                continue
            best_lam = min(rows, key=lambda x: x["nmse"])["lam"]
            rows = sorted([r for r in rows if r["lam"] == best_lam],
                          key=lambda x: x["tv_weight"])
            tvs = [rec["tv_weight"] for rec in rows]
            nmses = [rec["nmse"] for rec in rows]
            ax.plot(tvs, nmses, marker="o",
                    label=f"±{r:.0f}°  step {s:.0f}°  [λ={best_lam:g}]", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("TV weight")
    ax.set_ylabel("NMSE")
    ax.set_title(f"TV-weight sensitivity for {PRETTY['mse_tv_s3im']} (best λ per cell)",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


# ── Volume-view (projection / slice) figures ─────────────────────────────

def _load_recons() -> dict[str, np.ndarray]:
    if not RECONS_NPZ.exists():
        raise FileNotFoundError(
            f"{RECONS_NPZ} not found. Run benchmarks/run_ssim_ablation.py first."
        )
    with np.load(RECONS_NPZ) as data:
        return {key: data[key] for key in data.files}


def _axis_projections(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return vol.sum(axis=0), vol.sum(axis=1), vol.sum(axis=2)


def _middle_slices(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ny, Nz, Nx = vol.shape
    return vol[Ny // 2], vol[:, Nz // 2], vol[:, :, Nx // 2]


_PROJ_ROW_LABELS = (
    "Σ along Y\n(view ZX)",
    "Σ along Z\n(view YX)",
    "Σ along X\n(view YZ)",
)
_SLICE_ROW_LABELS = (
    "mid-Y slice\n(view ZX)",
    "mid-Z slice\n(view YX)",
    "mid-X slice\n(view YZ)",
)


def _grid_figure(
    rows: list[list[np.ndarray]],
    col_titles: list[str],
    row_labels: list[str],
    out_path: Path,
    suptitle: str,
    cmap: str = "turbo",
    vmin: float | None = 0.0,
) -> None:
    n_rows = len(rows)
    n_cols = len(rows[0])
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols, 2.6 * n_rows),
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
                ax.set_title(col_titles[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def _build_cols(
    recons: dict[str, np.ndarray],
    range_deg: float, step_deg: float,
    best_per_cell_map: dict[tuple[float, float, str], dict] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Returns ([GT, ...recons], [GT title, ...loss titles])."""
    cols = [recons["phantom"]]
    titles = ["Ground truth"]
    for kind in LOSS_KINDS:
        key = f"r{int(range_deg)}_s{int(step_deg)}_{kind}"
        if key not in recons:
            raise KeyError(f"missing {key} in {RECONS_NPZ}")
        cols.append(recons[key])
        # Annotate the title with the best (λ, TV) used in this cell.
        title = PRETTY[kind]
        if best_per_cell_map is not None:
            rec = best_per_cell_map.get((range_deg, step_deg, kind))
            tag = _hp_tag(rec) if rec is not None else ""
            if tag:
                title = f"{title}\n[{tag}]"
        titles.append(title)
    return cols, titles


def projection_figure(
    recons: dict[str, np.ndarray], range_deg: float, step_deg: float,
    out_path: Path,
    best_per_cell_map: dict | None = None,
) -> None:
    cols, titles = _build_cols(recons, range_deg, step_deg, best_per_cell_map)
    projs = [_axis_projections(v) for v in cols]
    rows = [[projs[c][axis] for c in range(len(cols))] for axis in range(3)]
    _grid_figure(
        rows, titles, list(_PROJ_ROW_LABELS),
        out_path,
        suptitle=(f"Axis-summed projections — tilt range ±{int(range_deg)}°, "
                  f"step {int(step_deg)}°"),
    )


def slice_figure(
    recons: dict[str, np.ndarray], range_deg: float, step_deg: float,
    out_path: Path,
    best_per_cell_map: dict | None = None,
) -> None:
    cols, titles = _build_cols(recons, range_deg, step_deg, best_per_cell_map)
    slices = [_middle_slices(v) for v in cols]
    rows = [[slices[c][axis] for c in range(len(cols))] for axis in range(3)]
    _grid_figure(
        rows, titles, list(_SLICE_ROW_LABELS),
        out_path,
        suptitle=(f"Central slices — tilt range ±{int(range_deg)}°, "
                  f"step {int(step_deg)}°"),
    )


# ── Driver ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-range", type=float, default=None,
                        help="If set, only emit projection/slice figs for this tilt range")
    parser.add_argument("--view-step", type=float, default=None,
                        help="If set, only emit projection/slice figs for this tilt step")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    print(f"Loaded {len(results)} records from {RESULTS_JSON.relative_to(REPO_ROOT)}")

    heatmap_grid(results, "nmse", lower_is_better=True,
                 out_path=OUT_DIR / "nmse_heatmap.png")
    heatmap_grid(results, "ssim", lower_is_better=False,
                 out_path=OUT_DIR / "ssim_heatmap.png")
    if any("ssim_3d" in r for r in results):
        heatmap_grid(results, "ssim_3d", lower_is_better=False,
                     out_path=OUT_DIR / "ssim3d_heatmap.png")
    convergence_grid(results, OUT_DIR / "convergence.png")
    lambda_sensitivity(results, OUT_DIR / "lambda_sensitivity.png")
    tv_sensitivity(results, OUT_DIR / "tv_sensitivity.png")

    # Volume-view figures (need NPZ + phantom).
    try:
        recons = _load_recons()
    except FileNotFoundError as e:
        print(f"Skipping projection/slice figures: {e}")
        return
    if "phantom" not in recons:
        print("Skipping projection/slice figures: phantom missing from NPZ "
              "(re-run run_ssim_ablation.py to regenerate with phantom saved)")
        return

    best_map = best_per_cell(results)
    cells = sorted({(r["range_deg"], r["step_deg"]) for r in results})
    if args.view_range is not None and args.view_step is not None:
        cells = [(args.view_range, args.view_step)]

    print(f"Generating projection+slice figures for {len(cells)} cell(s):")
    for r, s in cells:
        tag = f"r{int(r)}_s{int(s)}"
        try:
            projection_figure(
                recons, r, s,
                CELL_DIR / f"{tag}_projections.png",
                best_per_cell_map=best_map,
            )
            slice_figure(
                recons, r, s,
                CELL_DIR / f"{tag}_slices.png",
                best_per_cell_map=best_map,
            )
        except KeyError as e:
            print(f"  skipping {tag}: {e}")


if __name__ == "__main__":
    main()
