"""Summary plotter for the method comparison sweep.

Reads ``benchmarks/method_comparison/<sub-dir>/results.json`` and emits:
    * ``summary_bars.png``    — NMSE + SSIM bar charts, one panel per cell
    * ``summary_heatmap.png`` — method × range heatmap (dirs 02/03/04 only)

Per-cell projection + slice grids are produced directly by
``benchmarks/run_method_comparison.py`` and are not regenerated here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "benchmarks" / "results" / "method_comparison"
SWEEP_DIRS = ("01_70tr_2s", "02_range_s5", "03_range_s2", "04_range_s10")

METHOD_ORDER = [
    "fbp", "sirt", "sart", "genfire", "resire",
    "ad", "ad_tv", "ad_tv_ssim", "ad_s3im", "ad_tv_s3im",
]
METHOD_COLORS = {
    "fbp":         "#888888",
    "sirt":        "#1f77b4",
    "sart":        "#17becf",
    "genfire":     "#ff7f0e",
    "resire":      "#e377c2",
    "ad":          "#7f7f7f",
    "ad_tv":       "#2ca02c",
    "ad_tv_ssim":  "#bcbd22",
    "ad_s3im":     "#d62728",
    "ad_tv_s3im":  "#9467bd",
}


def load_results(sub_dir: Path) -> list[dict]:
    with (sub_dir / "results.json").open() as f:
        return json.load(f)


def plot_bars(sub_dir: Path, results: list[dict]) -> None:
    cells = sorted({(r["range_deg"], r["step_deg"]) for r in results})
    n_cells = len(cells)
    n_cols = min(3, n_cells)
    n_rows = (n_cells + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.0 * n_cols, 3.6 * n_rows),
                             squeeze=False)
    for k, (r, s) in enumerate(cells):
        ax = axes[k // n_cols, k % n_cols]
        rows = [next(rec for rec in results
                     if rec["range_deg"] == r and rec["step_deg"] == s
                     and rec["method"] == m)
                for m in METHOD_ORDER]
        n_ang = rows[0]["n_angles"]
        values = [rec["ssim_3d"] for rec in rows]
        colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
        bars = ax.bar(METHOD_ORDER, values, color=colors,
                      edgecolor="black", linewidth=0.4)
        ax.set_ylabel("SSIM (3D)", fontsize=9)
        ax.set_ylim(0.7, 1.0)
        ax.set_title(f"±{r:.0f}°, step {s:.0f}°  ({n_ang} angles)",
                     fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    max(v, 0.7) + 0.005,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=0)
    # Hide unused axes
    for k in range(len(cells), n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")
    fig.suptitle(f"SSIM (3D) — {sub_dir.name}", fontsize=12, y=1.00)
    fig.tight_layout()
    out = sub_dir / "summary_bars.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_heatmap(sub_dir: Path, results: list[dict]) -> None:
    ranges = sorted({r["range_deg"] for r in results}, reverse=True)
    if len(ranges) < 2:
        return  # nothing to heatmap
    step = results[0]["step_deg"]

    ssim_grid = np.zeros((len(METHOD_ORDER), len(ranges)), dtype=np.float32)
    for j, r in enumerate(ranges):
        for i, m in enumerate(METHOD_ORDER):
            rec = next(rec for rec in results
                       if rec["range_deg"] == r and rec["method"] == m)
            ssim_grid[i, j] = rec["ssim_3d"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(ssim_grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ranges)))
    ax.set_xticklabels([f"±{int(r)}°" for r in ranges])
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER)
    ax.set_title(f"SSIM (3D) — step {step:.0f}°", fontsize=11)
    mid = ssim_grid.mean()
    for j in range(ssim_grid.shape[1]):
        for i in range(ssim_grid.shape[0]):
            ax.text(j, i, f"{ssim_grid[i, j]:.3f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if ssim_grid[i, j] < mid else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"Method × tilt-range — {sub_dir.name}", fontsize=12)
    fig.tight_layout()
    out = sub_dir / "summary_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    for name in SWEEP_DIRS:
        sub_dir = OUT_ROOT / name
        results_path = sub_dir / "results.json"
        if not results_path.exists():
            print(f"skip {name} (no results.json)")
            continue
        results = load_results(sub_dir)
        print(f"\n=== {name} — {len(results)} runs ===")
        plot_bars(sub_dir, results)
        plot_heatmap(sub_dir, results)


if __name__ == "__main__":
    main()
