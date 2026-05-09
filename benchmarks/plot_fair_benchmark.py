"""Plot fair benchmark results — perfect tilt series, no pose optimization.

Reads ``benchmarks/fair_benchmark_results.json`` and renders two panels:
  Left  — SSIM bar chart sorted ascending (higher = better)
  Right — Iterations-per-second bar chart, log scale (higher = better)

Output: ``benchmarks/figures/fig_fair_ssim.png``

Usage:
    python benchmarks/plot_fair_benchmark.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

DATA = json.loads((ROOT / "fair_benchmark_results.json").read_text())
rows = DATA["rows"]

DISPLAY = {
    "ft-fbp":            "FastTomo FBP",
    "ft-sirt":           "FastTomo SIRT",
    "ft-sart":           "FastTomo SART",
    "ft-wsirt":          "FastTomo W-SIRT",
    "ft-resire":         "FastTomo RESIRE",
    "ft-genfire":        "FastTomo GENFIRE",
    "ft-ad":             "FastTomo AD",
    "ft-ad-tv":          "FastTomo AD + TV",
    "ft-ad-tv-ssim":     "FastTomo AD + TV + SSIM",
    "astra-sirt":        "ASTRA SIRT",
    "astra-fbp":         "ASTRA FBP",
    "torch_radon-sirt":  "TorchRadon iter-FBP",
    "torch_radon-fbp":   "TorchRadon FBP",
}

FT_COLOR    = "#1f77b4"   # blue
FT_AD_COLOR = "#d62728"   # red
ASTRA_COLOR = "#7f7f7f"   # grey
TR_COLOR    = "#bcbd22"   # olive


def color_for(method: str) -> str:
    if method.startswith("ft-ad"):
        return FT_AD_COLOR
    if method.startswith("ft-"):
        return FT_COLOR
    if method.startswith("astra"):
        return ASTRA_COLOR
    if method.startswith("torch_radon"):
        return TR_COLOR
    return "#444444"


def _by_method(name: str) -> dict | None:
    for r in rows:
        if r["method"] == name:
            return r
    return None


plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

# Ordered list for display (all known methods; missing ones are skipped).
ALL_METHODS = [
    "ft-fbp",
    "astra-fbp",
    "torch_radon-fbp",
    "torch_radon-sirt",
    "astra-sirt",
    "ft-wsirt",
    "ft-genfire",
    "ft-resire",
    "ft-sart",
    "ft-sirt",
    "ft-ad",
    "ft-ad-tv",
    "ft-ad-tv-ssim",
]
avail = [m for m in ALL_METHODS if _by_method(m) is not None]

# Iterative methods only (for it/s panel — exclude one-shot FBP).
ITER_METHODS = [m for m in avail if _by_method(m)["n_iter"] > 1]

legend_items = [
    Patch(facecolor=FT_AD_COLOR, edgecolor="black", label="FastTomo (autograd)"),
    Patch(facecolor=FT_COLOR,    edgecolor="black", label="FastTomo (numerical)"),
    Patch(facecolor=ASTRA_COLOR, edgecolor="black", label="ASTRA"),
    Patch(facecolor=TR_COLOR,    edgecolor="black", label="TorchRadon"),
]

fig, (ax_ssim, ax_ips) = plt.subplots(1, 2, figsize=(16.0, 6.0))
fig.suptitle(
    "Reconstruction quality & speed — perfect tilt series, no pose optimization",
    fontsize=14, y=1.01,
)

# ---------------------------------------------------------------------------
# Left panel: SSIM
# ---------------------------------------------------------------------------
ssim_vals  = [_by_method(m)["ssim"]   for m in avail]
labels     = [DISPLAY.get(m, m)       for m in avail]
colors     = [color_for(m)            for m in avail]

order = np.argsort(ssim_vals)          # ascending → best bar on top after invert_yaxis
ssim_s   = [ssim_vals[i] for i in order]
labels_s = [labels[i]    for i in order]
colors_s = [colors[i]    for i in order]

bars = ax_ssim.barh(range(len(ssim_s)), ssim_s, color=colors_s,
                    edgecolor="black", linewidth=0.5)
ax_ssim.set_yticks(range(len(ssim_s)))
ax_ssim.set_yticklabels(labels_s)
ax_ssim.invert_yaxis()
ax_ssim.set_xlabel("3D SSIM vs ground-truth phantom  (perfect data)")
ax_ssim.set_title("Reconstruction quality  —  higher is better")
ax_ssim.set_xlim(0.0, 1.05)
ax_ssim.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, ssim_s):
    ax_ssim.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=10)


# ---------------------------------------------------------------------------
# Right panel: iterations per second (log scale, iterative methods only)
# ---------------------------------------------------------------------------
ips_vals  = [_by_method(m)["it_per_s"] for m in ITER_METHODS]
it_labels = [DISPLAY.get(m, m)         for m in ITER_METHODS]
it_colors = [color_for(m)              for m in ITER_METHODS]

order2 = np.argsort(ips_vals)[::-1]
ips_s   = [ips_vals[i]  for i in order2]
ilabs_s = [it_labels[i] for i in order2]
icols_s = [it_colors[i] for i in order2]
meths_s = [ITER_METHODS[i] for i in order2]

bars2 = ax_ips.barh(range(len(ips_s)), ips_s, color=icols_s,
                    edgecolor="black", linewidth=0.5)
ax_ips.set_yticks(range(len(ips_s)))
ax_ips.set_yticklabels(ilabs_s)
ax_ips.invert_yaxis()
ax_ips.set_xscale("log")
ax_ips.set_xlabel("Iterations per second  (200³ volume, 29 angles)")
ax_ips.set_title("Throughput  —  higher is better")
ax_ips.grid(axis="x", which="both", alpha=0.3)

# Annotate each bar with it/s value; flag TorchRadon extrapolation.
for bar, val, m in zip(bars2, ips_s, meths_s):
    note = "  (extrapolated)" if m == "torch_radon-sirt" else ""
    ax_ips.text(val * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f} it/s{note}", va="center", fontsize=10)
ax_ips.set_xlim(right=ax_ips.get_xlim()[1] * 6)

fig.tight_layout()
fig.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_items, loc="lower center",
           bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=11)
out = FIG_DIR / "fig_fair_ssim.png"
fig.savefig(out)
plt.close(fig)
print(f"saved {out}")
