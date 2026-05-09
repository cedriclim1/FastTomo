"""Render the four presentation figures from the JSON outputs of
``benchmarks/run_presentation.py`` and ``benchmarks/run_scaling.py``.

Outputs PNGs into ``benchmarks/figures/``. Designed to drop into slides:
14 pt labels, dark-on-white, no extra chrome.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

PRES = json.loads((ROOT / "presentation_results.json").read_text())
SCALE = json.loads((ROOT / "scaling_results.json").read_text())

# Display labels and colour palette per algorithm family.
DISPLAY = {
    "ft-fbp-cached":          "FastTomo FBP",
    "ft-sirt-cached":         "FastTomo SIRT",
    "ft-resire":              "FastTomo RESIRE",
    "ft-genfire":             "FastTomo GENFIRE",
    "ft-ad-cached":           "FastTomo AD",
    "ft-ad-with-pose-cached": "FastTomo AD + Pose + TV",
    "ft-ad-pose-notv":        "FastTomo AD + Pose (no TV)",
    "ft-ad-pose-tv-ssim":     "FastTomo AD + Pose + TV + SSIM",
    "astra-sirt":             "ASTRA SIRT",
    "astra-fbp":              "ASTRA FBP",
    "torch_radon-sirt":       "TorchRadon iter-FBP",
    "torch_radon-fbp":        "TorchRadon FBP",
}

# Two-tone scheme: FastTomo highlighted, others muted.
FT_COLOR     = "#1f77b4"   # blue
FT_AD_COLOR  = "#d62728"   # red — flag the AD methods
ASTRA_COLOR  = "#7f7f7f"   # grey
TR_COLOR     = "#bcbd22"   # olive

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


def _by_method(rows, name):
    for r in rows:
        if r["method"] == name:
            return r
    return None


# ---------------------------------------------------------------------------
# Figure 1 — Speedup bar chart (iterations per second, log scale)
# ---------------------------------------------------------------------------
iter_methods = [
    "ft-sirt-cached",
    "ft-ad-cached",
    "ft-ad-with-pose-cached",
    "ft-resire",
    "ft-genfire",
    "astra-sirt",
    "torch_radon-sirt",
]
rows = PRES["rows"]
ips = [_by_method(rows, m)["it_per_s"] for m in iter_methods]
labels = [DISPLAY[m] for m in iter_methods]
colors = [color_for(m) for m in iter_methods]

fig, ax = plt.subplots(figsize=(8.5, 4.8))
order = np.argsort(ips)[::-1]
ips_s    = [ips[i]    for i in order]
labels_s = [labels[i] for i in order]
colors_s = [colors[i] for i in order]

bars = ax.barh(range(len(ips_s)), ips_s, color=colors_s, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(ips_s)))
ax.set_yticklabels(labels_s)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Iterations per second  (200³ volume, 29 angles)")
ax.set_title("Iterative reconstruction throughput  —  higher is better")
ax.grid(axis="x", which="both", alpha=0.3)

# Annotate each bar with the speedup vs ASTRA SIRT.
astra_ips = _by_method(rows, "astra-sirt")["it_per_s"]
for bar, val, m in zip(bars, ips_s,
                       [iter_methods[i] for i in order]):
    speedup = val / astra_ips
    if m == "astra-sirt":
        txt = f"{val:,.0f} it/s   (1× baseline)"
    else:
        sp = f"{speedup:.1f}×" if speedup < 5 else f"{speedup:.0f}×"
        txt = f"{val:,.0f} it/s   ({sp} ASTRA SIRT)"
    ax.text(val * 1.05, bar.get_y() + bar.get_height()/2, txt,
            va="center", fontsize=10)

ax.set_xlim(right=ax.get_xlim()[1] * 8)  # room for annotations
fig.savefig(FIG_DIR / "fig1_speedup_itps.png")
plt.close(fig)
print(f"saved {FIG_DIR/'fig1_speedup_itps.png'}")


# ---------------------------------------------------------------------------
# Figure 2 — Wall-clock time for matched workload (1000 iters, except TR=10)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.8))
times  = [_by_method(rows, m)["time_s"] for m in iter_methods]
niters = [_by_method(rows, m)["n_iter"] for m in iter_methods]

# Normalize all to "seconds for 1000 iterations" so it's a fair view.
times_1k = [t * (1000.0 / n) for t, n in zip(times, niters)]
order = np.argsort(times_1k)
ts_s     = [times_1k[i] for i in order]
labels_s = [labels[i]   for i in order]
colors_s = [colors[i]   for i in order]
methods_s = [iter_methods[i] for i in order]

bars = ax.barh(range(len(ts_s)), ts_s, color=colors_s, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(ts_s)))
ax.set_yticklabels(labels_s)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Wall-clock seconds for 1000 iterations  (200³ volume, 29 angles)")
ax.set_title("Time to converge  —  shorter is better")
ax.grid(axis="x", which="both", alpha=0.3)

for bar, val, m in zip(bars, ts_s, methods_s):
    note = ""
    if m == "torch_radon-sirt":
        note = "  (extrapolated — TR semi-converges at ~10 iter)"
    ax.text(val * 1.1, bar.get_y() + bar.get_height()/2,
            f"{val:.2f} s{note}", va="center", fontsize=10)
ax.set_xlim(right=ax.get_xlim()[1] * 8)
fig.savefig(FIG_DIR / "fig2_walltime_1000iter.png")
plt.close(fig)
print(f"saved {FIG_DIR/'fig2_walltime_1000iter.png'}")


# ---------------------------------------------------------------------------
# Figure 3 — SSIM bar chart over all methods
# ---------------------------------------------------------------------------
ssim_methods = [
    "ft-fbp-cached",
    "astra-fbp",
    "torch_radon-fbp",
    "torch_radon-sirt",
    "astra-sirt",
    "ft-genfire",
    "ft-resire",
    "ft-sirt-cached",
    "ft-ad-cached",
    "ft-ad-pose-notv",
    "ft-ad-with-pose-cached",
    "ft-ad-pose-tv-ssim",
]
fig, ax = plt.subplots(figsize=(9.0, 5.5))
ssim_methods_avail = [m for m in ssim_methods if _by_method(rows, m) is not None]
ssim_vals = [_by_method(rows, m)["ssim"] for m in ssim_methods_avail]
labels    = [DISPLAY[m] for m in ssim_methods_avail]
colors    = [color_for(m) for m in ssim_methods_avail]
order = np.argsort(ssim_vals)
ssim_s   = [ssim_vals[i] for i in order]
labels_s = [labels[i]    for i in order]
colors_s = [colors[i]    for i in order]
ssim_methods_s = [ssim_methods_avail[i] for i in order]

bars = ax.barh(range(len(ssim_s)), ssim_s, color=colors_s,
               edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(ssim_s)))
ax.set_yticklabels(labels_s)
ax.set_xlabel("3D SSIM vs ground-truth phantom  (1° tilt-axis-error data)")
ax.set_title("Reconstruction quality  —  higher is better")
ax.set_xlim(0.0, 1.05)
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, ssim_s):
    ax.text(val + 0.012, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
            va="center", fontsize=10)

# Legend by family
from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor=FT_AD_COLOR, edgecolor="black",
          label="FastTomo (autograd)"),
    Patch(facecolor=FT_COLOR,    edgecolor="black", label="FastTomo (numerical)"),
    Patch(facecolor=ASTRA_COLOR, edgecolor="black", label="ASTRA"),
    Patch(facecolor=TR_COLOR,    edgecolor="black", label="TorchRadon"),
]
ax.legend(handles=legend_items, loc="lower center",
          bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)
fig.savefig(FIG_DIR / "fig3_ssim.png")
plt.close(fig)
print(f"saved {FIG_DIR/'fig3_ssim.png'}")


# ---------------------------------------------------------------------------
# Figure 4 — Volume-size scaling (log-log) from scaling_results.json
# ---------------------------------------------------------------------------
scale_rows = SCALE["rows"]
sizes = sorted({r["N"] for r in scale_rows})
methods_sc = sorted({r["method"] for r in scale_rows})

fig, ax = plt.subplots(figsize=(7.5, 5.0))
sc_disp = {
    "ft-sirt-cached":          ("FastTomo SIRT",      FT_COLOR,    "o"),
    "ft-ad-with-pose-cached":  ("FastTomo AD + pose", FT_AD_COLOR, "s"),
}
for m, (label, color, marker) in sc_disp.items():
    rows_m = [r for r in scale_rows if r["method"] == m]
    rows_m.sort(key=lambda r: r["N"])
    xs = [r["N"]      for r in rows_m]
    ys = [r["time_s"] for r in rows_m]
    ax.loglog(xs, ys, marker=marker, color=color, linewidth=2,
              markersize=8, label=label)

# Reference O(N^3) line, anchored to the smallest SIRT point
sirt0 = next(r for r in scale_rows
             if r["method"] == "ft-sirt-cached" and r["N"] == sizes[0])
n0, t0 = sirt0["N"], sirt0["time_s"]
xs_ref = np.array(sizes, dtype=float)
ys_ref = t0 * (xs_ref / n0) ** 3
ax.loglog(xs_ref, ys_ref, "k--", alpha=0.5, label="O(N³) reference")

ax.set_xlabel("Cubic volume edge N (voxels)")
ax.set_ylabel(f"Wall-clock time for 100 iterations (s)")
ax.set_title("Volume-size scaling  —  29 angles, cached projector")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left")

# Annotate each AD point with the AD/SIRT ratio.
ad_rows   = sorted([r for r in scale_rows if r["method"] == "ft-ad-with-pose-cached"],
                   key=lambda r: r["N"])
sirt_rows = sorted([r for r in scale_rows if r["method"] == "ft-sirt-cached"],
                   key=lambda r: r["N"])
for ad, sirt in zip(ad_rows, sirt_rows):
    ratio = ad["time_s"] / sirt["time_s"]
    ax.annotate(f"{ratio:.0f}×", xy=(ad["N"], ad["time_s"]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=10, color=FT_AD_COLOR)

fig.savefig(FIG_DIR / "fig4_volume_scaling.png")
plt.close(fig)
print(f"saved {FIG_DIR/'fig4_volume_scaling.png'}")


# ---------------------------------------------------------------------------
# Figure 5 — CPU throughput (FastTomo CPU vs ASTRA CPU, OpenMP)
# ---------------------------------------------------------------------------
cpu_rows = PRES.get("cpu_rows", [])
if cpu_rows:
    DISPLAY_CPU = {
        "ft-fbp-cpu":          "FastTomo CPU FBP",
        "ft-sirt-cpu":         "FastTomo CPU SIRT",
        "ft-ad-cpu":           "FastTomo CPU AD",
        "ft-ad-with-pose-cpu": "FastTomo CPU AD + pose",
        "astra-fbp-cpu":       "ASTRA CPU FBP",
        "astra-sirt-cpu":      "ASTRA CPU SIRT",
    }
    cpu_methods = ["ft-sirt-cpu", "ft-ad-cpu", "ft-ad-with-pose-cpu",
                   "astra-sirt-cpu", "ft-fbp-cpu",  "astra-fbp-cpu"]
    cpu_ips = []
    cpu_labels = []
    cpu_colors = []
    for m in cpu_methods:
        r = _by_method(cpu_rows, m)
        if r is None:
            continue
        cpu_ips.append(r["it_per_s"])
        cpu_labels.append(DISPLAY_CPU[m])
        # color_for needs the original (pre-cpu) method root
        cpu_colors.append(color_for(m.replace("-cpu", "")))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    order = np.argsort(cpu_ips)[::-1]
    cpu_ips_s    = [cpu_ips[i]    for i in order]
    cpu_labels_s = [cpu_labels[i] for i in order]
    cpu_colors_s = [cpu_colors[i] for i in order]
    cpu_methods_o = [cpu_methods[i] for i in order]

    bars = ax.barh(range(len(cpu_ips_s)), cpu_ips_s,
                   color=cpu_colors_s, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(cpu_ips_s)))
    ax.set_yticklabels(cpu_labels_s)
    ax.invert_yaxis()
    ax.set_xscale("log")
    omp = PRES.get("omp_num_threads", 32)
    ax.set_xlabel(f"Iterations per second  (200³ volume, 29 angles, "
                  f"OpenMP × {omp} threads)")
    ax.set_title("CPU throughput  —  higher is better")
    ax.grid(axis="x", which="both", alpha=0.3)

    # Speedup vs ASTRA CPU SIRT for the SIRT pair, vs ASTRA CPU FBP for the FBP pair.
    base_sirt = _by_method(cpu_rows, "astra-sirt-cpu")
    base_fbp  = _by_method(cpu_rows, "astra-fbp-cpu")
    base_sirt_v = base_sirt["it_per_s"] if base_sirt else None
    base_fbp_v  = base_fbp["it_per_s"]  if base_fbp  else None
    for bar, val, m in zip(bars, cpu_ips_s, cpu_methods_o):
        if m == "astra-sirt-cpu":
            txt = f"{val:.2f} it/s   (1× baseline)"
        elif m == "astra-fbp-cpu":
            txt = f"{val:.2f} it/s   (1× FBP baseline)"
        elif m == "ft-sirt-cpu" and base_sirt_v:
            sp = val / base_sirt_v
            txt = f"{val:.2f} it/s   ({sp:.1f}× ASTRA CPU SIRT)"
        elif m == "ft-fbp-cpu" and base_fbp_v:
            sp = val / base_fbp_v
            txt = f"{val:.2f} it/s   ({sp:.1f}× ASTRA CPU FBP)"
        elif m in ("ft-ad-cpu", "ft-ad-with-pose-cpu") and base_sirt_v:
            sp = val / base_sirt_v
            txt = f"{val:.2f} it/s   ({sp:.1f}× ASTRA CPU SIRT)"
        else:
            txt = f"{val:.2f} it/s"
        ax.text(val * 1.05, bar.get_y() + bar.get_height()/2, txt,
                va="center", fontsize=10)
    ax.set_xlim(right=ax.get_xlim()[1] * 8)
    fig.savefig(FIG_DIR / "fig5_cpu_throughput.png")
    plt.close(fig)
    print(f"saved {FIG_DIR/'fig5_cpu_throughput.png'}")


# ---------------------------------------------------------------------------
# Figure 6 — CPU vs GPU throughput across algorithms (SIRT + AD families)
# ---------------------------------------------------------------------------
if cpu_rows:
    cpugpu_methods = [
        ("astra-sirt-cpu",         "ASTRA SIRT  (CPU)",         ASTRA_COLOR, "//"),
        ("ft-sirt-cpu",            "FastTomo SIRT  (CPU)",      FT_COLOR,    "//"),
        ("ft-ad-cpu",              "FastTomo AD  (CPU)",        FT_AD_COLOR, "//"),
        ("ft-ad-with-pose-cpu",    "FastTomo AD + pose  (CPU)", FT_AD_COLOR, "//"),
        ("astra-sirt",             "ASTRA SIRT  (GPU)",         ASTRA_COLOR, ""),
        ("ft-sirt-cached",         "FastTomo SIRT  (GPU)",      FT_COLOR,    ""),
        ("ft-ad-cached",           "FastTomo AD  (GPU)",        FT_AD_COLOR, ""),
        ("ft-ad-with-pose-cached", "FastTomo AD + pose  (GPU)", FT_AD_COLOR, ""),
    ]
    vals: list[float] = []
    lbls: list[str]   = []
    cols: list[str]   = []
    hats: list[str]   = []
    for m, label, color, hatch in cpugpu_methods:
        r = _by_method(cpu_rows, m) or _by_method(rows, m)
        if r is None:
            continue
        vals.append(r["it_per_s"])
        lbls.append(label)
        cols.append(color)
        hats.append(hatch)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    order = np.argsort(vals)
    vals_s = [vals[i] for i in order]
    lbls_s = [lbls[i] for i in order]
    cols_s = [cols[i] for i in order]
    hats_s = [hats[i] for i in order]

    bars = ax.barh(range(len(vals_s)), vals_s, color=cols_s,
                   edgecolor="black", linewidth=0.5, hatch=hats_s)
    ax.set_yticks(range(len(vals_s)))
    ax.set_yticklabels(lbls_s)
    ax.set_xscale("log")
    omp = PRES.get("omp_num_threads", 32)
    ax.set_xlabel(f"Iterations per second  (200³, 29 angles; CPU = OpenMP × {omp})")
    ax.set_title("CPU vs GPU across algorithms  —  hatched = CPU, solid = GPU")
    ax.grid(axis="x", which="both", alpha=0.3)

    base = _by_method(cpu_rows, "astra-sirt-cpu")["it_per_s"]
    for bar, val in zip(bars, vals_s):
        sp = val / base
        sp_txt = f"{sp:.1f}×" if sp < 5 else f"{sp:,.0f}×"
        ax.text(val * 1.05, bar.get_y() + bar.get_height()/2,
                f"{val:>9,.2f} it/s   ({sp_txt} ASTRA CPU SIRT)",
                va="center", fontsize=10)
    ax.set_xlim(right=ax.get_xlim()[1] * 60)
    fig.savefig(FIG_DIR / "fig6_cpu_vs_gpu.png")
    plt.close(fig)
    print(f"saved {FIG_DIR/'fig6_cpu_vs_gpu.png'}")


print("\nAll figures written to:", FIG_DIR)
