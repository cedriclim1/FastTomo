"""Method comparison sweep — 10 reconstruction algorithms × 4 tilt configurations.

Compares classical methods (FBP / SIRT / SART / GENFIRE / RESIRE) against the
AD family (plain MSE, +TV, +TV+SSIM-local, +λ·S3IM, +TV+λ·S3IM) on the
non-atomic quantem phantom with perfect (no-noise, no-pose-error) sinograms.

Sweep groups (each lands in benchmarks/method_comparison/<sub-dir>/):
    01_70tr_2s     single cell: ±70°, 2° step (71 angles)
    02_range_s5    ±r ∈ {70,60,50,40,30,20}, step 5°
    03_range_s2    ±r ∈ {70,60,50,40,30,20}, step 2°
    04_range_s10   ±r ∈ {70,60,50,40,30,20}, step 10°

Hyperparameters held constant for this pass:
    TV weight = 0.1, SSIM weight = 0.1, λ_S3IM = 0.1
    AD: 1000 iter, Adam lr=0.05, cosine schedule, clamp ≥0
    Classical methods: library defaults

CLI:
    python benchmarks/run_method_comparison.py --sweep 01     # smoke
    python benchmarks/run_method_comparison.py --sweep all    # full (~110 min)
"""

from __future__ import annotations

import argparse
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
OUT_ROOT = REPO_ROOT / "benchmarks" / "results" / "method_comparison"

# Hyperparameters (locked for this pass)
TV_WEIGHT = 3.0
SSIM_WEIGHT = 0.1
LAMBDA_S3IM = 0.1
S3IM_PATCH_PIXELS = 4096
S3IM_KERNEL_SIZE = 4
S3IM_N_PATCHES = 1
AD_N_ITER = 1000
AD_LR = 0.05
SEED = 0

# Classical method iteration counts (library defaults)
SIRT_N_ITER = 50
SART_N_ITER = 5
GENFIRE_N_ITER = 100
RESIRE_N_ITER = 100

METHOD_ORDER = [
    "fbp", "sirt", "sart", "genfire", "resire",
    "ad", "ad_tv", "ad_tv_ssim", "ad_s3im", "ad_tv_s3im",
]

AD_RECIPES = {
    "ad":          {"tv": False, "ssim": False, "s3im": False},
    "ad_tv":       {"tv": True,  "ssim": False, "s3im": False},
    "ad_tv_ssim":  {"tv": True,  "ssim": True,  "s3im": False},
    "ad_s3im":     {"tv": False, "ssim": False, "s3im": True},
    "ad_tv_s3im":  {"tv": True,  "ssim": False, "s3im": True},
}

SWEEPS: dict[str, list[tuple[float, float]]] = {
    "01": [(70.0, 2.0)],
    "02": [(70.0, 5.0), (60.0, 5.0), (50.0, 5.0), (40.0, 5.0), (30.0, 5.0), (20.0, 5.0)],
    "03": [(70.0, 2.0), (60.0, 2.0), (50.0, 2.0), (40.0, 2.0), (30.0, 2.0), (20.0, 2.0)],
    "04": [(70.0, 10.0), (60.0, 10.0), (50.0, 10.0), (40.0, 10.0), (30.0, 10.0), (20.0, 10.0)],
}
SWEEP_DIRS = {
    "01": "01_70tr_2s",
    "02": "02_range_s5",
    "03": "03_range_s2",
    "04": "04_range_s10",
}


@dataclass
class Result:
    sweep: str
    range_deg: float
    step_deg: float
    n_angles: int
    method: str
    nmse: float
    ssim_3d: float
    time_s: float
    ad_losses: list[float] = field(default_factory=list)


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


def run_classical(method: str, sino: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    if method == "fbp":
        return ft.run_fbp(sino, angles, positivity=True, mask=True)
    if method == "sirt":
        return ft.run_sirt(sino, angles, n_iter=SIRT_N_ITER,
                           positivity=True, mask=True)
    if method == "sart":
        return ft.run_sart(sino, angles, n_iter=SART_N_ITER,
                           positivity=True, mask=True)
    if method == "genfire":
        return ft.run_genfire(sino, angles, n_iter=GENFIRE_N_ITER)
    if method == "resire":
        return ft.run_resire(sino, angles, n_iter=RESIRE_N_ITER)
    raise ValueError(f"unknown classical method: {method}")


def reconstruct_ad(
    sino: torch.Tensor, angles: torch.Tensor,
    *, recipe: str, seed: int = SEED,
) -> tuple[torch.Tensor, list[float]]:
    flags = AD_RECIPES[recipe]
    n_angles, Ny, N_det = sino.shape
    proj = ft.ParallelProjector((Ny, N_det, N_det), angles, device=sino.device,
                                N_det=N_det)
    torch.manual_seed(seed)
    vol = torch.zeros(Ny, N_det, N_det, device=sino.device, requires_grad=True)
    opt = torch.optim.Adam([vol], lr=AD_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, AD_N_ITER)
    s3im_gen = torch.Generator(device=sino.device).manual_seed(seed + 1)

    losses: list[float] = []
    for _ in range(AD_N_ITER):
        opt.zero_grad(set_to_none=True)
        pred = proj.forward(vol)
        loss = F.mse_loss(pred, sino)
        if flags["tv"]:
            loss = loss + TV_WEIGHT * ft.tv_loss_3d(vol)
        if flags["ssim"]:
            loss = loss + SSIM_WEIGHT * ft.tilt_ssim_loss(pred, sino)
        if flags["s3im"]:
            loss = loss + LAMBDA_S3IM * ft.s3im_loss(
                pred, sino,
                patch_pixels=S3IM_PATCH_PIXELS, kernel_size=S3IM_KERNEL_SIZE,
                n_patches=S3IM_N_PATCHES, generator=s3im_gen,
            )
        losses.append(float(loss.item()))
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            vol.clamp_(min=0)
    return vol.detach(), losses


def axis_projections(v: np.ndarray):
    return v.sum(axis=0), v.sum(axis=1), v.sum(axis=2)


def middle_slices(v: np.ndarray):
    Ny, Nz, Nx = v.shape
    return v[Ny // 2], v[:, Nz // 2], v[:, :, Nx // 2]


PROJ_ROW_LABELS = (
    "Σ along Y\n(view ZX)",
    "Σ along Z\n(view YX)",
    "Σ along X\n(view YZ)",
)
SLICE_ROW_LABELS = (
    "mid-Y slice\n(view ZX)",
    "mid-Z slice\n(view YX)",
    "mid-X slice\n(view YZ)",
)


def emit_grid(
    rows: list[list[np.ndarray]],
    col_titles: list[str],
    row_labels: tuple[str, ...],
    out_path: Path,
    suptitle: str,
    cmap: str = "turbo",
    vmin: float | None = 0.0,
) -> None:
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols, 2.6 * n_rows),
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
                ax.set_title(col_titles[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_sweep(sweep_id: str, device: torch.device) -> None:
    cells = SWEEPS[sweep_id]
    out_dir = OUT_ROOT / SWEEP_DIRS[sweep_id]
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    phantom = load_phantom(device)
    phantom_np = phantom.detach().cpu().numpy().astype(np.float32)

    results: list[Result] = []
    vol_cache: dict[tuple[float, float, str], np.ndarray] = {}

    total = len(cells) * len(METHOD_ORDER)
    idx = 0
    for r, s in cells:
        n_angles = int(round(2 * r / s)) + 1
        angles = torch.linspace(-r, r, n_angles, device=device)
        with torch.no_grad():
            sino = ft.forward_project(phantom, angles)
        print(f"\n=== [{sweep_id}] ±{r:.0f}°, step {s:.0f}° → {n_angles} angles ===")

        for method in METHOD_ORDER:
            idx += 1
            t0 = time.perf_counter()
            ad_losses: list[float] = []
            if method in AD_RECIPES:
                vol, ad_losses = reconstruct_ad(sino, angles, recipe=method)
            else:
                vol = run_classical(method, sino, angles)
            dt = time.perf_counter() - t0
            nm = nmse(vol, phantom)
            ss = ssim3d(vol, phantom)
            print(f"[{idx:>3}/{total}] {method:<12s} "
                  f"NMSE={nm:.4f}  SSIM3D={ss:.4f}  ({dt:.1f}s)")
            results.append(Result(
                sweep=sweep_id, range_deg=r, step_deg=s, n_angles=n_angles,
                method=method, nmse=nm, ssim_3d=ss, time_s=dt,
                ad_losses=ad_losses,
            ))
            vol_cache[(r, s, method)] = vol.detach().cpu().numpy().astype(np.float32)
            del vol; torch.cuda.empty_cache()

    # Save JSON + NPZ
    with (out_dir / "results.json").open("w") as f:
        json.dump([asdict(rec) for rec in results], f, indent=2)
    print(f"\nSaved {len(results)} results → {out_dir.relative_to(REPO_ROOT)}/results.json")

    npz_payload = {"phantom": phantom_np}
    for (r, s, m), arr in vol_cache.items():
        npz_payload[f"r{int(r)}_s{int(s)}_{m}"] = arr
    np.savez_compressed(out_dir / "recons.npz", **npz_payload)
    print(f"Saved phantom + {len(vol_cache)} recons → {out_dir.relative_to(REPO_ROOT)}/recons.npz")

    # Per-cell projection + slice grids
    print("\nGenerating per-cell figures:")
    for r, s in cells:
        cols = [phantom_np]
        titles = ["GT"]
        for method in METHOD_ORDER:
            cols.append(vol_cache[(r, s, method)])
            rec = next(rr for rr in results
                       if rr.range_deg == r and rr.step_deg == s and rr.method == method)
            titles.append(f"{method}\nNMSE={rec.nmse:.3f}\nSSIM={rec.ssim_3d:.3f}")
        tag = f"r{int(r)}_s{int(s)}"

        projs = [axis_projections(v) for v in cols]
        proj_rows = [[projs[c][axis] for c in range(len(cols))] for axis in range(3)]
        emit_grid(
            proj_rows, titles, PROJ_ROW_LABELS,
            cells_dir / f"{tag}_projections.png",
            suptitle=f"Axis-summed projections — ±{r:.0f}°, step {s:.0f}°  ({sweep_id})",
        )
        print(f"  wrote {cells_dir.relative_to(REPO_ROOT)}/{tag}_projections.png")

        sl = [middle_slices(v) for v in cols]
        slice_rows = [[sl[c][axis] for c in range(len(cols))] for axis in range(3)]
        emit_grid(
            slice_rows, titles, SLICE_ROW_LABELS,
            cells_dir / f"{tag}_slices.png",
            suptitle=f"Central slices — ±{r:.0f}°, step {s:.0f}°  ({sweep_id})",
        )
        print(f"  wrote {cells_dir.relative_to(REPO_ROOT)}/{tag}_slices.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=["01", "02", "03", "04", "all"],
                        default="all", help="which sub-sweep to run (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    ids = ["01", "02", "03", "04"] if args.sweep == "all" else [args.sweep]
    t_total = time.perf_counter()
    for sid in ids:
        run_sweep(sid, device)
    print(f"\nAll done in {(time.perf_counter() - t_total) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
