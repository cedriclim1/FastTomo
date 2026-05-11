"""Focused λ sweep on the multiplex + TV recipe at the easy cell.

Runs the ``L2 + TV + λ·S3IM`` reconstruction at ±70° / 5° step (29 angles, no
missing wedge) for a fine λ sweep, holding TV at the project default. Also runs
the four reference recipes (L2, L2+TV, L2+TV+SSIM_local, plain multiplex at the
same λ values, no TV) once each at this cell for context.

Useful for picking a good λ when angular coverage is full: in this regime the
multiplex term is competing only against measurement noise, not missing-wedge
artefacts, so the optimum is much lower than in the missing-wedge cells.

Outputs:
    benchmarks/ssim-ablation/lambda_focus_results.json
    benchmarks/ssim-ablation/lambda_focus.png
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
OUT_DIR = REPO_ROOT / "benchmarks" / "results" / "ssim-ablation"
OUT_JSON = OUT_DIR / "lambda_focus_results.json"
OUT_PNG = OUT_DIR / "lambda_focus.png"

# Focused experiment knobs
RANGE_DEG = 70.0
STEP_DEG = 5.0
TV_WEIGHT = 1.0           # default; what every other comparator uses too
SSIM_WEIGHT = 0.5         # only used by mse_tv_ssim reference
LAMBDAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
S3IM_PATCH_PIXELS = 4096
S3IM_KERNEL_SIZE = 4
S3IM_N_PATCHES = 1
N_ITER = 1000
LR = 0.05
SEED = 0


@dataclass
class Result:
    recipe: str               # human-readable label
    lam: float                # λ on S3IM term (0 if not used)
    tv_weight: float
    nmse: float
    ssim_3d: float
    final_loss: float
    time_s: float
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
    *, use_tv: bool, use_local_ssim: bool, use_s3im: bool,
    lam: float, tv_weight: float, seed: int,
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
        loss = F.mse_loss(pred, sino)
        if use_tv:
            loss = loss + tv_weight * ft.tv_loss_3d(vol)
        if use_local_ssim:
            loss = loss + SSIM_WEIGHT * ft.tilt_ssim_loss(pred, sino)
        if use_s3im:
            loss = loss + lam * ft.s3im_loss(
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


def main() -> None:
    device = torch.device("cuda")
    phantom = load_phantom(device)
    n_angles = int(round(2 * RANGE_DEG / STEP_DEG)) + 1
    angles = torch.linspace(-RANGE_DEG, RANGE_DEG, n_angles, device=device)
    with torch.no_grad():
        sino = ft.forward_project(phantom, angles)
    print(f"sinogram {tuple(sino.shape)} — ±{RANGE_DEG:.0f}°, step {STEP_DEG:.0f}°")

    results: list[Result] = []

    # ── References (one run each) ────────────────────────────────────
    refs = [
        ("L2",                    dict(use_tv=False, use_local_ssim=False, use_s3im=False)),
        ("L2 + TV",               dict(use_tv=True,  use_local_ssim=False, use_s3im=False)),
        ("L2 + TV + SSIM (local)",dict(use_tv=True,  use_local_ssim=True,  use_s3im=False)),
    ]
    for label, kwargs in refs:
        t0 = time.perf_counter()
        vol, losses = reconstruct(
            sino, angles, **kwargs, lam=0.0, tv_weight=TV_WEIGHT, seed=SEED,
        )
        dt = time.perf_counter() - t0
        nm = nmse(vol, phantom); ss = ssim3d(vol, phantom)
        print(f"  {label:<30s}  NMSE={nm:.4f}  SSIM3D={ss:.4f}  ({dt:.1f}s)")
        results.append(Result(label, 0.0, TV_WEIGHT if "TV" in label else 0.0,
                              nm, ss, losses[-1], dt, losses))
        del vol; torch.cuda.empty_cache()

    # ── λ sweep for the two multiplex recipes ────────────────────────
    for label, kwargs in (
        ("Multiplex (L2 + λ·S3IM)",   dict(use_tv=False, use_local_ssim=False, use_s3im=True)),
        ("L2 + TV + λ·S3IM",          dict(use_tv=True,  use_local_ssim=False, use_s3im=True)),
    ):
        for lam in LAMBDAS:
            t0 = time.perf_counter()
            vol, losses = reconstruct(
                sino, angles, **kwargs, lam=lam, tv_weight=TV_WEIGHT, seed=SEED,
            )
            dt = time.perf_counter() - t0
            nm = nmse(vol, phantom); ss = ssim3d(vol, phantom)
            print(f"  {label:<30s} λ={lam:>7.3f}  NMSE={nm:.4f}  SSIM3D={ss:.4f}  ({dt:.1f}s)")
            results.append(Result(label, lam,
                                  TV_WEIGHT if "TV" in label else 0.0,
                                  nm, ss, losses[-1], dt, losses))
            del vol; torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved {len(results)} results → {OUT_JSON.relative_to(REPO_ROOT)}")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ref_colors = {
        "L2": "#888888", "L2 + TV": "#1f77b4",
        "L2 + TV + SSIM (local)": "#2ca02c",
    }
    sweep_colors = {
        "Multiplex (L2 + λ·S3IM)": "#d62728",
        "L2 + TV + λ·S3IM":        "#9467bd",
    }

    for ax, metric, label, lower_better in (
        (axes[0], "nmse",    "NMSE",          True),
        (axes[1], "ssim_3d", "SSIM (3D)",     False),
    ):
        # Reference horizontal lines
        for ref_label in ref_colors:
            rec = next(r for r in results if r.recipe == ref_label)
            ax.axhline(getattr(rec, metric), color=ref_colors[ref_label],
                       linestyle="--", lw=1.2, label=ref_label)
        # Multiplex sweep curves
        for sweep_label, c in sweep_colors.items():
            xs, ys = [], []
            for r in results:
                if r.recipe == sweep_label:
                    xs.append(r.lam); ys.append(getattr(r, metric))
            ax.plot(xs, ys, marker="o", color=c, lw=1.5, label=sweep_label)
        ax.set_xscale("log")
        ax.set_xlabel("λ")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        title = f"{label} vs λ — ±{RANGE_DEG:.0f}°, {STEP_DEG:.0f}° step, TV={TV_WEIGHT}"
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {OUT_PNG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
