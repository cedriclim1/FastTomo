"""SSIM-loss ablation on the non-atomic quantem phantom.

Compares five AD-loss recipes across a 3×3 grid of (tilt range × tilt step):

    1. ``mse``           — F.mse_loss(pred, sino)
    2. ``mse_tv``        — + tv_w · tv_loss_3d(vol)
    3. ``mse_tv_ssim``   — + tv_w · tv + ssim_w · tilt_ssim_loss(pred, sino)
    4. ``mse_s3im``      — F.mse_loss + λ · s3im_loss(pred, sino)            (paper-strict multiplex)
    5. ``mse_tv_s3im``   — F.mse_loss + tv_w · tv + λ · s3im_loss(pred, sino) (stacked)

For the multiplex cells (4, 5) the script sweeps λ ∈ {0.1, 1.0, 10.0}, giving
9 × (1 + 1 + 1 + 3 + 3) = 81 reconstructions in the full sweep. Each
reconstruction is a 1000-iter Adam loop over a fresh zero-init volume; the
ground-truth phantom and a freshly-forward-projected (zero pose-error)
sinogram are used so the only difference between cells is the loss function.

Outputs:
    benchmarks/ssim_ablation_results.json   — list of result records
    data/ssim_ablation_recons.npz           — top-1 volume per (config, loss kind)

Smoke test:  python benchmarks/run_ssim_ablation.py --quick
Full sweep:  python benchmarks/run_ssim_ablation.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim

import fasttomo as ft

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
QUANTEM_DATA_DIR = Path("/home/cedlim/quantem/quantem-tutorials/data")
OUT_DIR = REPO_ROOT / "benchmarks" / "results" / "ssim-ablation-03-tv3-tiltsweep"
OUT_JSON = OUT_DIR / "results.json"
OUT_RECONS = OUT_DIR / "recons.npz"


# ── Sweep grid ────────────────────────────────────────────────────────────

# Expanded tilt grid for direct cross-cell comparison.
TILT_RANGES = (70.0, 55.0, 40.0, 25.0)       # ±deg
TILT_STEPS = (2.0, 5.0, 8.0, 12.0)           # deg
# TV weight pinned at 3 across every TV-using recipe — the "unified
# regularizer strength" knob the user wanted in this comparison.
LAMBDA_S3IM = (1.0,)
TV_WEIGHTS_S3IM = (3.0,)
LOSS_KINDS = ("mse", "mse_tv", "mse_tv_ssim", "mse_s3im", "mse_tv_s3im")

# Fixed regulariser hyperparams
TV_WEIGHT_DEFAULT = 3.0
SSIM_WEIGHT = 0.5            # weight on tilt_ssim_loss in mse_tv_ssim cell (original default)
S3IM_PATCH_PIXELS = 4096     # 64×64 stochastic patch
S3IM_KERNEL_SIZE = 4
S3IM_N_PATCHES = 1           # M=1; paper notes M=1 ≈ M=10
N_ITER = 1000
LR = 0.05
SEED = 0


@dataclass
class Result:
    range_deg: float
    step_deg: float
    loss_kind: str
    lam: float
    tv_weight: float
    nmse: float
    ssim: float              # mean of per-XZ-slice 2D SSIM (legacy)
    ssim_3d: float           # full-volume 3D SSIM via skimage
    final_loss: float
    time_s: float
    n_angles: int
    losses: list[float] = field(default_factory=list)


# ── Phantom + on-the-fly perfect sinogram ─────────────────────────────────

def load_phantom(device: torch.device) -> torch.Tensor:
    """Load the quantem phantom with the same axis reorient as load_fair_tilt_series."""
    phantom_np = np.load(QUANTEM_DATA_DIR / "phantom.npy").astype(np.float32)
    phantom_np = phantom_np.transpose(2, 0, 1)[::-1].copy()
    return torch.from_numpy(phantom_np).to(device)


def make_perfect_sino(
    phantom: torch.Tensor, range_deg: float, step_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-project the phantom over the requested tilt schedule."""
    n_angles = int(round(2 * range_deg / step_deg)) + 1
    angles = torch.linspace(-range_deg, range_deg, n_angles, device=phantom.device)
    with torch.no_grad():
        sino = ft.forward_project(phantom, angles)
    return sino, angles


# ── Metrics ───────────────────────────────────────────────────────────────

def nmse(vol: torch.Tensor, gt: torch.Tensor) -> float:
    return float(((vol - gt) ** 2).sum() / (gt ** 2).sum())


def volume_ssim(vol: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean SSIM over central XZ slices (one per Y row) using skimage."""
    vol_np = vol.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    drange = float(gt_np.max() - gt_np.min())
    if drange == 0.0:
        return 0.0
    Ny = vol_np.shape[0]
    idxs = np.linspace(0, Ny - 1, num=min(Ny, 16), dtype=int)
    return float(np.mean([
        sk_ssim(vol_np[i], gt_np[i], data_range=drange) for i in idxs
    ]))


def volume_ssim_3d(vol: torch.Tensor, gt: torch.Tensor) -> float:
    """Full-volume 3D SSIM via skimage (Wang-2004 with a 3D Gaussian window).

    Computes one scalar SSIM over the entire [Ny, Nz, Nx] volume — the
    "proper" 3D metric, treating the volume as a single 3D image rather than
    a stack of 2D slices.
    """
    vol_np = vol.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    drange = float(gt_np.max() - gt_np.min())
    if drange == 0.0:
        return 0.0
    return float(sk_ssim(
        vol_np, gt_np, data_range=drange,
        win_size=7, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False,
    ))


# ── Reconstruction loop ───────────────────────────────────────────────────

def reconstruct(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    *,
    loss_kind: str,
    lam: float,
    tv_weight: float,
    n_iter: int,
    lr: float,
    seed: int,
) -> tuple[torch.Tensor, list[float]]:
    """Custom Adam loop. Returns (final_volume, losses_per_iter)."""
    device = sinogram.device
    n_angles, Ny, N_det = sinogram.shape
    vol_shape = (Ny, N_det, N_det)

    proj = ft.ParallelProjector(vol_shape, angles_deg, device=device, N_det=N_det)

    torch.manual_seed(seed)
    vol = torch.zeros(vol_shape, device=device, requires_grad=True)
    opt = torch.optim.Adam([vol], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter)

    s3im_gen = torch.Generator(device=device).manual_seed(seed + 1)

    losses: list[float] = []
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        pred = proj.forward(vol)
        loss = F.mse_loss(pred, sinogram)
        if loss_kind in ("mse_tv", "mse_tv_ssim", "mse_tv_s3im"):
            loss = loss + tv_weight * ft.tv_loss_3d(vol)
        if loss_kind == "mse_tv_ssim":
            loss = loss + SSIM_WEIGHT * ft.tilt_ssim_loss(pred, sinogram)
        if loss_kind in ("mse_s3im", "mse_tv_s3im"):
            loss = loss + lam * ft.s3im_loss(
                pred, sinogram,
                patch_pixels=S3IM_PATCH_PIXELS,
                kernel_size=S3IM_KERNEL_SIZE,
                n_patches=S3IM_N_PATCHES,
                generator=s3im_gen,
            )
        losses.append(float(loss.item()))
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            vol.clamp_(min=0)

    return vol.detach(), losses


# ── Sweep driver ──────────────────────────────────────────────────────────

def sub_runs_for_kind(kind: str, *, quick: bool) -> list[tuple[float, float]]:
    """Return list of (λ, tv_weight) pairs to evaluate for a given loss kind.

    * ``mse`` / ``mse_tv`` / ``mse_tv_ssim`` — one run with λ=0, TV at default.
    * ``mse_s3im``                          — λ sweep, no TV.
    * ``mse_tv_s3im``                       — full (TV × λ) grid (the user's
      "try a few TV losses" request).
    """
    if kind == "mse_tv_s3im":
        lams = (LAMBDA_S3IM[1],) if quick else LAMBDA_S3IM
        tvs = (TV_WEIGHT_DEFAULT,) if quick else TV_WEIGHTS_S3IM
        return [(lam, tv) for tv in tvs for lam in lams]
    if kind == "mse_s3im":
        lams = (LAMBDA_S3IM[1],) if quick else LAMBDA_S3IM
        # TV irrelevant for this recipe — keep tv at default for the record.
        return [(lam, TV_WEIGHT_DEFAULT) for lam in lams]
    # Pure / non-stochastic recipes: single run at default TV (no λ).
    return [(0.0, TV_WEIGHT_DEFAULT)]


def run_sweep(
    *,
    quick: bool,
    n_iter: int,
    device: torch.device,
) -> list[Result]:
    phantom = load_phantom(device)

    if quick:
        ranges = (70.0,)
        steps = (5.0,)
    else:
        ranges = TILT_RANGES
        steps = TILT_STEPS

    # For NPZ output: keep the best (lowest NMSE) volume per (range, step, kind).
    best_vols: dict[tuple[float, float, str], tuple[float, np.ndarray]] = {}

    results: list[Result] = []
    total_cells = sum(
        len(sub_runs_for_kind(k, quick=quick))
        for _r in ranges for _s in steps for k in LOSS_KINDS
    )

    cell_idx = 0
    for r in ranges:
        for s in steps:
            sino, angles = make_perfect_sino(phantom, r, s)
            print(f"\n=== range ±{r:.0f}°, step {s:.0f}° → {sino.shape[0]} angles ===")
            for kind in LOSS_KINDS:
                for lam, tv_w in sub_runs_for_kind(kind, quick=quick):
                    cell_idx += 1
                    t0 = time.perf_counter()
                    vol, losses = reconstruct(
                        sino, angles,
                        loss_kind=kind, lam=lam, tv_weight=tv_w,
                        n_iter=n_iter, lr=LR, seed=SEED,
                    )
                    dt = time.perf_counter() - t0
                    nm = nmse(vol, phantom)
                    ss = volume_ssim(vol, phantom)
                    ss3 = volume_ssim_3d(vol, phantom)
                    fl = losses[-1]
                    res = Result(
                        range_deg=r, step_deg=s, loss_kind=kind,
                        lam=lam, tv_weight=tv_w,
                        nmse=nm, ssim=ss, ssim_3d=ss3,
                        final_loss=fl, time_s=dt,
                        n_angles=int(sino.shape[0]),
                        losses=losses,
                    )
                    results.append(res)
                    print(
                        f"[{cell_idx:>3}/{total_cells}] "
                        f"{kind:>14s} λ={lam:>5.2f} tv={tv_w:>6.2f}  "
                        f"NMSE={nm:.4f}  SSIM2D={ss:.4f}  SSIM3D={ss3:.4f}  "
                        f"final_loss={fl:.4e}  ({dt:.1f}s)"
                    )

                    key = (r, s, kind)
                    prev = best_vols.get(key)
                    if prev is None or nm < prev[0]:
                        best_vols[key] = (nm, vol.detach().cpu().numpy())

                    del vol
                    torch.cuda.empty_cache()

    # Save NPZ of best volumes + GT phantom
    npz_payload = {"phantom": phantom.detach().cpu().numpy().astype(np.float32)}
    for (r, s, kind), (_nm, arr) in best_vols.items():
        key = f"r{int(r)}_s{int(s)}_{kind}"
        npz_payload[key] = arr.astype(np.float32)
    OUT_RECONS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_RECONS, **npz_payload)
    print(f"\nSaved phantom + {len(npz_payload) - 1} best volumes → {OUT_RECONS}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-iter", type=int, default=N_ITER,
                        help=f"AD iterations per run (default {N_ITER})")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 1 (range, step) × 5 losses, 1 λ for s3im cells")
    parser.add_argument("--out", default=str(OUT_JSON),
                        help=f"Output JSON path (default {OUT_JSON})")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    print(f"Device: {device}")
    print(f"Mode: {'quick smoke test' if args.quick else 'full sweep'}")
    print(f"Iterations per run: {args.n_iter}")

    results = run_sweep(quick=args.quick, n_iter=args.n_iter, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved {len(results)} results → {out_path}")


if __name__ == "__main__":
    main()
