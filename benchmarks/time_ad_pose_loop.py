"""
Per-section CUDA-event timing of the AD-with-pose reconstruction loop on the
atomic-resolution HEA dataset (210^3 volume, 210-detector sinogram).

Goal: decide whether `vol.clamp_(min=0); vol *= mask` is worth fusing into a
single CUDA kernel. Prints a per-section ms/iter and % breakdown, with the
clamp+mask combined number called out at the bottom.

--grad-mode {both,vol,mats}
  both  — vol+euler+shifts all require grad (original behaviour)
  vol   — only vol requires grad; BP kernel runs, grad_rotmat is gated off
  mats  — only euler requires grad; grad_rotmat kernel runs, BP is gated off

Usage:
    python benchmarks/time_ad_pose_loop.py [--device cuda:1] [--measure 200]
    python benchmarks/time_ad_pose_loop.py --device cuda:3 --grad-mode vol
    python benchmarks/time_ad_pose_loop.py --device cuda:3 --grad-mode mats
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import fasttomo as ft
from fasttomo.autograd import _euler_zyz_to_rotmat


HEA_DIR = "/home/cedlim/FastTomo/fasttomo-wiki/raw/code/hea"


def _load_atomic_dataset(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tilt_series = np.load(f"{HEA_DIR}/tilt_series.npy")
    tilt_angles = np.load(f"{HEA_DIR}/phis.npy")
    tilt_series = np.array([np.rot90(img) for img in tilt_series])
    tilt_series /= tilt_series.max()
    return (
        torch.from_numpy(tilt_series).float().to(device),
        torch.from_numpy(tilt_angles).float().to(device),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--measure", type=int, default=200)
    p.add_argument("--vol", type=int, default=210, help="volume edge length")
    p.add_argument(
        "--grad-mode",
        choices=["both", "vol", "mats"],
        default="both",
        help=(
            "both=vol+euler+shifts all require_grad (default); "
            "vol=only vol requires grad (BP kernel only); "
            "mats=only euler requires grad (grad_rotmat kernel only)"
        ),
    )
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("this script requires a CUDA device")
    # torch.cuda.Event(enable_timing=True) records on the *current* CUDA device,
    # not the device of the tensors being timed. Without setting the current
    # device explicitly, events get created on cuda:0 even when our work runs
    # on cuda:1+, which corrupts kernel state and surfaces as illegal memory
    # access in subsequent launches.
    torch.cuda.set_device(device)

    grad_mode = args.grad_mode

    tilt_series, tilt_angles = _load_atomic_dataset(device)
    n_angles = len(tilt_angles)
    N = args.vol

    # Set requires_grad per mode
    vol_requires_grad = grad_mode in ("both", "vol")
    euler_requires_grad = grad_mode in ("both", "mats")
    shifts_requires_grad = grad_mode == "both"

    vol = torch.zeros(N, N, N, device=device, requires_grad=vol_requires_grad)
    euler = torch.zeros(n_angles, 3, device=device)
    euler[:, 1] = tilt_angles
    euler.requires_grad = euler_requires_grad
    shifts = torch.zeros(n_angles, 2, device=device, requires_grad=shifts_requires_grad)
    mask = ft.cylindrical_mask(N, N, device=device)

    proj_rot = ft.RotmatProjector((N, N, N), n_angles=n_angles,
                                  device=device, N_det=N)

    n_total = args.warmup + args.measure
    # Optimisers are always created for all params so the loop structure is
    # unchanged.  In vol/mats mode the optimisers for the non-grad params
    # will no-op at step() time (no .grad set).
    opt_v = torch.optim.Adam([vol],    lr=0.05)
    opt_e = torch.optim.Adam([euler],  lr=1e-2)
    opt_s = torch.optim.Adam([shifts], lr=1e-2)
    sched_v = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, n_total)
    sched_e = torch.optim.lr_scheduler.CosineAnnealingLR(opt_e, n_total)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, n_total)

    tv_weight = 5e-2
    sparsity_weight = 3.0

    sections = ["zero_grad", "euler_to_rotmat", "fp", "shifts", "mse",
                "sparsity", "tv", "backward", "opt_step", "sched",
                "clamp", "mask"]
    starts: dict[str, list[torch.cuda.Event]] = {s: [] for s in sections}
    ends:   dict[str, list[torch.cuda.Event]] = {s: [] for s in sections}

    def step(record: bool) -> None:
        def tic(name: str) -> None:
            if not record:
                return
            e = torch.cuda.Event(enable_timing=True); e.record(); starts[name].append(e)

        def toc(name: str) -> None:
            if not record:
                return
            e = torch.cuda.Event(enable_timing=True); e.record(); ends[name].append(e)

        tic("zero_grad")
        opt_v.zero_grad(); opt_e.zero_grad(); opt_s.zero_grad()
        toc("zero_grad")

        tic("euler_to_rotmat")
        mats = _euler_zyz_to_rotmat(euler)
        toc("euler_to_rotmat")

        tic("fp")
        pred = proj_rot.forward(vol, mats)
        toc("fp")

        tic("shifts")
        pred = ft.apply_shifts_2d(pred, shifts)
        toc("shifts")

        tic("mse")
        loss = torch.nn.functional.mse_loss(pred, tilt_series)
        toc("mse")

        tic("sparsity")
        # In mats mode vol has no grad path — adding vol.mean() would be a
        # constant w.r.t. the only differentiable param (euler) and wastes
        # time building a graph leaf.  Only add when vol is differentiable.
        if vol_requires_grad:
            loss = loss + sparsity_weight * vol.mean()
        toc("sparsity")

        tic("tv")
        if vol_requires_grad:
            loss = loss + tv_weight * ft.tv_loss_3d(vol)
        toc("tv")

        tic("backward")
        loss.backward()
        toc("backward")

        tic("opt_step")
        opt_v.step(); opt_e.step(); opt_s.step()
        toc("opt_step")

        tic("sched")
        sched_v.step(); sched_e.step(); sched_s.step()
        toc("sched")

        with torch.no_grad():
            tic("clamp")
            vol.clamp_(min=0)
            toc("clamp")

            tic("mask")
            vol.mul_(mask)
            toc("mask")

    # Print configuration header before warmup so logs are unambiguous.
    print()
    print(f"grad-mode={grad_mode}  device={args.device}  vol={N}^3  "
          f"n_angles={n_angles}  warmup={args.warmup}  measure={args.measure}")
    print(f"vol.requires_grad={vol_requires_grad}  "
          f"euler.requires_grad={euler_requires_grad}  "
          f"shifts.requires_grad={shifts_requires_grad}")
    print()

    for _ in range(args.warmup):
        step(record=False)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(args.measure):
        step(record=True)
    torch.cuda.synchronize(device)
    wall_total_ms = (time.perf_counter() - t0) * 1e3

    totals_ms = {
        s: sum(a.elapsed_time(b) for a, b in zip(starts[s], ends[s]))
        for s in sections
    }
    iter_ms = wall_total_ms / args.measure
    section_sum_ms = sum(totals_ms.values()) / args.measure

    print(f"{'section':<16} {'ms/iter':>9} {'% iter':>8}")
    print("-" * 36)
    for s in sections:
        per = totals_ms[s] / args.measure
        print(f"{s:<16} {per:>9.4f} {100*per/iter_ms:>7.2f}%")
    print("-" * 36)
    print(f"{'sum(events)':<16} {section_sum_ms:>9.4f} {100*section_sum_ms/iter_ms:>7.2f}%")
    print(f"{'wall/iter':<16} {iter_ms:>9.4f} {'100.00%':>8}")
    print(f"throughput:      {1e3/iter_ms:>9.1f} it/s")
    print()

    cm_ms = (totals_ms["clamp"] + totals_ms["mask"]) / args.measure
    cm_pct = 100 * cm_ms / iter_ms
    print(f"clamp + mask combined: {cm_ms:.4f} ms/iter  ({cm_pct:.2f}% of iter)")
    verdict = "WORTH fusing" if cm_pct >= 5.0 else "NOT worth fusing"
    print(f"=> {verdict} (threshold = 5%)")


if __name__ == "__main__":
    main()
