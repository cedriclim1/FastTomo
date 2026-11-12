"""Multi-GPU speedup of the ``devices=`` keyword against one GPU.

Times each algorithm on a sphere phantom (``N^3``, single-axis tilt series)
on 1, 2, ... GPUs and reports seconds per iteration, the speedup over one
GPU and the relative L2 difference from the one-GPU volume.

    python benchmarks/scripts/run_multigpu.py --sizes 256 512 --gpus 1 2 4 \
        --out benchmarks/results/multigpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time

import torch

import fasttomo as ft
from fasttomo.configs import AdReconConfig, OptimConfig, PoseRefinementConfig, RegularizerConfig


def phantom(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ax = torch.arange(n, dtype=torch.float32)
    vol = torch.zeros(n, n, n)
    for _ in range(12):
        c = (0.25 + 0.5 * torch.rand(3, generator=g)) * n
        r = (0.04 + 0.08 * float(torch.rand(1, generator=g))) * n
        d = ((ax[:, None, None] - c[0]) ** 2 + (ax[None, :, None] - c[1]) ** 2
             + (ax[None, None, :] - c[2]) ** 2)
        vol += (d < r * r).float() * float(torch.rand(1, generator=g) + 0.5)
    return vol


def _ad(pose: bool, n_iter: int) -> AdReconConfig:
    cfg = AdReconConfig(optim=OptimConfig(n_iter=n_iter, lr_volume=1e-2),
                        reg=RegularizerConfig(tv_weight=1e-3))
    if pose:
        cfg.pose = PoseRefinementConfig(refine_pose=True, refine_shifts=True,
                                        anchor_reference=True)
    return cfg


ALGOS = {
    "fbp": (1, lambda s, a, n, d: ft.run_fbp(s, a, devices=d)),
    "sirt": (50, lambda s, a, n, d: ft.run_sirt(s, a, n, positivity=True, devices=d)),
    "sart": (3, lambda s, a, n, d: ft.run_sart(s, a, n, positivity=True, devices=d)),
    "fista": (50, lambda s, a, n, d: ft.run_fista(s, a, n, positivity=True, devices=d)),
    "fista_tv": (30, lambda s, a, n, d: ft.run_fista_tv(s, a, n, tv_weight=1e-2,
                                                       positivity=True, devices=d)),
    "resire": (50, lambda s, a, n, d: ft.run_resire(s, a, n, devices=d)),
    "grad_descent": (50, lambda s, a, n, d: ft.run_grad_descent(
        s, a, n, tv_weight=1e-3, positivity=True, devices=d)),
    "ad_recon": (50, lambda s, a, n, d: ft.run_ad_recon(s, a, n, config=_ad(False, n),
                                                       devices=d)),
    "ad_recon_pose": (50, lambda s, a, n, d: ft.run_ad_recon(s, a, n, config=_ad(True, n),
                                                            devices=d)),
}


def _sync(n_gpus: int) -> None:
    for i in range(n_gpus):
        torch.cuda.synchronize(i)


def time_run(fn, sino, ang, n_iter, devices, n_gpus, repeats):
    fn(sino, ang, min(n_iter, 2), devices)
    best, out = float("inf"), None
    for _ in range(repeats):
        _sync(n_gpus)
        t0 = time.perf_counter()
        out = fn(sino, ang, n_iter, devices)
        _sync(n_gpus)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    p.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--views", type=int, default=71)
    p.add_argument("--algos", nargs="+", default=list(ALGOS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    n_avail = torch.cuda.device_count()
    gpus = [g for g in args.gpus if g <= n_avail]
    meta = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                n_gpus=n_avail, host=platform.node(), views=args.views,
                commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip())
    rows = []
    for n in args.sizes:
        ang = torch.linspace(-70, 70, args.views)
        sino = ft.forward_project(phantom(n).cuda(0), ang.cuda(0)).contiguous()
        for name in args.algos:
            n_iter, fn = ALGOS[name]
            ref, t1 = None, None
            for g in gpus:
                devices = None if g == 1 else list(range(g))
                t, out = time_run(fn, sino, ang, n_iter, devices, max(gpus), args.repeats)
                vol = out[0] if isinstance(out, tuple) else out
                if ref is None:
                    ref, t1 = vol, t
                diff = float((vol.double() - ref.double()).norm() / ref.double().norm())
                row = dict(size=n, views=args.views, algo=name, gpus=g, n_iter=n_iter,
                           seconds=t, s_per_iter=t / n_iter, speedup=t1 / t,
                           rel_l2_vs_1gpu=diff)
                rows.append(row)
                print(f"{name:14s} N={n:4d} gpus={g} {t:8.3f}s {t / n_iter * 1e3:9.2f} ms/it "
                      f"x{t1 / t:5.2f} rel_l2={diff:.2e}", flush=True)
                if args.out:
                    with open(args.out, "w") as f:
                        json.dump(dict(meta=meta, rows=rows), f, indent=1)
                del out, vol
            del ref
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
