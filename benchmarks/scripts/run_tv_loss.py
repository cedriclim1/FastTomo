"""Microbenchmark: fused CUDA isotropic 3-D TV loss vs pure-PyTorch reference.

Compares the inline TV expression used in the notebook / previous
``run_grad_descent`` (the "old" path — three slice-subtractions, square-sum,
sqrt, mean, all materialized as full-volume intermediates) against
``fasttomo.tv_loss_3d`` (the fused kernel — one read pass + atomicAdd
reduction in forward, one voxel-driven gather in backward).

Reports per-call wall time for forward-only, backward-only, and
forward+backward, plus peak CUDA memory of one fwd+bwd cycle. Sweeps a
couple of cubic volume sizes so the cost vs problem-size trend is visible.

Usage:
    conda run -n FastTomo python -m benchmarks.run_tv_loss
    conda run -n FastTomo python -m benchmarks.run_tv_loss --shapes 128 200 256
    conda run -n FastTomo python -m benchmarks.run_tv_loss --shapes 200 --samples 200
"""

from __future__ import annotations

import argparse
import time

import torch

import fasttomo as ft


# ── reference (pure-PyTorch isotropic TV) ────────────────────────────────

def tv_pytorch(volume: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """The exact expression replaced by ``ft.tv_loss_3d``.

    Allocates three volume-sized diff tensors, a squared-sum tensor and a
    sqrt'd tensor — every call rebuilds the full autograd graph for the
    backward.
    """
    dy = volume[1:, :, :] - volume[:-1, :, :]
    dz = volume[:, 1:, :] - volume[:, :-1, :]
    dx = volume[:, :, 1:] - volume[:, :, :-1]
    return (dx[:-1, :-1, :] ** 2 + dy[:, :-1, :-1] ** 2
            + dz[:-1, :, :-1] ** 2 + eps).sqrt().mean()


# ── timing helpers ────────────────────────────────────────────────────────

def time_loop(fn, samples: int, warmup: int) -> float:
    """Mean ms/call. Synchronizes before/after the timed block."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(samples):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / samples * 1e3


def time_forward(loss_fn, vol: torch.Tensor, samples: int, warmup: int) -> float:
    """Forward-only — no graph retention; ``vol`` keeps requires_grad=False."""
    vol_d = vol.detach()
    return time_loop(lambda: loss_fn(vol_d), samples, warmup)


def time_fwd_bwd(loss_fn, vol_template: torch.Tensor, samples: int, warmup: int) -> float:
    """One full optimizer-iteration cost: build graph, backward, zero grad."""
    vol = vol_template.detach().clone().requires_grad_(True)

    def step():
        if vol.grad is not None:
            vol.grad = None
        loss_fn(vol).backward()

    return time_loop(step, samples, warmup)


def time_backward(loss_fn, vol_template: torch.Tensor, samples: int, warmup: int) -> float:
    """Backward-only — by subtraction (fwd+bwd) − fwd. Negative values clipped to 0."""
    t_fb = time_fwd_bwd(loss_fn, vol_template, samples, warmup)
    t_fw = time_forward(loss_fn, vol_template, samples, warmup)
    return max(0.0, t_fb - t_fw)


def peak_mem_fwd_bwd(loss_fn, vol_template: torch.Tensor) -> float:
    """Peak CUDA bytes during one fwd+bwd cycle, in MiB."""
    vol = vol_template.detach().clone().requires_grad_(True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    loss_fn(vol).backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# ── correctness check (sanity) ────────────────────────────────────────────

def correctness_check(vol: torch.Tensor, eps: float) -> tuple[float, float]:
    """(forward rel-err, backward max-rel-err) between fused and reference."""
    vol_a = vol.detach().clone().requires_grad_(True)
    vol_b = vol.detach().clone().requires_grad_(True)

    loss_a = ft.tv_loss_3d(vol_a, eps=eps);   loss_a.backward()
    loss_b = tv_pytorch(vol_b, eps=eps);       loss_b.backward()

    f_rel = (loss_a - loss_b).abs().item() / max(loss_b.abs().item(), 1e-12)
    g_diff  = (vol_a.grad - vol_b.grad).abs().max().item()
    g_scale = vol_b.grad.abs().max().clamp_min(1e-12).item()
    return f_rel, g_diff / g_scale


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shapes", type=int, nargs="+",
                        default=[128, 200, 256, 320],
                        help="cubic edge lengths to sweep")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"device={device}  samples={args.samples}  warmup={args.warmup}  "
          f"eps={args.eps:g}")
    print(f"shapes (cubic edge): {args.shapes}\n")

    rows: list[dict] = []
    for N in args.shapes:
        vol = torch.randn(N, N, N, device=device, dtype=torch.float32)

        # correctness sanity (cheap)
        f_rel, g_rel = correctness_check(vol, args.eps)

        # forward-only
        fw_pt    = time_forward(lambda v: tv_pytorch(v, args.eps), vol,
                                 args.samples, args.warmup)
        fw_fused = time_forward(lambda v: ft.tv_loss_3d(v, args.eps), vol,
                                 args.samples, args.warmup)

        # full step (fwd + bwd)
        fb_pt    = time_fwd_bwd(lambda v: tv_pytorch(v, args.eps), vol,
                                 args.samples, args.warmup)
        fb_fused = time_fwd_bwd(lambda v: ft.tv_loss_3d(v, args.eps), vol,
                                 args.samples, args.warmup)

        # backward-only (subtraction)
        bw_pt    = max(0.0, fb_pt    - fw_pt)
        bw_fused = max(0.0, fb_fused - fw_fused)

        # peak memory of one fwd+bwd
        mem_pt    = peak_mem_fwd_bwd(lambda v: tv_pytorch(v, args.eps), vol)
        mem_fused = peak_mem_fwd_bwd(lambda v: ft.tv_loss_3d(v, args.eps), vol)

        rows.append({
            "N": N,
            "fw_pt": fw_pt, "fw_fused": fw_fused,
            "bw_pt": bw_pt, "bw_fused": bw_fused,
            "fb_pt": fb_pt, "fb_fused": fb_fused,
            "mem_pt": mem_pt, "mem_fused": mem_fused,
            "f_rel": f_rel, "g_rel": g_rel,
        })

        print(f"── shape {N}^3 ──  (fwd rel-err {f_rel:.2e}, bwd max-rel-err {g_rel:.2e})")
        print(f"  {'phase':<14} {'PyTorch (ms)':>14} {'Fused (ms)':>12} {'speedup':>10}")
        print(f"  {'forward':<14} {fw_pt:>14.3f} {fw_fused:>12.3f} "
              f"{fw_pt / max(fw_fused, 1e-9):>9.2f}x")
        print(f"  {'backward':<14} {bw_pt:>14.3f} {bw_fused:>12.3f} "
              f"{bw_pt / max(bw_fused, 1e-9):>9.2f}x")
        print(f"  {'fwd+bwd':<14} {fb_pt:>14.3f} {fb_fused:>12.3f} "
              f"{fb_pt / max(fb_fused, 1e-9):>9.2f}x")
        print(f"  peak mem (fwd+bwd):  PyTorch {mem_pt:7.1f} MiB   "
              f"Fused {mem_fused:7.1f} MiB   "
              f"({mem_pt / max(mem_fused, 1e-9):.2f}x reduction)\n")

    # ── compact summary table ──
    print("Summary (fwd+bwd, the optimizer-step cost):")
    print(f"  {'N^3':>6} {'PyTorch ms':>12} {'Fused ms':>10} "
          f"{'speedup':>10} {'mem PT MiB':>12} {'mem fused MiB':>14}")
    for r in rows:
        speedup = r["fb_pt"] / max(r["fb_fused"], 1e-9)
        print(f"  {r['N']:>6} {r['fb_pt']:>12.3f} {r['fb_fused']:>10.3f} "
              f"{speedup:>9.2f}x {r['mem_pt']:>12.1f} {r['mem_fused']:>14.1f}")


if __name__ == "__main__":
    main()
