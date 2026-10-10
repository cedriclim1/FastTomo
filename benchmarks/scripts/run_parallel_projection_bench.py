"""
Parallel-beam CUDA forward/back-projection timings (kernel, cached
projector, functional one-shot API).

    python benchmarks/scripts/run_parallel_projection_bench.py [--sizes 128,256,512]
        [--angles 61,181] [--ny-offsets 0] [--iters 20] [--csv out.csv]

For every volume N^3 (Ny = N - offset, so offsets 1, 2, 3 give Ny % 4 != 0),
view count n (angles linspace(-70, 70, n)) and dtype (fp32, fp16) it prints
the median of ``--iters`` CUDA-event timings (after 3 warm-up calls) of:

* ``kernel``      the cached kernel alone (texture already filled, no upload);
* ``cached``      ``ParallelProjector.forward/back`` (upload + kernel + Python,
                  including the fp16 overflow check);
* ``functional``  ``fasttomo.forward_project/back_project``;
* ``host``        host time per functional call in a loop of 20 calls
                  without synchronising in between (how much the host blocks).

Run it on an idle GPU; record the GPU model and the commit hash with the
numbers. Only uses APIs that exist on main since PR #96, so the same script
measures a baseline build.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import time

import torch

import fasttomo as ft
import fasttomo._core as C
from fasttomo._backend import _stream_ptr


def timeit(fn, iters: int, warm: int = 3) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def host_ms(fn, n: int = 20) -> float:
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return 1e3 * (t1 - t0) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="128,256,512")
    ap.add_argument("--angles", default="61,181")
    ap.add_argument("--ny-offsets", default="0")
    ap.add_argument("--dtypes", default="float32,float16")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=ft.__file__.rsplit("/", 2)[0]).stdout.strip()
    except Exception:
        rev = "?"
    head = f"# {torch.cuda.get_device_name()} torch {torch.__version__} fasttomo {ft.__file__} @ {rev}"
    print(head)
    rows = ["N,Ny,nang,dtype,path,op,ms"]
    print(rows[0])
    for N in (int(v) for v in args.sizes.split(",")):
        for off in (int(v) for v in args.ny_offsets.split(",")):
            Ny = N - off
            for nang in (int(v) for v in args.angles.split(",")):
                ang = torch.linspace(-70, 70, nang, device=dev)
                for dname in args.dtypes.split(","):
                    dt = getattr(torch, dname)
                    g = torch.Generator(device=dev).manual_seed(0)
                    scale = 0.01 if dt == torch.float16 else 1.0
                    vol = (torch.rand(Ny, N, N, device=dev, generator=g) * scale).to(dt)
                    P = ft.ParallelProjector((Ny, N, N), ang, device=dev)
                    sino = P.forward(vol)
                    code = 0 if dt == torch.float32 else 1
                    op = P._op
                    vh, sh = op._vol_cache(), op._sino_cache(nang)
                    s = _stream_ptr(dev)
                    out_s, out_v = torch.empty_like(sino), torch.empty_like(vol)
                    ca, sa = op._cos, op._sin
                    C.tc_upload(sh, sino.data_ptr(), s)
                    it = args.iters
                    cases = [
                        ("kernel", "FP", lambda: C.forward_project_parallel_cuda_cached(
                            code, vh, vol.data_ptr(), out_s.data_ptr(), ca.data_ptr(), sa.data_ptr(),
                            Ny, N, N, nang, N, s, upload=False)),
                        ("kernel", "BP", lambda: C.back_project_parallel_cuda_cached(
                            code, sh, sino.data_ptr(), out_v.data_ptr(), ca.data_ptr(), sa.data_ptr(),
                            Ny, N, N, nang, N, s, upload=False)),
                        ("cached", "FP", lambda: P.forward(vol)),
                        ("cached", "BP", lambda: P.back(sino)),
                        ("functional", "FP", lambda: ft.forward_project(vol, ang)),
                        ("functional", "BP", lambda: ft.back_project(sino, ang, Nz=N)),
                    ]
                    for path, name, fn in cases:
                        r = f"{N},{Ny},{nang},{dname},{path},{name},{timeit(fn, it):.4f}"
                        rows.append(r)
                        print(r, flush=True)
                    for name, fn in (("FP", lambda: ft.forward_project(vol, ang)),
                                     ("BP", lambda: ft.back_project(sino, ang, Nz=N))):
                        r = f"{N},{Ny},{nang},{dname},host,{name},{host_ms(fn):.4f}"
                        rows.append(r)
                        print(r, flush=True)
                    del P, op, vol, sino, out_s, out_v
                    torch.cuda.empty_cache()
    if args.csv:
        with open(args.csv, "w") as f:
            f.write(head + "\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
