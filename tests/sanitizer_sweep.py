"""Size sweep over every texture / projection CUDA wrapper, for compute-sanitizer.

Not collected by pytest (no ``test_`` prefix): the sanitizer slows kernels
down by orders of magnitude and CI has no GPU. Run on a free GPU with

    compute-sanitizer --tool memcheck  --error-exitcode 1 python tests/sanitizer_sweep.py
    compute-sanitizer --tool racecheck --error-exitcode 1 python tests/sanitizer_sweep.py

(append ``tv`` to run only the TV-loss kernels, ``parallel`` to run only the
parallel-beam FP/BP, texture-pool and packed-texture variants).

Sizes 1, 15, 17, 33 and 37 cover single-element, partial-block (16x16,
32x8 and 8x8x4 blocks), multi-block and prime extents; Ny = 1 takes the
1-channel parallel textures, every Ny >= 2 the packed ones (2 fp32 slices
per float2 texel, 4 fp16 slices per texel, zero-padded when Ny is not a
multiple).
"""

import torch

import fasttomo as ft
from fasttomo.autograd import _euler_zyz_to_rotmat, _grad_rotmat_raw


def sweep(sizes=(1, 15, 17, 33, 37), extra_ny=(4, 8)) -> None:
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    for n in sizes:
        for Ny in sorted({n, *extra_ny}):
            Nz, Nx, n_ang = n, max(n, 2), max(n // 2, 1)
            angles = torch.linspace(-60, 60, n_ang, device=dev)
            euler = torch.stack([angles * 0.1, angles, -angles * 0.1], dim=1)
            mats = _euler_zyz_to_rotmat(euler).reshape(n_ang, 9).contiguous()
            vol = torch.rand(Ny, Nz, Nx, generator=g).to(dev)
            for dtype in (torch.float32, torch.float16):
                v = vol.to(dtype)
                s = ft.forward_project(v, angles)
                ft.back_project(s, angles, Nz=Nz)
                pp = ft.ParallelProjector((Ny, Nz, Nx), angles)
                pp.back(pp.forward(v))
                s3 = ft.forward_project_3d(v, euler)
                ft.back_project_3d(s3, euler, Nz=Nz)
                rp = ft.RotmatProjector((Ny, Nz, Nx), n_angles=n_ang, device=dev)
                m = mats.clone().requires_grad_(True)
                rp.forward(v, m).float().sum().backward()
                rp.back(s3, mats)
                del pp, rp
            a = angles.clone().requires_grad_(True)
            ft.forward_project(vol, a).sum().backward()
            _grad_rotmat_raw(vol, torch.rand(n_ang, Ny, Nx, device=dev), mats)
            if min(Ny, Nz, Nx) >= 2:
                ft.tv_loss_3d(vol.clone().requires_grad_(True)).backward()
                sino = ft.forward_project(vol, angles)
                ft.run_mbir(sino, angles, n_iter=1, multires_levels=1)
            torch.cuda.synchronize()
            print(f"ok n={n} Ny={Ny}", flush=True)
    mbir_sweep_variants(g)
    clip_and_tv_variants(g)
    sirt_variants(g)
    parallel_variants(g)


def parallel_variants(g) -> None:
    """The parallel-beam FP/BP kernels (1-channel, float2 and 4-channel half
    textures) for Ny = 1..8 (every Ny % 4, padded last texels), N_det above
    and below Nx: the one-shot binding, the pooled functional path, a cached
    projector, CUDA-resident NaN angles (the checked BP loop), the fp16
    non-finite report word, packed tc_download, and run_sirt's packed fused
    kernels (eager and graph) on odd Ny."""
    from fasttomo._operator import _TEXTURE_POOL
    from fasttomo.algorithms import algebraic
    from fasttomo.autograd import _back_project_parallel_raw, _forward_project_parallel_raw
    import fasttomo._core as core
    from fasttomo._backend import _stream_ptr

    dev = torch.device("cuda")
    for (Nz, Nx), N_det in (((1, 2), 3), ((15, 17), 13), ((33, 31), 37)):
        for Ny in (1, 2, 3, 4, 5, 6, 7, 8):
            n_ang = 7
            angles = torch.linspace(-90, 90, n_ang, device=dev)
            bad = angles.clone()
            bad[3] = float("nan")
            vol = torch.rand(Ny, Nz, Nx, generator=g).to(dev)
            for dtype in (torch.float32, torch.float16):
                v = vol.to(dtype)
                s = _forward_project_parallel_raw(v, angles, N_det)
                _back_project_parallel_raw(s, angles, Nz, Nx=Nx)
                s = ft.forward_project(v, angles, N_det=N_det)
                ft.back_project(s, angles, Nz=Nz, Nx=Nx)
                ft.forward_project(v, bad, N_det=N_det)
                ft.back_project(s, bad, Nz=Nz, Nx=Nx)
                pp = ft.ParallelProjector((Ny, Nz, Nx), angles, N_det=N_det)
                pp.back(pp.forward(v))
                if dtype == torch.float16:
                    try:
                        ft.back_project(torch.full_like(s, 3e4), angles, Nz=Nz, Nx=Nx)
                    except OverflowError:
                        pass
                    h = core.tc_create(0, 1, 4, Ny, Nz, Nx) if Ny > 1 else None
                else:
                    h = core.tc_create(0, 0, 2, Ny, Nz, Nx) if Ny > 1 else None
                if h is not None:
                    out = torch.empty_like(v)
                    core.tc_upload(h, v.data_ptr(), _stream_ptr(dev))
                    core.tc_download(h, out.data_ptr(), _stream_ptr(dev))
                    torch.cuda.synchronize()
                    core.tc_destroy(h)
                sino = s.float().to(dtype)
                sm = (torch.rand(Ny, Nz, N_det, generator=g) > 0.3).to(dev, dtype)
                for kw in (dict(), dict(positivity=True, support_mask=sm, return_loss=True)):
                    algebraic._SIRT_OPERATORS.clear()
                    ft.run_sirt(sino, angles, 2, Nz=Nz, **kw)
                    ft.run_sirt(sino, angles, 2, Nz=Nz, **kw)
                del pp
            torch.cuda.synchronize()
            print(f"ok parallel Ny={Ny} Nz={Nz} Nx={Nx} N_det={N_det}", flush=True)
    algebraic._SIRT_OPERATORS.clear()
    _TEXTURE_POOL.clear()


def sirt_variants(g) -> None:
    """run_sirt's fused kernels (csrc/cuda/sirt_update.cu) on the graph and
    eager paths: residual and update into global memory (scalar and 16-byte
    vectorised), into 1-channel and 4-channel packed fp16 textures through
    their surfaces, the texture-resident graph volume (surface reads and
    tc_download), rotmat (3-D surface), broadcast and full support masks."""
    from fasttomo.algorithms import algebraic

    dev = torch.device("cuda")
    for (Ny, Nz, Nx), n_ang in (((1, 1, 2), 1), ((3, 5, 7), 4), ((4, 17, 33), 7),
                                ((8, 16, 24), 9)):
        angles = torch.linspace(-60, 60, n_ang, device=dev)
        euler = torch.stack([angles * 0.1, angles, -angles * 0.1], dim=1)
        vol = torch.rand(Ny, Nz, Nx, generator=g).to(dev)
        sm = (torch.rand(Ny, Nz, Nx, generator=g) > 0.3).float().to(dev)
        for dtype in (torch.float32, torch.float16):
            sino = ft.forward_project(vol, angles, N_det=Nx).to(dtype)
            sino3 = ft.forward_project_3d(vol, euler, N_det=Nx).to(dtype)
            for kw in (dict(), dict(positivity=True, mask=True, sparsity=1e-3, omega=0.7),
                       dict(support_mask=sm.to(dtype), return_loss=True)):
                algebraic._SIRT_OPERATORS.clear()
                ft.run_sirt(sino, angles, 3, Nz=Nz, **kw)
                ft.run_sirt(sino, angles, 3, Nz=Nz, **kw)          # graph replay
            ft.run_sirt(sino3, angles, 2, Nz=Nz, euler_zyz_deg=euler, positivity=True)
            # the fused kernels into global memory (no texture target)
            algebraic._fused_update(vol.to(dtype), vol.to(dtype), vol.to(dtype), 0.5,
                                    algebraic.ConstraintConfig(positivity=True),
                                    eager_order=True)
            algebraic._fused_residual(sino, sino, sino, want_raw=True)
        algebraic._SIRT_OPERATORS.clear()
        torch.cuda.synchronize()
        print(f"ok sirt shape={(Ny, Nz, Nx)} n_angles={n_ang}", flush=True)


def clip_and_tv_variants(g) -> None:
    """The ray-clipped rotmat forward on its edge cases (axis-aligned rays,
    a detector wider than the volume so rays miss it, a narrower one,
    scaled matrices, partial 16x16 tiles) and the two-pass TV forward over
    several blocks with partial warps."""
    dev = torch.device("cuda")
    euler = torch.tensor([[0.0, 0.0, 0.0], [0.0, 90.0, 0.0], [90.0, 90.0, 0.0],
                          [12.3, 27.1, -7.9], [41.7, -33.2, 16.4]], device=dev)
    mats = _euler_zyz_to_rotmat(euler).reshape(-1, 9)
    mats = torch.cat([mats, mats[:1] * 0.37, mats[-1:] * 1.9]).contiguous()
    for (Ny, Nz, Nx), N_det in (((9, 5, 12), 29), ((12, 11, 9), 5), ((37, 30, 41), 45)):
        vol = torch.rand(Ny, Nz, Nx, generator=g).to(dev)
        for dtype in (torch.float32, torch.float16):
            rp = ft.RotmatProjector((Ny, Nz, Nx), n_angles=mats.shape[0], device=dev,
                                    N_det=N_det)
            m = mats.clone().requires_grad_(True)
            rp.forward(vol.to(dtype), m).float().sum().backward()
            del rp
        torch.cuda.synchronize()
        print(f"ok clip shape={(Ny, Nz, Nx)} N_det={N_det}", flush=True)
    tv_variants(g)


def tv_variants(g) -> None:
    """The 3-D and planar (XZ) TV kernels, forward and backward: several
    blocks with partial warps, one corner per row, a single row, Ny = 1,
    and the degenerate shapes that launch nothing."""
    dev = torch.device("cuda")
    for shape in ((40, 70, 67), (2, 2, 2), (5, 300, 7), (3, 17, 2), (1, 33, 65), (7, 2, 33),
                  (1, 1, 9), (4, 9, 1)):
        vol = torch.rand(*shape, generator=g).to(dev)
        ft.tv_loss_3d(vol.clone().requires_grad_(True)).backward()
        ft.tv_loss_xz(vol.clone().requires_grad_(True)).backward()
        torch.cuda.synchronize()
        print(f"ok tv shape={shape}", flush=True)


def mbir_sweep_variants(g) -> None:
    """Every MBIR ICD sweep kernel: the serial kernel with 1, 2, 4, 8 angle
    slots per lane and the generic kernel (> 256 angles), and the parallel
    group kernel (up to 256 angles), with and without lambdas and the Huber
    weight, on odd sizes whose rotated projections leave the detector. The
    larger slices (> 64 voxels) exercise the serial kernel's record-ring
    refills and groups with several members; the explicit-order path runs
    through the serial kernel."""
    from fasttomo.algorithms import mbir as M
    dev = torch.device("cuda")
    for n_ang in (9, 61, 100, 140, 260):
        for Ny, Nz, Nx, N_det in ((1, 5, 7, 7), (3, 9, 11, 11), (2, 13, 29, 31), (1, 40, 37, 37)):
            angles = torch.linspace(-75, 75, n_ang, device=dev)
            vol = torch.rand(Ny, Nz, Nx, generator=g).to(dev)
            a = angles * (torch.pi / 180)
            sino = M._mbir_forward_project(vol, N_det, torch.cos(a), torch.sin(a))
            lam = (torch.rand(sino.shape, generator=g) + 0.5).to(dev)
            kernels = ["serial"]
            if M._mbir_resolve_sweep("auto", dev, torch.cos(a), Nz, Nx, N_det):
                kernels.append("parallel")
            for kernel in kernels:
                for kw in ({}, {"lambdas": lam}, {"huber_T": 2.0},
                           {"lambdas": lam, "huber_T": 2.0}):
                    ft.run_mbir(sino, angles, n_iter=2, Nz=Nz, multires_levels=1,
                                sweep_kernel=kernel, **kw)
            key = M._mbir_perm_key(1, 0)
            perm = torch.from_numpy(M._mbir_permutation(Ny, Nz * Nx, key)).to(dev)
            v = torch.zeros(Ny, Nz, Nx, device=dev)
            M._mbir_icd_sweep_cuda(v, sino.clone(), lam, v.clone(),
                                   torch.empty(Ny * Nz * Nx * 2, device=dev), torch.cos(a),
                                   torch.sin(a), perm_key=key, perm=perm, qggmrf_p=1.2,
                                   qggmrf_q=2.0, qggmrf_c=1e-3, sigma_f=0.5, huber_T=2.0,
                                   huber_delta=0.5, sigma=1.0)
            torch.cuda.synchronize()
            print(f"ok mbir n_angles={n_ang} shape={(Ny, Nz, Nx, N_det)} {kernels}", flush=True)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["tv"]:          # only the TV kernels (quick)
        tv_variants(torch.Generator(device="cpu").manual_seed(0))
    elif sys.argv[1:] == ["parallel"]:  # only the parallel-beam kernels
        parallel_variants(torch.Generator(device="cpu").manual_seed(0))
    else:
        sweep()
