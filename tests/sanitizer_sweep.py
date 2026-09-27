"""Size sweep over every texture / projection CUDA wrapper, for compute-sanitizer.

Not collected by pytest (no ``test_`` prefix): the sanitizer slows kernels
down by orders of magnitude and CI has no GPU. Run on a free GPU with

    compute-sanitizer --tool memcheck  --error-exitcode 1 python tests/sanitizer_sweep.py
    compute-sanitizer --tool racecheck --error-exitcode 1 python tests/sanitizer_sweep.py

Sizes 1, 15, 17, 33 and 37 cover single-element, partial-block (16x16 and
8x8x4 blocks), multi-block and prime extents; Ny = 4 and 8 take the 4-channel
packed fp16 path, the others the 1-channel path.
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


def mbir_sweep_variants(g) -> None:
    """Every MBIR ICD sweep kernel: 1, 2, 4, 8 angle slots per lane and the
    generic kernel (> 256 angles), with and without lambdas and the Huber
    weight, on odd sizes whose rotated projections leave the detector."""
    dev = torch.device("cuda")
    for n_ang in (9, 61, 100, 140, 260):
        for Ny, Nz, Nx in ((1, 5, 7), (3, 9, 11)):
            angles = torch.linspace(-75, 75, n_ang, device=dev)
            vol = torch.rand(Ny, Nz, Nx, generator=g).to(dev)
            sino = ft.forward_project(vol, angles)
            lam = (torch.rand(sino.shape, generator=g) + 0.5).to(dev)
            for kw in ({}, {"lambdas": lam}, {"huber_T": 2.0}, {"lambdas": lam, "huber_T": 2.0}):
                ft.run_mbir(sino, angles, n_iter=2, multires_levels=1, **kw)
            torch.cuda.synchronize()
            print(f"ok mbir n_angles={n_ang} shape={(Ny, Nz, Nx)}", flush=True)


if __name__ == "__main__":
    sweep()
