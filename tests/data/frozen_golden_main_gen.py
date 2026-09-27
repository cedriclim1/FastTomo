"""Generate tests/data/frozen_golden_main.pt (read by tests/test_frozen_identity.py).

Run against origin/main's fasttomo at 0bf45d6 (the API before the section-6
refactor: support=, ConstraintConfig(mask=, sparsity=)). Needs a CUDA GPU:

    python tests/data/frozen_golden_main_gen.py tests/data/frozen_golden_main.pt

Every case here is mirrored, with the new keyword names plus the new
optional keywords at their no-op values, in test_frozen_identity.py.
"""
import sys

import torch

import fasttomo as ft


def inputs(device):
    g = torch.Generator().manual_seed(5)
    Ny, N, n = 3, 20, 13
    vol = torch.zeros(Ny, N, N)
    vol[:, 5:15, 6:13] = 1.0
    vol[1, 8:11, 8:12] = 2.0
    vol += 0.1 * torch.rand(Ny, N, N, generator=g)
    ang = torch.linspace(-65, 65, n)
    eul = torch.stack([torch.linspace(-3, 3, n), ang, torch.linspace(2, -2, n)], 1)
    sm = torch.zeros(1, N, N)
    sm[:, 3:17, 3:17] = 1.0
    vol, ang, sm = vol.to(device), ang.to(device), sm.to(device)
    return ft.forward_project(vol, ang), ang, eul, sm, N


def cases_old():
    out = {}
    sino, ang, _, _, N = inputs("cpu")
    out["genfire_cpu"] = ft.run_genfire(sino, ang, n_iter=12, Nz=N, return_loss=True)
    out["genfire_cpu_plain"] = ft.run_genfire(
        sino, ang, n_iter=8, Nz=N, mask=False, resolution_extension=False,
        sparsity=1e-3, positivity=False, return_loss=True)
    out["genfire_cpu_os"] = ft.run_genfire(sino, ang, n_iter=5, Nz=N,
                                           oversampling_ratio=3.0, cutoff_dist=0.7)
    sino, ang, eul, sm, N = inputs("cuda")
    out["genfire_cuda"] = ft.run_genfire(sino, ang, n_iter=12, Nz=N, return_loss=True)
    out["resire"] = ft.run_resire(sino, ang, n_iter=6, Nz=N, return_loss=True)
    out["resire_support"] = ft.run_resire(sino, ang, n_iter=4, Nz=N, support=sm)
    out["resire_mask"] = ft.run_resire(sino, ang, n_iter=4, Nz=N, mask=True,
                                       sparsity=1e-3, positivity=False, step_size=0.5)
    out["resire_euler"] = ft.run_resire(sino, ang, n_iter=4, Nz=N, euler_zyz_deg=eul,
                                        return_loss=True)
    C, O, R, K, P = (ft.AdReconConfig, ft.OptimConfig, ft.RegularizerConfig,
                     ft.ConstraintConfig, ft.PoseRefinementConfig)
    out["ad"] = ft.run_ad_recon(sino, ang, Nz=N, config=C(optim=O(n_iter=6)),
                                return_loss=True)
    out["ad_reg"] = ft.run_ad_recon(sino, ang, Nz=N, config=C(
        optim=O(n_iter=6, loss_fn="l1"),
        reg=R(tv_weight=1e-3, tv_xz_weight=1e-3, sparsity=1e-3),
        constraints=K(positivity=True, mask=True, sparsity=1e-3)), return_loss=True)
    out["ad_pose"] = ft.run_ad_recon(sino, ang, Nz=N, config=C(
        optim=O(n_iter=6),
        pose=P(refine_pose=True, refine_shifts=True, in_plane_weight=1e-2,
               anchor_reference=True)), return_loss=True, return_pose=True)
    out["ad_staged"] = ft.run_ad_recon(sino, ang, Nz=N, config=C(
        optim=O(n_iter=8), constraints=K(sparsity=1e-3),
        pose=P(refine_shifts=True, pose_warmup_iters=2, alternating_period=2)),
        return_loss=True, return_pose=True)
    out["ad_support"] = ft.run_ad_recon(sino, ang, Nz=N, config=C(optim=O(n_iter=5)),
                                        support_mask=sm)
    return out


def to_cpu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, tuple):
        return tuple(to_cpu(x) for x in v)
    return v


if __name__ == "__main__":
    torch.save({k: to_cpu(v) for k, v in cases_old().items()}, sys.argv[1])
