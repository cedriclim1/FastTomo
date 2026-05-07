"""Automatic-differentiation reconstruction loops, extracted from
``Notebooks/opt_test.ipynb``.

Two flavours:

* :func:`ad_reconstruct` — volume-only Adam, uses ``forward_project`` (the
  static single-axis operator). Cannot recover pose error; this is the
  "without pose optimization" baseline.
* :func:`ad_reconstruct_with_pose` — additionally treats the per-projection
  ZYZ Euler triple and 2D detector shifts as learnable parameters, each with
  its own Adam optimiser and cosine schedule. Uses ``forward_project_3d``.

Both apply MSE data fidelity, isotropic Charbonnier-style TV regularisation,
and a positivity projection after each step. Defaults match the notebook.

Each function accepts an optional ``tv_fn`` callable so callers can swap
between the inline pure-PyTorch expression (``_tv_charbonnier``, default for
historical comparability) and the fused CUDA kernel (``ft.tv_loss_3d``). This
is what ``benchmarks/run_tv_loss_iterative.py`` uses to measure the end-to-end
speedup of the fused TV reduction on full reconstruction loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

import fasttomo as ft
from fasttomo.autograd import _euler_zyz_to_rotmat


TVFn = Callable[[torch.Tensor], torch.Tensor]


def _tv_charbonnier(vol: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Isotropic Charbonnier TV from opt_test.ipynb (pure-PyTorch reference)."""
    dy = vol[1:, :, :] - vol[:-1, :, :]
    dz = vol[:, 1:, :] - vol[:, :-1, :]
    dx = vol[:, :, 1:] - vol[:, :, :-1]
    return (
        dx[:-1, :-1, :] ** 2
        + dy[:, :-1, :-1] ** 2
        + dz[:-1, :, :-1] ** 2
        + eps
    ).sqrt().mean()


def _tv_fused(vol: torch.Tensor) -> torch.Tensor:
    """Fused CUDA isotropic Charbonnier TV via ``ft.tv_loss_3d``."""
    return ft.tv_loss_3d(vol, eps=1e-8)


@dataclass
class ADResult:
    volume: torch.Tensor
    losses: list[float]
    euler_zyz_deg: torch.Tensor | None = None  # [n_angles, 3], with-pose only
    shifts: torch.Tensor | None = None         # [n_angles, 2], with-pose only


def ad_reconstruct(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    *,
    vol_shape: tuple[int, int, int] | None = None,
    n_iter: int = 1000,
    lr: float = 0.05,
    tv_weight: float = 1.0,
    tv_fn: TVFn = _tv_charbonnier,
) -> ADResult:
    """Volume-only AD reconstruction with the static ``forward_project``.

    No pose optimisation: the geometry is whatever ``angles_deg`` says, and any
    in-plane / tilt-axis error in the sinogram cannot be recovered.
    """
    device = sinogram.device
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    Nz = N_det
    if vol_shape is None:
        vol_shape = (Ny, Nz, Nx)

    vol = torch.zeros(vol_shape, device=device, requires_grad=True)
    opt = torch.optim.Adam([vol], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter)

    losses: list[float] = []
    for _ in range(n_iter):
        opt.zero_grad()
        pred = ft.forward_project(vol, angles_deg, N_det=N_det)
        loss = F.mse_loss(pred, sinogram) + tv_weight * tv_fn(vol)
        losses.append(float(loss.item()))
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            vol.clamp_(min=0)

    return ADResult(volume=vol.detach(), losses=losses)


def ad_reconstruct_with_pose(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    *,
    vol_shape: tuple[int, int, int] | None = None,
    n_iter: int = 1000,
    lr_vol: float = 0.05,
    lr_ang: float = 1.0,
    lr_shf: float = 0.1,
    tv_weight: float = 1.0,
    tv_fn: TVFn = _tv_charbonnier,
) -> ADResult:
    """AD reconstruction that jointly optimises volume, ZYZ Euler triple, and
    per-projection 2D shifts. Uses ``forward_project_3d``.

    Initial Euler is the single-axis embedding (φ=0, θ=angles_deg, ψ=0); shifts
    start at zero. Each parameter has its own Adam + cosine LR schedule, with
    learning rates from the notebook.
    """
    device = sinogram.device
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    Nz = N_det
    if vol_shape is None:
        vol_shape = (Ny, Nz, Nx)

    vol = torch.zeros(vol_shape, device=device, requires_grad=True)
    euler = torch.zeros(n_angles, 3, device=device)
    euler[:, 1] = angles_deg
    euler.requires_grad_(True)
    shifts = torch.zeros(n_angles, 2, device=device, requires_grad=True)

    opt_vol = torch.optim.Adam([vol], lr=lr_vol)
    opt_ang = torch.optim.Adam([euler], lr=lr_ang)
    opt_shf = torch.optim.Adam([shifts], lr=lr_shf)
    sched_vol = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vol, n_iter)
    sched_ang = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ang, n_iter)
    sched_shf = torch.optim.lr_scheduler.CosineAnnealingLR(opt_shf, n_iter)

    losses: list[float] = []
    for _ in range(n_iter):
        opt_vol.zero_grad()
        opt_ang.zero_grad()
        opt_shf.zero_grad()
        pred = ft.forward_project_3d(vol, euler, N_det=N_det, shifts=shifts)
        loss = F.mse_loss(pred, sinogram) + tv_weight * tv_fn(vol)
        losses.append(float(loss.item()))
        loss.backward()
        opt_vol.step()
        opt_ang.step()
        opt_shf.step()
        sched_vol.step()
        sched_ang.step()
        sched_shf.step()
        with torch.no_grad():
            vol.clamp_(min=0)

    return ADResult(
        volume=vol.detach(),
        losses=losses,
        euler_zyz_deg=euler.detach(),
        shifts=shifts.detach(),
    )


def ad_reconstruct_cached(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    *,
    vol_shape: tuple[int, int, int] | None = None,
    n_iter: int = 1000,
    lr: float = 0.05,
    tv_weight: float = 1.0,
    tv_fn: TVFn = _tv_charbonnier,
) -> ADResult:
    """Same as :func:`ad_reconstruct` but routes FP/BP through ``ParallelProjector``.

    The projector owns a persistent ``cudaArray`` + ``cudaTextureObject``, so
    each iteration pays only ``cudaMemcpy3DAsync`` and the kernel launch
    instead of a fresh array allocation. Identical numerics; faster.
    """
    device = sinogram.device
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    Nz = N_det
    if vol_shape is None:
        vol_shape = (Ny, Nz, Nx)

    proj = ft.ParallelProjector(vol_shape, angles_deg, device=device, N_det=N_det)

    vol = torch.zeros(vol_shape, device=device, requires_grad=True)
    opt = torch.optim.Adam([vol], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter)

    losses: list[float] = []
    for _ in range(n_iter):
        opt.zero_grad()
        pred = proj.forward(vol)
        loss = F.mse_loss(pred, sinogram) + tv_weight * tv_fn(vol)
        losses.append(float(loss.item()))
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            vol.clamp_(min=0)

    return ADResult(volume=vol.detach(), losses=losses)


def ad_reconstruct_with_pose_cached(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    *,
    vol_shape: tuple[int, int, int] | None = None,
    n_iter: int = 1000,
    lr_vol: float = 0.05,
    lr_ang: float = 1.0,
    lr_shf: float = 0.1,
    tv_weight: float = 1.0,
    tv_fn: TVFn = _tv_charbonnier,
) -> ADResult:
    """Same as :func:`ad_reconstruct_with_pose` but routes FP through
    ``RotmatProjector``. Euler→rotmat conversion stays in pure PyTorch so
    autograd chains ∂L/∂M into ∂L/∂euler. Shifts are applied post-projection
    via :func:`fasttomo.apply_shifts_2d` (same as the uncached path).
    """
    device = sinogram.device
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    Nz = N_det
    if vol_shape is None:
        vol_shape = (Ny, Nz, Nx)

    proj = ft.RotmatProjector(vol_shape, n_angles, device=device, N_det=N_det)

    vol = torch.zeros(vol_shape, device=device, requires_grad=True)
    euler = torch.zeros(n_angles, 3, device=device)
    euler[:, 1] = angles_deg
    euler.requires_grad_(True)
    shifts = torch.zeros(n_angles, 2, device=device, requires_grad=True)

    opt_vol = torch.optim.Adam([vol], lr=lr_vol)
    opt_ang = torch.optim.Adam([euler], lr=lr_ang)
    opt_shf = torch.optim.Adam([shifts], lr=lr_shf)
    sched_vol = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vol, n_iter)
    sched_ang = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ang, n_iter)
    sched_shf = torch.optim.lr_scheduler.CosineAnnealingLR(opt_shf, n_iter)

    losses: list[float] = []
    for _ in range(n_iter):
        opt_vol.zero_grad()
        opt_ang.zero_grad()
        opt_shf.zero_grad()
        mats = _euler_zyz_to_rotmat(euler)                 # [n_angles, 3, 3]
        pred = proj.forward(vol, mats)                     # [n_angles, Ny, N_det]
        pred = ft.apply_shifts_2d(pred, shifts)
        loss = F.mse_loss(pred, sinogram) + tv_weight * tv_fn(vol)
        losses.append(float(loss.item()))
        loss.backward()
        opt_vol.step()
        opt_ang.step()
        opt_shf.step()
        sched_vol.step()
        sched_ang.step()
        sched_shf.step()
        with torch.no_grad():
            vol.clamp_(min=0)

    return ADResult(
        volume=vol.detach(),
        losses=losses,
        euler_zyz_deg=euler.detach(),
        shifts=shifts.detach(),
    )
