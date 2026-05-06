"""
torch.autograd.Function wrappers for FastTomo's projection kernels.

Each FP operator's backward calls its BP partner; each BP operator's backward
calls its FP partner. This makes `forward_project`, `back_project`, and the
3D / RESIRE variants first-class autograd ops — `loss.backward()` flows
gradients into a `requires_grad=True` volume (or sinogram) tensor.

Caveats:
- The CUDA texture-based BPs are only approximate adjoints (~15-65 % relative
  error in the inner-product test). Sufficient as descent directions (SIRT and
  RESIRE rely on this), but `torch.autograd.gradcheck` will fail by design.
- Gradients are NOT propagated through angle inputs (`angles_deg` /
  `euler_zyz_deg`). Adding angle differentiability would require new
  C/CUDA Jacobian kernels; out of scope here.

See: fasttomo-wiki/wiki/concepts/autograd-projectors.md
"""

from __future__ import annotations

import math

import numpy as np
import torch

import fasttomo._core as _core


# ── ZYZ Euler → rotation matrix ──────────────────────────────────────────────

def _euler_zyz_to_rotmat(euler_zyz_deg: torch.Tensor) -> torch.Tensor:
    """
    Convert ZYZ Euler angles to 3×3 rotation matrices (row-major).

    euler_zyz_deg: [n_angles, 3]  — columns are (φ, θ, ψ) in degrees.
    Convention: R = R_Z(ψ) · R_Y(θ) · R_Z(φ)  (RESIRE convention).
    """
    r   = math.pi / 180.0
    phi   = euler_zyz_deg[:, 0].float() * r
    theta = euler_zyz_deg[:, 1].float() * r
    psi   = euler_zyz_deg[:, 2].float() * r

    cp, sp = torch.cos(phi),   torch.sin(phi)
    ct, st = torch.cos(theta), torch.sin(theta)
    cu, su = torch.cos(psi),   torch.sin(psi)

    r00 =  cu*ct*cp - su*sp;  r01 = -cu*ct*sp - su*cp;  r02 = cu*st
    r10 =  su*ct*cp + cu*sp;  r11 = -su*ct*sp + cu*cp;  r12 = su*st
    r20 = -st*cp;             r21 =  st*sp;              r22 = ct

    row0 = torch.stack([r00, r01, r02], dim=1)
    row1 = torch.stack([r10, r11, r12], dim=1)
    row2 = torch.stack([r20, r21, r22], dim=1)
    return torch.stack([row0, row1, row2], dim=1)


# ── plain (non-autograd) projection helpers ─────────────────────────────────
#
# These are the bodies of the original public wrappers, lifted verbatim. The
# autograd Function classes below call them so that `Function.backward`
# invocations of the adjoint partner cannot recurse into autograd.

def _forward_project_raw(
    volume: torch.Tensor,
    angles_deg: torch.Tensor,
    N_det: int = -1,
) -> torch.Tensor:
    angles_np = (angles_deg.detach().cpu().float() * (math.pi / 180.0)).numpy()

    if volume.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        Ny, Nz, Nx = volume.shape
        n_angles    = len(angles_np)
        if N_det < 0:
            N_det = Nx
        vol  = volume.contiguous().float()
        sino = torch.empty(n_angles, Ny, N_det,
                           dtype=torch.float32, device=volume.device)
        _core.forward_project_cuda(
            vol.data_ptr(), sino.data_ptr(),
            angles_np, Ny, Nz, Nx, n_angles, N_det,
        )
        return sino

    vol_np  = volume.detach().float().numpy()
    sino_np = _core.forward_project_cpu(vol_np, angles_np, N_det)
    return torch.from_numpy(sino_np)


def _back_project_raw(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    Nz: int = -1,
) -> torch.Tensor:
    angles_np = (angles_deg.detach().cpu().float() * (math.pi / 180.0)).numpy()

    if sinogram.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        n_angles, Ny, N_det = sinogram.shape
        Nx = N_det
        if Nz < 0:
            Nz = N_det
        sino = sinogram.contiguous().float()
        vol  = torch.empty(Ny, Nz, Nx,
                           dtype=torch.float32, device=sinogram.device)
        _core.back_project_cuda(
            sino.data_ptr(), vol.data_ptr(),
            angles_np, Ny, Nz, Nx, n_angles, N_det,
        )
        return vol

    sino_np = sinogram.detach().float().numpy()
    vol_np  = _core.back_project_cpu(sino_np, angles_np, Nz)
    return torch.from_numpy(vol_np)


def _forward_project_3d_raw(
    volume: torch.Tensor,
    euler_zyz_deg: torch.Tensor,
    N_det: int = -1,
) -> torch.Tensor:
    if not volume.is_cuda:
        raise RuntimeError(
            "forward_project_3d requires CUDA — move volume to GPU first"
        )
    if not _core.cuda_available:
        raise RuntimeError("fasttomo was compiled without CUDA support")

    Ny, Nz, Nx = volume.shape
    n_angles = euler_zyz_deg.shape[0]
    if N_det < 0:
        N_det = Nx

    vol  = volume.contiguous().float()
    sino = torch.empty(n_angles, Ny, N_det, dtype=torch.float32, device=vol.device)

    mats      = _euler_zyz_to_rotmat(euler_zyz_deg).to(device=vol.device)
    mats_flat = mats.reshape(n_angles, 9).contiguous()

    _core.forward_project_rotmat_cuda(
        vol.data_ptr(), sino.data_ptr(), mats_flat.data_ptr(),
        Ny, Nz, Nx, n_angles, N_det,
    )
    return sino


def _back_project_3d_raw(
    sinogram: torch.Tensor,
    euler_zyz_deg: torch.Tensor,
    Nz: int = -1,
) -> torch.Tensor:
    if not sinogram.is_cuda:
        raise RuntimeError(
            "back_project_3d requires CUDA — move sinogram to GPU first"
        )
    if not _core.cuda_available:
        raise RuntimeError("fasttomo was compiled without CUDA support")

    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    sino = sinogram.contiguous().float()
    vol  = torch.empty(Ny, Nz, Nx, dtype=torch.float32, device=sino.device)

    mats      = _euler_zyz_to_rotmat(euler_zyz_deg).to(device=sino.device)
    mats_flat = mats.reshape(n_angles, 9).contiguous()

    _core.back_project_rotmat_cuda(
        sino.data_ptr(), vol.data_ptr(), mats_flat.data_ptr(),
        Ny, Nz, Nx, n_angles, N_det,
    )
    return vol


def _forward_project_resire_raw(
    volume: torch.Tensor,
    euler_zyz_deg: torch.Tensor,
    N_det: int = -1,
) -> torch.Tensor:
    if not volume.is_cuda:
        raise RuntimeError(
            "forward_project_resire requires CUDA — move volume to GPU first"
        )
    if not _core.cuda_available:
        raise RuntimeError("fasttomo was compiled without CUDA support")

    Ny, Nz, Nx = volume.shape
    n_angles = euler_zyz_deg.shape[0]
    if N_det < 0:
        N_det = Nx

    vol  = volume.contiguous().float()
    sino = torch.zeros(n_angles, Ny, N_det, dtype=torch.float32, device=vol.device)

    mats      = _euler_zyz_to_rotmat(euler_zyz_deg).to(device=vol.device)
    mats_flat = mats.reshape(n_angles, 9).contiguous()

    _core.forward_project_resire_cuda(
        vol.data_ptr(), sino.data_ptr(), mats_flat.data_ptr(),
        Ny, Nz, Nx, n_angles, N_det,
    )
    return sino * 0.25


def _back_project_resire_raw(
    sinogram: torch.Tensor,
    euler_zyz_deg: torch.Tensor,
    Nz: int = -1,
) -> torch.Tensor:
    if not sinogram.is_cuda:
        raise RuntimeError(
            "back_project_resire requires CUDA — move sinogram to GPU first"
        )
    if not _core.cuda_available:
        raise RuntimeError("fasttomo was compiled without CUDA support")

    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    sino = sinogram.contiguous().float()
    vol  = torch.zeros(Ny, Nz, Nx, dtype=torch.float32, device=sino.device)

    mats      = _euler_zyz_to_rotmat(euler_zyz_deg).to(device=sino.device)
    mats_flat = mats.reshape(n_angles, 9).contiguous()

    _core.back_project_resire_cuda(
        sino.data_ptr(), vol.data_ptr(), mats_flat.data_ptr(),
        Ny, Nz, Nx, n_angles, N_det,
    )
    return vol * 0.25


# ── autograd.Function wrappers ───────────────────────────────────────────────
#
# Pattern for every pair (FP, BP):  FP.backward = BP_raw,  BP.backward = FP_raw.
# Non-tensor params (shape ints) live on `ctx` as Python attributes; tensors
# (the angle table) go through `save_for_backward`. We never return a gradient
# for the angles tensor — angle differentiability is deferred (would need
# Jacobian kernels w.r.t. rotation parameters).

class _FP2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, angles_deg, N_det):
        ctx.save_for_backward(angles_deg)
        ctx.Nz = volume.shape[-2]
        return _forward_project_raw(volume, angles_deg, N_det)

    @staticmethod
    def backward(ctx, grad_sino):
        (angles_deg,) = ctx.saved_tensors
        grad_vol = _back_project_raw(grad_sino.contiguous(), angles_deg, Nz=ctx.Nz)
        return grad_vol, None, None


class _BP2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, angles_deg, Nz):
        ctx.save_for_backward(angles_deg)
        ctx.N_det = sinogram.shape[-1]
        return _back_project_raw(sinogram, angles_deg, Nz)

    @staticmethod
    def backward(ctx, grad_vol):
        (angles_deg,) = ctx.saved_tensors
        grad_sino = _forward_project_raw(grad_vol.contiguous(), angles_deg, N_det=ctx.N_det)
        return grad_sino, None, None


class _FP3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, euler_zyz_deg, N_det):
        ctx.save_for_backward(euler_zyz_deg)
        ctx.Nz = volume.shape[-2]
        return _forward_project_3d_raw(volume, euler_zyz_deg, N_det)

    @staticmethod
    def backward(ctx, grad_sino):
        (euler_zyz_deg,) = ctx.saved_tensors
        grad_vol = _back_project_3d_raw(grad_sino.contiguous(), euler_zyz_deg, Nz=ctx.Nz)
        return grad_vol, None, None


class _BP3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, euler_zyz_deg, Nz):
        ctx.save_for_backward(euler_zyz_deg)
        ctx.N_det = sinogram.shape[-1]
        return _back_project_3d_raw(sinogram, euler_zyz_deg, Nz)

    @staticmethod
    def backward(ctx, grad_vol):
        (euler_zyz_deg,) = ctx.saved_tensors
        grad_sino = _forward_project_3d_raw(grad_vol.contiguous(), euler_zyz_deg, N_det=ctx.N_det)
        return grad_sino, None, None


class _FPResire(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, euler_zyz_deg, N_det):
        ctx.save_for_backward(euler_zyz_deg)
        ctx.Nz = volume.shape[-2]
        return _forward_project_resire_raw(volume, euler_zyz_deg, N_det)

    @staticmethod
    def backward(ctx, grad_sino):
        (euler_zyz_deg,) = ctx.saved_tensors
        grad_vol = _back_project_resire_raw(grad_sino.contiguous(), euler_zyz_deg, Nz=ctx.Nz)
        return grad_vol, None, None


class _BPResire(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, euler_zyz_deg, Nz):
        ctx.save_for_backward(euler_zyz_deg)
        ctx.N_det = sinogram.shape[-1]
        return _back_project_resire_raw(sinogram, euler_zyz_deg, Nz)

    @staticmethod
    def backward(ctx, grad_vol):
        (euler_zyz_deg,) = ctx.saved_tensors
        grad_sino = _forward_project_resire_raw(grad_vol.contiguous(), euler_zyz_deg, N_det=ctx.N_det)
        return grad_sino, None, None
