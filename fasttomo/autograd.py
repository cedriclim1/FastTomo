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
- 2D tilt-angle gradient: `_FP2D` computes ∂L/∂θ via `_grad_angle_raw`
  (CPU + CUDA).
- 3D rotation-matrix gradient: `_FP3D` takes the [n_angles, 3, 3] rotation
  matrix tensor directly so PyTorch can trace through `_euler_zyz_to_rotmat`
  outside the Function boundary. Backward calls `_grad_rotmat_raw` for ∂L/∂M
  and lets autograd chain to ∂L/∂euler_zyz_deg automatically (CPU + CUDA).
  RESIRE's rotation-matrix gradient is intentionally not exposed — angular
  refinement uses brute-force search (`refine_angles`) rather than autograd
  through the RESIRE inner loop.

See: fasttomo-wiki/wiki/concepts/autograd-projectors.md
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

import fasttomo._core as _core


# ── private guards ───────────────────────────────────────────────────────────

def _assert_cuda(t, name="op"):
    if not t.is_cuda:
        raise RuntimeError(f"{name} requires a CUDA tensor; got {t.device}")

def _ensure_fp32(t, op_name):
    if t.dtype != torch.float32:
        raise TypeError(f"{op_name} is fp32-only (got {t.dtype}). "
                        "Cast your tensor to float32 first.")


# ── thread-count resolution ──────────────────────────────────────────────────
#
# Resolution order for the CPU 3D rotmat kernels:
#   1. Explicit num_threads kwarg on the Python call
#   2. OMP_NUM_THREADS environment variable
#   3. 1 (serial)
#
# Step 3 is deliberate — OpenMP's stock default of `omp_get_max_threads()`
# (≈ all cores) would silently parallelize when the user didn't ask. The
# library defaults to predictable serial execution; the caller opts in.

def _resolve_num_threads(num_threads: int | None) -> int:
    if num_threads is not None:
        return max(1, int(num_threads))
    env = os.environ.get("OMP_NUM_THREADS")
    if env is not None and env.strip():
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return 1


# ── dtype dispatch ─────────────────────────────────────────────────────────
#
# The 4 in-scope projection ops (FP/BP scalar + FP/BP rotmat) preserve the
# I/O dtype of the input tensor. On CUDA the kernels are templated on
# float32 / __half (single-channel fp16 textures, fp32 accumulators). On CPU
# kernels stay fp32 only — fp16 inputs are upcast on entry and the output is
# downcast back to the user's dtype. BF16 raises (no native CUDA texture
# support; see fasttomo-wiki/wiki/concepts/half-precision-projection.md).

def _io_dtype_code(t: torch.Tensor) -> int:
    """Map a tensor dtype to the io_dtype int the bindings expect.
    Returns 0 for float32, 1 for float16. Raises on bfloat16 (no native
    CUDA texture support) and other unsupported dtypes."""
    if t.dtype == torch.float32:
        return 0
    if t.dtype == torch.float16:
        return 1
    if t.dtype == torch.bfloat16:
        raise TypeError(
            "FastTomo projection ops do not support torch.bfloat16: CUDA "
            "textures have no BF16 channel descriptor. Cast to torch.float16 "
            "(half precision, mirrors TorchRadon) or torch.float32."
        )
    raise TypeError(
        f"FastTomo projection ops only support float32 and float16 "
        f"(got {t.dtype}). BF16 is intentionally not supported."
    )


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

def _trig_on_device(angles_deg: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin of `angles_deg` (degrees) computed on `device`. Always
    contiguous float32. Caller passes the resulting data_ptr()s through to
    the CUDA bindings."""
    a = angles_deg.detach().to(device=device, dtype=torch.float32) * (math.pi / 180.0)
    return torch.cos(a).contiguous(), torch.sin(a).contiguous()


def _stream_ptr(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def _forward_project_raw(
    volume: torch.Tensor,
    angles_deg: torch.Tensor,
    N_det: int = -1,
) -> torch.Tensor:
    io_dtype = _io_dtype_code(volume)
    if volume.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        Ny, Nz, Nx = volume.shape
        n_angles   = angles_deg.numel()
        if N_det < 0:
            N_det = Nx
        vol  = volume.contiguous()
        sino = torch.empty(n_angles, Ny, N_det,
                           dtype=volume.dtype, device=volume.device)
        cos_t, sin_t = _trig_on_device(angles_deg, volume.device)
        _core.forward_project_cuda(
            io_dtype,
            vol.data_ptr(), sino.data_ptr(),
            cos_t.data_ptr(), sin_t.data_ptr(),
            Ny, Nz, Nx, n_angles, N_det,
            _stream_ptr(volume.device),
        )
        return sino

    # CPU: kernel is fp32-only; upcast on entry, downcast on exit.
    angles_np = (angles_deg.detach().cpu().float() * (math.pi / 180.0)).numpy()
    vol_np  = volume.detach().float().numpy()
    sino_np = _core.forward_project_cpu(vol_np, angles_np, N_det)
    out = torch.from_numpy(sino_np)
    return out.to(volume.dtype) if volume.dtype != torch.float32 else out


def _back_project_raw(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    Nz: int = -1,
) -> torch.Tensor:
    io_dtype = _io_dtype_code(sinogram)
    if sinogram.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        n_angles, Ny, N_det = sinogram.shape
        Nx = N_det
        if Nz < 0:
            Nz = N_det
        sino = sinogram.contiguous()
        vol  = torch.empty(Ny, Nz, Nx,
                           dtype=sinogram.dtype, device=sinogram.device)
        cos_t, sin_t = _trig_on_device(angles_deg, sinogram.device)
        _core.back_project_cuda(
            io_dtype,
            sino.data_ptr(), vol.data_ptr(),
            cos_t.data_ptr(), sin_t.data_ptr(),
            Ny, Nz, Nx, n_angles, N_det,
            _stream_ptr(sinogram.device),
        )
        return vol

    # CPU: fp32-only; upcast on entry, downcast on exit.
    angles_np = (angles_deg.detach().cpu().float() * (math.pi / 180.0)).numpy()
    sino_np = sinogram.detach().float().numpy()
    vol_np  = _core.back_project_cpu(sino_np, angles_np, Nz)
    out = torch.from_numpy(vol_np)
    return out.to(sinogram.dtype) if sinogram.dtype != torch.float32 else out


def _forward_project_3d_raw(
    volume: torch.Tensor,
    euler_zyz_deg: torch.Tensor,
    N_det: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    mats      = _euler_zyz_to_rotmat(euler_zyz_deg).to(device=volume.device)
    n_angles  = euler_zyz_deg.shape[0]
    mats_flat = mats.reshape(n_angles, 9).contiguous()
    return _forward_project_3d_mats_raw(volume, mats_flat, N_det, num_threads)


def _back_project_3d_raw(
    sinogram: torch.Tensor,
    euler_zyz_deg: torch.Tensor,
    Nz: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    mats      = _euler_zyz_to_rotmat(euler_zyz_deg).to(device=sinogram.device)
    n_angles  = euler_zyz_deg.shape[0]
    mats_flat = mats.reshape(n_angles, 9).contiguous()
    return _back_project_3d_mats_raw(sinogram, mats_flat, Nz, num_threads)


def _forward_project_3d_mats_raw(
    volume: torch.Tensor,
    mats_flat: torch.Tensor,   # [n_angles, 9] float32, on volume.device
    N_det: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    """3D rotmat FP that bypasses the euler→rotmat step (mats already given).
    Used by _FP3D's autograd Function so the conversion lives in PyTorch.
    Dispatches to CPU or CUDA based on `volume.device`. ``num_threads`` only
    applies to the CPU path (the CUDA kernel manages its own grid).

    Preserves the volume's dtype on the output sinogram. Matrices stay
    float32. CPU path upcasts fp16 inputs to fp32 internally."""
    io_dtype = _io_dtype_code(volume)
    Ny, Nz, Nx = volume.shape
    n_angles = mats_flat.shape[0]
    if N_det < 0:
        N_det = Nx

    if volume.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        vol  = volume.contiguous()
        sino = torch.empty(n_angles, Ny, N_det, dtype=volume.dtype, device=vol.device)
        _core.forward_project_rotmat_cuda(
            io_dtype,
            vol.data_ptr(), sino.data_ptr(), mats_flat.data_ptr(),
            Ny, Nz, Nx, n_angles, N_det,
            _stream_ptr(vol.device),
        )
        return sino

    nt = _resolve_num_threads(num_threads)
    vol_np  = volume.detach().contiguous().float().numpy()
    mats_np = mats_flat.detach().contiguous().float().cpu().numpy()
    sino_np = _core.forward_project_3d_cpu(vol_np, mats_np, N_det, nt)
    out = torch.from_numpy(sino_np)
    return out.to(volume.dtype) if volume.dtype != torch.float32 else out


def _back_project_3d_mats_raw(
    sinogram: torch.Tensor,
    mats_flat: torch.Tensor,   # [n_angles, 9] float32, on sinogram.device
    Nz: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    """Adjoint of _forward_project_3d_mats_raw — used inside _FP3D.backward.
    Dispatches to CPU or CUDA based on `sinogram.device`. ``num_threads`` only
    applies to the CPU path. Preserves the sinogram's dtype on the output
    volume."""
    io_dtype = _io_dtype_code(sinogram)
    n_angles, Ny, N_det = sinogram.shape
    Nx = N_det
    if Nz < 0:
        Nz = N_det

    if sinogram.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        sino = sinogram.contiguous()
        vol  = torch.empty(Ny, Nz, Nx, dtype=sinogram.dtype, device=sino.device)
        _core.back_project_rotmat_cuda(
            io_dtype,
            sino.data_ptr(), vol.data_ptr(), mats_flat.data_ptr(),
            Ny, Nz, Nx, n_angles, N_det,
            _stream_ptr(sino.device),
        )
        return vol

    nt = _resolve_num_threads(num_threads)
    sino_np = sinogram.detach().contiguous().float().numpy()
    mats_np = mats_flat.detach().contiguous().float().cpu().numpy()
    vol_np  = _core.back_project_3d_cpu(sino_np, mats_np, Nz, nt)
    out = torch.from_numpy(vol_np)
    return out.to(sinogram.dtype) if sinogram.dtype != torch.float32 else out


def _grad_rotmat_raw(
    volume: torch.Tensor,
    grad_sino: torch.Tensor,
    mats_flat: torch.Tensor,   # [n_angles, 9] float32, on volume.device
    num_threads: int | None = None,
) -> torch.Tensor:
    """
    Returns [n_angles, 9] gradient of the loss w.r.t. the rotation-matrix
    entries used by forward_project_3d. PyTorch autograd then chains this
    through `_euler_zyz_to_rotmat` to give ∂L/∂euler_zyz_deg. Dispatches to
    CPU or CUDA based on `volume.device`. ``num_threads`` only applies to
    the CPU path.

    Always runs in fp32 (gradient kernels stay at full precision); fp16
    inputs are upcast on entry. Output grad is fp32 — autograd's chain rule
    will downcast as needed.
    """
    n_angles = mats_flat.shape[0]
    Ny, Nz, Nx = volume.shape
    _, _, N_det = grad_sino.shape

    if volume.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        vol  = volume.contiguous().float()
        gs   = grad_sino.contiguous().float()
        grad_mats = torch.zeros(n_angles, 9, dtype=torch.float32, device=vol.device)
        _core.grad_rotmat_cuda(
            vol.data_ptr(), gs.data_ptr(), mats_flat.data_ptr(), grad_mats.data_ptr(),
            Ny, Nz, Nx, n_angles, N_det,
            _stream_ptr(vol.device),
        )
        return grad_mats

    nt = _resolve_num_threads(num_threads)
    vol_np  = volume.detach().contiguous().float().numpy()
    gs_np   = grad_sino.detach().contiguous().float().numpy()
    mats_np = mats_flat.detach().contiguous().float().cpu().numpy()
    grad_np = _core.grad_rotmat_cpu(vol_np, gs_np, mats_np, nt)
    return torch.from_numpy(grad_np)


def _forward_project_resire_raw(
    volume: torch.Tensor,
    euler_zyz_deg: torch.Tensor,
    N_det: int = -1,
) -> torch.Tensor:
    _assert_cuda(volume, "forward_project_resire")
    if not _core.cuda_available:
        raise RuntimeError("fasttomo was compiled without CUDA support")
    _ensure_fp32(volume, "forward_project_resire")

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
        _stream_ptr(vol.device),
    )
    return sino * 0.25


def _back_project_resire_raw(
    sinogram: torch.Tensor,
    euler_zyz_deg: torch.Tensor,
    Nz: int = -1,
) -> torch.Tensor:
    _assert_cuda(sinogram, "back_project_resire")
    if not _core.cuda_available:
        raise RuntimeError("fasttomo was compiled without CUDA support")
    _ensure_fp32(sinogram, "back_project_resire")

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
        _stream_ptr(sino.device),
    )
    return vol * 0.25


# ── autograd.Function wrappers ───────────────────────────────────────────────
#
# Pattern for every pair (FP, BP):  FP.backward = BP_raw,  BP.backward = FP_raw.
# Non-tensor params (shape ints) live on `ctx` as Python attributes; tensors
# (the angle table) go through `save_for_backward`. We never return a gradient
# for the angles tensor — angle differentiability is deferred (would need
# Jacobian kernels w.r.t. rotation parameters).

def _grad_angle_raw(
    volume: torch.Tensor,
    grad_sino: torch.Tensor,
    angles_deg: torch.Tensor,
) -> torch.Tensor:
    """
    Returns [n_angles] gradient of the loss w.r.t. tilt angles in degrees.

    Computed analytically: at each ray step, ∂sino/∂θ = ∂vol/∂x·(−z) + ∂vol/∂z·x,
    where the spatial derivatives use bilinear finite differences of the volume.
    """
    n_angles, Ny, N_det = grad_sino.shape
    _, Nz, Nx = volume.shape

    if volume.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        cos_t, sin_t = _trig_on_device(angles_deg, volume.device)
        grad_a = torch.zeros(n_angles, dtype=torch.float32, device=volume.device)
        _core.grad_angle_cuda(
            volume.data_ptr(), grad_sino.data_ptr(),
            cos_t.data_ptr(), sin_t.data_ptr(), grad_a.data_ptr(),
            Ny, Nz, Nx, n_angles, N_det,
            _stream_ptr(volume.device),
        )
        return grad_a * (math.pi / 180.0)

    angles_np = (angles_deg.detach().cpu().float() * (math.pi / 180.0)).numpy().astype(np.float32)
    vol_np = volume.detach().float().numpy()
    gs_np  = grad_sino.detach().float().numpy()
    grad_np = _core.grad_angle_cpu(vol_np, gs_np, angles_np, N_det)
    return torch.from_numpy(grad_np) * (math.pi / 180.0)


class _FP2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, angles_deg, N_det):
        ctx.save_for_backward(volume.detach(), angles_deg)
        ctx.Nz = volume.shape[-2]
        return _forward_project_raw(volume, angles_deg, N_det)

    @staticmethod
    def backward(ctx, grad_sino):
        volume, angles_deg = ctx.saved_tensors
        grad_sino = grad_sino.contiguous()
        grad_vol = _back_project_raw(grad_sino, angles_deg, Nz=ctx.Nz)
        grad_angles = None
        if ctx.needs_input_grad[1]:
            grad_angles = _grad_angle_raw(volume, grad_sino, angles_deg)
        return grad_vol, grad_angles, None


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
    """
    Texture-based 3D rotmat forward projection, differentiable w.r.t. both
    `volume` and the rotation matrix `mats` ([n_angles, 3, 3]). The Euler →
    rotmat conversion lives in pure PyTorch (`_euler_zyz_to_rotmat`) outside
    this Function, so autograd chains ∂L/∂M into ∂L/∂euler_zyz_deg.

    ``num_threads`` is the resolved CPU OpenMP thread count; CUDA path ignores
    it. It is stashed on ctx so the backward (BP + grad_rotmat) uses the same
    parallelism the forward did.
    """

    @staticmethod
    def forward(ctx, volume, mats, N_det, num_threads):
        mats_flat = mats.detach().reshape(-1, 9).contiguous().to(
            device=volume.device, dtype=torch.float32
        )
        ctx.save_for_backward(volume.detach(), mats_flat)
        ctx.Nz = volume.shape[-2]
        ctx.num_threads = num_threads
        return _forward_project_3d_mats_raw(
            volume.detach(), mats_flat, N_det, num_threads
        )

    @staticmethod
    def backward(ctx, grad_sino):
        volume, mats_flat = ctx.saved_tensors
        grad_sino = grad_sino.contiguous()
        grad_vol = None
        if ctx.needs_input_grad[0]:
            grad_vol = _back_project_3d_mats_raw(
                grad_sino, mats_flat, Nz=ctx.Nz, num_threads=ctx.num_threads
            )
        grad_mats = None
        if ctx.needs_input_grad[1]:
            grad_mats = _grad_rotmat_raw(
                volume, grad_sino, mats_flat, num_threads=ctx.num_threads
            ).reshape(-1, 3, 3)
        return grad_vol, grad_mats, None, None


class _BP3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, euler_zyz_deg, Nz, num_threads):
        ctx.save_for_backward(euler_zyz_deg)
        ctx.N_det = sinogram.shape[-1]
        ctx.num_threads = num_threads
        return _back_project_3d_raw(sinogram, euler_zyz_deg, Nz, num_threads)

    @staticmethod
    def backward(ctx, grad_vol):
        (euler_zyz_deg,) = ctx.saved_tensors
        grad_sino = _forward_project_3d_raw(
            grad_vol.contiguous(), euler_zyz_deg,
            N_det=ctx.N_det, num_threads=ctx.num_threads,
        )
        return grad_sino, None, None, None


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


# ── cached autograd Functions ───────────────────────────────────────────────
#
# Used by ParallelProjector / RotmatProjector. Each Function takes a projector
# instance via ctx (non-tensor, non-grad) and routes through projector._fp_raw
# / ._bp_raw, which call the *_cuda_cached entry points and reuse the
# projector's persistent TextureCache. The angle-gradient path is intentionally
# absent on the parallel cached pair (use the functional API for that); the
# rotmat cached pair forwards mats grad through the existing grad_rotmat
# kernel (no texture, so caching gives nothing there).


class _FP2DCached(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, projector):
        ctx.projector = projector
        return projector._fp_raw(volume)

    @staticmethod
    def backward(ctx, grad_sino):
        grad_vol = ctx.projector._bp_raw(grad_sino.contiguous())
        return grad_vol, None


class _BP2DCached(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, projector):
        ctx.projector = projector
        return projector._bp_raw(sinogram)

    @staticmethod
    def backward(ctx, grad_vol):
        grad_sino = ctx.projector._fp_raw(grad_vol.contiguous())
        return grad_sino, None


class _FP3DCached(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, mats_flat, projector):
        ctx.projector = projector
        ctx.save_for_backward(volume.detach(), mats_flat.detach())
        return projector._fp_raw(volume, mats_flat)

    @staticmethod
    def backward(ctx, grad_sino):
        volume, mats_flat = ctx.saved_tensors
        proj = ctx.projector
        grad_sino = grad_sino.contiguous()
        grad_vol = None
        if ctx.needs_input_grad[0]:
            grad_vol = proj._bp_raw(grad_sino, mats_flat)
        grad_mats = None
        if ctx.needs_input_grad[1]:
            grad_mats = _grad_rotmat_raw(volume, grad_sino, mats_flat)
        return grad_vol, grad_mats, None


class _BP3DCached(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, mats_flat, projector):
        ctx.projector = projector
        ctx.save_for_backward(mats_flat.detach())
        return projector._bp_raw(sinogram, mats_flat)

    @staticmethod
    def backward(ctx, grad_vol):
        (mats_flat,) = ctx.saved_tensors
        proj = ctx.projector
        grad_sino = proj._fp_raw(grad_vol.contiguous(), mats_flat)
        # grad w.r.t. mats: needs the volume that produced grad_vol — adjoint
        # path doesn't have it, so we return None. Pose optimisation should
        # use forward (FP) as the differentiable handle on mats, not back.
        return grad_sino, None, None


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


