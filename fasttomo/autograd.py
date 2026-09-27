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

import contextlib
import math
import os

import torch

import fasttomo._core as _core


def _on_device(t: torch.Tensor):
    """Pin the current CUDA device to ``t.device`` for the enclosed block.

    Any CUDA allocation or kernel launch issued by ``_core.*_cuda*`` runs on
    the *current* device, not the device of the data pointer. Without this
    guard, calling a CUDA op on cuda:1 while the process default is cuda:0
    allocates textures on cuda:0 and crashes the kernel with
    ``cudaErrorIllegalAddress``. No-op on CPU tensors.
    """
    return torch.cuda.device(t.device) if t.is_cuda else contextlib.nullcontext()


# ── private guards ───────────────────────────────────────────────────────────

def _assert_cuda(t, name="op"):
    if not t.is_cuda:
        raise RuntimeError(f"{name} requires a CUDA tensor; got {t.device}")

def _ensure_fp32(t, op_name):
    if t.dtype != torch.float32:
        raise TypeError(f"{op_name} is fp32-only (got {t.dtype}). "
                        "Cast your tensor to float32 first.")


def _assert_dtype_match(a: torch.Tensor, b: torch.Tensor,
                        a_name: str, b_name: str, op_name: str) -> None:
    if a.dtype != b.dtype:
        raise TypeError(
            f"{op_name}: {a_name} ({a.dtype}) and {b_name} ({b.dtype}) "
            "must have the same dtype. Cast one to match the other."
        )


def _check_inputs(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor | None,
    Nz: int,
    *,
    allowed_dtypes: tuple[torch.dtype, ...],
    op_name: str,
    require_cuda: bool = False,
) -> tuple[int, int, int, int, torch.device]:
    """Validate and resolve sinogram/angles inputs for reconstruction algorithms.

    Returns (n_angles, Ny, N_det, Nz_resolved, device). Never coerces.

    Raises:
        RuntimeError  if require_cuda and sinogram is not on CUDA.
        TypeError     if sinogram.dtype not in allowed_dtypes.
        ValueError    if sinogram is not 3-D, angles_deg is not 1-D,
                      or sinogram.shape[0] != len(angles_deg).
    """
    if require_cuda and not sinogram.is_cuda:
        raise RuntimeError(
            f"{op_name} requires a CUDA tensor; got {sinogram.device}"
        )

    if sinogram.dtype not in allowed_dtypes:
        names = " or ".join(str(d) for d in allowed_dtypes)
        raise TypeError(
            f"{op_name} requires {names} sinogram (got {sinogram.dtype}). "
            "Cast with .float() or .half() first."
        )

    if sinogram.ndim != 3:
        raise ValueError(
            f"{op_name}: sinogram must be 3-D [n_angles, Ny, N_det], "
            f"got shape {tuple(sinogram.shape)}"
        )

    n_angles, Ny, N_det = sinogram.shape

    if angles_deg is not None:
        if angles_deg.ndim != 1:
            raise ValueError(
                f"{op_name}: angles_deg must be 1-D [n_angles], "
                f"got shape {tuple(angles_deg.shape)}"
            )
        if len(angles_deg) != n_angles:
            raise ValueError(
                f"{op_name}: sinogram has {n_angles} projections but "
                f"angles_deg has {len(angles_deg)} entries."
            )

    Nz_resolved = N_det if Nz < 0 else Nz
    return n_angles, Ny, N_det, Nz_resolved, sinogram.device


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
# The autograd Function classes below call these so that `Function.backward`
# invocations of the adjoint partner cannot recurse into autograd.
#
# Both backends share one binding convention (`csrc/bindings_*.cpp`): every
# array is passed as the integer ``data_ptr()`` of a tensor followed by the
# dims, so a raw pointer is only valid for a C-contiguous tensor of the
# kernel's dtype. `_launch` is the single place that turns tensors into
# pointers, and it refuses anything else.

def _trig_on_device(angles_deg: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin of `angles_deg` (degrees) computed on `device`. Always
    contiguous float32. The same tables feed the CPU and the CUDA kernels."""
    a = angles_deg.detach().to(device=device, dtype=torch.float32) * (math.pi / 180.0)
    return torch.cos(a).contiguous(), torch.sin(a).contiguous()


def _stream_ptr(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def _kernel_input(t: torch.Tensor, fp32_only: bool = False) -> torch.Tensor:
    """``t`` as the kernels read it: detached and C-contiguous, float32 on CPU
    (and for fp32-only kernels), the caller's fp32/fp16 on CUDA. Raises on
    other dtypes (bf16 included) via `_io_dtype_code`."""
    _io_dtype_code(t)
    t = t.detach()
    if fp32_only or not t.is_cuda:
        t = t.float()
    return t.contiguous()


def _mats_input(mats_flat: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Rotation matrices as the kernels read them: dense float32 [n, 9]."""
    return mats_flat.detach().reshape(-1, 9).to(device=device, dtype=torch.float32).contiguous()


def _data_ptr(t: torch.Tensor, name: str, device: torch.device | None = None) -> int:
    """``t.data_ptr()`` for a kernel binding, refusing a strided tensor (the
    kernels index dense row-major buffers) or one on the wrong device."""
    if not t.is_contiguous():
        raise ValueError(f"{name} must be C-contiguous (stride {t.stride()})")
    if device is not None and t.device != device:
        raise ValueError(f"{name} is on {t.device}, expected {device}")
    return t.data_ptr()


def _launch(op: str, out: torch.Tensor, *args, io_dtype: int | None = None,
            num_threads: int | None = None) -> torch.Tensor:
    """Run ``_core.<op>_cuda`` or ``_core.<op>_cpu`` on ``out``'s device.

    Tensor arguments become ``data_ptr()`` ints after checking that each is
    C-contiguous and on ``out``'s device; other arguments pass through. The
    CUDA call gets ``io_dtype`` prepended (when given) and the current stream
    appended, and runs under the device guard; the CPU call gets the resolved
    ``num_threads`` appended. Returns ``out``.
    """
    ptrs = [_data_ptr(a, f"{op} argument {i}", out.device)
            if isinstance(a, torch.Tensor) else a
            for i, a in enumerate(args)]
    if out.is_cuda:
        if not _core.cuda_available:
            raise RuntimeError("fasttomo was compiled without CUDA support")
        if io_dtype is not None:
            ptrs.insert(0, io_dtype)
        with _on_device(out):
            getattr(_core, op + "_cuda")(*ptrs, _stream_ptr(out.device))
    else:
        getattr(_core, op + "_cpu")(*ptrs, _resolve_num_threads(num_threads))
    return out


def _forward_project_raw(
    volume: torch.Tensor,
    angles_deg: torch.Tensor,
    N_det: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    """Single-axis FP. Output dtype = volume dtype (CPU computes in fp32)."""
    vol = _kernel_input(volume)
    Ny, Nz, Nx = vol.shape
    n_angles = angles_deg.numel()
    N_det = Nx if N_det < 0 else N_det
    cos_a, sin_a = _trig_on_device(angles_deg, vol.device)
    sino = vol.new_empty(n_angles, Ny, N_det)
    _launch("forward_project_parallel", sino,
            vol, sino, cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det,
            io_dtype=_io_dtype_code(vol), num_threads=num_threads)
    return sino.to(volume.dtype)


def _back_project_raw(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    Nz: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    """Single-axis BP. Output dtype = sinogram dtype (CPU computes in fp32)."""
    sino = _kernel_input(sinogram)
    n_angles, Ny, N_det = sino.shape
    Nx = N_det
    Nz = N_det if Nz < 0 else Nz
    cos_a, sin_a = _trig_on_device(angles_deg, sino.device)
    vol = sino.new_empty(Ny, Nz, Nx)
    _launch("back_project_parallel", vol,
            sino, vol, cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det,
            io_dtype=_io_dtype_code(sino), num_threads=num_threads)
    return vol.to(sinogram.dtype)


def _forward_project_3d_raw(
    volume: torch.Tensor,
    euler_zyz_deg: torch.Tensor,
    N_det: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    mats = _euler_zyz_to_rotmat(euler_zyz_deg).reshape(-1, 9)
    return _forward_project_3d_mats_raw(volume, mats, N_det, num_threads)


def _back_project_3d_raw(
    sinogram: torch.Tensor,
    euler_zyz_deg: torch.Tensor,
    Nz: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    mats = _euler_zyz_to_rotmat(euler_zyz_deg).reshape(-1, 9)
    return _back_project_3d_mats_raw(sinogram, mats, Nz, num_threads)


def _forward_project_3d_mats_raw(
    volume: torch.Tensor,
    mats_flat: torch.Tensor,   # [n_angles, 9]
    N_det: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    """Rotmat FP that bypasses the euler→rotmat step (mats already given).
    Used by _FP3D's autograd Function so the conversion lives in PyTorch.
    ``num_threads`` only applies to the CPU path. Output dtype = volume
    dtype; the matrices are always read as float32."""
    vol = _kernel_input(volume)
    Ny, Nz, Nx = vol.shape
    mats = _mats_input(mats_flat, vol.device)
    n_angles = mats.shape[0]
    N_det = Nx if N_det < 0 else N_det
    sino = vol.new_empty(n_angles, Ny, N_det)
    _launch("forward_project_rotmat", sino,
            vol, sino, mats, Ny, Nz, Nx, n_angles, N_det,
            io_dtype=_io_dtype_code(vol), num_threads=num_threads)
    return sino.to(volume.dtype)


def _back_project_3d_mats_raw(
    sinogram: torch.Tensor,
    mats_flat: torch.Tensor,   # [n_angles, 9]
    Nz: int = -1,
    num_threads: int | None = None,
) -> torch.Tensor:
    """Adjoint partner of _forward_project_3d_mats_raw (used inside
    _FP3D.backward). Output dtype = sinogram dtype."""
    sino = _kernel_input(sinogram)
    n_angles, Ny, N_det = sino.shape
    Nx = N_det
    Nz = N_det if Nz < 0 else Nz
    mats = _mats_input(mats_flat, sino.device)
    vol = sino.new_empty(Ny, Nz, Nx)
    _launch("back_project_rotmat", vol,
            sino, vol, mats, Ny, Nz, Nx, n_angles, N_det,
            io_dtype=_io_dtype_code(sino), num_threads=num_threads)
    return vol.to(sinogram.dtype)


def _grad_rotmat_raw(
    volume: torch.Tensor,
    grad_sino: torch.Tensor,
    mats_flat: torch.Tensor,   # [n_angles, 9]
    num_threads: int | None = None,
) -> torch.Tensor:
    """
    Returns [n_angles, 9] gradient of the loss w.r.t. the rotation-matrix
    entries used by forward_project_3d. PyTorch autograd then chains this
    through `_euler_zyz_to_rotmat` to give ∂L/∂euler_zyz_deg. ``num_threads``
    only applies to the CPU path.

    Always runs in fp32 (gradient kernels stay at full precision); fp16
    inputs are upcast on entry. Output grad is fp32 — autograd's chain rule
    will downcast as needed.
    """
    vol = _kernel_input(volume, fp32_only=True)
    gs = _kernel_input(grad_sino, fp32_only=True).to(vol.device)
    Ny, Nz, Nx = vol.shape
    mats = _mats_input(mats_flat, vol.device)
    n_angles = mats.shape[0]
    N_det = gs.shape[2]
    grad_mats = torch.zeros(n_angles, 9, dtype=torch.float32, device=vol.device)
    return _launch("grad_rotmat", grad_mats,
                   vol, gs, mats, grad_mats, Ny, Nz, Nx, n_angles, N_det,
                   num_threads=num_threads)


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

    with _on_device(vol):
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

    with _on_device(sino):
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
    num_threads: int | None = None,
) -> torch.Tensor:
    """
    Returns [n_angles] gradient of the loss w.r.t. tilt angles in degrees.

    Computed analytically: at each ray step, ∂sino/∂θ = ∂vol/∂x·(−z) + ∂vol/∂z·x,
    where the spatial derivatives use bilinear finite differences of the volume.
    The kernels are fp32-only, so fp16 or strided inputs are densified to fp32.
    """
    vol = _kernel_input(volume, fp32_only=True)
    gs = _kernel_input(grad_sino, fp32_only=True).to(vol.device)
    Ny, Nz, Nx = vol.shape
    n_angles, _, N_det = gs.shape
    cos_a, sin_a = _trig_on_device(angles_deg, vol.device)
    grad_a = torch.zeros(n_angles, dtype=torch.float32, device=vol.device)
    _launch("grad_angle", grad_a,
            vol, gs, cos_a, sin_a, grad_a, Ny, Nz, Nx, n_angles, N_det,
            num_threads=num_threads)
    return grad_a * (math.pi / 180.0)


class _FP2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, angles_deg, N_det, num_threads):
        ctx.save_for_backward(volume.detach(), angles_deg)
        ctx.Nz = volume.shape[-2]
        ctx.num_threads = num_threads
        return _forward_project_raw(volume, angles_deg, N_det, num_threads)

    @staticmethod
    def backward(ctx, grad_sino):
        volume, angles_deg = ctx.saved_tensors
        grad_sino = grad_sino.contiguous()
        grad_vol = _back_project_raw(grad_sino, angles_deg, Nz=ctx.Nz,
                                     num_threads=ctx.num_threads)
        grad_angles = None
        if ctx.needs_input_grad[1]:
            grad_angles = _grad_angle_raw(volume, grad_sino, angles_deg,
                                          num_threads=ctx.num_threads)
        return grad_vol, grad_angles, None, None


class _BP2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, angles_deg, Nz, num_threads):
        ctx.save_for_backward(angles_deg)
        ctx.N_det = sinogram.shape[-1]
        ctx.num_threads = num_threads
        return _back_project_raw(sinogram, angles_deg, Nz, num_threads)

    @staticmethod
    def backward(ctx, grad_vol):
        (angles_deg,) = ctx.saved_tensors
        grad_sino = _forward_project_raw(grad_vol.contiguous(), angles_deg,
                                         N_det=ctx.N_det,
                                         num_threads=ctx.num_threads)
        return grad_sino, None, None, None


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


# ── 3-D isotropic TV loss (fused CUDA kernel) ──────────────────────────────

def _tv_loss_iso_3d_raw(volume: torch.Tensor, eps: float) -> torch.Tensor:
    _assert_cuda(volume, "tv_loss_3d")
    _ensure_fp32(volume, "tv_loss_3d")
    if volume.ndim != 3:
        raise ValueError(f"tv_loss_3d expects a 3-D [Ny, Nz, Nx] tensor, "
                         f"got shape {tuple(volume.shape)}")
    vol = volume.contiguous()
    Ny, Nz, Nx = vol.shape
    acc = torch.zeros(1, dtype=torch.float32, device=vol.device)
    stream = torch.cuda.current_stream(vol.device).cuda_stream
    with _on_device(vol):
        _core.tv_loss_iso_3d_cuda(
            vol.data_ptr(), acc.data_ptr(),
            Ny, Nz, Nx, float(eps), stream,
        )
    N = max(1, (Ny - 1) * (Nz - 1) * (Nx - 1))
    return acc.squeeze(0) / float(N)


def _tv_loss_iso_3d_grad_raw(volume: torch.Tensor, grad_out: torch.Tensor,
                             eps: float) -> torch.Tensor:
    vol = volume.contiguous()
    Ny, Nz, Nx = vol.shape
    N = max(1, (Ny - 1) * (Nz - 1) * (Nx - 1))
    g_scaled = (grad_out.detach().to(dtype=torch.float32, device=vol.device)
                .reshape(1).contiguous() / float(N))
    grad_vol = torch.zeros_like(vol)
    stream = torch.cuda.current_stream(vol.device).cuda_stream
    with _on_device(vol):
        _core.tv_loss_iso_3d_grad_cuda(
            vol.data_ptr(), g_scaled.data_ptr(), grad_vol.data_ptr(),
            Ny, Nz, Nx, float(eps), stream,
        )
    return grad_vol


class _TVLossIso3D(torch.autograd.Function):
    """Fused isotropic 3-D TV loss with analytic gradient.

    Forward returns a scalar tensor (shape ``[]``); backward populates the
    gradient w.r.t. ``volume`` via a single voxel-driven kernel. Unlike the
    projection ops, this Function's adjoint is exact (it implements the
    closed-form gradient of a smooth scalar function), so ``gradcheck``
    would pass — we still verify against the pure-PyTorch reference instead
    of relying on it, matching the rest of the test suite.
    """

    @staticmethod
    def forward(ctx, volume, eps):
        ctx.save_for_backward(volume)
        ctx.eps = float(eps)
        return _tv_loss_iso_3d_raw(volume, eps)

    @staticmethod
    def backward(ctx, grad_out):
        (volume,) = ctx.saved_tensors
        grad_vol = _tv_loss_iso_3d_grad_raw(volume, grad_out, ctx.eps)
        return grad_vol, None


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
        sino = projector._fp_raw(volume, mats_flat)
        # Snapshot which volume upload the cache currently holds.
        ctx.vol_cache_epoch = getattr(projector, "_vol_cache_epoch", None)
        return sino

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
            # Fast path only if this backward still corresponds to the most
            # recent volume upload. Otherwise fall back to the uncached kernel
            # so multi-forward single-backward graphs stay correct.
            if getattr(ctx, "vol_cache_epoch", None) == getattr(proj, "_vol_cache_epoch", None):
                grad_mats = proj._grad_mats_raw(volume, grad_sino, mats_flat)
            else:
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


