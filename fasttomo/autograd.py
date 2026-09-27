"""
Autograd and raw kernel layer of FastTomo's projections.

* The raw layer (`_launch` and the ``*_raw`` functions): one uncached call of
  a CPU or CUDA kernel, no autograd. The projection operators
  (fasttomo/_operator.py) are built on it.
* Two generic Functions, `_Project` and `_BackProject`, make any projection
  operator differentiable: FP.backward = the operator's BP (and its
  ``grad_geom`` for the angles / rotation matrices), BP.backward = its FP.
  ``forward_project``, ``back_project``, the ``*_3d`` variants, the projector
  classes and the algorithms all go through them (ARCH P11-F2), so
  ``loss.backward()`` flows gradients into a ``requires_grad=True`` volume,
  sinogram, tilt-angle or rotation-matrix tensor.
* The RESIRE pair keeps its own Functions `_FPResire` / `_BPResire` around
  the frozen RESIRE wrappers (the RESIRE operator calls them).
* The fused 3-D TV loss `_TVLossIso3D`.

Caveats:
- The CUDA texture-based BPs are only approximate adjoints (~15-65 % relative
  error in the inner-product test). Sufficient as descent directions (SIRT and
  RESIRE rely on this), but `torch.autograd.gradcheck` will fail by design.
- Single-axis tilt-angle gradient: `_grad_angle_raw` computes ∂L/∂θ
  (CPU + CUDA).
- 3D rotation-matrix gradient: the rotmat operators take the rotation
  matrices directly so PyTorch can trace through `euler_zyz_to_rotmat`
  outside the Function boundary; `_grad_rotmat_raw` (or the cached operator's
  texture-reusing kernel) gives ∂L/∂M and autograd chains to
  ∂L/∂euler_zyz_deg (CPU + CUDA). RESIRE's rotation-matrix gradient is
  intentionally not exposed — angular refinement uses brute-force search
  (`refine_angles`) rather than autograd through the RESIRE inner loop.

See: fasttomo-wiki/wiki/concepts/autograd-projectors.md
"""

from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable

import fasttomo._core as _core
# Moved to fasttomo._backend (ARCH P8); imported here so the raw layer below
# and the old ``fasttomo.autograd._X`` import paths keep working.
from fasttomo._backend import (  # noqa: F401
    _assert_cuda,
    _assert_dtype_match,
    _check_shapes,
    _data_ptr,
    _ensure_fp32,
    _io_dtype_code,
    _on_device,
    _resolve_num_threads,
    _stream_ptr,
)


# ── ZYZ Euler → rotation matrix ──────────────────────────────────────────────

def euler_zyz_to_rotmat(euler_zyz_deg: torch.Tensor) -> torch.Tensor:
    """Convert ZYZ Euler angles to 3×3 rotation matrices.

    Parameters
    ----------
    euler_zyz_deg : Tensor, shape ``[n_angles, 3]``
        Columns are ``(phi, theta, psi)`` in degrees. ``theta`` is the tilt
        angle; a single-axis tilt series is ``(0, angle, 0)``.

    Returns
    -------
    Tensor, shape ``[n_angles, 3, 3]``, float32, on the input's device.
        Row-major ``R = R_Z(psi) · R_Y(theta) · R_Z(phi)`` (the RESIRE
        convention used by ``forward_project_3d``, ``RotmatProjector`` and
        ``run_ad_recon``'s pose refinement). Differentiable: autograd traces
        through it, which is how pose refinement gets ∂L/∂euler.
    """
    r   = math.pi / 180.0
    phi   = euler_zyz_deg[:, 0].float() * r
    theta = euler_zyz_deg[:, 1].float() * r
    psi   = euler_zyz_deg[:, 2].float() * r

    cp, sp = torch.cos(phi),   torch.sin(phi)
    ct, st = torch.cos(theta), torch.sin(theta)
    cu, su = torch.cos(psi),   torch.sin(psi)

    r00, r01, r02 = cu*ct*cp - su*sp, -cu*ct*sp - su*cp, cu*st
    r10, r11, r12 = su*ct*cp + cu*sp, -su*ct*sp + cu*cp, su*st
    r20, r21, r22 = -st*cp, st*sp, ct

    row0 = torch.stack([r00, r01, r02], dim=1)
    row1 = torch.stack([r10, r11, r12], dim=1)
    row2 = torch.stack([r20, r21, r22], dim=1)
    return torch.stack([row0, row1, row2], dim=1)


# Private name kept for existing imports (benchmark scripts, tests).
_euler_zyz_to_rotmat = euler_zyz_to_rotmat


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
    a = angles_deg.detach().reshape(-1).to(device=device, dtype=torch.float32) * (math.pi / 180.0)
    return torch.cos(a).contiguous(), torch.sin(a).contiguous()


def _kernel_input(t: torch.Tensor, name: str, ndim: int = 3,
                  fp32_only: bool = False) -> torch.Tensor:
    """``t`` as the kernels read it: detached and C-contiguous, float32 on CPU
    (and for fp32-only kernels), the caller's fp32/fp16 on CUDA. Raises on
    other dtypes (bf16 included) via `_io_dtype_code`, and ValueError if
    ``t`` is not ``ndim``-D (so the caller can unpack its dims)."""
    _io_dtype_code(t)
    if t.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {tuple(t.shape)}")
    t = t.detach()
    if fp32_only or not t.is_cuda:
        t = t.float()
    return t.contiguous()


def _mats_input(mats_flat: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Rotation matrices as the kernels read them: dense float32 [n, 9].
    Accepts [n, 9] or [n, 3, 3]; the count ``n`` is checked by `_launch`."""
    if tuple(mats_flat.shape[1:]) not in ((9,), (3, 3)):
        raise ValueError("rotation matrices must have shape [n_angles, 9] or "
                         f"[n_angles, 3, 3], got {tuple(mats_flat.shape)}")
    return mats_flat.detach().reshape(-1, 9).to(device=device, dtype=torch.float32).contiguous()


def _launch(op: str, out: torch.Tensor, *args, shapes, io_dtype: int | None = None,
            num_threads: int | None = None, label: str | None = None) -> torch.Tensor:
    """Run ``_core.<op>_cuda`` or ``_core.<op>_cpu`` on ``out``'s device.

    ``shapes`` lists ``(name, expected_shape)`` for every tensor argument, in
    order; each tensor must match it exactly (`_check_shapes`), be
    C-contiguous and live on ``out``'s device before it becomes a
    ``data_ptr()`` int. Other arguments pass through. The CUDA call gets
    ``io_dtype`` prepended (when given) and the current stream appended, and
    runs under the device guard; the CPU call gets the resolved
    ``num_threads`` appended. Returns ``out``.

    Zero-size inputs or outputs (``n_angles``, ``Ny``, ``Nz``, ``Nx`` or
    ``N_det`` of 0) raise ValueError on both backends. The CUDA kernels
    cannot launch an empty grid or build an empty texture, and the CPU
    kernels used to return empty tensors instead; one backend-independent
    error is the chosen contract. ``label`` (default ``op``) heads that
    error: the public function a one-shot call came from.
    """
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    if len(shapes) != len(tensors):
        raise RuntimeError(f"fasttomo internal error: {op} declares "
                           f"{len(shapes)} shapes for {len(tensors)} tensors")
    _check_shapes(op, [(n, t, sh) for (n, sh), t in zip(shapes, tensors)])
    for name, shape in shapes:
        if 0 in tuple(shape):
            raise ValueError(
                f"{label or op}: {name} has a zero-size dimension {tuple(shape)}; every "
                "dimension (n_angles, Ny, Nz, Nx, N_det) must be at least 1")
    names = iter(n for n, _ in shapes)
    ptrs = [_data_ptr(a, f"{op}: {next(names)}", out.device)
            if isinstance(a, torch.Tensor) else a
            for a in args]
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


def _forward_project_parallel_raw(
    volume: torch.Tensor,
    angles_deg: torch.Tensor,
    N_det: int = -1,
    num_threads: int | None = None,
    *,
    label: str | None = None,
) -> torch.Tensor:
    """Single-axis FP. Output dtype = volume dtype (CPU computes in fp32)."""
    vol = _kernel_input(volume, "volume")
    Ny, Nz, Nx = vol.shape
    n_angles = angles_deg.numel()
    N_det = Nx if N_det < 0 else N_det
    cos_a, sin_a = _trig_on_device(angles_deg, vol.device)
    sino = vol.new_empty(n_angles, Ny, N_det)
    _launch("forward_project_parallel", sino,
            vol, sino, cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det,
            shapes=[("volume", (Ny, Nz, Nx)), ("sinogram", (n_angles, Ny, N_det)),
                    ("cos", (n_angles,)), ("sin", (n_angles,))],
            io_dtype=_io_dtype_code(vol), num_threads=num_threads, label=label)
    return sino.to(volume.dtype)


def _back_project_parallel_raw(
    sinogram: torch.Tensor,
    angles_deg: torch.Tensor,
    Nz: int = -1,
    num_threads: int | None = None,
    *,
    Nx: int = -1,
    label: str | None = None,
) -> torch.Tensor:
    """Single-axis BP. Output dtype = sinogram dtype (CPU computes in fp32).
    ``Nx`` and ``Nz`` default to N_det; pass the volume's Nx when the
    detector is wider or narrower than the volume (the kernels centre both)."""
    sino = _kernel_input(sinogram, "sinogram")
    n_angles, Ny, N_det = sino.shape
    Nx = N_det if Nx < 0 else Nx
    Nz = N_det if Nz < 0 else Nz
    cos_a, sin_a = _trig_on_device(angles_deg, sino.device)
    vol = sino.new_empty(Ny, Nz, Nx)
    _launch("back_project_parallel", vol,
            sino, vol, cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det,
            shapes=[("sinogram", (n_angles, Ny, N_det)), ("volume", (Ny, Nz, Nx)),
                    ("angles", (n_angles,)), ("angles", (n_angles,))],
            io_dtype=_io_dtype_code(sino), num_threads=num_threads, label=label)
    return vol.to(sinogram.dtype)


def _forward_project_rotmat_raw(
    volume: torch.Tensor,
    mats_flat: torch.Tensor,   # [n_angles, 9]
    N_det: int = -1,
    num_threads: int | None = None,
    *,
    label: str | None = None,
) -> torch.Tensor:
    """Rotmat FP from rotation matrices (the euler→rotmat conversion lives in
    PyTorch, outside the autograd Function, so ∂L/∂euler chains through it).
    ``num_threads`` only applies to the CPU path. Output dtype = volume
    dtype; the matrices are always read as float32."""
    vol = _kernel_input(volume, "volume")
    Ny, Nz, Nx = vol.shape
    mats = _mats_input(mats_flat, vol.device)
    n_angles = mats.shape[0]
    N_det = Nx if N_det < 0 else N_det
    sino = vol.new_empty(n_angles, Ny, N_det)
    _launch("forward_project_rotmat", sino,
            vol, sino, mats, Ny, Nz, Nx, n_angles, N_det,
            shapes=[("volume", (Ny, Nz, Nx)), ("sinogram", (n_angles, Ny, N_det)),
                    ("matrices", (n_angles, 9))],
            io_dtype=_io_dtype_code(vol), num_threads=num_threads, label=label)
    return sino.to(volume.dtype)


def _back_project_rotmat_raw(
    sinogram: torch.Tensor,
    mats_flat: torch.Tensor,   # [n_angles, 9]
    Nz: int = -1,
    num_threads: int | None = None,
    *,
    Nx: int = -1,
    label: str | None = None,
) -> torch.Tensor:
    """Adjoint partner of _forward_project_rotmat_raw. Output dtype =
    sinogram dtype. ``Nx`` and ``Nz``
    default to N_det."""
    sino = _kernel_input(sinogram, "sinogram")
    n_angles, Ny, N_det = sino.shape
    Nx = N_det if Nx < 0 else Nx
    Nz = N_det if Nz < 0 else Nz
    mats = _mats_input(mats_flat, sino.device)
    vol = sino.new_empty(Ny, Nz, Nx)
    _launch("back_project_rotmat", vol,
            sino, vol, mats, Ny, Nz, Nx, n_angles, N_det,
            shapes=[("sinogram", (n_angles, Ny, N_det)), ("volume", (Ny, Nz, Nx)),
                    ("matrices", (n_angles, 9))],
            io_dtype=_io_dtype_code(sino), num_threads=num_threads, label=label)
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
    through `euler_zyz_to_rotmat` to give ∂L/∂euler_zyz_deg. ``num_threads``
    only applies to the CPU path.

    Always runs in fp32 (gradient kernels stay at full precision); fp16
    inputs are upcast on entry. Output grad is fp32 — autograd's chain rule
    will downcast as needed.
    """
    vol = _kernel_input(volume, "volume", fp32_only=True)
    gs = _kernel_input(grad_sino, "grad_sino", fp32_only=True).to(vol.device)
    Ny, Nz, Nx = vol.shape
    mats = _mats_input(mats_flat, vol.device)
    n_angles = mats.shape[0]
    N_det = gs.shape[2]
    grad_mats = torch.zeros(n_angles, 9, dtype=torch.float32, device=vol.device)
    return _launch("grad_rotmat", grad_mats,
                   vol, gs, mats, grad_mats, Ny, Nz, Nx, n_angles, N_det,
                   shapes=[("volume", (Ny, Nz, Nx)), ("grad_sino", (n_angles, Ny, N_det)),
                           ("matrices", (n_angles, 9)), ("grad_mats", (n_angles, 9))],
                   num_threads=num_threads)


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
    vol = _kernel_input(volume, "volume", fp32_only=True)
    gs = _kernel_input(grad_sino, "grad_sino", fp32_only=True).to(vol.device)
    Ny, Nz, Nx = vol.shape
    n_angles, _, N_det = gs.shape
    cos_a, sin_a = _trig_on_device(angles_deg, vol.device)
    grad_a = torch.zeros(n_angles, dtype=torch.float32, device=vol.device)
    _launch("grad_angle", grad_a,
            vol, gs, cos_a, sin_a, grad_a, Ny, Nz, Nx, n_angles, N_det,
            shapes=[("volume", (Ny, Nz, Nx)), ("grad_sino", (n_angles, Ny, N_det)),
                    ("angles", (n_angles,)), ("angles", (n_angles,)),
                    ("grad_angles", (n_angles,))],
            num_threads=num_threads)
    return grad_a * (math.pi / 180.0)


def _check_resire_shapes(op: str, data: torch.Tensor, data_name: str,
                         euler_zyz_deg: torch.Tensor,
                         n_views: int | None = None) -> None:
    """Shape guard for the frozen RESIRE wrappers (validation only; the
    kernel calls and their numerics are untouched).

    ``data`` (volume or sinogram) must be 3-D with no zero-size dim, and
    ``euler_zyz_deg`` must be ``[n_views, 3]``. For the back-projection
    ``n_views`` is the sinogram's view count; the matrices are read per view,
    so a count mismatch would otherwise surface as an opaque reshape error
    (or, for a column count other than 3, silently use the wrong angles)."""
    if data.ndim != 3:
        raise ValueError(f"{op}: {data_name} must be 3-D, got shape {tuple(data.shape)}")
    if data.numel() == 0:
        raise ValueError(f"{op}: {data_name} has a zero-size dimension "
                         f"{tuple(data.shape)}")
    if euler_zyz_deg.ndim != 2 or euler_zyz_deg.shape[1] != 3:
        raise ValueError(f"{op}: euler_zyz_deg must have shape [n_angles, 3], "
                         f"got {tuple(euler_zyz_deg.shape)}")
    if euler_zyz_deg.shape[0] == 0:
        raise ValueError(f"{op}: euler_zyz_deg has no views")
    if n_views is not None and euler_zyz_deg.shape[0] != n_views:
        raise ValueError(f"{op}: sinogram has {n_views} views but euler_zyz_deg "
                         f"has {euler_zyz_deg.shape[0]} rows")


def _forward_project_resire_raw(
    volume: torch.Tensor,
    euler_zyz_deg: torch.Tensor,
    N_det: int = -1,
) -> torch.Tensor:
    _assert_cuda(volume, "forward_project_resire")
    if not _core.cuda_available:
        raise RuntimeError("fasttomo was compiled without CUDA support")
    _ensure_fp32(volume, "forward_project_resire")
    _check_resire_shapes("forward_project_resire", volume, "volume", euler_zyz_deg)
    if N_det == 0:
        raise ValueError("forward_project_resire: N_det must be positive")

    Ny, Nz, Nx = volume.shape
    n_angles = euler_zyz_deg.shape[0]
    if N_det < 0:
        N_det = Nx

    vol  = volume.contiguous().float()
    sino = torch.zeros(n_angles, Ny, N_det, dtype=torch.float32, device=vol.device)

    mats      = euler_zyz_to_rotmat(euler_zyz_deg).to(device=vol.device)
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
    *,
    Nx: int = -1,
) -> torch.Tensor:
    _assert_cuda(sinogram, "back_project_resire")
    if not _core.cuda_available:
        raise RuntimeError("fasttomo was compiled without CUDA support")
    _ensure_fp32(sinogram, "back_project_resire")
    _check_resire_shapes("back_project_resire", sinogram, "sinogram", euler_zyz_deg,
                         n_views=sinogram.shape[0] if sinogram.ndim == 3 else None)
    if Nz == 0 or Nx == 0:
        raise ValueError("back_project_resire: Nz and Nx must be positive")

    n_angles, Ny, N_det = sinogram.shape
    if Nx < 0:
        Nx = N_det
    if Nz < 0:
        Nz = N_det

    sino = sinogram.contiguous().float()
    vol  = torch.zeros(Ny, Nz, Nx, dtype=torch.float32, device=sino.device)

    mats      = euler_zyz_to_rotmat(euler_zyz_deg).to(device=sino.device)
    mats_flat = mats.reshape(n_angles, 9).contiguous()

    with _on_device(sino):
        _core.back_project_resire_cuda(
            sino.data_ptr(), vol.data_ptr(), mats_flat.data_ptr(),
            Ny, Nz, Nx, n_angles, N_det,
            _stream_ptr(sino.device),
        )
    return vol * 0.25


# ── RESIRE Functions (frozen wrappers) ─────────────────────────────────────

class _FPResire(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, euler_zyz_deg, N_det):
        ctx.save_for_backward(euler_zyz_deg)
        ctx.Nz = volume.shape[-2]
        ctx.Nx = volume.shape[-1]
        return _forward_project_resire_raw(volume, euler_zyz_deg, N_det)

    @staticmethod
    def backward(ctx, grad_sino):
        (euler_zyz_deg,) = ctx.saved_tensors
        grad_vol = _back_project_resire_raw(grad_sino.contiguous(), euler_zyz_deg,
                                            Nz=ctx.Nz, Nx=ctx.Nx)
        return grad_vol, None, None


class _BPResire(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, euler_zyz_deg, Nz, Nx=-1):
        ctx.save_for_backward(euler_zyz_deg)
        ctx.N_det = sinogram.shape[-1]
        return _back_project_resire_raw(sinogram, euler_zyz_deg, Nz, Nx=Nx)

    @staticmethod
    def backward(ctx, grad_vol):
        (euler_zyz_deg,) = ctx.saved_tensors
        grad_sino = _forward_project_resire_raw(grad_vol.contiguous(), euler_zyz_deg, N_det=ctx.N_det)
        return grad_sino, None, None, None


# ── generic projection Functions over a projection operator ────────────────
#
# ``op`` is a fasttomo._operator.ProjectionOperator: ``op.fp`` / ``op.bp`` are
# its raw (non-autograd) forward and back projection, ``op.grad_geom`` the
# gradient w.r.t. its per-call geometry ``geom`` (tilt angles or rotation
# matrices; ``None`` = the operator's default). The adapter decides how the
# kernels run (CPU, CUDA one-shot, CUDA cached); these two Functions only
# wire the adjoint pairs: FP.backward = BP (+ grad_geom), BP.backward = FP.


class _Project(torch.autograd.Function):
    """``op.fp(volume, geom)``, differentiable w.r.t. ``volume`` (via
    ``op.bp``) and ``geom`` (via ``op.grad_geom``)."""

    @staticmethod
    def forward(ctx, volume, geom, op):
        ctx.op = op
        ctx.geom_shape = None if geom is None else geom.shape
        need_geom = ctx.needs_input_grad[1]
        ctx.save_for_backward(volume.detach() if need_geom else None,
                              None if geom is None else geom.detach())
        sino = op.fp(volume, geom)
        # e.g. which volume upload a cached texture holds right now; handed
        # back to op.grad_geom, which alone knows what it means
        ctx.token = op.fp_token()
        return sino

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_sino):
        # The adjoint kernels are not themselves differentiable: a double
        # backward (create_graph=True) raises instead of silently dropping
        # the second-order term.
        volume, geom = ctx.saved_tensors
        op = ctx.op
        grad_sino = grad_sino.contiguous()
        grad_vol = op.bp(grad_sino, geom) if ctx.needs_input_grad[0] else None
        grad_geom = None
        if ctx.needs_input_grad[1]:
            grad_geom = op.grad_geom(volume, grad_sino, geom, token=ctx.token)
            grad_geom = grad_geom.reshape(ctx.geom_shape)
        return grad_vol, grad_geom, None


class _BackProject(torch.autograd.Function):
    """``op.bp(sinogram, geom)``, differentiable w.r.t. ``sinogram`` (via
    ``op.fp``). No geometry gradient: it would need the volume that
    produced the incoming gradient; use the forward projection as the
    differentiable handle on the geometry."""

    @staticmethod
    def forward(ctx, sinogram, geom, op):
        ctx.op = op
        ctx.save_for_backward(None if geom is None else geom.detach())
        return op.bp(sinogram, geom)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_vol):
        (geom,) = ctx.saved_tensors
        return ctx.op.fp(grad_vol.contiguous(), geom), None, None


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
    stream = _stream_ptr(vol.device)
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
    stream = _stream_ptr(vol.device)
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
