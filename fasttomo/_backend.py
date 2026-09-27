"""
Backend plumbing shared by the projection layer and the algorithms.

The CUDA device guard, the current-stream pointer, the dtype codes the
bindings and the texture cache expect, the input guards, the ``data_ptr``
and shape checks in front of every raw kernel call, and the OpenMP
thread-count policy of the CPU kernels. Everything here is private; the
old import paths (``fasttomo.autograd._on_device``,
``fasttomo.projector._dtype_to_tc_code``, ...) keep working through
re-imports in those modules.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os

import torch


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
# fp16 results beyond the fp16 range raise OverflowError instead of
# returning inf (`_check_fp16_range` below).

# fp16 range: the kernels accumulate in fp32 and round the result to fp16
# once, so a projection whose value exceeds the fp16 range (|x| > 65504)
# would come back as +-inf. Typical triggers: the back-projection of an
# unnormalised sinogram (a unit-density object 1024 voxels wide has ray
# sums ~800; summed over 180 views that is ~1.4e5), a forward projection of
# a large volume with values of ~20 or more, FBP of such data, or FISTA's
# first gradient BP(-sinogram). Such a result raises OverflowError instead
# of returning inf; scale the data into range (for example divide the
# sinogram by its maximum, or by n_angles before a back-projection, and
# undo it on the fp32 result) or use fp32.
#
# When: a projection op (functional API, projector classes, run_fbp,
# refine_angles, autograd backward) checks its own fp16 result right away,
# which reads one flag back to the host (a stream sync per fp16 call on
# CUDA). The fp16 iterative algorithms (run_sirt, run_sart, run_fista,
# run_fista_tv) instead OR a device-side flag per projection and read it
# once when they return (`_fp16_overflow_checked`): a per-call sync cost
# 1.7-1.8x on 64^3 fp16 FISTA/SIRT. Nothing is checked while a CUDA graph
# is being captured (run_sirt's graph path; SIRT normalises its residual
# and stays far from the limit), and fp32 is never checked.

_FP16_MAX = 65504.0

# None: check every fp16 projection at once. A list: an fp16 algorithm is
# running and collects device-side flags (`_fp16_overflow_checked`).
_fp16_overflow_flags: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "fasttomo_fp16_overflow_flags", default=None)


def _overflow_error(op_name: str, what: str) -> OverflowError:
    return OverflowError(
        f"{op_name}: {what} exceeds the fp16 range (|x| > {_FP16_MAX:g}) and "
        "overflowed to inf. The kernels accumulate in fp32, but fp16 results "
        "are stored in fp16. Scale the input into range (for example divide "
        "the sinogram by its maximum or by n_angles, and multiply the fp32 "
        "result back) or use float32."
    )


def _check_fp16_range(out: torch.Tensor, inp: torch.Tensor, op_name: str) -> None:
    """Raise OverflowError when an fp16 projection overflowed: ``out`` holds
    an inf although ``inp`` (the projected volume or sinogram) is finite.
    No-op for other dtypes and during CUDA-graph capture; inside an fp16
    algorithm only a device-side flag is recorded (see the comment above)."""
    if out.dtype != torch.float16:
        return
    if out.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    flags = _fp16_overflow_flags.get()
    if flags is not None:
        # One reduction kernel, no sync: the fp32 sum of an fp16 tensor is
        # finite unless it holds an inf or NaN (|sum| <= 65504 * numel).
        flags.append(out.sum(dtype=torch.float32))
        return
    if not bool(torch.isinf(out).any()):
        return
    if not bool(torch.isfinite(inp).all()):
        return          # the inf came in with the input
    raise _overflow_error(op_name, "the fp16 projection result")


def _fp16_overflow_checked(fn):
    """Decorator for the fp16-capable iterative algorithms: their
    projections record device-side overflow flags, read once on return; an
    overflow (with a finite sinogram) raises OverflowError then."""
    name = fn.__name__

    @functools.wraps(fn)
    def wrapper(sinogram, *args, **kwargs):
        if not (isinstance(sinogram, torch.Tensor) and sinogram.dtype == torch.float16):
            return fn(sinogram, *args, **kwargs)
        flags: list = []
        token = _fp16_overflow_flags.set(flags)
        try:
            result = fn(sinogram, *args, **kwargs)
        finally:
            _fp16_overflow_flags.reset(token)
        if (flags and not bool(torch.isfinite(torch.stack(flags)).all())
                and bool(torch.isfinite(sinogram).all())):
            raise _overflow_error(name, "a forward or back-projection of this fp16 run")
        return result

    return wrapper


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


def _stream_ptr(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def _data_ptr(t: torch.Tensor, name: str, device: torch.device | None = None) -> int:
    """``t.data_ptr()`` for a kernel binding, refusing a strided tensor (the
    kernels index dense row-major buffers) or one on the wrong device."""
    if not t.is_contiguous():
        raise ValueError(f"{name} must be C-contiguous (stride {t.stride()})")
    if device is not None and t.device != device:
        raise ValueError(f"{name} is on {t.device}, expected {device}")
    return t.data_ptr()


def _check_shapes(op: str, items) -> None:
    """Raise ValueError unless every ``(name, tensor, expected_shape)`` in
    ``items`` has exactly that shape (``tensor`` may be None: skipped). The
    kernels receive bare pointers plus dims, so a buffer that is smaller than
    the dims say is read or written out of bounds; this is the one place
    that prevents it."""
    for name, t, shape in items:
        if t is not None and tuple(t.shape) != tuple(shape):
            raise ValueError(
                f"{op}: {name} has shape {tuple(t.shape)}, expected "
                f"{tuple(shape)} (sizes must agree with the other inputs)")


# Texture-cache kind enum mirrors texture_cache.h:
_TC_VOL_LAYERED  = 0
_TC_SINO_LAYERED = 1
_TC_VOL_3D       = 2

# Texture-cache dtype enum mirrors texture_cache.h tc_dtype_t:
_TC_DTYPE_FP32 = 0
_TC_DTYPE_FP16 = 1


def _dtype_to_tc_code(dtype: torch.dtype) -> int:
    """Translate a torch dtype to the tc_dtype_t int used by tc_create."""
    if dtype == torch.float32:
        return _TC_DTYPE_FP32
    if dtype == torch.float16:
        return _TC_DTYPE_FP16
    if dtype == torch.bfloat16:
        raise TypeError(
            "Projector textures do not support torch.bfloat16: CUDA "
            "textures have no BF16 channel descriptor. Cast to "
            "torch.float16 or torch.float32."
        )
    raise TypeError(
        f"Projector textures only support float32 and float16 (got {dtype})."
    )
