"""
The projection operator: the one internal seam for "A and Aᵀ of this
geometry on this device" (ARCH P1).

Every forward/back projection in FastTomo goes through a
:class:`ProjectionOperator` built by :func:`_make_operator`: the functional
API (``forward_project`` and friends build a transient one per call), the
public projector classes (``ParallelProjector`` / ``RotmatProjector`` own
one), and every algorithm (``_build_fp_bp`` and the loops of FBP, SART,
wSIRT, grad-descent, AD, RESIRE and ``refine_angles``).

An operator has two layers:

* raw, no autograd: ``fp(volume, geom)``, ``bp(sinogram, geom)`` and, where
  the geometry is differentiable, ``grad_geom(volume, grad_sino, geom)``;
* differentiable: ``project(volume, geom)`` / ``backproject(sinogram, geom)``
  run the raw layer through the generic autograd Functions
  ``_Project`` / ``_BackProject`` (fasttomo/autograd.py), whose backward is
  ``bp`` / ``fp`` / ``grad_geom`` of the same operator.

``geom`` is the per-call geometry: the tilt angles in degrees ``[n]`` for a
``"parallel"`` operator (``None`` = the angles it was built with), the
rotation matrices ``[n, 9]`` or ``[n, 3, 3]`` for ``"rotmat"``, the ZYZ Euler
angles ``[n, 3]`` for ``"resire"``. The view count ``n`` may differ between
calls (SART's subsets, ``refine_angles``' candidates); the volume shape and
``N_det`` are fixed.

Adapters (chosen by ``_make_operator``):

``UncachedOperator``     every call goes through the raw ``_launch`` layer
                         (fasttomo/autograd.py). The CPU adapter; on CUDA it
                         is the one-shot path: the functional API and FBP.
                         Its textures come from a small process-wide pool
                         (:class:`_TexturePool`) instead of being allocated
                         and freed (with a stream sync) on every call.
``CudaCachedOperator``   persistent ``TextureCache`` handles reused across
                         calls (what the cached projectors always did); the
                         parallel angle gradient falls back to the uncached
                         ``grad_angle`` kernel.
``ResireOperator``       the frozen RESIRE wrappers and their autograd
                         Functions, called exactly as before.
``_UserProjectorOperator``  a user-supplied projector object (``projector=``
                         in the algorithms): its public ``forward``/``back``.

Cached and uncached CUDA adapters run the same kernels with the same texture
layout, so they agree bit for bit (``tests/test_parity.py``).
"""

from __future__ import annotations

import itertools
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass

import torch

import fasttomo._core as _core
from fasttomo._backend import (
    _TC_SINO_LAYERED,
    _TC_VOL_3D,
    _TC_VOL_LAYERED,
    _check_finite_geometry,
    _check_fp16_range,
    _dtype_to_tc_code,
    _io_dtype_code,
    _stream_ptr,
)
from fasttomo.autograd import (
    _BackProject,
    _BPResire,
    _FPResire,
    _Project,
    _back_project_parallel_raw,
    _back_project_resire_raw,
    _back_project_rotmat_raw,
    _forward_project_parallel_raw,
    _forward_project_resire_raw,
    _forward_project_rotmat_raw,
    _grad_angle_raw,
    _grad_rotmat_raw,
)

_KINDS = ("parallel", "rotmat", "resire")


@dataclass(frozen=True, eq=False)
class Geometry:
    """What an operator projects between.

    ``kind``        ``"parallel"`` (single tilt axis), ``"rotmat"`` (a 3x3
                    rotation per view) or ``"resire"`` (the RESIRE kernels).
    ``vol_shape``   ``(Ny, Nz, Nx)``.
    ``N_det``       detector width.
    ``n_angles``    nominal view count (the default ``geom`` has it).
    ``angles_deg``  parallel only: the default tilt angles in degrees.

    A dimension of ``-1`` means "taken from the input at call time" (the
    functional API's transient operators; uncached adapter only).
    """
    kind: str
    vol_shape: tuple[int, int, int]
    N_det: int
    n_angles: int
    angles_deg: torch.Tensor | None = None


def _word_ptr(word: torch.Tensor | None) -> int:
    """Binding pointer of an optional non-finite report word (0 = none)."""
    return 0 if word is None else word.data_ptr()


class ProjectionOperator:
    """Forward/back projection for one geometry on one device.

    Subclasses implement the raw ``fp``/``bp`` (and ``grad_geom`` where the
    geometry is differentiable). ``project``/``backproject`` validate the
    input (device, shape; the dtype is locked on first use) and run the raw
    layer under autograd. Attributes: ``kind``, ``Ny``, ``Nz``, ``Nx``,
    ``N_det``, ``n_angles``, ``device``, ``dtype`` (``None`` until first
    use); properties ``vol_shape`` and ``sino_shape``.
    """

    kind: str = ""
    # run_sirt's CUDA-graph cache; only an operator that owns textures
    # (CudaCachedOperator) keeps graphs, since they capture its handles.
    _sirt_graph_cache: dict | None = None
    _sirt_graph_bad_keys: set | None = None

    def __init__(self, geometry: Geometry, device: torch.device | str,
                 dtype: torch.dtype | None = None, name: str | None = None) -> None:
        if geometry.kind not in _KINDS:
            raise ValueError(f"unknown projection geometry {geometry.kind!r}")
        self._name = name or type(self).__name__
        self.kind = geometry.kind
        # any negative size means "from the input" (the functional API's
        # N_det=-1 / Nz=-1 convention accepts every negative value)
        self.Ny, self.Nz, self.Nx, self.N_det, self.n_angles = (
            max(int(v), -1) for v in (*geometry.vol_shape, geometry.N_det, geometry.n_angles))
        dims = dict(Ny=self.Ny, Nz=self.Nz, Nx=self.Nx, N_det=self.N_det,
                    n_angles=self.n_angles)
        if 0 in dims.values():
            # -1 ("from the input") dims are not known yet: leave them out
            known = {k: v for k, v in dims.items() if v != -1}
            raise ValueError(
                f"{self._name}: every dimension must be at least 1; got a "
                f"zero-size dimension in {known}")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            # torch.device("cuda") -> "cuda:N": unambiguous device checks, and
            # the device any texture of this operator is bound to.
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.dtype: torch.dtype | None = dtype
        a = geometry.angles_deg
        self._angles_deg = None if a is None else a.detach()

    @property
    def vol_shape(self) -> tuple[int, int, int]:
        return (self.Ny, self.Nz, self.Nx)

    @property
    def sino_shape(self) -> tuple[int, int, int]:
        return (self.n_angles, self.Ny, self.N_det)

    # ── raw layer (no autograd) ──────────────────────────────────────────
    def fp(self, volume: torch.Tensor, geom: torch.Tensor | None = None) -> torch.Tensor:
        """Forward-project ``volume`` -> ``[n, Ny, N_det]``."""
        raise NotImplementedError

    def bp(self, sinogram: torch.Tensor, geom: torch.Tensor | None = None) -> torch.Tensor:
        """Back-project ``sinogram`` -> ``[Ny, Nz, Nx]``."""
        raise NotImplementedError

    def grad_geom(self, volume: torch.Tensor, grad_sino: torch.Tensor,
                  geom: torch.Tensor | None = None, token=None) -> torch.Tensor:
        """∂L/∂geom given ∂L/∂sino of ``fp(volume, geom)``: ``[n]`` (per
        degree) for parallel, ``[n, 9]`` for rotmat. ``token`` is what
        ``fp_token()`` returned right after that ``fp``."""
        raise NotImplementedError(f"{self._name} has no geometry gradient")

    def fp_token(self):
        """Opaque token identifying the state ``fp`` left behind (the
        cached rotmat adapter's volume-texture upload). ``None`` if the
        adapter keeps none."""
        return None

    def _nonfinite_word(self, x: torch.Tensor) -> torch.Tensor | None:
        """A zeroed int32 device word for ``fp(x, nonfinite=...)`` /
        ``bp(x, nonfinite=...)`` when this adapter's kernels report the
        non-finite values they store (the fp16 parallel CUDA kernels: 1 =
        an inf, 2 = a NaN), else None. The fp16 range check then reads the
        word instead of scanning the result (`_backend._check_fp16_range`).
        None while a CUDA graph is being captured (nothing is checked)."""
        return None

    def _parallel_fp16_word(self, x: torch.Tensor) -> torch.Tensor | None:
        if (self.kind != "parallel" or not x.is_cuda or x.dtype != torch.float16
                or torch.cuda.is_current_stream_capturing()):
            return None
        return torch.zeros(1, dtype=torch.int32, device=x.device)

    # ── differentiable layer ─────────────────────────────────────────────
    def project(self, volume: torch.Tensor, geom: torch.Tensor | None = None) -> torch.Tensor:
        """Differentiable ``fp`` (w.r.t. ``volume`` and ``geom``)."""
        self._check_device(volume, "volume")
        self._check_shape(volume, "volume", self.vol_shape)
        self._lock_dtype(volume)
        return _Project.apply(volume, geom, self)

    def backproject(self, sinogram: torch.Tensor, geom: torch.Tensor | None = None) -> torch.Tensor:
        """Differentiable ``bp`` (w.r.t. ``sinogram``)."""
        self._check_device(sinogram, "sinogram")
        self._check_shape(sinogram, "sinogram", (self._count(geom), self.Ny, self.N_det))
        self._lock_dtype(sinogram)
        return _BackProject.apply(sinogram, geom, self)

    # ── helpers ──────────────────────────────────────────────────────────
    def _count(self, geom: torch.Tensor | None) -> int:
        """View count of a call with ``geom``."""
        if geom is None:
            return self.n_angles
        return int(geom.numel()) if self.kind == "parallel" else int(geom.shape[0])

    def _angles(self, geom: torch.Tensor | None) -> torch.Tensor:
        if geom is not None:
            return geom
        if self._angles_deg is None:
            raise ValueError(f"{self._name}: no default angles; pass them per call")
        return self._angles_deg

    def _rotmat_geom(self, geom: torch.Tensor | None) -> torch.Tensor:
        if geom is None:
            raise ValueError(f"{self._name}: a {self.kind} projection needs its "
                             "rotation matrices on every call")
        return geom

    def _lock_dtype(self, t: torch.Tensor) -> int:
        """The io_dtype code of ``t``. The first call locks ``self.dtype``;
        a later call with another dtype raises."""
        code = _io_dtype_code(t)
        if self.dtype is None:
            self.dtype = t.dtype
        elif t.dtype != self.dtype:
            raise TypeError(
                f"{self._name} dtype lock-in mismatch: it was set up for "
                f"{self.dtype}, got {t.dtype}. Construct a new projector for "
                "a different dtype."
            )
        return code

    def _check_device(self, t: torch.Tensor, name: str) -> None:
        if t.device != self.device:
            raise ValueError(
                f"{self._name}: {name} tensor on {t.device} but "
                f"projector was built on {self.device}. Move the tensor with "
                f".to({self.device}) or rebuild the projector on the right "
                f"device."
            )

    def _check_shape(self, t: torch.Tensor, name: str, expected: tuple[int, ...]) -> None:
        actual = tuple(int(v) for v in t.shape)
        if len(actual) != len(expected) or any(
                e >= 0 and a != e for a, e in zip(actual, expected)):
            raise ValueError(f"{self._name}: {name} must have shape {expected}, "
                             f"got {actual}.")


class UncachedOperator(ProjectionOperator):
    """Every call through the raw kernel layer (``_launch``): the CPU
    adapter, and the one-shot CUDA path (a scoped texture per call).
    ``num_threads`` drives the CPU kernels."""

    def __init__(self, geometry: Geometry, device, dtype=None, name=None,
                 num_threads: int | None = None) -> None:
        if geometry.kind == "resire":
            raise ValueError("use ResireOperator for the RESIRE geometry")
        super().__init__(geometry, device, dtype, name)
        self.num_threads = num_threads
        # an explicit name (a public function) also heads the raw layer's
        # zero-size errors; otherwise they name the kernel
        self._label = name

    def fp(self, volume, geom=None, *, nonfinite=None):
        if self.kind == "parallel":
            return _forward_project_parallel_raw(volume, self._angles(geom), self.N_det,
                                                 self.num_threads, label=self._label,
                                                 nonfinite=nonfinite, texture=_TEXTURE_POOL)
        return _forward_project_rotmat_raw(volume, self._rotmat_geom(geom), self.N_det,
                                           self.num_threads, label=self._label,
                                           texture=_TEXTURE_POOL)

    def bp(self, sinogram, geom=None, *, nonfinite=None):
        if self.kind == "parallel":
            return _back_project_parallel_raw(sinogram, self._angles(geom), self.Nz,
                                              self.num_threads, Nx=self.Nx, label=self._label,
                                              nonfinite=nonfinite, texture=_TEXTURE_POOL)
        return _back_project_rotmat_raw(sinogram, self._rotmat_geom(geom), self.Nz,
                                        self.num_threads, Nx=self.Nx, label=self._label,
                                        texture=_TEXTURE_POOL)

    _nonfinite_word = ProjectionOperator._parallel_fp16_word

    def grad_geom(self, volume, grad_sino, geom=None, token=None):
        if self.kind == "parallel":
            return _grad_angle_raw(volume, grad_sino, self._angles(geom), self.num_threads)
        return _grad_rotmat_raw(volume, grad_sino, self._rotmat_geom(geom), self.num_threads)


class CudaCachedOperator(ProjectionOperator):
    """Persistent CUDA textures for one geometry, reused across calls.

    One volume ``TextureCache`` (layered for parallel, 3-D for rotmat) and
    one sinogram cache per view count, created on first use and released
    when the operator is collected (deferred while a CUDA graph is being
    captured). The I/O dtype locks on first use; parallel caches pack
    Y-slices into one texel (2 per float2 texel for fp32, 4 for fp16; see
    ``_channels``), exactly like the one-shot wrappers, so cached and
    uncached results are identical.

    Gradients w.r.t. the geometry: parallel falls back to the uncached
    ``grad_angle`` kernel; rotmat samples the volume texture its forward
    uploaded when that upload is still the latest (the ``fp_token`` epoch),
    and falls back to the uncached ``grad_rotmat`` kernel otherwise.

    Also owns ``run_sirt``'s CUDA-graph cache (the graphs capture these
    textures, so they must die with them).

    Direct texture writes (kernel audit F6): the caches are created with a
    surface, so run_sirt's fused kernels can write the updated volume and
    the residual straight into the textures. ``_mark_vol_written(volume)``
    records that the volume texture now holds ``volume``; the next ``fp`` of
    that same tensor (same storage, same autograd version counter, no upload
    since) skips its ``tc_upload``. ``_mark_sino_written(n)`` /
    ``bp_from_texture`` do the same for the sinogram texture; ``bp`` of a
    tensor always uploads. ``_forget_texture_writes()`` drops both records.
    ``upload_counts`` counts performed and skipped uploads (a debug hook for
    tests and measurements; CUDA-graph replays are not counted).

    Streams: every call uploads into the same texture arrays, so a call on
    one stream must not start before the calls already queued on another
    stream have finished reading them (it overwrote the array under them:
    two interleaved streams gave wrong projections in 16 of 20 trials). A
    call from a different stream than the previous one therefore makes the
    current stream wait for the previous stream (``wait_stream``, no host
    sync, only on a stream switch). Nothing is done while a CUDA graph is
    being captured (run_sirt synchronises the device around its capture).
    """

    def __init__(self, geometry: Geometry, device, dtype=None, name=None) -> None:
        if geometry.kind == "resire":
            raise ValueError("use ResireOperator for the RESIRE geometry")
        super().__init__(geometry, device, dtype, name)
        if self.device.type != "cuda":
            raise ValueError(f"{self._name} requires a CUDA device")
        if min(self.Ny, self.Nz, self.Nx, self.N_det, self.n_angles) < 1:
            raise ValueError(f"{self._name}: every dimension must be at least 1")
        if dtype is not None:
            _dtype_to_tc_code(dtype)
        self._cos = self._sin = None
        if self._angles_deg is not None:
            self._cos, self._sin = self._default_trig(self._angles_deg)
        self._vol_handle: int | None = None
        self._sino_handles: dict[int, int] = {}
        # Bumped by every volume upload, so a backward can tell whether the
        # volume texture still holds its forward's volume.
        self._vol_cache_epoch = 0
        # what a direct (surface) write left in the textures, see above
        self._vol_tex_token: tuple | None = None
        self._sino_tex_written: set[int] = set()
        self.upload_counts = {"vol": 0, "vol_skipped": 0, "sino": 0, "sino_skipped": 0}
        self._sirt_graph_cache = {}
        self._sirt_graph_bad_keys = set()
        # the stream of the latest call that used the textures
        self._last_stream: torch.cuda.Stream | None = None

    def _order_after_other_streams(self) -> None:
        """Make the current stream wait for the previous call's stream when
        they differ, so this call's upload cannot overwrite a texture that a
        kernel queued there is still reading (see the class docstring)."""
        cur = torch.cuda.current_stream(self.device)
        last = self._last_stream
        if last is None or last == cur:
            self._last_stream = cur
            return
        if torch.cuda.is_current_stream_capturing():
            return
        cur.wait_stream(last)
        self._last_stream = cur

    def _default_trig(self, angles_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _check_finite_geometry(angles_deg, "angles_deg")     # host angles only
        a_rad = angles_deg.to(device=self.device, dtype=torch.float32) * (math.pi / 180.0)
        return a_rad.cos().contiguous(), a_rad.sin().contiguous()

    def _set_default_angles(self, angles_deg: torch.Tensor) -> None:
        """Replace the default tilt angles (same count) in place: the cos/sin
        buffers keep their storage, so a CUDA graph captured with this
        operator reads the new angles on its next replay. Used by run_sirt
        when it reuses an internal operator (``algebraic._SirtOperatorCache``)."""
        a = angles_deg.detach()
        if self._cos is None or a.numel() != self.n_angles:
            raise ValueError(f"{self._name}: expected {self.n_angles} default angles, "
                             f"got {a.numel()}")
        self._angles_deg = a
        cos_a, sin_a = self._default_trig(a)
        with torch.cuda.device(self.device):
            self._cos.copy_(cos_a)
            self._sin.copy_(sin_a)

    def _channels(self) -> int:
        # Y-slice packing of the parallel textures (TorchRadon's
        # batch-channel-packing trick), the layout the one-shot wrappers use:
        # _core.parallel_channels (2 slices per float2 texel for fp32, 4 per
        # fp16 texel; texture_cache.h). Rotmat needs continuous Y
        # interpolation (tex3D) and stays single-channel.
        if self.kind == "parallel":
            return int(_core.parallel_channels(_dtype_to_tc_code(self.dtype), self.Ny))
        return 1

    def _vol_cache(self) -> int:
        if self._vol_handle is None:
            kind = _TC_VOL_LAYERED if self.kind == "parallel" else _TC_VOL_3D
            self._vol_handle = int(_core.tc_create(
                kind, _dtype_to_tc_code(self.dtype), self._channels(),
                self.Ny, self.Nz, self.Nx, surface=True))
        return self._vol_handle

    def _sino_cache(self, n: int) -> int:
        h = self._sino_handles.get(n)
        if h is None:
            h = self._sino_handles[n] = int(_core.tc_create(
                _TC_SINO_LAYERED, _dtype_to_tc_code(self.dtype), self._channels(),
                n, self.Ny, self.N_det, surface=True))
        return h

    # ── direct texture writes (run_sirt's fused kernels) ────────────────
    def _vol_write_target(self) -> int | None:
        """The volume texture handle if kernels can write it directly (it
        has a surface), else None. Needs the dtype locked. The caller writes
        on the current stream, ordered after other streams' reads."""
        if self.dtype is None:
            return None
        with torch.cuda.device(self.device):
            self._order_after_other_streams()
            h = self._vol_cache()
        return h if _core.tc_has_surface(h) else None

    def _sino_write_target(self, n: int) -> int | None:
        """The sinogram texture handle for ``n`` views if kernels can write
        it directly, else None. Needs the dtype locked. The caller writes on
        the current stream, ordered after other streams' reads."""
        if self.dtype is None:
            return None
        with torch.cuda.device(self.device):
            self._order_after_other_streams()
            h = self._sino_cache(n)
        return h if _core.tc_has_surface(h) else None

    def _mark_vol_written(self, volume: torch.Tensor) -> None:
        """A kernel just wrote ``volume``'s values into the volume texture."""
        self._vol_cache_epoch += 1
        self._vol_tex_token = (volume.data_ptr(), volume._version, self._vol_cache_epoch)

    def _mark_sino_written(self, n: int) -> None:
        """A kernel just wrote a sinogram of ``n`` views into that texture."""
        self._sino_tex_written.add(n)

    def _forget_texture_writes(self) -> None:
        """Drop the direct-write records (the next fp/bp uploads). The epoch
        moves on too: a CUDA graph may have rewritten the volume texture
        since the last recorded write."""
        if self._vol_tex_token is not None or self._sino_tex_written:
            self._vol_cache_epoch += 1
        self._vol_tex_token = None
        self._sino_tex_written.clear()

    def _vol_texture_holds(self, volume: torch.Tensor) -> bool:
        return self._vol_tex_token == (
            volume.data_ptr(), volume._version, self._vol_cache_epoch)

    def _upload_vol(self, volume: torch.Tensor) -> None:
        """Upload ``volume`` (contiguous, the operator's shape and dtype) into
        the volume texture and record it, as a skipped ``fp`` upload would
        need (run_sirt's CUDA graph: once per call instead of per iteration)."""
        self._lock_dtype(volume)
        self._check_input(volume, "volume", self.vol_shape)
        vol = volume.contiguous()
        with torch.cuda.device(self.device):
            self._order_after_other_streams()
            _core.tc_upload(self._vol_cache(), vol.data_ptr(), _stream_ptr(self.device))
        self.upload_counts["vol"] += 1
        self._mark_vol_written(vol)

    def _download_vol(self, out: torch.Tensor) -> None:
        """Copy the volume texture's contents into ``out`` (contiguous, the
        operator's shape and dtype): the volume a CUDA graph kept only in
        the texture (run_sirt)."""
        self._check_input(out, "volume", self.vol_shape)
        if out.dtype != self.dtype or not out.is_contiguous():
            raise ValueError(f"{self._name}: _download_vol needs a contiguous {self.dtype} tensor")
        with torch.cuda.device(self.device):
            self._order_after_other_streams()
            _core.tc_download(self._vol_cache(), out.data_ptr(), _stream_ptr(self.device))

    def _trig(self, geom):
        if geom is None and self._cos is not None:
            return self._cos, self._sin
        a = self._angles(geom).detach().reshape(-1)
        _check_finite_geometry(a, "angles_deg")               # host angles only
        a = a.to(device=self.device, dtype=torch.float32) * (math.pi / 180.0)
        return torch.cos(a).contiguous(), torch.sin(a).contiguous()

    def _mats(self, geom) -> torch.Tensor:
        mats = self._rotmat_geom(geom)
        if mats.device != self.device:
            raise ValueError(f"{self._name}: mats tensor on {mats.device} but "
                             f"projector was built on {self.device}.")
        return mats.detach().reshape(-1, 9).to(torch.float32).contiguous()

    def _check_input(self, t, name, shape):
        # The kernels read t through a raw pointer with the cache's dims.
        self._check_device(t, name)
        self._check_shape(t, name, shape)

    _nonfinite_word = ProjectionOperator._parallel_fp16_word

    def fp(self, volume, geom=None, *, nonfinite=None):
        io_dtype = self._lock_dtype(volume)
        self._check_input(volume, "volume", self.vol_shape)
        with torch.cuda.device(self.device):
            self._order_after_other_streams()
            h = self._vol_cache()
            vol = volume.contiguous()
            # the texture already holds `vol` (a direct write, no upload since)
            upload = not self._vol_texture_holds(vol)
            if self.kind == "parallel":
                cos_a, sin_a = self._trig(geom)
                n = cos_a.numel()
                sino = torch.empty(n, self.Ny, self.N_det, dtype=self.dtype, device=self.device)
                _core.forward_project_parallel_cuda_cached(
                    io_dtype, h, vol.data_ptr(), sino.data_ptr(),
                    cos_a.data_ptr(), sin_a.data_ptr(),
                    self.Ny, self.Nz, self.Nx, n, self.N_det, _stream_ptr(self.device),
                    upload=upload, nonfinite_ptr=_word_ptr(nonfinite))
            else:
                mats = self._mats(geom)
                n = mats.shape[0]
                sino = torch.empty(n, self.Ny, self.N_det, dtype=self.dtype, device=self.device)
                _core.forward_project_rotmat_cuda_cached(
                    io_dtype, h, vol.data_ptr(), sino.data_ptr(), mats.data_ptr(),
                    self.Ny, self.Nz, self.Nx, n, self.N_det, _stream_ptr(self.device),
                    upload=upload)
        if upload:
            self._vol_cache_epoch += 1
            self._vol_tex_token = None
            self.upload_counts["vol"] += 1
        else:
            # unchanged epoch: fp_token() still names the texture's contents
            self.upload_counts["vol_skipped"] += 1
        return sino

    def bp(self, sinogram, geom=None, *, nonfinite=None):
        io_dtype = self._lock_dtype(sinogram)
        return self._bp(io_dtype, sinogram.contiguous(), geom, nonfinite)

    def bp_from_texture(self, geom=None) -> torch.Tensor:
        """Back-project the sinogram a kernel wrote into the texture for
        this view count (``_mark_sino_written``), without an upload."""
        if self.dtype is None:
            raise RuntimeError(f"{self._name}: bp_from_texture before any upload")
        word = None
        if self.dtype == torch.float16:     # fp32: nothing is checked
            word = self._nonfinite_word(torch.empty(0, dtype=self.dtype, device=self.device))
        vol = self._bp(0 if self.dtype == torch.float32 else 1, None, geom, word)
        # the fp16 overflow check `backproject` runs (inside an fp16
        # run_sirt: a device-side flag); the input sinogram is the texture,
        # so a zero stands in for it (an inf is then never blamed on the
        # input, which the fused residual kernel produced finite). fp32 has
        # no check: no zero tensor (one fill kernel per SIRT iteration).
        if vol.dtype == torch.float16:
            _check_fp16_range(vol, vol.new_zeros(()), self._name, nonfinite=word)
        return vol

    def _bp(self, io_dtype, sino, geom, nonfinite=None):
        """``bp`` of ``sino``, or (``sino`` None) of the texture's contents."""
        with torch.cuda.device(self.device):
            self._order_after_other_streams()
            vol = torch.empty(self.Ny, self.Nz, self.Nx, dtype=self.dtype, device=self.device)
            if self.kind == "parallel":
                cos_a, sin_a = self._trig(geom)
                n = cos_a.numel()
            else:
                mats = self._mats(geom)
                n = mats.shape[0]
            if sino is None:
                if n not in self._sino_tex_written:
                    raise RuntimeError(f"{self._name}: no sinogram of {n} views was "
                                       "written into the texture")
                self.upload_counts["sino_skipped"] += 1
                sino_ptr, upload = 0, False
            else:
                self._check_input(sino, "sinogram", (n, self.Ny, self.N_det))
                self._sino_tex_written.discard(n)
                self.upload_counts["sino"] += 1
                sino_ptr, upload = sino.data_ptr(), True
            if self.kind == "parallel":
                _core.back_project_parallel_cuda_cached(
                    io_dtype, self._sino_cache(n), sino_ptr, vol.data_ptr(),
                    cos_a.data_ptr(), sin_a.data_ptr(),
                    self.Ny, self.Nz, self.Nx, n, self.N_det, _stream_ptr(self.device),
                    upload=upload, nonfinite_ptr=_word_ptr(nonfinite))
            else:
                _core.back_project_rotmat_cuda_cached(
                    io_dtype, self._sino_cache(n), sino_ptr, vol.data_ptr(),
                    mats.data_ptr(),
                    self.Ny, self.Nz, self.Nx, n, self.N_det, _stream_ptr(self.device),
                    upload=upload)
        return vol

    def fp_token(self):
        return self._vol_cache_epoch

    def grad_geom(self, volume, grad_sino, geom=None, token=None):
        if self.kind == "parallel":
            return _grad_angle_raw(volume, grad_sino, self._angles(geom))
        if token is not None and token == self._vol_cache_epoch:
            return self._grad_mats_cached(volume, grad_sino, geom)
        # A later forward overwrote the volume texture (several forwards
        # before one backward): the uncached kernel uploads `volume` itself.
        return _grad_rotmat_raw(volume, grad_sino, self._rotmat_geom(geom))

    def _grad_mats_cached(self, volume, grad_sino, geom) -> torch.Tensor:
        """∂L/∂M from the volume texture the latest ``fp`` uploaded (point-
        filter view), saving the upload ``_grad_rotmat_raw`` would do. The
        caller guarantees that upload was ``volume`` (the ``fp_token``)."""
        assert self._vol_handle is not None, "requires a prior fp()"
        io_dtype = _io_dtype_code(volume)
        mats = self._mats(geom)
        n = mats.shape[0]
        Ny, Nz, Nx = volume.shape
        N_det = grad_sino.shape[2]
        grad_mats = torch.zeros(n, 9, dtype=torch.float32, device=volume.device)
        # the gradient kernel accumulates in fp32
        gs = grad_sino.contiguous().float()
        with torch.cuda.device(self.device):
            self._order_after_other_streams()
            _core.grad_rotmat_cuda_cached(
                io_dtype, self._vol_handle,
                gs.data_ptr(), mats.data_ptr(), grad_mats.data_ptr(),
                Ny, Nz, Nx, n, N_det, _stream_ptr(self.device))
        return grad_mats

    def __del__(self) -> None:
        # _core.tc_destroy synchronizes the device before freeing, so an FP/BP
        # still queued on any stream finishes reading the texture first. GC may
        # fire on another current device, so the release pins ours.
        try:
            handles = [h for h in (getattr(self, "_vol_handle", None),
                                   *getattr(self, "_sino_handles", {}).values())
                       if h is not None]
            self._vol_handle = None
            self._sino_handles = {}
            if handles:
                _release_handles(self.device, handles)
        except Exception:
            pass


class ResireOperator(ProjectionOperator):
    """The RESIRE voxel-driven projectors (CUDA, fp32; frozen numerics).

    ``geom`` is the ZYZ Euler table ``[n, 3]`` in degrees. ``fp``/``bp``
    call the frozen raw wrappers and ``project``/``backproject`` the RESIRE
    autograd Functions, with exactly the arguments ``forward_project_resire``
    / ``back_project_resire`` pass. No geometry gradient (``refine_angles``
    searches instead)."""

    def fp(self, volume, geom=None):
        return _forward_project_resire_raw(volume, self._rotmat_geom(geom), self.N_det)

    def bp(self, sinogram, geom=None):
        return _back_project_resire_raw(sinogram, self._rotmat_geom(geom), self.Nz, Nx=self.Nx)

    def project(self, volume, geom=None):
        return _FPResire.apply(volume, self._rotmat_geom(geom), self.N_det)

    def backproject(self, sinogram, geom=None):
        return _BPResire.apply(sinogram, self._rotmat_geom(geom), self.Nz, self.Nx)


class _UserProjectorOperator(ProjectionOperator):
    """A projector object passed by the caller (``projector=``): anything
    with ``forward(volume[, mats])`` and ``back(sinogram[, mats])``, such as
    ``ParallelProjector`` / ``RotmatProjector`` (which validate, then call
    their own operator) or a duck-typed stand-in. Its ``run_sirt`` graph
    cache is the projector's, if it has one."""

    kind = "user"

    def __init__(self, projector) -> None:     # no geometry of its own
        self._projector = projector
        self._name = type(projector).__name__

    @property
    def _sirt_graph_cache(self):
        return getattr(self._projector, "_sirt_graph_cache", None)

    @property
    def _sirt_graph_bad_keys(self):
        return getattr(self._projector, "_sirt_graph_bad_keys", None)

    def project(self, volume, geom=None):
        p = self._projector
        return p.forward(volume) if geom is None else p.forward(volume, geom)

    def backproject(self, sinogram, geom=None):
        p = self._projector
        return p.back(sinogram) if geom is None else p.back(sinogram, geom)

    fp = project
    bp = backproject


def _make_operator(
    geometry: Geometry,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    projector: object | None = None,
    *,
    cached: bool | None = None,
    num_threads: int | None = None,
    name: str | None = None,
) -> ProjectionOperator:
    """The operator for ``geometry`` on ``device``: the one place that picks
    an adapter.

    * ``projector`` given (the algorithms' ``projector=``): wrap it
      (:class:`_UserProjectorOperator`), whatever ``geometry`` says.
    * ``geometry.kind == "resire"``: :class:`ResireOperator`.
    * CUDA and ``cached`` not ``False``: :class:`CudaCachedOperator`.
      One-shot callers pass ``cached=False`` (the functional API, FBP) and
      get the scoped-texture path, which frees its texture right away.
    * otherwise (CPU, or CUDA uncached): :class:`UncachedOperator`.

    ``dtype`` pre-locks the I/O dtype (else the first call does);
    ``num_threads`` drives the CPU kernels; ``name`` is used in error
    messages (the owning projector's class name).
    """
    if projector is not None:
        return _UserProjectorOperator(projector)
    device = torch.device(device)
    if geometry.kind == "resire":
        return ResireOperator(geometry, device, dtype, name)
    if device.type == "cuda" and cached is not False:
        return CudaCachedOperator(geometry, device, dtype, name)
    return UncachedOperator(geometry, device, dtype, name, num_threads=num_threads)


# ── texture release (deferred while a CUDA graph is being captured) ─────────
#
# Handles of operators collected while this thread was capturing a CUDA
# graph. Releasing then is illegal: in the default (global) capture mode
# the device sync and cudaFreeArray are prohibited for the whole thread,
# whatever device the capture runs on, and would invalidate the capture.
# They wait here until _drain_deferred_handles() runs after the capture
# (run_sirt calls it) or the next operator is released.
# Limitation: only a capture on the *calling* thread is detected. If GC
# finalizes an operator on thread B while thread A captures in global mode,
# the free still breaks A's capture; keep projectors alive across captures
# in threaded code.
_deferred_handles: list[tuple[torch.device, int]] = []


def _capturing(device: torch.device) -> bool:
    """True if this thread is capturing a CUDA graph on the current device
    or on ``device`` (the operator's)."""
    if torch.cuda.is_current_stream_capturing():
        return True
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _destroy(device: torch.device, handle: int) -> None:
    try:
        with torch.cuda.device(device):
            _core.tc_destroy(handle)
    except Exception:
        pass   # best effort: one failing handle must not strand the rest


def _drain_deferred_handles() -> None:
    """Release the handles deferred during a CUDA-graph capture. No-op while
    a capture is still running on the current device."""
    if not _deferred_handles or torch.cuda.is_current_stream_capturing():
        return
    while _deferred_handles:
        _destroy(*_deferred_handles.pop())


def _release_handles(device: torch.device, handles: list[int]) -> None:
    if _capturing(device):
        _deferred_handles.extend((device, h) for h in handles)
        return
    for h in handles:
        _destroy(device, h)
    _drain_deferred_handles()


# ── texture pool of the one-shot CUDA path (functional API, FBP) ────────────

class _TexturePool:
    """Texture caches of the one-shot CUDA path, kept between calls.

    ``forward_project`` / ``back_project`` / ``*_3d`` and ``run_fbp`` on CUDA
    used to build a texture per call (``cudaMalloc3DArray`` + texture object),
    run the kernel, then synchronise the stream and free it: +0.15 ms per
    call at 128-256^3 and +1.5 ms at 512^3 fp32, and the host blocked until
    the kernel had finished, so a Python loop of calls could not queue work.
    Now `_launch` leases a cache from this pool, runs the cached kernel
    (upload + launch, stream-ordered, no sync) and returns the cache.

    * Keyed by (device index, texture kind, I/O dtype, channels, dims): a
      cache is only reused for the same shape and layout.
    * At most ``maxsize`` (4) caches are kept, least recently used evicted;
      memory retained is up to 4 textures, e.g. a 512^3 fp32 volume texture
      is 512 MiB (outside PyTorch's allocator, not counted by
      ``torch.cuda.memory_allocated``). :func:`release_cached_projectors`
      (fasttomo.algorithms) frees them; so does an allocation failure of a
      new texture (the pool is emptied and the allocation retried once).
    * Thread-safe: a lease takes the cache out of the pool (``checkout``)
      until the call has queued its work, so concurrent calls never share
      one (a second one is created instead).
    * Streams: the lease records the stream it ran on. A later call on
      another stream makes its stream wait for that one first
      (``wait_stream``, no host sync), so its upload cannot overwrite the
      texture while the earlier kernel still reads it (the ordering
      ``CudaCachedOperator`` applies since PR #95). Destroying a cache
      synchronises the device first.
    * CUDA-graph capture: not used (a captured graph would keep reading a
      cache the pool may evict later); a call during capture takes the
      one-shot binding as before.
    """

    def __init__(self, maxsize: int = 4) -> None:
        self.maxsize = maxsize
        # serial -> (key, handle, stream of the last lease); oldest first
        self._free: OrderedDict[int, tuple] = OrderedDict()
        self._serial = itertools.count()
        self._lock = threading.Lock()
        self.created = 0       # debug counter (tests): textures created

    def launch(self, op: str, spec: tuple, ptrs: list, device: torch.device,
               kwargs: dict) -> None:
        """Run ``_core.<op>_cuda_cached`` for the one-shot binding's
        arguments ``ptrs`` (io_dtype first) on a leased cache of layout
        ``spec`` = (kind, io_dtype, channels, dim_a, dim_b, dim_c)."""
        stream_ptr = _stream_ptr(device)
        if torch.cuda.is_current_stream_capturing():
            getattr(_core, op + "_cuda")(*ptrs, stream_ptr, **kwargs)
            return
        key = (device.index, *spec)
        handle, last = self._checkout(key, device)
        cur = torch.cuda.current_stream(device)
        try:
            if last is not None and last != cur:
                cur.wait_stream(last)
            getattr(_core, op + "_cuda_cached")(ptrs[0], handle, *ptrs[1:], stream_ptr,
                                                upload=True, **kwargs)
        finally:
            self._checkin(key, handle, cur)

    def _checkout(self, key: tuple, device: torch.device) -> tuple[int, object]:
        with self._lock:
            for serial, (k, handle, last) in self._free.items():
                if k == key:
                    del self._free[serial]
                    return handle, last
        kind, dtype, channels, a, b, c = key[1:]
        try:
            handle = int(_core.tc_create(kind, dtype, channels, a, b, c))
        except RuntimeError:
            # most likely out of memory: free what the pool holds, retry once
            if not self.clear(device):
                raise
            handle = int(_core.tc_create(kind, dtype, channels, a, b, c))
        self.created += 1
        return handle, None

    def _checkin(self, key: tuple, handle: int, stream) -> None:
        with self._lock:
            self._free[next(self._serial)] = (key, handle, stream)
            evicted = []
            while len(self._free) > self.maxsize:
                evicted.append(self._free.popitem(last=False)[1])
        for k, h, _ in evicted:
            _release_handles(torch.device("cuda", k[0]), [h])

    def clear(self, device: torch.device | None = None) -> int:
        """Destroy the pooled caches (of ``device`` only, if given); returns
        how many. Caches leased by a running call are not affected."""
        with self._lock:
            keep, drop = OrderedDict(), []
            for serial, entry in self._free.items():
                if device is None or entry[0][0] == device.index:
                    drop.append(entry)
                else:
                    keep[serial] = entry
            self._free = keep
        for k, h, _ in drop:
            _release_handles(torch.device("cuda", k[0]), [h])
        return len(drop)

    def __len__(self) -> int:
        return len(self._free)


_TEXTURE_POOL = _TexturePool()


def _release_texture_pool() -> int:
    """Free the one-shot path's pooled textures (release_cached_projectors)."""
    if not _TEXTURE_POOL:
        return 0
    return _TEXTURE_POOL.clear()
