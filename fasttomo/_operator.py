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
                         is the one-shot path (a scoped texture per call).
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

import math
from dataclasses import dataclass

import torch

import fasttomo._core as _core
from fasttomo._backend import (
    _TC_SINO_LAYERED,
    _TC_VOL_3D,
    _TC_VOL_LAYERED,
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

    def fp(self, volume, geom=None):
        if self.kind == "parallel":
            return _forward_project_parallel_raw(volume, self._angles(geom), self.N_det,
                                                 self.num_threads, label=self._label)
        return _forward_project_rotmat_raw(volume, self._rotmat_geom(geom), self.N_det,
                                           self.num_threads, label=self._label)

    def bp(self, sinogram, geom=None):
        if self.kind == "parallel":
            return _back_project_parallel_raw(sinogram, self._angles(geom), self.Nz,
                                              self.num_threads, Nx=self.Nx, label=self._label)
        return _back_project_rotmat_raw(sinogram, self._rotmat_geom(geom), self.Nz,
                                        self.num_threads, Nx=self.Nx, label=self._label)

    def grad_geom(self, volume, grad_sino, geom=None, token=None):
        if self.kind == "parallel":
            return _grad_angle_raw(volume, grad_sino, self._angles(geom), self.num_threads)
        return _grad_rotmat_raw(volume, grad_sino, self._rotmat_geom(geom), self.num_threads)


class CudaCachedOperator(ProjectionOperator):
    """Persistent CUDA textures for one geometry, reused across calls.

    One volume ``TextureCache`` (layered for parallel, 3-D for rotmat) and
    one sinogram cache per view count, created on first use and released
    when the operator is collected (deferred while a CUDA graph is being
    captured). The I/O dtype locks on first use; fp16 parallel caches pack
    4 Y-slices per texel when ``Ny % 4 == 0``, exactly like the one-shot
    wrappers, so cached and uncached results are identical.

    Gradients w.r.t. the geometry: parallel falls back to the uncached
    ``grad_angle`` kernel; rotmat samples the volume texture its forward
    uploaded when that upload is still the latest (the ``fp_token`` epoch),
    and falls back to the uncached ``grad_rotmat`` kernel otherwise.

    Also owns ``run_sirt``'s CUDA-graph cache (the graphs capture these
    textures, so they must die with them).

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
            a_rad = (self._angles_deg.to(device=self.device, dtype=torch.float32)
                     * (math.pi / 180.0))
            self._cos = a_rad.cos().contiguous()
            self._sin = a_rad.sin().contiguous()
        self._vol_handle: int | None = None
        self._sino_handles: dict[int, int] = {}
        # Bumped by every volume upload, so a backward can tell whether the
        # volume texture still holds its forward's volume.
        self._vol_cache_epoch = 0
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

    def _channels(self) -> int:
        # 4-channel packed-Y mode for fp16 parallel caches when Ny % 4 == 0
        # (TorchRadon's batch-channel-packing trick); rotmat needs continuous
        # Y interpolation (tex3D) and stays single-channel.
        if self.kind == "parallel" and self.dtype == torch.float16 and self.Ny % 4 == 0:
            return 4
        return 1

    def _vol_cache(self) -> int:
        if self._vol_handle is None:
            kind = _TC_VOL_LAYERED if self.kind == "parallel" else _TC_VOL_3D
            self._vol_handle = int(_core.tc_create(
                kind, _dtype_to_tc_code(self.dtype), self._channels(),
                self.Ny, self.Nz, self.Nx))
        return self._vol_handle

    def _sino_cache(self, n: int) -> int:
        h = self._sino_handles.get(n)
        if h is None:
            h = self._sino_handles[n] = int(_core.tc_create(
                _TC_SINO_LAYERED, _dtype_to_tc_code(self.dtype), self._channels(),
                n, self.Ny, self.N_det))
        return h

    def _trig(self, geom):
        if geom is None and self._cos is not None:
            return self._cos, self._sin
        a = self._angles(geom)
        a = a.detach().reshape(-1).to(device=self.device, dtype=torch.float32) * (math.pi / 180.0)
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

    def fp(self, volume, geom=None):
        io_dtype = self._lock_dtype(volume)
        self._check_input(volume, "volume", self.vol_shape)
        with torch.cuda.device(self.device):
            self._order_after_other_streams()
            h = self._vol_cache()
            vol = volume.contiguous()
            if self.kind == "parallel":
                cos_a, sin_a = self._trig(geom)
                n = cos_a.numel()
                sino = torch.empty(n, self.Ny, self.N_det, dtype=self.dtype, device=self.device)
                _core.forward_project_parallel_cuda_cached(
                    io_dtype, h, vol.data_ptr(), sino.data_ptr(),
                    cos_a.data_ptr(), sin_a.data_ptr(),
                    self.Ny, self.Nz, self.Nx, n, self.N_det, _stream_ptr(self.device))
            else:
                mats = self._mats(geom)
                n = mats.shape[0]
                sino = torch.empty(n, self.Ny, self.N_det, dtype=self.dtype, device=self.device)
                _core.forward_project_rotmat_cuda_cached(
                    io_dtype, h, vol.data_ptr(), sino.data_ptr(), mats.data_ptr(),
                    self.Ny, self.Nz, self.Nx, n, self.N_det, _stream_ptr(self.device))
        self._vol_cache_epoch += 1
        return sino

    def bp(self, sinogram, geom=None):
        io_dtype = self._lock_dtype(sinogram)
        with torch.cuda.device(self.device):
            self._order_after_other_streams()
            sino = sinogram.contiguous()
            vol = torch.empty(self.Ny, self.Nz, self.Nx, dtype=self.dtype, device=self.device)
            if self.kind == "parallel":
                cos_a, sin_a = self._trig(geom)
                n = cos_a.numel()
                self._check_input(sinogram, "sinogram", (n, self.Ny, self.N_det))
                _core.back_project_parallel_cuda_cached(
                    io_dtype, self._sino_cache(n), sino.data_ptr(), vol.data_ptr(),
                    cos_a.data_ptr(), sin_a.data_ptr(),
                    self.Ny, self.Nz, self.Nx, n, self.N_det, _stream_ptr(self.device))
            else:
                mats = self._mats(geom)
                n = mats.shape[0]
                self._check_input(sinogram, "sinogram", (n, self.Ny, self.N_det))
                _core.back_project_rotmat_cuda_cached(
                    io_dtype, self._sino_cache(n), sino.data_ptr(), vol.data_ptr(),
                    mats.data_ptr(),
                    self.Ny, self.Nz, self.Nx, n, self.N_det, _stream_ptr(self.device))
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
