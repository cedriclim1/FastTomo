"""
Cached projector classes for tight inner loops.

The functional API (``forward_project``, ``back_project``, ``forward_project_3d``,
``back_project_3d``) allocates a fresh ``cudaArray`` and ``cudaTextureObject``
on every call. For one-shot users that's negligible; for an N-iteration AD or
SIRT loop it's 2N alloc/free pairs on identical-shaped arrays. The projector
classes here own a persistent ``TextureCache`` per geometry and reuse it across
all calls, leaving only ``cudaMemcpy3DAsync`` per FP/BP — closing the per-call
kernel-overhead gap with TorchRadon.

Two flavours, mirroring the two functional FP/BP pairs:

  ParallelProjector  — single-axis tilt, scalar angles fixed at construction.
                       Differentiable w.r.t. volume / sinogram only. Ideal for
                       AD reconstruction without pose optimisation, FBP, SIRT.
  RotmatProjector    — full ZYZ rotation, ``[n_angles, 3, 3]`` rotation matrix
                       passed each call. Differentiable w.r.t. volume,
                       sinogram, and matrices (the ∂/∂M kernel samples the
                       volume texture, and reuses the one the forward
                       uploaded).

Both classes are CUDA-only — they wrap the ``*_cuda_cached`` entry points.
On CPU you should use the functional API.

Usage (AD loop, no pose):

    proj = ParallelProjector((Ny, Nz, Nx), angles_deg, device=vol.device)
    for _ in range(n_iter):
        pred = proj.forward(vol)
        loss = ((pred - sino) ** 2).mean()
        loss.backward()
        ...

Usage (AD loop, with pose):

    proj = RotmatProjector((Ny, Nz, Nx), N_det=Nx, device=vol.device,
                           n_angles=n_angles)
    for _ in range(n_iter):
        mats = euler_zyz_to_rotmat(euler).reshape(n_angles, 9).contiguous()
        pred = proj.forward(vol, mats)
        ...

The functional API is unchanged; you can mix cached and uncached calls freely.
"""

from __future__ import annotations

import math

import torch

import fasttomo._core as _core
from fasttomo.autograd import (
    _FPParallelCached, _BPParallelCached, _FPRotmatCached, _BPRotmatCached, _io_dtype_code,
    _on_device,
)


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


class _ProjectorBase:
    """Private base class holding state and logic shared by ParallelProjector
    and RotmatProjector.

    Subclasses must call ``super().__init__(...)`` with the shared fields
    listed below and then initialise their own projector-specific state.
    """

    def __init__(
        self,
        Ny: int,
        Nz: int,
        Nx: int,
        N_det: int,
        n_angles: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        dims = dict(Ny=Ny, Nz=Nz, Nx=Nx, N_det=N_det, n_angles=n_angles)
        if min(dims.values()) < 1:
            raise ValueError(
                f"{type(self).__name__}: every dimension must be at least 1, "
                f"got {dims}")
        self.Ny       = Ny
        self.Nz       = Nz
        self.Nx       = Nx
        self.N_det    = N_det
        self.n_angles = n_angles
        # Normalise the device so torch.device("cuda") becomes torch.device("cuda:N")
        # — needed for unambiguous device comparison in _check_device. The cudaArray
        # this projector will hold is bound to whichever index resolves now, so
        # capturing it explicitly is also more correct than carrying an unindexed
        # alias around.
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device   = device
        self.dtype: torch.dtype | None = dtype
        self._vol_handle:  int | None = None
        self._sino_handle: int | None = None

    def _lock_dtype(self, t: torch.Tensor) -> int:
        """Return the io_dtype int. Locks self.dtype on first call; raises
        on mismatch afterwards.  The error message uses ``type(self).__name__``
        so subclass-specific names appear in the message."""
        code = _io_dtype_code(t)
        if self.dtype is None:
            self.dtype = t.dtype
        elif t.dtype != self.dtype:
            raise TypeError(
                f"{type(self).__name__} dtype lock-in mismatch: cache was "
                f"allocated for {self.dtype}, got {t.dtype}. Construct a "
                f"new projector for a different dtype."
            )
        return code

    def _check_device(self, t: torch.Tensor, name: str) -> None:
        """Raise ValueError if ``t`` does not live on ``self.device``.

        Catches the device-mismatch case early with a clear message instead of
        letting a downstream kernel hit ``cudaErrorIllegalAddress``. Cached
        textures are tied to the projector's construction device; mixing
        devices on the same projector is never legal.
        """
        if t.device != self.device:
            raise ValueError(
                f"{type(self).__name__}: {name} tensor on {t.device} but "
                f"projector was built on {self.device}. Move the tensor with "
                f".to({self.device}) or rebuild the projector on the right "
                f"device."
            )

    def _check_shape(
        self,
        t: torch.Tensor,
        name: str,
        expected: tuple[int, ...],
    ) -> None:
        actual = tuple(int(v) for v in t.shape)
        if actual != expected:
            raise ValueError(
                f"{type(self).__name__}: {name} must have shape {expected}, "
                f"got {actual}."
            )

    def _check_volume_shape(self, volume: torch.Tensor) -> None:
        self._check_shape(volume, "volume", (self.Ny, self.Nz, self.Nx))

    def _check_sinogram_shape(self, sinogram: torch.Tensor) -> None:
        self._check_shape(sinogram, "sinogram", (self.n_angles, self.Ny, self.N_det))

    def __del__(self) -> None:
        # Tear down explicitly. ``_core.tc_destroy`` synchronizes the device
        # before freeing, so an FP/BP still queued on any stream (the last
        # call of run_sirt, say) finishes reading the texture first. GC may
        # fire while a different CUDA device is current, so pin the
        # projector's construction device.
        try:
            handles = [h for h in (getattr(self, "_vol_handle", None),
                                   getattr(self, "_sino_handle", None))
                       if h is not None]
            self._vol_handle = None
            self._sino_handle = None
            if handles:
                _release_handles(self.device, handles)
        except Exception:
            pass


# Handles of projectors collected while this thread was capturing a CUDA
# graph. Releasing then is illegal: in the default (global) capture mode
# the device sync and cudaFreeArray are prohibited for the whole thread,
# whatever device the capture runs on, and would invalidate the capture.
# They wait here until _drain_deferred_handles() runs after the capture
# (run_sirt calls it) or the next projector is released.
# Limitation (as on main, where cudaFreeArray had the same effect): only a
# capture on the *calling* thread is detected. If GC finalizes a projector
# on thread B while thread A captures in global mode, the free still breaks
# A's capture; keep projectors alive across captures in threaded code.
_deferred_handles: list[tuple[torch.device, int]] = []


def _capturing(device: torch.device) -> bool:
    """True if this thread is capturing a CUDA graph on the current device
    or on ``device`` (the projector's)."""
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


class ParallelProjector(_ProjectorBase):
    """Cached parallel-beam projector for fixed-angle tight loops.

    The angle tensor is stored at construction; cos/sin are precomputed once
    on ``device`` and reused on every call. Two ``TextureCache`` handles (vol
    + sino) are allocated lazily on first ``forward()`` / ``back()`` and held
    until the projector is destroyed.

    The functional ``forward_project`` / ``back_project`` API stays for
    one-shot calls and for differentiable-angle optimisation. Use this class
    when angles are fixed and FP/BP run many times on the same volume shape.

    ``_sirt_graph_cache``: lazily-initialised ``dict`` keyed by reconstruction
    configuration tuples; holds captured ``torch.cuda.CUDAGraph`` objects used
    by ``run_sirt`` for constant-time replay of the SIRT iteration body.
    """

    def __init__(
        self,
        vol_shape: tuple[int, int, int],   # (Ny, Nz, Nx)
        angles_deg: torch.Tensor,
        device: torch.device | str | None = None,
        N_det: int | None = None,
    ):
        Ny, Nz, Nx = (int(v) for v in vol_shape)
        if device is None:
            device = angles_deg.device if angles_deg.is_cuda else torch.device("cuda")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("ParallelProjector requires a CUDA device")

        a_rad = (
            angles_deg.detach().to(device=device, dtype=torch.float32)
            * (math.pi / 180.0)
        )
        self._cos = a_rad.cos().contiguous()
        self._sin = a_rad.sin().contiguous()
        n_angles = int(a_rad.numel())

        # I/O dtype is locked on the first FP/BP call. Subsequent calls with
        # a mismatched dtype raise — matches the existing shape-mismatch
        # contract in the *_cuda_cached entry points.
        super().__init__(
            Ny=Ny,
            Nz=Nz,
            Nx=Nx,
            N_det=int(N_det) if N_det is not None else Nx,
            n_angles=n_angles,
            device=device,
        )

    def _channels(self) -> int:
        """4-channel packed-Y mode is auto-enabled for fp16 caches when
        Ny % 4 == 0 (TorchRadon's batch-channel-packing trick). Otherwise
        falls back to single-channel. fp32 caches stay 1-channel."""
        if self.dtype == torch.float16 and self.Ny % 4 == 0:
            return 4
        return 1

    def _ensure_vol_cache(self) -> None:
        if self._vol_handle is None:
            assert self.dtype is not None, "_lock_dtype must be called first"
            with torch.cuda.device(self.device):
                self._vol_handle = int(_core.tc_create(
                    _TC_VOL_LAYERED, _dtype_to_tc_code(self.dtype), self._channels(),
                    self.Ny, self.Nz, self.Nx,
                ))

    def _ensure_sino_cache(self) -> None:
        if self._sino_handle is None:
            assert self.dtype is not None, "_lock_dtype must be called first"
            with torch.cuda.device(self.device):
                self._sino_handle = int(_core.tc_create(
                    _TC_SINO_LAYERED, _dtype_to_tc_code(self.dtype), self._channels(),
                    self.n_angles, self.Ny, self.N_det,
                ))

    # ── raw (non-autograd) entry points used by the autograd Function ──
    def _fp_raw(self, volume: torch.Tensor) -> torch.Tensor:
        io_dtype = self._lock_dtype(volume)
        with torch.cuda.device(self.device):
            self._ensure_vol_cache()
            vol = volume.contiguous()
            sino = torch.empty(
                self.n_angles, self.Ny, self.N_det,
                dtype=self.dtype, device=self.device,
            )
            _core.forward_project_parallel_cuda_cached(
                io_dtype,
                self._vol_handle,
                vol.data_ptr(), sino.data_ptr(),
                self._cos.data_ptr(), self._sin.data_ptr(),
                self.Ny, self.Nz, self.Nx, self.n_angles, self.N_det,
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        return sino

    def _bp_raw(self, sinogram: torch.Tensor) -> torch.Tensor:
        io_dtype = self._lock_dtype(sinogram)
        with torch.cuda.device(self.device):
            self._ensure_sino_cache()
            sino = sinogram.contiguous()
            vol = torch.empty(
                self.Ny, self.Nz, self.Nx,
                dtype=self.dtype, device=self.device,
            )
            _core.back_project_parallel_cuda_cached(
                io_dtype,
                self._sino_handle,
                sino.data_ptr(), vol.data_ptr(),
                self._cos.data_ptr(), self._sin.data_ptr(),
                self.Ny, self.Nz, self.Nx, self.n_angles, self.N_det,
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        return vol

    # ── public differentiable API ──
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        self._check_device(volume, "volume")
        self._check_volume_shape(volume)
        return _FPParallelCached.apply(volume, self)

    def back(self, sinogram: torch.Tensor) -> torch.Tensor:
        self._check_device(sinogram, "sinogram")
        self._check_sinogram_shape(sinogram)
        return _BPParallelCached.apply(sinogram, self)


class RotmatProjector(_ProjectorBase):
    """Cached 3D rotation-matrix projector for tight loops.

    Unlike ``ParallelProjector`` the rotation matrices are passed at every
    call (since pose optimisation makes them learnable). Only the texture
    caches are persistent; the matrix tensor is consumed via its ``.data_ptr()``
    on each FP/BP. Differentiable w.r.t. ``volume``, ``sinogram``, and
    ``matrices``: the ∂/∂M kernel reads the volume through the point-filter
    view of the forward's texture cache, falling back to a one-shot texture
    if the cache has been overwritten by a later forward.

    ``_sirt_graph_cache``: lazily-initialised ``dict`` keyed by reconstruction
    configuration tuples; holds captured ``torch.cuda.CUDAGraph`` objects used
    by ``run_sirt`` for constant-time replay of the SIRT iteration body.
    """

    def __init__(
        self,
        vol_shape: tuple[int, int, int],   # (Ny, Nz, Nx)
        n_angles: int,
        device: torch.device | str,
        N_det: int | None = None,
    ):
        Ny, Nz, Nx = (int(v) for v in vol_shape)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("RotmatProjector requires a CUDA device")

        # Locked on first FP/BP call (see ParallelProjector for the rationale).
        super().__init__(
            Ny=Ny,
            Nz=Nz,
            Nx=Nx,
            N_det=int(N_det) if N_det is not None else Nx,
            n_angles=int(n_angles),
            device=device,
        )
        # Monotonic token for the current contents of _vol_handle.
        # Every _fp_raw upload increments this so autograd can detect when a
        # saved forward volume has been overwritten by a later forward call.
        self._vol_cache_epoch = 0

    def _ensure_vol_cache(self) -> None:
        # Rotmat kernels need continuous Y interpolation (tex3D), so they
        # can't pack 4 Y-slices into channels — channels stays 1.
        if self._vol_handle is None:
            assert self.dtype is not None, "_lock_dtype must be called first"
            with torch.cuda.device(self.device):
                self._vol_handle = int(_core.tc_create(
                    _TC_VOL_3D, _dtype_to_tc_code(self.dtype), 1,
                    self.Ny, self.Nz, self.Nx,
                ))

    def _ensure_sino_cache(self) -> None:
        if self._sino_handle is None:
            assert self.dtype is not None, "_lock_dtype must be called first"
            with torch.cuda.device(self.device):
                self._sino_handle = int(_core.tc_create(
                    _TC_SINO_LAYERED, _dtype_to_tc_code(self.dtype), 1,
                    self.n_angles, self.Ny, self.N_det,
                ))

    def _fp_raw(self, volume: torch.Tensor, mats_flat: torch.Tensor) -> torch.Tensor:
        io_dtype = self._lock_dtype(volume)
        with torch.cuda.device(self.device):
            self._ensure_vol_cache()
            vol = volume.contiguous()
            mats = mats_flat.detach().to(torch.float32).contiguous()
            sino = torch.empty(
                self.n_angles, self.Ny, self.N_det,
                dtype=self.dtype, device=self.device,
            )
            _core.forward_project_rotmat_cuda_cached(
                io_dtype,
                self._vol_handle,
                vol.data_ptr(), sino.data_ptr(), mats.data_ptr(),
                self.Ny, self.Nz, self.Nx, self.n_angles, self.N_det,
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        self._vol_cache_epoch += 1
        return sino

    def _bp_raw(self, sinogram: torch.Tensor, mats_flat: torch.Tensor) -> torch.Tensor:
        io_dtype = self._lock_dtype(sinogram)
        with torch.cuda.device(self.device):
            self._ensure_sino_cache()
            sino = sinogram.contiguous()
            mats = mats_flat.detach().to(torch.float32).contiguous()
            vol = torch.empty(
                self.Ny, self.Nz, self.Nx,
                dtype=self.dtype, device=self.device,
            )
            _core.back_project_rotmat_cuda_cached(
                io_dtype,
                self._sino_handle,
                sino.data_ptr(), vol.data_ptr(), mats.data_ptr(),
                self.Ny, self.Nz, self.Nx, self.n_angles, self.N_det,
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        return vol

    def _grad_mats_raw(
        self,
        volume: torch.Tensor,
        grad_sino: torch.Tensor,
        mats_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Cached ∂L/∂M kernel. Requires _vol_handle to be already populated by a
        prior forward(). Saves the volume upload that _grad_rotmat_raw would do.

        Invariant: forward() populated _vol_handle this iteration; PyTorch autograd
        guarantees `volume` (saved via save_for_backward) hasn't been mutated;
        therefore the cache's content matches `volume`."""
        assert self._vol_handle is not None, (
            "_grad_mats_raw requires forward() to have been called this iteration"
        )
        io_dtype = _io_dtype_code(volume)
        n_angles = mats_flat.shape[0]
        Ny, Nz, Nx = volume.shape
        _, _, N_det = grad_sino.shape
        grad_mats = torch.zeros(n_angles, 9, dtype=torch.float32, device=volume.device)
        # grad_sino comes from the BP path's autograd; force fp32 to match the kernel's
        # current accumulator. (Future fp16 grad_rotmat would template this.)
        gs = grad_sino.contiguous().float()
        # The kernel reads mats as a dense float32 [n_angles, 9] buffer.
        mats = mats_flat.detach().to(torch.float32).contiguous()
        with torch.cuda.device(self.device):
            _core.grad_rotmat_cuda_cached(
                io_dtype,
                self._vol_handle,
                gs.data_ptr(), mats.data_ptr(), grad_mats.data_ptr(),
                Ny, Nz, Nx, n_angles, N_det,
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        return grad_mats

    def _normalize_and_validate_mats(self, mats: torch.Tensor) -> torch.Tensor:
        # The rotmat kernels read mats as a dense float32 buffer through a raw
        # pointer, so fp16/fp64 or strided inputs would be misread. Cast and
        # densify here (both are differentiable, so pose gradients still flow
        # back to the caller's dtype).
        if not mats.is_floating_point():
            raise TypeError(
                f"{type(self).__name__}: mats must be a floating tensor, "
                f"got {mats.dtype}."
            )
        if mats.ndim == 3:
            self._check_shape(mats, "mats", (self.n_angles, 3, 3))
            return mats.reshape(self.n_angles, 9).to(torch.float32).contiguous()
        if mats.ndim == 2:
            self._check_shape(mats, "mats", (self.n_angles, 9))
            return mats.to(torch.float32).contiguous()
        raise ValueError(
            f"{type(self).__name__}: mats must have shape "
            f"({self.n_angles}, 3, 3) or ({self.n_angles}, 9), "
            f"got {tuple(int(v) for v in mats.shape)}."
        )

    def forward(self, volume: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
        self._check_device(volume, "volume")
        self._check_volume_shape(volume)
        self._check_device(mats, "mats")
        mats_flat = self._normalize_and_validate_mats(mats)
        return _FPRotmatCached.apply(volume, mats_flat, self)

    def back(self, sinogram: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
        self._check_device(sinogram, "sinogram")
        self._check_sinogram_shape(sinogram)
        self._check_device(mats, "mats")
        mats_flat = self._normalize_and_validate_mats(mats)
        return _BPRotmatCached.apply(sinogram, mats_flat, self)
