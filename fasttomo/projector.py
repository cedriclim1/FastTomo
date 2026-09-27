"""
Projector classes for tight inner loops.

The functional API (``forward_project``, ``back_project``, ``forward_project_3d``,
``back_project_3d``) builds a one-shot operator per call: on CUDA a fresh
``cudaArray`` and ``cudaTextureObject`` every time. For one-shot users that's
negligible; for an N-iteration AD or SIRT loop it's 2N alloc/free pairs on
identical-shaped arrays. A projector owns one projection operator
(`fasttomo._operator`) for its geometry: on CUDA a persistent
``TextureCache`` per geometry, reused across all calls, leaving only
``cudaMemcpy3DAsync`` per FP/BP (closing the per-call kernel-overhead gap
with TorchRadon); on CPU the OpenMP kernels.

Two flavours, mirroring the two functional FP/BP pairs:

  ParallelProjector  — single-axis tilt, scalar angles fixed at construction.
                       Differentiable w.r.t. volume / sinogram only. Ideal for
                       AD reconstruction without pose optimisation, FBP, SIRT.
  RotmatProjector    — full ZYZ rotation, ``[n_angles, 3, 3]`` rotation matrix
                       passed each call. Differentiable w.r.t. volume,
                       sinogram, and matrices (on CUDA the ∂/∂M kernel samples
                       the volume texture the forward uploaded).

Both work on CUDA (cached textures) and, since the operator seam, on CPU
(``device="cpu"``; no cache to keep, same results as the functional API).

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

The functional API is unchanged; you can mix projector and functional calls
freely.
"""

from __future__ import annotations

import torch

from fasttomo._backend import (  # noqa: F401  (re-exported for old import paths)
    _TC_DTYPE_FP16,
    _TC_DTYPE_FP32,
    _TC_SINO_LAYERED,
    _TC_VOL_3D,
    _TC_VOL_LAYERED,
    _dtype_to_tc_code,
    _io_dtype_code,
    _stream_ptr,
)
from fasttomo._operator import (  # noqa: F401  (texture release moved to _operator)
    Geometry,
    _capturing,
    _deferred_handles,
    _destroy,
    _drain_deferred_handles,
    _make_operator,
    _release_handles,
)


class _ProjectorBase:
    """Shared front of ParallelProjector and RotmatProjector: owns the
    projection operator (``self._op``) and exposes its geometry as
    ``Ny``, ``Nz``, ``Nx``, ``N_det``, ``n_angles``, ``device``, ``dtype``.
    Validation (device, shape, dtype lock-in) happens in the operator."""

    def __init__(self, geometry: Geometry, device: torch.device | str,
                 num_threads: int | None) -> None:
        Ny, Nz, Nx = geometry.vol_shape
        dims = dict(Ny=Ny, Nz=Nz, Nx=Nx, N_det=geometry.N_det, n_angles=geometry.n_angles)
        if min(dims.values()) < 1:
            raise ValueError(
                f"{type(self).__name__}: every dimension must be at least 1, "
                f"got {dims}")
        self._op = _make_operator(geometry, device, num_threads=num_threads,
                                  name=type(self).__name__)
        op = self._op
        self.Ny, self.Nz, self.Nx = op.vol_shape
        self.N_det = op.N_det
        self.n_angles = op.n_angles
        self.device = op.device

    @property
    def dtype(self) -> torch.dtype | None:
        """The I/O dtype, locked by the first FP/BP call (``None`` before)."""
        return self._op.dtype

    # run_sirt's CUDA-graph cache for this projector: configuration key ->
    # captured graph and its static buffers, and the keys whose capture
    # failed. Owned by the operator so the graphs die with its textures
    # (``None`` on CPU, where run_sirt never captures).
    @property
    def _sirt_graph_cache(self) -> dict | None:
        return self._op._sirt_graph_cache

    @property
    def _sirt_graph_bad_keys(self) -> set | None:
        return self._op._sirt_graph_bad_keys

    def _check_device(self, t: torch.Tensor, name: str) -> None:
        """Raise ValueError if ``t`` does not live on ``self.device``."""
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


def _default_device(angles_deg: torch.Tensor) -> torch.device:
    return angles_deg.device if angles_deg.is_cuda else torch.device("cuda")


class ParallelProjector(_ProjectorBase):
    """Parallel-beam projector for fixed-angle tight loops.

    The angle tensor is stored at construction; on CUDA cos/sin are
    precomputed once on ``device`` and two ``TextureCache`` handles (vol +
    sino) are allocated lazily on first ``forward()`` / ``back()`` and held
    until the projector is destroyed. ``device`` defaults to the angles'
    CUDA device, else ``"cuda"``; ``device="cpu"`` runs the CPU kernels
    (``num_threads`` as in the functional API).

    The functional ``forward_project`` / ``back_project`` API stays for
    one-shot calls and for differentiable-angle optimisation. Use this class
    when angles are fixed and FP/BP run many times on the same volume shape.

    ``_sirt_graph_cache`` holds the CUDA graphs ``run_sirt`` captures with
    this projector, keyed by reconstruction configuration, for
    constant-time replay of the SIRT iteration body.
    """

    def __init__(
        self,
        vol_shape: tuple[int, int, int],   # (Ny, Nz, Nx)
        angles_deg: torch.Tensor,
        device: torch.device | str | None = None,
        N_det: int | None = None,
        *,
        num_threads: int | None = None,
    ):
        Ny, Nz, Nx = (int(v) for v in vol_shape)
        device = torch.device(device) if device is not None else _default_device(angles_deg)
        geometry = Geometry(
            "parallel", (Ny, Nz, Nx), int(N_det) if N_det is not None else Nx,
            int(angles_deg.numel()), angles_deg)
        super().__init__(geometry, device, num_threads)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Forward-project ``volume [Ny, Nz, Nx]`` to ``[n_angles, Ny, N_det]``
        (differentiable w.r.t. ``volume``). The first call locks the dtype."""
        return self._op.project(volume)

    def back(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Back-project ``sinogram [n_angles, Ny, N_det]`` to ``[Ny, Nz, Nx]``
        (differentiable w.r.t. ``sinogram``)."""
        return self._op.backproject(sinogram)


class RotmatProjector(_ProjectorBase):
    """3D rotation-matrix projector for tight loops.

    Unlike ``ParallelProjector`` the rotation matrices are passed at every
    call (since pose optimisation makes them learnable). On CUDA only the
    texture caches are persistent; the matrix tensor is consumed via its
    ``.data_ptr()`` on each FP/BP. Differentiable w.r.t. ``volume``,
    ``sinogram``, and ``matrices``: on CUDA the ∂/∂M kernel reads the volume
    through the point-filter view of the forward's texture cache, falling
    back to a one-shot texture if the cache has been overwritten by a later
    forward. ``device="cpu"`` runs the CPU kernels.

    ``run_sirt`` never captures a CUDA graph with this projector (rotation
    matrices are a per-call input), so its ``_sirt_graph_cache`` stays empty.
    """

    def __init__(
        self,
        vol_shape: tuple[int, int, int],   # (Ny, Nz, Nx)
        n_angles: int,
        device: torch.device | str,
        N_det: int | None = None,
        *,
        num_threads: int | None = None,
    ):
        Ny, Nz, Nx = (int(v) for v in vol_shape)
        geometry = Geometry(
            "rotmat", (Ny, Nz, Nx), int(N_det) if N_det is not None else Nx, int(n_angles))
        super().__init__(geometry, device, num_threads)

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
        """Forward-project ``volume [Ny, Nz, Nx]`` with rotation matrices
        ``mats [n_angles, 3, 3]`` (or ``[n_angles, 9]``, e.g. from
        :func:`fasttomo.euler_zyz_to_rotmat`). Differentiable w.r.t.
        ``volume`` and ``mats``."""
        self._check_device(mats, "mats")
        return self._op.project(volume, self._normalize_and_validate_mats(mats))

    def back(self, sinogram: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
        """Back-project ``sinogram [n_angles, Ny, N_det]`` with ``mats`` to
        ``[Ny, Nz, Nx]`` (differentiable w.r.t. ``sinogram``)."""
        self._check_device(mats, "mats")
        return self._op.backproject(sinogram, self._normalize_and_validate_mats(mats))
