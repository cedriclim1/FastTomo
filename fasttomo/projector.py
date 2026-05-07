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
                       sinogram, and matrices (the ∂/∂M kernel doesn't use a
                       texture, so no caching needed there).

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
        mats = _euler_zyz_to_rotmat(euler).reshape(n_angles, 9).contiguous()
        pred = proj.forward(vol, mats)
        ...

The functional API is unchanged; you can mix cached and uncached calls freely.
"""

from __future__ import annotations

import math

import torch

import fasttomo._core as _core
from fasttomo.autograd import _FP2DCached, _BP2DCached, _FP3DCached, _BP3DCached


# Texture-cache kind enum mirrors texture_cache.h:
_TC_VOL_LAYERED  = 0
_TC_SINO_LAYERED = 1
_TC_VOL_3D       = 2


class ParallelProjector:
    """Cached parallel-beam projector for fixed-angle tight loops.

    The angle tensor is stored at construction; cos/sin are precomputed once
    on ``device`` and reused on every call. Two ``TextureCache`` handles (vol
    + sino) are allocated lazily on first ``forward()`` / ``back()`` and held
    until the projector is destroyed.

    The functional ``forward_project`` / ``back_project`` API stays for
    one-shot calls and for differentiable-angle optimisation. Use this class
    when angles are fixed and FP/BP run many times on the same volume shape.
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
        self.n_angles = int(a_rad.numel())

        self.Ny    = Ny
        self.Nz    = Nz
        self.Nx    = Nx
        self.N_det = int(N_det) if N_det is not None else Nx
        self.device = device

        self._vol_handle:  int | None = None
        self._sino_handle: int | None = None

    def _ensure_vol_cache(self) -> None:
        if self._vol_handle is None:
            self._vol_handle = int(_core.tc_create(_TC_VOL_LAYERED, self.Ny, self.Nz, self.Nx))

    def _ensure_sino_cache(self) -> None:
        if self._sino_handle is None:
            self._sino_handle = int(
                _core.tc_create(_TC_SINO_LAYERED, self.n_angles, self.Ny, self.N_det)
            )

    # ── raw (non-autograd) entry points used by the autograd Function ──
    def _fp_raw(self, volume: torch.Tensor) -> torch.Tensor:
        self._ensure_vol_cache()
        vol = volume.contiguous()
        sino = torch.empty(
            self.n_angles, self.Ny, self.N_det,
            dtype=torch.float32, device=self.device,
        )
        _core.forward_project_cuda_cached(
            self._vol_handle,
            vol.data_ptr(), sino.data_ptr(),
            self._cos.data_ptr(), self._sin.data_ptr(),
            self.Ny, self.Nz, self.Nx, self.n_angles, self.N_det,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        return sino

    def _bp_raw(self, sinogram: torch.Tensor) -> torch.Tensor:
        self._ensure_sino_cache()
        sino = sinogram.contiguous()
        vol = torch.empty(
            self.Ny, self.Nz, self.Nx,
            dtype=torch.float32, device=self.device,
        )
        _core.back_project_cuda_cached(
            self._sino_handle,
            sino.data_ptr(), vol.data_ptr(),
            self._cos.data_ptr(), self._sin.data_ptr(),
            self.Ny, self.Nz, self.Nx, self.n_angles, self.N_det,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        return vol

    # ── public differentiable API ──
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return _FP2DCached.apply(volume, self)

    def back(self, sinogram: torch.Tensor) -> torch.Tensor:
        return _BP2DCached.apply(sinogram, self)

    def __del__(self) -> None:
        # Tear down explicitly. tc_destroy is safe-on-null and frees both the
        # cudaArray and the textureObject.
        try:
            if getattr(self, "_vol_handle", None) is not None:
                _core.tc_destroy(self._vol_handle)
                self._vol_handle = None
            if getattr(self, "_sino_handle", None) is not None:
                _core.tc_destroy(self._sino_handle)
                self._sino_handle = None
        except Exception:
            pass


class RotmatProjector:
    """Cached 3D rotation-matrix projector for tight loops.

    Unlike ``ParallelProjector`` the rotation matrices are passed at every
    call (since pose optimisation makes them learnable). Only the texture
    caches are persistent; the matrix tensor is consumed via its ``.data_ptr()``
    on each FP/BP. Differentiable w.r.t. ``volume``, ``sinogram``, and
    ``matrices`` (the ∂/∂M kernel doesn't use a texture so it stays uncached).
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
        self.Ny       = Ny
        self.Nz       = Nz
        self.Nx       = Nx
        self.N_det    = int(N_det) if N_det is not None else Nx
        self.n_angles = int(n_angles)
        self.device   = device

        self._vol_handle:  int | None = None
        self._sino_handle: int | None = None

    def _ensure_vol_cache(self) -> None:
        if self._vol_handle is None:
            self._vol_handle = int(_core.tc_create(_TC_VOL_3D, self.Ny, self.Nz, self.Nx))

    def _ensure_sino_cache(self) -> None:
        if self._sino_handle is None:
            self._sino_handle = int(
                _core.tc_create(_TC_SINO_LAYERED, self.n_angles, self.Ny, self.N_det)
            )

    def _fp_raw(self, volume: torch.Tensor, mats_flat: torch.Tensor) -> torch.Tensor:
        self._ensure_vol_cache()
        vol = volume.contiguous()
        mats = mats_flat.detach().contiguous()
        sino = torch.empty(
            self.n_angles, self.Ny, self.N_det,
            dtype=torch.float32, device=self.device,
        )
        _core.forward_project_rotmat_cuda_cached(
            self._vol_handle,
            vol.data_ptr(), sino.data_ptr(), mats.data_ptr(),
            self.Ny, self.Nz, self.Nx, self.n_angles, self.N_det,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        return sino

    def _bp_raw(self, sinogram: torch.Tensor, mats_flat: torch.Tensor) -> torch.Tensor:
        self._ensure_sino_cache()
        sino = sinogram.contiguous()
        mats = mats_flat.detach().contiguous()
        vol = torch.empty(
            self.Ny, self.Nz, self.Nx,
            dtype=torch.float32, device=self.device,
        )
        _core.back_project_rotmat_cuda_cached(
            self._sino_handle,
            sino.data_ptr(), vol.data_ptr(), mats.data_ptr(),
            self.Ny, self.Nz, self.Nx, self.n_angles, self.N_det,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        return vol

    def forward(self, volume: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
        # accept either [n_angles, 3, 3] (default API) or [n_angles, 9]
        mats_flat = mats.reshape(self.n_angles, 9) if mats.ndim == 3 else mats
        return _FP3DCached.apply(volume, mats_flat, self)

    def back(self, sinogram: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
        mats_flat = mats.reshape(self.n_angles, 9) if mats.ndim == 3 else mats
        return _BP3DCached.apply(sinogram, mats_flat, self)

    def __del__(self) -> None:
        try:
            if getattr(self, "_vol_handle", None) is not None:
                _core.tc_destroy(self._vol_handle)
                self._vol_handle = None
            if getattr(self, "_sino_handle", None) is not None:
                _core.tc_destroy(self._sino_handle)
                self._sino_handle = None
        except Exception:
            pass
