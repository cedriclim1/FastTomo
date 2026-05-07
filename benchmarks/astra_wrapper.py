"""ASTRA parallel-beam wrapper, sliced over Y to match FastTomo's
``[n_angles, Ny, N_det]`` convention.

ASTRA's 2D parallel-beam FBP runs per Y-slice; the wrapper hides the loop and
the ASTRA resource cleanup. Inputs and outputs are numpy float32; conversion
to/from torch lives in the caller.
"""

from __future__ import annotations

from typing import Iterator

import astra
import numpy as np


class AstraParallelBeam:
    """Per-slice 2D parallel-beam projector / FBP backend.

    The instance owns a ``vol_geom`` and ``proj_geom`` sized to the sinogram's
    detector width. Slices are reconstructed at ``[Nz=N_det, Nx=N_det]`` to
    match FastTomo's default square reconstruction.
    """

    def __init__(self, n_det: int, angles_rad: np.ndarray):
        self.n_det = int(n_det)
        self.angles_rad = np.asarray(angles_rad, dtype=np.float32)
        self.vol_geom = astra.create_vol_geom(n_det, n_det)  # (Nz rows, Nx cols)
        self.proj_geom = astra.create_proj_geom(
            "parallel", 1.0, n_det, self.angles_rad
        )

    # --- internal helpers ----------------------------------------------------

    def _alloc_slice(self, sino_2d: np.ndarray) -> tuple[int, int]:
        sino_id = astra.data2d.create("-sino", self.proj_geom, sino_2d)
        rec_id = astra.data2d.create("-vol", self.vol_geom)
        return sino_id, rec_id

    def _free(self, *ids: int) -> None:
        for i in ids:
            astra.data2d.delete(i)

    # --- public API ----------------------------------------------------------

    def fbp(self, sinogram: np.ndarray, positivity: bool = False) -> np.ndarray:
        """Filtered back-projection of a 3D sinogram.

        sinogram: [n_angles, Ny, N_det] float32
        returns:  [Ny, Nz=N_det, Nx=N_det] float32
        """
        n_angles, Ny, N_det = sinogram.shape
        if N_det != self.n_det:
            raise ValueError(f"detector size {N_det} != wrapper {self.n_det}")
        rec = np.empty((Ny, N_det, N_det), dtype=np.float32)
        for y in range(Ny):
            sino_id, rec_id = self._alloc_slice(sinogram[:, y, :])
            cfg = astra.astra_dict("FBP_CUDA")
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = rec_id
            alg_id = astra.algorithm.create(cfg)
            try:
                astra.algorithm.run(alg_id)
                rec[y] = astra.data2d.get(rec_id)
            finally:
                astra.algorithm.delete(alg_id)
                self._free(sino_id, rec_id)
        if positivity:
            np.clip(rec, 0.0, None, out=rec)
        return rec

    def sirt(
        self,
        sinogram: np.ndarray,
        num_iter: int = 100,
        positivity: bool = True,
    ) -> np.ndarray:
        """ASTRA SIRT_CUDA per Y-slice.

        ``positivity=True`` sets ASTRA's MinConstraint=0.0 inside SIRT, so
        every iteration projects onto the non-negative orthant.
        """
        n_angles, Ny, N_det = sinogram.shape
        if N_det != self.n_det:
            raise ValueError(f"detector size {N_det} != wrapper {self.n_det}")
        rec = np.empty((Ny, N_det, N_det), dtype=np.float32)
        for y in range(Ny):
            sino_id, rec_id = self._alloc_slice(sinogram[:, y, :])
            cfg = astra.astra_dict("SIRT_CUDA")
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = rec_id
            if positivity:
                cfg["option"] = {"MinConstraint": 0.0}
            alg_id = astra.algorithm.create(cfg)
            try:
                astra.algorithm.run(alg_id, num_iter)
                rec[y] = astra.data2d.get(rec_id)
            finally:
                astra.algorithm.delete(alg_id)
                self._free(sino_id, rec_id)
        return rec
