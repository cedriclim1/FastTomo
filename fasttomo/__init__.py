# Import torch first so its bundled libcudart is loaded into the process before
# fasttomo._core is imported. The CUDA wheel is intentionally built without
# bundling libcudart so the symbol resolves to whatever PyTorch already loaded
# — this is what lets one wheel work across PyTorch's CUDA versions.
import torch  # noqa: F401

import warnings as _warnings
from fasttomo import _core as _core_mod
if (not getattr(_core_mod, "cuda_available", False)) and torch.cuda.is_available():
    _warnings.warn(
        "fasttomo was built without CUDA support but torch.cuda.is_available() is True. "
        "CPU kernels will be used; passing CUDA tensors will raise RuntimeError. "
        "Rebuild with `pip install -e . --config-settings=cmake.define.FASTTOMO_USE_CUDA=ON` "
        "after activating the conda env that ships nvcc.",
        RuntimeWarning,
        stacklevel=2,
    )
del _warnings, _core_mod

from fasttomo.projection import (
    forward_project,
    back_project,
    forward_project_3d,
    back_project_3d,
    forward_project_resire,
    back_project_resire,
)
from fasttomo.utilities import (
    apply_shifts,
    apply_shifts_2d,
    estimate_shifts_xcorr,
    ramp_filter,
    cylindrical_mask,
    tv_loss_3d,
)
from fasttomo.algorithms import (
    run_fbp,
    run_weighted_backprojection,
    run_sirt,
    run_wsirt,
    run_genfire,
    run_resire,
    run_grad_descent,
    refine_angles,
)
from fasttomo.projector import ParallelProjector, RotmatProjector

__all__ = [
    "forward_project",
    "back_project",
    "forward_project_3d",
    "back_project_3d",
    "forward_project_resire",
    "back_project_resire",
    "apply_shifts",
    "apply_shifts_2d",
    "estimate_shifts_xcorr",
    "ramp_filter",
    "cylindrical_mask",
    "tv_loss_3d",
    "run_fbp",
    "run_weighted_backprojection",
    "run_sirt",
    "run_wsirt",
    "run_genfire",
    "run_resire",
    "run_grad_descent",
    "refine_angles",
    "ParallelProjector",
    "RotmatProjector",
]
