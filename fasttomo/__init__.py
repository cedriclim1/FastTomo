# Import torch first so its bundled libcudart is loaded into the process before
# fasttomo._core is imported. The CUDA wheel is intentionally built without
# bundling libcudart so the symbol resolves to whatever PyTorch already loaded
# — this is what lets one wheel work across PyTorch's CUDA versions.
import torch  # noqa: F401

import sys as _sys
import warnings as _warnings
from pathlib import Path as _Path
from packaging.version import Version as _Version
from fasttomo import _core as _core_mod

# Hard requirement: PyTorch 2.11 is the only release built against CUDA 13.0,
# which is the toolkit FastTomo's CUDA kernels target. Older torch will
# either fail to load _core or load an incompatible libcudart.
_torch_ver = _Version(torch.__version__.split("+")[0])
if _torch_ver < _Version("2.11"):
    raise ImportError(
        f"fasttomo requires torch>=2.11 (found {torch.__version__}). "
        "Run `pip install --upgrade 'torch>=2.11'`."
    )

if (not getattr(_core_mod, "cuda_available", False)) and torch.cuda.is_available():
    _warnings.warn(
        "fasttomo was built without CUDA support but torch.cuda.is_available() is True. "
        "CPU kernels will be used; passing CUDA tensors will raise RuntimeError. "
        "Rebuild with `pip install -e . --config-settings=cmake.define.FASTTOMO_USE_CUDA=ON` "
        "after activating the conda env that ships nvcc.",
        RuntimeWarning,
        stacklevel=2,
    )

# When the binary is a CUDA build, torch must also report CUDA 13.0 — any
# mismatch usually crashes at first kernel launch.
if getattr(_core_mod, "cuda_available", False) and torch.version.cuda is not None:
    _parts = torch.version.cuda.split(".")
    _cu_major = int(_parts[0])
    _cu_minor = int(_parts[1]) if len(_parts) > 1 else 0
    if (_cu_major, _cu_minor) != (13, 0):
        _warnings.warn(
            f"fasttomo._core was built against CUDA 13.0 but torch reports "
            f"CUDA {torch.version.cuda}. PyTorch 2.11 only ships against "
            "CUDA 13.0; mixed runtimes can crash at first kernel launch.",
            RuntimeWarning,
            stacklevel=2,
        )
    del _parts, _cu_major, _cu_minor

del _warnings, _Version, _torch_ver, _core_mod

def _repair_stale_editable_algorithms_mapping() -> None:
    """Patch stale scikit-build editable maps from pre-ARCH-P8 installs.

    Before the split, ``fasttomo.algorithms`` was a module file
    ``fasttomo/algorithms.py``. After the split it is a package at
    ``fasttomo/algorithms/__init__.py``. A user who updates the source tree
    without reinstalling the editable package can keep an old redirect entry
    that points at the deleted file, making ``import fasttomo`` fail with
    FileNotFoundError.
    """
    pkg_dir = _Path(__file__).resolve().parent
    old_algorithms_py = str(pkg_dir / "algorithms.py")
    new_algorithms_init = str(pkg_dir / "algorithms" / "__init__.py")
    new_algorithms_pkg = str(pkg_dir / "algorithms")
    if not _Path(new_algorithms_init).exists():
        return
    for finder in _sys.meta_path:
        if finder.__class__.__name__ != "ScikitBuildRedirectingFinder":
            continue
        mapping = getattr(finder, "known_source_files", None)
        if not isinstance(mapping, dict):
            continue
        if mapping.get("fasttomo.algorithms") != old_algorithms_py:
            continue
        mapping["fasttomo.algorithms"] = new_algorithms_init
        locations = getattr(finder, "submodule_search_locations", None)
        if isinstance(locations, dict):
            locations["fasttomo.algorithms"] = [new_algorithms_pkg]

_repair_stale_editable_algorithms_mapping()

from fasttomo.projection import (
    forward_project,
    back_project,
    forward_project_3d,
    back_project_3d,
    forward_project_resire,
    back_project_resire,
    euler_zyz_to_rotmat,
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
    run_sart,
    run_wsirt,
    run_genfire,
    run_resire,
    run_fista,
    run_fista_tv,
    run_grad_descent,
    run_ad_recon,
    run_mbir,
    refine_angles,
)
from fasttomo.projector import ParallelProjector, RotmatProjector
from fasttomo.regularizers import tilt_ssim_loss, s3im_loss, logsum_sparsity, tv_loss_xz
from fasttomo.configs import (
    AdReconConfig,
    ConstraintConfig,
    MbirConfig,
    OptimConfig,
    PoseRefinementConfig,
    RegularizerConfig,
)

__all__ = [
    "forward_project",
    "back_project",
    "forward_project_3d",
    "back_project_3d",
    "forward_project_resire",
    "back_project_resire",
    "euler_zyz_to_rotmat",
    "apply_shifts",
    "apply_shifts_2d",
    "estimate_shifts_xcorr",
    "ramp_filter",
    "cylindrical_mask",
    "tv_loss_3d",
    "run_fbp",
    "run_weighted_backprojection",
    "run_sirt",
    "run_sart",
    "run_wsirt",
    "run_genfire",
    "run_resire",
    "run_fista",
    "run_fista_tv",
    "run_grad_descent",
    "run_ad_recon",
    "run_mbir",
    "refine_angles",
    "ParallelProjector",
    "RotmatProjector",
    "tilt_ssim_loss",
    "s3im_loss",
    "logsum_sparsity",
    "tv_loss_xz",
    "AdReconConfig",
    "ConstraintConfig",
    "MbirConfig",
    "OptimConfig",
    "PoseRefinementConfig",
    "RegularizerConfig",
]
