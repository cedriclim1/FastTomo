"""
Backward-compatibility shim for the legacy ``fasttomo.torch_ops`` module.

The public API was split across ``projection.py``, ``utilities.py``, and
``algorithms.py``. Keep this module so existing user code importing
``fasttomo.torch_ops`` continues to work.
"""

from __future__ import annotations

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
    "run_fbp",
    "run_weighted_backprojection",
    "run_sirt",
    "run_wsirt",
    "run_genfire",
    "run_resire",
    "run_grad_descent",
    "refine_angles",
]
