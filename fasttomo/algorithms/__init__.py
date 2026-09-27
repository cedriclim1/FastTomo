"""
Reconstruction algorithms for FastTomo.

Conventions
===========

Every ``run_*`` function follows the conventions below; the function
docstrings only describe what is specific to each algorithm and refer here
for the rest.

Shapes and units
    Volume ``[Ny, Nz, Nx]``, sinogram ``[n_angles, Ny, N_det]``, angles a
    1-D tensor in degrees. ``Ny`` runs along the tilt axis. ``Nz`` (keyword,
    ``-1``) defaults to ``N_det``; ``Nx = N_det``.

Call signature
    ``run_x(sinogram, angles_deg, n_iter, *, ...)``: the sinogram, the
    angles and, for iterative algorithms, ``n_iter`` are positional; every
    other argument is keyword-only, so a call written for an older
    positional order raises TypeError instead of binding to another
    parameter. Exceptions: the direct methods (``run_fbp``,
    ``run_weighted_backprojection``) have no ``n_iter``, and ``run_genfire``
    takes ``n_iter`` as a keyword for one release (its third positional
    used to be ``Nz``). Integers (``n_iter``, ``Nz``) may be Python, numpy
    or 0-d tensor integers. Renamed keywords keep working for one release
    with a DeprecationWarning (see CHANGELOG.md).

Returns
    ``volume`` (``[Ny, Nz, Nx]``, the sinogram's dtype and device), or
    ``(volume, loss_history)`` with ``return_loss=True``: a float32 CPU
    tensor with one entry per iteration run (each algorithm's docstring
    says which loss it records). Algorithm-specific extras are appended
    after these: ``(..., euler, shifts)`` with ``run_ad_recon``'s
    ``return_pose=True``, ``(..., d_offsets, sigma)`` with ``run_mbir``'s
    ``return_state=True``.

dtypes
    fp32 everywhere; fp16 also for FBP, SIRT, SART, FISTA and FISTA-TV
    (output in the input dtype). Anything else (fp64, bf16) raises
    TypeError; nothing is cast silently. A ``warm_start`` must have the
    sinogram's dtype.

Support: ``mask`` and ``support_mask``
    ``mask=True`` restricts the volume to the inscribed cylinder of the XZ
    plane; ``support_mask`` is an arbitrary support tensor of shape
    ``[Nz, Nx]``, ``[1, Nz, Nx]`` or ``[Ny, Nz, Nx]``, any real or bool
    dtype and any device (cast and moved internally), values expected in
    ``{0, 1}`` (soft values weight voxels). The two are mutually exclusive
    (ValueError); a wrong shape raises ValueError. Out-of-support voxels of
    the result are exactly 0 for a binary mask. Where it is applied:

    ==============================  =====================================
    algorithm                       support applied
    ==============================  =====================================
    run_fbp, run_weighted_back...   once, to the final volume
    run_sirt, run_wsirt, run_resire every iteration, after the update
    run_sart                        every outer iteration (after all
                                    subsets)
    run_fista                       every iteration, in the projection P_C
    run_fista_tv                    every iteration, inside the TV prox
    run_genfire                     every iteration in real space (on the
                                    oversampled grid) and to the output
    run_grad_descent, run_ad_recon  every optimiser step (``no_grad``)
    run_mbir                        initial volume of each level and after
                                    every outer ICD sweep, not inside the
                                    voxel update
    ==============================  =====================================

    ``run_ad_recon`` takes the cylinder as ``ConstraintConfig.cylinder``.
    The contract is frozen for the desktop app (2026-06-12 handoff), which
    probes ``inspect.signature(fn).parameters`` for ``support_mask``.

Constraints order
    After each update: support (multiply), positivity (``positivity=True``,
    clamp at 0), soft-threshold (``sparsity=t``: ``sign(v) * max(|v| - t,
    0)``, the proximal step of ``t * ||v||_1``). See
    :mod:`fasttomo.constraints`.

``loss_callback`` (frozen, 2026-06-11 handoff)
    Called once per iteration as ``loss_callback(i, loss)``: ``i`` is the
    0-based iteration index, ``loss`` the plain CPU float also stored in
    ``loss_history[i]`` (NaN/Inf passed through). A callback that declares
    a ``components`` keyword (or ``**kwargs``) also receives
    ``components={...}``: CPU floats keyed ``"fidelity"`` plus only the
    regularisers active in the run, summing to ``loss``. A raising callback
    is logged and called again on the next iteration; the reconstruction
    never stops because of it. The loss (and its host sync) is only
    computed when ``return_loss`` or a callback asks for it; computing it
    does not change the iterates (one exception: CUDA ``run_sirt`` then
    runs its eager loop instead of the CUDA graph, which can differ in the
    last bit when ``omega != 1``). The desktop app probes
    ``inspect.signature(fn).parameters`` for ``loss_callback``.

``warm_start``
    Initial volume ``[Ny, Nz, Nx]`` (default zeros), in the sinogram's
    dtype; moved to the sinogram's device and copied, never modified.

``progress``
    ``True`` shows a tqdm progress bar (stderr); off by default.

``num_threads``
    OpenMP threads for the CPU kernels: keyword, else ``OMP_NUM_THREADS``,
    else 1. Ignored on CUDA.

Package layout
==============

The former single ``algorithms.py`` is split by algorithm family (ARCH P8);
this package re-exports every name the module had, so
``fasttomo.algorithms.run_sirt`` and ``from fasttomo.algorithms import
_prepare_recon`` keep working.

==============  ======================================================
module          contents
==============  ======================================================
``_common``     constants, `_prepare_recon`, `_LossRecorder`,
                deprecated-keyword helpers, `_build_fp_bp`
``analytic``    ``run_fbp``, ``run_weighted_backprojection`` (W-BP)
``algebraic``   ``run_sirt`` (+ CUDA graph), ``run_sart``, ``run_wsirt``
``fista``       ``run_fista``, ``run_fista_tv``, TV prox, Lipschitz
``ad``          ``run_ad_recon``, ``run_grad_descent``
``genfire``     ``run_genfire`` (frozen, moved verbatim)
``resire``      ``run_resire`` (frozen, moved verbatim)
``alignment``   ``refine_angles``
``mbir``        ``run_mbir``
==============  ======================================================
"""

from __future__ import annotations

from fasttomo.algorithms._common import (
    _as_float,
    _as_int,
    _assert_projector_output_shape,
    _assert_warm_start_shape,
    _build_fp_bp,
    _callback_accepts_components,
    _check_inputs,
    _deprecated_kwarg,
    _emit_loss,
    _finish_iters,
    _GENFIRE_RES_EXTENSION_PIVOTS,
    _is_default,
    _LIPSCHITZ_SAFETY,
    _log,
    _loss_float,
    _LossRecorder,
    _MBIR_BLANK_STD,
    _MBIR_CONVERGENCE_STRIDE,
    _MBIR_SIGMA_MIN_RATIO,
    _MBIR_WEIGHT_EPS,
    _NORM_EPS,
    _prepare_recon,
    _ReconSetup,
    _reject_unknown_kwargs,
    _REL_EPS,
    _S3IM_SEED,
    _TV_DUAL_STEP,
    _WBP_MIN_THRESHOLD,
)
from fasttomo.algorithms.analytic import (
    run_fbp,
    run_weighted_backprojection,
    _bp_gather,
    _compute_exact_wf,
    _make_wbp,
    _wbp_backproject,
    _WBP_CHUNK_VOXELS,
    _wbp_margin,
)
from fasttomo.algorithms.algebraic import (
    run_sart,
    run_sirt,
    run_wsirt,
    _run_sirt_cuda_graph,
    _sirt_iter_inplace,
)
from fasttomo.algorithms.fista import (
    run_fista,
    run_fista_tv,
    _estimate_lipschitz,
    _fista_loop,
    _project_dual_ball,
    _tv_div,
    _tv_grad,
    _tv_prox_3d,
)
from fasttomo.algorithms.ad import (
    run_ad_recon,
    run_grad_descent,
    _ad_loss_terms,
    _ad_project_constraints,
    _ad_recon_active_steps,
    _ad_recon_step_schedule,
    _ad_sparsity_term,
    _ad_step,
    _AdRun,
    _prepare_ad_recon_pose_init,
)
from fasttomo.algorithms.genfire import (
    run_genfire,
    _genfire_hermitian_sym,
)
from fasttomo.algorithms.resire import (
    run_resire,
)
from fasttomo.algorithms.alignment import (
    refine_angles,
)
from fasttomo.algorithms.mbir import (
    run_mbir,
    _downsample_sino_det,
    _generate_permutation,
    _huber_surr_weight,
    _mbir_forward_project,
    _mbir_icd_sweep_cpu,
    _mbir_initial_residual,
    _mbir_update_offset_variance,
    _MbirState,
    _resize_volume_zx,
    _run_mbir_single_scale,
)

# Names the pre-split module imported and therefore exposed as attributes
# (``fasttomo.algorithms.ParallelProjector`` etc.); kept so old attribute
# paths keep resolving. To monkeypatch what an algorithm calls, patch the
# submodule that uses the name (e.g. ``fasttomo.algorithms.ad``).
import fasttomo._core as _core
from fasttomo._backend import (
    _assert_dtype_match,
    _check_shapes,
    _data_ptr,
    _resolve_num_threads,
)
from fasttomo.autograd import _launch, euler_zyz_to_rotmat
from fasttomo.configs import (
    AdReconConfig,
    ConstraintConfig,
    MbirConfig,
    OptimConfig,
    PoseRefinementConfig,
    RegularizerConfig,
)
from fasttomo.constraints import (
    _maybe_cylindrical_mask,
    _resolve_support_mask,
    _soft_threshold,
    cylindrical_mask,
)
from fasttomo.projection import (
    apply_shifts_2d,
    back_project,
    back_project_3d,
    back_project_resire,
    forward_project,
    forward_project_3d,
    forward_project_resire,
)
from fasttomo.projector import ParallelProjector, RotmatProjector, _drain_deferred_handles
from fasttomo.regularizers import s3im_loss, tilt_ssim_loss, tv_loss_3d, tv_loss_xz
from fasttomo.utilities import ramp_filter

__all__ = [
    "run_fbp",
    "run_weighted_backprojection",
    "run_sirt",
    "run_sart",
    "run_wsirt",
    "run_fista",
    "run_fista_tv",
    "run_genfire",
    "run_resire",
    "run_grad_descent",
    "run_ad_recon",
    "run_mbir",
    "refine_angles",
]
