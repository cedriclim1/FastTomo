"""
Algebraic reconstruction: SIRT (with its CUDA-graph fast path), SART and
weighted SIRT (wSIRT).
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from fasttomo._operator import Geometry, _drain_deferred_handles, _make_operator
from fasttomo.autograd import euler_zyz_to_rotmat
from fasttomo.configs import ConstraintConfig
from fasttomo.algorithms._common import (
    _log,
    _assert_projector_output_shape,
    _deprecated_kwarg,
    _reject_unknown_kwargs,
    _as_int,
    _prepare_recon,
    _LossRecorder,
    _build_fp_bp,
    _reciprocal_or_zero,
)
from fasttomo.algorithms.analytic import _make_wbp


def run_sirt(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 50,
    *,
    omega: float = 1.0,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,   # precomputed support, enforced each iteration
    sparsity: float = 0.0,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — use 3D projector if set
    projector: object | None = None,            # ParallelProjector / RotmatProjector for cached FP/BP
    warm_start: torch.Tensor | None = None,     # [Ny, Nz, Nx]; its dtype must match the sinogram
    num_threads: int | None = None,             # CPU paths
    return_loss: bool = False,
    loss_callback: Callable | None = None,      # per-iteration loss streaming hook
    progress: bool = False,
    **deprecated: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Simultaneous Iterative Reconstruction Technique.

    ``x <- x + omega * C * A^T (R * (b - A x))`` with the voxel and ray
    normalisations ``C = 1 / A^T 1`` and ``R = 1 / A 1``, then the
    constraints (support, positivity, soft-threshold ``sparsity``) every
    iteration. Recorded loss: the data MSE ``mean((b - A x)^2)`` *before*
    each update (components ``{"fidelity": loss}``).

    On CUDA the projections go through a cached projector (built internally
    unless ``projector`` is given) and, when no loss is needed, ``n_iter > 1``
    and ``euler_zyz_deg`` is None, the iteration body runs as a captured
    CUDA graph cached on the projector.

    Parameters
    ----------
    sinogram, angles_deg, n_iter
        ``[n_angles, Ny, N_det]`` fp32/fp16, ``[n_angles]`` degrees, and the
        number of iterations (default 50).
    omega : float
        Relaxation (default 1.0).
    Nz, positivity, mask, support_mask, sparsity, warm_start, num_threads,
    return_loss, loss_callback, progress
        See Conventions of :mod:`fasttomo.algorithms`. ``warm_start``'s dtype
        is the compute dtype (it must match the sinogram's).
    euler_zyz_deg : Tensor ``[n_angles, 3]``, optional
        ZYZ Euler angles (degrees, the convention of
        :func:`fasttomo.euler_zyz_to_rotmat`); switches to the rotation-matrix
        projector.
    projector : ParallelProjector or RotmatProjector, optional
        Reuse its texture cache (and, on CUDA, CUDA graphs) across calls. Must
        be sized for this volume and sinogram; its angles replace
        ``angles_deg``, and a RotmatProjector needs ``euler_zyz_deg``.

    Returns
    -------
    volume, or ``(volume, loss_history)``; see Conventions.
    """
    warm_start = _deprecated_kwarg(deprecated, "volume", "warm_start", warm_start, "run_sirt")
    _reject_unknown_kwargs(deprecated, "run_sirt")
    setup = _prepare_recon(
        "run_sirt", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        allowed_dtypes=(torch.float32, torch.float16), warm_start=warm_start,
        positivity=positivity, mask=mask, support_mask=support_mask, sparsity=sparsity,
    )
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)
    volume, constraints, n_iter = setup.volume0, setup.constraints, setup.n_iter

    _fp, _bp, op = _build_fp_bp(
        sinogram, angles_deg, (Ny, Nz, Nx), N_det, n_angles,
        device, num_threads, projector, euler_zyz_deg,
    )

    # --- normalisation matrices (computed once with the chosen projector) ---
    ones_sino = torch.ones_like(sinogram)
    C_vol     = _bp(ones_sino)
    _assert_projector_output_shape(C_vol, (Ny, Nz, Nx), "back-projection", "run_sirt")
    C_vol     = _reciprocal_or_zero(C_vol)

    ones_vol  = torch.ones_like(volume)
    R_sino    = _fp(ones_vol)
    _assert_projector_output_shape(R_sino, (n_angles, Ny, N_det), "forward-projection", "run_sirt")
    R_sino    = _reciprocal_or_zero(R_sino)

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_sirt")

    # --- CUDA Graph capture path ----------------------------------------
    # Gate: CUDA input, no per-iteration loss tracking (which calls .item()).
    # Rotmat mode (euler_zyz_deg) is kept on eager path because the per-call
    # matrix tensor can vary, while cached graphs assume fixed captured inputs.
    # If capture fails, fall back to the eager loop below.
    if sinogram.is_cuda and not rec.wants_loss and n_iter > 1 and euler_zyz_deg is None:
        graph_result = _run_sirt_cuda_graph(
            volume=volume,
            sinogram=sinogram,
            n_iter=n_iter,
            omega=omega,
            C_vol=C_vol,
            R_sino=R_sino,
            _fp=_fp,
            _bp=_bp,
            constraints=constraints,
            projector=op,
            key=(Ny, Nz, Nx, n_angles, N_det,
                 sinogram.dtype,
                 float(omega),
                 constraints.positivity,
                 float(constraints.soft_threshold),
                 tuple(constraints.support_mask.shape)
                 if constraints.support_mask is not None else None,
                 euler_zyz_deg is not None),
            on_replay=rec.tick,
        )
        if graph_result is not None:
            return rec.finish(graph_result)

    # --- eager iteration loop (fallback / CPU / loss-tracking path) ------
    # The loss is the data MSE before the update (SIRT's loss is pure
    # fidelity; sparsity acts as a proximal constraint, not a loss term).
    for it in rec.iterate(n_iter):
        raw_res  = sinogram - _fp(volume)
        if rec.wants_loss:
            rec.record(it, raw_res.pow(2).mean())
        residual = raw_res * R_sino
        volume   = volume + omega * C_vol * _bp(residual)
        volume   = constraints.apply(volume)

    return rec.finish(volume)


def run_sart(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,                   # [n_angles] degrees
    n_iter: int = 5,
    *,
    subset_size: int = 1,
    omega: float = 1.0,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    sparsity: float = 0.0,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — use 3D projector if set
    warm_start: torch.Tensor | None = None,     # [Ny, Nz, Nx]; its dtype must match the sinogram
    num_threads: int | None = None,             # CPU paths
    return_loss: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
    **deprecated: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """SART / OS-SART (Andersen & Kak 1984; Wang & Yu 2007).

    For each subset ``S`` of consecutive angles (``[0:subset_size]``,
    ``[subset_size:2*subset_size]``, ...; the last may be shorter):
    ``x += omega * C_S * A_S^T ((b_S - A_S x) / R_S)`` with
    ``R_S = A_S 1`` and ``C_S = 1 / A_S^T 1``. ``subset_size=1`` is plain
    SART, ``subset_size=n_angles`` one SIRT step. The constraints run once
    per outer iteration, after all subsets. Recorded loss: the full-sinogram
    data MSE after the subset sweep, before the constraints (one extra
    forward projection per iteration, only when the loss is needed).

    A reference implementation: no CUDA graph (on CUDA the projections do
    reuse cached textures). Use :func:`run_sirt` when speed matters.

    Parameters
    ----------
    sinogram, angles_deg, n_iter
        fp32/fp16 sinogram, degrees, outer iterations (default 5).
    subset_size : int
        Angles per subset (default 1; clamped to ``[1, n_angles]``).
    omega : float
        Relaxation (default 1.0).
    Nz, positivity, mask, support_mask, sparsity, warm_start, num_threads,
    return_loss, loss_callback, progress
        See Conventions of :mod:`fasttomo.algorithms`.
    euler_zyz_deg
        As in :func:`run_sirt`.

    Returns
    -------
    volume, or ``(volume, loss_history)``; see Conventions.
    """
    warm_start = _deprecated_kwarg(deprecated, "volume", "warm_start", warm_start, "run_sart")
    _reject_unknown_kwargs(deprecated, "run_sart")
    setup = _prepare_recon(
        "run_sart", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        allowed_dtypes=(torch.float32, torch.float16), warm_start=warm_start,
        positivity=positivity, mask=mask, support_mask=support_mask, sparsity=sparsity,
    )
    subset_size = _as_int(subset_size, "subset_size", "run_sart", minimum=None)
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)
    volume, compute_dtype, n_iter = setup.volume0, setup.compute_dtype, setup.n_iter
    constraints = setup.constraints

    # One projection operator for every subset (CUDA: one cached volume
    # texture plus one sinogram texture per subset size); each subset
    # passes its own angles or rotation matrices, computed once here.
    op = _make_operator(
        Geometry("parallel" if euler_zyz_deg is None else "rotmat", (Ny, Nz, Nx), N_det,
                 n_angles, angles_deg if euler_zyz_deg is None else None),
        device, compute_dtype, num_threads=num_threads)

    def _subset_geom(idx: torch.Tensor | None) -> torch.Tensor | None:
        if euler_zyz_deg is None:
            return None if idx is None else angles_deg[idx]
        e = euler_zyz_deg if idx is None else euler_zyz_deg[idx]
        return euler_zyz_to_rotmat(e).to(device=device)

    # Build subsets: list of index tensors
    clamp = max(1, min(subset_size, n_angles))
    all_idx = torch.arange(n_angles, dtype=torch.long)
    subsets = list(torch.split(all_idx, clamp))
    geoms = [_subset_geom(idx) for idx in subsets]

    # Pre-compute per-subset normalisations (done once, outside the outer loop).
    ones_vol = torch.ones(Ny, Nz, Nx, dtype=compute_dtype, device=device)
    R_list: list[torch.Tensor] = []  # row normalisers [|S|, Ny, N_det]
    C_list: list[torch.Tensor] = []  # col normalisers [Ny, Nz, Nx]
    for idx, geom in zip(subsets, geoms):
        R_s = op.project(ones_vol, geom)
        R_s = _reciprocal_or_zero(R_s)
        R_list.append(R_s)

        # ones_sino for this subset
        ones_sino_s = torch.ones(len(idx), Ny, N_det, dtype=compute_dtype, device=device)
        C_s = op.backproject(ones_sino_s, geom)
        C_s = _reciprocal_or_zero(C_s)
        C_list.append(C_s)

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_sart")
    full_geom = _subset_geom(None) if rec.wants_loss else None

    # --- outer iteration loop ---
    for it in rec.iterate(n_iter):
        # --- inner subset loop ---
        for idx, geom, R_s, C_s in zip(subsets, geoms, R_list, C_list):
            sino_s = sinogram[idx]                   # [|S|, Ny, N_det]
            pred_s = op.project(volume, geom)        # [|S|, Ny, N_det]
            r_s    = sino_s - pred_s                 # [|S|, Ny, N_det]
            volume = volume + omega * C_s * op.backproject(r_s * R_s, geom)

        # Loss over the full sinogram after the subset sweep, before the
        # constraints (costs one extra full forward projection).
        if rec.wants_loss:
            with torch.no_grad():
                raw_res = sinogram - op.project(volume, full_geom)
                rec.record(it, raw_res.pow(2).mean())

        volume = constraints.apply(volume)

    return rec.finish(volume)


def run_wsirt(
    sinogram: torch.Tensor,         # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,       # [n_angles] degrees
    n_iter: int = 50,
    *,
    omega: float = 1.0,
    Nz: int = -1,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    sparsity: float = 0.0,
    threshold_scale: float = 1.0,
    warm_start: torch.Tensor | None = None,
    num_threads: int | None = None,
    return_loss: bool = False,
    loss_callback: Callable | None = None,
    progress: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Weighted SIRT (Wolf et al. 2014). fp32, CPU and CUDA.

    ``x <- x + omega * W-BP(b - A x)``, then the constraints, where W-BP is
    the exact weighted back-projection of :func:`run_weighted_backprojection`
    (edge margin, voxel-driven on both backends). It does not diverge at the
    default ``omega=1``; CPU and CUDA agree to ~2e-4, the floor set by the
    CUDA texture unit's 8-bit interpolation weights. Recorded loss: the data
    MSE before each update.

    Parameters
    ----------
    sinogram, angles_deg, n_iter
        fp32 sinogram, degrees, iterations (default 50).
    omega : float
        Relaxation (default 1.0).
    threshold_scale : float
        Missing-wedge threshold of the weighting filter (default 1.0).
    Nz, positivity, mask, support_mask, sparsity, warm_start, num_threads,
    return_loss, loss_callback, progress
        See Conventions of :mod:`fasttomo.algorithms` (``num_threads`` drives
        the CPU forward projection and the CPU W-BP back-projection).

    Returns
    -------
    volume, or ``(volume, loss_history)``; see Conventions.
    """
    setup = _prepare_recon(
        "run_wsirt", sinogram, angles_deg, Nz=Nz, n_iter=n_iter, warm_start=warm_start,
        positivity=positivity, mask=mask, support_mask=support_mask, sparsity=sparsity,
    )
    N_det, Nz, Nx, device = setup.N_det, setup.Nz, setup.Nx, setup.device
    volume, constraints, n_iter = setup.volume0, setup.constraints, setup.n_iter

    # exact W-BP operator, built once per tilt geometry
    _wbp = _make_wbp(angles_deg, Nz, Nx, N_det, device, threshold_scale, num_threads)
    op = _make_operator(Geometry("parallel", (setup.Ny, Nz, Nx), N_det, setup.n_angles,
                                 angles_deg), device, sinogram.dtype, num_threads=num_threads)

    rec = _LossRecorder(return_loss, loss_callback, progress, n_iter, desc="run_wsirt")

    for it in rec.iterate(n_iter):
        proj     = op.project(volume)
        residual = sinogram - proj
        if rec.wants_loss:
            rec.record(it, residual.pow(2).mean())
        volume   = volume + omega * _wbp(residual)
        volume   = constraints.apply(volume)

    return rec.finish(volume)


# ---------------------------------------------------------------------------
# CUDA Graph helper for run_sirt
# ---------------------------------------------------------------------------

def _run_sirt_cuda_graph(
    volume: torch.Tensor,
    sinogram: torch.Tensor,
    n_iter: int,
    omega: float,
    C_vol: torch.Tensor,
    R_sino: torch.Tensor,
    _fp,
    _bp,
    constraints: ConstraintConfig,
    projector,
    key: tuple,
    on_replay: Callable[[], None] | None = None,
) -> torch.Tensor | None:
    """
    Run ``n_iter`` SIRT iterations via a captured CUDA Graph.

    Returns the reconstructed volume tensor, or ``None`` if capture was
    skipped / failed (caller should fall back to the eager loop).

    The graph is cached on ``projector._sirt_graph_cache[key]``; keys whose
    capture failed are kept in ``projector._sirt_graph_bad_keys`` so later
    calls with that projector skip capture. Both are declared by the
    projector base class; a projector without them (a duck-typed object)
    runs the eager loop.
    """
    cache = getattr(projector, "_sirt_graph_cache", None)
    bad_keys = getattr(projector, "_sirt_graph_bad_keys", None)
    if cache is None or bad_keys is None or key in bad_keys:
        return None

    # CUDA Graph capture is rooted on the *current* CUDA device. When the
    # caller's tensors live on a non-default device (e.g. cuda:1 while the
    # process default is cuda:0), torch.cuda.graph(g) captures on the wrong
    # device's stream — the graph captures nothing and the failed capture
    # leaves the runtime in a state that crashes the eager fallback. Pin
    # the current device to the tensor's device for capture and replay.
    target_device = volume.device

    if key not in cache:
        # --- build a new graph for this key ---
        try:
            # Allocate persistent static buffers. The constraint mask is a
            # graph input too (captured by pointer inside apply_inplace_),
            # so it gets its own static buffer refreshed on every replay —
            # otherwise a later call with a different mask of the same shape
            # would silently replay the first call's mask.
            vol_static = volume.clone()
            sino_static = sinogram.clone()
            C_static = C_vol.clone()
            R_static = R_sino.clone()
            mask_static = (
                constraints.support_mask.clone()
                if constraints.support_mask is not None else None
            )
            static_constraints = ConstraintConfig(
                positivity=constraints.positivity,
                support_mask=mask_static,
                soft_threshold=constraints.soft_threshold,
            )

            # Warmup: one eager iteration with the static buffer.
            # Populates cuFFT plans, allocator pool, projector texture caches.
            with torch.no_grad():
                _sirt_iter_inplace(
                    vol_static, sino_static, C_static, R_static, omega,
                    _fp, _bp, static_constraints,
                )

            # Capture the iteration body on the correct device. Use an
            # explicit Stream pinned to the target device so torch.cuda.graph
            # captures on the right device — the implicit side stream is
            # created lazily on the first call's device and reused for later
            # calls, which corrupts cross-device captures.
            try:
                with torch.cuda.device(target_device):
                    torch.cuda.synchronize(target_device)
                    capture_stream = torch.cuda.Stream(device=target_device)
                    g = torch.cuda.CUDAGraph()
                    with torch.no_grad(), torch.cuda.graph(g, stream=capture_stream):
                        _sirt_iter_inplace(
                            vol_static, sino_static, C_static, R_static, omega,
                            _fp, _bp, static_constraints,
                        )
                    torch.cuda.synchronize(target_device)
            finally:
                # Free texture caches of projectors collected during the
                # capture (their release was deferred, see projector.py).
                _drain_deferred_handles()

            cache[key] = (g, vol_static, sino_static, C_static, R_static,
                          mask_static)

        except Exception as exc:
            _log.warning(
                "run_sirt: CUDA Graph capture failed for key %s (%s). "
                "Falling back to eager loop for this and future calls "
                "with this configuration and projector.",
                key, exc,
            )
            bad_keys.add(key)
            return None

    g, vol_static, sino_static, C_static, R_static, mask_static = cache[key]

    # Reset the captured static buffer to the caller's starting volume so the
    # warmup iteration that ran inside the cache-build branch above does not
    # advance the reconstruction. After this copy each replay applies one
    # SIRT iteration, yielding exactly n_iter updates for both first-call
    # and cache-hit paths.
    vol_static.copy_(volume)
    # Keep graph inputs fresh on cache hits; captured CUDA graphs replay with
    # pointer-stable tensors, so dynamic call-time tensors must be copied into
    # those static buffers before replay.
    sino_static.copy_(sinogram)
    C_static.copy_(C_vol)
    R_static.copy_(R_sino)
    if mask_static is not None:
        mask_static.copy_(constraints.support_mask)

    with torch.cuda.device(target_device):
        for _ in range(n_iter):
            g.replay()
            if on_replay is not None:
                on_replay()
        torch.cuda.synchronize(target_device)

    return vol_static.clone()


def _sirt_iter_inplace(
    vol_static: torch.Tensor,
    sinogram: torch.Tensor,
    C_vol: torch.Tensor,
    R_sino: torch.Tensor,
    omega: float,
    _fp,
    _bp,
    constraints: ConstraintConfig,
) -> None:
    """
    One in-place SIRT iteration on ``vol_static``.

    All writes go through in-place ops (``add_``, ``mul_``, ``clamp_``,
    ``copy_``) so that ``vol_static``'s memory address is stable —
    a requirement for CUDA Graph capture to replay correctly.
    """
    raw_res  = sinogram - _fp(vol_static)
    residual = raw_res * R_sino
    update   = _bp(residual)
    # Avoid creating a temporary for `update * C_vol` to keep allocations
    # inside the graph's memory pool.
    update.mul_(C_vol)
    vol_static.add_(update, alpha=omega)
    constraints.apply_inplace_(vol_static)
