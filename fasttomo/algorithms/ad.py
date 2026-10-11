"""
Automatic-differentiation reconstructions: ``run_ad_recon`` (volume plus
optional pose refinement, regularisers, staged schedules) and the plain
autograd reference ``run_grad_descent``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from fasttomo._operator import Geometry, _make_operator
from fasttomo.autograd import euler_zyz_to_rotmat
from fasttomo.projection import apply_shifts_2d
from fasttomo.constraints import _soft_threshold_
from fasttomo.regularizers import _S3IMTarget, _TiltSSIMTarget, tv_loss_3d, tv_loss_xz
from fasttomo.projector import ParallelProjector, RotmatProjector
from fasttomo.configs import (
    AdReconConfig,
    ConstraintConfig,
    OptimConfig,
    PoseRefinementConfig,
    RegularizerConfig,
)
from fasttomo.algorithms._common import (
    _S3IM_SEED,
    _assert_projector_output_shape,
    _prepare_recon,
    _DeviceLossRecorder,
    _LossRecorder,
    _emit_volume,
)


def _ad_recon_step_schedule(
    iteration: int,
    *,
    pose_warmup_iters: int = 0,
    alternating_period: int = 0,
) -> tuple[bool, bool]:
    """Return ``(step_volume, step_pose)`` for one AD iteration.

    Defaults deliberately return the historical joint schedule. Warmup is
    always volume-only. Alternation starts with a volume-only block, then a
    pose-only block, and repeats.
    """
    if iteration < pose_warmup_iters:
        return True, False
    if alternating_period == 0:
        return True, True
    block = (iteration - pose_warmup_iters) // alternating_period
    step_volume = block % 2 == 0
    return step_volume, not step_volume


def _ad_recon_active_steps(
    n_iter: int,
    *,
    pose_warmup_iters: int = 0,
    alternating_period: int = 0,
) -> tuple[int, int]:
    """``(volume_steps, pose_steps)``: how many of the ``n_iter`` iterations
    step the volume optimiser and the pose/shift optimisers. Each cosine
    schedule uses its optimiser's count as ``T_max`` so its learning rate
    reaches ~0 at the end of that optimiser's own active iterations."""
    vol = pose = 0
    for it in range(n_iter):
        step_volume, step_pose = _ad_recon_step_schedule(
            it, pose_warmup_iters=pose_warmup_iters,
            alternating_period=alternating_period)
        vol += step_volume
        pose += step_pose
    return vol, pose


def _prepare_ad_recon_pose_init(
    value: object,
    *,
    expected_shape: tuple[int, int],
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Convert and validate one engine-native AD pose initialiser."""
    try:
        initial = torch.as_tensor(value, dtype=torch.float32, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if tuple(initial.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(initial.shape)}"
        )
    if not bool(torch.isfinite(initial).all()):
        raise ValueError(f"{name} must contain only finite values")
    return initial.detach().clone()


def _adam(param: torch.Tensor, lr: float) -> torch.optim.Adam:
    """Adam on one leaf: the fused CUDA implementation for a CUDA fp32
    leaf (one kernel per step, no volume-sized temporary; A6-6), else the
    default one. Fused and foreach Adam agree to ~1 ulp per step."""
    fused = param.is_cuda and param.dtype == torch.float32
    return torch.optim.Adam([param], lr=lr, fused=True if fused else None)


def _ad_sparsity_term(volume: torch.Tensor, weight: float) -> torch.Tensor:
    """L1 sparsity penalty for ``run_ad_recon``: ``weight * mean(|volume|)``.

    Must use the magnitude, as documented on ``RegularizerConfig.sparsity``.
    A signed ``mean(volume)`` is a linear term whose gradient is the same
    constant for every voxel, so without positivity it drags the whole
    volume negative instead of promoting sparsity.
    """
    return weight * volume.abs().mean()


@dataclass
class _AdRun:
    """State of one `run_ad_recon` call that its loop reads and updates."""
    sinogram: torch.Tensor
    angles_deg: torch.Tensor              # nominal tilt angles, on the device
    volume: torch.Tensor                  # leaf tensor, requires_grad
    fp: Callable[..., torch.Tensor]       # fp(volume) or fp(volume, mats), differentiable
    op: Any                               # the projection operator (kept alive)
    tilt_angle_opt: torch.Tensor | None   # [n_angles, 3] ZYZ Euler, requires_grad
    shifts: torch.Tensor | None           # [n_angles, 2], requires_grad
    optimisers: list                      # [volume, (angles), (shifts)]
    schedulers: list
    optim_cfg: OptimConfig
    reg_cfg: RegularizerConfig
    constraints_cfg: ConstraintConfig
    pose_cfg: PoseRefinementConfig | None
    refine_pose: bool
    refine_shifts: bool
    ref_idx: int | None                   # anchored view (anchor_reference)
    support: torch.Tensor | None          # resolved support, [1|Ny, Nz, Nx]
    s3im_gen: torch.Generator | None
    ssim_target: _TiltSSIMTarget | None   # tilt_ssim_loss target statistics
    s3im_target: _S3IMTarget | None       # s3im_loss constants
    default_step_schedule: bool
    pose_warmup_iters: int
    alternating_period: int
    pred_shape: tuple[int, int, int]      # (n_angles, Ny, N_det)


def _ad_pose_euler(run: _AdRun) -> torch.Tensor | None:
    """The ``[n, 3]`` ZYZ Euler angles of this iteration's forward
    projection, built from the pose leaf inside the graph.

    With ``antisymmetric_z``, ``psi = -phi`` is part of the graph, so the
    gradient that reaches ``phi`` is the total derivative ``dL/dphi -
    dL/dpsi`` on the constraint set (A6-3; stepping ``phi`` with the
    partial ``dL/dphi`` and overwriting ``psi`` converged to the wrong pose).
    ``psi`` is written as ``(psi_leaf + phi).detach() - phi``: exactly
    ``-phi`` once the post-step constraint has set ``psi_leaf = -phi``, and
    in iteration 0 the initialiser's own ``psi`` up to rounding (``(psi0 +
    phi) - phi`` can differ from ``psi0`` by one ulp), which keeps the
    documented behaviour of ``pose_init_euler_deg``. The leaf's ``psi``
    column gets no gradient.
    """
    tilt = run.tilt_angle_opt
    if tilt is None or not (run.pose_cfg is not None and run.pose_cfg.antisymmetric_z):
        return tilt
    phi = tilt[:, 0]
    psi = (tilt[:, 2] + phi).detach() - phi
    return torch.stack([phi, tilt[:, 1], psi], dim=1)


def _ad_loss_terms(run: _AdRun, pred: torch.Tensor,
                   euler: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Data-fit and regularisation terms, keyed for the loss_callback
    components contract (only the active ones). Insertion order is the
    summation order of the total loss (float addition is not associative),
    kept identical to the historical loop. ``euler``: `_ad_pose_euler`."""
    optim_cfg, reg_cfg, pose_cfg = run.optim_cfg, run.reg_cfg, run.pose_cfg
    volume, sinogram, tilt = run.volume, run.sinogram, euler
    terms: dict[str, torch.Tensor] = {}
    if optim_cfg.loss_fn == "mse":
        terms["fidelity"] = F.mse_loss(pred, sinogram)
    else:
        terms["fidelity"] = F.l1_loss(pred, sinogram)
    if reg_cfg.tv_weight > 0.0:
        terms["tv"] = reg_cfg.tv_weight * tv_loss_3d(volume)
    if reg_cfg.tv_xz_weight > 0.0:
        terms["tv_xz"] = reg_cfg.tv_xz_weight * tv_loss_xz(volume)
    if reg_cfg.sparsity > 0.0:
        terms["sparsity"] = _ad_sparsity_term(volume, reg_cfg.sparsity)
    if pose_cfg is not None and pose_cfg.in_plane_weight > 0.0 and tilt is not None:
        terms["in_plane"] = pose_cfg.in_plane_weight * (
            tilt[:, 0].pow(2).mean()
            + tilt[:, 2].pow(2).mean()
        )
    if reg_cfg.s3im_weight > 0.0:
        terms["s3im"] = reg_cfg.s3im_weight * run.s3im_target(pred, run.s3im_gen)
    if reg_cfg.tilt_ssim_weight > 0.0:
        terms["tilt_ssim"] = reg_cfg.tilt_ssim_weight * run.ssim_target(pred)
    return terms


def _ad_step(run: _AdRun, it: int, rec: _LossRecorder) -> bool:
    """One `run_ad_recon` iteration up to the optimiser steps: forward
    projection, loss, backward, the pose-gradient masks and the scheduled
    optimiser/scheduler steps. Returns whether the volume was stepped."""
    for opt in run.optimisers:
        opt.zero_grad()

    tilt, shifts = run.tilt_angle_opt, run.shifts
    step_volume = step_pose = True
    if not run.default_step_schedule:
        step_volume, step_pose = _ad_recon_step_schedule(
            it,
            pose_warmup_iters=run.pose_warmup_iters,
            alternating_period=run.alternating_period,
        )
        # Only the leaves stepped this iteration need a gradient: a
        # volume-only iteration then skips grad_rotmat and the shift
        # backward, a pose-only one the volume back-projection. The
        # gradients that are computed do not depend on which other leaves
        # require grad, so the result is bit-identical to computing all.
        run.volume.requires_grad_(step_volume)
        for leaf in (tilt, shifts):
            if leaf is not None:
                leaf.requires_grad_(step_pose)

    # forward project
    euler = _ad_pose_euler(run)
    if run.refine_pose and euler is not None:
        mats = euler_zyz_to_rotmat(euler)
        pred = run.fp(run.volume, mats)
    else:
        pred = run.fp(run.volume)

    # A user-supplied projector built for another geometry would make
    # the loss broadcast silently (e.g. [1, Ny, N] against the sinogram).
    # Checked before the shift correction, whose own error ("shifts has
    # N entries but sinogram has M angles") would name the wrong culprit.
    _assert_projector_output_shape(
        pred, run.pred_shape, "forward-projection", "run_ad_recon",
    )

    # optional shift correction
    if run.refine_shifts and shifts is not None:
        pred = apply_shifts_2d(pred, shifts)

    terms = _ad_loss_terms(run, pred, euler)
    loss = terms["fidelity"]
    for key, term in terms.items():
        if key != "fidelity":
            loss = loss + term

    loss.backward()

    if run.ref_idx is not None:
        if tilt is not None and tilt.grad is not None:
            tilt.grad[run.ref_idx] = 0.0
        if shifts is not None and shifts.grad is not None:
            shifts.grad[run.ref_idx] = 0.0

    if run.default_step_schedule:
        # Keep the default path exactly identical to the historical loop.
        for opt in run.optimisers:
            opt.step()
        for sched in run.schedulers:
            sched.step()
    else:
        step_flags = [step_volume] + [step_pose] * (len(run.optimisers) - 1)
        for opt, should_step in zip(run.optimisers, step_flags):
            if should_step:
                opt.step()
        # An inactive optimiser's LR must not decay before its stage runs.
        for sched, should_step in zip(run.schedulers, step_flags):
            if should_step:
                sched.step()
    # The loss of the iterate entering this iteration (before its step),
    # recorded after the step kernels are queued so that a callback's host
    # sync does not sit between the forward and the backward pass.
    if rec.wants_loss:
        rec.record(it, loss, terms)
    return step_volume


def _ad_project_constraints(run: _AdRun, step_volume: bool) -> None:
    """The post-step constraints of `run_ad_recon`, inside ``no_grad`` on
    the leaf tensors: volume positivity/support/soft-threshold only when
    the volume was stepped, then the tilt bound and ``psi = -phi``."""
    volume, tilt = run.volume, run.tilt_angle_opt
    constraints_cfg, pose_cfg = run.constraints_cfg, run.pose_cfg
    with torch.no_grad():
        if step_volume:
            if constraints_cfg.positivity:
                volume.clamp_(min=0.0)
            if run.support is not None:
                volume.mul_(run.support)
            if constraints_cfg.soft_threshold > 0.0:
                # Post-step L1 soft-threshold (ConstraintConfig.soft_threshold),
                # last: positivity -> support mask -> soft-threshold.
                _soft_threshold_(volume, constraints_cfg.soft_threshold)
        tilt_bound_deg = pose_cfg.tilt_bound_deg if pose_cfg is not None else None
        if tilt_bound_deg is not None and run.refine_pose and tilt is not None:
            tilt[:, 1].clamp_(
                min=run.angles_deg - tilt_bound_deg,
                max=run.angles_deg + tilt_bound_deg,
            )
        antisymmetric_z = pose_cfg is not None and pose_cfg.antisymmetric_z
        if antisymmetric_z and run.refine_pose and tilt is not None:
            tilt[:, 2] = -tilt[:, 0]


def run_ad_recon(
    sinogram: torch.Tensor,            # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor,          # [n_angles] degrees
    n_iter: int | None = None,         # None: config.optim.n_iter
    *,
    Nz: int | None = None,
    config: AdReconConfig | None = None,
    support_mask: torch.Tensor | None = None,
    projector: object | None = None,
    warm_start: torch.Tensor | None = None,
    return_loss: bool = False,
    return_pose: bool = False,
    loss_callback: Callable | None = None,
    volume_callback: Callable[[int, torch.Tensor], None] | None = None,
    progress: bool = False,
) -> torch.Tensor | tuple:
    """Autograd reconstruction with optional pose and shift refinement (CUDA, fp32).

    Adam on the volume (and, with ``config.pose``, on per-view ZYZ Euler
    angles and detector shifts) minimising the data term plus the
    regularisers of ``config.reg``. Knobs live on :class:`AdReconConfig` and
    its sub-configs; see those classes for per-field semantics. Recorded
    loss: the total loss of each iteration, with components ``fidelity``
    plus the active regularisers (``tv``, ``tv_xz``, ``sparsity``,
    ``in_plane``, ``s3im``, ``tilt_ssim``), each already weighted.
    ``loss_history[i]`` (and the callback's value for ``i``) is the loss of
    the iterate *entering* iteration ``i``, the one whose gradient that
    iteration steps with: before its optimiser step and constraints (like
    SIRT's residual before the update). ``loss_history[-1]`` is therefore
    not the loss of the returned volume (they differ by the last step, ~0
    with the cosine schedule, whose last learning rate is ~0).

    Parameters
    ----------
    sinogram, angles_deg
        ``[n_angles, Ny, N_det]`` fp32 CUDA tensor and ``[n_angles]`` degrees.
    n_iter : int, optional
        Iterations; ``None`` (default) uses ``config.optim.n_iter``. Given, it
        overrides the config without modifying it.
    config : AdReconConfig, optional
        Optimiser, regulariser, constraint and pose knobs; ``None`` means
        ``AdReconConfig()`` (volume-only refinement). ``config.validate()``
        runs on entry, so fields edited after construction are checked too.
    support_mask : Tensor, optional
        Support enforced after every optimiser step (see Conventions). Same as
        ``config.constraints.support_mask``; setting both, or this and
        ``config.constraints.cylinder``, raises ValueError.
    projector : ParallelProjector or RotmatProjector, optional
        Prebuilt projector: a RotmatProjector when ``config.pose.refine_pose``,
        else a ParallelProjector (ValueError otherwise). Built internally when
        None.
    Nz, warm_start, return_loss, loss_callback, progress
        See Conventions of :mod:`fasttomo.algorithms`.
    return_pose : bool
        Also return ``(euler, shifts)``: the refined ``[n_angles, 3]`` Euler
        angles and ``[n_angles, 2]`` shifts on CPU, each None if not refined.
    volume_callback : callable, optional
        Optional hook invoked once per iteration, after the optimiser
        step(s) and the in-place constraints, as ``volume_callback(i,
        volume)`` — 0-based index and the live volume detached from autograd
        (``[Ny, Nz, Nx]`` float32 on the CUDA device; the same
        representation that is returned). It shares storage with the
        optimised parameter, so the callback must not modify it and must
        ``.clone()`` it to keep it. Exceptions raised by the callback are
        logged and swallowed.

    Returns
    -------
    ``volume``, ``(volume, loss_history)``, ``(volume, euler, shifts)`` or
    ``(volume, loss_history, euler, shifts)``; see Conventions.

    Notes
    -----
    Staging: ``PoseRefinementConfig.pose_warmup_iters`` and
    ``alternating_period`` schedule volume-only, pose-only and joint
    iterations; ``n_iter`` counts all of them and each cosine schedule spans
    only its optimiser's active iterations. See :class:`PoseRefinementConfig`
    for the Euler convention and how the pose constraints treat an
    initialiser.

    Constraints, after every optimiser step inside ``torch.no_grad``:
    in iterations that step the volume, apply positivity, the support, then
    ``ConstraintConfig.soft_threshold`` (all three are skipped in pose-only
    iterations so a soft support does not attenuate a frozen volume);
    always apply the tilt bound and ``psi = -phi`` to pose parameters.
    Positivity and a non-negative support commute, so this keeps the
    constraints order of the Conventions. ``RegularizerConfig.sparsity`` is
    different: a loss term ``sparsity * mean(|f|)`` acting through the
    gradient; both may be set.

    CUDA only: CPU input raises RuntimeError.
    """
    if config is None:
        config = AdReconConfig()
    # The dataclasses are mutable, so construction-time checks can be
    # bypassed by later edits; validate the whole tree here.
    config.validate()

    optim_cfg = config.optim
    reg_cfg = config.reg
    constraints_cfg = config.constraints
    pose_cfg = config.pose

    refine_pose = pose_cfg is not None and pose_cfg.refine_pose
    refine_shifts = pose_cfg is not None and pose_cfg.refine_shifts
    anchor_reference = pose_cfg is not None and pose_cfg.anchor_reference

    # Projector class/mode mismatch validation
    if projector is not None:
        if refine_pose and not isinstance(projector, RotmatProjector):
            raise ValueError(
                f"refine_pose=True requires a RotmatProjector, but got "
                f"{type(projector).__name__}. Either pass a RotmatProjector "
                f"or set refine_pose=False."
            )
        if not refine_pose and not isinstance(projector, ParallelProjector):
            raise ValueError(
                f"refine_pose=False requires a ParallelProjector, but got "
                f"{type(projector).__name__}. Either pass a ParallelProjector "
                f"or set refine_pose=True."
            )

    # ── support: ConstraintConfig.mask (bool True → cylinder; tensor → that
    # support) or the support_mask kwarg, one support per reconstruction ──
    if support_mask is not None:
        if constraints_cfg.cylinder or constraints_cfg.support_mask is not None:
            raise ValueError(
                "run_ad_recon: support_mask and ConstraintConfig.cylinder/support_mask "
                "are mutually exclusive — pass one support per reconstruction."
            )
    else:
        support_mask = constraints_cfg.support_mask

    setup = _prepare_recon(
        "run_ad_recon", sinogram, angles_deg, Nz=Nz,
        n_iter=optim_cfg.n_iter if n_iter is None else n_iter,
        require_cuda=True, warm_start=warm_start,
        mask=constraints_cfg.cylinder, support_mask=support_mask,
    )
    n_angles, Ny, N_det, Nz, Nx, device = (
        setup.n_angles, setup.Ny, setup.N_det, setup.Nz, setup.Nx, setup.device)
    n_iter = setup.n_iter
    angles_deg = angles_deg.float().to(device)

    ref_idx: int | None = None
    if anchor_reference:
        ref_idx = int(angles_deg.abs().argmin())

    # ── initialise volume ──────────────────────────────────────────────────
    volume = setup.volume0.requires_grad_(True)

    # ── projection operator (the caller's projector, else cached textures) ─
    op = _make_operator(
        Geometry("rotmat" if refine_pose else "parallel", (Ny, Nz, Nx), N_det, n_angles,
                 None if refine_pose else angles_deg),
        device, torch.float32, projector)

    # ── initialise pose / shift parameters ────────────────────────────────
    tilt_angle_opt: torch.Tensor | None = None
    shifts: torch.Tensor | None = None

    if refine_pose:
        if pose_cfg is not None and pose_cfg.pose_init_euler_deg is not None:
            tilt_angle_opt = _prepare_ad_recon_pose_init(
                pose_cfg.pose_init_euler_deg,
                expected_shape=(n_angles, 3),
                device=device,
                name="pose_init_euler_deg",
            )
        else:
            tilt_angle_opt = torch.zeros(
                n_angles, 3, dtype=torch.float32, device=device
            )
            tilt_angle_opt[:, 1] = angles_deg
        tilt_angle_opt.requires_grad_(True)

    if refine_shifts:
        if pose_cfg is not None and pose_cfg.pose_init_shifts is not None:
            shifts = _prepare_ad_recon_pose_init(
                pose_cfg.pose_init_shifts,
                expected_shape=(n_angles, 2),
                device=device,
                name="pose_init_shifts",
            )
            shifts.requires_grad_(True)
        else:
            shifts = torch.zeros(
                n_angles, 2, dtype=torch.float32, device=device,
                requires_grad=True,
            )

    # ── build optimiser list ───────────────────────────────────────────────
    opt_vol = _adam(volume, optim_cfg.lr_volume)
    optimisers = [opt_vol]

    opt_angles: torch.optim.Optimizer | None = None
    opt_shifts: torch.optim.Optimizer | None = None

    if refine_pose and tilt_angle_opt is not None and pose_cfg is not None:
        opt_angles = _adam(tilt_angle_opt, pose_cfg.lr_angles)
        optimisers.append(opt_angles)
    if refine_shifts and shifts is not None and pose_cfg is not None:
        opt_shifts = _adam(shifts, pose_cfg.lr_shifts)
        optimisers.append(opt_shifts)

    pose_warmup_iters = pose_cfg.pose_warmup_iters if pose_cfg is not None else 0
    alternating_period = pose_cfg.alternating_period if pose_cfg is not None else 0
    default_step_schedule = pose_warmup_iters == 0 and alternating_period == 0
    if (refine_pose or refine_shifts) and pose_warmup_iters >= n_iter > 0:
        warnings.warn(
            f"run_ad_recon: pose_warmup_iters={pose_warmup_iters} >= "
            f"n_iter={n_iter}, so the pose/shift parameters never step.",
            stacklevel=2,
        )

    # ── cosine LR schedulers ───────────────────────────────────────────────
    # Each scheduler steps only when its optimiser steps, so T_max is that
    # optimiser's number of active iterations (n_iter on the default joint
    # schedule); every LR then anneals to ~0 by the end of its own stage.
    schedulers: list = []
    if optim_cfg.cosine_schedule:
        vol_steps, pose_steps = _ad_recon_active_steps(
            n_iter,
            pose_warmup_iters=pose_warmup_iters,
            alternating_period=alternating_period,
        )
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, vol_steps if opt is opt_vol else pose_steps))
            for opt in optimisers
        ]

    # ── S3IM generator (seeded once; reused each iteration) ───────────────
    _s3im_gen: torch.Generator | None = None
    s3im_target: _S3IMTarget | None = None
    if reg_cfg.s3im_weight > 0.0:
        _s3im_gen = torch.Generator(device=device).manual_seed(_S3IM_SEED)
        s3im_target = _S3IMTarget(sinogram)

    # ── tilt SSIM: the sinogram's statistics, computed once ───────────────
    ssim_target = _TiltSSIMTarget(sinogram) if reg_cfg.tilt_ssim_weight > 0.0 else None

    # ── iteration ─────────────────────────────────────────────────────────
    run = _AdRun(
        sinogram=sinogram, angles_deg=angles_deg, volume=volume, fp=op.project, op=op,
        tilt_angle_opt=tilt_angle_opt, shifts=shifts,
        optimisers=optimisers, schedulers=schedulers,
        optim_cfg=optim_cfg, reg_cfg=reg_cfg, constraints_cfg=constraints_cfg,
        pose_cfg=pose_cfg, refine_pose=refine_pose, refine_shifts=refine_shifts,
        ref_idx=ref_idx, support=setup.constraints.support_mask, s3im_gen=_s3im_gen,
        ssim_target=ssim_target, s3im_target=s3im_target,
        default_step_schedule=default_step_schedule,
        pose_warmup_iters=pose_warmup_iters, alternating_period=alternating_period,
        pred_shape=(n_angles, Ny, N_det),
    )
    rec = _DeviceLossRecorder(return_loss, loss_callback, progress, n_iter,
                              desc="run_ad_recon")
    for it in rec.iterate(n_iter):
        step_volume = _ad_step(run, it, rec)
        _ad_project_constraints(run, step_volume)
        if volume_callback is not None:
            _emit_volume(volume_callback, it, volume.detach())
    rec.close()

    out = volume.detach()

    # ── build return tuple ─────────────────────────────────────────────────
    if not return_loss and not return_pose:
        return out

    result: list = [out]
    if return_loss:
        result.append(rec.history())
    if return_pose:
        refined_euler = tilt_angle_opt.detach().cpu() if tilt_angle_opt is not None else None
        refined_shifts = shifts.detach().cpu() if shifts is not None else None
        result.append(refined_euler)
        result.append(refined_shifts)

    return tuple(result)


# ── autograd-based reference reconstruction ──────────────────────────────

_LANDWEBER_POWER_ITERS = 20
_LANDWEBER_SEED = 0x5EED


def _landweber_step(op, geom, volume: torch.Tensor, n_meas: int) -> float:
    """``1 / L`` for the gradient of ``mean((A x - b)^2)``: ``L = 2 ||B A||
    / n_meas`` with ``||B A||`` from a seeded power iteration on the
    operator's raw ``fp`` / ``bp`` (the global RNG is not touched), times a
    1 % safety margin. One host sync per power step, at setup only."""
    g = torch.Generator().manual_seed(_LANDWEBER_SEED)
    v = torch.randn(volume.shape, generator=g, dtype=volume.dtype).to(volume.device)
    v = v / v.norm()
    norm = 0.0
    with torch.no_grad():
        for _ in range(_LANDWEBER_POWER_ITERS):
            w = op.bp(op.fp(v, geom), geom)
            norm = float(w.norm())
            if not norm > 0.0:          # A v = 0 (or non-finite): keep 1e-2
                return 1e-2
            v = w / norm
    return n_meas / (2.0 * norm * 1.01)

def run_grad_descent(
    sinogram: torch.Tensor,                     # [n_angles, Ny, N_det]
    angles_deg: torch.Tensor | None = None,     # [n_angles] — single-axis
    n_iter: int = 200,
    *,
    lr: float | None = None,
    Nz: int | None = None,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    positivity: bool = False,
    mask: bool = False,
    support_mask: torch.Tensor | None = None,
    tv_weight: float = 0.0,
    tv_eps: float = 1e-8,
    euler_zyz_deg: torch.Tensor | None = None,  # [n_angles, 3] — 3D mode
    use_resire_projector: bool = False,
    warm_start: torch.Tensor | None = None,
    num_threads: int | None = None,             # CPU paths
    return_loss: bool = False,
    loss_callback: Callable | None = None,
    volume_callback: Callable[[int, torch.Tensor], None] | None = None,
    progress: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference autograd reconstruction with a ``torch.optim`` optimiser. fp32.

    Minimises ``mean((A x - b)^2) + tv_weight * TV(x)`` over the volume with
    ``optimizer_cls`` (Adam by default; SGD gives Landweber; LBFGS works, its
    closure is re-evaluated as it needs). Constraints are applied in place
    under ``torch.no_grad`` after each step, the autograd-safe way to project
    a leaf tensor. Recorded loss: the total loss of each step (the first
    closure evaluation for LBFGS), components ``fidelity`` and ``tv``,
    evaluated at the iterate entering the step (before the update and the
    constraints).

    Not a replacement for SIRT/RESIRE: use it for a custom differentiable
    loss or a learnable prior. Peak memory ~7 N^3 vs SIRT's ~5 N^3.

    Parameters
    ----------
    sinogram : Tensor ``[n_angles, Ny, N_det]`` fp32
    angles_deg : Tensor ``[n_angles]``, optional
        Single-axis angles; exactly one of ``angles_deg`` and ``euler_zyz_deg``.
    n_iter : int
        Optimiser steps (default 200).
    lr : float, optional
        Learning rate. Default (None): 1e-2, except for
        ``optimizer_cls=torch.optim.SGD`` (or a subclass), where it is
        ``1 / L``, ``L = 2 ||B A|| / M`` the Lipschitz constant of the
        gradient of the data term ``mean((A x - b)^2)`` (``M`` sinogram
        entries, ``B`` the back-projection, ``||B A||`` from 20 power
        iterations with a fixed seed): plain gradient descent at the
        classic Landweber step. A fixed 1e-2 was 3 400 to 13 400 times
        smaller than that at 64^3 to 256^3, so SGD barely moved. The TV
        term is not included (its gradient is not Lipschitz for small
        ``tv_eps``); lower ``lr`` if SGD with a large ``tv_weight``
        oscillates. Adam's step does not depend on the gradient's scale, so
        its default is unchanged.
    optimizer_cls : type
        Any ``torch.optim.Optimizer`` class.
    tv_weight, tv_eps : float
        3-D isotropic TV weight (0 disables it) and the smoothing constant
        inside the square root (:func:`fasttomo.tv_loss_3d`).
    euler_zyz_deg : Tensor ``[n_angles, 3]``, optional
        Rotation-matrix projector.
    use_resire_projector : bool
        With ``euler_zyz_deg``: RESIRE's projector instead (CUDA only).
    volume_callback : callable, optional
        Optional hook invoked once per optimizer step, after the in-place
        constraints, as ``volume_callback(i, volume)`` — 0-based index and
        the live volume detached from autograd (``[Ny, Nz, Nx]`` float32 on
        its device; the same representation that is returned). It shares
        storage with the optimised parameter, so the callback must not
        modify it and must ``.clone()`` it to keep it. Exceptions are logged
        and swallowed. With LBFGS one call corresponds to one ``step()``
        (which may evaluate the closure several times).
    Nz, positivity, mask, support_mask, warm_start, num_threads, return_loss,
    loss_callback, progress
        See Conventions of :mod:`fasttomo.algorithms`.

    Returns
    -------
    volume, or ``(volume, loss_history)``; see Conventions.
    """
    if (angles_deg is None) == (euler_zyz_deg is None):
        raise ValueError(
            "Provide exactly one of angles_deg (single-axis) or "
            "euler_zyz_deg (3D ZYZ), not both or neither."
        )

    setup = _prepare_recon(
        "run_grad_descent", sinogram, angles_deg, Nz=Nz, n_iter=n_iter,
        warm_start=warm_start, positivity=positivity, mask=mask, support_mask=support_mask,
    )
    N_det = setup.N_det
    constraints, n_iter = setup.constraints, setup.n_iter

    volume = setup.volume0.requires_grad_(True)

    # parallel: the angles are the operator's; rotmat: matrices computed
    # once; RESIRE: its wrappers take the Euler angles themselves
    if euler_zyz_deg is None:
        kind, geom = "parallel", None
    elif use_resire_projector:
        kind, geom = "resire", euler_zyz_deg
    else:
        kind, geom = "rotmat", euler_zyz_to_rotmat(euler_zyz_deg.detach()).to(setup.device)
    op = _make_operator(
        Geometry(kind, (setup.Ny, setup.Nz, setup.Nx), N_det, setup.n_angles,
                 angles_deg if kind == "parallel" else None),
        setup.device, sinogram.dtype, num_threads=num_threads)

    def _fp(v):
        return op.project(v, geom)

    if lr is None:
        lr = (_landweber_step(op, geom, volume, sinogram.numel())
              if issubclass(optimizer_cls, torch.optim.SGD) else 1e-2)
    opt = optimizer_cls([volume], lr=lr)

    is_lbfgs = isinstance(opt, torch.optim.LBFGS)
    rec = _DeviceLossRecorder(return_loss, loss_callback, progress, n_iter,
                              desc="run_grad_descent")

    # Loss terms of the first closure evaluation of each step: the loss
    # LBFGS.step returns is that evaluation's (it may evaluate several times).
    terms: dict[str, torch.Tensor] = {}

    def closure():
        opt.zero_grad()
        sino_pred = _fp(volume)
        fidelity = ((sino_pred - sinogram) ** 2).mean()
        loss = fidelity
        tv = None
        if tv_weight > 0.0:
            tv = tv_weight * tv_loss_3d(volume, eps=tv_eps)
            loss = loss + tv
        if not terms:
            terms["fidelity"] = fidelity.detach()
            if tv is not None:
                terms["tv"] = tv.detach()
        loss.backward()
        return loss

    for it in rec.iterate(n_iter):
        terms.clear()
        if is_lbfgs:
            loss_val = opt.step(closure)
        else:
            loss_val = closure()
            opt.step()
        if rec.wants_loss:
            rec.record(it, loss_val.detach(), terms)
        with torch.no_grad():
            constraints.apply_inplace_(volume)
        if volume_callback is not None:
            _emit_volume(volume_callback, it, volume.detach())

    return rec.finish(volume.detach())
