"""Multi-GPU execution of the reconstruction algorithms (the ``devices=`` keyword).

One host thread per GPU runs the serial code on its share of the problem,
as ASTRA's ``CompositeGeometryManager`` does for its multi-GPU FP/BP/FDK.
Two decompositions:

slab split (FBP, SIRT, SART, FISTA, FISTA-TV, RESIRE; AD and gradient descent
without pose)
    Single-axis parallel beam: volume slice ``y`` only meets detector row
    ``y``, so the problem separates over ``Ny`` (ASTRA's split of the volume
    and the projections into blocks of detector rows). Each GPU solves a
    contiguous block of rows; the blocks are concatenated. Couplings are
    handled exactly: losses are row-weighted means, FISTA's Lipschitz
    constant comes from one power iteration over all blocks, and the
    FISTA-TV prox exchanges a halo of ``tv_inner_iter + 2`` slices per
    iteration (the dependency cone of the inner FGP/Chambolle loop), and
    the autograd solvers' 3-D TV one halo row per step (`_slab_tv`).
view split (AD with pose refinement, gradient descent with euler_zyz_deg)
    Pose couples the rows, so each GPU keeps a replica of the volume and a
    block of views. The data term is weighted by the
    block's share of the views, the volume regularisers are added on the
    first GPU only, and the volume gradients are summed over the GPUs
    before every optimiser step (data-parallel training, as in DDP), so
    the replicas take identical steps.

The worker threads see their role through `_rank_context`; serial calls
(no context) run the unchanged code path.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.cuda.comm

from fasttomo.algorithms._common import (
    _callback_accepts_components,
    _emit_loss,
    _emit_pose,
    _emit_volume,
)

_LOCAL = threading.local()


@dataclass
class _RankContext:
    rank: int
    n_ranks: int
    data_weight: float = 1.0
    primary: bool = True
    rows: tuple[int, int, int] | None = None
    halo: Callable[[torch.Tensor], tuple[torch.Tensor | None, torch.Tensor | None]] | None = None
    grad_sync: Callable[[list[torch.Tensor | None]], None] | None = None
    ref_idx: int | None = None
    tv_prox: Callable[..., torch.Tensor] | None = None


def _rank_context() -> _RankContext | None:
    return getattr(_LOCAL, "ctx", None)


def _resolve_devices(devices: Any, sinogram: torch.Tensor,
                     op_name: str) -> list[torch.device] | None:
    """The CUDA devices of a ``devices=`` argument, or None for a serial run."""
    if devices is None:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError(f"{op_name}: devices= needs CUDA, which is not available.")
    count = torch.cuda.device_count()
    if isinstance(devices, str) and devices == "all":
        devs = [torch.device("cuda", i) for i in range(count)]
    elif isinstance(devices, int) and not isinstance(devices, bool):
        if not 1 <= devices <= count:
            raise ValueError(f"{op_name}: devices={devices} but {count} CUDA devices are visible.")
        devs = [torch.device("cuda", i) for i in range(devices)]
    elif isinstance(devices, Sequence) and not isinstance(devices, str):
        devs = []
        for d in devices:
            dev = torch.device("cuda", d) if isinstance(d, int) else torch.device(d)
            if dev.type != "cuda":
                raise ValueError(f"{op_name}: devices must be CUDA devices, got {d!r}.")
            if dev.index is None:
                dev = torch.device("cuda", torch.cuda.current_device())
            if dev.index >= count:
                raise ValueError(f"{op_name}: {dev} does not exist ({count} CUDA devices).")
            devs.append(dev)
        if len(set(devs)) != len(devs):
            raise ValueError(f"{op_name}: devices contains duplicates: {list(devices)!r}.")
    else:
        raise TypeError(
            f"{op_name}: devices must be None, 'all', an int or a sequence of devices, "
            f"got {devices!r}.")
    if not devs:
        raise ValueError(f"{op_name}: devices is empty.")
    if not sinogram.is_cuda:
        raise ValueError(f"{op_name}: devices= needs a CUDA sinogram.")
    return devs if len(devs) > 1 else None


def _split(n: int, parts: int) -> list[tuple[int, int]]:
    """``n`` items as at most ``parts`` contiguous, balanced, non-empty blocks."""
    parts = max(1, min(parts, n))
    base, extra = divmod(n, parts)
    bounds, s = [], 0
    for k in range(parts):
        e = s + base + (1 if k < extra else 0)
        bounds.append((s, e))
        s = e
    return bounds


class _Group:
    """One worker thread per device, with barriers that break on any failure."""

    def __init__(self, devices: list[torch.device]) -> None:
        self.devices = devices
        self.n = len(devices)
        self._barriers: list[threading.Barrier] = []

    def barrier(self, action: Callable[[], None] | None = None) -> threading.Barrier:
        b = threading.Barrier(self.n, action=action)
        self._barriers.append(b)
        return b

    def run(self, fn: Callable[[int, torch.device], Any],
            contexts: list[_RankContext] | None = None) -> list[Any]:
        results: list[Any] = [None] * self.n
        errors: list[BaseException | None] = [None] * self.n

        def worker(r: int) -> None:
            _LOCAL.ctx = contexts[r] if contexts is not None else None
            try:
                with torch.cuda.device(self.devices[r]):
                    results[r] = fn(r, self.devices[r])
                    torch.cuda.current_stream(self.devices[r]).synchronize()
            except BaseException as exc:
                errors[r] = exc
                for b in self._barriers:
                    b.abort()
            finally:
                _LOCAL.ctx = None

        threads = [threading.Thread(target=worker, args=(r,), name=f"fasttomo-gpu{r}")
                   for r in range(self.n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        real = [e for e in errors if e is not None
                and not isinstance(e, threading.BrokenBarrierError)]
        if real:
            raise real[0]
        for e in errors:
            if e is not None:
                raise e
        return results


class _PerIteration:
    """Collects one value per rank and iteration; ``merge(it, values)`` runs
    once per iteration, in iteration order, when the last rank has reported."""

    def __init__(self, n: int, merge: Callable[[int, list[Any]], None]) -> None:
        self.n = n
        self.merge = merge
        self._lock = threading.Lock()
        self._parts: dict[int, list[Any]] = {}
        self._counts: dict[int, int] = {}

    def submit(self, rank: int, it: int, value: Any) -> None:
        with self._lock:
            parts = self._parts.setdefault(it, [None] * self.n)
            parts[rank] = value
            self._counts[it] = self._counts.get(it, 0) + 1
            if self._counts[it] < self.n:
                return
            del self._parts[it], self._counts[it]
            self.merge(it, parts)


def _loss_merger(loss_callback: Callable | None, weights: list[float]) -> _PerIteration | None:
    if loss_callback is None:
        return None
    send = _callback_accepts_components(loss_callback)

    def merge(it: int, parts: list[tuple[float, dict | None]]) -> None:
        total = math.fsum(w * p[0] for w, p in zip(weights, parts))
        comps = None
        if send:
            keys: list[str] = []
            for _, c in parts:
                for k in (c or {}):
                    if k not in keys:
                        keys.append(k)
            comps = {k: math.fsum(w * (c or {}).get(k, 0.0) for w, (_, c) in zip(weights, parts))
                     for k in keys}
        _emit_loss(loss_callback, send, it, total, comps)

    return _PerIteration(len(weights), merge)


def _rank_loss_callback(merger: _PerIteration | None, rank: int) -> Callable | None:
    if merger is None:
        return None

    def cb(it: int, loss: float, components: dict | None = None) -> None:
        merger.submit(rank, it, (loss, components))

    return cb


def _merge_history(histories: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    total = sum(w * h.double() for w, h in zip(weights, histories))
    return total.float()


def _slab_of(t: torch.Tensor | None, s: int, e: int, Ny: int,
             device: torch.device) -> torch.Tensor | None:
    """Rows ``s:e`` of a per-row ``[Ny, ...]`` tensor; a broadcast one as is."""
    if t is None:
        return None
    if t.ndim == 3 and t.shape[0] == Ny and Ny > 1:
        t = t[s:e]
    return t.to(device).contiguous()


def _gather_rows(slabs: list[torch.Tensor], bounds: list[tuple[int, int]],
                 a: int, b: int, device: torch.device) -> torch.Tensor:
    """Rows ``a:b`` of the volume whose row blocks ``bounds`` are ``slabs``."""
    pieces = []
    for (s, e), slab in zip(bounds, slabs):
        lo, hi = max(a, s), min(b, e)
        if lo < hi:
            pieces.append(slab[lo - s:hi - s].to(device))
    return torch.cat(pieces, 0)


def _slab_run(fn: Callable, devices: list[torch.device], sinogram: torch.Tensor,
              angles_deg: torch.Tensor, args: tuple, kwargs: dict, *,
              support_mask: torch.Tensor | None = None,
              warm_start: torch.Tensor | None = None,
              return_loss: bool = False,
              loss_callback: Callable | None = None,
              volume_callback: Callable | None = None,
              progress: bool = False,
              has_loss: bool = True,
              rank_kwargs: Callable[[int, torch.device], dict] | None = None,
              contexts: list[_RankContext] | None = None,
              bounds: list[tuple[int, int]] | None = None,
              group: _Group | None = None,
              loss_weights: list[float] | None = None):
    """Run ``fn`` (a serial ``run_*``) on row blocks of ``sinogram``, one per device."""
    Ny = sinogram.shape[1]
    bounds = bounds or _split(Ny, len(devices))
    devices = devices[:len(bounds)]
    weights = loss_weights or [(e - s) / Ny for s, e in bounds]
    out_dev = sinogram.device
    group = group or _Group(devices)
    loss_merge = _loss_merger(loss_callback, weights)
    vol_merge = None
    if volume_callback is not None:
        vol_merge = _PerIteration(len(devices), lambda it, parts: _emit_volume(
            volume_callback, it, torch.cat(parts, 0)))

    def work(r: int, dev: torch.device):
        s, e = bounds[r]
        kw = dict(kwargs)
        kw["support_mask"] = _slab_of(support_mask, s, e, Ny, dev)
        if warm_start is not None:
            kw["warm_start"] = warm_start[s:e].to(dev).contiguous()
        if has_loss:
            kw["return_loss"] = return_loss
            kw["loss_callback"] = _rank_loss_callback(loss_merge, r)
        if vol_merge is not None:
            kw["volume_callback"] = lambda it, v: vol_merge.submit(r, it, v.to(out_dev, copy=True))
        if progress:
            kw["progress"] = r == 0
        if rank_kwargs is not None:
            kw.update(rank_kwargs(r, dev, (s, e)))
        sino_r = sinogram[:, s:e].to(dev).contiguous()
        return fn(sino_r, angles_deg, *args, **kw)

    results = group.run(work, contexts)
    vols = [res[0] if isinstance(res, tuple) else res for res in results]
    volume = torch.cat([v.to(out_dev) for v in vols], 0)
    if has_loss and return_loss:
        return volume, _merge_history([res[1] for res in results], weights)
    return volume


def _allreduce_sum(group: _Group) -> Callable[[list[torch.Tensor | None]], None]:
    """A collective: every rank passes its tensors (same shapes, same Nones);
    each returns with them replaced in place by the sum over the ranks."""
    slots: list[list[torch.Tensor | None] | None] = [None] * group.n

    def action() -> None:
        for i in range(len(slots[0])):
            ts = [s[i] for s in slots]
            if ts[0] is None:
                continue
            total = torch.cuda.comm.reduce_add(ts, destination=ts[0].device)
            torch.cuda.comm.broadcast(total, out=ts)

    barrier = group.barrier(action)

    def sync(tensors: list[torch.Tensor | None]) -> None:
        ctx = _rank_context()
        slots[ctx.rank] = tensors
        barrier.wait()

    return sync


def _view_run(fn: Callable, devices: list[torch.device], sinogram: torch.Tensor,
              angles_deg: torch.Tensor, args: tuple, kwargs: dict, *,
              warm_start: torch.Tensor | None = None,
              support_mask: torch.Tensor | None = None,
              return_loss: bool = False,
              loss_callback: Callable | None = None,
              volume_callback: Callable | None = None,
              pose_callback: Callable | None = None,
              return_pose: bool = False,
              progress: bool = False,
              ref_idx: int | None = None,
              rank_kwargs: Callable[[int, torch.device, tuple[int, int]], dict] | None = None):
    """Run ``fn`` (``run_ad_recon`` or ``run_grad_descent``) data-parallel over
    view blocks with a replicated volume and summed volume gradients."""
    n_views = sinogram.shape[0]
    bounds = _split(n_views, len(devices))
    devices = devices[:len(bounds)]
    n = len(devices)
    out_dev = sinogram.device
    group = _Group(devices)
    sync = _allreduce_sum(group)
    contexts = []
    for r, (s, e) in enumerate(bounds):
        local_ref = ref_idx - s if ref_idx is not None and s <= ref_idx < e else None
        contexts.append(_RankContext(rank=r, n_ranks=n, data_weight=(e - s) / n_views,
                                     primary=r == 0, grad_sync=sync, ref_idx=local_ref))
    loss_merge = _loss_merger(loss_callback, [1.0] * n)
    pose_merge = None
    if pose_callback is not None:
        def merge_pose(it: int, parts: list[tuple]) -> None:
            eul = None if parts[0][0] is None else torch.cat([p[0] for p in parts], 0)
            sh = None if parts[0][1] is None else torch.cat([p[1] for p in parts], 0)
            _emit_pose(pose_callback, it, eul, sh)
        pose_merge = _PerIteration(n, merge_pose)

    def work(r: int, dev: torch.device):
        s, e = bounds[r]
        kw = dict(kwargs)
        if warm_start is not None:
            kw["warm_start"] = warm_start.to(dev)
        if support_mask is not None:
            kw["support_mask"] = support_mask.to(dev)
        kw["return_loss"] = return_loss
        kw["loss_callback"] = _rank_loss_callback(loss_merge, r)
        if volume_callback is not None and r == 0:
            kw["volume_callback"] = volume_callback
        if pose_merge is not None:
            def pcb(it, eul, sh):
                pose_merge.submit(r, it, (
                    None if eul is None else eul.to(out_dev, copy=True),
                    None if sh is None else sh.to(out_dev, copy=True)))
            kw["pose_callback"] = pcb
        if return_pose:
            kw["return_pose"] = True
        if progress:
            kw["progress"] = r == 0
        if rank_kwargs is not None:
            kw.update(rank_kwargs(r, dev, (s, e)))
        sino_r = sinogram[s:e].to(dev).contiguous()
        ang_r = angles_deg[s:e] if angles_deg is not None else None
        return fn(sino_r, ang_r, *args, **kw)

    results = group.run(work, contexts)
    first = results[0] if isinstance(results[0], tuple) else (results[0],)
    out: list[Any] = [first[0].to(out_dev)]
    k = 1
    if return_loss:
        out.append(_merge_history([res[k] for res in results], [1.0] * n))
        k += 1
    if return_pose:
        for j in (k, k + 1):
            parts = [res[j] for res in results]
            out.append(None if parts[0] is None else torch.cat(parts, 0))
    return out[0] if len(out) == 1 else tuple(out)


def _slab_lipschitz(devices: list[torch.device], bounds: list[tuple[int, int]],
                    sinogram: torch.Tensor, angles_deg: torch.Tensor, Nz: int, Nx: int,
                    dtype: torch.dtype, n_iter: int, num_threads: int | None) -> float:
    """`fista._estimate_lipschitz` of the whole operator, run on the row blocks:
    same start vector, the norm of ``A^T A v`` summed over the blocks."""
    from fasttomo.algorithms._common import _build_fp_bp
    from fasttomo.algorithms.fista import _LIPSCHITZ_SAFETY, _NORM_EPS, _normalise

    n_angles, Ny, N_det = sinogram.shape
    v = torch.randn((Ny, Nz, Nx), device=sinogram.device, dtype=dtype)
    if dtype == torch.float32:
        v = v / (v.norm() + _NORM_EPS)
    else:
        v = (v.float() / (torch.linalg.vector_norm(v, dtype=torch.float32) + _NORM_EPS)).to(dtype)
    group = _Group(devices[:len(bounds)])
    sq: list[torch.Tensor | None] = [None] * group.n
    nrm: list[torch.Tensor | None] = [None]

    def action() -> None:
        nrm[0] = torch.stack([t.to(group.devices[0]) for t in sq]).sum().sqrt()

    barrier = group.barrier(action)

    def work(r: int, dev: torch.device) -> float:
        s, e = bounds[r]
        sino_r = sinogram[:, s:e].to(dev).contiguous()
        _fp, _bp, op = _build_fp_bp(sino_r, angles_deg, (e - s, Nz, Nx), N_det, n_angles,
                                    dev, num_threads, None, None)
        v_r = v[s:e].to(dev).contiguous()
        out = 0.0
        for _ in range(n_iter):
            AtAv = _bp(_fp(v_r))
            sq[r] = torch.linalg.vector_norm(AtAv, dtype=torch.float32).square()
            barrier.wait()
            n_r = nrm[0].to(dev)
            v_r = _normalise(AtAv, n_r)
            out = n_r
        del op
        return 0.0 if n_iter == 0 else float(out)

    return group.run(work)[0] * _LIPSCHITZ_SAFETY


def _tv_prox_halo(group: _Group, bounds: list[tuple[int, int]], Ny: int,
                  halo: int, support_mask: torch.Tensor | None) -> Callable[..., torch.Tensor]:
    """FISTA-TV's prox on row blocks: each block is extended by ``halo`` rows
    of its neighbours' prox input, proxed, and cropped back. Exact when
    ``halo`` covers the inner loop's dependency cone (one row per iteration)."""
    from fasttomo.algorithms.fista import _tv_prox_3d

    slots: list[torch.Tensor | None] = [None] * group.n
    posted = group.barrier()
    read = group.barrier()
    per_row = support_mask is not None and support_mask.ndim == 3 and support_mask.shape[0] == Ny

    def prox(u: torch.Tensor, lam: float, positivity: bool, mask: torch.Tensor | None,
             n_inner: int, mode: str) -> torch.Tensor:
        r = _rank_context().rank
        s, e = bounds[r]
        a, b = max(0, s - halo), min(Ny, e + halo)
        slots[r] = u
        posted.wait()
        parts = []
        if a < s:
            parts.append(_gather_rows(slots, bounds, a, s, u.device))
        parts.append(u)
        if e < b:
            parts.append(_gather_rows(slots, bounds, e, b, u.device))
        ext = torch.cat(parts, 0) if len(parts) > 1 else u
        read.wait()
        if per_row and mask is not None:
            mask = support_mask[a:b].to(device=u.device, dtype=mask.dtype).contiguous()
        x = _tv_prox_3d(ext, lam, positivity, mask, n_inner, mode)
        return x[s - a:e - a].contiguous()

    return prox


def _check_slab_split(op_name: str, projector: object | None,
                      euler_zyz_deg: torch.Tensor | None) -> None:
    if projector is not None:
        raise ValueError(f"{op_name}: devices= and projector= are mutually exclusive "
                         "(a projector holds one device's full-volume textures).")
    if euler_zyz_deg is not None:
        raise ValueError(f"{op_name}: devices= needs single-axis angles; a euler_zyz_deg "
                         "geometry does not separate along the tilt axis.")


def _slab_fista(fn: Callable, devices: list[torch.device], sinogram: torch.Tensor,
                angles_deg: torch.Tensor, n_iter: int, kwargs: dict, *,
                n_power_iter: int, L: float | None, tv_halo: int | None = None,
                support_mask: torch.Tensor | None = None, **run_kw):
    """`_slab_run` for FISTA / FISTA-TV: one global ``L``, and with ``tv_halo``
    the TV prox of `_tv_prox_halo`."""
    _, Ny, N_det = sinogram.shape
    bounds = _split(Ny, len(devices))
    devices = devices[:len(bounds)]
    Nz = kwargs.get("Nz")
    Nz = N_det if Nz is None or int(Nz) < 0 else int(Nz)
    if L is None:
        L = _slab_lipschitz(devices, bounds, sinogram, angles_deg, Nz, N_det,
                            sinogram.dtype, int(n_power_iter), kwargs.get("num_threads"))
    group = _Group(devices)
    contexts = None
    if tv_halo is not None:
        prox = _tv_prox_halo(group, bounds, Ny, tv_halo, support_mask)
        contexts = [_RankContext(rank=r, n_ranks=len(devices), tv_prox=prox)
                    for r in range(len(devices))]
    return _slab_run(fn, devices, sinogram, angles_deg, (n_iter,), dict(kwargs, L=L),
                     support_mask=support_mask, contexts=contexts, bounds=bounds,
                     group=group, **run_kw)


_HALO_TERM = "_tv_halo"


def _row_halo(group: _Group, bounds: list[tuple[int, int]]) -> Callable:
    """Each rank's neighbour rows ``s - 1`` and ``e`` (None at the volume edges)."""
    slots: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * group.n
    posted = group.barrier()
    read = group.barrier()

    def halo(volume: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        r = _rank_context().rank
        v = volume.detach()
        slots[r] = (v[:1].clone(), v[-1:].clone())
        posted.wait()
        lo = slots[r - 1][1].to(v.device) if r > 0 else None
        hi = slots[r + 1][0].to(v.device) if r < group.n - 1 else None
        read.wait()
        return lo, hi

    return halo


def _slab_tv(ctx: _RankContext, volume: torch.Tensor,
             tv_fn: Callable[[torch.Tensor], torch.Tensor]):
    """``(owned, hidden)`` for a row block of a global 3-D TV mean: ``owned`` is
    the block's share of the TV value (its corners ``s .. e-1``); ``hidden`` is
    corner ``s-1``, counted by the block below, added to the backward only so
    that rows ``s .. e-1`` receive every corner's gradient."""
    s, e, Ny = ctx.rows
    lo, hi = ctx.halo(volume)
    ext = volume if hi is None else torch.cat([volume, hi], 0)
    owned = tv_fn(ext) * ((ext.shape[0] - 1) / (Ny - 1))
    hidden = None
    if lo is not None:
        hidden = tv_fn(torch.cat([lo, volume[:1]], 0)) * (1.0 / (Ny - 1))
    return owned, hidden


def _slab_autograd(fn: Callable, devices: list[torch.device], sinogram: torch.Tensor,
                   angles_deg: torch.Tensor, args: tuple, kwargs: dict, **run_kw):
    """`_slab_run` for the autograd solvers: row-weighted data and volume terms,
    3-D TV through `_slab_tv` with a one-row halo exchanged every step."""
    Ny = sinogram.shape[1]
    bounds = _split(Ny, len(devices))
    devices = devices[:len(bounds)]
    group = _Group(devices)
    halo = _row_halo(group, bounds)
    contexts = [_RankContext(rank=r, n_ranks=len(devices), data_weight=(e - s) / Ny,
                             rows=(s, e, Ny), halo=halo)
                for r, (s, e) in enumerate(bounds)]
    return _slab_run(fn, devices, sinogram, angles_deg, args, kwargs, contexts=contexts,
                     bounds=bounds, group=group, loss_weights=[1.0] * len(devices), **run_kw)
