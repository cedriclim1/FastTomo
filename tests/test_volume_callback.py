"""
Volume-callback contract tests.

Iterative algorithms accept an optional ``volume_callback`` kwarg invoked once
per outer iteration as ``volume_callback(i, volume)`` with a 0-based index and
the live iterate (same shape/dtype/device as the returned volume). A raising
callback must never abort the reconstruction, and supplying a callback must
not change the result. ``run_ad_recon``'s ``pose_callback(i, euler, shifts)``
follows the same contract for the live pose parameters.
"""

import inspect
import logging

import numpy as np
import pytest
import torch

import fasttomo as ft
from conftest import needs_cuda
from fasttomo.algorithms import _emit_pose, _emit_volume

N_ANGLES = 5
NY = 16
N = 16
N_ITER = 3

SUPPORTED = [
    ft.run_sirt,
    ft.run_sart,
    ft.run_fista,
    ft.run_fista_tv,
    ft.run_resire,
    ft.run_grad_descent,
    ft.run_ad_recon,
    ft.run_genfire,
    ft.run_mbir,
]


class VolumeRecorder:
    def __init__(self):
        self.indices: list[int] = []
        self.volumes: list[torch.Tensor] = []

    def __call__(self, i, volume):
        self.indices.append(i)
        self.volumes.append(volume.clone())


def make_sino(device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    vol = torch.zeros(NY, N, N, device=device)
    q = N // 4
    vol[:, q:3 * q, q:3 * q] = 1.0
    angles = torch.linspace(-60.0, 60.0, N_ANGLES, device=device)
    sino = ft.forward_project(vol, angles).contiguous()
    return sino, angles


def _euler(angles):
    e = torch.zeros(angles.numel(), 3, device=angles.device)
    e[:, 1] = angles
    return e


def _ad_config():
    return ft.AdReconConfig(optim=ft.OptimConfig(n_iter=N_ITER, lr_volume=0.05))


RUNNERS = {
    "sirt_graph": lambda s, a, **kw: ft.run_sirt(s, a, n_iter=N_ITER, **kw),
    "sirt_eager": lambda s, a, **kw: ft.run_sirt(
        s, a, n_iter=N_ITER, euler_zyz_deg=_euler(a), **kw
    ),
    "sart": lambda s, a, **kw: ft.run_sart(s, a, n_iter=N_ITER, subset_size=2, **kw),
    "fista": lambda s, a, **kw: ft.run_fista(s, a, n_iter=N_ITER, positivity=True, **kw),
    "fista_tv": lambda s, a, **kw: ft.run_fista_tv(
        s, a, tv_weight=1e-3, n_iter=N_ITER, positivity=True, **kw
    ),
    "resire": lambda s, a, **kw: ft.run_resire(s, a, n_iter=N_ITER, **kw),
    "grad_descent": lambda s, a, **kw: ft.run_grad_descent(
        s, a, n_iter=N_ITER, positivity=True, **kw
    ),
    "ad_recon": lambda s, a, **kw: ft.run_ad_recon(s, a, config=_ad_config(), **kw),
    "mbir": lambda s, a, **kw: ft.run_mbir(
        s, a, n_iter=N_ITER, convergence_tol=0.0, **kw
    ),
}


ATOMIC_SCATTER = {"resire"}


def _assert_same_result(name, out, ref):
    if name in ATOMIC_SCATTER:
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    else:
        assert torch.equal(out, ref)


def _run_seeded(runner, sino, angles, **kw):
    torch.manual_seed(0)
    return runner(sino, angles, **kw)


def test_emit_volume_passes_index_and_tensor():
    seen = []
    vol = torch.arange(6.0).reshape(1, 2, 3)
    for i in range(4):
        _emit_volume(lambda it, v: seen.append((it, v)), i, vol)
    assert [it for it, _ in seen] == [0, 1, 2, 3]
    assert all(v is vol for _, v in seen)


def test_emit_volume_swallows_and_logs_exceptions(caplog):
    calls = []

    def bad_cb(i, volume):
        calls.append(i)
        raise RuntimeError("boom")

    vol = torch.zeros(1, 2, 2)
    with caplog.at_level(logging.WARNING, logger="fasttomo.algorithms"):
        for i in range(3):
            _emit_volume(bad_cb, i, vol)
    assert calls == [0, 1, 2]
    warnings = [r for r in caplog.records if "volume_callback raised" in r.getMessage()]
    assert len(warnings) == 3
    assert all(r.exc_info is not None for r in warnings)


def test_emit_volume_does_not_swallow_base_exceptions():
    def interrupt(i, volume):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _emit_volume(interrupt, 0, torch.zeros(1))


@pytest.mark.parametrize("fn", SUPPORTED, ids=lambda f: f.__name__)
def test_volume_callback_in_signature(fn):
    param = inspect.signature(fn).parameters["volume_callback"]
    assert param.default is None


def test_run_fbp_has_no_volume_callback():
    assert "volume_callback" not in inspect.signature(ft.run_fbp).parameters


@needs_cuda
@pytest.mark.parametrize("name", list(RUNNERS))
def test_volume_callback_contract(name):
    runner = RUNNERS[name]
    sino, angles = make_sino()
    cb = VolumeRecorder()
    out_cb = _run_seeded(runner, sino, angles, volume_callback=cb)
    out_ref = _run_seeded(runner, sino, angles)

    assert cb.indices == list(range(N_ITER))
    for v in cb.volumes:
        assert v.shape == out_cb.shape == (NY, N, N)
        assert v.dtype == out_cb.dtype
        assert v.device == out_cb.device
    torch.testing.assert_close(cb.volumes[-1], out_cb, rtol=0, atol=0)
    _assert_same_result(name, out_cb, out_ref)


@needs_cuda
def test_genfire_volume_callback_contract():
    sino, angles = make_sino()
    cb = VolumeRecorder()
    out_cb, hist = ft.run_genfire(
        sino, angles, n_iter=N_ITER, return_loss=True, volume_callback=cb
    )
    out_ref, hist_ref = ft.run_genfire(sino, angles, n_iter=N_ITER, return_loss=True)

    assert cb.indices == list(range(N_ITER))
    for v in cb.volumes:
        assert v.shape == out_cb.shape == (NY, N, N)
        assert v.dtype == out_cb.dtype
        assert v.device == out_cb.device
    best = int(np.argmin(hist.numpy()))
    scale = out_cb.abs().max().item()
    torch.testing.assert_close(cb.volumes[best], out_cb, rtol=1e-4, atol=1e-5 * scale)
    assert torch.equal(out_cb, out_ref)
    assert torch.equal(hist, hist_ref)


@needs_cuda
def test_sirt_graph_path_kept_with_volume_callback():
    sino, angles = make_sino()
    proj = ft.ParallelProjector((NY, N, N), angles, device=sino.device, N_det=N)
    cb = VolumeRecorder()
    out = ft.run_sirt(sino, angles, n_iter=N_ITER, projector=proj, volume_callback=cb)
    assert len(getattr(proj, "_sirt_graph_cache", {})) == 1
    assert cb.indices == list(range(N_ITER))
    assert not torch.equal(cb.volumes[0], cb.volumes[-1])
    torch.testing.assert_close(cb.volumes[-1], out, rtol=0, atol=0)


@needs_cuda
@pytest.mark.parametrize("n_iter", [2, N_ITER])
def test_sirt_graph_intermediate_volumes_match_shorter_runs(n_iter):
    sino, angles = make_sino()
    cb = VolumeRecorder()
    ft.run_sirt(sino, angles, n_iter=N_ITER + 2, volume_callback=cb)
    ref = ft.run_sirt(sino, angles, n_iter=n_iter)
    assert torch.equal(cb.volumes[n_iter - 1], ref)


@needs_cuda
@pytest.mark.parametrize("name", ["sirt_graph", "fista", "ad_recon"])
def test_raising_volume_callback_does_not_abort(name):
    runner = RUNNERS[name]
    sino, angles = make_sino()
    calls = []

    def bad_cb(i, volume):
        calls.append(i)
        raise RuntimeError("boom")

    out = _run_seeded(runner, sino, angles, volume_callback=bad_cb)
    ref = _run_seeded(runner, sino, angles)
    assert calls == list(range(N_ITER))
    _assert_same_result(name, out, ref)


def test_mbir_volume_callback_rejects_multires():
    sino = torch.zeros(N_ANGLES, 2, N)
    angles = torch.linspace(-60.0, 60.0, N_ANGLES)
    with pytest.raises(ValueError, match="volume_callback is not supported by run_mbir"):
        ft.run_mbir(sino, angles, multires_levels=2, volume_callback=lambda i, v: None)


class PoseRecorder:
    def __init__(self, order=None):
        self.indices: list[int] = []
        self.live: list[tuple] = []
        self.poses: list[tuple] = []
        self.order = order

    def __call__(self, i, euler, shifts):
        self.indices.append(i)
        self.live.append((euler, shifts))
        self.poses.append(tuple(None if t is None else t.clone() for t in (euler, shifts)))
        if self.order is not None:
            self.order.append(("pose", i))


def _pose_config(**pose):
    return ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=N_ITER, lr_volume=0.05),
        pose=ft.PoseRefinementConfig(lr_angles=1.0, lr_shifts=1.0, **pose),
    )


def test_emit_pose_passes_index_and_tensors():
    seen = []
    euler, shifts = torch.zeros(N_ANGLES, 3), torch.ones(N_ANGLES, 2)
    for i in range(3):
        _emit_pose(lambda it, e, s: seen.append((it, e, s)), i, euler, None if i == 1 else shifts)
    assert [it for it, _, _ in seen] == [0, 1, 2]
    assert all(e is euler for _, e, _ in seen)
    assert [s is shifts for _, _, s in seen] == [True, False, True]
    assert seen[1][2] is None


def test_emit_pose_swallows_and_logs_exceptions(caplog):
    calls = []

    def bad_cb(i, euler, shifts):
        calls.append(i)
        raise RuntimeError("boom")

    with caplog.at_level(logging.WARNING, logger="fasttomo.algorithms"):
        for i in range(3):
            _emit_pose(bad_cb, i, None, None)
    assert calls == [0, 1, 2]
    warnings = [r for r in caplog.records if "pose_callback raised" in r.getMessage()]
    assert len(warnings) == 3
    assert all(r.exc_info is not None for r in warnings)


def test_emit_pose_does_not_swallow_base_exceptions():
    def interrupt(i, euler, shifts):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _emit_pose(interrupt, 0, None, None)


def test_pose_callback_in_ad_recon_signature():
    param = inspect.signature(ft.run_ad_recon).parameters["pose_callback"]
    assert param.default is None
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


@needs_cuda
@pytest.mark.parametrize("antisymmetric_z", [False, True])
def test_pose_callback_contract(antisymmetric_z):
    sino, angles = make_sino()
    order: list[tuple[str, int]] = []
    cb = PoseRecorder(order)
    cfg = _pose_config(refine_pose=True, refine_shifts=True, antisymmetric_z=antisymmetric_z)
    torch.manual_seed(0)
    out, euler, shifts = ft.run_ad_recon(
        sino, angles, config=cfg, return_pose=True, pose_callback=cb,
        volume_callback=lambda i, v: order.append(("volume", i)),
    )
    torch.manual_seed(0)
    ref, euler_ref, shifts_ref = ft.run_ad_recon(sino, angles, config=cfg, return_pose=True)

    assert cb.indices == list(range(N_ITER))
    assert order == [(kind, i) for i in range(N_ITER) for kind in ("volume", "pose")]
    for e, s in cb.live:
        assert e.shape == (N_ANGLES, 3) and s.shape == (N_ANGLES, 2)
        assert e.dtype == s.dtype == torch.float32
        assert e.device.type == s.device.type == "cuda"
        assert not e.requires_grad and not s.requires_grad
    assert len({e.data_ptr() for e, _ in cb.live}) == 1
    assert len({s.data_ptr() for _, s in cb.live}) == 1
    assert not torch.equal(cb.poses[0][1], cb.poses[-1][1])
    torch.testing.assert_close(cb.poses[-1][0].cpu(), euler, rtol=0, atol=0)
    torch.testing.assert_close(cb.poses[-1][1].cpu(), shifts, rtol=0, atol=0)
    if antisymmetric_z:
        for e, _ in cb.poses:
            torch.testing.assert_close(e[:, 2], -e[:, 0], rtol=0, atol=0)
    assert torch.equal(out, ref)
    assert torch.equal(euler, euler_ref) and torch.equal(shifts, shifts_ref)


@needs_cuda
@pytest.mark.parametrize(
    ("pose", "has_euler", "has_shifts"),
    [
        (None, False, False),
        ({"refine_shifts": True}, False, True),
        ({"refine_pose": True}, True, False),
    ],
)
def test_pose_callback_unrefined_groups_are_none(pose, has_euler, has_shifts):
    sino, angles = make_sino()
    cb = PoseRecorder()
    cfg = _ad_config() if pose is None else _pose_config(**pose)
    ft.run_ad_recon(sino, angles, config=cfg, pose_callback=cb)
    assert cb.indices == list(range(N_ITER))
    for e, s in cb.live:
        assert (e is not None) == has_euler
        assert (s is not None) == has_shifts


@needs_cuda
def test_raising_pose_callback_does_not_abort():
    sino, angles = make_sino()
    cfg = _pose_config(refine_pose=True, refine_shifts=True)
    calls = []

    def bad_cb(i, euler, shifts):
        calls.append(i)
        raise RuntimeError("boom")

    torch.manual_seed(0)
    out = ft.run_ad_recon(sino, angles, config=cfg, return_pose=True, pose_callback=bad_cb)
    torch.manual_seed(0)
    ref = ft.run_ad_recon(sino, angles, config=cfg, return_pose=True)
    assert calls == list(range(N_ITER))
    for a, b in zip(out, ref):
        assert torch.equal(a, b)
