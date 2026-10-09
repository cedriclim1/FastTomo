"""
Volume-callback contract tests.

Iterative algorithms accept an optional ``volume_callback`` kwarg invoked once
per outer iteration as ``volume_callback(i, volume)`` with a 0-based index and
the live iterate (same shape/dtype/device as the returned volume). A raising
callback must never abort the reconstruction, and supplying a callback must
not change the result.
"""

import inspect
import logging

import numpy as np
import pytest
import torch

import fasttomo as ft
from fasttomo.algorithms import _emit_volume

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA GPU available"
)

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


def _ad_config():
    return ft.AdReconConfig(optim=ft.OptimConfig(n_iter=N_ITER, lr_volume=0.05))


RUNNERS = {
    "sirt_graph": lambda s, a, **kw: ft.run_sirt(s, a, n_iter=N_ITER, **kw),
    "sirt_eager": lambda s, a, **kw: ft.run_sirt(
        s, a, n_iter=N_ITER, return_loss=True, **kw
    )[0],
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
        s, a, outer_iters=N_ITER, convergence_tol=0.0, **kw
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


@requires_cuda
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


@requires_cuda
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


@requires_cuda
def test_sirt_graph_path_kept_with_volume_callback():
    sino, angles = make_sino()
    proj = ft.ParallelProjector((NY, N, N), angles, device=sino.device, N_det=N)
    cb = VolumeRecorder()
    out = ft.run_sirt(sino, angles, n_iter=N_ITER, projector=proj, volume_callback=cb)
    assert len(getattr(proj, "_sirt_graph_cache", {})) == 1
    assert cb.indices == list(range(N_ITER))
    assert not torch.equal(cb.volumes[0], cb.volumes[-1])
    torch.testing.assert_close(cb.volumes[-1], out, rtol=0, atol=0)


@requires_cuda
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
