"""
Loss-callback streaming contract tests.

Contract (app handoff, 2026-06-11): iterative algorithms accept an optional,
``inspect.signature``-introspectable ``loss_callback`` kwarg invoked once per
iteration as ``loss_callback(i, loss)`` — 0-based index, plain CPU float,
NaN/Inf passed through, matching ``loss_history[i]``. Callbacks accepting a
``components`` keyword additionally receive per-term floats whose sum ≈ the
total. A raising callback must never abort the reconstruction, and omitting
the kwarg must leave behaviour unchanged.
"""

import inspect
import math

import pytest
import torch

import fasttomo as ft

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA GPU available"
)

N = 32
N_ANGLES = 12
N_ITER = 6


# ── helpers ────────────────────────────────────────────────────────────────

def make_sino(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-project a centred square phantom into a small tilt series."""
    vol = torch.zeros(2, N, N, device=device)
    q = N // 4
    vol[:, q:3*q, q:3*q] = 1.0
    angles = torch.linspace(-60.0, 60.0, N_ANGLES, device=device)
    return ft.forward_project(vol, angles), angles


class Recorder:
    """2-positional callback (the app's current forwarder shape)."""

    def __init__(self):
        self.calls: list[tuple[int, float]] = []

    def __call__(self, i, loss):
        self.calls.append((i, loss))


class ComponentRecorder:
    """Components-aware callback (the extended forwarder shape)."""

    def __init__(self):
        self.calls: list[tuple[int, float, dict]] = []

    def __call__(self, i, loss, components=None):
        self.calls.append((i, loss, components))


# ── acceptance 1: kwarg is introspectable ──────────────────────────────────

@pytest.mark.parametrize(
    "fn", [ft.run_sirt, ft.run_ad_recon, ft.run_fista, ft.run_fista_tv]
)
def test_loss_callback_in_signature(fn):
    assert "loss_callback" in inspect.signature(fn).parameters


# ── acceptance 2 + 3: indices/values match loss_history, plain floats ──────

def test_sirt_callback_matches_loss_history():
    sino, angles = make_sino()
    cb = Recorder()
    _, hist = ft.run_sirt(
        sino, angles, n_iter=N_ITER, Nz=N, return_loss=True, loss_callback=cb
    )
    assert [i for i, _ in cb.calls] == list(range(N_ITER))
    for i, loss in cb.calls:
        assert type(loss) is float
        assert loss == pytest.approx(hist[i].item(), rel=1e-6)


def test_fista_callback_matches_loss_history():
    sino, angles = make_sino()
    cb = Recorder()
    _, hist = ft.run_fista(
        sino, angles, n_iter=N_ITER, Nz=N, return_loss=True, loss_callback=cb
    )
    assert [i for i, _ in cb.calls] == list(range(N_ITER))
    for i, loss in cb.calls:
        assert type(loss) is float
        assert loss == pytest.approx(hist[i].item(), rel=1e-6)


def test_fista_tv_callback_matches_loss_history():
    sino, angles = make_sino()
    cb = Recorder()
    _, hist = ft.run_fista_tv(
        sino, angles, tv_weight=1e-3, n_iter=N_ITER, Nz=N,
        return_loss=True, loss_callback=cb,
    )
    assert [i for i, _ in cb.calls] == list(range(N_ITER))
    for i, loss in cb.calls:
        assert loss == pytest.approx(hist[i].item(), rel=1e-6)


def test_sirt_callback_without_return_loss():
    """The callback works on its own; return_loss is independent."""
    sino, angles = make_sino()
    cb = Recorder()
    vol = ft.run_sirt(sino, angles, n_iter=N_ITER, Nz=N, loss_callback=cb)
    assert vol.shape == (2, N, N)
    assert len(cb.calls) == N_ITER


def test_sirt_nan_loss_still_invokes_callback():
    sino, angles = make_sino()
    sino[0, 0, 0] = float("nan")
    cb = Recorder()
    ft.run_sirt(sino, angles, n_iter=3, Nz=N, loss_callback=cb)
    assert len(cb.calls) == 3
    assert all(math.isnan(loss) for _, loss in cb.calls)


# ── acceptance 4: raising callback does not abort ──────────────────────────

def test_sirt_raising_callback_does_not_abort():
    sino, angles = make_sino()
    calls = []

    def bad_cb(i, loss):
        calls.append(i)
        raise RuntimeError("boom")

    vol, hist = ft.run_sirt(
        sino, angles, n_iter=N_ITER, Nz=N, return_loss=True, loss_callback=bad_cb
    )
    # every iteration still invoked the callback and the run completed
    assert calls == list(range(N_ITER))
    assert vol.shape == (2, N, N)
    assert len(hist) == N_ITER


# ── acceptance 5: components ───────────────────────────────────────────────

def test_sirt_components_fidelity_only():
    """SIRT's recorded loss is pure data fidelity (sparsity is a prox)."""
    sino, angles = make_sino()
    cb = ComponentRecorder()
    ft.run_sirt(
        sino, angles, n_iter=N_ITER, Nz=N, sparsity=1e-4, loss_callback=cb
    )
    assert len(cb.calls) == N_ITER
    for i, loss, comps in cb.calls:
        assert set(comps) == {"fidelity"}
        assert type(comps["fidelity"]) is float
        assert comps["fidelity"] == pytest.approx(loss)


def test_var_keyword_callback_receives_components():
    sino, angles = make_sino()
    seen = []
    ft.run_sirt(
        sino, angles, n_iter=2, Nz=N,
        loss_callback=lambda i, loss, **kw: seen.append(kw),
    )
    assert all("components" in kw for kw in seen)


# ── acceptance 6: omitting the callback is behaviour-identical ─────────────

def test_sirt_omitting_callback_identical():
    sino, angles = make_sino()
    vol_a, hist_a = ft.run_sirt(sino, angles, n_iter=N_ITER, Nz=N, return_loss=True)
    vol_b, hist_b = ft.run_sirt(
        sino, angles, n_iter=N_ITER, Nz=N, return_loss=True,
        loss_callback=lambda i, loss: None,
    )
    assert torch.equal(vol_a, vol_b)
    assert torch.equal(hist_a, hist_b)


# ── run_ad_recon (CUDA-only) ───────────────────────────────────────────────

@requires_cuda
class TestAdReconCallback:
    def _run(self, cb, *, reg=None, n_iter=N_ITER, return_loss=True):
        sino, angles = make_sino("cuda")
        cfg = ft.AdReconConfig(
            optim=ft.OptimConfig(n_iter=n_iter, lr_volume=0.05),
            reg=reg or ft.RegularizerConfig(),
        )
        return ft.run_ad_recon(
            sino, angles, config=cfg, return_loss=return_loss, loss_callback=cb
        )

    def test_matches_loss_history(self):
        cb = Recorder()
        _, hist = self._run(cb)
        assert [i for i, _ in cb.calls] == list(range(N_ITER))
        for i, loss in cb.calls:
            assert type(loss) is float
            assert loss == pytest.approx(hist[i].item(), rel=1e-6)

    def test_components_sum_to_total(self):
        cb = ComponentRecorder()
        reg = ft.RegularizerConfig(tv_weight=1e-3, sparsity=1e-3)
        _, hist = self._run(cb, reg=reg)
        assert len(cb.calls) == N_ITER
        for i, loss, comps in cb.calls:
            # only the active terms, no 'total' key
            assert set(comps) == {"fidelity", "tv", "sparsity"}
            assert all(type(v) is float for v in comps.values())
            assert sum(comps.values()) == pytest.approx(loss, rel=1e-4)
            assert loss == pytest.approx(hist[i].item(), rel=1e-6)

    def test_inactive_terms_emit_no_keys(self):
        cb = ComponentRecorder()
        self._run(cb)  # no regularisers active
        for _, _, comps in cb.calls:
            assert set(comps) == {"fidelity"}

    def test_raising_callback_does_not_abort(self):
        calls = []

        def bad_cb(i, loss):
            calls.append(i)
            raise RuntimeError("boom")

        vol, hist = self._run(bad_cb)
        assert calls == list(range(N_ITER))
        assert len(hist) == N_ITER


@requires_cuda
def test_sirt_cuda_callback_matches_loss_history():
    """A callback forces the eager loop past the CUDA-Graph fast path."""
    sino, angles = make_sino("cuda")
    cb = Recorder()
    _, hist = ft.run_sirt(
        sino, angles, n_iter=N_ITER, Nz=N, return_loss=True, loss_callback=cb
    )
    assert [i for i, _ in cb.calls] == list(range(N_ITER))
    for i, loss in cb.calls:
        assert loss == pytest.approx(hist[i].item(), rel=1e-6)
