"""
Contracts every reconstruction algorithm honours (ARCH P13): one capability
table, ``ALGOS``, and parametrised tests for ``loss_callback``,
``return_loss``, ``warm_start``, ``support_mask``, ``progress``, the
positional order, dtype acceptance, ``num_threads`` and the deprecated
aliases, run on CPU and on CUDA; then the frozen 2026-06-11 loss-callback
streaming contract; then the public import surface.

Former test_api_contracts.py (its N/N_ANGLES/N_ITER renamed SIZE/VIEWS/
ITERS; the uniform-keyword tests now also run on CUDA) and
test_loss_callback.py (every test body unchanged).

Loss-callback contract (app handoff, 2026-06-11): iterative algorithms accept an optional,
``inspect.signature``-introspectable ``loss_callback`` kwarg invoked once per
iteration as ``loss_callback(i, loss)`` — 0-based index, plain CPU float,
NaN/Inf passed through, matching ``loss_history[i]``. Callbacks accepting a
``components`` keyword additionally receive per-term floats whose sum ≈ the
total. A raising callback must never abort the reconstruction, and omitting
the kwarg must leave behaviour unchanged.

The desktop app probes ``inspect.signature(fn).parameters`` for
``loss_callback`` and ``support_mask`` (2026-06 handoff docs), so those must
be named parameters, not ``**kwargs``.
"""

from __future__ import annotations

import inspect
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from conftest import DTYPE_CAPS, HAS_CUDA, needs_cuda

import fasttomo as ft
from fasttomo import algorithms as algo

# ════════════════════════════════════════════════════════════════════════════
# Part 1: API contracts
# ════════════════════════════════════════════════════════════════════════════

# Problem size of the API-contract tests (the loss-callback tests below
# keep their own SIZE, VIEWS, ITERS).
SIZE, NY, VIEWS, ITERS = 16, 2, 10, 3


@dataclass(frozen=True)
class Caps:
    """What an algorithm advertises."""
    iterative: bool = True
    dtypes: tuple = (torch.float32,)
    cuda_only: bool = False
    warm_identity: bool = True       # n_iter=0 returns the warm start unchanged
    n_iter_positional: bool = True   # (sinogram, angles, n_iter) positional
    extra: dict = field(default_factory=dict)


ALGOS = {
    "run_fbp": Caps(iterative=False, dtypes=(torch.float32, torch.float16),
                    n_iter_positional=False),
    "run_weighted_backprojection": Caps(iterative=False, n_iter_positional=False),
    "run_sirt": Caps(dtypes=(torch.float32, torch.float16)),
    "run_sart": Caps(dtypes=(torch.float32, torch.float16), extra={"subset_size": 4}),
    "run_wsirt": Caps(),
    "run_fista": Caps(dtypes=(torch.float32, torch.float16)),
    "run_fista_tv": Caps(dtypes=(torch.float32, torch.float16), extra={"tv_weight": 1e-3}),
    # GENFIRE's warm start seeds the Fourier iterate; n_iter is keyword-only
    # for one release (its old third positional was Nz).
    "run_genfire": Caps(warm_identity=False, n_iter_positional=False),
    "run_resire": Caps(cuda_only=True),
    "run_grad_descent": Caps(),
    "run_ad_recon": Caps(cuda_only=True),
    "run_mbir": Caps(extra={"convergence_tol": 0.0}),
}
ITERATIVE = [k for k, c in ALGOS.items() if c.iterative]


def _fn(name):
    return getattr(ft, name)


def _device(caps: Caps) -> str:
    if caps.cuda_only:
        if not HAS_CUDA:
            pytest.skip("CUDA-only algorithm, no CUDA GPU")
        return "cuda"
    return "cpu"


def _cases(names):
    """(name, device) parameters: CPU and CUDA for every algorithm with a
    CPU path, CUDA only for the CUDA-only ones. IDs read ``run_sirt-cpu``."""
    out = []
    for name in names:
        devices = ["cuda"] if ALGOS[name].cuda_only else ["cpu", "cuda"]
        for device in devices:
            marks = needs_cuda if device == "cuda" else ()
            out.append(pytest.param(name, device, marks=marks, id=f"{name}-{device}"))
    return out


def _data(device, dtype=torch.float32):
    vol = torch.zeros(NY, SIZE, SIZE, device=device)
    vol[:, 4:12, 5:11] = 1.0
    ang = torch.linspace(-60, 60, VIEWS, device=device)
    return ft.forward_project(vol, ang).to(dtype), ang


def _call(name, sino, ang, **kw):
    caps = ALGOS[name]
    kw = {**caps.extra, **kw}
    if caps.iterative:
        kw.setdefault("n_iter", ITERS)
    return _fn(name)(sino, ang, Nz=SIZE, **kw)


# ── signature: the frozen names are introspectable ─────────────────────────

@pytest.mark.parametrize("name", list(ALGOS))
def test_signature_exposes_support_mask_by_name(name):
    params = inspect.signature(_fn(name)).parameters
    assert "support_mask" in params
    assert params["support_mask"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("name", ITERATIVE)
def test_signature_exposes_uniform_iterative_keywords(name):
    params = inspect.signature(_fn(name)).parameters
    for kw in ("loss_callback", "warm_start", "progress", "return_loss"):
        assert kw in params, kw
        assert params[kw].kind is inspect.Parameter.KEYWORD_ONLY, kw
    assert "n_iter" in params


@pytest.mark.parametrize("name", list(ALGOS))
def test_positional_order(name):
    """(sinogram, angles_deg[, n_iter]) positionally, everything after it
    keyword-only."""
    params = list(inspect.signature(_fn(name)).parameters.values())
    positional = [p.name for p in params if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    expected = ["sinogram", "angles_deg"]
    if ALGOS[name].n_iter_positional:
        expected.append("n_iter")
    assert positional == expected


# ── the uniform keywords do what they advertise ────────────────────────────

class _Calls:
    def __init__(self):
        self.calls = []

    def __call__(self, i, loss, components=None):
        self.calls.append((i, loss, components))


@pytest.mark.parametrize(("name", "device"), _cases(ITERATIVE))
def test_loss_callback_and_return_loss(name, device):
    sino, ang = _data(device)
    cb = _Calls()
    vol, hist = _call(name, sino, ang, return_loss=True, loss_callback=cb)
    assert vol.shape == (NY, SIZE, SIZE)
    assert hist.dtype == torch.float32 and hist.device.type == "cpu"
    assert [c[0] for c in cb.calls] == list(range(ITERS)) == list(range(len(hist)))
    for (_, loss, comps), h in zip(cb.calls, hist.tolist()):
        assert type(loss) is float
        assert loss == pytest.approx(h, rel=1e-6, abs=1e-12)
        assert isinstance(comps, dict) and "fidelity" in comps
        assert sum(comps.values()) == pytest.approx(loss, rel=1e-4, abs=1e-9)


@pytest.mark.parametrize(("name", "device"), _cases(ITERATIVE))
def test_raising_callback_does_not_abort(name, device):
    sino, ang = _data(device)
    seen = []

    def bad(i, loss):
        seen.append(i)
        raise RuntimeError("boom")

    _call(name, sino, ang, loss_callback=bad)
    assert seen == list(range(ITERS))


@pytest.mark.parametrize(("name", "device"), _cases(ITERATIVE))
def test_callback_does_not_change_the_result(name, device):
    sino, ang = _data(device)
    torch.manual_seed(0)
    a = _call(name, sino, ang)
    torch.manual_seed(0)
    b = _call(name, sino, ang, loss_callback=lambda i, loss: None)
    if name in ("run_resire", "run_genfire") and sino.is_cuda:
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)   # atomics
    else:
        assert torch.equal(a, b)


@pytest.mark.parametrize(("name", "device"), _cases(ITERATIVE))
def test_warm_start(name, device):
    caps = ALGOS[name]
    sino, ang = _data(device)
    warm = torch.rand(NY, SIZE, SIZE, generator=torch.Generator().manual_seed(1)).to(device)
    keep = warm.clone()
    out = _call(name, sino, ang, warm_start=warm, n_iter=0 if caps.warm_identity else 2)
    assert torch.equal(warm, keep)                 # the caller's tensor is not modified
    if caps.warm_identity:
        assert torch.equal(out, warm)
    else:
        assert not torch.equal(out, _call(name, sino, ang, n_iter=2))
    with pytest.raises(ValueError, match="warm-start volume must have shape"):
        _call(name, sino, ang, warm_start=warm[:1])
    with pytest.raises(TypeError):
        _call(name, sino, ang, warm_start=warm.double())


@pytest.mark.parametrize(("name", "device"), _cases(list(ALGOS)))
def test_support_mask(name, device):
    """Out-of-support voxels are exactly 0, the mask excludes mask=True and a
    wrong shape raises ValueError (the 2026-06 support_mask contract)."""
    sino, ang = _data(device)
    M = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    M[3:13, 3:13] = True                           # a CPU bool mask is accepted
    out = _call(name, sino, ang, support_mask=M)
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == (NY, SIZE, SIZE)
    assert (out[:, ~M.to(out.device)] == 0).all()
    assert out[:, M.to(out.device)].abs().max() > 0
    if name != "run_ad_recon":                     # AD: the cylinder lives on the config
        with pytest.raises(ValueError, match="mutually exclusive"):
            _call(name, sino, ang, support_mask=M, mask=True)
    with pytest.raises(ValueError, match="support_mask"):
        _call(name, sino, ang, support_mask=torch.ones(SIZE + 1, SIZE))


@pytest.mark.parametrize(("name", "device"), _cases(ITERATIVE))
def test_progress_bar(name, device, capsys):
    sino, ang = _data(device)
    _call(name, sino, ang, progress=True)
    assert name in capsys.readouterr().err        # tqdm writes to stderr
    _call(name, sino, ang)
    assert capsys.readouterr().err == ""          # off by default


# ── dtypes: accept what is advertised, raise on anything else ──────────────

@pytest.mark.parametrize(("name", "device"), _cases(list(ALGOS)))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, torch.bfloat16])
def test_dtype_accept_or_raise(name, device, dtype):
    caps = ALGOS[name]
    sino, ang = _data(device, dtype)
    if dtype in caps.dtypes:
        out = _call(name, sino, ang)
        assert out.dtype == dtype
    else:
        with pytest.raises(TypeError):
            _call(name, sino, ang)


# ── num_threads reaches the CPU kernels ───────────────────────────────────

@pytest.mark.parametrize("name", [k for k, c in ALGOS.items() if not c.cuda_only
                                  and k not in ("run_weighted_backprojection", "run_genfire")])
def test_num_threads_honoured_on_cpu(name, monkeypatch):
    """Every CPU kernel call resolves the thread count from the kwarg (MBIR
    used to hardcode 4). run_weighted_backprojection's CPU path is pure
    PyTorch; GENFIRE's CPU gridding kernel (frozen) is serial and takes no
    thread count."""
    import fasttomo.autograd as ag
    seen = []
    real = ag._resolve_num_threads

    def spy(n):
        seen.append(n)
        return real(n)

    monkeypatch.setattr(ag, "_resolve_num_threads", spy)
    monkeypatch.setattr(algo.mbir, "_resolve_num_threads", spy)
    monkeypatch.setattr(ft.projection, "_resolve_num_threads", spy)
    sino, ang = _data("cpu")
    seen.clear()
    _call(name, sino, ang, num_threads=3)
    assert seen and all(n == 3 for n in seen), seen


def test_mbir_default_threads_follow_the_library_convention(monkeypatch):
    calls = []
    real = algo._core.mbir_icd_sweep_cpu
    monkeypatch.setattr(algo._core, "mbir_icd_sweep_cpu",
                        lambda **kw: (calls.append(kw["num_threads"]), real(**kw))[1])
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    sino, ang = _data("cpu")
    ft.run_mbir(sino, ang, 2, Nz=SIZE, convergence_tol=0.0)
    assert calls == [1, 1]                         # kwarg -> OMP_NUM_THREADS -> 1
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    calls.clear()
    ft.run_mbir(sino, ang, 2, Nz=SIZE, convergence_tol=0.0)
    assert calls == [2, 2]


# ── stale calls fail loudly ────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    pytest.param(n, marks=needs_cuda if ALGOS[n].cuda_only else (), id=n) for n in ALGOS])
def test_old_positional_calls_raise(name):
    """Everything after the core positionals is keyword-only: a call written
    for an old positional order (Nz, omega, tv_weight, ... in 3rd/4th place)
    raises TypeError instead of binding to another parameter."""
    caps = ALGOS[name]
    sino, ang = _data(_device(caps))
    n_pos = 3 if caps.n_iter_positional else 2
    with pytest.raises(TypeError):
        _fn(name)(sino, ang, *([2] * (n_pos - 2)), 16)


def test_fista_tv_old_order_raises():
    sino, ang = _data("cpu")
    with pytest.raises(TypeError):
        ft.run_fista_tv(sino, ang, 0.01)           # was tv_weight
    with pytest.raises(TypeError):
        ft.run_fista_tv(sino, ang, 0.01, 5)


def test_genfire_old_order_raises():
    sino, ang = _data("cpu")
    with pytest.raises(TypeError):
        ft.run_genfire(sino, ang, 16)              # was Nz


def test_unknown_keyword_raises():
    sino, ang = _data("cpu")
    with pytest.raises(TypeError, match="unexpected keyword argument 'omgea'"):
        ft.run_sirt(sino, ang, 2, omgea=1.0)


def test_integer_arguments_are_checked():
    sino, ang = _data("cpu")
    with pytest.raises(TypeError, match="n_iter"):
        ft.run_sirt(sino, ang, 2.5)
    with pytest.raises(ValueError, match="n_iter"):
        ft.run_sirt(sino, ang, -1)
    out = ft.run_sirt(sino, ang, np.int64(2), Nz=np.int64(SIZE))   # numpy integers work
    assert out.shape == (NY, SIZE, SIZE)


# ── deprecated aliases: warn, keep working, clash with the new name ────────

@pytest.mark.parametrize("name", ["run_sirt", "run_sart", "run_fista", "run_fista_tv"])
def test_volume_alias_of_warm_start(name):
    sino, ang = _data("cpu")
    warm = torch.full((NY, SIZE, SIZE), 0.1)
    torch.manual_seed(0)
    with pytest.warns(DeprecationWarning, match="'volume' is deprecated, use 'warm_start'"):
        old = _call(name, sino, ang, volume=warm)
    torch.manual_seed(0)
    assert torch.equal(old, _call(name, sino, ang, warm_start=warm))
    with pytest.raises(TypeError, match="both"):
        _call(name, sino, ang, volume=warm, warm_start=warm)


@pytest.mark.skipif(not HAS_CUDA, reason="RESIRE is CUDA-only")
def test_resire_support_alias():
    sino, ang = _data("cuda")
    M = torch.ones(1, SIZE, SIZE, device="cuda")
    with pytest.warns(DeprecationWarning, match="'support' is deprecated"):
        ft.run_resire(sino, ang, 2, Nz=SIZE, support=M)
    with pytest.raises(ValueError, match="mutually exclusive"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ft.run_resire(sino, ang, 2, Nz=SIZE, support=M, mask=True)


def test_mbir_aliases():
    sino, ang = _data("cpu")
    with pytest.warns(DeprecationWarning, match="'outer_iters' is deprecated"):
        a = ft.run_mbir(sino, ang, Nz=SIZE, outer_iters=2)
    assert torch.equal(a, ft.run_mbir(sino, ang, 2, Nz=SIZE))
    with pytest.warns(DeprecationWarning, match="return_history is deprecated"):
        vol, d, sigma, hist = ft.run_mbir(sino, ang, 2, Nz=SIZE, return_history=True)
    vol2, loss, d2, sigma2 = ft.run_mbir(sino, ang, 2, Nz=SIZE, return_loss=True,
                                         return_state=True)
    assert torch.equal(vol, vol2) and torch.equal(d, d2) and sigma == sigma2
    assert isinstance(hist, list) and loss.shape == (2,)
    with pytest.raises(TypeError):
        ft.run_mbir(sino, ang, 2, Nz=SIZE, return_history=True, return_loss=True)


def test_constraint_config_aliases():
    M = torch.ones(1, 4, 4)
    with pytest.warns(DeprecationWarning, match="ConstraintConfig.mask is deprecated"):
        c = ft.ConstraintConfig(mask=True)
    assert c.cylinder and c.support_mask is None
    with pytest.warns(DeprecationWarning):
        c = ft.ConstraintConfig(mask=M, sparsity=0.1)
    assert c.support_mask is M and c.soft_threshold == 0.1 and not c.cylinder
    with pytest.warns(DeprecationWarning):
        assert c.sparsity == 0.1 and c.mask is M
    with pytest.warns(DeprecationWarning):
        c.mask = True
    assert c.cylinder and c.support_mask is None
    with pytest.warns(DeprecationWarning):
        c.sparsity = 0.2
    assert c.soft_threshold == 0.2
    with pytest.raises(TypeError):
        ft.ConstraintConfig(True, True)            # everything after positivity is keyword-only
    with pytest.raises(TypeError), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ft.ConstraintConfig(sparsity=0.1, soft_threshold=0.2)


def test_constraint_config_apply_honours_cylinder():
    """apply() used to ignore a bool mask silently."""
    v = torch.ones(2, 8, 8)
    out = ft.ConstraintConfig(cylinder=True).apply(v)
    torch.testing.assert_close(out, v * ft.cylindrical_mask(8, 8).unsqueeze(0))
    with pytest.raises(ValueError, match="mutually exclusive"):
        ft.ConstraintConfig(cylinder=True, support_mask=torch.ones(8, 8)).validate()


def test_upsample_factor_removed():
    sino, ang = _data("cpu")
    vol = ft.run_fbp(sino, ang, Nz=SIZE)
    with pytest.raises(TypeError, match="upsample_factor was removed"):
        ft.estimate_shifts_xcorr(vol, sino, ang, upsample_factor=10)


# ── MBIR: support, config, warm start ──────────────────────────────────────

@pytest.mark.parametrize("levels", [1, 2])
def test_mbir_support_keeps_the_error_sinogram_consistent(levels):
    """The mask is applied after each ICD sweep and the removed part is
    forward-projected into e, so the recorded loss equals the data MSE
    recomputed from the returned volume and offsets."""
    sino, ang = _data("cpu")
    M = torch.zeros(SIZE, SIZE)
    M[3:13, 3:13] = 1.0
    vol, loss, d, _ = ft.run_mbir(sino, ang, 4, Nz=SIZE, multires_levels=levels, support_mask=M,
                                  return_loss=True, return_state=True, sigma_f=0.1)
    assert (vol[:, M == 0] == 0).all()
    a = ang * math.pi / 180
    e = sino - algo._mbir_forward_project(vol, SIZE, torch.cos(a), torch.sin(a)) - d.view(-1, 1, 1)
    assert loss[-1].item() == pytest.approx(e.pow(2).mean().item(), rel=1e-5)
    # an all-ones support is a no-op
    assert torch.equal(ft.run_mbir(sino, ang, 3, Nz=SIZE, multires_levels=levels,
                                   support_mask=torch.ones(SIZE, SIZE)),
                       ft.run_mbir(sino, ang, 3, Nz=SIZE, multires_levels=levels))


def test_mbir_config():
    sino, ang = _data("cpu")
    a = ft.run_mbir(sino, ang, 2, Nz=SIZE, huber_T=3.0, sigma_f=0.5)
    b = ft.run_mbir(sino, ang, 2, Nz=SIZE, config=ft.MbirConfig(huber_T=3.0, sigma_f=0.5))
    assert torch.equal(a, b)
    with pytest.raises(TypeError, match="either through config or as keywords"):
        ft.run_mbir(sino, ang, 2, Nz=SIZE, huber_T=3.0, config=ft.MbirConfig())
    defaults = inspect.signature(ft.run_mbir).parameters
    for f in ft.MbirConfig.__dataclass_fields__:
        default = getattr(ft.MbirConfig(), f)
        assert defaults[f].default == default or (default != default), f


# ── misc public surface ────────────────────────────────────────────────────

def test_euler_zyz_to_rotmat_public():
    from fasttomo.autograd import _euler_zyz_to_rotmat
    assert ft.euler_zyz_to_rotmat is _euler_zyz_to_rotmat
    R = ft.euler_zyz_to_rotmat(torch.tensor([[10.0, 30.0, -5.0]]))
    torch.testing.assert_close(R[0] @ R[0].T, torch.eye(3), atol=1e-6, rtol=0)


def test_configs_accept_numpy_and_tensor_scalars():
    """#82 review N1: parameter sweeps built with numpy pass numpy scalars."""
    ft.OptimConfig(n_iter=np.int64(10), lr_volume=np.float32(0.01)).validate()
    ft.OptimConfig(lr_volume=torch.tensor(0.01)).validate()
    ft.PoseRefinementConfig(refine_pose=True, lr_angles=np.float32(0.01),
                            pose_warmup_iters=np.int64(2))
    ft.RegularizerConfig(tv_weight=np.float16(0.1)).validate()
    with pytest.raises(ValueError):
        ft.OptimConfig(n_iter=True).validate()
    with pytest.raises(ValueError):
        ft.OptimConfig(lr_volume=np.float32("nan")).validate()


def test_loss_recorder_skips_the_loss_when_nobody_wants_it():
    rec = algo._LossRecorder(False, None, False, 3)
    assert not rec.wants_loss
    rec.record(0, torch.tensor(1.0))
    assert rec.losses == []
    rec = algo._LossRecorder(True, None, False, 3)
    rec.record(0, torch.tensor(2.0))
    assert rec.losses == [2.0]
    with pytest.raises(TypeError, match="callable"):
        algo._LossRecorder(False, "not callable", False, 3)


# ── SIRT CUDA-graph path with progress=True (#85 review follow-up) ─────────

@needs_cuda
def test_sirt_graph_path_taken_with_progress(capsys):
    """progress=True keeps run_sirt on the CUDA-graph fast path (the bar
    ticks once per replay through ``on_replay``) and gives the eager loop's
    result bit for bit at omega=1 (``return_loss=True`` forces the eager
    loop)."""
    sino, ang = _data("cuda")
    proj = ft.ParallelProjector((NY, SIZE, SIZE), ang, device="cuda")
    graph = ft.run_sirt(sino, ang, 5, Nz=SIZE, projector=proj, progress=True)
    assert len(proj._sirt_graph_cache) == 1        # captured, not the eager loop
    err = capsys.readouterr().err
    assert "run_sirt" in err and "5/5" in err      # the bar reached the end
    eager, _ = ft.run_sirt(sino, ang, 5, Nz=SIZE, return_loss=True)
    assert torch.equal(graph, eager)


# ── dtype capability tables agree ──────────────────────────────────────────

def test_conftest_dtype_caps_match_algos():
    """conftest.DTYPE_CAPS (used by the ``dtype`` fixture) and ALGOS
    advertise the same dtypes and backends for every algorithm."""
    for name, caps in ALGOS.items():
        backends = ("cuda",) if caps.cuda_only else ("cpu", "cuda")
        assert set(DTYPE_CAPS[name]) == set(backends), name
        for be in backends:
            assert set(DTYPE_CAPS[name][be]) == set(caps.dtypes), (name, be)


# ── public import surface (ARCH P8 package split) ──────────────────────────
#
# Public names of ``fasttomo`` at main 2a58dc8 (before fasttomo/algorithms.py
# became a package). The split must neither drop nor add a public name.
FASTTOMO_PUBLIC_BEFORE_SPLIT = [
    "AdReconConfig", "ConstraintConfig", "MbirConfig", "OptimConfig",
    "ParallelProjector", "PoseRefinementConfig", "RegularizerConfig",
    "RotmatProjector", "algorithms", "apply_shifts", "apply_shifts_2d",
    "autograd", "back_project", "back_project_3d", "back_project_resire",
    "configs", "constraints", "cylindrical_mask", "estimate_shifts_xcorr",
    "euler_zyz_to_rotmat", "forward_project", "forward_project_3d",
    "forward_project_resire", "logsum_sparsity", "projection", "projector",
    "ramp_filter", "refine_angles", "regularizers", "run_ad_recon",
    "run_fbp", "run_fista", "run_fista_tv", "run_genfire",
    "run_grad_descent", "run_mbir", "run_resire", "run_sart", "run_sirt",
    "run_weighted_backprojection", "run_wsirt", "s3im_loss",
    "tilt_ssim_loss", "torch", "tv_loss_3d", "tv_loss_xz", "utilities",
]

# Every name ``fasttomo.algorithms`` exposed as a module at 2a58dc8, except
# the stdlib / third-party module aliases it happened to import (torch, np,
# F, math, typing helpers, ...). Tests, benchmarks and the desktop app may
# import any of these from ``fasttomo.algorithms``.
ALGORITHMS_NAMES_BEFORE_SPLIT = [
    "AdReconConfig", "ConstraintConfig", "MbirConfig", "OptimConfig",
    "ParallelProjector", "PoseRefinementConfig", "RegularizerConfig",
    "RotmatProjector", "_AdRun", "_GENFIRE_RES_EXTENSION_PIVOTS",
    "_LIPSCHITZ_SAFETY", "_LossRecorder", "_MBIR_BLANK_STD",
    "_MBIR_CONVERGENCE_STRIDE", "_MBIR_SIGMA_MIN_RATIO",
    "_MBIR_WEIGHT_EPS", "_MbirState", "_NORM_EPS", "_REL_EPS",
    "_ReconSetup", "_S3IM_SEED", "_TV_DUAL_STEP", "_WBP_CHUNK_VOXELS",
    "_WBP_MIN_THRESHOLD", "_ad_loss_terms", "_ad_project_constraints",
    "_ad_recon_active_steps", "_ad_recon_step_schedule",
    "_ad_sparsity_term", "_ad_step", "_as_float", "_as_int",
    "_assert_dtype_match", "_assert_projector_output_shape",
    "_assert_warm_start_shape", "_bp_gather", "_build_fp_bp",
    "_callback_accepts_components", "_check_inputs", "_check_shapes",
    "_compute_exact_wf", "_core", "_data_ptr", "_deprecated_kwarg",
    "_downsample_sino_det", "_drain_deferred_handles", "_emit_loss",
    "_estimate_lipschitz", "_finish_iters", "_fista_loop",
    "_generate_permutation", "_genfire_hermitian_sym",
    "_huber_surr_weight", "_is_default", "_launch", "_log", "_loss_float",
    "_make_wbp", "_maybe_cylindrical_mask", "_mbir_forward_project",
    "_mbir_icd_sweep_cpu", "_mbir_initial_residual",
    "_mbir_update_offset_variance", "_prepare_ad_recon_pose_init",
    "_prepare_recon", "_project_dual_ball", "_reject_unknown_kwargs",
    "_resize_volume_zx", "_resolve_num_threads", "_resolve_support_mask",
    "_run_mbir_single_scale", "_run_sirt_cuda_graph", "_sirt_iter_inplace",
    "_soft_threshold", "_tv_div", "_tv_grad", "_tv_prox_3d",
    "_wbp_backproject", "_wbp_margin", "apply_shifts_2d", "back_project",
    "back_project_3d", "back_project_resire", "cylindrical_mask",
    "euler_zyz_to_rotmat", "forward_project", "forward_project_3d",
    "forward_project_resire", "ramp_filter", "refine_angles",
    "run_ad_recon", "run_fbp", "run_fista", "run_fista_tv", "run_genfire",
    "run_grad_descent", "run_mbir", "run_resire", "run_sart", "run_sirt",
    "run_weighted_backprojection", "run_wsirt", "s3im_loss",
    "tilt_ssim_loss", "tv_loss_3d", "tv_loss_xz",
]


def test_fasttomo_public_names_unchanged():
    public = sorted(n for n in dir(ft) if not n.startswith("_"))
    assert public == sorted(FASTTOMO_PUBLIC_BEFORE_SPLIT)
    assert sorted(ft.__all__) == sorted(n for n in FASTTOMO_PUBLIC_BEFORE_SPLIT
                                        if n not in ("algorithms", "autograd", "configs",
                                                     "constraints", "projection", "projector",
                                                     "regularizers", "torch", "utilities"))


def test_algorithms_package_reexports_old_names():
    missing = [n for n in ALGORITHMS_NAMES_BEFORE_SPLIT if not hasattr(algo, n)]
    assert missing == []


def test_algorithms_public_objects_live_in_submodules():
    """The re-export is the same object the submodule defines."""
    import fasttomo.algorithms.ad as ad
    import fasttomo.algorithms.algebraic as algebraic
    import fasttomo.algorithms.alignment as alignment
    import fasttomo.algorithms.analytic as analytic
    import fasttomo.algorithms.fista as fista
    import fasttomo.algorithms.genfire as genfire
    import fasttomo.algorithms.mbir as mbir
    import fasttomo.algorithms.resire as resire
    owners = {
        "run_fbp": analytic, "run_weighted_backprojection": analytic,
        "run_sirt": algebraic, "run_sart": algebraic, "run_wsirt": algebraic,
        "run_fista": fista, "run_fista_tv": fista, "run_genfire": genfire,
        "run_resire": resire, "run_grad_descent": ad, "run_ad_recon": ad,
        "run_mbir": mbir, "refine_angles": alignment,
    }
    assert sorted(owners) == sorted(algo.__all__)
    for name, mod in owners.items():
        assert getattr(algo, name) is getattr(mod, name) is getattr(ft, name), name


def test_old_private_import_paths_still_resolve():
    """Helpers moved to fasttomo._backend / projection / regularizers keep
    their old import paths (benchmarks and tests import some of them)."""
    import fasttomo._backend as be
    from fasttomo import autograd, projection, projector, regularizers, utilities
    for name in ("_on_device", "_assert_cuda", "_ensure_fp32", "_assert_dtype_match",
                 "_resolve_num_threads", "_io_dtype_code", "_stream_ptr", "_data_ptr",
                 "_check_shapes"):
        assert getattr(autograd, name) is getattr(be, name), name
    for name in ("_dtype_to_tc_code", "_TC_VOL_LAYERED", "_TC_SINO_LAYERED", "_TC_VOL_3D",
                 "_TC_DTYPE_FP32", "_TC_DTYPE_FP16", "_io_dtype_code"):
        assert getattr(projector, name) is getattr(be, name), name
    assert autograd._euler_zyz_to_rotmat is ft.euler_zyz_to_rotmat
    assert utilities.tv_loss_3d is regularizers.tv_loss_3d is ft.tv_loss_3d
    assert utilities.apply_shifts is projection.apply_shifts is ft.apply_shifts
    assert utilities.apply_shifts_2d is projection.apply_shifts_2d is ft.apply_shifts_2d


def test_utilities_projection_import_cycle_is_gone():
    """projection no longer imports utilities (the cycle ARCH P8 names)."""
    import ast
    import pathlib
    src = pathlib.Path(ft.projection.__file__).read_text()
    imported = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    assert "fasttomo.utilities" not in imported


# ════════════════════════════════════════════════════════════════════════════
# Part 2: loss_callback streaming contract (2026-06-11 handoff)
# ════════════════════════════════════════════════════════════════════════════

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
