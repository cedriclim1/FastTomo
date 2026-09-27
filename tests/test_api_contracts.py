"""
API contracts shared by every reconstruction algorithm (section 6 of the
phase-1 plan; to be folded into test_contracts.py by the test restructure).

One capability table, ``ALGOS``, and parametrised tests that each algorithm
honours what it advertises: ``loss_callback``, ``warm_start``,
``support_mask``, ``progress``, ``return_loss``, the positional order,
dtype acceptance, ``num_threads`` and the deprecated aliases. The
desktop app probes ``inspect.signature(fn).parameters`` for
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

import fasttomo as ft
from fasttomo import algorithms as algo

HAS_CUDA = torch.cuda.is_available()
N, NY, N_ANGLES, N_ITER = 16, 2, 10, 3


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


def _data(device, dtype=torch.float32):
    vol = torch.zeros(NY, N, N, device=device)
    vol[:, 4:12, 5:11] = 1.0
    ang = torch.linspace(-60, 60, N_ANGLES, device=device)
    return ft.forward_project(vol, ang).to(dtype), ang


def _call(name, sino, ang, **kw):
    caps = ALGOS[name]
    kw = {**caps.extra, **kw}
    if caps.iterative:
        kw.setdefault("n_iter", N_ITER)
    return _fn(name)(sino, ang, Nz=N, **kw)


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


@pytest.mark.parametrize("name", ITERATIVE)
def test_loss_callback_and_return_loss(name):
    caps = ALGOS[name]
    sino, ang = _data(_device(caps))
    cb = _Calls()
    vol, hist = _call(name, sino, ang, return_loss=True, loss_callback=cb)
    assert vol.shape == (NY, N, N)
    assert hist.dtype == torch.float32 and hist.device.type == "cpu"
    assert [c[0] for c in cb.calls] == list(range(N_ITER)) == list(range(len(hist)))
    for (_, loss, comps), h in zip(cb.calls, hist.tolist()):
        assert type(loss) is float
        assert loss == pytest.approx(h, rel=1e-6, abs=1e-12)
        assert isinstance(comps, dict) and "fidelity" in comps
        assert sum(comps.values()) == pytest.approx(loss, rel=1e-4, abs=1e-9)


@pytest.mark.parametrize("name", ITERATIVE)
def test_raising_callback_does_not_abort(name):
    caps = ALGOS[name]
    sino, ang = _data(_device(caps))
    seen = []

    def bad(i, loss):
        seen.append(i)
        raise RuntimeError("boom")

    _call(name, sino, ang, loss_callback=bad)
    assert seen == list(range(N_ITER))


@pytest.mark.parametrize("name", ITERATIVE)
def test_callback_does_not_change_the_result(name):
    caps = ALGOS[name]
    sino, ang = _data(_device(caps))
    torch.manual_seed(0)
    a = _call(name, sino, ang)
    torch.manual_seed(0)
    b = _call(name, sino, ang, loss_callback=lambda i, loss: None)
    if name in ("run_resire", "run_genfire") and sino.is_cuda:
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)   # atomics
    else:
        assert torch.equal(a, b)


@pytest.mark.parametrize("name", ITERATIVE)
def test_warm_start(name):
    caps = ALGOS[name]
    device = _device(caps)
    sino, ang = _data(device)
    warm = torch.rand(NY, N, N, generator=torch.Generator().manual_seed(1)).to(device)
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


@pytest.mark.parametrize("name", list(ALGOS))
def test_support_mask(name):
    """Out-of-support voxels are exactly 0, the mask excludes mask=True and a
    wrong shape raises ValueError (the 2026-06 support_mask contract)."""
    caps = ALGOS[name]
    sino, ang = _data(_device(caps))
    M = torch.zeros(N, N, dtype=torch.bool)
    M[3:13, 3:13] = True                           # a CPU bool mask is accepted
    out = _call(name, sino, ang, support_mask=M)
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == (NY, N, N)
    assert (out[:, ~M.to(out.device)] == 0).all()
    assert out[:, M.to(out.device)].abs().max() > 0
    if name != "run_ad_recon":                     # AD: the cylinder lives on the config
        with pytest.raises(ValueError, match="mutually exclusive"):
            _call(name, sino, ang, support_mask=M, mask=True)
    with pytest.raises(ValueError, match="support_mask"):
        _call(name, sino, ang, support_mask=torch.ones(N + 1, N))


@pytest.mark.parametrize("name", ITERATIVE)
def test_progress_bar(name, capsys):
    caps = ALGOS[name]
    sino, ang = _data(_device(caps))
    _call(name, sino, ang, progress=True)
    assert name in capsys.readouterr().err        # tqdm writes to stderr
    _call(name, sino, ang)
    assert capsys.readouterr().err == ""          # off by default


# ── dtypes: accept what is advertised, raise on anything else ──────────────

@pytest.mark.parametrize("name", list(ALGOS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, torch.bfloat16])
def test_dtype_accept_or_raise(name, dtype):
    caps = ALGOS[name]
    sino, ang = _data(_device(caps), dtype)
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
    monkeypatch.setattr(algo, "_resolve_num_threads", spy)
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
    ft.run_mbir(sino, ang, 2, Nz=N, convergence_tol=0.0)
    assert calls == [1, 1]                         # kwarg -> OMP_NUM_THREADS -> 1
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    calls.clear()
    ft.run_mbir(sino, ang, 2, Nz=N, convergence_tol=0.0)
    assert calls == [2, 2]


# ── stale calls fail loudly ────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(ALGOS))
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
    out = ft.run_sirt(sino, ang, np.int64(2), Nz=np.int64(N))   # numpy integers work
    assert out.shape == (NY, N, N)


# ── deprecated aliases: warn, keep working, clash with the new name ────────

@pytest.mark.parametrize("name", ["run_sirt", "run_sart", "run_fista", "run_fista_tv"])
def test_volume_alias_of_warm_start(name):
    sino, ang = _data("cpu")
    warm = torch.full((NY, N, N), 0.1)
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
    M = torch.ones(1, N, N, device="cuda")
    with pytest.warns(DeprecationWarning, match="'support' is deprecated"):
        ft.run_resire(sino, ang, 2, Nz=N, support=M)
    with pytest.raises(ValueError, match="mutually exclusive"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ft.run_resire(sino, ang, 2, Nz=N, support=M, mask=True)


def test_mbir_aliases():
    sino, ang = _data("cpu")
    with pytest.warns(DeprecationWarning, match="'outer_iters' is deprecated"):
        a = ft.run_mbir(sino, ang, Nz=N, outer_iters=2)
    assert torch.equal(a, ft.run_mbir(sino, ang, 2, Nz=N))
    with pytest.warns(DeprecationWarning, match="return_history is deprecated"):
        vol, d, sigma, hist = ft.run_mbir(sino, ang, 2, Nz=N, return_history=True)
    vol2, loss, d2, sigma2 = ft.run_mbir(sino, ang, 2, Nz=N, return_loss=True,
                                         return_state=True)
    assert torch.equal(vol, vol2) and torch.equal(d, d2) and sigma == sigma2
    assert isinstance(hist, list) and loss.shape == (2,)
    with pytest.raises(TypeError):
        ft.run_mbir(sino, ang, 2, Nz=N, return_history=True, return_loss=True)


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
    vol = ft.run_fbp(sino, ang, Nz=N)
    with pytest.raises(TypeError, match="upsample_factor was removed"):
        ft.estimate_shifts_xcorr(vol, sino, ang, upsample_factor=10)


# ── MBIR: support, config, warm start ──────────────────────────────────────

@pytest.mark.parametrize("levels", [1, 2])
def test_mbir_support_keeps_the_error_sinogram_consistent(levels):
    """The mask is applied after each ICD sweep and the removed part is
    forward-projected into e, so the recorded loss equals the data MSE
    recomputed from the returned volume and offsets."""
    sino, ang = _data("cpu")
    M = torch.zeros(N, N)
    M[3:13, 3:13] = 1.0
    vol, loss, d, _ = ft.run_mbir(sino, ang, 4, Nz=N, multires_levels=levels, support_mask=M,
                                  return_loss=True, return_state=True, sigma_f=0.1)
    assert (vol[:, M == 0] == 0).all()
    a = ang * math.pi / 180
    e = sino - algo._mbir_forward_project(vol, N, torch.cos(a), torch.sin(a)) - d.view(-1, 1, 1)
    assert loss[-1].item() == pytest.approx(e.pow(2).mean().item(), rel=1e-5)
    # an all-ones support is a no-op
    assert torch.equal(ft.run_mbir(sino, ang, 3, Nz=N, multires_levels=levels,
                                   support_mask=torch.ones(N, N)),
                       ft.run_mbir(sino, ang, 3, Nz=N, multires_levels=levels))


def test_mbir_config():
    sino, ang = _data("cpu")
    a = ft.run_mbir(sino, ang, 2, Nz=N, huber_T=3.0, sigma_f=0.5)
    b = ft.run_mbir(sino, ang, 2, Nz=N, config=ft.MbirConfig(huber_T=3.0, sigma_f=0.5))
    assert torch.equal(a, b)
    with pytest.raises(TypeError, match="either through config or as keywords"):
        ft.run_mbir(sino, ang, 2, Nz=N, huber_T=3.0, config=ft.MbirConfig())
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
