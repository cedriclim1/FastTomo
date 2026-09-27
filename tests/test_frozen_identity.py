"""
Bit-identity of the frozen algorithms (GENFIRE, RESIRE) and of run_ad_recon
across the section-6 API refactor.

GENFIRE and RESIRE are frozen: the refactor only routed their validation
through `_prepare_recon` and added keyword-only options (loss_callback,
progress, warm_start, support_mask). run_ad_recon's loop was split into
helpers. `tests/data/frozen_golden_main.pt` holds their outputs on
origin/main 0bf45d6 (generator: `tests/data/frozen_golden_main_gen.py`, old
keyword names). Here the same calls use the new names and pass every new
option at its no-op value, and must reproduce the golden:

* GENFIRE on CPU and the AD reconstructions are deterministic: torch.equal,
  but only in the environment the golden was made on, because the result
  depends on the torch FFT / reduction backends: GENFIRE-CPU on Linux x86_64
  with torch 2.12 (elsewhere 1e-6), AD on sm_120 with torch 2.12 (elsewhere
  1e-5), like the RESIRE golden in test_algorithms_genfire_resire.py.
* GENFIRE on CUDA (gridding atomics) and RESIRE (forward-projection
  atomics) are not bit-reproducible even main-vs-main: 1e-5 tolerance.
"""

import platform
from pathlib import Path

import pytest
import torch

import fasttomo as ft

GOLDEN = Path(__file__).parent / "data" / "frozen_golden_main.pt"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU available")


def _inputs(device):
    g = torch.Generator().manual_seed(5)
    Ny, N, n = 3, 20, 13
    vol = torch.zeros(Ny, N, N)
    vol[:, 5:15, 6:13] = 1.0
    vol[1, 8:11, 8:12] = 2.0
    vol += 0.1 * torch.rand(Ny, N, N, generator=g)
    ang = torch.linspace(-65, 65, n)
    eul = torch.stack([torch.linspace(-3, 3, n), ang, torch.linspace(2, -2, n)], 1)
    sm = torch.zeros(1, N, N)
    sm[:, 3:17, 3:17] = 1.0
    vol, ang, sm = vol.to(device), ang.to(device), sm.to(device)
    return ft.forward_project(vol, ang), ang, eul, sm, N


class _Calls:
    def __init__(self):
        self.losses = []

    def __call__(self, i, loss):
        self.losses.append(loss)


def _flat(v):
    if isinstance(v, tuple):
        return [x for e in v for x in _flat(e)]
    return [v.detach().cpu() if isinstance(v, torch.Tensor) else v]


def _check(now, golden, exact: bool, key: str, tol: float = 1e-5):
    a, b = _flat(now), _flat(golden)
    assert len(a) == len(b), key
    for x, y in zip(a, b):
        if not isinstance(x, torch.Tensor):
            assert x == y, key
        elif exact:
            assert torch.equal(x, y), key
        else:
            assert x.shape == y.shape, key
            torch.testing.assert_close(x, y, rtol=tol, atol=tol, msg=key)


def _exact_env() -> bool:
    return torch.cuda.get_device_capability() == (12, 0) and torch.__version__.startswith("2.12.")


def _genfire_cpu_exact_env() -> bool:
    """The environment frozen_golden_main.pt's GENFIRE-CPU keys were made in
    (torch 2.12 CPU FFT on Linux x86_64). Elsewhere the comparison falls back
    to 1e-6 (#85 review N7)."""
    return (torch.__version__.startswith("2.12.") and platform.system() == "Linux"
            and platform.machine() == "x86_64")


def _golden():
    return torch.load(GOLDEN, weights_only=False)


def test_genfire_cpu_identical_to_main():
    golden = _golden()
    exact = _genfire_cpu_exact_env()
    sino, ang, _, _, N = _inputs("cpu")
    cb = _Calls()
    out = ft.run_genfire(sino, ang, n_iter=12, Nz=N, return_loss=True, loss_callback=cb,
                         progress=False, warm_start=None, support_mask=None)
    _check(out, golden["genfire_cpu"], exact, "genfire_cpu", tol=1e-6)
    assert cb.losses == out[1].tolist()          # the callback streams the history
    out = ft.run_genfire(sino, ang, n_iter=8, Nz=N, mask=False, resolution_extension=False,
                         sparsity=1e-3, positivity=False, return_loss=True,
                         loss_callback=_Calls(), warm_start=None, support_mask=None)
    _check(out, golden["genfire_cpu_plain"], exact, "genfire_cpu_plain", tol=1e-6)
    out = ft.run_genfire(sino, ang, n_iter=5, Nz=N, oversampling_ratio=3.0, cutoff_dist=0.7,
                         progress=False)
    _check(out, golden["genfire_cpu_os"], exact, "genfire_cpu_os", tol=1e-6)


@needs_cuda
def test_genfire_cuda_matches_main():
    golden = _golden()
    sino, ang, _, _, N = _inputs("cuda")
    out = ft.run_genfire(sino, ang, n_iter=12, Nz=N, return_loss=True, loss_callback=_Calls(),
                         progress=False, warm_start=None, support_mask=None)
    _check(out, golden["genfire_cuda"], False, "genfire_cuda")


@needs_cuda
def test_resire_matches_main():
    golden = _golden()
    sino, ang, eul, sm, N = _inputs("cuda")
    cb = _Calls()
    out = ft.run_resire(sino, ang, n_iter=6, Nz=N, return_loss=True, loss_callback=cb,
                        progress=False, warm_start=None, support_mask=None)
    _check(out, golden["resire"], False, "resire")
    assert cb.losses == out[1].tolist()
    out = ft.run_resire(sino, ang, n_iter=4, Nz=N, support_mask=sm, loss_callback=_Calls())
    _check(out, golden["resire_support"], False, "resire_support")
    out = ft.run_resire(sino, ang, n_iter=4, Nz=N, mask=True, sparsity=1e-3, positivity=False,
                        step_size=0.5, warm_start=None)
    _check(out, golden["resire_mask"], False, "resire_mask")
    out = ft.run_resire(sino, ang, n_iter=4, Nz=N, euler_zyz_deg=eul, return_loss=True,
                        loss_callback=_Calls(), progress=False)
    _check(out, golden["resire_euler"], False, "resire_euler")


@needs_cuda
def test_ad_recon_identical_to_main():
    golden = _golden()
    exact = _exact_env()
    sino, ang, _, sm, N = _inputs("cuda")
    C, Opt, R, K, P = (ft.AdReconConfig, ft.OptimConfig, ft.RegularizerConfig,
                     ft.ConstraintConfig, ft.PoseRefinementConfig)
    extra = dict(loss_callback=_Calls(), progress=False, warm_start=None)
    out = ft.run_ad_recon(sino, ang, Nz=N, config=C(optim=Opt(n_iter=6)), return_loss=True, **extra)
    _check(out, golden["ad"], exact, "ad")
    out = ft.run_ad_recon(sino, ang, Nz=N, config=C(
        optim=Opt(n_iter=6, loss_fn="l1"),
        reg=R(tv_weight=1e-3, tv_xz_weight=1e-3, sparsity=1e-3),
        constraints=K(positivity=True, cylinder=True, soft_threshold=1e-3)),
        return_loss=True, **extra)
    _check(out, golden["ad_reg"], exact, "ad_reg")
    out = ft.run_ad_recon(sino, ang, Nz=N, config=C(
        optim=Opt(n_iter=6),
        pose=P(refine_pose=True, refine_shifts=True, in_plane_weight=1e-2,
               anchor_reference=True)), return_loss=True, return_pose=True, **extra)
    _check(out, golden["ad_pose"], exact, "ad_pose")
    out = ft.run_ad_recon(sino, ang, Nz=N, config=C(
        optim=Opt(n_iter=8), constraints=K(soft_threshold=1e-3),
        pose=P(refine_shifts=True, pose_warmup_iters=2, alternating_period=2)),
        return_loss=True, return_pose=True, **extra)
    _check(out, golden["ad_staged"], exact, "ad_staged")
    # the n_iter positional overrides config.optim.n_iter
    out = ft.run_ad_recon(sino, ang, 5, Nz=N, config=C(optim=Opt(n_iter=99)), support_mask=sm,
                          **extra)
    _check(out, golden["ad_support"], exact, "ad_support")
