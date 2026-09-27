"""
Bit-identity of the frozen algorithms (GENFIRE, RESIRE) and of run_ad_recon
across the section-6 API refactor.

GENFIRE and RESIRE are frozen: the refactor only routed their validation
through `_prepare_recon` and added keyword-only options (loss_callback,
progress, warm_start, support_mask). run_ad_recon's loop was split into
helpers. `tests/data/frozen_golden_main.pt` holds their outputs on
origin/main 0bf45d6 (generator: `tests/data/frozen_golden_main_gen.py`, old
keyword names), except the five run_ad_recon keys, regenerated for later
deliberate AD changes by `tests/data/regenerate_golden_keys.py` (phase 1b:
deterministic grad_rotmat, fused Adam, the antisymmetric_z total
derivative). Here the same calls use the new names and pass every new
option at its no-op value, and must reproduce the golden:

* GENFIRE on CPU and the AD reconstructions are deterministic: torch.equal,
  but only where the machine's arithmetic fingerprint (``tests/_golden.py``)
  is one the golden stores, because the last bit depends on the CPU's SIMD /
  FFT paths and the GPU, not on the torch version: GENFIRE-CPU against the
  CPU fingerprint (elsewhere 1e-5 of the output's largest value, since
  FFT round-off follows the transform's norm), AD against the CUDA fingerprint
  (elsewhere 1e-5), like the RESIRE golden in
  test_algorithms_genfire_resire.py. A fallback is reported as a
  ``GoldenNotExactWarning``. (Made with torch 2.12.1; torch.equal
  re-verified on 2.14.0+cu130, where the fingerprints were recorded.)
* GENFIRE on CUDA (gridding atomics) and RESIRE (forward-projection
  atomics) are not bit-reproducible even main-vs-main: 1e-5 tolerance
  (for GENFIRE relative to the output's largest value).
"""

from pathlib import Path

import pytest
import torch
from _golden import exact_env, peak_close

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


def _check(now, golden, exact: bool, key: str, tol: float = 1e-5, fft: bool = False):
    """Compare ``now`` with ``golden``: bit for bit when ``exact``, else at
    ``tol`` elementwise, or, for FFT-based outputs (``fft=True``: GENFIRE),
    at ``tol`` of the tensor's largest value (``_golden.peak_close``). A
    failure reports the measured difference, not just the key."""
    a, b = _flat(now), _flat(golden)
    assert len(a) == len(b), key
    for i, (x, y) in enumerate(zip(a, b)):
        if not isinstance(x, torch.Tensor):
            assert x == y, f"{key}[{i}]: {x!r} != {y!r}"
        elif exact:
            assert torch.equal(x, y), f"{key}[{i}]: not bit-identical"
        elif fft:
            ok, diff, peak = peak_close(x, y, tol)
            assert ok, (f"{key}[{i}]: max|diff| {diff:.3e} exceeds {tol:.0e} of "
                        f"max(1, max|golden|) with max|golden| {peak:.3e} (shape "
                        f"{tuple(x.shape)} vs {tuple(y.shape)})")
        else:
            torch.testing.assert_close(x, y, rtol=tol, atol=tol,
                                       msg=lambda m, i=i: f"{key}[{i}]: {m}")


def _golden():
    return torch.load(GOLDEN, weights_only=False)


def test_genfire_cpu_identical_to_main():
    golden = _golden()
    exact = exact_env(golden, "cpu", what=GOLDEN.name)
    sino, ang, _, _, N = _inputs("cpu")
    cb = _Calls()
    out = ft.run_genfire(sino, ang, n_iter=12, Nz=N, return_loss=True, loss_callback=cb,
                         progress=False, warm_start=None, support_mask=None)
    _check(out, golden["genfire_cpu"], exact, "genfire_cpu", tol=1e-5, fft=True)
    assert cb.losses == out[1].tolist()          # the callback streams the history
    out = ft.run_genfire(sino, ang, n_iter=8, Nz=N, mask=False, resolution_extension=False,
                         sparsity=1e-3, positivity=False, return_loss=True,
                         loss_callback=_Calls(), warm_start=None, support_mask=None)
    _check(out, golden["genfire_cpu_plain"], exact, "genfire_cpu_plain", tol=1e-5, fft=True)
    out = ft.run_genfire(sino, ang, n_iter=5, Nz=N, oversampling_ratio=3.0, cutoff_dist=0.7,
                         progress=False)
    _check(out, golden["genfire_cpu_os"], exact, "genfire_cpu_os", tol=1e-5, fft=True)


@needs_cuda
def test_genfire_cuda_matches_main():
    golden = _golden()
    sino, ang, _, _, N = _inputs("cuda")
    out = ft.run_genfire(sino, ang, n_iter=12, Nz=N, return_loss=True, loss_callback=_Calls(),
                         progress=False, warm_start=None, support_mask=None)
    _check(out, golden["genfire_cuda"], False, "genfire_cuda", fft=True)


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


def ad_cases() -> dict:
    """The run_ad_recon golden cases (CUDA): key -> zero-argument call
    returning that key's output. Also used by
    tests/data/regenerate_golden_keys.py, so the test and the regeneration
    cannot drift apart."""
    sino, ang, _, sm, N = _inputs("cuda")
    C, Opt, R, K, P = (ft.AdReconConfig, ft.OptimConfig, ft.RegularizerConfig,
                       ft.ConstraintConfig, ft.PoseRefinementConfig)

    def extra():
        return dict(loss_callback=_Calls(), progress=False, warm_start=None)

    return {
        "ad": lambda: ft.run_ad_recon(sino, ang, Nz=N, config=C(optim=Opt(n_iter=6)),
                                      return_loss=True, **extra()),
        "ad_reg": lambda: ft.run_ad_recon(sino, ang, Nz=N, config=C(
            optim=Opt(n_iter=6, loss_fn="l1"),
            reg=R(tv_weight=1e-3, tv_xz_weight=1e-3, sparsity=1e-3),
            constraints=K(positivity=True, cylinder=True, soft_threshold=1e-3)),
            return_loss=True, **extra()),
        "ad_pose": lambda: ft.run_ad_recon(sino, ang, Nz=N, config=C(
            optim=Opt(n_iter=6),
            pose=P(refine_pose=True, refine_shifts=True, in_plane_weight=1e-2,
                   anchor_reference=True)), return_loss=True, return_pose=True, **extra()),
        "ad_staged": lambda: ft.run_ad_recon(sino, ang, Nz=N, config=C(
            optim=Opt(n_iter=8), constraints=K(soft_threshold=1e-3),
            pose=P(refine_shifts=True, pose_warmup_iters=2, alternating_period=2)),
            return_loss=True, return_pose=True, **extra()),
        # the n_iter positional overrides config.optim.n_iter
        "ad_support": lambda: ft.run_ad_recon(sino, ang, 5, Nz=N,
                                              config=C(optim=Opt(n_iter=99)),
                                              support_mask=sm, **extra()),
    }


@needs_cuda
def test_ad_recon_identical_to_main():
    """The five AD keys were regenerated for deliberate changes (phase 1b:
    deterministic grad_rotmat for ad_pose; fused Adam for all five and the
    antisymmetric_z total derivative for ad_pose) with
    tests/data/regenerate_golden_keys.py; they pin that behaviour bit for
    bit where the fingerprint matches."""
    golden = _golden()
    exact = exact_env(golden, "cuda", what=GOLDEN.name)
    for key, call in ad_cases().items():
        _check(call(), golden[key], exact, key)
