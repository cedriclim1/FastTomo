"""
MBIR (run_mbir): ICD reconstruction, anomaly modelling, offsets and
variance, multires, input validation and the raw sweep bindings.

CPU <-> CUDA agreement lives in test_parity.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from _golden import exact_env, tensors
from conftest import (
    make_angles,
    needs_cuda,
)

import fasttomo as ft


# The CPU MBIR reference exists for two reasons:
#   1. validation when no GPU is present
#   2. a numerically-comparable target for the CUDA kernel
#
# `test_mbir_cpu_runs` exercises the C kernel and ensures the algorithm
# converges on the CPU path. The CPU <-> CUDA consistency tests live in
# test_parity.py and are skipped without CUDA.
def _mbir_sino_cpu(phantom: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """CPU voxel-driven bilinear FP (the same model used inside CPU ICD)."""
    from fasttomo.autograd import _launch, _trig_on_device
    Ny, Nz, Nx = phantom.shape
    N_det = Nx
    n_angles = angles_deg.numel()
    vol = phantom.float().contiguous()
    cos_a, sin_a = _trig_on_device(angles_deg, vol.device)
    sino = torch.empty(n_angles, Ny, N_det)
    return _launch("mbir_forward_project", sino,
                   vol, sino, cos_a, sin_a, Ny, Nz, Nx, n_angles, N_det,
                   shapes=[("volume", (Ny, Nz, Nx)), ("sino", (n_angles, Ny, N_det)),
                           ("cos", (n_angles,)), ("sin", (n_angles,))],
                   num_threads=1)


# All MBIR tests use the project's own voxel-driven bilinear scatter forward
# model (`mbir_forward_project_cuda`) to generate the input sinogram. This
# is the same A used inside the ICD column updates, so the algorithm is
# being tested as a self-adjoint MAP estimator under its own forward
# model — the cleanest possible correctness check.
def _mbir_sino_self(phantom: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """Forward-project using MBIR's own A (voxel-driven bilinear scatter)
    rather than FastTomo's ray-marching `forward_project`. This is what
    the ICD column updates are inverting, so tests using this FP can
    expect near-perfect recovery of the input phantom."""
    import math

    import fasttomo._core as _core
    Ny, Nz, Nx = phantom.shape
    n_angles = angles_deg.shape[0]
    N_det = Nx
    sino = torch.zeros(n_angles, Ny, N_det, device=phantom.device, dtype=torch.float32)
    angles_rad = (angles_deg.float().to(phantom.device) * (math.pi / 180.0)).contiguous()
    cos_a = torch.cos(angles_rad).contiguous()
    sin_a = torch.sin(angles_rad).contiguous()
    _core.mbir_forward_project_cuda(
        phantom.contiguous().data_ptr(), sino.data_ptr(),
        cos_a.data_ptr(), sin_a.data_ptr(),
        Ny, Nz, Nx, n_angles, N_det,
        torch.cuda.current_stream(phantom.device).cuda_stream,
    )
    return sino


# ── end to end ────────────────────────────────────────────────────────────────


def test_mbir_cpu_runs():
    """CPU MBIR end-to-end on a tiny phantom; assert convergence."""
    Ny = Nz = Nx = 16
    n_angles = 24
    phantom = torch.zeros(Ny, Nz, Nx)
    phantom[4:12, 4:12, 4:12] = 1.0
    angles = make_angles(n_angles, max_deg=60.0)
    sino = _mbir_sino_cpu(phantom, angles)
    vol = ft.run_mbir(
        sino, angles,
        sigma_f=0.1, huber_T=float("inf"),
        estimate_offset=False, estimate_variance=True,
        n_iter=30, multires_levels=1,
    )
    rel = (vol - phantom).norm().item() / phantom.norm().item()
    assert rel < 0.30, f"CPU MBIR rel error {rel:.4f} > 0.30"


DEVICE = torch.device("cuda")


@needs_cuda
def test_mbir_perfect_data_recovers_phantom():
    """Conventional MBIR (T=∞, no anomaly modeling) on a noiseless solid
    cube should recover the phantom to high accuracy. Validates the
    surrogate-MM ICD inner loop + qGGMRF prior + variance auto-tuning."""
    Ny = Nz = Nx = 32
    n_angles = 40
    phantom = torch.zeros(Ny, Nz, Nx, device=DEVICE)
    phantom[8:24, 8:24, 8:24] = 1.0
    angles = make_angles(n_angles, max_deg=70.0, device=DEVICE)

    sino = _mbir_sino_self(phantom, angles)
    vol = ft.run_mbir(
        sino, angles,
        sigma_f=0.1,
        huber_T=float("inf"),
        estimate_offset=False,
        estimate_variance=True,
        n_iter=80,
        multires_levels=1,
    )
    rel_err = (vol - phantom).norm().item() / phantom.norm().item()
    assert rel_err < 0.10, f"MBIR rel error {rel_err:.4f} > 0.10"


@needs_cuda
def test_mbir_pe_rejects_bragg_streak():
    """The whole point of MBIR-AM-PE: with anomaly modeling, a single
    high-magnitude Bragg-style streak in one tilt's detector strip is
    suppressed in the reconstruction. Without anomaly modeling, the
    streak corrupts the volume.

    We assert MBIR-AM has lower RMSE than conventional MBIR on the
    Bragg-corrupted data — the qualitative paper result, scaled to
    32³ so it runs in a few seconds. """
    Ny = Nz = Nx = 32
    n_angles = 40
    phantom = torch.zeros(Ny, Nz, Nx, device=DEVICE)
    phantom[8:24, 8:24, 8:24] = 1.0
    angles = make_angles(n_angles, max_deg=70.0, device=DEVICE)

    sino_clean = _mbir_sino_self(phantom, angles)

    # Inject a Bragg-like streak: at one tilt, multiply a 4-pixel-wide
    # detector strip by 5× across all Y rows. This corrupts that
    # projection at known coordinates.
    sino_corrupt = sino_clean.clone()
    angle_idx = n_angles // 3
    sino_corrupt[angle_idx, :, 14:18] *= 5.0

    # Conventional MBIR — no anomaly handling.
    vol_conv = ft.run_mbir(
        sino_corrupt, angles,
        sigma_f=0.1, huber_T=float("inf"),
        estimate_offset=False, estimate_variance=True,
        n_iter=80, multires_levels=1,
    )
    err_conv = (vol_conv - phantom).norm().item()

    # MBIR-AM-PE — Bragg anomaly modeling on.
    vol_am = ft.run_mbir(
        sino_corrupt, angles,
        sigma_f=0.1, huber_T=3.0, huber_delta=0.5,
        estimate_offset=False, estimate_variance=True,
        n_iter=80, multires_levels=1,
    )
    err_am = (vol_am - phantom).norm().item()

    # AM should beat conventional MBIR on this corrupted data.
    assert err_am < err_conv, (
        f"MBIR-AM error {err_am:.4f} not lower than conventional MBIR "
        f"{err_conv:.4f} — anomaly modeling not effective"
    )


@needs_cuda
def test_mbir_seed_reproducibility():
    """The voxel-update permutation is the only stochastic step. With a
    fixed seed two runs must produce bitwise-identical reconstructions."""
    Ny = Nz = Nx = 16
    angles = make_angles(20, max_deg=60.0, device=DEVICE)
    phantom = torch.zeros(Ny, Nz, Nx, device=DEVICE)
    phantom[4:12, 4:12, 4:12] = 1.0
    sino = _mbir_sino_self(phantom, angles)

    kw = dict(
        sigma_f=0.1, huber_T=float("inf"),
        estimate_offset=False, estimate_variance=False,
        n_iter=20, multires_levels=1, sigma_init=0.1,
    )
    v1 = ft.run_mbir(sino, angles, seed=42, **kw)
    v2 = ft.run_mbir(sino, angles, seed=42, **kw)
    assert torch.equal(v1, v2), "MBIR not reproducible with the same seed"


# ── voxel order (generated in the CUDA kernel, mirrored in NumPy) ─────────────

_PERM_SIZES = [(1, 1), (3, 2), (4, 5), (2, 144), (5, 1000), (3, 4097), (2, 40000)]


@pytest.mark.parametrize("Ny,nvox", _PERM_SIZES)
def test_mbir_permutation_rows_are_distinct_permutations(Ny, nvox):
    """Each slice's order is a permutation of range(nvox) (cycle-walking
    included: 4097 and 40000 are not powers of four), slices and outer
    iterations get different orders, and the order is a pure function of
    (seed, iteration)."""
    from fasttomo.algorithms.mbir import _mbir_perm_key, _mbir_permutation
    p = _mbir_permutation(Ny, nvox, _mbir_perm_key(7, 0))
    assert p.dtype == np.int32 and p.shape == (Ny, nvox)
    for row in p:
        assert np.array_equal(np.sort(row), np.arange(nvox))
    assert np.array_equal(p, _mbir_permutation(Ny, nvox, _mbir_perm_key(7, 0)))
    if nvox >= 16:
        assert not np.array_equal(p, _mbir_permutation(Ny, nvox, _mbir_perm_key(7, 1)))
        assert not np.array_equal(p, _mbir_permutation(Ny, nvox, _mbir_perm_key(8, 0)))
        if Ny > 1:
            assert not np.array_equal(p[0], p[1])


@needs_cuda
@pytest.mark.parametrize("Ny,nvox", _PERM_SIZES)
@pytest.mark.parametrize("seed,it", [(0, 0), (3, 1), (-5, 7), (2**40 + 1, 99)])
def test_mbir_permutation_cuda_matches_numpy(Ny, nvox, seed, it):
    """The order the CUDA sweep builds in the kernel is the NumPy order the
    CPU sweep gets, element for element (so CPU and CUDA visit the voxels
    in the same order and the parity tests compare like with like)."""
    import fasttomo._core as _core
    from fasttomo.algorithms.mbir import _mbir_perm_key, _mbir_permutation
    key = _mbir_perm_key(seed, it)
    out = torch.full((Ny, nvox), -1, dtype=torch.int32, device=DEVICE)
    _core.mbir_permutation_cuda(out_ptr=out.data_ptr(), Ny=Ny, nvox=nvox, perm_key=key,
                                stream_ptr=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert np.array_equal(out.cpu().numpy(), _mbir_permutation(Ny, nvox, key))


@needs_cuda
def test_mbir_cuda_in_kernel_order_matches_explicit_perm():
    """The sweep with perm_ptr=0 (order built in the kernel) equals the sweep
    given the same order as an explicit [Ny, Nz*Nx] array, bit for bit; the
    raw binding and the validated wrapper agree."""
    import fasttomo._core as _core
    from fasttomo.algorithms.mbir import (_mbir_icd_sweep_cuda, _mbir_perm_key,
                                          _mbir_permutation)
    g = torch.Generator().manual_seed(4)
    Ny, Nz, Nx, n = 5, 11, 13, 17
    ang = make_angles(n, max_deg=70.0) * math.pi / 180
    cos_a, sin_a = torch.cos(ang).to(DEVICE), torch.sin(ang).to(DEVICE)
    vol0 = torch.rand(Ny, Nz, Nx, generator=g).to(DEVICE)
    e0 = torch.randn(n, Ny, Nx, generator=g).to(DEVICE)
    lam = (torch.rand(n, Ny, Nx, generator=g) + 0.5).to(DEVICE)
    key = _mbir_perm_key(11, 2)
    perm = torch.from_numpy(_mbir_permutation(Ny, Nz * Nx, key)).to(DEVICE)
    work = torch.empty(Ny * Nz * Nx * 4, device=DEVICE)
    outs = []
    for perm_ptr in (0, perm.data_ptr()):
        vol, e = vol0.clone(), e0.clone()
        _core.mbir_icd_sweep_cuda(
            vol_ptr=vol.data_ptr(), e_sino_ptr=e.data_ptr(), vol_snapshot_ptr=vol0.data_ptr(),
            lambdas_ptr=lam.data_ptr(), perm_ptr=perm_ptr, perm_key=key,
            work_ptr=work.data_ptr(),
            cos_ptr=cos_a.data_ptr(), sin_ptr=sin_a.data_ptr(),
            Ny=Ny, Nz=Nz, Nx=Nx, n_angles=n, N_det=Nx,
            qggmrf_p=1.2, qggmrf_q=2.0, qggmrf_c=1e-3, sigma_f=0.3,
            huber_T=2.0, huber_delta=0.5, sigma=0.7,
            stream_ptr=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        outs.append((vol, e))
    for p in (None, perm):
        vol, e = vol0.clone(), e0.clone()
        _mbir_icd_sweep_cuda(vol, e, lam, vol0, work, cos_a.float(), sin_a.float(),
                             perm_key=key, perm=p, qggmrf_p=1.2, qggmrf_q=2.0, qggmrf_c=1e-3,
                             sigma_f=0.3, huber_T=2.0, huber_delta=0.5, sigma=0.7)
        torch.cuda.synchronize()
        outs.append((vol, e))
    assert not torch.equal(outs[0][0], vol0)
    for v, e in outs[1:]:
        assert torch.equal(outs[0][0], v) and torch.equal(outs[0][1], e)


@needs_cuda
def test_mbir_cuda_explicit_order_is_validated():
    """The explicit-order path (perm_ptr) is checked like every other input:
    dtype, shape, value range, and each row a permutation (#89 review)."""
    from fasttomo.algorithms.mbir import _mbir_icd_sweep_cuda, _mbir_permutation
    Ny, Nz, Nx, n = 2, 3, 4, 5
    vol = torch.zeros(Ny, Nz, Nx, device=DEVICE)
    e = torch.zeros(n, Ny, Nx, device=DEVICE)
    c = torch.ones(n, device=DEVICE)
    work = torch.empty(Ny * Nz * Nx * 4, device=DEVICE)
    good = torch.from_numpy(_mbir_permutation(Ny, Nz * Nx, 7)).to(DEVICE)
    kw = dict(perm_key=7, qggmrf_p=1.2, qggmrf_q=2.0, qggmrf_c=1e-3, sigma_f=1.0,
              huber_T=float("inf"), huber_delta=0.5, sigma=1.0)
    _mbir_icd_sweep_cuda(vol, e, None, vol.clone(), work, c, c, perm=good, **kw)
    dup = good.clone()
    dup[1, 0] = dup[1, 1]
    high = good.clone()
    high[0, 3] = Nz * Nx
    neg = good.clone()
    neg[1, 2] = -1
    for bad, err, match in ((good.long(), TypeError, "int32"),
                            (good[:, :-1].contiguous(), ValueError, "shape"),
                            (high, ValueError, r"\[0, 12\)"),
                            (neg, ValueError, r"\[0, 12\)"),
                            (dup, ValueError, "permutation"),
                            (good.cpu(), TypeError, "cuda")):
        with pytest.raises(err, match=match):
            _mbir_icd_sweep_cuda(vol, e, None, vol.clone(), work, c, c, perm=bad, **kw)
    with pytest.raises(ValueError, match="work"):
        _mbir_icd_sweep_cuda(vol, e, None, vol.clone(), work[:-1], c, c, **kw)


# ── CUDA sweep variants ───────────────────────────────────────────────────────

# 9, 61, 100, 140 and 260 angles take the 1-, 2-, 4- and 8-slot register
# kernels and the generic kernel (> 256 angles).
_SWEEP_ANGLES = (9, 61, 100, 140, 260)
_SWEEP_GOLDEN = Path(__file__).parent / "data" / "mbir_sweep_golden.pt"


def _sweep_golden_cases(pre_restructure: bool = False) -> dict[str, torch.Tensor]:
    """Volume and error sinogram after three CUDA sweeps for every sweep
    variant (see tests/data/mbir_sweep_golden_gen.py). ``pre_restructure``
    calls the binding without ``work_ptr`` (the older signature)."""
    import fasttomo._core as _core
    from fasttomo.algorithms.mbir import _mbir_perm_key
    out = {}
    g = torch.Generator().manual_seed(123)
    Ny, Nz, Nx, Nd = 2, 9, 11, 11
    for n in _SWEEP_ANGLES:
        ang = torch.linspace(-75, 75, n) * math.pi / 180
        cos_a, sin_a = torch.cos(ang).to(DEVICE), torch.sin(ang).to(DEVICE)
        vol0 = torch.rand(Ny, Nz, Nx, generator=g)
        vol0 = (vol0 * (torch.rand(Ny, Nz, Nx, generator=g) > 0.3)).to(DEVICE)
        e0 = torch.randn(n, Ny, Nd, generator=g).to(DEVICE)
        lam_t = (torch.rand(n, Ny, Nd, generator=g) + 0.2).to(DEVICE)
        for use_lam in (False, True):
            for T in (float("inf"), 1.5):
                vol, e = vol0.clone(), e0.clone()
                work = torch.empty(Ny * Nz * Nx * 4, device=DEVICE)
                extra = {} if pre_restructure else {"work_ptr": work.data_ptr()}
                for it in range(3):
                    snap = vol.clone()
                    _core.mbir_icd_sweep_cuda(
                        vol_ptr=vol.data_ptr(), e_sino_ptr=e.data_ptr(),
                        vol_snapshot_ptr=snap.data_ptr(),
                        lambdas_ptr=lam_t.data_ptr() if use_lam else 0,
                        perm_ptr=0, perm_key=_mbir_perm_key(9, it),
                        cos_ptr=cos_a.data_ptr(), sin_ptr=sin_a.data_ptr(),
                        Ny=Ny, Nz=Nz, Nx=Nx, n_angles=n, N_det=Nd,
                        qggmrf_p=1.2, qggmrf_q=2.0, qggmrf_c=1e-3, sigma_f=0.4,
                        huber_T=T, huber_delta=0.5, sigma=0.8,
                        stream_ptr=torch.cuda.current_stream().cuda_stream, **extra)
                    torch.cuda.synchronize()
                key = f"n{n}_lam{int(use_lam)}_T{T}"
                out[key + "/vol"] = vol.cpu()
                out[key + "/e"] = e.cpu()
    return out


@needs_cuda
def test_mbir_cuda_sweep_matches_pre_restructure_golden():
    """The one-warp-per-slice sweep (prepare kernel, register-resident e,
    fused warp reduction) reproduces the previous 64-thread kernel bit for
    bit on sm_120, for every variant: same order, same arithmetic, same
    summation order. On other GPUs the compiler may contract differently, so
    where the CUDA arithmetic fingerprint (tests/_golden.py) is not one the
    golden stores the comparison falls back to 1e-5 (with a
    GoldenNotExactWarning)."""
    raw = torch.load(_SWEEP_GOLDEN, weights_only=True)
    golden = tensors(raw)
    now = _sweep_golden_cases()
    assert set(now) == set(golden)
    exact = exact_env(raw, "cuda", what=_SWEEP_GOLDEN.name)
    bad = [k for k in sorted(now)
           if not (torch.equal(now[k], golden[k]) if exact
                   else torch.allclose(now[k], golden[k], rtol=1e-5, atol=1e-5))]
    assert not bad, f"differ from the pre-restructure sweep (exact={exact}): {bad}"


@needs_cuda
def test_mbir_cuda_sweep_requires_work_buffer():
    import fasttomo._core as _core
    v = torch.zeros(1, 2, 2, device=DEVICE)
    e = torch.zeros(3, 1, 2, device=DEVICE)
    c = torch.ones(3, device=DEVICE)
    with pytest.raises(ValueError, match="work buffer"):
        _core.mbir_icd_sweep_cuda(
            vol_ptr=v.data_ptr(), e_sino_ptr=e.data_ptr(), vol_snapshot_ptr=v.data_ptr(),
            lambdas_ptr=0, perm_ptr=0, perm_key=0, work_ptr=0,
            cos_ptr=c.data_ptr(), sin_ptr=c.data_ptr(),
            Ny=1, Nz=2, Nx=2, n_angles=3, N_det=2,
            qggmrf_p=1.2, qggmrf_q=2.0, qggmrf_c=1e-3, sigma_f=1.0,
            huber_T=float("inf"), huber_delta=0.5, sigma=1.0)


@needs_cuda
def test_mbir_estimate_offset_disabled_keeps_d_zero():
    """When estimate_offset=False, the per-tilt offset must remain at its
    initial value (zero). The kernel still receives the offset pointer but
    the host wrapper bypasses the closed-form update."""
    Ny = Nz = Nx = 16
    angles = make_angles(20, max_deg=60.0, device=DEVICE)
    phantom = torch.zeros(Ny, Nz, Nx, device=DEVICE)
    phantom[4:12, 4:12, 4:12] = 1.0
    sino = _mbir_sino_self(phantom, angles)

    _, d_offsets, _ = ft.run_mbir(
        sino, angles,
        sigma_f=0.1, huber_T=float("inf"),
        estimate_offset=False, estimate_variance=False,
        n_iter=10, multires_levels=1, sigma_init=0.1,
        return_state=True,
    )
    assert torch.all(d_offsets == 0.0), (
        f"d_offsets non-zero when estimate_offset=False: {d_offsets}"
    )


def test_mbir_fp_matches_standard_fp_off_axis_voxel(backend):
    """Cross-projector regression: MBIR's own forward projector
    (mbir_forward_project_cpu / _cuda) and the standard ft.forward_project
    must agree on where a single off-Z-centre voxel lands on the detector.
    A Z-symmetric phantom hides the sign bug; placing the voxel at iz=8
    (well off the iz=Nz//2 centre) makes the sign error visible as a
    shifted detector hit.

    Replaces test_cpu.py::test_mbir_fp_matches_standard_fp_off_axis_voxel
    and test_cuda.py::test_mbir_fp_matches_standard_fp_off_axis_voxel_cuda
    (the centre of mass is computed in fp64 on both)."""
    mbir_sino = {"cpu": _mbir_sino_cpu, "cuda": _mbir_sino_self}[backend.type]
    Ny, Nz, Nx = 4, 32, 32
    N_det = Nx
    n_angles = 12
    phantom = torch.zeros(Ny, Nz, Nx, device=backend)
    phantom[Ny // 2, 8, Nx // 2] = 1.0   # off-centre in Z, on-centre in X
    angles_deg = make_angles(n_angles, max_deg=60.0, device=backend)

    sino_mbir = mbir_sino(phantom, angles_deg)
    sino_std = ft.forward_project(phantom, angles_deg)

    det_idx = np.arange(N_det, dtype=np.float64)
    iy = Ny // 2
    for ia in range(n_angles):
        m = sino_mbir[ia, iy].cpu().numpy().astype(np.float64)
        s = sino_std[ia, iy].cpu().numpy().astype(np.float64)
        m_sum, s_sum = m.sum(), s.sum()
        if m_sum < 1e-12 and s_sum < 1e-12:
            continue
        com_m = (m * det_idx).sum() / max(m_sum, 1e-12)
        com_s = (s * det_idx).sum() / max(s_sum, 1e-12)
        assert abs(com_m - com_s) < 0.75, (
            f"angle {ia}: MBIR {backend} FP COM {com_m:.2f} vs standard FP COM "
            f"{com_s:.2f} (diff {abs(com_m - com_s):.3f} > 0.75 px)"
        )


# Per-backend problem sizes of test_mbir_reconstruction_from_standard_fp_sinogram:
# (Ny, Nz, Nx), Z-asymmetric block (z0, z1, x0, x1), n_angles, n_iter, rel_err bound.
# The CPU case is small and tolerates more MBIR <-> ray-marching model mismatch.
_MBIR_STD_FP_CASES = {
    "cpu": ((4, 24, 24), (16, 22, 4, 10), 30, 50, 0.35),
    "cuda": ((16, 32, 32), (20, 28, 8, 16), 40, 60, 0.25),
}


def test_mbir_reconstruction_from_standard_fp_sinogram(backend):
    """End-to-end regression on the real bug path: sinogram built with the
    standard ft.forward_project (as user code and the notebook do) is given to
    run_mbir. With the z-sign bug the flip makes rel_err >> 1; post-fix it
    stays below the per-backend bound (0.35 CPU, 0.25 CUDA).

    Replaces test_cpu.py::test_mbir_reconstruction_from_standard_fp_sinogram_cpu
    and test_cuda.py::test_mbir_reconstruction_from_standard_fp_sinogram."""
    (Ny, Nz, Nx), (z0, z1, x0, x1), n_angles, n_iter, bound = _MBIR_STD_FP_CASES[backend.type]
    phantom = torch.zeros(Ny, Nz, Nx, device=backend)
    phantom[:, z0:z1, x0:x1] = 1.0   # Z-asymmetric block
    angles = make_angles(n_angles, max_deg=70.0, device=backend)
    sino = ft.forward_project(phantom, angles)
    vol = ft.run_mbir(
        sino, angles,
        sigma_f=0.1, huber_T=float("inf"),
        estimate_offset=False, estimate_variance=True,
        n_iter=n_iter, multires_levels=1,
    )
    rel_err = (vol - phantom).norm().item() / phantom.norm().item()
    assert rel_err < bound, (
        f"MBIR rel error vs phantom {rel_err:.4f} > {bound} "
        "(possible projector sign regression)"
    )


# ── input validation ──────────────────────────────────────────────────────────


def test_mbir_multires_levels_zero_raises():
    sino = torch.zeros(5, 2, 16)
    angles = torch.linspace(-70, 70, 5)
    with pytest.raises(ValueError, match="multires_levels must be >= 1"):
        ft.run_mbir(sino, angles, multires_levels=0)


def test_mbir_angles_2d_raises():
    sino = torch.zeros(5, 2, 16)
    with pytest.raises(ValueError, match="1-D"):
        ft.run_mbir(sino, torch.zeros(5, 1))


def test_mbir_angles_count_mismatch_raises():
    sino = torch.zeros(5, 2, 16)
    angles = torch.linspace(-70, 70, 7)
    with pytest.raises(ValueError, match="5 projections"):
        ft.run_mbir(sino, angles)


def test_mbir_lambdas_shape_mismatch_raises():
    sino = torch.zeros(5, 2, 16)
    angles = torch.linspace(-70, 70, 5)
    bad_lambdas = torch.ones(5, 2, 8)  # wrong N_det
    with pytest.raises(ValueError, match="lambdas must match"):
        ft.run_mbir(sino, angles, lambdas=bad_lambdas)


def test_mbir_lambdas_negative_raises():
    sino = torch.zeros(5, 2, 16)
    angles = torch.linspace(-70, 70, 5)
    bad_lambdas = -torch.ones_like(sino)
    with pytest.raises(ValueError, match="non-negative"):
        ft.run_mbir(sino, angles, lambdas=bad_lambdas)


def test_mbir_lambdas_nonfinite_raises():
    sino = torch.zeros(5, 2, 16)
    angles = torch.linspace(-70, 70, 5)
    bad_lambdas = torch.ones_like(sino)
    bad_lambdas[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ft.run_mbir(sino, angles, lambdas=bad_lambdas)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_mbir_invalid_sigma_init_raises(value):
    sino = torch.zeros(5, 2, 16)
    angles = torch.linspace(-70, 70, 5)
    with pytest.raises(ValueError, match="sigma_init"):
        ft.run_mbir(sino, angles, sigma_init=value)


@pytest.mark.parametrize("value", [0.0, -1e-3, float("nan")])
def test_mbir_invalid_sigma_min_raises(value):
    sino = torch.zeros(5, 2, 16)
    angles = torch.linspace(-70, 70, 5)
    with pytest.raises(ValueError, match="sigma_min"):
        ft.run_mbir(sino, angles, sigma_min=value)


def test_mbir_zero_lambdas_accepted():
    """lambdas == 0 (fully masked pixels) is valid and must still run."""
    sino = torch.randn(5, 2, 16)
    angles = torch.linspace(-70, 70, 5)
    lam = torch.ones_like(sino)
    lam[:, :, :4] = 0.0
    vol = ft.run_mbir(sino, angles, lambdas=lam, n_iter=2, multires_levels=1)
    assert torch.isfinite(vol).all()


def test_mbir_multires_not_divisible_raises():
    # N_det=14 is not divisible by coarse_factor=4 (multires_levels=3).
    sino14 = torch.zeros(5, 2, 14)
    angles = torch.linspace(-70, 70, 5)
    with pytest.raises(ValueError, match="divisible by"):
        ft.run_mbir(sino14, angles, multires_levels=3)


def test_mbir_multires_even_runs():
    """Baseline: 2-level multires with power-of-2 detector width should run."""
    sino = torch.randn(5, 2, 16)
    angles = torch.linspace(-70, 70, 5)
    vol = ft.run_mbir(sino, angles, multires_levels=2, n_iter=2)
    assert vol.shape == (2, 16, 16)


def _mbir_cylinder(device, N=32):
    z = torch.arange(N) - (N - 1) / 2
    r2 = z[:, None] ** 2 + z[None, :] ** 2
    vol = ((r2 <= 10 ** 2).float() + 0.5 * (r2 <= 4 ** 2).float()).expand(2, N, N)
    vol = vol.contiguous().to(device)
    ang = torch.linspace(-70, 70, 41, device=device)
    return vol, ang, ft.forward_project(vol, ang), (r2 <= 7 ** 2).to(device)


def test_mbir_multires_hands_over_fine_grid_values(backend, monkeypatch):
    """A level with factor f works in voxels f fine voxels wide, so its
    values are f times the fine-grid ones (a ray crosses f times fewer
    voxels). The volume handed from the coarse level (f = 2) to the fine one
    must be halved: it arrived at twice the object's level (scope report 2g
    [S], confirmed: 2.3 instead of 1.15), which the fine level then had to
    iterate away. A fine-grid warm start is doubled for the coarse level."""
    from fasttomo.algorithms import mbir as M

    vol, ang, sino, inner = _mbir_cylinder(backend)
    seen = []
    real = M._run_mbir_single_scale

    def spy(*a, **kw):
        st = kw["init_state"]
        seen.append((kw["Nz"], None if st is None else st.volume.clone()))
        return real(*a, **kw)

    monkeypatch.setattr(M, "_run_mbir_single_scale", spy)
    kw = dict(estimate_offset=False, estimate_variance=False, seed=1)
    ft.run_mbir(sino, ang, n_iter=20, multires_levels=2, **kw)
    (nz0, v0), (nz1, v1) = seen
    assert (nz0, nz1, v0) == (16, 32, None)
    level = vol[:, inner].mean().item()
    assert abs(v1[:, inner].mean().item() / level - 1.0) < 0.05
    # a fine-grid warm start reaches the coarse level at twice its values
    seen.clear()
    ft.run_mbir(sino, ang, n_iter=1, multires_levels=2, warm_start=vol, **kw)
    coarse = seen[0][1]
    torch.testing.assert_close(coarse, 2.0 * M._resize_volume_zx(vol, 16, 16))


def test_mbir_two_levels_converge_to_the_single_level_solution(backend):
    """Coarse-to-fine reaches the single-level MAP estimate: at 60
    iterations per level the two agree to 2e-3 relative (5e-4 measured on a
    64^2 cylinder, 1e-3 before the hand-over rescale), and at 5 iterations
    the two-level run is already at the object's level."""
    vol, ang, sino, inner = _mbir_cylinder(backend)
    kw = dict(estimate_offset=False, estimate_variance=False, seed=1)
    one = ft.run_mbir(sino, ang, n_iter=60, **kw)
    two = ft.run_mbir(sino, ang, n_iter=60, multires_levels=2, **kw)
    assert ((two - one).norm() / one.norm()).item() < 2e-3
    early = ft.run_mbir(sino, ang, n_iter=5, multires_levels=2, **kw)
    assert abs(early[:, inner].mean().item() / vol[:, inner].mean().item() - 1.0) < 0.05


def test_mbir_two_levels_with_variance_estimation_reach_the_single_level_solution(backend):
    """With estimate_variance=True the 2-level and 1-level runs reach the
    same fixed point, only slowly (#95 review: 17-22 % apart at 30-150
    iterations). On this noise-free 32^2 cylinder: 26 / 11 / 1.9 % apart at
    30 / 300 / 1200 iterations per level, with the same final sigma; the
    slow part is the joint estimate itself (a single level still moves
    too), see ``multires_levels`` in run_mbir's docstring."""
    vol, ang, sino, _ = _mbir_cylinder(backend)
    kw = dict(n_iter=1200, estimate_offset=False, estimate_variance=True,
              convergence_tol=0.0, return_state=True, seed=1)
    one, _, s1 = ft.run_mbir(sino, ang, **kw)
    two, _, s2 = ft.run_mbir(sino, ang, multires_levels=2, **kw)
    assert ((two - one).norm() / one.norm()).item() < 0.05
    assert abs(s2 / s1 - 1.0) < 1e-2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("sigma_init", [None, 0.1])
def test_mbir_rejects_non_finite_sinogram(backend, bad, sigma_init):
    """A NaN/inf pixel raises a clear error. Without sigma_init it used to
    say 'sigma_init must be finite' (the auto sigma is sinogram.std()); with
    sigma_init it ran and returned NaN voxels."""
    sino = torch.rand(6, 2, 8, device=backend)
    sino[2, 1, 3] = bad
    with pytest.raises(ValueError, match="sinogram contains NaN or inf"):
        ft.run_mbir(sino, torch.linspace(-40, 40, 6, device=backend),
                    n_iter=1, multires_levels=1, sigma_init=sigma_init)


# ── strided buffers ───────────────────────────────────────────────────────────


def test_mbir_noncontiguous_lambdas_matches_contiguous():
    """Regression (#58): a strided lambdas view must behave like its dense copy.

    The MBIR sweeps read lambdas through a raw linear buffer, so an
    ``expand``ed view used to be indexed as if it were dense (reading past
    the real storage) and gave a different reconstruction.
    """
    torch.manual_seed(0)
    Ny = Nz = Nx = 8
    n_angles = 5
    angles = make_angles(n_angles, max_deg=40.0)
    phantom = torch.rand(Ny, Nz, Nx)
    sino = ft.forward_project(phantom, angles)

    lam_expand = (torch.rand(1, Ny, Nx) + 0.5).expand(n_angles, Ny, Nx)
    assert not lam_expand.is_contiguous()
    lam_contig = lam_expand.contiguous()

    kw = dict(
        Nz=Nz,
        sigma_f=0.1,
        huber_T=float("inf"),
        estimate_offset=False,
        estimate_variance=False,
        n_iter=4,
        multires_levels=1,
        seed=11,
    )
    vol_contig = ft.run_mbir(sino, angles, lambdas=lam_contig, **kw)
    vol_expand = ft.run_mbir(sino, angles, lambdas=lam_expand, **kw)

    assert torch.allclose(vol_contig, vol_expand, atol=1e-6, rtol=1e-6)


def test_mbir_cpu_sweep_rejects_noncontiguous_buffers():
    """The CPU MBIR sweep writes volume / e_sino through raw pointers, so a
    strided buffer must raise rather than be misread or silently copied (#58)."""
    from fasttomo.algorithms import _mbir_icd_sweep_cpu
    from fasttomo.autograd import _trig_on_device

    Ny, Nz, Nx, n_angles = 2, 4, 4, 3
    N_det = Nx
    cos_a, sin_a = _trig_on_device(torch.linspace(-30, 30, n_angles), torch.device("cpu"))
    perm = torch.arange(Nz * Nx, dtype=torch.int32).repeat(Ny, 1)

    def call(volume, e_sino, lambdas):
        return _mbir_icd_sweep_cpu(
            volume, e_sino, lambdas, torch.zeros(Ny, Nz, Nx), perm,
            cos_a, sin_a,
            qggmrf_p=1.2, qggmrf_q=2.0, qggmrf_c=0.01, sigma_f=1.0,
            huber_T=float("inf"), huber_delta=1.0, sigma=1.0, num_threads=1,
        )

    vol = torch.zeros(Ny, Nz, Nx)
    e = torch.ones(n_angles, Ny, N_det)
    lam = torch.ones(n_angles, Ny, N_det)
    call(vol, e.clone(), lam)  # dense buffers are accepted

    lam_strided = torch.ones(1, Ny, N_det).expand(n_angles, Ny, N_det)
    with pytest.raises(ValueError, match="lambdas must be C-contiguous"):
        call(vol, e.clone(), lam_strided)

    vol_strided = torch.zeros(Ny, Nx, Nz).transpose(1, 2)
    with pytest.raises(ValueError, match="volume must be C-contiguous"):
        call(vol_strided, e.clone(), lam)


@needs_cuda
def test_mbir_noncontiguous_lambdas_matches_contiguous_cuda():
    """Regression (#58): the CUDA sweep reads lambdas through a raw pointer,
    so an ``expand``ed view must give the same volume as its dense copy.

    The view is expanded along Ny from a [n_angles, 1, N_det] base with
    different values per angle, and a filler tensor is allocated right after
    the base storage. Without the fix the kernel reads the base linearly:
    past its first rows it lands in the filler (1e3), not in a copy of the
    right values. (The first version of this test expanded along the angle
    axis and allocated the dense copy right after the base, so the stray
    reads found exactly the right values and the test passed on the unfixed
    code.)"""
    Ny = Nz = Nx = 16
    n_angles = 20
    angles = make_angles(n_angles, max_deg=60.0, device=DEVICE)
    phantom = torch.zeros(Ny, Nz, Nx, device=DEVICE)
    phantom[4:12, 4:12, 4:12] = 1.0
    sino = _mbir_sino_self(phantom, angles)

    g = torch.Generator(device="cpu").manual_seed(0)
    lam_base = (torch.rand(n_angles, 1, Nx, generator=g) + 0.5).to(DEVICE)
    filler = torch.full((1 << 16,), 1e3, device=DEVICE)   # noqa: F841 — keeps the memory after lam_base
    lam_expand = lam_base.expand(n_angles, Ny, Nx)
    assert not lam_expand.is_contiguous()
    lam_contig = lam_expand.contiguous()

    kw = dict(
        sigma_f=0.1, huber_T=float("inf"),
        estimate_offset=False, estimate_variance=False,
        n_iter=10, multires_levels=1, sigma_init=0.1, seed=3,
    )
    v_contig = ft.run_mbir(sino, angles, lambdas=lam_contig, **kw)
    v_expand = ft.run_mbir(sino, angles, lambdas=lam_expand, **kw)
    assert torch.allclose(v_contig, v_expand, atol=1e-6, rtol=1e-6)


# ── offset / variance update ──────────────────────────────────────────────────


def _offset_variance_ref(e, d, sigma, sigma_min, T, delta, lam, est_off, est_var):
    """fp64 NumPy reference of the closed-form MBIR updates."""
    e = e.astype(np.float64).copy()
    d = d.astype(np.float64).copy()
    lam = np.ones_like(e) if lam is None else lam.astype(np.float64)

    def weights(e):
        r = np.abs(e * np.sqrt(lam) / sigma)
        if not math.isfinite(T):
            return np.ones_like(r)
        return np.where(r < T, 1.0, delta * T / np.maximum(r, 1e-30))

    if est_off:
        bw = weights(e)
        for k in range(e.shape[0]):
            den = (bw[k] * lam[k]).sum()
            if den < 1e-30:
                continue
            dd = (bw[k] * lam[k] * e[k]).sum() / den
            d[k] += dd
            e[k] -= dd
    s = sigma
    if est_var:
        bw = weights(e)
        s = max(sigma_min, math.sqrt((bw * lam * e * e).sum() / e.size))
    return e, d, s


@pytest.mark.parametrize("T", [float("inf"), 1.5])
@pytest.mark.parametrize("use_lam", [False, True])
@pytest.mark.parametrize("flags", [(True, True), (True, False), (False, True), (False, False)])
def test_mbir_offset_variance_matches_reference(backend, T, use_lam, flags):
    """The single offset/variance implementation (both backends) against an
    fp64 reference, with Huber weights, per-pixel lambdas and one view whose
    weights are all zero (its offset must stay put, as in the removed C code)."""
    from fasttomo.algorithms import _mbir_update_offset_variance
    rng = np.random.default_rng(13)
    e = (rng.standard_normal((5, 3, 7)) + rng.standard_normal((5, 1, 1))).astype(np.float32)
    e[1, 0, 2] = 9.0                                   # Huber outlier
    lam = (rng.random((5, 3, 7)) + 0.5).astype(np.float32) if use_lam else None
    if use_lam:
        lam[4] = 0.0                                   # a view with no weight
    d0 = rng.standard_normal(5).astype(np.float32)
    e_ref, d_ref, s_ref = _offset_variance_ref(e, d0, 0.8, 1e-3, T, 0.5, lam, *flags)

    e_t = torch.from_numpy(e).to(backend)
    d_new, s_new = _mbir_update_offset_variance(
        e_t, torch.from_numpy(d0).to(backend), 0.8, 1e-3, T, 0.5,
        torch.from_numpy(lam).to(backend) if use_lam else None, *flags)
    torch.testing.assert_close(e_t.cpu().double(), torch.from_numpy(e_ref), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(d_new.cpu().double(), torch.from_numpy(d_ref), rtol=1e-6, atol=1e-6)
    assert s_new == pytest.approx(s_ref, rel=1e-6)
    if use_lam and flags[0]:
        assert d_new[4].item() == d0[4]


def test_mbir_cpu_uses_the_python_offset_variance_update(monkeypatch):
    """The CPU path calls _mbir_update_offset_variance once per outer
    iteration (before P12 the C sweep did its own copy of the update)."""
    import fasttomo.algorithms as algo
    calls = []
    real = algo._mbir_update_offset_variance

    def counting(*args, **kwargs):
        calls.append(args[-2:])
        return real(*args, **kwargs)

    monkeypatch.setattr(algo.mbir, "_mbir_update_offset_variance", counting)
    sino = torch.rand(6, 2, 8, generator=torch.Generator().manual_seed(14))
    ft.run_mbir(sino, torch.linspace(-40, 40, 6), n_iter=3, multires_levels=1,
                convergence_tol=0.0, estimate_offset=True, estimate_variance=True)
    assert calls == [(True, True)] * 3
