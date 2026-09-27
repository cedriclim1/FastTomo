"""
MBIR (run_mbir): ICD reconstruction, anomaly modelling, offsets and
variance, multires, input validation and the raw sweep bindings.

CPU <-> CUDA agreement lives in test_parity.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
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
