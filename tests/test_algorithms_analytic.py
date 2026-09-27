"""
Direct reconstructions: FBP (run_fbp) and weighted back-projection
(run_weighted_backprojection). W-BP operator details are in test_wsirt.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from conftest import (
    make_angles,
    needs_cuda,
    phantom_2d,
    rel_err,
)

import fasttomo as ft

# ── FBP ───────────────────────────────────────────────────────────────────────


# ── FBP against an independent fp64 FBP of an analytic cylinder ───────────
#
# CPU run_fbp back-projects with the shift-invariant gather (the CUDA BP's
# operator with exact weights), no longer with the CPU forward projector's
# ray-driven transpose, whose flat-field pattern gave 0.9-2.5 % RMS error
# (0.08-0.3 max) against this reference (phase-1b CPU audit A2).

def _cylinder_sinogram(N, r, ang_deg, Ny=2):
    """Pixel-averaged exact chords of a centred unit-density cylinder."""
    u = np.arange(N) - (N - 1) / 2.0
    sub = (np.arange(16) + 0.5) / 16 - 0.5
    p = (2 * np.sqrt(np.clip(r ** 2 - (u[:, None] + sub[None, :]) ** 2, 0, None))).mean(1)
    return np.broadcast_to(p, (len(ang_deg), Ny, N)).copy()


def _fbp_fp64(sino, ang_deg, Nz, Nx):
    """Independent fp64 FBP: spatial Ram-Lak linear convolution (h[0] = 1/4,
    h[odd n] = -1/(pi n)^2), linear-interpolation gather back-projection
    (zero outside the detector), pi / n_angles."""
    n, Ny, Nd = sino.shape
    k = np.arange(-(Nd - 1), Nd)
    h = np.zeros(k.shape)
    h[k == 0] = 0.25
    odd = k % 2 == 1
    h[odd] = -1.0 / (math.pi * k[odd]) ** 2
    q = np.apply_along_axis(lambda row: np.convolve(row, h)[Nd - 1:2 * Nd - 1], 2, sino)
    x = np.arange(Nx) - (Nx - 1) / 2
    z = np.arange(Nz) - (Nz - 1) / 2
    out = np.zeros((Ny, Nz, Nx))
    for a, th in enumerate(np.deg2rad(ang_deg)):
        uu = x[None, :] * math.cos(th) + z[:, None] * math.sin(th) + (Nd - 1) / 2
        i0 = np.floor(uu).astype(int)
        w = uu - i0
        rows = np.pad(q[a], ((0, 0), (1, 1)))
        r0 = rows[:, np.clip(i0 + 1, 0, Nd + 1)]
        r1 = rows[:, np.clip(i0 + 2, 0, Nd + 1)]
        out += (1 - w) * r0 + w * r1
    return out * math.pi / n


@pytest.mark.parametrize("n,rng", [(90, (0, 180)), (41, (-60, 60))])
def test_fbp_matches_fp64_fbp_of_a_cylinder(backend, n, rng):
    N, r = 64, 20.0
    ang = np.linspace(rng[0], rng[1], n, endpoint=(rng != (0, 180)))
    s64 = _cylinder_sinogram(N, r, ang)
    ref = _fbp_fp64(s64, ang, N, N)
    rec = ft.run_fbp(torch.tensor(s64, dtype=torch.float32, device=backend),
                     torch.tensor(ang, dtype=torch.float32, device=backend)).double().cpu().numpy()
    rms = float(np.sqrt(((rec - ref) ** 2).mean()))
    # CPU: exact fp32 weights (measured 3-5e-7); CUDA: 8-bit texture weights (~1e-4)
    assert rms < (2e-6 if backend.type == "cpu" else 5e-4), rms


@needs_cuda
def test_fbp_cpu_matches_cuda():
    """Both backends back-project with the same gather operator: they differ
    by the CUDA texture unit's 8-bit weights only (1.1-1.6e-4 measured; the
    CPU ray-driven BP was 1.6-3.9e-2 away)."""
    N, n = 64, 61
    ang = torch.linspace(-60, 60, n)
    vol = torch.zeros(3, N, N)
    vol[:, 16:48, 20:44] = 1.0
    vol[1, 24:40, 26:30] = 2.0
    sino = ft.forward_project(vol, ang)
    cpu = ft.run_fbp(sino, ang, mask=True)
    gpu = ft.run_fbp(sino.cuda(), ang.cuda(), mask=True).cpu()
    assert float((gpu - cpu).norm() / cpu.norm()) < 6e-4


def test_cpu_fbp_uses_the_gather_kernel_and_stays_differentiable(monkeypatch):
    """CPU run_fbp back-projects with the C gather kernel (num_threads passed
    through); its autograd backward is the CPU forward projection of the
    upstream gradient, the pairing CUDA's back-projection has."""
    from fasttomo import _core
    from fasttomo.algorithms.analytic import _GatherBackProject

    calls = []
    real = _core.back_project_gather_parallel_cpu
    monkeypatch.setattr(_core, "back_project_gather_parallel_cpu",
                        lambda *a: (calls.append(a[-1]), real(*a))[1])
    ang = torch.linspace(-60, 60, 13)
    sino = torch.rand(13, 2, 24, generator=torch.Generator().manual_seed(2))
    rec = ft.run_fbp(sino, ang, num_threads=3)
    assert calls == [3] and rec.shape == (2, 24, 24)

    s = sino.clone().requires_grad_(True)
    w = torch.rand(2, 24, 24, generator=torch.Generator().manual_seed(3))
    (_GatherBackProject.apply(s, ang, 24, 24, 1) * w).sum().backward()
    assert torch.equal(s.grad, ft.forward_project(w, ang))
    s2 = sino.clone().requires_grad_(True)
    ft.run_fbp(s2, ang).sum().backward()
    assert s2.grad is not None and torch.isfinite(s2.grad).all()


def test_fbp_shape(backend):
    """Replaces test_cpu.py::test_fbp_shape and test_cuda.py::test_fbp_shape_cuda."""
    N      = 64
    angles = make_angles(60, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fbp(sino, angles, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == backend.type


def _cylinder_fbp_level(backend, radius, tilt, N=128):
    """Mean FBP value over the inner half-radius of a unit-density cylinder
    (a disk in XZ, constant along Y) of ``radius`` voxels. ``tilt=90``:
    180 equally spaced views over [-90, 90); else ``linspace(-tilt, tilt)``
    at the same 1-degree spacing."""
    z = torch.arange(N) - (N - 1) / 2
    r2 = z[:, None] ** 2 + z[None, :] ** 2
    vol = (r2 <= radius ** 2).float().expand(2, N, N).contiguous().to(backend)
    if tilt == 90:
        ang = torch.arange(180) * 1.0 - 90
    else:
        ang = torch.linspace(-tilt, tilt, 2 * tilt + 1)
    ang = ang.to(backend)
    rec = ft.run_fbp(ft.forward_project(vol, ang), ang)
    return rec[:, r2 <= (radius / 2) ** 2].mean().item()


@pytest.mark.parametrize("radius", [20, 40])
def test_fbp_angular_scale_keeps_mean_level_at_limited_tilt(backend, radius):
    """Scope report 2g [S]: FBP scales by pi / n_angles, which looks like
    it assumes 180 degrees of coverage, so +-70 degrees was suspected to be
    off by 180/140. pi / n_angles is pi times the mean over the measured
    views: it keeps the mean level correct, which is what this pins. For a
    rotationally symmetric object every view carries the same ray integral,
    so the level at +-70 equals the one at 180 degrees; scaling by the
    actual spacing (140 degrees / n) would make it 0.78 of that and fail
    here. This does not make +-70 unbiased: anisotropic objects still show
    an orientation-dependent bias at limited tilt (the missing wedge; a 4:1
    ellipse reads 1.20 or 0.50 depending on its orientation, #95 review),
    which no angular scale removes."""
    full = _cylinder_fbp_level(backend, radius, 90)
    limited = _cylinder_fbp_level(backend, radius, 70)
    assert abs(limited / full - 1.0) < 5e-3, (limited, full)


@pytest.mark.parametrize("tilt", [90, 70])
@pytest.mark.parametrize("radius", [20, 40, 60])
def test_fbp_cylinder_absolute_value(backend, radius, tilt):
    """A unit-density cylinder reconstructs at 1 (inner half-radius mean)
    at 180 degrees and at +-70 degrees, from a small one to one filling the
    field of view. The unpadded |xi| ramp gave 0.96 / 0.83 / 0.57."""
    level = _cylinder_fbp_level(backend, radius, tilt)
    assert abs(level - 1.0) < 5e-3, level


def test_ramp_filter_is_the_linear_ram_lak_convolution():
    """ramp_filter == the direct (linear, unpadded-wraparound-free)
    convolution of each row with h[0] = 1/4, h[odd n] = -1/(pi n)^2."""
    import math

    g = torch.Generator().manual_seed(3)
    for N_det in (7, 16, 33):
        sino = torch.rand(2, 3, N_det, generator=g, dtype=torch.float64)
        n = torch.arange(-(N_det - 1), N_det, dtype=torch.float64)
        h = torch.where(n.abs() % 2 == 1, -1.0 / (math.pi * n) ** 2, torch.zeros_like(n))
        h[N_det - 1] = 0.25
        ref = torch.stack([torch.stack([
            torch.tensor([(row * h.flip(0)[N_det - 1 - u: 2 * N_det - 1 - u]).sum().item()
                          for u in range(N_det)], dtype=torch.float64)
            for row in view]) for view in sino])
        out = ft.ramp_filter(sino.float())
        assert out.shape == sino.shape and out.dtype == torch.float32
        torch.testing.assert_close(out.double(), ref, rtol=0, atol=1e-6)


def test_fbp_better_than_plain_backproject(backend):
    """FBP reconstruction MSE should beat raw (unfiltered) back-projection.

    Replaces test_cpu.py::test_fbp_better_than_plain_backproject and
    test_cuda.py::test_fbp_better_than_plain_backproject_cuda."""
    N      = 64
    angles = make_angles(90, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    def norm_mse(a, b):
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        b = (b - b.min()) / (b.max() - b.min() + 1e-9)
        return ((a - b) ** 2).mean().item()

    bp_mse  = norm_mse(ft.back_project(sino, angles, Nz=N), vol)
    fbp_mse = norm_mse(ft.run_fbp(sino, angles, Nz=N), vol)
    assert fbp_mse < bp_mse, (
        f"FBP MSE ({fbp_mse:.4f}) should be < raw BP MSE ({bp_mse:.4f})"
    )


def test_fbp_mask_uses_projector_output_shape():
    """mask=True must follow the projector output geometry.

    Cached-projector FBP can reconstruct with Nz != N_det. The mask must be
    built from the reconstructed volume shape, not from the input sinogram.
    """
    n_angles, Ny, N_det = 7, 2, 16
    Nz_proj = 20
    sino = torch.randn(n_angles, Ny, N_det)
    angles = make_angles(n_angles)

    class DummyProjector:
        def back(self, _sino: torch.Tensor) -> torch.Tensor:
            return torch.ones(Ny, Nz_proj, N_det, dtype=_sino.dtype, device=_sino.device)

    rec = ft.run_fbp(sino, angles, Nz=-1, mask=True, projector=DummyProjector())
    assert rec.shape == (Ny, Nz_proj, N_det)


DEVICE = torch.device("cuda")


@needs_cuda
def test_run_fbp_fp16_matches_fp32():
    """fp16 run_fbp recon should match fp32 to within ~1% relative on
    a structured phantom (FFT in fp16 is accepted, see run_fbp docstring)."""
    N      = 64
    angles = make_angles(60, device=DEVICE)
    vol    = phantom_2d(N, device=DEVICE).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec_fp32 = ft.run_fbp(sino, angles, Nz=N, mask=True)
    rec_fp16 = ft.run_fbp(sino.half(), angles, Nz=N, mask=True)
    assert rec_fp16.dtype == torch.float16
    # Looser tolerance because fp16 FFT compounds the precision loss.
    assert rel_err(rec_fp16, rec_fp32) < 5e-2


# ── W-BP ──────────────────────────────────────────────────────────────────────


def test_wbp_shape(backend):
    """Replaces test_cpu.py::test_wbp_shape and test_cuda.py::test_wbp_shape_cuda."""
    N      = 32
    angles = make_angles(20, device=backend)
    vol    = phantom_2d(N, device=backend).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_weighted_backprojection(sino, angles, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == backend.type


# ── support_mask ──────────────────────────────────────────────────────────────


def _box_support(Ny: int, N: int) -> torch.Tensor:
    """Binary [Ny, N, N] support: 1 inside the central half-box, 0 outside."""
    M = torch.zeros(Ny, N, N)
    q = N // 4
    M[:, q:3*q, q:3*q] = 1.0
    return M


def test_fbp_support_mask_equals_post_multiply():
    """FBP applies the support once at the end: == unmasked recon * mask."""
    N      = 32
    angles = make_angles(20)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)

    M = _box_support(1, N)
    rec_masked = ft.run_fbp(sino, angles, Nz=N, support_mask=M)
    rec_plain  = ft.run_fbp(sino, angles, Nz=N)
    assert torch.equal(rec_masked, rec_plain * M)


def test_fbp_support_mask_reproduces_cylinder():
    """Passing the cylinder as a support tensor reproduces mask=True exactly."""
    N      = 32
    angles = make_angles(20)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)

    M = ft.cylindrical_mask(N, N).unsqueeze(0)
    rec_support  = ft.run_fbp(sino, angles, Nz=N, support_mask=M)
    rec_cylinder = ft.run_fbp(sino, angles, Nz=N, mask=True)
    assert torch.equal(rec_support, rec_cylinder)


def test_fbp_support_mask_with_mask_true_raises():
    N      = 32
    angles = make_angles(10)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ft.run_fbp(sino, angles, Nz=N, mask=True,
                   support_mask=_box_support(1, N))


def test_fbp_support_mask_bad_shape_raises():
    N      = 32
    angles = make_angles(10)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)
    with pytest.raises(ValueError, match="support_mask"):
        ft.run_fbp(sino, angles, Nz=N, support_mask=torch.ones(N + 1, N))
    with pytest.raises(ValueError, match="support_mask"):
        ft.run_fbp(sino, angles, Nz=N, support_mask=torch.ones(3, N, N))


def _half_support(N: int, left: bool) -> torch.Tensor:
    """Binary [1, N, N] support covering the left or right half in x."""
    M = torch.zeros(1, N, N, device=DEVICE)
    if left:
        M[:, :, : N // 2] = 1.0
    else:
        M[:, :, N // 2:] = 1.0
    return M


@needs_cuda
def test_fbp_support_mask_cuda():
    """support_mask on CUDA: equals post-multiply; bool tensors accepted."""
    N      = 32
    angles = make_angles(20, device=DEVICE)
    sino   = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles)

    M = _half_support(N, left=True)
    rec_masked = ft.run_fbp(sino, angles, Nz=N, support_mask=M)
    rec_plain  = ft.run_fbp(sino, angles, Nz=N)
    assert torch.equal(rec_masked, rec_plain * M)

    rec_bool = ft.run_fbp(sino, angles, Nz=N, support_mask=M.bool())
    assert torch.equal(rec_bool, rec_masked)


@needs_cuda
def test_fbp_support_mask_fp16_volume_fp32_mask_cuda():
    """fp16 sinogram + fp32 mask: mask is cast, dtype preserved end-to-end."""
    N      = 32
    angles = make_angles(20, device=DEVICE)
    sino   = ft.forward_project(phantom_2d(N, device=DEVICE).unsqueeze(0), angles).half()

    M   = _half_support(N, left=True)            # float32
    rec = ft.run_fbp(sino, angles, Nz=N, support_mask=M)
    assert rec.dtype == torch.float16
    assert rec[M.expand_as(rec) == 0].abs().max().item() == 0.0


# ── W-BP weighting-filter cache (solver audit P3) ────────────────────────────


def test_wbp_filter_cache_is_bit_identical(backend):
    """The exact weighting filter is built once per geometry and reused:
    a repeated run_wbp returns the same bits, the cached filter is the one a
    fresh build gives, and angles / sizes / threshold are part of the key."""
    from fasttomo.algorithms import analytic as an

    an._WF_CACHE.clear()
    N = 24
    ang = make_angles(19, device=backend)
    vol = phantom_2d(N, device=backend).unsqueeze(0).repeat(2, 1, 1)
    sino = ft.forward_project(vol, ang)
    a = ft.run_wbp(sino, ang)
    assert len(an._WF_CACHE) == 1
    b = ft.run_wbp(sino, ang)
    assert len(an._WF_CACHE) == 1 and torch.equal(a, b)
    wf = an._cached_exact_wf(ang, N, N, N, sino.device, 1.0)
    fresh = an._compute_exact_wf(ang, N, N, N, sino.device, 1.0)
    assert torch.equal(wf, fresh)
    # angles on the other side of the host/device boundary hit the same entry
    an._cached_exact_wf(ang.cpu(), N, N, N, sino.device, 1.0)
    assert len(an._WF_CACHE) == 1
    ft.run_wbp(sino, ang, threshold_scale=2.0)
    ft.run_wbp(sino, ang + 0.5)
    ft.run_wbp(sino, ang, Nz=20)
    assert len(an._WF_CACHE) == 4
    for i in range(an._WF_CACHE_SIZE + 2):          # bounded LRU
        an._cached_exact_wf(ang + i + 1.0, 8, 8, 8, sino.device, 1.0)
    assert len(an._WF_CACHE) == an._WF_CACHE_SIZE
    an._WF_CACHE.clear()


# ── FBP apodisation window (solver audit H4) ─────────────────────────────────


def _windowed_ramp_reference(sino: torch.Tensor, window: str) -> torch.Tensor:
    """fp64 NumPy reference of the apodised ramp filter: the spatial
    Ram-Lak kernel (h[0] = 1/4, h[odd n] = -1/(pi n)^2) zero-padded to the
    next power of two P >= 2 N - 1, its spectrum times the window at
    f = k / P, applied to each zero-padded row and cropped."""
    import math

    import numpy as np

    N = sino.shape[-1]
    P = 1 << max(1, (2 * N - 1).bit_length())
    n = np.arange(P)
    n = np.where(n < P // 2, n, n - P)
    h = np.where(np.abs(n) % 2 == 1, -1.0 / (math.pi * np.where(n == 0, 1, n)) ** 2, 0.0)
    h[0] = 0.25
    f = np.arange(P // 2 + 1) / P
    W = {"hann": 0.5 + 0.5 * np.cos(2 * math.pi * f),
         "hamming": 0.54 + 0.46 * np.cos(2 * math.pi * f),
         "none": np.ones_like(f)}[window]
    H = np.fft.rfft(h).real * W
    x = sino.double().cpu().numpy()
    out = np.fft.irfft(np.fft.rfft(x, n=P, axis=-1) * H, n=P, axis=-1)[..., :N]
    return torch.from_numpy(out)


@pytest.mark.parametrize("window", ["none", "hann", "hamming"])
def test_fbp_window_filter_matches_fp64_reference(backend, window):
    """The apodised ramp filter run_fbp applies equals an independent fp64
    reference to fp32 rounding, for odd and even detector widths."""
    from fasttomo.algorithms.analytic import _fbp_filter_fp32

    g = torch.Generator().manual_seed(5)
    for N_det in (7, 16, 33, 128):
        sino = torch.rand(3, 2, N_det, generator=g).to(backend)
        got = _fbp_filter_fp32(sino, window).double().cpu()
        ref = _windowed_ramp_reference(sino, window)
        assert (got - ref).abs().max() <= 1e-6 * ref.abs().max(), (N_det, window)


def test_fbp_window_keyword(backend):
    """window="none" is the former run_fbp bit for bit; Hann and Hamming keep
    the mean level of a unit cylinder, cut the noise of a noisy sinogram,
    and run_fbp(window=w) is the back-projection of the apodised filter."""
    import math

    N, radius = 96, 30
    z = torch.arange(N) - (N - 1) / 2
    r2 = z[:, None] ** 2 + z[None, :] ** 2
    vol = (r2 <= radius ** 2).float().expand(2, N, N).contiguous().to(backend)
    ang = (torch.arange(180) * 1.0 - 90).to(backend)
    sino = ft.forward_project(vol, ang)
    g = torch.Generator().manual_seed(0)
    noisy = sino + 0.5 * torch.randn(sino.shape, generator=g).to(backend)
    inner = r2 <= (radius / 2) ** 2
    assert torch.equal(ft.run_fbp(sino, ang), ft.run_fbp(sino, ang, window="none"))
    std_none = ft.run_fbp(noisy, ang)[:, inner].std()
    for w in ("hann", "hamming"):
        rec = ft.run_fbp(sino, ang, window=w)
        assert abs(rec[:, inner].mean().item() - 1.0) < 5e-3, w
        assert ft.run_fbp(noisy, ang, window=w)[:, inner].std() < 0.8 * std_none, w
    from fasttomo.algorithms.analytic import _fbp_filter_fp32, _fbp_gather_backproject
    filtered = _fbp_filter_fp32(sino, "hann")
    if backend.type == "cpu":        # run_fbp's CPU back-projection is the gather
        want = _fbp_gather_backproject(filtered, ang, N, N, None) * (math.pi / 180)
    else:
        want = ft.back_project(filtered, ang) * (math.pi / 180)
    torch.testing.assert_close(ft.run_fbp(sino, ang, window="hann"), want,
                               rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="window"):
        ft.run_fbp(sino, ang, window="shepp")


def test_release_cached_projectors_drops_wbp_filters(backend):
    """release_cached_projectors() also frees the cached W-BP weighting
    filters (it still returns the number of run_sirt configurations)."""
    from fasttomo.algorithms import analytic as an

    ang = make_angles(9, device=backend)
    sino = ft.forward_project(phantom_2d(16, device=backend).unsqueeze(0), ang)
    ft.release_cached_projectors()
    ft.run_wbp(sino, ang)
    assert len(an._WF_CACHE) == 1
    assert ft.release_cached_projectors() == 0
    assert len(an._WF_CACHE) == 0
