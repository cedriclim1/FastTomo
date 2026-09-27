"""
Sampling-boundary rule and kink gradients of the projectors (phase 1b PR G).

The parallel-beam operators (CPU and CUDA forward, the CPU ray-driven
transpose, grad_angle) use the rotmat path's *loose* rule: the slice is a sum
of bilinear tent functions, zero outside, so a sample contributes while any
corner of its stencil is inside, fx in (-1, Nx), fz in (-1, Nz). The former
tight rule, fx in [0, Nx-1), zeroed the last detector column at 0 deg,
never sampled the last z row for odd Nz and projected Nz = 1 / Nx = 1 volumes
to zero. The geometry gradients (grad_angle, grad_rotmat) average the two
one-sided derivatives at samples exactly on a grid line, so they equal the
central finite difference at grid-aligned views (0 deg, +-90 deg, pure tilts,
the identity pose) too.

References are independent fp64 NumPy models (``_fp_parallel_ref``,
``_fp_rotmat_ref`` in test_kernel_correctness.py) and analytic line integrals.
Tolerances: CPU vs fp64 1e-5 relative; the CUDA forward 2e-3 (the texture
unit's 8-bit filter weights); gradients vs central FD 1e-4 (measured 1e-8 to
1.2e-6 on both backends).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from conftest import needs_cuda
from test_kernel_correctness import _fd_grad_mats, _fp_parallel_ref, _fp_rotmat_ref

import fasttomo as ft
import fasttomo._core as C
from fasttomo.autograd import (
    _back_project_parallel_raw,
    _forward_project_parallel_raw,
    _grad_angle_raw,
    _grad_rotmat_raw,
    euler_zyz_to_rotmat,
)

HAS_CUDA = torch.cuda.is_available()
DEVICES = ["cpu", pytest.param("cuda", marks=needs_cuda)]


def _tol(device):
    return 1e-5 if device == "cpu" else 2e-3


def _maxrel(a, b):
    a = torch.as_tensor(a, dtype=torch.float64).cpu()
    b = torch.as_tensor(b, dtype=torch.float64).cpu()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


# ── 1. edge detectors at axis-aligned views ───────────────────────────────

_SHAPES_2D = [(8, 8), (9, 9), (9, 8), (8, 9)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("Nz,Nx", _SHAPES_2D)
def test_axis_aligned_views_see_every_column_and_row(device, Nz, Nx):
    """0 deg: detector i (N_det = Nx) sees column i, 180 deg column Nx-1-i;
    +-90 deg (N_det = Nz): row i / row Nz-1-i. Every edge detector sees its
    full edge column / row (the tight rule gave 0 on the last one)."""
    g = torch.Generator().manual_seed(Nz * 10 + Nx)
    vol = torch.rand(3, Nz, Nx, generator=g) + 0.5           # non-zero on every face
    col, row = vol.sum(1), vol.sum(2)                          # [Ny, Nx], [Ny, Nz]
    v = vol.to(device)
    a = torch.tensor([0.0, 180.0], device=device)
    s = ft.forward_project(v, a).cpu()
    assert _maxrel(s[0], col) < _tol(device)
    assert _maxrel(s[1], col.flip(-1)) < _tol(device)
    a = torch.tensor([90.0, -90.0], device=device)
    s = ft.forward_project(v, a, N_det=Nz).cpu()
    assert _maxrel(s[0], row) < _tol(device)
    assert _maxrel(s[1], row.flip(-1)) < _tol(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("Nz", [8, 9])
def test_last_z_row_is_sampled_at_zero_degrees(device, Nz):
    """A volume that is 1 only on its last (or first) z row projects to 1 on
    every detector at 0 deg; with a wider detector the two outer detectors
    sit exactly on the tent ends and see 0."""
    for iz in (0, Nz - 1):
        vol = torch.zeros(2, Nz, 7)
        vol[:, iz] = 1.0
        s = ft.forward_project(vol.to(device), torch.zeros(1, device=device), N_det=9).cpu()
        want = torch.tensor([0.0] + [1.0] * 7 + [0.0]).expand(2, 9)
        assert _maxrel(s[0], want) < _tol(device), iz


# ── 2. constant volume: mass ──────────────────────────────────────────────

def _chord_square(t, th, half):
    c, s = math.cos(th), math.sin(th)
    lo, hi = -1e18, 1e18
    for p0, d in ((t * c, -s), (t * s, c)):
        if abs(d) < 1e-15:
            if abs(p0) > half:
                return 0.0
            continue
        a, b = (-half - p0) / d, (half - p0) / d
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return max(0.0, hi - lo)


@pytest.mark.parametrize("device", DEVICES)
def test_constant_volume_mass(device):
    """Axis-aligned rays through a constant N^2 slice sum to N exactly (the
    tent basis is a partition of unity for any sampling phase). At oblique
    views the unit-step ray sum follows the chord of the pixel square
    [-N/2, N/2]^2 to the discretisation error of the ray-driven model (the
    design study measured 0.026 N on interior detectors and 0.126 N on the
    edge ones at N = 16; the tight rule gave 0.17 N and 1.0 N)."""
    N = 16
    vol = torch.ones(2, N, N, device=device)
    s = ft.forward_project(vol, torch.tensor([0.0, 90.0, -90.0, 180.0], device=device)).cpu()
    assert _maxrel(s, torch.full_like(s, float(N))) < _tol(device)
    ang = [1.0, 17.3]
    s = ft.forward_project(vol, torch.tensor(ang, device=device)).cpu()[:, 0].double()
    ref = torch.tensor([[_chord_square(i - 0.5 * (N - 1), math.radians(a), N / 2)
                         for i in range(N)] for a in ang], dtype=torch.float64)
    err = (s - ref).abs() / N
    assert err[:, 2:-2].max() < 0.03
    assert err.max() < 0.13


# ── 3. flip symmetries ────────────────────────────────────────────────────

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("Nz,Nx", [(16, 16), (17, 17), (16, 17)])
def test_flip_and_half_turn_symmetry(device, Nz, Nx):
    """FP(v, th) = flip_det FP(flip_x v, -th) = flip_det FP(v, th + 180) on
    volumes that touch every face (the tight rule broke both by up to 0.8 of
    the maximum at 0 deg and 0.14 at 90 / 180 deg)."""
    g = torch.Generator().manual_seed(Nz + Nx)
    vol = (torch.rand(2, Nz, Nx, generator=g) + 0.2).to(device)
    ang = torch.tensor([0.0, 17.3, 45.0, 60.0, 90.0], device=device)
    s = ft.forward_project(vol, ang).cpu()
    s_flip = ft.forward_project(vol.flip(-1), -ang).cpu().flip(-1)
    s_half = ft.forward_project(vol, ang + 180.0).cpu().flip(-1)
    assert _maxrel(s_flip, s) < 2 * _tol(device)
    assert _maxrel(s_half, s) < 2 * _tol(device)


# ── 4. unified semantics: parallel = rotmat at pure tilts ─────────────────

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,N_det", [((3, 11, 12), 12), ((2, 9, 8), 13), ((1, 1, 9), 9),
                                          ((2, 9, 1), 9), ((2, 2, 7), 7)])
def test_parallel_forward_equals_rotmat_pure_tilt(device, shape, N_det):
    """forward_project(v, th) == forward_project_3d(v, (0, th, 0)) on
    box-filling volumes, degenerate Nz = 1 / Nx = 1 / Nz = 2 slices included
    (the tight rule projected those to 0, or to half)."""
    g = torch.Generator().manual_seed(sum(shape))
    vol = (torch.rand(*shape, generator=g) + 0.5).to(device)
    ang = torch.tensor([-61.3, -30.0, 0.0, 17.3, 45.0, 90.0])
    eul = torch.stack([torch.zeros(6), ang, torch.zeros(6)], 1)
    par = ft.forward_project(vol, ang.to(device), N_det=N_det).cpu()
    rot = ft.forward_project_3d(vol, eul.to(device), N_det=N_det).cpu()
    assert par.abs().min(dim=-1).values.max() > 0
    assert _maxrel(par, rot) < _tol(device)
    ref = _fp_parallel_ref(vol.double().cpu().numpy(), ang.double().numpy(), N_det)
    assert _maxrel(par, ref) < _tol(device)


# ── 5. the CPU pair: exact transpose ──────────────────────────────────────

def test_cpu_forward_and_back_projection_are_exact_transposes():
    """Dense check at Nz != Nx, N_det != Nx with grid-aligned views: the
    matrices of the forward and of the ray-driven back projection are each
    other's transpose bit for bit (a unit input sums the same fp32 weights
    in the same step order on both sides), outer ring included."""
    Nz, Nx, Nd = 12, 11, 14
    ang = torch.tensor([-90.0, -37.9, 0.0, 22.5, 45.0, 90.0])
    n = len(ang)
    A = torch.stack([_forward_project_parallel_raw(e.reshape(1, Nz, Nx), ang, Nd).reshape(-1)
                     for e in torch.eye(Nz * Nx)], 1)                    # [n*Nd, Nz*Nx]
    B = torch.stack([_back_project_parallel_raw(e.reshape(n, 1, Nd), ang, Nz, Nx=Nx).reshape(-1)
                     for e in torch.eye(n * Nd)], 1)                     # [Nz*Nx, n*Nd]
    assert torch.equal(A.T, B)
    edge = A.reshape(n, Nd, Nz, Nx)
    assert edge[:, :, :, -1].abs().sum() > 0 and edge[:, :, -1, :].abs().sum() > 0


@pytest.mark.parametrize("Ny", [1, 2, 16, 17, 33])
def test_cpu_pair_inner_products(Ny):
    g = torch.Generator().manual_seed(Ny)
    Nz, Nx, Nd = 10, 13, 15
    ang = torch.tensor([-70.0, -45.0, 0.0, 12.7, 90.0])
    x = torch.rand(Ny, Nz, Nx, generator=g) + 0.5
    y = torch.randn(5, Ny, Nd, generator=g)
    lhs = (_forward_project_parallel_raw(x, ang, Nd).double() * y.double()).sum()
    rhs = (x.double() * _back_project_parallel_raw(y, ang, Nz, Nx=Nx).double()).sum()
    assert abs(float(lhs - rhs)) < 1e-5 * float(lhs.abs())


# ── 6. grad_angle = central finite difference ─────────────────────────────

def _volumes(shape, seed):
    g = torch.Generator().manual_seed(seed)
    Ny, Nz, Nx = shape
    z = torch.arange(Nz, dtype=torch.float64) - 0.5 * (Nz - 1)
    x = torch.arange(Nx, dtype=torch.float64) - 0.5 * (Nx - 1)
    s = 0.3 * max(Nz, Nx)
    blob = torch.exp(-((x[None, :] - 1.0) ** 2 + (z[:, None] + 0.5) ** 2) / (2 * s * s))
    return {"ones": torch.ones(shape),
            "rand": torch.rand(*shape, generator=g) + 0.5,
            "smooth": blob.float().expand(Ny, Nz, Nx).contiguous()}


def _fd_angles(vol, w, ang, eps=1e-5):
    v, wn = vol.double().numpy(), w.double().numpy()
    fd = np.zeros(len(ang))
    for i in range(len(ang)):
        e = eps * (np.arange(len(ang)) == i)
        fd[i] = ((wn * _fp_parallel_ref(v, ang + e, wn.shape[-1])).sum()
                 - (wn * _fp_parallel_ref(v, ang - e, wn.shape[-1])).sum()) / (2 * eps)
    return fd                                                 # per degree


def _grad_angle_exact_trig(vol, w, ang, device):
    """grad_angle through the raw binding with exact trig tables: cos/sin of
    multiples of 90 deg are exactly 0 / +-1 (torch's fp32 cos(90 deg) is
    -4.4e-8, which moves the far samples of a +-90 / 180 deg view just off
    their grid line; see the note in test_grad_angle_matches_fd_at_grid_aligned_views)."""
    r = np.deg2rad(ang)
    c, s = np.cos(r), np.sin(r)
    c[np.abs(c) < 1e-12] = 0.0
    s[np.abs(s) < 1e-12] = 0.0
    ct = torch.tensor(c, dtype=torch.float32, device=device)
    st = torch.tensor(s, dtype=torch.float32, device=device)
    v = vol.to(device).contiguous()
    gs = w.to(device).contiguous()
    out = torch.zeros(len(ang), device=device)
    Ny, Nz, Nx = v.shape
    args = (v.data_ptr(), gs.data_ptr(), ct.data_ptr(), st.data_ptr(), out.data_ptr(),
            Ny, Nz, Nx, len(ang), gs.shape[-1])
    if device == "cpu":
        C.grad_angle_cpu(*args)
    else:
        C.grad_angle_cuda(*args, stream_ptr=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
    return out.cpu().double() * (math.pi / 180.0)


_GA_SHAPES = [((2, 11, 12), 12), ((2, 9, 9), 9), ((1, 8, 9), 11), ((1, 1, 7), 7), ((1, 6, 1), 5)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,N_det", _GA_SHAPES)
def test_grad_angle_matches_fd_at_grid_aligned_views(device, shape, N_det):
    """At 0, +-90, 45 and 180 deg (samples on grid lines) and at generic
    views, on volumes that touch every face, the analytic gradient is the
    central FD of the forward (the forward difference was 0.2-26x off at the
    grid-aligned views). The kernels are called with exact trig tables: with
    torch's fp32 table a +-90 / 180 deg view is 2.5e-6 / 5e-6 deg past the
    alignment, and a sample whose offset survives fp32 rounding is off the
    grid line, where the one-sided derivative of that view is the exact one.
    0 and 45 deg are exact in fp32 (checked at the API level below)."""
    ang = np.array([0.0, 90.0, -90.0, 45.0, 180.0, -47.3, 8.1])
    g = torch.Generator().manual_seed(7)
    for kind, vol in _volumes(shape, 3).items():
        w = torch.randn(len(ang), shape[0], N_det, generator=g)
        fd = _fd_angles(vol, w, ang)
        got = _grad_angle_exact_trig(vol, w, ang, device)
        assert _maxrel(got, fd) < 1e-4, kind


@pytest.mark.parametrize("device", DEVICES)
def test_grad_angle_api_matches_fd_at_zero_and_45_degrees(device):
    ang = np.array([0.0, 45.0, -12.9, 33.7])
    g = torch.Generator().manual_seed(8)
    for kind, vol in _volumes((3, 11, 12), 4).items():
        w = torch.randn(len(ang), 3, 12, generator=g)
        fd = _fd_angles(vol, w, ang)
        got = _grad_angle_raw(vol.to(device), w.to(device),
                              torch.tensor(ang, dtype=torch.float32, device=device))
        assert _maxrel(got, fd) < 1e-4, kind


def test_grad_angle_is_symmetric_under_x_flip():
    """dL/dth(v, th) = -dL/dth(flip_x v, -th) with the flipped weights, at
    0 deg too (the forward difference broke it at every grid-aligned view)."""
    g = torch.Generator().manual_seed(9)
    vol = torch.rand(2, 9, 10, generator=g) + 0.5
    w = torch.randn(3, 2, 10, generator=g)
    ang = torch.tensor([0.0, 20.0, -33.0])
    a = _grad_angle_raw(vol, w, ang)
    b = _grad_angle_raw(vol.flip(-1), w.flip(-1), -ang)
    assert _maxrel(-b, a) < 1e-5


# ── 7. grad_rotmat = central finite difference at kinks ───────────────────

def _smooth_top(shape):
    """Smooth, and non-zero on the top y slice and on the faces."""
    Ny, Nz, Nx = shape
    y, z, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) - 0.5 * (n - 1)
                               for n in shape), indexing="ij")
    s = 0.3 * max(shape)
    return torch.exp(-((x - 1) ** 2 + (y - 0.5 * Ny + 0.5) ** 2 + (z + 0.5) ** 2)
                     / (2 * s * s)).float()


_POSES = {
    "identity": [[0.0, 0.0, 0.0]],
    "pure_tilts": [[0.0, -47.3, 0.0], [0.0, 12.1, 0.0], [0.0, 90.0, 0.0]],
    "generic": [[-4.0, 33.0, 2.5], [3.1, -58.0, -1.7]],
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,N_det", [((6, 10, 11), 12), ((5, 9, 9), 9), ((1, 7, 8), 9)])
@pytest.mark.parametrize("pose", list(_POSES))
def test_grad_rotmat_matches_fd_at_kinks(device, shape, N_det, pose):
    """All 9 entries of dL/dM equal the central FD at the identity view and
    at pure tilts (every sample has wy == 0) as well as at generic poses, on
    face-touching volumes (the forward difference was 20 % to 5.7x off at the
    kinks). Matrices with exact 0 / +-1 entries (the fp32 table of the
    90 deg tilt has cos = -4.4e-8, see the grad_angle note above)."""
    mats = euler_zyz_to_rotmat(torch.tensor(_POSES[pose])).reshape(-1, 9)
    mats = torch.where(mats.abs() < 1e-6, torch.zeros_like(mats), mats)
    g = torch.Generator().manual_seed(21)
    vols = {"smooth": _smooth_top(shape), "rand": torch.rand(*shape, generator=g) + 0.5}
    for kind, vol in vols.items():
        w = torch.randn(mats.shape[0], shape[0], N_det, generator=g)
        fd = _fd_grad_mats(vol.double().numpy(), mats.double().numpy(), w.double().numpy())
        got = _grad_rotmat_raw(vol.to(device), w.to(device), mats.to(device))
        assert _maxrel(got, fd) < 1e-4, kind


@pytest.mark.parametrize("device", DEVICES)
def test_euler_gradient_at_pure_tilt_matches_fd(device):
    """d<w, FP3d(v, euler)>/d euler at a pure tilt (the pose-refinement
    start) is the central FD in all three angles (dphi / dpsi were 3-10 %
    off with the forward difference)."""
    shape = (6, 10, 11)
    vol = _smooth_top(shape) + 0.2
    eul = torch.tensor([[0.0, -31.0, 0.0], [0.0, 17.0, 0.0]])
    g = torch.Generator().manual_seed(5)
    w = torch.randn(2, shape[0], shape[2], generator=g)
    e = eul.clone().to(device).requires_grad_(True)
    (ft.forward_project_3d(vol.to(device), e) * w.to(device)).sum().backward()

    def loss(E):
        r = math.pi / 180.0
        out = 0.0
        for i, (p, t, s) in enumerate(E):
            cp, sp, ct, st, cu, su = (math.cos(p * r), math.sin(p * r), math.cos(t * r),
                                      math.sin(t * r), math.cos(s * r), math.sin(s * r))
            M = np.array([cu * ct * cp - su * sp, -cu * ct * sp - su * cp, cu * st,
                          su * ct * cp + cu * sp, -su * ct * sp + cu * cp, su * st,
                          -st * cp, st * sp, ct])
            out += (w[i].double().numpy() * _fp_rotmat_ref(vol.double().numpy(), M[None],
                                                           shape[2])[0]).sum()
        return out

    E0 = eul.double().numpy()
    fd = np.zeros_like(E0)
    eps = 1e-5
    for i in range(E0.shape[0]):
        for k in range(3):
            Ep, Em = E0.copy(), E0.copy()
            Ep[i, k] += eps
            Em[i, k] -= eps
            fd[i, k] = (loss(Ep) - loss(Em)) / (2 * eps)
    assert _maxrel(e.grad, fd) < 1e-3


# ── 8. regression: edge columns of reconstructions, refine_angles ─────────

def _chord_rect(t, th, cx, cz, a, b):
    c, s = math.cos(th), math.sin(th)
    p, d = (t * c - cx, t * s - cz), (-s, c)
    lo, hi = -1e18, 1e18
    for k, h in ((0, a), (1, b)):
        if abs(d[k]) < 1e-15:
            if abs(p[k]) > h:
                return 0.0
            continue
        u, v = (-h - p[k]) / d[k], (h - p[k]) / d[k]
        lo, hi = max(lo, min(u, v)), min(hi, max(u, v))
    return max(0.0, hi - lo)


def _slab(N):
    """ET slab filling the field of view laterally, plus a feature at the
    right edge: (density, (cx, cz, half width, half height))."""
    return [(1.0, (0.0, 0.0, N / 2, 0.22 * N)), (0.6, (0.42 * N, 0.0, 0.08 * N, 0.1 * N))]


def _slab_data(N, angs, Ny=2):
    """Analytic line integrals (no inverse crime) and the pixel-averaged
    truth of the slab."""
    hd = 0.5 * (N - 1)
    y = np.array([[sum(r * _chord_rect(i - hd, math.radians(a), *p) for r, p in _slab(N))
                   for i in range(N)] for a in angs])
    c = np.arange(N) - 0.5 * (N - 1)
    img = np.zeros((N, N))
    for r, (cx, cz, a, b) in _slab(N):
        ox = np.clip(np.minimum(c + 0.5, cx + a) - np.maximum(c - 0.5, cx - a), 0, None)
        oz = np.clip(np.minimum(c + 0.5, cz + b) - np.maximum(c - 0.5, cz - b), 0, None)
        img += r * oz[:, None] * ox[None, :]
    return torch.tensor(y, dtype=torch.float32)[:, None].repeat(1, Ny, 1), img


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("alg", ["sirt", "fista"])
def test_reconstruction_edge_columns_are_unbiased(device, alg):
    """On analytic data of a slab that fills the field of view, the first
    and the last column of the reconstruction have the same, small error.
    Measured at N = 32, 100 iterations (col 0 / col N-1 mean error):
    tight rule CPU sirt +0.13 / +0.52, CUDA fista +0.18 / +1.52; loose rule
    +0.042 / +0.044 and +0.019 / +0.023 on both backends."""
    N = 32
    angs = np.linspace(-60, 60, 61)
    y, truth = _slab_data(N, angs)
    a = torch.tensor(angs, dtype=torch.float32, device=device)
    if alg == "sirt":
        x = ft.run_sirt(y.to(device), a, 100)
    else:
        torch.manual_seed(0)
        x = ft.run_fista(y.to(device), a, 100)
    e = x[1].double().cpu().numpy() - truth
    assert abs(e[:, -1].mean() - e[:, 0].mean()) < 0.02
    assert max(abs(e[:, 0].mean()), abs(e[:, -1].mean())) < 0.08


@pytest.mark.parametrize("device", DEVICES)
def test_refine_angles_keeps_true_angles_on_loose_data(device):
    """refine_angles started at the true angles of pure-tilt rotmat data of
    a slab that fills the field of view stays there (the tight parallel
    search walked off by 0.45 deg rms, 1.5 deg max, on this case)."""
    Ny, N = 2, 32
    y_, z_, x_ = np.meshgrid(np.arange(Ny) - 0.5, np.arange(N) - 15.5, np.arange(N) - 15.5,
                             indexing="ij")
    vol = ((np.abs(z_) < 7) * (1.0 + 0.3 * np.sin(x_ / 2.5) + 0.2 * np.cos(y_ + z_ / 1.5)))
    vol = torch.tensor(vol + 0.8 * np.exp(-((z_ - 2) ** 2 + (x_ - 12) ** 2) / 4.0),
                       dtype=torch.float32, device=device)
    true = torch.linspace(-60, 60, 21)
    eul = torch.stack([0 * true, true, 0 * true], 1).to(device)
    sino = ft.forward_project_3d(vol, eul)
    a = true.clone()
    for rg in (1.0, 0.1, 0.01):
        a = ft.refine_angles(vol, sino, a.to(device), search_range_deg=rg, n_steps=21)
    err = (a.cpu() - true).abs()
    assert err.max() < 0.01


# ── 10. adversarial inputs ────────────────────────────────────────────────

@pytest.mark.parametrize("device", DEVICES)
def test_zero_and_large_face_touching_volumes(device):
    ang = torch.tensor([0.0, 30.0, 90.0], device=device)
    assert ft.forward_project(torch.zeros(2, 9, 8, device=device), ang).abs().max() == 0
    g = torch.Generator().manual_seed(3)
    big = (1e4 * (torch.rand(2, 9, 8, generator=g) + 0.5)).to(device)
    s = ft.forward_project(big, ang).cpu()
    ref = _fp_parallel_ref(big.double().cpu().numpy(), ang.double().cpu().numpy(), 8)
    assert _maxrel(s, ref) < _tol(device)


@needs_cuda
@pytest.mark.parametrize("Ny", [5, 6, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_packed_channels_match_single_slices_on_face_touching_volume(Ny, dtype):
    """The float2 (fp32) and 4-slice fp16 packed textures give every slice
    the bits of the 1-channel kernel (Ny = 1), outer ring included."""
    g = torch.Generator().manual_seed(Ny)
    vol = (torch.rand(Ny, 11, 10, generator=g) + 0.5).to("cuda", dtype)
    ang = torch.tensor([-90.0, -17.3, 0.0, 45.0, 90.0], device="cuda")
    packed = ft.forward_project(vol, ang, N_det=12)
    for iy in range(Ny):
        single = ft.forward_project(vol[iy:iy + 1].contiguous(), ang, N_det=12)
        assert torch.equal(packed[:, iy:iy + 1], single), iy
