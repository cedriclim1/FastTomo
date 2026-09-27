"""
Python-side correctness tests (PHASE1 section 4).

Each block names the item it covers. Tests are parametrised over the
``backend`` fixture: the CPU case always runs, the CUDA case skips when no
GPU is visible. Random inputs use fixed seeds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import fasttomo as ft

HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA GPU available")

BACKENDS = ["cpu", pytest.param("cuda", marks=needs_cuda)]


@pytest.fixture(params=BACKENDS)
def backend(request) -> torch.device:
    return torch.device(request.param)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double().cpu()
    b = b.detach().double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _blob(Ny: int, Nz: int, Nx: int, device, c=(0.1, -0.2), s=4.0) -> torch.Tensor:
    """Smooth Gaussian blob, constant along Y. Smooth inputs keep the CUDA
    texture back-projection close to the exact adjoint (~1 % instead of
    15-180 % for white noise), so the inner-product test is meaningful."""
    z = torch.arange(Nz, device=device) - (Nz - 1) / 2
    x = torch.arange(Nx, device=device) - (Nx - 1) / 2
    g = torch.exp(-((z[:, None] - c[0] * Nz) ** 2 + (x[None, :] - c[1] * Nx) ** 2) / (2 * s * s))
    return g.expand(Ny, Nz, Nx).contiguous()


# ── back_project(Nx=...) and autograd with N_det != Nx ─────────────────────

NX_NDET = [(24, 32), (32, 24), (24, 24)]


@pytest.mark.parametrize("Nx,N_det", NX_NDET)
def test_back_project_nx_shape_and_default(backend, Nx, N_det):
    """back_project(Nx=...) returns [Ny, Nz, Nx]; the default is still N_det."""
    ang = torch.linspace(-60, 60, 17, device=backend)
    sino = torch.randn(17, 3, N_det, generator=torch.Generator().manual_seed(1)).to(backend)
    vol = ft.back_project(sino, ang, Nz=20, Nx=Nx)
    assert vol.shape == (3, 20, Nx) and vol.dtype == torch.float32
    assert ft.back_project(sino, ang, Nz=20).shape == (3, 20, N_det)
    assert torch.equal(ft.back_project(sino, ang, Nz=20),
                       ft.back_project(sino, ang, Nz=20, Nx=N_det))


@pytest.mark.parametrize("Nx,N_det", NX_NDET)
def test_forward_back_adjoint_nx_ne_ndet(backend, Nx, N_det):
    """<A x, y> == <x, A^T y> with A = forward_project(., N_det) on a
    [Ny, Nz, Nx] volume and A^T = back_project(., Nx=Nx).

    CPU: the scatter BP is the exact transpose, so random inputs agree to
    fp32 round-off. CUDA: the texture-gather BP is an approximate adjoint;
    with smooth inputs it is within ~1.1 % for every geometry, including the
    square one, so a 3 % bound catches any mis-centring of Nx vs N_det.
    """
    Ny, Nz, n = 2, 20, 17
    ang = torch.linspace(-60, 60, n, device=backend)
    if backend.type == "cpu":
        g = torch.Generator().manual_seed(0)
        x = torch.randn(Ny, Nz, Nx, generator=g)
        y = torch.randn(n, Ny, N_det, generator=g)
        tol = 1e-5
    else:
        x = _blob(Ny, Nz, Nx, backend)
        y = ft.forward_project(_blob(Ny, Nz, Nx, backend, c=(-0.15, 0.1), s=3.0), ang, N_det=N_det)
        tol = 3e-2
    lhs = (ft.forward_project(x, ang, N_det=N_det) * y).double().sum()
    rhs = (x * ft.back_project(y, ang, Nz=Nz, Nx=Nx)).double().sum()
    assert float(abs(lhs - rhs) / abs(lhs)) < tol


@pytest.mark.parametrize("Nx,N_det", [(24, 32), (32, 24)])
def test_forward_project_autograd_nx_ne_ndet(backend, Nx, N_det):
    """Backward of forward_project with N_det != Nx used to raise
    ('_FPParallelBackward returned an invalid gradient'). Now the volume
    gradient is back_project(grad, Nx=Nx) and the angle gradient exists."""
    Ny, Nz, n = 3, 20, 17
    g = torch.Generator().manual_seed(2)
    x = torch.randn(Ny, Nz, Nx, generator=g).to(backend).requires_grad_(True)
    w = torch.randn(n, Ny, N_det, generator=g).to(backend)
    ang = torch.linspace(-60, 60, n, device=backend).requires_grad_(True)
    (ft.forward_project(x, ang, N_det=N_det) * w).sum().backward()
    assert x.grad.shape == x.shape and ang.grad.shape == ang.shape
    assert torch.isfinite(ang.grad).all()
    ref = ft.back_project(w, ang.detach(), Nz=Nz, Nx=Nx)
    assert torch.equal(x.grad, ref)

    # and back_project's own backward is forward_project(., N_det)
    s = w.clone().requires_grad_(True)
    (ft.back_project(s, ang.detach(), Nz=Nz, Nx=Nx) * x.detach()).sum().backward()
    assert torch.equal(s.grad, ft.forward_project(x.detach(), ang.detach(), N_det=N_det))


def test_forward_project_angle_grad_fd_nx_ne_ndet():
    """CPU angle gradient with N_det > Nx matches central finite differences
    of the loss (the forward kernel and grad_angle share one operator)."""
    Ny, Nz, Nx, N_det, n = 1, 24, 24, 34, 5
    x = _blob(Ny, Nz, Nx, "cpu", s=3.0) + 0.5 * _blob(Ny, Nz, Nx, "cpu", c=(-0.2, 0.2), s=2.0)
    w = torch.linspace(0.5, 1.5, N_det).expand(n, Ny, N_det).contiguous()
    ang = torch.tensor([-50.0, -20.0, 5.0, 30.0, 55.0], requires_grad=True)
    (ft.forward_project(x, ang, N_det=N_det) * w).sum().backward()
    h = 1e-2
    fd = []
    for i in range(n):
        a_p = ang.detach().clone()
        a_p[i] += h
        a_m = ang.detach().clone()
        a_m[i] -= h
        lp = (ft.forward_project(x, a_p, N_det=N_det) * w).double().sum()
        lm = (ft.forward_project(x, a_m, N_det=N_det) * w).double().sum()
        fd.append(float((lp - lm) / (2 * h)))
    fd = torch.tensor(fd)
    assert _rel(ang.grad, fd) < 2e-2, (ang.grad, fd)


@pytest.mark.parametrize("Nx,N_det", [(24, 32), (32, 24)])
def test_back_project_nx_matches_centre_crop(backend, Nx, N_det):
    """Geometry check independent of the adjoint: the gather BP evaluates
    each voxel at its physical position, so BP onto a narrower volume is the
    centre crop of BP onto a wider one (even width difference). Exact on
    CUDA; on CPU (ray-driven scatter) the outermost voxel ring differs
    because the wider volume's bilinear stencils straddle the crop edge."""
    Ny, Nz, n = 2, 20, 17
    ang = torch.linspace(-60, 60, n, device=backend)
    y = torch.randn(n, Ny, N_det, generator=torch.Generator().manual_seed(3)).to(backend)
    narrow, wide = min(Nx, N_det), max(Nx, N_det)
    off = (wide - narrow) // 2
    small = ft.back_project(y, ang, Nz=Nz, Nx=narrow)
    big = ft.back_project(y, ang, Nz=Nz, Nx=wide)[:, :, off:off + narrow]
    if backend.type == "cuda":
        assert torch.equal(small, big)
    else:
        torch.testing.assert_close(small[:, 1:-1, 1:-1], big[:, 1:-1, 1:-1], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("Nx,N_det", [(24, 32), (32, 24)])
def test_forward_project_3d_autograd_nx_ne_ndet(backend, Nx, N_det):
    """Same fix for the rotation-matrix pair: forward_project_3d with
    N_det != Nx has a working backward, equal to back_project_3d(Nx=Nx), and
    back_project_3d(Nx=...) is the centre crop of the wider BP."""
    Ny, Nz, n = 3, 20, 9
    g = torch.Generator().manual_seed(4)
    x = torch.randn(Ny, Nz, Nx, generator=g).to(backend).requires_grad_(True)
    w = torch.randn(n, Ny, N_det, generator=g).to(backend)
    eul = torch.zeros(n, 3, device=backend)
    eul[:, 1] = torch.linspace(-60, 60, n, device=backend)
    eul.requires_grad_(True)
    (ft.forward_project_3d(x, eul, N_det=N_det) * w).sum().backward()
    assert x.grad.shape == x.shape and eul.grad.shape == eul.shape
    ref = ft.back_project_3d(w, eul.detach(), Nz=Nz, Nx=Nx)
    assert torch.equal(x.grad, ref)
    s = w.clone().requires_grad_(True)
    (ft.back_project_3d(s, eul.detach(), Nz=Nz, Nx=Nx) * x.detach()).sum().backward()
    assert torch.equal(s.grad, ft.forward_project_3d(x.detach(), eul.detach(), N_det=N_det))
    narrow, wide = min(Nx, N_det), max(Nx, N_det)
    off = (wide - narrow) // 2
    small = ft.back_project_3d(w, eul.detach(), Nz=Nz, Nx=narrow)
    big = ft.back_project_3d(w, eul.detach(), Nz=Nz, Nx=wide)[:, :, off:off + narrow]
    torch.testing.assert_close(small, big, rtol=0, atol=0)


# ── ConstraintConfig.soft_threshold (formerly sparsity) honoured by run_ad_recon ─────────────────────

class _MeanZProjector:
    """CPU stand-in for ParallelProjector: every view is the Z-mean of the
    volume. Linear, so the fidelity gradient is non-zero for every voxel."""

    def __init__(self, n_angles: int):
        self.n_angles = n_angles

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return volume.mean(dim=1).unsqueeze(0).expand(self.n_angles, -1, -1)


def _cpu_ad_recon(monkeypatch, *, n_angles=4, Ny=2, N=6, projector=None):
    """Patch run_ad_recon's CUDA-only input check and projector class so the
    loop runs on CPU with ``projector`` (default `_MeanZProjector`)."""
    import fasttomo.algorithms as algo

    proj = projector if projector is not None else _MeanZProjector(n_angles)

    def fake_check_inputs(sinogram, angles_deg, Nz, **kwargs):
        n, ny, nd = sinogram.shape
        return n, ny, nd, (nd if Nz < 0 else Nz), sinogram.device

    monkeypatch.setattr(algo, "ParallelProjector", type(proj))
    monkeypatch.setattr(algo, "_check_inputs", fake_check_inputs)
    return proj


@pytest.mark.parametrize("lam,expected", [(0.0, 0.2), (0.05, 0.15), (0.3, 0.0)])
def test_ad_recon_constraint_sparsity_soft_thresholds(monkeypatch, lam, expected):
    """One Adam step from a zero volume moves every voxel by exactly
    lr_volume (Adam's first step is lr * g / (|g| + eps)); the post-step
    soft-threshold then leaves |v| = max(lr - lam, 0). Before the fix
    ConstraintConfig.sparsity was ignored and |v| stayed 0.2."""
    proj = _cpu_ad_recon(monkeypatch)
    sino = torch.full((4, 2, 6), 3.0)           # target above the start: v grows
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=1, lr_volume=0.2, cosine_schedule=False),
        constraints=ft.ConstraintConfig(soft_threshold=lam),
    )
    rec = ft.run_ad_recon(sino, torch.linspace(-30, 30, 4), Nz=6,
                          projector=proj, config=cfg)
    torch.testing.assert_close(rec, torch.full_like(rec, expected), rtol=0, atol=1e-6)


def test_ad_recon_constraint_order_mask_then_threshold(monkeypatch):
    """The soft-threshold runs after the support mask. A soft mask tells
    the orders apart: every voxel is +0.2 after the first Adam step;
    mask-then-threshold gives max(m * 0.2 - lam, 0), threshold-then-mask
    gives m * max(0.2 - lam, 0). With m = 0.5, lam = 0.05 these are 0.05 and
    0.075. (With a binary mask both orders agree, so a binary mask could not
    pin the order.) A voxel outside the mask (m = 0) ends at 0 either way,
    and positivity commutes with a non-negative mask, so its position
    relative to the mask is not observable."""
    proj = _cpu_ad_recon(monkeypatch)
    mask = torch.ones(6, 6)
    mask[:, :2] = 0.5                       # soft region
    mask[:, 4:] = 0.0                       # outside the support
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=1, lr_volume=0.2, cosine_schedule=False),
        constraints=ft.ConstraintConfig(positivity=True, soft_threshold=0.05),
    )
    rec = ft.run_ad_recon(torch.full((4, 2, 6), 3.0), torch.linspace(-30, 30, 4),
                          Nz=6, projector=proj, config=cfg, support_mask=mask)
    torch.testing.assert_close(rec[:, :, :2], torch.full((2, 6, 2), 0.05), rtol=0, atol=1e-6)
    torch.testing.assert_close(rec[:, :, 2:4], torch.full((2, 6, 2), 0.15), rtol=0, atol=1e-6)
    assert torch.equal(rec[:, :, 4:], torch.zeros(2, 6, 2))
    # and ConstraintConfig.apply uses the same order on the same input
    applied = ft.ConstraintConfig(positivity=True, support_mask=mask.unsqueeze(0), soft_threshold=0.05).apply(
        torch.full((2, 6, 6), 0.2))
    torch.testing.assert_close(rec, applied, rtol=0, atol=1e-6)


def test_ad_recon_constraint_sparsity_matches_config_apply(monkeypatch):
    """Several iterations: run_ad_recon's result equals a hand-written loop
    (Adam step, then ConstraintConfig.apply) with the same settings."""
    proj = _cpu_ad_recon(monkeypatch)
    g = torch.Generator().manual_seed(5)
    sino = torch.rand(4, 2, 6, generator=g) * 2 - 0.5
    cons = ft.ConstraintConfig(positivity=False, soft_threshold=0.02)
    cfg = ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=6, lr_volume=0.1, cosine_schedule=False),
        constraints=cons,
    )
    rec = ft.run_ad_recon(sino, torch.linspace(-30, 30, 4), Nz=6,
                          projector=proj, config=cfg)

    v = torch.zeros(2, 6, 6, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.1)
    for _ in range(6):
        opt.zero_grad()
        torch.nn.functional.mse_loss(proj.forward(v), sino).backward()
        opt.step()
        with torch.no_grad():
            v.copy_(cons.apply(v))
    assert torch.equal(rec, v.detach())
    cfg.constraints = ft.ConstraintConfig(positivity=False)
    unthresholded = ft.run_ad_recon(sino, torch.linspace(-30, 30, 4), Nz=6,
                                    projector=proj, config=cfg)
    assert not torch.equal(rec, unthresholded)


@needs_cuda
def test_ad_recon_constraint_sparsity_cuda():
    """Real projector: the soft-threshold zeroes the small background voxels
    that plain AD leaves as noise, and sparsity=0 keeps the default result
    bit-identical."""
    torch.manual_seed(0)
    N, n = 32, 25
    ang = torch.linspace(-60, 60, n, device="cuda")
    vol = torch.zeros(2, N, N, device="cuda")
    vol[:, 10:20, 12:22] = 1.0
    sino = ft.forward_project(vol, ang)
    sino = sino + 0.05 * torch.randn_like(sino)
    base = ft.AdReconConfig(optim=ft.OptimConfig(n_iter=40, lr_volume=0.05))
    rec0 = ft.run_ad_recon(sino, ang, Nz=N, config=base)
    rec0b = ft.run_ad_recon(sino, ang, Nz=N, config=ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=40, lr_volume=0.05),
        constraints=ft.ConstraintConfig(soft_threshold=0.0)))
    assert torch.equal(rec0, rec0b)
    rec = ft.run_ad_recon(sino, ang, Nz=N, config=ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=40, lr_volume=0.05),
        constraints=ft.ConstraintConfig(soft_threshold=0.01)))
    bg = vol == 0
    assert (rec[bg] == 0).float().mean() > 0.5 > (rec0[bg] == 0).float().mean()
    assert _rel(rec, vol) < 1.0


# ── wSIRT: functional recovery ─────────────────────────────────────────────

def _wsirt_phantom(device, N=48):
    vol = torch.zeros(2, N, N, device=device)
    vol[:, 12:36, 14:30] = 1.0
    vol[1, 20:28, 20:26] = 2.0
    ang = torch.linspace(-70, 70, 41, device=device)
    return vol, ang, ft.forward_project(vol, ang)


def test_wsirt_recovers_phantom(backend):
    """wSIRT on a +-70 deg, 41-view phantom: the data residual falls by more
    than 100x within 5 iterations, and the volume error after 20 iterations
    is below 35 % (limited-angle floor: SIRT reaches ~29 %). Output has the
    [Ny, Nz, Nx] shape, fp32 dtype and the input device."""
    vol, ang, sino = _wsirt_phantom(backend)
    rec, loss = ft.run_wsirt(sino, ang, n_iter=20, Nz=48, return_loss=True)
    assert rec.shape == vol.shape and rec.dtype == torch.float32
    assert rec.device.type == backend.type
    assert loss.shape == (20,) and torch.isfinite(loss).all()
    assert loss[5] < loss[0] / 100
    assert _rel(rec, vol) < 0.35


@needs_cuda
def test_wsirt_converges_faster_than_sirt_cuda():
    """The point of wSIRT (Wolf et al. 2014): the exact-filter W-BP step gets
    close in a handful of iterations where plain SIRT is still far away."""
    vol, ang, sino = _wsirt_phantom(torch.device("cuda"))
    w = ft.run_wsirt(sino, ang, n_iter=5, Nz=48)
    s = ft.run_sirt(sino, ang, n_iter=5, Nz=48)
    assert _rel(w, vol) < 0.8 * _rel(s, vol)


def test_wsirt_constraints(backend):
    """positivity, mask (cylinder) and the sparsity soft-threshold are
    honoured on every backend."""
    vol, ang, sino = _wsirt_phantom(backend)
    rec = ft.run_wsirt(sino, ang, n_iter=5, Nz=48, positivity=True, mask=True)
    assert rec.min() >= 0
    outside = ft.cylindrical_mask(48, 48, device=backend) == 0
    assert (rec[:, outside] == 0).all()
    plain = ft.run_wsirt(sino, ang, n_iter=5, Nz=48)
    sparse = ft.run_wsirt(sino, ang, n_iter=5, Nz=48, sparsity=0.05)
    assert (sparse == 0).sum() > (plain == 0).sum()


# ── apply_shifts_2d ────────────────────────────────────────────────────────

def _shift_ref(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """fp64 reference: out[y, x] = bilinear sample of img at (y - sy, x - sx)
    with zeros outside [0, N-1] (grid_sample align_corners=True, padding
    zeros). Positive shifts move content toward larger indices."""
    Ny, Nx = img.shape
    out = np.zeros_like(img, dtype=np.float64)
    for y in range(Ny):
        for x in range(Nx):
            fy, fx = y - sy, x - sx
            y0, x0 = math.floor(fy), math.floor(fx)
            wy, wx = fy - y0, fx - x0
            acc = 0.0
            for dy, wyy in ((0, 1 - wy), (1, wy)):
                for dx, wxx in ((0, 1 - wx), (1, wx)):
                    yy, xx = y0 + dy, x0 + dx
                    if 0 <= yy < Ny and 0 <= xx < Nx:
                        acc += wyy * wxx * img[yy, xx]
            out[y, x] = acc
    return out


@pytest.mark.parametrize("shifts", [
    [(3.0, 0.0), (-2.0, 0.0)],        # integer, detector axis
    [(0.0, 2.0), (0.0, -1.0)],        # integer, tilt axis
    [(0.25, 0.0), (-1.75, 0.0)],      # sub-pixel, detector axis
    [(0.0, 0.5), (0.0, -0.3)],        # sub-pixel, tilt axis
    [(1.4, -0.6), (-0.2, 2.7)],       # both axes
])
def test_apply_shifts_2d_matches_reference(backend, shifts):
    g = torch.Generator().manual_seed(6)
    sino = torch.rand(2, 7, 11, generator=g)
    sh = torch.tensor(shifts, dtype=torch.float32)
    out = ft.apply_shifts_2d(sino.to(backend), sh.to(backend))
    assert out.shape == sino.shape and out.dtype == torch.float32
    ref = np.stack([_shift_ref(sino[i].double().numpy(), *shifts[i]) for i in range(2)])
    torch.testing.assert_close(out.cpu().double(), torch.from_numpy(ref), rtol=1e-5, atol=1e-5)


def test_apply_shifts_2d_integer_is_roll_with_zero_fill(backend):
    """Integer shifts are an exact translation with zero fill."""
    sino = torch.arange(2 * 5 * 9, dtype=torch.float32).reshape(2, 5, 9).to(backend)
    out = ft.apply_shifts_2d(sino, torch.tensor([[2.0, 1.0], [-3.0, 0.0]], device=backend))
    exp = torch.zeros_like(sino)
    exp[0, 1:, 2:] = sino[0, :-1, :-2]
    exp[1, :, :-3] = sino[1, :, 3:]
    torch.testing.assert_close(out, exp, rtol=0, atol=1e-4)


def test_apply_shifts_2d_zero_is_identity_and_grad(backend):
    """Zero shift is the identity; the shifts gradient matches central
    finite differences at a sub-pixel point (grid_sample autograd)."""
    g = torch.Generator().manual_seed(7)
    sino = torch.rand(3, 6, 10, generator=g, dtype=torch.float64).float().to(backend)
    torch.testing.assert_close(ft.apply_shifts_2d(sino, torch.zeros(3, 2, device=backend)),
                               sino, rtol=0, atol=1e-6)
    w = torch.rand(3, 6, 10, generator=g).to(backend)
    sh = torch.tensor([[0.3, -0.4], [1.2, 0.6], [-0.7, 0.2]], device=backend, requires_grad=True)
    (ft.apply_shifts_2d(sino, sh) * w).sum().backward()
    h = 1e-2
    fd = torch.zeros(3, 2)
    for i in range(3):
        for j in range(2):
            p = sh.detach().clone()
            p[i, j] += h
            m = sh.detach().clone()
            m[i, j] -= h
            fd[i, j] = ((ft.apply_shifts_2d(sino, p) - ft.apply_shifts_2d(sino, m)) * w).double().sum() / (2 * h)
    assert _rel(sh.grad, fd) < 1e-3


def test_apply_shifts_2d_rejects_bad_shapes():
    sino = torch.zeros(3, 4, 5)
    with pytest.raises(ValueError, match=r"\[n_angles, 2\]"):
        ft.apply_shifts_2d(sino, torch.zeros(3))
    with pytest.raises(ValueError, match="3 angles"):
        ft.apply_shifts_2d(sino, torch.zeros(2, 2))


# ── tv_loss_xz and logsum_sparsity vs NumPy fp64 references ────────────────

def _tv_xz_ref(v: np.ndarray, eps: float) -> float:
    dz = v[:, 1:, :-1] - v[:, :-1, :-1]
    dx = v[:, :-1, 1:] - v[:, :-1, :-1]
    return float(np.sqrt(dz ** 2 + dx ** 2 + eps).mean())


def _fd_grad(f, v: np.ndarray, h: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(v)
    it = np.nditer(v, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        vp = v.copy()
        vp[i] += h
        vm = v.copy()
        vm[i] -= h
        g[i] = (f(vp) - f(vm)) / (2 * h)
    return g


@pytest.mark.parametrize("shape", [(1, 5, 7), (3, 4, 6), (2, 2, 2)])
def test_tv_loss_xz_value_and_grad(backend, shape):
    """Value and gradient against an fp64 NumPy reference (gradient by
    central finite differences). The Y axis is never differenced: a volume
    that varies only along Y has zero TV-xz and zero gradient."""
    eps = 1e-3
    rng = np.random.default_rng(8)
    v = rng.standard_normal(shape)
    t = torch.tensor(v, dtype=torch.float32, device=backend, requires_grad=True)
    loss = ft.tv_loss_xz(t, eps=eps)
    assert loss.shape == () and loss.dtype == torch.float32
    assert loss.item() == pytest.approx(_tv_xz_ref(v, eps), rel=1e-5)
    loss.backward()
    ref = _fd_grad(lambda a: _tv_xz_ref(a, eps), v)
    torch.testing.assert_close(t.grad.cpu().double(), torch.from_numpy(ref), rtol=1e-3, atol=1e-5)

    y_only = torch.arange(shape[0], dtype=torch.float32, device=backend)[:, None, None]
    y_only = y_only.expand(shape).clone().requires_grad_(True)
    val = ft.tv_loss_xz(y_only, eps=0.0)
    val.backward()
    assert val.item() == 0.0


def test_tv_loss_xz_rejects_bad_input():
    with pytest.raises(TypeError):
        ft.tv_loss_xz(torch.zeros(2, 3, 3, dtype=torch.float64))
    with pytest.raises(ValueError):
        ft.tv_loss_xz(torch.zeros(3, 3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_logsum_sparsity_value_and_grad(backend, dtype):
    """mean(log(1 + |v| / eps)) and its gradient sign(v) / (eps + |v|) / N,
    checked against NumPy fp64; the gradient at v == 0 is 0."""
    eps = 0.05
    v = np.array([[-2.0, -0.3, 0.0], [1e-4, 0.7, 5.0]])
    t = torch.tensor(v, dtype=dtype, device=backend, requires_grad=True)
    loss = ft.logsum_sparsity(t, eps=eps)
    assert loss.shape == () and loss.dtype == dtype
    ref = np.log1p(np.abs(v) / eps).mean()
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    assert loss.item() == pytest.approx(ref, rel=tol)
    loss.backward()
    gref = np.sign(v) / (eps + np.abs(v)) / v.size
    torch.testing.assert_close(t.grad.cpu().double(), torch.from_numpy(gref), rtol=tol * 10, atol=tol)
    assert t.grad[0, 2].item() == 0.0

    # random volume: gradient also matches finite differences of the reference
    rng = np.random.default_rng(9)
    v = rng.standard_normal((2, 3, 4))
    t = torch.tensor(v, dtype=torch.float64, device=backend, requires_grad=True)
    ft.logsum_sparsity(t, eps=eps).backward()
    fd = _fd_grad(lambda a: float(np.log1p(np.abs(a) / eps).mean()), v)
    torch.testing.assert_close(t.grad.cpu(), torch.from_numpy(fd), rtol=1e-5, atol=1e-8)


# ── RESIRE raw wrappers: shape guard, outputs unchanged ────────────────────
#
# RESIRE is frozen: the guard only adds checks before the unchanged kernel
# calls. The golden file was generated by `_resire_cases()` running against
# origin/main's fasttomo/ (98fa473). The RESIRE forward kernel scatters with
# fp32 atomicAdd, so for general angles it is not bit-reproducible even
# main-vs-main; those outputs are compared with torch.equal where the
# arithmetic is exact (identity rotation, dyadic volume: every partial sum
# is exactly representable, so accumulation order cannot matter) and with a
# tolerance at the atomics' noise level otherwise. Every back-projection
# (a gather, no atomics) is compared with torch.equal.

RESIRE_GOLDEN = __import__("pathlib").Path(__file__).parent / "data" / "resire_golden_main.pt"
RESIRE_EXACT = ("bp", "bp_nz", "bp_default", "fp_grad", "fp_exact", "fp_exact_ndet")
RESIRE_NOISY = ("fp", "fp_ndet", "bp_grad_sq", "run", "run_loss", "run_euler")


def _resire_cases() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    Ny, Nz, Nx, n = 5, 18, 22, 9
    vol = torch.rand(Ny, Nz, Nx, generator=g).cuda()
    eul = torch.stack([torch.linspace(-6, 6, n), torch.linspace(-60, 60, n),
                       torch.linspace(3, -3, n)], 1).cuda()
    sino = torch.rand(n, Ny, 26, generator=g).cuda()
    sino_sq = torch.rand(n, Ny, Nx, generator=g).cuda()
    ang = torch.linspace(-60, 60, n).cuda()
    out = {}
    out["fp"] = ft.forward_project_resire(vol, eul)
    out["fp_ndet"] = ft.forward_project_resire(vol, eul, N_det=26)
    out["bp"] = ft.back_project_resire(sino, eul, Nz=Nz)
    out["bp_nz"] = ft.back_project_resire(sino_sq, eul, Nz=11)
    out["bp_default"] = ft.back_project_resire(sino, eul)
    v = vol.clone().requires_grad_(True)
    (ft.forward_project_resire(v, eul) * sino_sq).sum().backward()
    out["fp_grad"] = v.grad
    s = sino_sq.clone().requires_grad_(True)
    (ft.back_project_resire(s, eul, Nz=Nz) * vol).sum().backward()
    out["bp_grad_sq"] = s.grad            # the RESIRE forward: atomics noise
    dyadic = torch.randint(0, 8, (Ny, Nz, Nx), generator=g).float().cuda() / 4
    ident = torch.zeros(4, 3, device="cuda")
    out["fp_exact"] = ft.forward_project_resire(dyadic, ident)
    out["fp_exact_ndet"] = ft.forward_project_resire(dyadic, ident, N_det=Nx + 4)
    sq = ft.forward_project_resire(torch.rand(Ny, Nx, Nx, generator=g).cuda(), eul)
    out["run"], out["run_loss"] = ft.run_resire(sq, ang, n_iter=4, Nz=Nx, return_loss=True)
    out["run_euler"] = ft.run_resire(sq, ang, n_iter=3, Nz=Nx, euler_zyz_deg=eul.cpu(),
                                     mask=True, sparsity=0.001)
    return {k: t.detach().cpu() for k, t in out.items()}


@needs_cuda
def test_resire_outputs_identical_to_main():
    golden = torch.load(RESIRE_GOLDEN)
    now = _resire_cases()
    assert set(golden) == set(RESIRE_EXACT) | set(RESIRE_NOISY) == set(now)
    for k in RESIRE_EXACT:
        assert torch.equal(now[k], golden[k]), k
    for k in RESIRE_NOISY:
        assert now[k].shape == golden[k].shape
        torch.testing.assert_close(now[k], golden[k], rtol=1e-5, atol=1e-5, msg=k)


@needs_cuda
@pytest.mark.parametrize("which,kwargs,match", [
    ("fp", dict(euler=(9, 2)), r"\[n_angles, 3\]"),
    ("fp", dict(euler=(9, 9)), r"\[n_angles, 3\]"),
    ("fp", dict(euler=(9,)), r"\[n_angles, 3\]"),
    ("fp", dict(euler=(0, 3)), "no views"),
    ("fp", dict(data=(0, 6, 6)), "zero-size"),
    ("fp", dict(N_det=0), "N_det must be positive"),
    ("bp", dict(euler=(8, 3)), "9 views but euler_zyz_deg has 8"),
    ("bp", dict(euler=(10, 3)), "9 views but euler_zyz_deg has 10"),
    ("bp", dict(euler=(9, 4)), r"\[n_angles, 3\]"),
    ("bp", dict(data=(9, 0, 6)), "zero-size"),
    ("bp", dict(Nz=0), "Nz and Nx must be positive"),
])
def test_resire_raw_wrappers_reject_bad_shapes(which, kwargs, match):
    """Mismatched sizes raise ValueError before any kernel launch."""
    eul = torch.zeros(kwargs.get("euler", (9, 3)), device="cuda")
    if which == "fp":
        vol = torch.rand(kwargs.get("data", (4, 6, 6)), device="cuda")
        with pytest.raises(ValueError, match=match):
            ft.forward_project_resire(vol, eul, N_det=kwargs.get("N_det", -1))
    else:
        sino = torch.rand(kwargs.get("data", (9, 4, 6)), device="cuda")
        with pytest.raises(ValueError, match=match):
            ft.back_project_resire(sino, eul, Nz=kwargs.get("Nz", -1))


@needs_cuda
def test_run_resire_rejects_mismatched_euler():
    """A 1-row Euler table used to broadcast the forward projection over the
    whole sinogram in the residual; now it raises before iterating."""
    sino = torch.rand(9, 4, 12, device="cuda")
    ang = torch.linspace(-60, 60, 9, device="cuda")
    with pytest.raises(ValueError, match=r"euler_zyz_deg must have shape \(9, 3\)"):
        ft.run_resire(sino, ang, n_iter=1, euler_zyz_deg=torch.zeros(1, 3))


@needs_cuda
@pytest.mark.parametrize("Nx,N_det", [(22, 26), (26, 22)])
def test_resire_autograd_nx_ne_ndet(Nx, N_det):
    """Same fix as the texture pair: forward_project_resire with N_det != Nx
    has a working backward (it raised 'invalid gradient' before), equal to
    back_project_resire(Nx=Nx); back_project_resire(Nx=...) of the wider
    volume centre-cropped equals the narrow one (gather, exact)."""
    g = torch.Generator().manual_seed(12)
    Ny, Nz, n = 4, 14, 7
    eul = torch.stack([torch.zeros(n), torch.linspace(-60, 60, n), torch.zeros(n)], 1).cuda()
    x = torch.rand(Ny, Nz, Nx, generator=g).cuda().requires_grad_(True)
    w = torch.rand(n, Ny, N_det, generator=g).cuda()
    (ft.forward_project_resire(x, eul, N_det=N_det) * w).sum().backward()
    assert x.grad.shape == x.shape
    assert torch.equal(x.grad, ft.back_project_resire(w, eul, Nz=Nz, Nx=Nx))
    s = w.clone().requires_grad_(True)
    (ft.back_project_resire(s, eul, Nz=Nz, Nx=Nx) * x.detach()).sum().backward()
    assert s.grad.shape == w.shape
    narrow, wide = min(Nx, N_det), max(Nx, N_det)
    off = (wide - narrow) // 2
    small = ft.back_project_resire(w, eul, Nz=Nz, Nx=narrow)
    big = ft.back_project_resire(w, eul, Nz=Nz, Nx=wide)[:, :, off:off + narrow]
    assert torch.equal(small, big)


# ── zero-size inputs: one ValueError on both backends ──────────────────────

def _zero_size_cases(dev):
    from fasttomo.autograd import _grad_angle_raw, _grad_rotmat_raw
    a = torch.linspace(-30, 30, 5, device=dev)
    e = torch.zeros(5, 3, device=dev)
    mats = torch.eye(3, device=dev).expand(5, 3, 3)
    v = torch.rand(2, 6, 6, device=dev)
    s = torch.rand(5, 2, 6, device=dev)
    return {
        "fp N_det=0": lambda: ft.forward_project(v, a, N_det=0),
        "fp no angles": lambda: ft.forward_project(v, a[:0]),
        "fp Ny=0": lambda: ft.forward_project(v[:0], a),
        "bp Nz=0": lambda: ft.back_project(s, a, Nz=0),
        "bp Nx=0": lambda: ft.back_project(s, a, Nx=0),
        "bp N_det=0": lambda: ft.back_project(s[:, :, :0], a),
        "fp3d N_det=0": lambda: ft.forward_project_3d(v, e, N_det=0),
        "fp3d no angles": lambda: ft.forward_project_3d(v, e[:0]),
        "bp3d Nz=0": lambda: ft.back_project_3d(s, e, Nz=0),
        "bp3d Ny=0": lambda: ft.back_project_3d(s[:, :0], e),
        "grad_rotmat": lambda: _grad_rotmat_raw(v, s[:, :, :0], mats),
        "grad_angle": lambda: _grad_angle_raw(v, s[:, :, :0], a),
    }


@pytest.mark.parametrize("case", list(_zero_size_cases("cpu")))
def test_zero_size_raises_value_error(backend, case):
    """Before: CPU returned empty tensors while CUDA raised a launch
    RuntimeError or a tc_create ValueError. Now both raise the same
    ValueError from the shared launch helper."""
    with pytest.raises(ValueError, match="zero-size dimension"):
        _zero_size_cases(backend)[case]()


@needs_cuda
@pytest.mark.parametrize("make", [
    lambda: ft.ParallelProjector((2, 0, 6), torch.zeros(5, device="cuda")),
    lambda: ft.ParallelProjector((2, 6, 6), torch.zeros(0, device="cuda")),
    lambda: ft.ParallelProjector((2, 6, 6), torch.zeros(5, device="cuda"), N_det=0),
    lambda: ft.RotmatProjector((0, 6, 6), n_angles=5, device="cuda"),
    lambda: ft.RotmatProjector((2, 6, 6), n_angles=5, device="cuda", N_det=0),
])
def test_projectors_reject_zero_size_geometry(make):
    with pytest.raises(ValueError, match="at least 1"):
        make()


# ── #78 review follow-ups ──────────────────────────────────────────────────

def test_ad_recon_shape_guard_runs_before_shift_correction(monkeypatch):
    """A projector built for the wrong number of views must give the
    projector-shape error even with refine_shifts=True; the shift correction
    used to run first and raise 'shifts has 5 entries but sinogram has 1
    angles', blaming the shifts."""

    class OneViewProjector:
        def forward(self, volume):
            return volume.mean(dim=1).unsqueeze(0)          # [1, Ny, Nx]

    proj = _cpu_ad_recon(monkeypatch, projector=OneViewProjector())
    cfg = ft.AdReconConfig(optim=ft.OptimConfig(n_iter=1),
                           pose=ft.PoseRefinementConfig(refine_shifts=True))
    with pytest.raises(ValueError, match="projector returned forward-projection"):
        ft.run_ad_recon(torch.zeros(5, 2, 6), torch.linspace(-20, 20, 5), Nz=6,
                        projector=proj, config=cfg)


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


# ── ARCH P12: one MBIR offset/variance implementation ──────────────────────

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

    monkeypatch.setattr(algo, "_mbir_update_offset_variance", counting)
    sino = torch.rand(6, 2, 8, generator=torch.Generator().manual_seed(14))
    ft.run_mbir(sino, torch.linspace(-40, 40, 6), n_iter=3, multires_levels=1,
                convergence_tol=0.0, estimate_offset=True, estimate_variance=True)
    assert calls == [(True, True)] * 3


@needs_cuda
def test_mbir_cpu_cuda_consistency_with_offset_and_variance():
    """test_mbir_cpu_cuda_consistency runs with estimation off; this one
    turns on the offset and variance updates, Huber weights and lambdas, so
    the shared update is exercised on both backends. The residual
    difference is float summation order in the sweeps (~3e-7 here)."""
    import fasttomo.algorithms as algo
    g = torch.Generator().manual_seed(21)
    Ny, N, n = 6, 16, 18
    ph = torch.zeros(Ny, N, N)
    ph[1:5, 4:12, 5:11] = 1.0
    ang = torch.linspace(-60, 60, n)
    a = ang * math.pi / 180
    sino = -algo._mbir_initial_residual(ph, torch.zeros(n, Ny, N), torch.cos(a), torch.sin(a))
    sino = sino + 0.3 * torch.linspace(-1, 1, n).view(n, 1, 1) + 0.05 * torch.randn(n, Ny, N, generator=g)
    sino[3, 2, 5] += 8.0
    lam = torch.rand(n, Ny, N, generator=g) + 0.5
    kw = dict(sigma_f=0.1, n_iter=8, multires_levels=1, seed=5, return_state=True,
              estimate_offset=True, estimate_variance=True, huber_T=3.0)
    vc, dc, sc = ft.run_mbir(sino, ang, lambdas=lam, **kw)
    vg, dg, sg = ft.run_mbir(sino.cuda(), ang.cuda(), lambdas=lam.cuda(), **kw)
    assert (dc.abs() > 0.1).any()                        # offsets were estimated
    torch.testing.assert_close(vg.cpu(), vc, rtol=0, atol=1e-4)
    torch.testing.assert_close(dg.cpu(), dc, rtol=0, atol=1e-4)
    assert sg == pytest.approx(sc, rel=1e-5)
