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
        a_p = ang.detach().clone(); a_p[i] += h
        a_m = ang.detach().clone(); a_m[i] -= h
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


# ── ConstraintConfig.sparsity honoured by run_ad_recon ─────────────────────

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
        constraints=ft.ConstraintConfig(sparsity=lam),
    )
    rec = ft.run_ad_recon(sino, torch.linspace(-30, 30, 4), Nz=6,
                          projector=proj, config=cfg)
    torch.testing.assert_close(rec, torch.full_like(rec, expected), rtol=0, atol=1e-6)


def test_ad_recon_constraint_sparsity_matches_config_apply(monkeypatch):
    """Several iterations: run_ad_recon's result equals a hand-written loop
    (Adam step, then ConstraintConfig.apply) with the same settings."""
    proj = _cpu_ad_recon(monkeypatch)
    g = torch.Generator().manual_seed(5)
    sino = torch.rand(4, 2, 6, generator=g) * 2 - 0.5
    cons = ft.ConstraintConfig(positivity=False, sparsity=0.02)
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
        constraints=ft.ConstraintConfig(sparsity=0.0)))
    assert torch.equal(rec0, rec0b)
    rec = ft.run_ad_recon(sino, ang, Nz=N, config=ft.AdReconConfig(
        optim=ft.OptimConfig(n_iter=40, lr_volume=0.05),
        constraints=ft.ConstraintConfig(sparsity=0.01)))
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
            p = sh.detach().clone(); p[i, j] += h
            m = sh.detach().clone(); m[i, j] -= h
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
        vp = v.copy(); vp[i] += h
        vm = v.copy(); vm[i] -= h
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
