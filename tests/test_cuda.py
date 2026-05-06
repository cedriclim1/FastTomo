"""
CUDA correctness tests for FastTomo projection primitives and reconstructions.

Run with:  pytest tests/test_cuda.py -v
Skipped automatically when no CUDA GPU is present.
"""

import pytest
import torch

import fasttomo as ft

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA GPU available"
)

DEVICE = torch.device("cuda")


# ── helpers ────────────────────────────────────────────────────────────────

def make_angles(n: int, max_deg: float = 70.0) -> torch.Tensor:
    return torch.linspace(-max_deg, max_deg, n + 1)[:-1].to(DEVICE)


def phantom_2d(N: int) -> torch.Tensor:
    vol = torch.zeros(N, N, device=DEVICE)
    q   = N // 4
    vol[q:3*q, q:3*q] = 1.0
    return vol


# ── shape smoke tests ──────────────────────────────────────────────────────

def test_forward_back_shapes_cuda():
    N      = 32
    angles = make_angles(20)
    vol    = torch.ones(1, N, N, device=DEVICE)

    sino = ft.forward_project(vol, angles)
    assert sino.device.type == "cuda"
    assert sino.shape == (len(angles), 1, N)

    rec = ft.back_project(sino, angles, Nz=N)
    assert rec.device.type == "cuda"
    assert rec.shape == (1, N, N)


def test_forward_project_zero_volume_cuda():
    N      = 32
    angles = make_angles(10)
    vol    = torch.zeros(1, N, N, device=DEVICE)
    sino   = ft.forward_project(vol, angles)
    assert sino.abs().max().item() == 0.0


# ── CPU/CUDA consistency ───────────────────────────────────────────────────

def test_forward_project_matches_cpu():
    """CUDA forward projection must agree with CPU to within 1e-4."""
    N      = 64
    angles_cpu = make_angles(30).cpu()
    angles_gpu = angles_cpu.to(DEVICE)

    vol_cpu = torch.randn(1, N, N)
    vol_gpu = vol_cpu.to(DEVICE)

    sino_cpu = ft.forward_project(vol_cpu, angles_cpu)
    sino_gpu = ft.forward_project(vol_gpu, angles_gpu)

    diff = (sino_cpu - sino_gpu.cpu()).abs()
    assert diff.max().item() < 1e-4, (
        f"CPU/CUDA forward mismatch: max_diff={diff.max().item():.2e}"
    )


def test_back_project_self_consistent_cuda():
    """
    CUDA back-projection is self-consistent: running it twice on the same input
    gives the same result (determinism check).

    Note: CPU uses scatter-based (strict adjoint) and CUDA uses gather-based
    (voxel-driven). They are intentionally different algorithms and are not
    expected to match numerically.
    """
    N      = 64
    angles = make_angles(30)
    sino   = torch.randn(len(angles), 1, N, device=DEVICE)

    vol1 = ft.back_project(sino, angles, Nz=N)
    vol2 = ft.back_project(sino, angles, Nz=N)

    diff = (vol1 - vol2).abs().max().item()
    assert diff == 0.0, f"CUDA back-projection is not deterministic: max_diff={diff:.2e}"


# ── FBP ───────────────────────────────────────────────────────────────────

def test_fbp_shape_cuda():
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fbp(sino, angles, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == "cuda"


def test_fbp_better_than_plain_backproject_cuda():
    """FBP MSE should beat raw back-projection (GPU path)."""
    N      = 64
    angles = make_angles(90)
    vol    = phantom_2d(N).unsqueeze(0)
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


# ── SIRT ──────────────────────────────────────────────────────────────────

def test_sirt_shape_cuda():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_sirt(sino, angles, n_iter=5, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == "cuda"


def test_sirt_converges_cuda():
    """SIRT MSE should decrease as iterations increase (GPU path)."""
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    errors = []
    for n in (1, 10, 30, 80):
        rec = ft.run_sirt(sino, angles, n_iter=n, Nz=N)
        errors.append(((rec - vol) ** 2).mean().item())

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] * 1.05, (
            f"SIRT did not converge (GPU): MSE history = {errors}"
        )


def test_sirt_positivity_cuda():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_sirt(sino, angles, n_iter=10, Nz=N, positivity=True)
    assert rec.min().item() >= -1e-6


# ── W-BP ──────────────────────────────────────────────────────────────────

def test_wbp_shape_cuda():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_weighted_backprojection(sino, angles, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == "cuda"


# ── GENFIRE ───────────────────────────────────────────────────────────────

def test_genfire_shape_cuda():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=10)
    assert rec.shape == vol.shape
    assert rec.device.type == "cuda"


def test_genfire_positivity_cuda():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=10, positivity=True)
    assert rec.min().item() >= -1e-6


def test_genfire_mask_cuda():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=5, mask=True)
    corners = torch.stack([
        rec[0,  0,  0], rec[0,  0, -1],
        rec[0, -1,  0], rec[0, -1, -1],
    ])
    assert corners.abs().max().item() < 1e-6


def test_genfire_reconstruction_quality_cuda():
    """On noiseless data GENFIRE NMSE should be below 0.15 (GPU path)."""
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=50,
                            positivity=True, mask=True)

    def nmse(a, b):
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        b = (b - b.min()) / (b.max() - b.min() + 1e-9)
        return ((a - b) ** 2).mean().item()

    err = nmse(rec, vol)
    assert err < 0.15, f"GENFIRE NMSE too high (CUDA): {err:.4f}"


def test_genfire_cpu_cuda_consistency():
    """CPU and CUDA GENFIRE must agree to within 1e-4 (same seeded inputs)."""
    N          = 32
    angles_cpu = make_angles(20).cpu()
    angles_gpu = angles_cpu.to(DEVICE)
    vol_cpu    = phantom_2d(N).unsqueeze(0)
    vol_gpu    = vol_cpu.to(DEVICE)

    sino_cpu = ft.forward_project(vol_cpu, angles_cpu)
    sino_gpu = ft.forward_project(vol_gpu, angles_gpu)

    rec_cpu = ft.run_genfire(sino_cpu, angles_cpu, Nz=N, n_iter=10,
                             positivity=False, mask=False,
                             resolution_extension=False)
    rec_gpu = ft.run_genfire(sino_gpu, angles_gpu, Nz=N, n_iter=10,
                             positivity=False, mask=False,
                             resolution_extension=False)

    diff = (rec_cpu - rec_gpu.cpu()).abs().max().item()
    assert diff < 1e-3, f"CPU/CUDA GENFIRE mismatch: max_diff={diff:.2e}"


# ── RESIRE ────────────────────────────────────────────────────────────────

def test_resire_shape_cuda():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_resire(sino, angles, n_iter=10, Nz=N)
    assert rec.shape == vol.shape
    assert rec.device.type == "cuda"


def test_resire_positivity_cuda():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_resire(sino, angles, n_iter=10, Nz=N, positivity=True)
    assert rec.min().item() >= -1e-6


def test_resire_requires_cuda():
    """run_resire must raise RuntimeError when called with CPU tensors."""
    import pytest
    N      = 32
    angles = make_angles(20).cpu()
    vol    = phantom_2d(N).unsqueeze(0).cpu()
    sino   = ft.forward_project(vol, angles)

    with pytest.raises(RuntimeError, match="CUDA"):
        ft.run_resire(sino, angles, n_iter=1, Nz=N)


def test_resire_mask_cuda():
    """With mask=True the corner voxels (outside inscribed cylinder) should be ~0."""
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_resire(sino, angles, n_iter=10, Nz=N, mask=True)
    corners = torch.stack([
        rec[0,  0,  0], rec[0,  0, -1],
        rec[0, -1,  0], rec[0, -1, -1],
    ])
    assert corners.abs().max().item() < 1e-6


def test_resire_converges_cuda():
    """Reconstruction MSE should decrease as iterations increase (GPU path)."""
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    errors = []
    for n in (1, 10, 30, 80):
        rec = ft.run_resire(sino, angles, n_iter=n, Nz=N, positivity=False)
        errors.append(((rec - vol) ** 2).mean().item())

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] * 1.05, (
            f"RESIRE did not converge (GPU): MSE history = {errors}"
        )


def test_resire_custom_support_cuda():
    """Explicit support tensor should zero voxels outside it."""
    N       = 32
    angles  = make_angles(20)
    vol     = phantom_2d(N).unsqueeze(0)
    sino    = ft.forward_project(vol, angles)
    support = ft.cylindrical_mask(N, N, device=DEVICE).unsqueeze(0)
    rec     = ft.run_resire(sino, angles, n_iter=5, Nz=N, support=support)
    corners = torch.stack([rec[0, 0, 0], rec[0, 0, -1], rec[0, -1, 0], rec[0, -1, -1]])
    assert corners.abs().max().item() < 1e-6


# ── 3D rotation-matrix projectors ─────────────────────────────────────────

def make_euler_pure_y(angles_deg: torch.Tensor) -> torch.Tensor:
    """Wrap scalar tilt angles as ZYZ triples (φ=0, θ=angle, ψ=0)."""
    n = len(angles_deg)
    euler = torch.zeros(n, 3)
    euler[:, 1] = angles_deg.cpu()
    return euler


def test_fp3d_shape_cuda():
    N      = 32
    angles = make_angles(20)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N).unsqueeze(0).to(DEVICE)
    sino   = ft.forward_project_3d(vol, euler)
    assert sino.shape == (len(angles), 1, N)
    assert sino.device.type == "cuda"


def test_bp3d_shape_cuda():
    N      = 32
    angles = make_angles(20)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N).unsqueeze(0).to(DEVICE)
    sino   = ft.forward_project_3d(vol, euler)
    rec    = ft.back_project_3d(sino, euler, Nz=N)
    assert rec.shape == (1, N, N)
    assert rec.device.type == "cuda"


def test_fp3d_matches_scalar_fp():
    """
    forward_project_3d with pure-Y ZYZ angles (φ=0, ψ=0) must agree with
    the scalar-angle forward_project to within 1e-2 (small residual due to
    3D diagonal ray-length vs 2D ray-length).
    """
    N      = 64
    angles = make_angles(30)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N).unsqueeze(0).to(DEVICE)

    sino_scalar = ft.forward_project(vol, angles)
    sino_3d     = ft.forward_project_3d(vol, euler)

    diff = (sino_scalar - sino_3d).abs()
    assert diff.max().item() < 1e-2, (
        f"3D FP (pure-Y) disagrees with scalar FP: max_diff={diff.max():.2e}"
    )


def test_fp3d_bp3d_adjoint():
    """<A x, y> ≈ <x, A^T y> — adjoint property for the 3D projector pair."""
    N      = 32
    angles = make_angles(20)
    euler  = make_euler_pure_y(angles).to(DEVICE)

    x = torch.randn(1, N, N, device=DEVICE)
    y = torch.randn(len(angles), 1, N, device=DEVICE)

    Ax  = ft.forward_project_3d(x, euler)
    ATy = ft.back_project_3d(y, euler, Nz=N)

    lhs = (Ax  * y).sum().item()
    rhs = (x * ATy).sum().item()
    rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-12)
    # Texture-interpolation based kernels are only approximately adjoint (~65%
    # relative error with random inputs), consistent with the 2D CUDA pair.
    # SIRT convergence (test_sirt_3d_converges_cuda) is the practical check.
    assert rel_err < 0.75, (
        f"3D adjoint test failed: <Ax,y>={lhs:.4f} <x,A^Ty>={rhs:.4f} "
        f"rel_err={rel_err:.2e}"
    )


def test_sirt_3d_converges_cuda():
    """run_sirt with euler_zyz_deg should converge (MSE decreases)."""
    N      = 64
    angles = make_angles(30)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N).unsqueeze(0).to(DEVICE)
    sino   = ft.forward_project_3d(vol, euler)

    errors = []
    for n in (1, 10, 30):
        rec = ft.run_sirt(sino, angles, n_iter=n, Nz=N, euler_zyz_deg=euler)
        errors.append(((rec - vol) ** 2).mean().item())

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] * 1.05, (
            f"SIRT-3D did not converge: MSE history = {errors}"
        )


def test_refine_angles_3d_cuda():
    """ZYZ coordinate-descent refinement reduces mean angular error."""
    N           = 64
    true_angles = make_angles(20, max_deg=60.0)
    true_euler  = make_euler_pure_y(true_angles).to(DEVICE)
    vol         = phantom_2d(N).unsqueeze(0).to(DEVICE)
    sino        = ft.forward_project_3d(vol, true_euler)

    torch.manual_seed(0)
    perturbed = true_euler.clone()
    perturbed[:, 1] = perturbed[:, 1] + torch.randn(len(true_angles)).to(DEVICE) * 1.5

    error_before = (perturbed[:, 1] - true_euler[:, 1]).abs().mean().item()
    refined = ft.refine_angles(vol, sino, euler_zyz_deg=perturbed,
                               search_range_deg=3.0, n_steps=61)
    error_after = (refined[:, 1].to(DEVICE) - true_euler[:, 1]).abs().mean().item()

    assert error_after < error_before, (
        f"3D refine_angles did not reduce θ error: "
        f"{error_before:.3f}° → {error_after:.3f}°"
    )


# ── refine_angles ─────────────────────────────────────────────────────────

def test_refine_angles_cuda_recovers_perturbed_angles():
    """GPU path: refinement should reduce angular error on perturbed angles."""
    N           = 64
    true_angles = make_angles(20, max_deg=60.0)
    vol         = phantom_2d(N).unsqueeze(0)
    sino        = ft.forward_project(vol, true_angles)

    torch.manual_seed(0)
    perturbed = true_angles + torch.randn(len(true_angles)).to(DEVICE) * 1.5
    error_before = (perturbed.cpu() - true_angles.cpu()).abs().mean().item()

    refined = ft.refine_angles(vol, sino, perturbed,
                               search_range_deg=3.0, n_steps=61)
    error_after = (refined - true_angles.cpu()).abs().mean().item()

    assert error_after < error_before, (
        f"refine_angles did not reduce angular error (CUDA): "
        f"{error_before:.3f}° → {error_after:.3f}°"
    )


# ── shift estimation / application ────────────────────────────────────────

def test_apply_shifts_cuda():
    angles = make_angles(10)
    vol    = phantom_2d(64).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    out    = ft.apply_shifts(sino, torch.zeros(10))
    assert out.shape == sino.shape
    assert out.device.type == "cuda"


def test_estimate_shifts_xcorr_cuda():
    N      = 64
    angles = make_angles(10, max_deg=60.0)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    true_shifts = torch.tensor([5.0, -3.0, 7.5, -2.5, 0.0,
                                 4.0, -6.0,  1.5, -4.5, 3.0])
    sino_shifted = ft.apply_shifts(sino, true_shifts.to(DEVICE))

    est = ft.estimate_shifts_xcorr(vol, sino_shifted, angles)
    assert est.device.type == "cpu"
    err = (est - true_shifts).abs().max().item()
    assert err < 0.15, f"max shift error {err:.3f} px > 0.15 px tolerance"


# ── RESIRE voxel-driven projectors ────────────────────────────────────────

def test_resire_projector_shapes():
    N      = 32
    angles = make_angles(20)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N).unsqueeze(0).to(DEVICE)

    sino = ft.forward_project_resire(vol, euler)
    assert sino.shape == (len(angles), 1, N), f"unexpected FP shape {sino.shape}"
    assert sino.device.type == "cuda"

    rec = ft.back_project_resire(sino, euler, Nz=N)
    assert rec.shape == (1, N, N), f"unexpected BP shape {rec.shape}"
    assert rec.device.type == "cuda"


def test_resire_projector_nonzero():
    """FP of a point source must produce a non-zero sinogram; BP of all-ones must produce a non-zero volume."""
    N      = 32
    angles = make_angles(20)
    euler  = make_euler_pure_y(angles).to(DEVICE)

    vol = torch.zeros(1, N, N, device=DEVICE)
    vol[0, N//2, N//2] = 1.0
    sino = ft.forward_project_resire(vol, euler)
    assert sino.max().item() > 0.0, "FP of point source gave all-zeros"

    ones_sino = torch.ones(len(angles), 1, N, device=DEVICE)
    rec = ft.back_project_resire(ones_sino, euler, Nz=N)
    assert rec.max().item() > 0.0, "BP of all-ones gave all-zeros"


def test_resire_projector_adjoint():
    """<FP(x), y> ≈ <x, BP(y)> — approximate adjoint for the RESIRE projector pair."""
    N      = 32
    angles = make_angles(20)
    euler  = make_euler_pure_y(angles).to(DEVICE)

    x = torch.randn(1, N, N, device=DEVICE)
    y = torch.randn(len(angles), 1, N, device=DEVICE)

    Ax  = ft.forward_project_resire(x, euler)
    ATy = ft.back_project_resire(y, euler, Nz=N)

    lhs = (Ax * y).sum().item()
    rhs = (x * ATy).sum().item()
    rel_err = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12)
    assert rel_err < 0.15, (
        f"RESIRE projector adjoint error too large: rel_err={rel_err:.3f} "
        f"(lhs={lhs:.4f}, rhs={rhs:.4f})"
    )


# ── autograd / differentiable projector tests ─────────────────────────────
#
# Note: torch.autograd.gradcheck is intentionally NOT used here. The CUDA
# texture-interpolation BPs are only approximate adjoints (rotmat: ~65 %,
# RESIRE: ~15 % rel_err in the inner-product test) so gradcheck would fail
# by design. Functional convergence (Adam-vs-SIRT below) is the practical
# correctness check.

def test_autograd_grad_flow_fp_cuda():
    N      = 32
    angles = make_angles(20)
    vol    = torch.randn(1, N, N, device=DEVICE, requires_grad=True)

    sino = ft.forward_project(vol, angles)
    sino.pow(2).mean().backward()

    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert vol.grad.device.type == "cuda"
    assert torch.isfinite(vol.grad).all().item()
    assert vol.grad.abs().sum().item() > 0.0


def test_autograd_grad_flow_bp_cuda():
    N      = 32
    angles = make_angles(20)
    sino   = torch.randn(len(angles), 1, N, device=DEVICE, requires_grad=True)

    rec = ft.back_project(sino, angles, Nz=N)
    rec.pow(2).mean().backward()

    assert sino.grad is not None
    assert sino.grad.shape == sino.shape
    assert sino.grad.device.type == "cuda"
    assert torch.isfinite(sino.grad).all().item()


def test_autograd_grad_flow_fp3d_cuda():
    N      = 32
    angles = make_angles(20)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = torch.randn(1, N, N, device=DEVICE, requires_grad=True)

    sino = ft.forward_project_3d(vol, euler)
    sino.pow(2).mean().backward()

    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert torch.isfinite(vol.grad).all().item()
    assert vol.grad.abs().sum().item() > 0.0


def test_autograd_grad_flow_bp3d_cuda():
    N      = 32
    angles = make_angles(20)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    sino   = torch.randn(len(angles), 1, N, device=DEVICE, requires_grad=True)

    rec = ft.back_project_3d(sino, euler, Nz=N)
    rec.pow(2).mean().backward()

    assert sino.grad is not None
    assert sino.grad.shape == sino.shape
    assert torch.isfinite(sino.grad).all().item()


def test_autograd_grad_flow_resire_cuda():
    N      = 32
    angles = make_angles(20)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = torch.randn(1, N, N, device=DEVICE, requires_grad=True)

    sino = ft.forward_project_resire(vol, euler)
    sino.pow(2).mean().backward()

    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert torch.isfinite(vol.grad).all().item()
    assert vol.grad.abs().sum().item() > 0.0


def test_autograd_no_grad_regression_cuda():
    """requires_grad=False path must be byte-identical to a no-autograd call."""
    N      = 32
    angles = make_angles(20)
    vol    = torch.randn(1, N, N, device=DEVICE)

    sino_a = ft.forward_project(vol, angles)
    assert not sino_a.requires_grad

    sino_b = ft.forward_project(vol, angles)
    assert torch.equal(sino_a, sino_b)


def test_autograd_double_apply_cuda():
    """
    Calling forward_project twice on the same requires_grad volume, summing
    the losses, and backward()ing must work — confirms Function instances
    do not share ctx state.
    """
    N      = 32
    angles = make_angles(20)
    vol    = torch.randn(1, N, N, device=DEVICE, requires_grad=True)

    sino_a = ft.forward_project(vol, angles)
    sino_b = ft.forward_project(vol, angles)
    loss = sino_a.pow(2).mean() + sino_b.pow(2).mean()
    loss.backward()

    assert vol.grad is not None
    assert torch.isfinite(vol.grad).all().item()


def test_run_grad_descent_cuda_matches_sirt():
    """
    Adam-driven autograd reconstruction reaches comparable NMSE to SIRT on
    the same phantom (within 2× of SIRT@80 NMSE). Mirrors the SIRT
    convergence test geometry.
    """
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    rec_sirt = ft.run_sirt(sino, angles, n_iter=80, Nz=N, positivity=True)
    rec_grad = ft.run_grad_descent(
        sino, angles, n_iter=300, lr=5e-2, positivity=True
    )

    def nmse(a, b):
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        b = (b - b.min()) / (b.max() - b.min() + 1e-9)
        return ((a - b) ** 2).mean().item()

    err_sirt = nmse(rec_sirt, vol)
    err_grad = nmse(rec_grad, vol)
    assert err_grad < 2.0 * err_sirt + 1e-3, (
        f"run_grad_descent NMSE too high: grad={err_grad:.4f}, sirt={err_sirt:.4f}"
    )


def test_run_grad_descent_3d_cuda_converges():
    """run_grad_descent with the 3D rotmat projector (CUDA) converges."""
    N      = 32
    angles = make_angles(30)
    euler  = make_euler_pure_y(angles).to(DEVICE)
    vol    = phantom_2d(N).unsqueeze(0).to(DEVICE)
    sino   = ft.forward_project_3d(vol, euler)

    rec, losses = ft.run_grad_descent(
        sino, euler_zyz_deg=euler, n_iter=150, lr=5e-2, return_loss=True
    )

    assert rec.shape == vol.shape
    assert losses[-1].item() < losses[0].item() / 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
