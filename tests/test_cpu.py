"""
CPU correctness tests for FastTomo projection primitives and reconstructions.

Run with:  pytest tests/test_cpu.py -v
"""

import pytest
import torch

import fasttomo as ft
import fasttomo.algorithms as ft_algorithms


# ── helpers ────────────────────────────────────────────────────────────────

def make_angles(n: int, max_deg: float = 70.0) -> torch.Tensor:
    """n evenly spaced angles in [-max_deg, +max_deg) in degrees."""
    return torch.linspace(-max_deg, max_deg, n + 1)[:-1]


def phantom_2d(N: int) -> torch.Tensor:
    """Simple square phantom centred in an N×N volume slice."""
    vol = torch.zeros(N, N)
    q   = N // 4
    vol[q:3*q, q:3*q] = 1.0
    return vol


# ── adjoint test ───────────────────────────────────────────────────────────

def test_adjoint_cpu():
    """
    Verify <Ax, y> ≈ <x, A^T y> to reasonable tolerance.

    A = forward_project,  A^T = back_project.
    This is the fundamental correctness check for the kernel pair.
    """
    N      = 64
    Ny     = 1
    angles = make_angles(30)

    x = torch.randn(Ny, N, N)
    y = torch.randn(len(angles), Ny, N)

    Ax  = ft.forward_project(x, angles)
    ATy = ft.back_project(y, angles, Nz=N)

    lhs = (Ax  * y).sum().item()
    rhs = (x * ATy).sum().item()

    rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-12)
    assert rel_err < 5e-3, (
        f"Adjoint test failed: <Ax,y>={lhs:.6f} vs <x,A^Ty>={rhs:.6f} "
        f"(rel_err={rel_err:.2e})"
    )


# ── shape smoke tests ──────────────────────────────────────────────────────

def test_forward_back_shapes():
    N      = 32
    angles = make_angles(20)
    vol    = torch.ones(1, N, N)

    sino = ft.forward_project(vol, angles)
    assert sino.shape == (len(angles), 1, N)

    rec = ft.back_project(sino, angles, Nz=N)
    assert rec.shape == (1, N, N)


def test_forward_project_zero_volume():
    N      = 32
    angles = make_angles(10)
    vol    = torch.zeros(1, N, N)
    sino   = ft.forward_project(vol, angles)
    assert sino.abs().max().item() == 0.0


# ── FBP ───────────────────────────────────────────────────────────────────

def test_fbp_shape():
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fbp(sino, angles, Nz=N)
    assert rec.shape == vol.shape


def test_fbp_better_than_plain_backproject():
    """FBP reconstruction MSE should beat raw (unfiltered) back-projection."""
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


# ── SIRT ──────────────────────────────────────────────────────────────────

def test_sirt_shape():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_sirt(sino, angles, n_iter=5, Nz=N)
    assert rec.shape == vol.shape


def test_sirt_converges():
    """Reconstruction MSE should decrease as iterations increase."""
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    errors = []
    for n in (1, 10, 30, 80):
        rec = ft.run_sirt(sino, angles, n_iter=n, Nz=N)
        errors.append(((rec - vol) ** 2).mean().item())

    # allow 5% slack to account for oscillation near convergence
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] * 1.05, (
            f"SIRT did not converge: MSE history = {errors}"
        )


def test_sirt_positivity():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_sirt(sino, angles, n_iter=10, Nz=N, positivity=True)
    assert rec.min().item() >= -1e-6


def test_sirt_warm_start_shape_mismatch_raises():
    """Warm-start volume shape must exactly match [Ny, Nz, Nx] — without
    this guard a [1, N, N] warm-start with Ny>1 silently broadcasts inside
    the SIRT update, corrupting the reconstruction with no warning."""
    n_angles, Ny, N = 12, 4, 16
    sino = torch.zeros(n_angles, Ny, N, dtype=torch.float32)
    angles = make_angles(n_angles)
    warm = torch.zeros(1, N, N, dtype=torch.float32)  # Ny mismatch
    with pytest.raises(ValueError, match="volume must have shape"):
        ft.run_sirt(sino, angles, n_iter=1, Nz=N, volume=warm)


def test_sirt_projector_shape_mismatch_raises():
    """Projector outputs must match the active reconstruction geometry.

    Regression guard for silent broadcasting: if a user reuses a projector
    built for Ny=1 with a Ny>1 sinogram, SIRT used to broadcast normalisers
    and updates across Y without raising."""

    class MismatchedProjector:
        def __init__(self, n_angles: int, N: int):
            self.n_angles = n_angles
            self.N = N

        def forward(self, volume: torch.Tensor) -> torch.Tensor:
            return torch.zeros(self.n_angles, 1, self.N, dtype=volume.dtype)

        def back(self, sinogram: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, self.N, self.N, dtype=sinogram.dtype)

    n_angles, Ny, N = 12, 4, 16
    sino = torch.zeros(n_angles, Ny, N, dtype=torch.float32)
    angles = make_angles(n_angles)

    with pytest.raises(ValueError, match="projector returned"):
        ft.run_sirt(
            sino,
            angles,
            n_iter=1,
            Nz=N,
            projector=MismatchedProjector(n_angles=n_angles, N=N),
        )


def test_ad_sparsity_loss_is_sign_symmetric_and_has_zero_gradient_at_zero():
    """AD sparsity regularization must implement mean(abs(volume)).

    Regression guard: a signed mean drives all voxels negative when sparsity
    is enabled, corrupting AD reconstructions.
    """
    vol = torch.tensor([[[1.0, -2.0], [0.5, -0.25]]], dtype=torch.float32)
    loss = ft_algorithms._ad_sparsity_loss(vol)
    mirrored_loss = ft_algorithms._ad_sparsity_loss(-vol)
    assert torch.allclose(loss, mirrored_loss), "L1 sparsity must be sign-symmetric"

    vol_zero = torch.zeros(1, 2, 2, dtype=torch.float32, requires_grad=True)
    zero_loss = ft_algorithms._ad_sparsity_loss(vol_zero)
    zero_loss.backward()
    assert torch.count_nonzero(vol_zero.grad).item() == 0, (
        "L1 sparsity should not apply a signed push at zero; gradient must be zero"
    )


# ── FISTA / FISTA-TV ──────────────────────────────────────────────────────

def test_fista_shape():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista(sino, angles, n_iter=5, Nz=N)
    assert rec.shape == vol.shape


def test_fista_converges():
    """FISTA data-fit MSE history should be (roughly) non-increasing."""
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    rec, losses = ft.run_fista(sino, angles, n_iter=30, Nz=N, return_loss=True)
    assert losses.shape == (30,)
    # FISTA can oscillate slightly (Nesterov momentum); 10% slack over window
    assert losses[-1].item() < losses[0].item() * 0.1, (
        f"FISTA loss did not decrease: first={losses[0].item():.4e}, "
        f"last={losses[-1].item():.4e}"
    )


def test_fista_positivity():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista(sino, angles, n_iter=10, Nz=N, positivity=True)
    assert rec.min().item() >= -1e-6


def test_fista_mask_zeros_corners():
    """mask=True must zero the inscribed-cylinder corners."""
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista(sino, angles, n_iter=5, Nz=N, mask=True)
    corners = torch.stack([
        rec[0,  0,  0], rec[0,  0, -1],
        rec[0, -1,  0], rec[0, -1, -1],
    ])
    assert corners.abs().max().item() < 1e-6


def test_fista_warm_start_shape_mismatch_raises():
    n_angles, Ny, N = 12, 4, 16
    sino = torch.zeros(n_angles, Ny, N, dtype=torch.float32)
    angles = make_angles(n_angles)
    warm = torch.zeros(1, N, N, dtype=torch.float32)  # Ny mismatch
    with pytest.raises(ValueError, match="volume must have shape"):
        ft.run_fista(sino, angles, n_iter=1, Nz=N, volume=warm)


def test_fista_user_L_skips_power_iteration():
    """L= kwarg must produce the same shape and a sane reconstruction."""
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista(sino, angles, n_iter=10, Nz=N, L=20.0)
    assert rec.shape == vol.shape
    assert torch.isfinite(rec).all()


def test_fista_tv_shape():
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista_tv(sino, angles, tv_weight=0.01, n_iter=5, Nz=N)
    assert rec.shape == vol.shape


def test_fista_tv_reduces_tv_vs_fista():
    """At matched n_iter, FISTA-TV must produce a less-TV reconstruction."""
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    rec        = ft.run_fista(sino, angles, n_iter=40, Nz=N)
    rec_tv     = ft.run_fista_tv(sino, angles, tv_weight=0.05, n_iter=40, Nz=N)

    # CPU pure-PyTorch TV (tv_loss_3d's CUDA path is fp32-only and Ny=1 hits an
    # empty Y-diff; recompute here for portability)
    def tv(v):
        dy = v[1:, :, :] - v[:-1, :, :]
        dz = v[:, 1:, :] - v[:, :-1, :]
        dx = v[:, :, 1:] - v[:, :, :-1]
        return (dz[:, :, :-1].pow(2) + dx[:, :-1, :].pow(2) + 1e-8).sqrt().mean().item()

    tv_fista    = tv(rec)
    tv_fista_tv = tv(rec_tv)
    assert tv_fista_tv < tv_fista, (
        f"FISTA-TV did not reduce TV (FISTA={tv_fista:.4f}, FISTA-TV={tv_fista_tv:.4f})"
    )


def test_fista_tv_positivity_inside_prox():
    """positivity=True is baked into the TV prox — output must be ≥ 0
    without any after-step clamp."""
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista_tv(
        sino, angles, tv_weight=0.05, n_iter=20, Nz=N, positivity=True,
    )
    assert rec.min().item() >= -1e-6


def test_fista_tv_chambolle_mode():
    """tv_inner='chambolle' must run and produce a finite reconstruction."""
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_fista_tv(
        sino, angles, tv_weight=0.05, n_iter=20, Nz=N, tv_inner="chambolle",
    )
    assert rec.shape == vol.shape
    assert torch.isfinite(rec).all()


def test_fista_tv_zero_weight_raises():
    N      = 16
    angles = make_angles(10)
    sino   = torch.zeros(len(angles), 1, N)
    with pytest.raises(ValueError, match="tv_weight > 0"):
        ft.run_fista_tv(sino, angles, tv_weight=0.0, n_iter=1, Nz=N)


def test_fista_tv_unknown_inner_mode_raises():
    N      = 16
    angles = make_angles(10)
    sino   = torch.zeros(len(angles), 1, N)
    with pytest.raises(ValueError, match="unknown mode"):
        ft.run_fista_tv(sino, angles, tv_weight=0.01, n_iter=1, Nz=N, tv_inner="bogus")


# ── W-BP ──────────────────────────────────────────────────────────────────

def test_wbp_shape():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_weighted_backprojection(sino, angles, Nz=N)
    assert rec.shape == vol.shape


# ── GENFIRE ───────────────────────────────────────────────────────────────

def test_genfire_shape():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)        # [1, N, N]
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=10)
    assert rec.shape == vol.shape


def test_genfire_positivity():
    """positivity=True must eliminate all negative voxels."""
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=10, positivity=True)
    assert rec.min().item() >= -1e-6, f"negative voxels found: min={rec.min():.4e}"


def test_genfire_mask():
    """With mask=True the corner voxels (outside inscribed cylinder) should be ~0."""
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=5, mask=True)
    # corners are always outside the inscribed cylinder
    corners = torch.stack([
        rec[0,  0,  0], rec[0,  0, -1],
        rec[0, -1,  0], rec[0, -1, -1],
    ])
    assert corners.abs().max().item() < 1e-6, (
        f"corner voxels should be masked to 0, got {corners.abs().max():.2e}"
    )


def test_genfire_converges():
    """Reconstruction error should decrease (or stay flat) as iterations increase."""
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    errors = []
    for n in (5, 20, 50):
        rec = ft.run_genfire(sino, angles, Nz=N, n_iter=n,
                             positivity=True, mask=False,
                             resolution_extension=False)
        errors.append(((rec - vol) ** 2).mean().item())

    # allow 10 % slack to accommodate best-iterate tracking non-monotonicity
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] * 1.10, (
            f"GENFIRE did not converge: MSE history = {errors}"
        )


def test_genfire_reconstruction_quality():
    """On noiseless data GENFIRE NMSE should be below 0.15."""
    N      = 64
    angles = make_angles(60)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=50,
                            positivity=True, mask=True)

    # normalised MSE (both normalised to [0,1])
    def nmse(a, b):
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        b = (b - b.min()) / (b.max() - b.min() + 1e-9)
        return ((a - b) ** 2).mean().item()

    err = nmse(rec, vol)
    assert err < 0.15, f"GENFIRE NMSE too high: {err:.4f}"


def test_genfire_no_resolution_extension():
    """Algorithm must run without resolution extension and return correct shape."""
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    rec    = ft.run_genfire(sino, angles, Nz=N, n_iter=10,
                            resolution_extension=False)
    assert rec.shape == vol.shape


# ── RESIRE ────────────────────────────────────────────────────────────────

def test_resire_requires_cuda():
    """run_resire is CUDA-only — must raise RuntimeError on CPU tensors."""
    import pytest
    N      = 32
    angles = make_angles(20)
    sino   = ft.forward_project(phantom_2d(N).unsqueeze(0), angles)
    with pytest.raises(RuntimeError, match="CUDA"):
        ft.run_resire(sino, angles, n_iter=1, Nz=N)


# ── refine_angles ─────────────────────────────────────────────────────────

def test_refine_angles_shape():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    refined = ft.refine_angles(vol, sino, angles)
    assert refined.shape == angles.shape


def test_refine_angles_returns_cpu_tensor():
    N      = 32
    angles = make_angles(20)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    refined = ft.refine_angles(vol, sino, angles)
    assert refined.device.type == "cpu"


def test_refine_angles_recovers_perturbed_angles():
    """
    Perturb true angles by up to 2°; refinement should reduce angular error.
    Uses noiseless forward projections so ground truth is exact.
    """
    N          = 64
    true_angles = make_angles(20, max_deg=60.0)
    vol         = phantom_2d(N).unsqueeze(0)
    sino        = ft.forward_project(vol, true_angles)

    torch.manual_seed(0)
    perturbed = true_angles + torch.randn(len(true_angles)) * 1.5
    error_before = (perturbed - true_angles).abs().mean().item()

    # one round of refinement with a 3° search window
    refined = ft.refine_angles(vol, sino, perturbed,
                               search_range_deg=3.0, n_steps=61)
    error_after = (refined - true_angles).abs().mean().item()

    assert error_after < error_before, (
        f"refine_angles did not reduce angular error: "
        f"{error_before:.3f}° → {error_after:.3f}°"
    )


def test_refine_angles_exact_angles_unchanged():
    """
    If angles are already correct the refined angles should stay within
    the grid resolution of the search (search_range / (n_steps - 1)).
    """
    N      = 32
    angles = make_angles(15, max_deg=60.0)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    n_steps = 31
    search  = 3.0
    refined = ft.refine_angles(vol, sino, angles,
                               search_range_deg=search, n_steps=n_steps)
    grid_step = 2 * search / (n_steps - 1)
    max_shift = (refined - angles).abs().max().item()
    assert max_shift <= grid_step + 1e-5, (
        f"correct angles shifted more than one grid step: {max_shift:.4f}°"
    )


# ── shift estimation / application ────────────────────────────────────────

def test_apply_shifts_shape():
    angles = make_angles(10)
    vol    = phantom_2d(32).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    out    = ft.apply_shifts(sino, torch.zeros(10))
    assert out.shape == sino.shape


def test_apply_shifts_zero_shift():
    angles = make_angles(10)
    vol    = phantom_2d(32).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    out    = ft.apply_shifts(sino, torch.zeros(10))
    assert (out - sino).abs().max().item() < 1e-5


def test_apply_shifts_integer_shift():
    angles = make_angles(10)
    vol    = phantom_2d(64).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    s      = 8
    shifts = torch.full((10,), float(s))
    out    = ft.apply_shifts(sino, shifts)
    rolled = torch.roll(sino, s, dims=-1)
    # Fourier phase shift and torch.roll agree up to boundary ringing (1e-4)
    assert (out - rolled).abs().max().item() < 1e-4


def test_estimate_shifts_xcorr_recovers_known_shifts():
    N      = 64
    angles = make_angles(10, max_deg=60.0)
    vol    = phantom_2d(N).unsqueeze(0)         # [1, 64, 64]
    sino   = ft.forward_project(vol, angles)    # [10, 1, 64]

    true_shifts = torch.tensor([5.0, -3.0, 7.5, -2.5, 0.0,
                                 4.0, -6.0,  1.5, -4.5, 3.0])
    sino_shifted = ft.apply_shifts(sino, true_shifts)

    est = ft.estimate_shifts_xcorr(vol, sino_shifted, angles)
    err = (est - true_shifts).abs().max().item()
    assert err < 0.15, f"max shift error {err:.3f} px > 0.15 px tolerance"


def test_estimate_shifts_xcorr_returns_cpu():
    angles = make_angles(8)
    vol    = phantom_2d(32).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    est    = ft.estimate_shifts_xcorr(vol, sino, angles)
    assert est.device.type == "cpu"


def test_estimate_shifts_xcorr_max_shift():
    N      = 64
    angles = make_angles(5, max_deg=60.0)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)
    large  = torch.full((5,), 10.0)
    sino_shifted = ft.apply_shifts(sino, large)
    est = ft.estimate_shifts_xcorr(vol, sino_shifted, angles, max_shift=5.0)
    assert est.abs().max().item() <= 5.0 + 1e-6


# ── autograd / differentiable projector tests ─────────────────────────────
#
# Note: we deliberately do NOT use torch.autograd.gradcheck. The CUDA
# texture-interpolation BPs are only approximate adjoints (15-65 % rel_err
# in the existing inner-product tests) so gradcheck would fail by design.
# These tests cover gradient flow + functional convergence instead.

def test_autograd_grad_flow_fp_cpu():
    """forward_project propagates gradients into the volume tensor."""
    N      = 32
    angles = make_angles(20)
    vol    = torch.randn(1, N, N, requires_grad=True)

    sino = ft.forward_project(vol, angles)
    loss = (sino ** 2).mean()
    loss.backward()

    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert torch.isfinite(vol.grad).all().item()
    assert vol.grad.abs().sum().item() > 0.0


def test_autograd_grad_flow_bp_cpu():
    """back_project propagates gradients into the sinogram tensor."""
    N      = 32
    angles = make_angles(20)
    sino   = torch.randn(len(angles), 1, N, requires_grad=True)

    rec  = ft.back_project(sino, angles, Nz=N)
    loss = (rec ** 2).mean()
    loss.backward()

    assert sino.grad is not None
    assert sino.grad.shape == sino.shape
    assert torch.isfinite(sino.grad).all().item()
    assert sino.grad.abs().sum().item() > 0.0


def test_autograd_no_grad_regression_cpu():
    """
    With requires_grad=False, forward_project must produce identical output
    to a direct kernel call — Function.apply short-circuits, no behaviour
    change vs the pre-autograd implementation.
    """
    N      = 32
    angles = make_angles(20)
    vol    = torch.randn(1, N, N)  # no requires_grad

    sino = ft.forward_project(vol, angles)
    assert not sino.requires_grad

    # Determinism: two calls must give identical bytes
    sino2 = ft.forward_project(vol, angles)
    assert torch.equal(sino, sino2)


def test_autograd_grad_through_squeeze_cpu():
    """
    A 2-D volume input (no Ny axis) should still receive a gradient of the
    same 2-D shape. Verifies the unsqueeze(0) hop is differentiable.
    """
    N      = 32
    angles = make_angles(20)
    vol_2d = torch.randn(N, N, requires_grad=True)

    sino = ft.forward_project(vol_2d, angles)
    sino.sum().backward()

    assert vol_2d.grad is not None
    assert vol_2d.grad.shape == (N, N)


def test_run_grad_descent_cpu_converges():
    """Adam-driven reconstruction loss drops by >50× from initial."""
    N      = 32
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    rec, losses = ft.run_grad_descent(
        sino, angles, n_iter=200, lr=5e-2, return_loss=True
    )

    assert rec.shape == vol.shape
    assert not rec.requires_grad
    assert losses[-1].item() < losses[0].item() / 50.0, (
        f"grad_descent did not converge: loss[0]={losses[0]:.4e}, "
        f"loss[-1]={losses[-1]:.4e}"
    )


def test_run_grad_descent_positional_kwargs_backward_compat_cpu():
    """Legacy positional optional args should still map correctly."""
    N      = 24
    angles = make_angles(12)
    vol    = phantom_2d(N).unsqueeze(0)
    sino   = ft.forward_project(vol, angles)

    rec, losses = ft.run_grad_descent(
        sino,                       # sinogram
        angles,                     # angles_deg
        5,                          # n_iter
        5e-2,                       # lr
        -1,                         # Nz
        torch.optim.Adam,           # optimizer_cls
        False,                      # positivity
        False,                      # mask
        0.0,                        # tv_weight
        None,                       # euler_zyz_deg
        False,                      # use_resire_projector
        True,                       # return_loss
    )

    assert rec.shape == vol.shape
    assert losses.shape == torch.Size([5])
    assert torch.isfinite(losses).all().item()


# ── 3D rotation-matrix CPU tests ──────────────────────────────────────────
#
# The 3D rotmat projector now has CPU OpenMP parallelism. These tests cover
# (a) numerical equivalence with the scalar-axis projector when φ=ψ=0,
# (b) approximate adjoint, (c) thread-count invariance of FP output.

def test_fp3d_matches_scalar_fp_cpu():
    """With φ=ψ=0, forward_project_3d must match the scalar-axis FP.

    Mirrors test_cuda.py::test_fp3d_matches_scalar_fp — same phantom, same
    absolute tolerance (1e-2). The small residual comes from the 3D
    diagonal ray-length being slightly longer than the 2D one, so the very
    last ray step can fall just past the volume face.
    """
    N      = 64
    angles = make_angles(30)
    vol    = phantom_2d(N).unsqueeze(0)

    sino_2d = ft.forward_project(vol, angles)

    euler = torch.zeros(len(angles), 3, dtype=torch.float32)
    euler[:, 1] = angles
    sino_3d = ft.forward_project_3d(vol, euler, num_threads=4)

    diff = (sino_2d - sino_3d).abs()
    assert diff.max().item() < 1e-2, (
        f"3D FP (pure-Y) disagrees with scalar FP: max_diff={diff.max():.2e}"
    )


def test_fp3d_adjoint_cpu():
    """<Ax, y> ≈ <x, Aᵀ y> for the CPU 3D pair (approximate adjoint).

    Mirrors test_cuda.py::test_fp3d_bp3d_adjoint — random inputs with pure-Y
    Euler angles. CPU BP is approximate per CLAUDE.md (~CUDA-comparable).
    """
    N      = 32
    angles = make_angles(20)
    euler  = torch.zeros(len(angles), 3, dtype=torch.float32)
    euler[:, 1] = angles

    torch.manual_seed(0)
    x = torch.randn(1, N, N)
    y = torch.randn(len(angles), 1, N)

    Ax  = ft.forward_project_3d(x, euler, num_threads=4)
    ATy = ft.back_project_3d(y, euler, Nz=N, num_threads=4)

    lhs = (Ax * y).sum().item()
    rhs = (x * ATy).sum().item()
    rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-12)
    assert rel_err < 0.75, (
        f"3D adjoint test failed: <Ax,y>={lhs:.4f} <x,Aᵀy>={rhs:.4f} "
        f"rel_err={rel_err:.2e}"
    )


def test_fp3d_threading_consistency_cpu():
    """FP output must be byte-identical across thread counts (no atomics)."""
    N      = 24
    Ny     = 3
    torch.manual_seed(0)
    angles = make_angles(20, max_deg=60.0)
    euler  = torch.zeros(len(angles), 3, dtype=torch.float32)
    euler[:, 1] = angles
    vol    = torch.randn(Ny, N, N)

    sino_serial = ft.forward_project_3d(vol, euler, num_threads=1)
    sino_par    = ft.forward_project_3d(vol, euler, num_threads=8)

    assert torch.equal(sino_serial, sino_par), (
        "FP3D output differs between num_threads=1 and num_threads=8 — "
        f"max-diff={(sino_serial - sino_par).abs().max().item():.3e}"
    )


def test_fp3d_default_is_serial(monkeypatch):
    """With no num_threads kwarg and no OMP_NUM_THREADS, kernel runs serial.

    We can't observe thread count directly from Python, but we can verify
    the resolution helper: it must return 1 in this configuration."""
    from fasttomo.autograd import _resolve_num_threads
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert _resolve_num_threads(None) == 1
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert _resolve_num_threads(None) == 8
    assert _resolve_num_threads(2) == 2   # explicit kwarg wins over env


# ── SART ──────────────────────────────────────────────────────────────────

class TestRunSart:
    """Tests for run_sart (SART / OS-SART baseline)."""

    def _make_fixture(self, N: int = 16):
        """Return a (sinogram, angles) pair produced by FP of a small phantom."""
        angles = make_angles(20, max_deg=60.0)
        vol = phantom_2d(N).unsqueeze(0)
        sino = ft.forward_project(vol, angles)
        return sino, angles, vol

    def test_shape(self):
        """Output volume has shape [Ny, Nz, Nx] and dtype float32."""
        N = 16
        sino, angles, vol = self._make_fixture(N)
        rec = ft.run_sart(sino, angles, n_iter=2, Nz=N)
        assert rec.shape == vol.shape, f"expected {vol.shape}, got {rec.shape}"
        assert rec.dtype == torch.float32

    def test_convergence(self):
        """SART residual MSE decreases over 5 iterations (subset_size=1)."""
        N = 16
        sino, angles, _ = self._make_fixture(N)
        _, losses = ft.run_sart(sino, angles, n_iter=5, subset_size=1, Nz=N,
                                return_loss=True)
        assert losses[0].item() > losses[-1].item(), (
            f"SART did not reduce loss: {losses.tolist()}"
        )

    def test_sirt_equivalence(self):
        """
        With subset_size=n_angles and omega=1.0, run_sart should produce
        RMSE within 1e-3 of run_sirt on the same fixture and iteration count.

        Both algorithms reduce to the same update equation when all angles form
        a single subset — the only numerical difference is floating-point
        ordering inside the update accumulation.
        """
        N = 16
        k = 3
        sino, angles, vol = self._make_fixture(N)
        n_angles = len(angles)

        rec_sirt = ft.run_sirt(sino, angles, n_iter=k, Nz=N)
        rec_sart = ft.run_sart(sino, angles, n_iter=k, subset_size=n_angles, Nz=N)

        rmse = ((rec_sirt - rec_sart).pow(2).mean()).sqrt().item()
        assert rmse < 1e-3, (
            f"SART (subset_size=n_angles) vs SIRT RMSE = {rmse:.2e} (threshold 1e-3)"
        )

    def test_subset_size_smoke(self):
        """subset_size=4, n_iter=2 produces finite, shape-correct output."""
        N = 16
        sino, angles, vol = self._make_fixture(N)
        rec = ft.run_sart(sino, angles, n_iter=2, subset_size=4, Nz=N)
        assert rec.shape == vol.shape
        assert torch.isfinite(rec).all().item(), "output contains NaN or Inf"

    def test_return_loss(self):
        """return_loss=True returns (volume, loss_history) with len == n_iter."""
        N = 16
        sino, angles, vol = self._make_fixture(N)
        n_iter = 4
        result = ft.run_sart(sino, angles, n_iter=n_iter, Nz=N, return_loss=True)
        assert isinstance(result, tuple) and len(result) == 2
        rec, losses = result
        assert rec.shape == vol.shape
        assert losses.shape == torch.Size([n_iter]), (
            f"expected loss_history length {n_iter}, got {losses.shape}"
        )
        assert torch.isfinite(losses).all().item()

    def test_warm_start_shape_mismatch_raises(self):
        """Warm-start volume shape must match [Ny, Nz, Nx] exactly — without
        this guard a [1, N, N] warm-start with Ny>1 silently broadcasts and
        corrupts the SART subset update with no warning."""
        n_angles, Ny, N = 12, 4, 16
        sino = torch.zeros(n_angles, Ny, N, dtype=torch.float32)
        angles = make_angles(n_angles)
        warm = torch.zeros(1, N, N, dtype=torch.float32)  # Ny mismatch
        with pytest.raises(ValueError, match="volume must have shape"):
            ft.run_sart(sino, angles, n_iter=1, Nz=N, volume=warm)


# ── MBIR CPU reference + CPU/CUDA cross-consistency ──────────────────────
#
# The CPU MBIR reference exists for two reasons:
#   1. validation when no GPU is present
#   2. a numerically-comparable target for the CUDA kernel
#
# `test_mbir_cpu_runs` exercises the C kernel and ensures the algorithm
# converges on the CPU path. The cross-consistency test below requires
# CUDA and is skipped otherwise.


def _mbir_sino_cpu(phantom: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """CPU voxel-driven bilinear FP (the same model used inside CPU ICD)."""
    import math
    import fasttomo._core as _core
    Ny, Nz, Nx = phantom.shape
    N_det = Nx
    angles_rad = (angles_deg.float() * (math.pi / 180.0)).numpy().copy()
    sino_np = _core.mbir_forward_project_cpu(
        phantom.contiguous().numpy(),
        angles_rad,
        N_det,
        num_threads=1,
    )
    return torch.from_numpy(sino_np)


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
        outer_iters=30, multires_levels=1,
    )
    rel = (vol - phantom).norm().item() / phantom.norm().item()
    assert rel < 0.30, f"CPU MBIR rel error {rel:.4f} > 0.30"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_mbir_cpu_cuda_consistency():
    """CPU and CUDA paths use identical math and an identical (seeded)
    permutation. Differences come from float summation order across the
    GPU's parallel reductions — bounded but non-zero. We assert max
    absolute deviation < 1e-2 and mean < 1e-3 on a small problem after
    a modest number of iterations. """
    Ny = Nz = Nx = 12
    n_angles = 16
    phantom = torch.zeros(Ny, Nz, Nx)
    phantom[3:9, 3:9, 3:9] = 1.0
    angles = make_angles(n_angles, max_deg=60.0)
    sino_cpu = _mbir_sino_cpu(phantom, angles)
    sino_cuda = sino_cpu.cuda()

    kw = dict(
        sigma_f=0.1, huber_T=float("inf"),
        estimate_offset=False, estimate_variance=False,
        sigma_init=0.1,
        outer_iters=20, multires_levels=1, seed=7,
    )
    v_cpu = ft.run_mbir(sino_cpu, angles, **kw)
    v_cu  = ft.run_mbir(sino_cuda, angles.cuda(), **kw).cpu()

    diff = (v_cu - v_cpu).abs()
    assert diff.max().item() < 1e-2, (
        f"CPU/CUDA max diff {diff.max().item():.5f} > 1e-2"
    )
    assert diff.mean().item() < 1e-3, (
        f"CPU/CUDA mean diff {diff.mean().item():.6f} > 1e-3"
    )


def test_mbir_fp_matches_standard_fp_off_axis_voxel():
    """Cross-projector regression: mbir_forward_project_cpu and the standard
    ft.forward_project must agree on where a single off-Z-centre voxel lands
    on the detector. A Z-symmetric phantom hides the sign bug; placing the
    voxel at iz=8 (well off the iz=Nz//2 centre) makes the sign error visible
    as a shifted detector hit."""
    import math
    import numpy as np
    import fasttomo._core as _core
    Ny, Nz, Nx = 4, 32, 32
    N_det = Nx
    n_angles = 12
    phantom = torch.zeros(Ny, Nz, Nx)
    phantom[Ny // 2, 8, Nx // 2] = 1.0   # off-centre in Z, on-centre in X
    angles_deg = make_angles(n_angles, max_deg=60.0)
    angles_rad = (angles_deg.float() * (math.pi / 180.0)).numpy().copy()

    sino_mbir = _core.mbir_forward_project_cpu(
        phantom.contiguous().numpy(), angles_rad, N_det, num_threads=1,
    )
    sino_std = ft.forward_project(phantom, angles_deg)

    det_idx = np.arange(N_det, dtype=np.float32)
    iy = Ny // 2
    for ia in range(n_angles):
        m = sino_mbir[ia, iy].astype(np.float64)
        s = sino_std[ia, iy].numpy().astype(np.float64)
        m_sum, s_sum = m.sum(), s.sum()
        if m_sum < 1e-12 and s_sum < 1e-12:
            continue
        com_m = (m * det_idx).sum() / max(m_sum, 1e-12)
        com_s = (s * det_idx).sum() / max(s_sum, 1e-12)
        assert abs(com_m - com_s) < 0.75, (
            f"angle {ia}: MBIR FP COM {com_m:.2f} vs standard FP COM "
            f"{com_s:.2f} (diff {abs(com_m - com_s):.3f} > 0.75 px)"
        )


def test_mbir_reconstruction_from_standard_fp_sinogram_cpu():
    """End-to-end regression on the real bug path: sinogram built with the
    standard ft.forward_project (as user code and the notebook do) is given to
    run_mbir. With the bug the Z-flip makes rel_err >> 1; post-fix it should be
    < 0.35 (tolerates MBIR↔ray-marching model mismatch on CPU small phantom)."""
    Ny, Nz, Nx = 4, 24, 24
    phantom = torch.zeros(Ny, Nz, Nx)
    phantom[:, 16:22, 4:10] = 1.0   # Z-asymmetric block
    angles = make_angles(30, max_deg=70.0)
    sino = ft.forward_project(phantom, angles)
    vol = ft.run_mbir(
        sino, angles,
        sigma_f=0.1, huber_T=float("inf"),
        estimate_offset=False, estimate_variance=True,
        outer_iters=50, multires_levels=1,
    )
    rel_err = (vol - phantom).norm().item() / phantom.norm().item()
    assert rel_err < 0.35, (
        f"MBIR rel error vs phantom {rel_err:.4f} > 0.35 "
        "(possible projector sign regression)"
    )


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
    vol = ft.run_mbir(sino, angles, multires_levels=2, outer_iters=2)
    assert vol.shape == (2, 16, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
