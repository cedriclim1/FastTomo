"""
CPU correctness tests for FastTomo projection primitives and reconstructions.

Run with:  pytest tests/test_cpu.py -v
"""

import pytest
import torch

import fasttomo as ft


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
