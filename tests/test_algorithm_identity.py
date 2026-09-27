"""
Bit-identity of the projection ops and of every algorithm that goes through
them, pinned before the projection-operator refactor (ARCH P1 / P11-F2).

`tests/data/algorithm_golden_main.pt` holds the outputs of `_cases(device)`
on origin/main 5362a6b (generator: `tests/data/algorithm_golden_main_gen.py`,
run against a checkout of that commit's ``fasttomo/``). Only the public API
is called, so the same function runs before and after the refactor.

Exception: ``cpu/mbir`` and ``cuda/mbir`` were regenerated when the MBIR
voxel order moved from a host NumPy ``rng.permutation`` per slice to a keyed
Feistel permutation built in the CUDA kernel (and mirrored in NumPy for the
CPU sweep). With the old order injected, the new code reproduced the old
golden bit for bit on both backends; only the order changed.

Exception: the ramp-filter and FBP keys (``_fft_ramp``, ``fbp``,
``fbp_mask``, ``fbp_h`` and, on CUDA, ``fbp_proj``) were regenerated when
``ramp_filter`` became the zero-padded spatial-domain Ram-Lak filter (the
unpadded ``|xi|`` filter reconstructed a unit-density cylinder at 0.57 to
0.96 instead of 1). Only those 9 keys were replaced; every other tensor of
the file is byte-identical to before.

GENFIRE, RESIRE and run_ad_recon have their own goldens
(test_frozen_identity.py, test_algorithms_genfire_resire.py).

Comparison:

* Keys listed in ``ATOMIC_KEYS`` come from CUDA kernels that accumulate with
  fp32 atomics (``grad_angle``, ``grad_rotmat``, the RESIRE forward
  projection, MBIR's forward projection), so they are not bit-reproducible
  even main-vs-main; they are compared at 2e-6 relative / absolute (the
  observed main-vs-main spread is below 1e-6). The TV-loss value is no longer
  one of them: since PR #92 its CUDA forward is a fixed-order two-pass
  reduction, so ``cuda/gd_tv`` and ``cuda/gd_tv_loss`` are ordinary
  deterministic keys.
* Every other key is deterministic and compared with ``torch.equal``, but
  only where the machine's arithmetic matches the one the golden was verified
  on: the golden stores the arithmetic fingerprints of those environments
  (``tests/_golden.py``: matmul, reductions, FFTs, transcendentals, fp16
  rounding, interpolation, grid_sample and a backward pass run on the
  device, plus the CPU SIMD dispatch
  level or the GPU compute capability), because the last bit depends on the
  CPU SIMD / BLAS / FFT paths and on the GPU's texture unit, not on the torch
  version (two CI runners with the same torch differ by a few ulp).
  Elsewhere they fall back to 1e-6 (CPU keys) or 1e-5 (CUDA keys), and fp16
  outputs to one fp16 ulp (2**-10 relative / absolute); for the keys in
  ``FFT_KEYS`` the absolute part of that tolerance is scaled by the tensor's
  largest value (FFT round-off follows the transform's norm, not each bin's
  size). The fallback is reported as a ``GoldenNotExactWarning``. The ``_fft_ramp`` / ``_fft_rfft2`` keys (the two torch.fft uses
  on their own) are ordinary deterministic keys; the arithmetic fingerprint
  covers the FFT.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from _golden import exact_env, peak_close, tensors

import fasttomo as ft

GOLDEN = Path(__file__).parent / "data" / "algorithm_golden_main.pt"
HAS_CUDA = torch.cuda.is_available()

# Tolerances where the arithmetic fingerprint does not match (see the module
# docstring).
LOOSE_TOL = {"cpu": 1e-6, "cuda": 1e-5}
FP16_TOL = 2.0 ** -10

# Outputs that go through torch.fft (ramp filter, FBP, the W-BP / wSIRT
# deconvolution). On the loose branch their tolerance is relative to the
# tensor's largest value (``_golden.peak_close``), because FFT round-off is
# set by the transform's norm, not by each bin (CI run 36331234339 failed an
# elementwise 1e-6 on a 3.8e-6 difference, 5.7e-8 of the peak). Names
# without the device prefix and the ``#i`` tuple index.
FFT_KEYS = {"_fft_ramp", "_fft_rfft2", "fbp", "fbp_mask", "fbp_h", "fbp_proj", "wbp",
            "wsirt", "wsirt_cons"}

# Outputs of fp32-atomics kernels (not reproducible run to run). cuda/gd_tv
# and cuda/gd_tv_loss are not here: the TV forward is a fixed-order reduction
# since PR #92, so they get the deterministic-key check.
ATOMIC_KEYS = {
    "cuda/fp_grad_angles",
    "cuda/fp3d_grad_euler",
    "cuda/fp3d_shift_grad_euler",
    "cuda/proj_rotmat_grad_mats",
    "cuda/proj_rotmat_grad_mats_stale",
    "cuda/gd_resire",
    "cuda/gd_resire_loss",
    "cuda/mbir",
}


# ── inputs ────────────────────────────────────────────────────────────────


def _inputs(device):
    g = torch.Generator().manual_seed(1234)
    Ny, N, n = 4, 16, 9
    vol = torch.zeros(Ny, N, N)
    vol[:, 4:12, 5:11] = 1.0
    vol[1:3, 6:9, 6:10] = 2.0
    vol += 0.05 * torch.rand(Ny, N, N, generator=g)
    ang = torch.linspace(-60, 60, n)
    eul = torch.stack([torch.linspace(-4, 4, n), ang, torch.linspace(3, -3, n)], 1)
    w_sino = torch.rand(n, Ny, N, generator=g)
    w_vol = torch.rand(Ny, N, N, generator=g)
    sm = torch.zeros(1, N, N)
    sm[:, 2:14, 2:14] = 1.0
    shifts = torch.stack([torch.linspace(-0.7, 0.7, n), torch.linspace(0.3, -0.3, n)], 1)
    rect = torch.rand(Ny, 12, N, generator=g)          # Nz != Nx volume
    w_wide = torch.rand(n, Ny, 20, generator=g)        # N_det = 20 != Nx
    t = dict(vol=vol, ang=ang, eul=eul, w_sino=w_sino, w_vol=w_vol, sm=sm,
             shifts=shifts, rect=rect, w_wide=w_wide)
    t = {k: v.to(device) for k, v in t.items()}
    t["sino"] = ft.forward_project(t["vol"], t["ang"])
    t["sino3d"] = ft.forward_project_3d(t["vol"], t["eul"])
    t["N"] = N
    return t


def _grad(fn, *leaves):
    leaves = [x.detach().clone().requires_grad_(True) for x in leaves]
    fn(*leaves).backward()
    return [x.grad for x in leaves]


# ── cases ─────────────────────────────────────────────────────────────────


def _projection_cases(t, dev, out):
    vol, ang, eul, ws, wv = t["vol"], t["ang"], t["eul"], t["w_sino"], t["w_vol"]
    N = t["N"]
    out["fp"] = ft.forward_project(vol, ang)
    out["fp_ndet"] = ft.forward_project(vol, ang, N_det=20)
    out["fp_rect"] = ft.forward_project(t["rect"], ang)
    out["bp"] = ft.back_project(ws, ang)
    out["bp_nz"] = ft.back_project(ws, ang, Nz=12)
    out["bp_wide"] = ft.back_project(t["w_wide"], ang, Nz=12, Nx=N)
    out["fp3d"] = ft.forward_project_3d(vol, eul)
    out["fp3d_ndet"] = ft.forward_project_3d(vol, eul, N_det=20)
    out["bp3d"] = ft.back_project_3d(ws, eul)
    out["bp3d_wide"] = ft.back_project_3d(t["w_wide"], eul, Nz=12, Nx=N)
    out["fp3d_shift"] = ft.forward_project_3d(vol, eul, shifts=t["shifts"])
    h = torch.float16
    out["fp_h"] = ft.forward_project(vol.to(h), ang)
    out["bp_h"] = ft.back_project(ws.to(h), ang)
    out["fp3d_h"] = ft.forward_project_3d(vol.to(h), eul)
    out["bp3d_h"] = ft.back_project_3d(ws.to(h), eul)

    # gradients through the Functions
    gv, ga = _grad(lambda v, a: (ft.forward_project(v, a) * ws).sum(), vol, ang)
    out["fp_grad_vol"], out["fp_grad_angles"] = gv, ga
    (gs,) = _grad(lambda s: (ft.back_project(s, ang, Nz=12) * t["rect"]).sum(), ws)
    out["bp_grad_sino"] = gs
    (gr,) = _grad(lambda v: (ft.forward_project(v, ang, N_det=20) * t["w_wide"]).sum(),
                  t["rect"])
    out["fp_rect_grad_vol"] = gr
    gv, ge = _grad(lambda v, e: (ft.forward_project_3d(v, e) * ws).sum(), vol, eul)
    out["fp3d_grad_vol"], out["fp3d_grad_euler"] = gv, ge
    gv, ge, gsh = _grad(lambda v, e, s: (ft.forward_project_3d(v, e, shifts=s) * ws).sum(),
                        vol, eul, t["shifts"])
    out["fp3d_shift_grad_vol"], out["fp3d_shift_grad_euler"] = gv, ge
    out["fp3d_shift_grad_shifts"] = gsh
    (gs,) = _grad(lambda s: (ft.back_project_3d(s, eul) * wv).sum(), ws)
    out["bp3d_grad_sino"] = gs
    (gs,) = _grad(lambda s: (ft.back_project(s.to(h), ang).float() * wv).sum(), ws)
    out["bp_h_grad_sino"] = gs

    if dev == "cuda":
        # the cached projectors (CUDA-only on main)
        proj = ft.ParallelProjector(tuple(vol.shape), ang, device=dev)
        out["proj_par_fp"] = proj.forward(vol)
        out["proj_par_bp"] = proj.back(ws)
        (out["proj_par_grad_vol"],) = _grad(lambda v: (proj.forward(v) * ws).sum(), vol)
        (out["proj_par_grad_sino"],) = _grad(lambda s: (proj.back(s) * wv).sum(), ws)
        ph = ft.ParallelProjector(tuple(vol.shape), ang, device=dev)
        out["proj_par_fp_h"] = ph.forward(vol.to(h))
        out["proj_par_bp_h"] = ph.back(ws.to(h))
        mats = ft.euler_zyz_to_rotmat(eul)
        rp = ft.RotmatProjector(tuple(vol.shape), n_angles=ang.numel(), device=dev)
        out["proj_rotmat_fp"] = rp.forward(vol, mats)
        out["proj_rotmat_bp"] = rp.back(ws, mats)
        gv, gm = _grad(lambda v, m: (rp.forward(v, m) * ws).sum(), vol, mats)
        out["proj_rotmat_grad_vol"], out["proj_rotmat_grad_mats"] = gv, gm
        (out["proj_rotmat_grad_sino"],) = _grad(lambda s: (rp.back(s, mats) * wv).sum(), ws)
        # a later forward overwrites the cached volume: the stale branch
        # takes the uncached gradient kernel
        va = vol.clone().requires_grad_(True)
        ma = mats.clone().requires_grad_(True)
        sa = rp.forward(va, ma)
        sb = rp.forward(vol * 0.5, mats)
        ((sa * ws).sum() + (sb * ws).sum()).backward()
        out["proj_rotmat_grad_mats_stale"] = ma.grad


def _algorithm_cases(t, dev, out):
    sino, ang, eul, N, sm = t["sino"], t["ang"], t["eul"], t["N"], t["sm"]
    h = torch.float16
    ang_cpu = ang.cpu()

    # FFT fingerprint: the two torch.fft uses below, on their own
    out["_fft_ramp"] = ft.ramp_filter(sino)
    out["_fft_rfft2"] = torch.view_as_real(torch.fft.rfft2(t["vol"]))
    out["fbp"] = ft.run_fbp(sino, ang)
    out["fbp_mask"] = ft.run_fbp(sino, ang, Nz=12, positivity=True, mask=True)
    out["fbp_h"] = ft.run_fbp(sino.to(h), ang)
    out["wbp"] = ft.run_weighted_backprojection(sino, ang, positivity=True)

    out["sirt"] = ft.run_sirt(sino, ang, 6)
    out["sirt_loss"] = ft.run_sirt(sino, ang, 6, return_loss=True)
    out["sirt_cons"] = ft.run_sirt(sino, ang, 5, Nz=12, positivity=True, mask=True,
                                   sparsity=1e-3, omega=0.8)
    out["sirt_support"] = ft.run_sirt(sino, ang, 5, support_mask=sm)
    out["sirt_euler"] = ft.run_sirt(t["sino3d"], ang, 5, euler_zyz_deg=eul, return_loss=True)
    out["sirt_h"] = ft.run_sirt(sino.to(h), ang, 4)
    out["sirt_warm"] = ft.run_sirt(sino, ang, 3, warm_start=out["sirt"].to(dev))

    out["sart"] = ft.run_sart(sino, ang, 3)
    out["sart_sub4"] = ft.run_sart(sino, ang, 3, subset_size=4, positivity=True,
                                   return_loss=True)
    out["sart_all"] = ft.run_sart(sino, ang, 2, subset_size=9, mask=True)
    out["sart_euler"] = ft.run_sart(t["sino3d"], ang, 2, subset_size=2, euler_zyz_deg=eul,
                                    return_loss=True)
    out["sart_h"] = ft.run_sart(sino.to(h), ang, 2, subset_size=3)

    out["wsirt"] = ft.run_wsirt(sino, ang, 4, return_loss=True)
    out["wsirt_cons"] = ft.run_wsirt(sino, ang, 3, positivity=True, support_mask=sm,
                                     sparsity=1e-3)

    # the Lipschitz power iteration starts from torch.randn (global RNG)
    torch.manual_seed(0)
    out["fista"] = ft.run_fista(sino, ang, 6, return_loss=True)
    out["fista_L"] = ft.run_fista(sino, ang, 5, L=400.0, positivity=True, mask=True)
    torch.manual_seed(0)
    out["fista_euler"] = ft.run_fista(t["sino3d"], ang, 4, euler_zyz_deg=eul)
    torch.manual_seed(0)
    out["fista_h"] = ft.run_fista(sino.to(h), ang, 4)
    torch.manual_seed(0)
    out["fista_tv"] = ft.run_fista_tv(sino, ang, 5, tv_weight=0.05, return_loss=True)
    torch.manual_seed(0)
    out["fista_tv_ch"] = ft.run_fista_tv(sino, ang, 4, tv_weight=0.05, tv_inner="chambolle",
                                         positivity=True)

    out["gd"], out["gd_loss"] = ft.run_grad_descent(sino, ang, 8, lr=0.05, return_loss=True)
    out["gd_euler"] = ft.run_grad_descent(t["sino3d"], None, 6, lr=0.05, euler_zyz_deg=eul,
                                          positivity=True)
    out["gd_sgd"] = ft.run_grad_descent(sino, ang, 5, lr=1e-3,
                                        optimizer_cls=torch.optim.SGD, mask=True)
    out["gd_tv"], out["gd_tv_loss"] = ft.run_grad_descent(sino, ang, 5, lr=0.05,
                                                          tv_weight=1e-2, return_loss=True)
    if dev == "cuda":
        out["gd_resire"], out["gd_resire_loss"] = ft.run_grad_descent(
            t["sino3d"], None, 5, lr=0.05, euler_zyz_deg=eul, use_resire_projector=True,
            return_loss=True)
        proj = ft.ParallelProjector((4, N, N), ang, device=dev)
        out["sirt_proj"] = ft.run_sirt(sino, ang, 5, projector=proj)
        out["sirt_proj_again"] = ft.run_sirt(sino, ang, 5, projector=proj, omega=0.5)
        torch.manual_seed(0)
        out["fista_proj"] = ft.run_fista(sino, ang, 4, projector=proj)
        out["fbp_proj"] = ft.run_fbp(sino, ang, projector=proj)
        rp = ft.RotmatProjector((4, N, N), n_angles=ang.numel(), device=dev)
        out["sirt_rotmat_proj"] = ft.run_sirt(t["sino3d"], ang, 4, euler_zyz_deg=eul,
                                              projector=rp)

    out["mbir"] = ft.run_mbir(sino, ang, n_iter=3, multires_levels=1, seed=3)

    vol = out["sirt"].to(dev)
    out["refine"] = ft.refine_angles(vol, sino, ang_cpu + 0.7, search_range_deg=2.0,
                                     n_steps=9)
    out["refine_zyz"] = ft.refine_angles(vol, t["sino3d"], euler_zyz_deg=eul.cpu() + 0.5,
                                         search_range_deg=1.5, n_steps=5)


def _cases(dev: str) -> dict[str, torch.Tensor]:
    """Every output, keyed ``"<device>/<case>"``, detached on the CPU."""
    t = _inputs(dev)
    out: dict[str, object] = {}
    _projection_cases(t, dev, out)
    _algorithm_cases(t, dev, out)
    flat = {}
    for k, v in out.items():
        if isinstance(v, tuple):
            for i, x in enumerate(v):
                flat[f"{dev}/{k}#{i}"] = x
        else:
            flat[f"{dev}/{k}"] = v
    return {k: v.detach().cpu() for k, v in flat.items()}


def all_cases() -> dict[str, torch.Tensor]:
    out = _cases("cpu")
    if HAS_CUDA:
        out.update(_cases("cuda"))
    return out


# ── tests ─────────────────────────────────────────────────────────────────


def _compare(dev: str) -> None:
    raw = torch.load(GOLDEN, weights_only=True)
    golden = tensors(raw)
    now = _cases(dev)
    want = {k for k in golden if k.startswith(dev + "/")}
    assert set(now) == want
    exact = exact_env(raw, dev, what=GOLDEN.name)
    bad = []
    for k in sorted(want):
        a, b = now[k], golden[k]
        assert a.shape == b.shape and a.dtype == b.dtype, k
        base = k.split("#")[0]
        diff = (a.double() - b.double()).abs().max().item() if a.numel() else 0.0
        scale = b.double().abs().max().item() if b.numel() else 0.0
        how = ""
        if base in ATOMIC_KEYS:
            ok = torch.allclose(a, b, rtol=2e-6, atol=2e-6)
            how = "atomics, 2e-6"
        elif exact:
            ok = torch.equal(a, b)
            how = "exact"
        else:
            tol = FP16_TOL if b.dtype == torch.float16 else LOOSE_TOL[dev]
            if base.split("/", 1)[1] in FFT_KEYS:
                ok = peak_close(a, b, tol)[0]
                how = f"FFT, {tol:.0e} of max(1, peak)"
            else:
                ok = torch.allclose(a, b, rtol=tol, atol=tol)
                how = f"{tol:.0e} elementwise"
        if not ok:
            bad.append(f"{k}: max|diff| {diff:.3e} (max|golden| {scale:.3e}; {how})")
    assert not bad, (f"{len(bad)} of {len(want)} outputs differ from the golden "
                     f"(exact={exact}, torch {torch.__version__}):\n"
                     + "\n".join(bad))


def test_cpu_outputs_identical_to_main():
    _compare("cpu")


@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA GPU available")
def test_cuda_outputs_identical_to_main():
    _compare("cuda")
