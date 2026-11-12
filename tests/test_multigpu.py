"""Multi-GPU ``devices=`` runs against the serial run on one GPU.

Slab split (FBP, SIRT, SART, FISTA, FISTA-TV, RESIRE): bit-identical volumes
(FISTA when L is given; its own power iteration matches to rounding). AD and
gradient descent (slab split with a one-row TV halo, or view split with
summed gradients) match to rounding; with Adam and a dominant TV term the
difference is bounded by the effect of a 1e-7 perturbation of the start,
since Adam's sign-like step amplifies rounding near zero gradients. Loss histories and callbacks are
checked too.
"""

from __future__ import annotations

import math

import pytest
import torch

import fasttomo as ft
from conftest import N_GPUS, needs_2_gpus, rel_l2
from fasttomo.algorithms import _multigpu
from fasttomo.configs import AdReconConfig, OptimConfig, PoseRefinementConfig, RegularizerConfig

pytestmark = needs_2_gpus

DEV = torch.device("cuda:0")
NY, N, NA = 23, 40, 31


def _data(seed: int = 0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    vol = torch.zeros(NY, N, N)
    yy, zz, xx = torch.meshgrid(torch.arange(NY), torch.arange(N), torch.arange(N), indexing="ij")
    for _ in range(6):
        c = torch.rand(3, generator=g) * torch.tensor([NY, N * 0.6, N * 0.6]) + torch.tensor(
            [0.0, N * 0.2, N * 0.2])
        r = 3 + 6 * float(torch.rand(1, generator=g))
        vol += ((yy - c[0]) ** 2 + (zz - c[1]) ** 2 + (xx - c[2]) ** 2 < r * r).float()
    ang = torch.linspace(-70, 70, NA)
    sino = ft.forward_project(vol.to(DEV), ang.to(DEV))
    sino = sino + 0.05 * sino.std() * torch.randn(sino.shape, generator=g).to(DEV)
    return sino.to(dtype).contiguous(), ang, vol


def _device_sets():
    sets = [[0, 1]]
    if N_GPUS >= 3:
        sets.append([0, 1, 2])
    if N_GPUS >= 4:
        sets.append("all")
    return sets


DEVICE_SETS = _device_sets()


def test_split_is_balanced_and_contiguous():
    assert _multigpu._split(10, 4) == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert _multigpu._split(2, 4) == [(0, 1), (1, 2)]
    assert _multigpu._split(7, 1) == [(0, 7)]


def test_resolve_devices():
    sino = torch.zeros(2, 3, 4, device=DEV)
    assert _multigpu._resolve_devices(None, sino, "f") is None
    assert _multigpu._resolve_devices([0], sino, "f") is None
    assert _multigpu._resolve_devices("all", sino, "f") == [
        torch.device("cuda", i) for i in range(N_GPUS)]
    assert _multigpu._resolve_devices(2, sino, "f") == [torch.device("cuda", i) for i in range(2)]
    assert _multigpu._resolve_devices(["cuda:1", 0], sino, "f") == [
        torch.device("cuda", 1), torch.device("cuda", 0)]
    with pytest.raises(ValueError, match="duplicates"):
        _multigpu._resolve_devices([0, 0], sino, "f")
    with pytest.raises(ValueError, match="CUDA devices"):
        _multigpu._resolve_devices(["cpu", 0], sino, "f")
    with pytest.raises(ValueError, match="CUDA sinogram"):
        _multigpu._resolve_devices([0, 1], sino.cpu(), "f")
    with pytest.raises(ValueError, match="visible"):
        _multigpu._resolve_devices(N_GPUS + 1, sino, "f")


def test_parallel_projection_separates_along_tilt_axis():
    """The premise of the slab split: rows of FP/BP are independent."""
    sino, ang, vol = _data()
    full_fp = ft.forward_project(vol.to(DEV), ang.to(DEV))
    full_bp = ft.back_project(sino, ang.to(DEV), Nz=N)
    for s, e in _multigpu._split(NY, 3):
        assert torch.equal(ft.forward_project(vol[s:e].contiguous().to(DEV), ang.to(DEV)),
                           full_fp[:, s:e])
        assert torch.equal(ft.back_project(sino[:, s:e].contiguous(), ang.to(DEV), Nz=N),
                           full_bp[s:e])


@pytest.mark.parametrize("devices", DEVICE_SETS)
@pytest.mark.parametrize("window", ["none", "hann"])
def test_fbp_bit_identical(devices, window):
    sino, ang, _ = _data()
    ref = ft.run_fbp(sino, ang, window=window, positivity=True, mask=True)
    out = ft.run_fbp(sino, ang, window=window, positivity=True, mask=True, devices=devices)
    assert out.device == sino.device
    assert torch.equal(out, ref)


def _row_support():
    m = torch.ones(NY, N, N)
    m[:, :4] = 0
    m[5:9, :, -6:] = 0
    return m


@pytest.mark.parametrize("devices", DEVICE_SETS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sirt_bit_identical(devices, dtype):
    sino, ang, _ = _data(dtype=dtype)
    kw = dict(positivity=True, support_mask=_row_support(), sparsity=1e-3, return_loss=True)
    ref, ref_loss = ft.run_sirt(sino, ang, 12, **kw)
    out, loss = ft.run_sirt(sino, ang, 12, devices=devices, **kw)
    assert torch.equal(out, ref)
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    torch.testing.assert_close(loss, ref_loss, rtol=rtol, atol=0)


@pytest.mark.parametrize("devices", DEVICE_SETS)
@pytest.mark.parametrize("view_order", ["sequential", "golden"])
def test_sart_bit_identical(devices, view_order):
    sino, ang, _ = _data()
    ws = torch.rand(NY, N, N, generator=torch.Generator().manual_seed(3))
    kw = dict(subset_size=4, view_order=view_order, positivity=True, mask=True,
              warm_start=ws.to(DEV), return_loss=True)
    ref, ref_loss = ft.run_sart(sino, ang, 3, **kw)
    out, loss = ft.run_sart(sino, ang, 3, devices=devices, **kw)
    assert torch.equal(out, ref)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=0)


@pytest.mark.parametrize("devices", DEVICE_SETS)
def test_fista_given_L_bit_identical(devices):
    sino, ang, _ = _data()
    ref = ft.run_fista(sino, ang, 15, L=2.0e3, positivity=True)
    out = ft.run_fista(sino, ang, 15, L=2.0e3, positivity=True, devices=devices)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("devices", DEVICE_SETS)
def test_fista_power_iteration_matches(devices):
    sino, ang, _ = _data()
    torch.manual_seed(5)
    ref, ref_loss = ft.run_fista(sino, ang, 15, positivity=True, return_loss=True)
    torch.manual_seed(5)
    out, loss = ft.run_fista(sino, ang, 15, positivity=True, return_loss=True, devices=devices)
    assert rel_l2(out, ref) < 1e-5
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=0)


@pytest.mark.parametrize("devices", DEVICE_SETS)
@pytest.mark.parametrize("tv_inner", ["fgp", "chambolle"])
@pytest.mark.parametrize("support", ["row", "cylinder"])
def test_fista_tv_halo_prox_bit_identical(devices, tv_inner, support):
    """With L given, the halo-exchanged TV prox reproduces the full prox exactly."""
    sino, ang, _ = _data()
    kw = dict(tv_weight=50.0, L=2.0e3, tv_inner=tv_inner, tv_inner_iter=6, positivity=True)
    kw.update(support_mask=_row_support()) if support == "row" else kw.update(mask=True)
    ref = ft.run_fista_tv(sino, ang, 8, **kw)
    out = ft.run_fista_tv(sino, ang, 8, devices=devices, **kw)
    assert torch.equal(out, ref)


def test_fista_tv_halo_wider_than_a_block():
    """Blocks thinner than the halo gather rows from several neighbours."""
    sino, ang, _ = _data()
    kw = dict(tv_weight=50.0, L=2.0e3, tv_inner_iter=12)
    ref = ft.run_fista_tv(sino, ang, 4, **kw)
    out = ft.run_fista_tv(sino, ang, 4, devices=list(range(min(N_GPUS, 4))), **kw)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("parts", [2, 3, 5])
def test_slab_tv_pieces_reproduce_global_tv_and_gradient(parts):
    """`_slab_tv`'s owned values sum to the global TV, and owned + hidden give
    every row block exactly the global gradient of its rows."""
    g = torch.Generator().manual_seed(7)
    vol = torch.rand(NY, 12, 14, generator=g).to(DEV)
    full = vol.clone().requires_grad_(True)
    ft.tv_loss_3d(full).backward()
    total = 0.0
    for r, (s, e) in enumerate(_multigpu._split(NY, parts)):
        block = vol[s:e].clone().requires_grad_(True)
        ctx = _multigpu._RankContext(
            rank=r, n_ranks=parts, rows=(s, e, NY),
            halo=lambda v, s=s, e=e: (vol[s - 1:s] if s > 0 else None,
                                      vol[e:e + 1] if e < NY else None))
        owned, hidden = _multigpu._slab_tv(ctx, block, ft.tv_loss_3d)
        (owned if hidden is None else owned + hidden).backward()
        total += float(owned)
        torch.testing.assert_close(block.grad, full.grad[s:e], rtol=1e-5, atol=1e-7)
    assert math.isclose(total, float(ft.tv_loss_3d(vol)), rel_tol=1e-6)


@pytest.mark.parametrize("devices", DEVICE_SETS)
def test_ad_recon_strong_tv_matches(devices):
    """A TV weight that dominates the data term, so a wrong halo would show."""
    sino, ang, _ = _data()
    cfg = _ad_config(False, 5e1)
    ref, ref_loss = ft.run_ad_recon(sino, ang, 30, config=cfg, return_loss=True)
    no_tv = ft.run_ad_recon(sino, ang, 30, config=_ad_config(False, 0.0))
    assert rel_l2(no_tv, ref) > 1e-2
    out, loss = ft.run_ad_recon(sino, ang, 30, config=cfg, return_loss=True, devices=devices)
    ws = 1e-7 * torch.rand(NY, N, N, generator=torch.Generator().manual_seed(1)).to(DEV)
    jitter = rel_l2(ft.run_ad_recon(sino, ang, 30, config=cfg, warm_start=ws), ref)
    assert rel_l2(out, ref) < 2 * jitter + 1e-5
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=0)


@pytest.mark.parametrize("devices", DEVICE_SETS)
def test_grad_descent_sgd_strong_tv_matches(devices):
    """Plain gradient steps with a dominant TV term: the slab TV halo gives
    the serial gradient, so the iterates agree to rounding."""
    sino, ang, _ = _data()
    kw = dict(lr=1e-4, tv_weight=1e2, positivity=True, mask=True,
              optimizer_cls=torch.optim.SGD, return_loss=True)
    ref, ref_loss = ft.run_grad_descent(sino, ang.to(DEV), 30, **kw)
    out, loss = ft.run_grad_descent(sino, ang.to(DEV), 30, devices=devices, **kw)
    assert rel_l2(out, ref) < 1e-6
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=0)


def test_grad_descent_euler_view_split_matches():
    sino, ang, _ = _data()
    eul = torch.zeros(NA, 3)
    eul[:, 1] = ang
    kw = dict(lr=5e-2, tv_weight=1e-3, positivity=True, return_loss=True,
              euler_zyz_deg=eul.to(DEV))
    ref, ref_loss = ft.run_grad_descent(sino, None, 20, **kw)
    out, loss = ft.run_grad_descent(sino, None, 20, devices=[0, 1], **kw)
    assert rel_l2(out, ref) < 1e-4
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=0)


@pytest.mark.parametrize("devices", DEVICE_SETS)
def test_grad_descent_adam_tv_matches(devices):
    sino, ang, _ = _data()
    kw = dict(lr=5e-2, tv_weight=1e-3, positivity=True, mask=True, return_loss=True)
    ref, ref_loss = ft.run_grad_descent(sino, ang.to(DEV), 30, **kw)
    out, loss = ft.run_grad_descent(sino, ang.to(DEV), 30, devices=devices, **kw)
    assert rel_l2(out, ref) < 1e-4
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=0)


def test_grad_descent_sgd_default_lr_matches():
    sino, ang, _ = _data()
    kw = dict(optimizer_cls=torch.optim.SGD, return_loss=True)
    ref, ref_loss = ft.run_grad_descent(sino, ang.to(DEV), 20, **kw)
    out, loss = ft.run_grad_descent(sino, ang.to(DEV), 20, devices=[0, 1], **kw)
    assert rel_l2(out, ref) < 1e-5
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=0)


def _ad_config(pose: bool, tv: float, loss_fn: str = "mse") -> AdReconConfig:
    cfg = AdReconConfig(optim=OptimConfig(n_iter=40, lr_volume=5e-2, loss_fn=loss_fn),
                        reg=RegularizerConfig(tv_weight=tv))
    if pose:
        cfg.pose = PoseRefinementConfig(refine_pose=True, refine_shifts=True,
                                        lr_angles=1e-2, lr_shifts=1e-2,
                                        anchor_reference=True, in_plane_weight=1e-3)
    return cfg


@pytest.mark.parametrize("devices", DEVICE_SETS)
@pytest.mark.parametrize(("pose", "tv", "loss_fn"),
                         [(False, 0.0, "mse"), (False, 1e-3, "l1"), (True, 1e-3, "mse")])
def test_ad_recon_matches(devices, pose, tv, loss_fn):
    sino, ang, _ = _data()
    cfg = _ad_config(pose, tv, loss_fn)
    kw = dict(config=cfg, return_loss=True, return_pose=pose)
    ref = ft.run_ad_recon(sino, ang, **kw)
    out = ft.run_ad_recon(sino, ang, devices=devices, **kw)
    assert rel_l2(out[0], ref[0]) < 1e-4
    torch.testing.assert_close(out[1], ref[1], rtol=1e-4, atol=0)
    if pose:
        torch.testing.assert_close(out[2], ref[2], rtol=0, atol=1e-4)
        torch.testing.assert_close(out[3], ref[3], rtol=0, atol=1e-4)
        assert torch.all(out[2][int(ang.abs().argmin())] == ref[2][int(ang.abs().argmin())])


def test_ad_recon_replicas_stay_identical():
    """Every rank steps the same summed gradient: the replicas never drift."""
    sino, ang, _ = _data()
    seen = {}
    real_run = _multigpu._Group.run

    def spy(self, fn, contexts=None):
        res = real_run(self, fn, contexts)
        seen["vols"] = [r[0] if isinstance(r, tuple) else r for r in res]
        return res

    _multigpu._Group.run = spy
    try:
        ft.run_ad_recon(sino, ang, config=_ad_config(True, 1e-3), devices=[0, 1])
    finally:
        _multigpu._Group.run = real_run
    a, b = seen["vols"]
    assert torch.equal(a, b.to(a.device))


@pytest.mark.parametrize("name", ["sirt", "ad"])
def test_callbacks_merged_per_iteration(name):
    sino, ang, _ = _data()
    losses, comps, vols = [], [], []

    def loss_cb(i, loss, components=None):
        losses.append((i, loss))
        comps.append(components)

    def vol_cb(i, v):
        vols.append((i, tuple(v.shape), v.device))

    if name == "sirt":
        out, hist = ft.run_sirt(sino, ang, 6, return_loss=True, loss_callback=loss_cb,
                                volume_callback=vol_cb, devices=[0, 1])
        n = 6
    else:
        out, hist = ft.run_ad_recon(sino, ang, 6, config=_ad_config(False, 1e-3),
                                    return_loss=True, loss_callback=loss_cb,
                                    volume_callback=vol_cb, devices=[0, 1])
        n = 6
    assert [i for i, _ in losses] == list(range(n))
    assert [i for i, _, _ in vols] == list(range(n))
    assert all(shape == tuple(out.shape) for _, shape, _ in vols)
    torch.testing.assert_close(torch.tensor([v for _, v in losses]), hist, rtol=1e-6, atol=0)
    for (_, total), c in zip(losses, comps):
        assert math.isclose(sum(c.values()), total, rel_tol=1e-5)


def test_pose_callback_gathers_all_views():
    sino, ang, _ = _data()
    got = []
    ft.run_ad_recon(sino, ang, 3, config=_ad_config(True, 0.0), devices=[0, 1],
                    pose_callback=lambda i, e, s: got.append((i, e.shape, s.shape)))
    assert got == [(i, (NA, 3), (NA, 2)) for i in range(3)]


def test_rejected_combinations():
    sino, ang, _ = _data()
    with pytest.raises(ValueError, match="single-axis"):
        ft.run_sirt(sino, ang, 2, euler_zyz_deg=torch.zeros(NA, 3), devices=[0, 1])
    with pytest.raises(ValueError, match="projector"):
        ft.run_fbp(sino, ang, projector=ft.ParallelProjector((NY, N, N), ang.to(DEV), device=DEV),
                   devices=[0, 1])
    cfg = _ad_config(False, 0.0)
    cfg.reg.s3im_weight = 0.1
    with pytest.raises(NotImplementedError, match="s3im"):
        ft.run_ad_recon(sino, ang, 2, config=cfg, devices=[0, 1])
    with pytest.raises(NotImplementedError, match="LBFGS"):
        ft.run_grad_descent(sino, ang.to(DEV), 2, optimizer_cls=torch.optim.LBFGS, devices=[0, 1])


def test_worker_error_propagates_without_deadlock():
    sino, ang, _ = _data()
    with pytest.raises(ValueError):
        ft.run_fista_tv(sino, ang, 3, tv_weight=1.0, support_mask=torch.ones(3, 3, 3),
                        devices=[0, 1])


def test_resire_projection_separates_along_tilt_axis():
    sino, ang, vol = _data()
    eul = torch.zeros(NA, 3)
    eul[:, 1] = ang
    eul = eul.to(DEV)
    full_fp = ft.forward_project_resire(vol.to(DEV), eul)
    full_bp = ft.back_project_resire(sino, eul, Nz=N)
    for s, e in _multigpu._split(NY, 3):
        torch.testing.assert_close(
            ft.forward_project_resire(vol[s:e].contiguous().to(DEV), eul), full_fp[:, s:e],
            rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            ft.back_project_resire(sino[:, s:e].contiguous(), eul, Nz=N), full_bp[s:e],
            rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("devices", DEVICE_SETS)
def test_resire_matches(devices):
    sino, ang, _ = _data()
    kw = dict(support_mask=_row_support(), sparsity=1e-3, return_loss=True)
    ref, ref_loss = ft.run_resire(sino, ang, 10, **kw)
    out, loss = ft.run_resire(sino, ang, 10, devices=devices, **kw)
    assert torch.equal(out, ref) or rel_l2(out, ref) < 1e-6
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=0)
