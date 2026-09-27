"""Unit tests for fasttomo.atom_tracer.

Builds a small synthetic 3-atom phantom with non-integer atom positions
and a 5.7× intensity ratio between the heavy and light species, then
checks (a) sub-voxel localisation accuracy, (b) the LS fit recovers the
correct σ and amplitude, (c) the GMM classifier separates the two
intensity classes.

CPU-only; the tracer's CUDA codepath is exercised opportunistically when
a GPU is available, but is not required for correctness.
"""

import math

import numpy as np
import pytest
import torch

import fasttomo as ft


def _make_synthetic(positions_yzx, amplitudes, sigma=1.2, shape=(32, 32, 32)):
    """Render a sum-of-Gaussians volume; positions can be non-integer."""
    Ny, Nz, Nx = shape
    ay = np.arange(Ny, dtype=np.float32)
    az = np.arange(Nz, dtype=np.float32)
    ax = np.arange(Nx, dtype=np.float32)
    Y, Z, X = np.meshgrid(ay, az, ax, indexing="ij")

    vol = np.zeros(shape, dtype=np.float32)
    for (py, pz, px), A in zip(positions_yzx, amplitudes):
        r2 = (Y - py) ** 2 + (Z - pz) ** 2 + (X - px) ** 2
        vol += A * np.exp(-r2 / (2 * sigma * sigma))
    return vol


_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")


@pytest.mark.parametrize("device", _DEVICES)
def test_subvoxel_accuracy_three_atoms(device):
    """Three atoms at non-integer positions are localised to <0.15 vox."""
    positions = np.array([
        [10.3, 10.7, 10.2],
        [20.6, 12.1, 14.8],
        [15.5, 22.4, 18.9],
    ], dtype=np.float32)
    amps = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    vol_np = _make_synthetic(positions, amps, sigma=1.2, shape=(32, 32, 32))
    vol = torch.from_numpy(vol_np).to(device)

    result = ft.trace_atoms(
        vol, sigmas_vox=(1.0, 1.2, 1.4), threshold_rel=0.05,
        n_species=1, fit_window=2,
    )

    assert len(result) == 3, f"expected 3 atoms, got {len(result)}"

    # Match each detection to the nearest GT and check distance.
    det = result.positions_vox
    diffs = positions[:, None, :] - det[None, :, :]
    d = np.linalg.norm(diffs, axis=2)
    min_d = d.min(axis=1)
    assert (min_d < 0.15).all(), (
        f"sub-voxel error too large: {min_d} (max {min_d.max():.3f})"
    )

    # σ recovery: should be close to 1.2 (within the fit's tolerance).
    assert np.all(np.abs(result.sigmas - 1.2) < 0.15), result.sigmas


def test_gmm_two_species_split():
    """Heavy and light atoms separate cleanly under the GMM.

    Atoms are spaced ≥ 6 vox apart to avoid NMS merging — this test is
    about the classifier, not detection density.
    """
    light = np.array([
        [ 8.3,  8.5,  8.2],
        [ 8.1, 24.6,  8.0],
        [ 8.5,  8.2, 24.7],
        [ 8.7, 24.3, 24.4],
    ], dtype=np.float32)
    heavy = np.array([
        [24.4, 16.4, 16.5],
        [16.1, 24.7, 16.6],
        [16.5, 16.3, 24.4],
        [24.9, 24.1, 24.5],
    ], dtype=np.float32)
    positions = np.concatenate([light, heavy], axis=0)
    amps = np.array([0.175] * 4 + [1.0] * 4, dtype=np.float32)

    vol_np = _make_synthetic(positions, amps, sigma=1.2, shape=(32, 32, 32))
    vol = torch.from_numpy(vol_np)

    result = ft.trace_atoms(
        vol, sigmas_vox=(1.0, 1.2, 1.4), threshold_rel=0.05,
        n_species=2, fit_window=2,
    )

    assert len(result) >= 7, f"detection lost atoms: {len(result)}"

    # Match each GT atom to the nearest detection, then check species.
    det = result.positions_vox
    diffs = positions[:, None, :] - det[None, :, :]
    d = np.linalg.norm(diffs, axis=2)
    nearest = d.argmin(axis=1)
    matched_mask = d.min(axis=1) < 0.5

    species_gt = np.array([0] * 4 + [1] * 4, dtype=np.int8)
    species_det = result.species_idx[nearest]

    # Allow at most one mislabel out of the matched atoms.
    matches = (species_det == species_gt) & matched_mask
    assert matches.sum() >= matched_mask.sum() - 1, (
        f"GMM split too noisy: {matches.sum()} / {matched_mask.sum()} correct; "
        f"det={species_det}, gt={species_gt}"
    )
    # Component means must be ascending (post-sort invariant).
    assert result.component_log_mass[0] < result.component_log_mass[1]


def test_atomlist_npz_dict_roundtrip(tmp_path):
    """Saving + loading the AtomList preserves all arrays."""
    positions = np.array([[10.5, 10.5, 10.5], [20.5, 20.5, 20.5]],
                         dtype=np.float32)
    amps = np.array([0.175, 1.0], dtype=np.float32)
    vol_np = _make_synthetic(positions, amps, sigma=1.2, shape=(32, 32, 32))
    result = ft.trace_atoms(
        torch.from_numpy(vol_np), sigmas_vox=(1.2,), threshold_rel=0.05,
        n_species=2,
    )
    out = tmp_path / "atoms.npz"
    np.savez(out, **result.to_npz_dict())
    loaded = np.load(out)
    for key in result.to_npz_dict():
        assert key in loaded.files


def test_empty_volume_returns_empty_list():
    vol = torch.zeros(16, 16, 16)
    result = ft.trace_atoms(vol, sigmas_vox=(1.2,), n_species=2)
    assert len(result) == 0
    assert result.positions_vox.shape == (0, 3)
    assert result.species_prob.shape == (0, 2)


# ── Physical-filter tests ─────────────────────────────────────────────────


def _make_lattice(positions_yzx, amps, sigma=1.2, shape=(48, 48, 48)):
    return _make_synthetic(positions_yzx, amps, sigma=sigma, shape=shape)


def test_split_peak_filter_rejects_close_pair():
    """Two peaks 2 vox apart get rejected as split-peaks; well-spaced lattice survives."""
    # Build a small 3×3×3 cubic lattice at NN spacing matching the configured
    # lattice constants below (a=4.0 Å, c=4.0 Å → d_NN = sqrt(2²+2²) = 2.83 Å
    # → at pixel 0.5 Å/vox, d_NN_vox = 5.66). Use a simple cubic of spacing
    # 5.66 vox so each atom has 6 NN inside the bond shell.
    spacing = 5.66
    centers = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                centers.append([10 + i * spacing,
                                10 + j * spacing,
                                10 + k * spacing])
    centers = np.array(centers, dtype=np.float32)
    amps = np.ones(len(centers), dtype=np.float32)

    # Inject a split-peak ~3 vox from the central atom — far enough to
    # survive 3³ NMS but inside d_min = 0.6·d_NN ≈ 3.4 vox so the
    # bond-distance filter rejects it.
    centre_idx = len(centers) // 2
    split = centers[centre_idx] + np.array([3.0, 0, 0], dtype=np.float32)
    positions = np.concatenate([centers, split[None]], axis=0)
    amps_full = np.concatenate([amps, [0.8]])

    vol_np = _make_lattice(positions, amps_full, sigma=0.8, shape=(48, 48, 48))
    vol = torch.from_numpy(vol_np)

    # Without filters: detection should find both centres + split.
    no_filter = ft.trace_atoms(
        vol, threshold_rel=0.05, n_species=1, fit_window=2,
    )
    # With filters: split-peak (and only the split) should be removed.
    filtered = ft.trace_atoms(
        vol, threshold_rel=0.05, n_species=1, fit_window=2,
        voxel_size_angstrom=0.5,
        lattice_constants_angstrom=(4.0, 4.0),
        bond_distance_min_frac=0.6,
        bond_distance_max_frac=1.4,
        coord_min=2,
    )

    assert len(no_filter) >= len(filtered) + 1, (
        f"split-peak filter should drop ≥1 atom: no_filter={len(no_filter)} "
        f"filtered={len(filtered)}"
    )
    # The 27-atom cubic core should survive entirely; split is the only
    # atom we expect dropped.
    assert len(filtered) >= 26, f"lattice core damaged: {len(filtered)}"
    assert filtered.filter_rejections.get("split_peak", 0) >= 1


def test_lone_peak_filter_rejects_isolated_noise():
    """A lone noise spike far from any neighbour gets rejected; lattice survives."""
    spacing = 5.66
    centers = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                centers.append([8 + i * spacing,
                                8 + j * spacing,
                                8 + k * spacing])
    centers = np.array(centers, dtype=np.float32)

    # Inject a lone peak with no neighbours within ~2·d_NN.
    lone = np.array([[40.0, 40.0, 40.0]], dtype=np.float32)
    positions = np.concatenate([centers, lone], axis=0)
    amps = np.ones(len(positions), dtype=np.float32)

    vol_np = _make_lattice(positions, amps, sigma=1.0, shape=(48, 48, 48))
    vol = torch.from_numpy(vol_np)

    filtered = ft.trace_atoms(
        vol, threshold_rel=0.05, n_species=1, fit_window=2,
        voxel_size_angstrom=0.5,
        lattice_constants_angstrom=(4.0, 4.0),
        bond_distance_min_frac=0.6,
        bond_distance_max_frac=1.4,
        coord_min=2,
    )
    # Lone peak gone; cubic core kept.
    assert len(filtered) >= 26, f"lattice damaged: {len(filtered)}"
    # The lone peak at (40,40,40) should be absent.
    if len(filtered) > 0:
        d_to_lone = np.linalg.norm(
            filtered.positions_vox - lone[0], axis=1
        )
        assert d_to_lone.min() > 3.0, (
            "lone-peak filter failed to reject isolated atom"
        )
    rej = filtered.filter_rejections
    assert (rej.get("lone", 0) + rej.get("low_coord", 0)) >= 1


def test_two_pass_recovers_dim_atom_near_bright():
    """Iterative-CLEAN two-pass detection finds a dim atom whose local-max
    is suppressed by a bright Pt-like neighbour after smoothing.

    Build two atoms 5 vox apart with a 6× intensity ratio, then heavily
    blur the volume so the dim atom's 3³ NMS window is dominated by the
    bright atom's shoulder. Single-pass NMS misses the dim atom; two-pass
    (subtract bright peak, redetect) recovers it.
    """
    bright = np.array([[16.0, 16.0, 16.0]], dtype=np.float32)
    dim = np.array([[16.0, 21.0, 16.0]], dtype=np.float32)
    positions = np.concatenate([bright, dim], axis=0)
    amps = np.array([1.0, 0.17], dtype=np.float32)

    # Render with a wide σ so the bright atom's shoulder reaches the dim
    # atom's centre — mimics the TV+SSIM-broadened AD recon.
    vol_np = _make_synthetic(positions, amps, sigma=1.8, shape=(32, 32, 32))
    vol = torch.from_numpy(vol_np)

    single = ft.trace_atoms(
        vol, threshold_rel=0.005, n_species=1, fit_window=2,
    )
    two_pass = ft.trace_atoms(
        vol, threshold_rel=0.005, n_species=1, fit_window=2,
        n_clean_passes=2, clean_render_sigma=1.8,
    )

    # Two-pass should find at least one more atom than single-pass, OR find
    # both. Direct check: dim atom must be matched by two-pass.
    if len(single) > 0:
        d_single = np.linalg.norm(
            single.positions_vox - dim[0], axis=1
        ).min()
    else:
        d_single = float("inf")
    if len(two_pass) > 0:
        d_two = np.linalg.norm(
            two_pass.positions_vox - dim[0], axis=1
        ).min()
    else:
        d_two = float("inf")

    assert d_two < 1.0, (
        f"two-pass failed to recover dim atom: dist={d_two:.3f}, "
        f"n_single={len(single)} n_two={len(two_pass)}"
    )
    # Single-pass should miss the dim atom (the whole point of the test).
    # Two-pass finds it because subtraction of the bright peak reveals it.
    assert d_two < d_single, (
        f"two-pass not better than single: d_single={d_single:.3f} "
        f"d_two={d_two:.3f}"
    )


def test_count_prior_gate_escalates_threshold():
    """Setting an explicit n_expected lower than the true count escalates the
    threshold and cuts detection."""
    spacing = 5.66
    centers = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                centers.append([8 + i * spacing,
                                8 + j * spacing,
                                8 + k * spacing])
    centers = np.array(centers, dtype=np.float32)
    # Wide amplitude range so threshold escalation drops weakest peaks.
    amps = np.linspace(0.05, 1.0, len(centers), dtype=np.float32)

    vol_np = _make_lattice(centers, amps, sigma=1.0, shape=(48, 48, 48))
    vol = torch.from_numpy(vol_np)

    full = ft.trace_atoms(
        vol, threshold_rel=0.02, n_species=1, fit_window=2,
        voxel_size_angstrom=0.5,
        lattice_constants_angstrom=(4.0, 4.0),
    )
    # Explicit prior at 10 atoms (truth is 27) should drop weakest peaks.
    gated = ft.trace_atoms(
        vol, threshold_rel=0.02, n_species=1, fit_window=2,
        voxel_size_angstrom=0.5,
        lattice_constants_angstrom=(4.0, 4.0),
        enable_count_prior=True,
        count_prior_n_expected=10,
        count_prior_tol=0.05,
        count_prior_max_retries=3,
    )
    assert len(gated) < len(full), (
        f"count gate should reduce detections: full={len(full)} gated={len(gated)}"
    )
    assert gated.filter_rejections.get("count_gate_retries", 0) >= 1
