"""Generate a physically-grounded BF-TEM phantom with crystalline-grain Bragg anomalies.

The phantom is an amorphous matrix (cylindrical low-intensity fill) containing
several crystalline grains (spheres of higher intensity). Each grain is assigned
a set of "Bragg tilt indices" — the specific tilts at which the grain satisfies
the Bragg condition and diffracts the beam *away* from the bright-field
detector. At each grain's Bragg tilts, the rays passing *through that grain*
have their intensity attenuated by `(1 - bragg_factor)` of the grain's
contribution, simulating the localised extinction that ruins FBP/SIRT
reconstructions and motivates MBIR-AM-PE's anomaly modelling.

This replaces the earlier random-sinogram-strip corruption: the new corruption
is spatially coherent with volume features and only affects the projection
rays that physically intersect each grain at its Bragg tilts.

Outputs:
    data/bftem_phantom.npy                       phantom volume [N, N, N]
    data/bftem_grains.npz                        grain metadata (centers, radii, intensities, bragg tilt assignments)
    data/bftem_sino_clean.npy                    clean forward projection
    data/bftem_sino_bragg.npy                    Bragg-corrupted forward projection
    data/bftem_bragg_mask.npy                    bool mask of corrupted (tilt, Y, det) pixels

Run:
    python benchmarks/scripts/gen_mbir_phantom_data.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import fasttomo as ft


# ── phantom construction ──────────────────────────────────────────────────


def make_bftem_phantom(
    N: int = 200,
    matrix_intensity: float = 0.3,
    grain_intensity: float = 0.6,
    matrix_radius_frac: float = 0.4,
    n_grains: int = 6,
    grain_radius: int = 12,
    seed: int = 0,
) -> tuple[np.ndarray, list[dict]]:
    """Build a [N, N, N] BF-TEM phantom: cylindrical amorphous matrix +
    several spherical crystalline grains.

    Returns
    -------
    phantom : [N, N, N] float32
    grains  : list of dicts {center: (iy, iz, ix), radius, intensity}
    """
    rng = np.random.default_rng(seed)
    phantom = np.zeros((N, N, N), dtype=np.float32)

    half = (N - 1) / 2.0
    matrix_radius = matrix_radius_frac * N

    # Cylindrical amorphous matrix (axis along Y).
    zz, xx = np.meshgrid(
        np.arange(N, dtype=np.float32) - half,
        np.arange(N, dtype=np.float32) - half,
        indexing="ij",
    )
    cyl_mask = (zz * zz + xx * xx) <= matrix_radius * matrix_radius
    phantom[:, cyl_mask] = matrix_intensity

    # Place non-overlapping crystalline grains inside the matrix.
    grains: list[dict] = []
    attempts_max = 200
    pad = grain_radius + 2
    for _ in range(n_grains):
        for _ in range(attempts_max):
            cy = int(rng.integers(pad, N - pad))
            cz = int(rng.integers(pad, N - pad))
            cx = int(rng.integers(pad, N - pad))
            # Keep grain centre inside the matrix region (allow grain to touch matrix edge).
            radial = float(np.hypot(cz - half, cx - half))
            if radial > matrix_radius - grain_radius * 0.5:
                continue
            # Check non-overlap with existing grains (clearance of 4 voxels).
            ok = True
            for g in grains:
                d = np.linalg.norm(np.array([cy, cz, cx]) - g["center"])
                if d < g["radius"] + grain_radius + 4:
                    ok = False
                    break
            if ok:
                grains.append({
                    "center": np.array([cy, cz, cx], dtype=np.int32),
                    "radius": grain_radius,
                    "intensity": grain_intensity,
                })
                break
        else:
            print(f"warning: only placed {len(grains)}/{n_grains} grains")
            break

    # Render each grain as a solid sphere (overwrites the matrix where it sits).
    iy_grid, iz_grid, ix_grid = np.meshgrid(
        np.arange(N), np.arange(N), np.arange(N), indexing="ij",
    )
    for g in grains:
        cy, cz, cx = g["center"]
        d2 = (iy_grid - cy) ** 2 + (iz_grid - cz) ** 2 + (ix_grid - cx) ** 2
        phantom[d2 <= g["radius"] ** 2] = g["intensity"]

    return phantom, grains


def assign_bragg_tilts(
    n_grains: int,
    n_angles: int,
    n_bragg_per_grain: int = 2,
    seed: int = 0,
) -> list[list[int]]:
    """For each grain, pick `n_bragg_per_grain` tilt indices at which it
    diffracts. Different grains get disjoint(ish) Bragg sets so the
    corruption is spread across the sinogram (more realistic than every
    grain extincting at the same tilt)."""
    rng = np.random.default_rng(seed + 100)
    out: list[list[int]] = []
    for _ in range(n_grains):
        out.append(rng.choice(n_angles, size=n_bragg_per_grain, replace=False).tolist())
    return out


# ── forward projection with grain-level Bragg corruption ──────────────────


def project_with_bragg(
    phantom: np.ndarray,
    grains: list[dict],
    bragg_assignments: list[list[int]],
    angles_deg: np.ndarray,
    bragg_factor: float = 0.15,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-project the phantom and apply per-grain Bragg corruption.

    At each grain's Bragg tilts, the rays passing through that grain have
    their intensity attenuated by `(1 - bragg_factor)` of the grain's
    contribution. `bragg_factor=0.15` means the grain contributes only 15%
    of its line-integral at the diffracting tilt (85% extinction).

    Returns
    -------
    clean_sino   : [n_angles, Ny, N_det] forward projection of the full phantom
    bragg_sino   : same shape, with grain extinction applied at the Bragg tilts
    bragg_mask   : bool array marking which sinogram entries were touched
                   (i.e. where the corruption is non-negligible)
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    vol_t = torch.from_numpy(phantom).to(dev).contiguous()
    ang_t = torch.from_numpy(angles_deg.astype(np.float32)).to(dev)

    clean_sino = ft.forward_project(vol_t, ang_t).cpu().numpy()
    bragg_sino = clean_sino.copy()
    bragg_mask = np.zeros_like(clean_sino, dtype=bool)

    n_angles, Ny, N_det = clean_sino.shape

    # Per-grain isolated forward projection, then attenuate at Bragg tilts.
    for i, g in enumerate(grains):
        grain_only = np.zeros_like(phantom)
        cy, cz, cx = g["center"]
        r = g["radius"]
        # Crop the sphere into the volume.
        yy, zz, xx = np.meshgrid(
            np.arange(Ny), np.arange(phantom.shape[1]), np.arange(phantom.shape[2]),
            indexing="ij",
        )
        d2 = (yy - cy) ** 2 + (zz - cz) ** 2 + (xx - cx) ** 2
        grain_only[d2 <= r * r] = g["intensity"]
        grain_t = torch.from_numpy(grain_only).to(dev).contiguous()
        grain_sino = ft.forward_project(grain_t, ang_t).cpu().numpy()

        for k_brg in bragg_assignments[i]:
            # Attenuate this grain's contribution at the Bragg tilt by (1 - bragg_factor).
            bragg_sino[k_brg] -= (1.0 - bragg_factor) * grain_sino[k_brg]
            # Record which detector pixels were touched (i.e. the grain's shadow).
            bragg_mask[k_brg] |= grain_sino[k_brg] > 1e-4

    return clean_sino, bragg_sino, bragg_mask


# ── CLI entry ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--n-angles", type=int, default=29)
    parser.add_argument("--n-grains", type=int, default=6)
    parser.add_argument("--grain-radius", type=int, default=12)
    parser.add_argument("--n-bragg-per-grain", type=int, default=2)
    parser.add_argument("--bragg-factor", type=float, default=0.15,
                        help="Fraction of grain intensity retained at Bragg tilt "
                             "(0 = full extinction, 1 = no Bragg). Default 0.15.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"building {args.N}^3 BF-TEM phantom with {args.n_grains} grains...")
    phantom, grains = make_bftem_phantom(
        N=args.N,
        n_grains=args.n_grains,
        grain_radius=args.grain_radius,
        seed=args.seed,
    )
    print(f"  placed {len(grains)} grains, max intensity {phantom.max():.3f}")

    angles = np.linspace(-70.0, 70.0, args.n_angles, dtype=np.float32)
    bragg_assignments = assign_bragg_tilts(
        len(grains), args.n_angles, args.n_bragg_per_grain, args.seed,
    )
    print(f"  Bragg tilt assignments per grain: {bragg_assignments}")

    print("forward-projecting phantom + per-grain Bragg corruption...")
    clean_sino, bragg_sino, bragg_mask = project_with_bragg(
        phantom, grains, bragg_assignments, angles,
        bragg_factor=args.bragg_factor,
        device=args.device,
    )

    # Save artifacts (all in data/, which is gitignored — recipe is the source of truth).
    np.save(args.out_dir / "bftem_phantom.npy", phantom)
    np.save(args.out_dir / "bftem_sino_clean.npy", clean_sino)
    np.save(args.out_dir / "bftem_sino_bragg.npy", bragg_sino)
    np.save(args.out_dir / "bftem_bragg_mask.npy", bragg_mask)
    np.savez(
        args.out_dir / "bftem_grains.npz",
        centers=np.array([g["center"] for g in grains]),
        radii=np.array([g["radius"] for g in grains]),
        intensities=np.array([g["intensity"] for g in grains]),
        bragg_tilts=np.array(bragg_assignments),
        angles_deg=angles,
        bragg_factor=args.bragg_factor,
    )

    print(f"wrote {args.out_dir}/bftem_phantom.npy           shape={phantom.shape}")
    print(f"wrote {args.out_dir}/bftem_sino_clean.npy        shape={clean_sino.shape}, max={clean_sino.max():.2f}")
    print(f"wrote {args.out_dir}/bftem_sino_bragg.npy        max={bragg_sino.max():.2f}, min={bragg_sino.min():.4f}")
    print(f"wrote {args.out_dir}/bftem_bragg_mask.npy        #corrupted={bragg_mask.sum()}")
    print(f"wrote {args.out_dir}/bftem_grains.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
