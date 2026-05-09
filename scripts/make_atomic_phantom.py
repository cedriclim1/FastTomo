#!/usr/bin/env python3
"""
Bimetallic L10-FePt nanoparticle phantom for atomic-resolution tomography.

Crystal structure: L10 FePt  (tetragonal, space group P4/mmm)
  a = 3.86 Å  (in-plane, Fe-Fe and Pt-Pt within each layer)
  c = 3.71 Å  (out-of-plane, Fe-layer to Pt-layer spacing = c/2 = 1.855 Å)

  Fe layers at z = 0,   c,   2c, …  → two atoms per layer at (0,0) and (a/2,a/2)
  Pt layers at z = c/2, 3c/2, …    → two atoms per layer at (a/2,0) and (0,a/2)

Nearest-neighbour distances:
  Fe–Fe within layer : a/√2 = 2.73 Å
  Pt–Pt within layer : a/√2 = 2.73 Å
  Fe–Pt cross-layer  : √((a/2)²+(c/2)²) = 2.68 Å  ← shortest bond

HAADF-STEM intensities scale as Z^1.7:
  I_Fe = (26/78)^1.7 ≈ 0.175   (normalised to Pt = 1.0)
  I_Pt = 1.0

Volume convention: [Ny, Nz, Nx]  (FastTomo standard)
  Crystal y-axis → Ny (tilt axis — independent slices)
  Crystal z-axis → Nz (depth     — L10 stacking direction)
  Crystal x-axis → Nx (lateral)

Outputs
-------
data/fept_phantom_200.npy          float32 [200,200,200] volume
data/fept_phantom_200_atoms.npz    atom positions (Å) + species labels
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ── parameters ─────────────────────────────────────────────────────────────────
N            = 200
PIXEL        = 0.5        # Å / voxel  →  box = 100 Å = 10 nm
RADIUS_ANG   = 32.0       # Å sphere radius → 6.4 nm diameter particle
SIGMA_ANG    = 0.60       # Å  Gaussian atom width (thermal smearing + probe PSF)
DISORD_ANG   = 0.12       # Å  rms per-axis thermal displacement
ANTISITE_FRAC = 0.03      # fraction of sites with swapped species (chemical disorder)
SEED         = 42

# L10 FePt lattice constants (Å)
a_latt, c_latt = 3.86, 3.71

# HAADF intensities normalised to Pt = 1
Z_Fe, Z_Pt = 26.0, 78.0
I_Fe = (Z_Fe / Z_Pt) ** 1.7   # ≈ 0.175
I_Pt = 1.0

# Wyckoff positions within one unit cell
#   Fe: 1a (0,0,0)  and  2c face-diagonal (a/2, a/2, 0)
#   Pt: 2e edge     (a/2, 0, c/2)  and  (0, a/2, c/2)
BASIS = [
    ( 0.0,       0.0,       0.0,       'Fe', I_Fe ),
    ( a_latt/2,  a_latt/2,  0.0,       'Fe', I_Fe ),
    ( a_latt/2,  0.0,       c_latt/2,  'Pt', I_Pt ),
    ( 0.0,       a_latt/2,  c_latt/2,  'Pt', I_Pt ),
]

# ── tile lattice inside sphere ─────────────────────────────────────────────────
rng   = np.random.default_rng(SEED)
n_max = int(RADIUS_ANG / min(a_latt, c_latt)) + 3

pos_x, pos_y, pos_z = [], [], []   # Å, relative to box centre
sp_list, int_list   = [], []

for iz in range(-n_max, n_max + 1):
    for iy in range(-n_max, n_max + 1):
        for ix in range(-n_max, n_max + 1):
            for bx, by, bz, sp, intens in BASIS:
                rx = ix * a_latt + bx
                ry = iy * a_latt + by
                rz = iz * c_latt + bz
                if rx**2 + ry**2 + rz**2 <= RADIUS_ANG**2:
                    pos_x.append(rx)
                    pos_y.append(ry)
                    pos_z.append(rz)
                    sp_list.append(sp)
                    int_list.append(intens)

pos_x = np.array(pos_x, dtype=np.float32)
pos_y = np.array(pos_y, dtype=np.float32)
pos_z = np.array(pos_z, dtype=np.float32)
species     = np.array(sp_list)
intensities = np.array(int_list, dtype=np.float32)
n_total = len(intensities)

# Anti-site disorder
n_swap = int(ANTISITE_FRAC * n_total)
swap_idx = rng.choice(n_total, size=n_swap, replace=False)
for idx in swap_idx:
    if species[idx] == 'Fe':
        species[idx]     = 'Pt'
        intensities[idx] = I_Pt
    else:
        species[idx]     = 'Fe'
        intensities[idx] = I_Fe

# Thermal displacements
pos_x += rng.normal(0, DISORD_ANG, n_total).astype(np.float32)
pos_y += rng.normal(0, DISORD_ANG, n_total).astype(np.float32)
pos_z += rng.normal(0, DISORD_ANG, n_total).astype(np.float32)

n_fe = (species == 'Fe').sum()
n_pt = (species == 'Pt').sum()
print(f"Atoms placed:  {n_total} total  |  {n_fe} Fe  |  {n_pt} Pt")
print(f"Anti-site swaps: {n_swap}")

# ── rasterise: point masses then Gaussian blur ─────────────────────────────────
# Volume layout: [Ny, Nz, Nx]
#   pos_y → first axis (Ny, tilt axis)
#   pos_z → second axis (Nz, depth / L10 stacking)
#   pos_x → third axis (Nx, lateral)
center = (N - 1) / 2.0

ivy = np.rint(pos_y / PIXEL + center).astype(np.int32)
ivz = np.rint(pos_z / PIXEL + center).astype(np.int32)
ivx = np.rint(pos_x / PIXEL + center).astype(np.int32)

mask = (
    (ivy >= 0) & (ivy < N) &
    (ivz >= 0) & (ivz < N) &
    (ivx >= 0) & (ivx < N)
)
vol = np.zeros((N, N, N), dtype=np.float32)
np.add.at(vol, (ivy[mask], ivz[mask], ivx[mask]), intensities[mask])

sigma_vox = SIGMA_ANG / PIXEL   # 1.2 voxels
vol = gaussian_filter(vol, sigma=sigma_vox).astype(np.float32)

print(f"Volume shape: {vol.shape}  dtype: {vol.dtype}")
print(f"  max={vol.max():.4f}  mean={vol[vol > vol.max()*0.01].mean():.4f}  "
      f"nonzero={np.count_nonzero(vol > 1e-4)}")

# ── save ───────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent.parent / "data"
out_dir.mkdir(exist_ok=True)

vol_path   = out_dir / "fept_phantom_200.npy"
atoms_path = out_dir / "fept_phantom_200_atoms.npz"

np.save(vol_path, vol)

# Save atoms in Å (relative to box centre), useful for validation
pos_ang = np.stack([pos_x, pos_y, pos_z], axis=1)   # (N_atoms, 3)
np.savez(atoms_path,
         positions_ang=pos_ang,
         species=species,
         intensities=intensities,
         pixel_angstrom=np.float32(PIXEL),
         sigma_angstrom=np.float32(SIGMA_ANG),
         radius_angstrom=np.float32(RADIUS_ANG),
         lattice_a=np.float32(a_latt),
         lattice_c=np.float32(c_latt))

print(f"\nSaved volume → {vol_path}")
print(f"Saved atoms  → {atoms_path}")

# ── summary ────────────────────────────────────────────────────────────────────
print("\n── Structure summary ──────────────────────────────────────────────────")
print(f"  Structure     : L10 FePt  (tetragonal a={a_latt} Å, c={c_latt} Å)")
print(f"  Nanoparticle  : spherical, r={RADIUS_ANG} Å ({2*RADIUS_ANG/10:.1f} nm diam)")
print(f"  Box           : {N}³ voxels @ {PIXEL} Å/vox = {N*PIXEL:.0f} Å = {N*PIXEL/10:.1f} nm")
print(f"  Atom Gaussian : σ = {SIGMA_ANG} Å = {sigma_vox:.2f} vox")
print(f"  Shortest bond : Fe–Pt ≈ 2.68 Å = {2.68/PIXEL:.1f} vox  (>> 2σ = {2*sigma_vox:.1f} vox)")
print(f"  Thermal disp  : {DISORD_ANG} Å rms/axis")
print(f"  Anti-sites    : {ANTISITE_FRAC*100:.0f}%  ({n_swap} atoms swapped)")
print(f"  I_Fe / I_Pt   : {I_Fe:.3f} / {I_Pt:.3f}")
